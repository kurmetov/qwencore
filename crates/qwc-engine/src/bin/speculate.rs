//! Спекулятивный decode на MTP-голове: черновики головой, проверка моделью.
//!
//! `cargo run --release -p qwc-engine --bin speculate -- --prompt-ids 9707 --max-new 64 --draft 2`
//!
//! Шаг устроен так: голова по цепочке предлагает k токенов (каждый стоит
//! чтения её 849 МБ плюс словарной проекции), модель проверяет их одним
//! проходом по весам, и принимается самый длинный совпавший префикс плюс один
//! свой токен сверху. Состояние DeltaNet при этом живёт на черновом слоте:
//! пока неизвестно, сколько принято, настоящее трогать нельзя.
//!
//! Флаг `--verify` прогоняет то же самое обычным greedy и сверяет токены: у
//! спекуляции нет права менять выход, только скорость.

use qwc_core::arch::{HIDDEN_SIZE, VOCAB_SIZE};
use qwc_cuda::paged_attention::KvCacheDtype;
use qwc_cuda::sampling::Argmax;
use qwc_cuda::{DeviceBuffer, Stream};
use qwc_engine::mtp::{self, MtpHead, MtpScratch};
use qwc_engine::{Executor, ExecutorConfig, ModelWeights};
use qwc_model::Checkpoint;
use std::path::PathBuf;
use std::time::Instant;

struct Speculator {
    head: MtpHead,
    scratch: MtpScratch,
    hidden: DeviceBuffer<u16>,
    draft_hidden: DeviceBuffer<u16>,
    logits: DeviceBuffer<f32>,
    argmax: Argmax,
    stream: Stream,
}

impl Speculator {
    fn new(checkpoint: &Checkpoint, context: usize) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            head: mtp::load(checkpoint)?,
            scratch: MtpScratch::new(1, context, KvCacheDtype::Fp8)?,
            hidden: DeviceBuffer::zeroed(HIDDEN_SIZE)?,
            draft_hidden: DeviceBuffer::zeroed(HIDDEN_SIZE)?,
            logits: DeviceBuffer::zeroed(VOCAB_SIZE)?,
            argmax: Argmax::new(1, VOCAB_SIZE)?,
            stream: Stream::new()?,
        })
    }

    /// Цепочка черновиков: первый идёт по скрытому состоянию модели, дальше
    /// голова продолжает по собственному выходу и собственному предсказанию.
    fn draft(
        &mut self,
        weights: &ModelWeights,
        token: u32,
        position: usize,
        depth: usize,
    ) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let mut drafts = Vec::with_capacity(depth);
        let mut input = token;
        for step in 0..depth {
            let source: *const DeviceBuffer<u16> = if step == 0 {
                &self.hidden
            } else {
                &self.draft_hidden
            };
            // SAFETY: буферы живут в self и не пересекаются с выходом.
            let source = unsafe { &*source };
            mtp::draft(
                &self.head,
                weights,
                &mut self.scratch,
                source,
                &[input],
                &[(position + step) as u32],
                &mut self.draft_hidden,
                &self.stream,
            )?;
            weights
                .lm_head
                .logits(&self.draft_hidden, &mut self.logits, 1, &self.stream)?;
            self.argmax.sample(&self.logits, 1, &self.stream)?;
            input = self.argmax.to_host(1)?[0];
            drafts.push(input);
        }
        Ok(drafts)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model = PathBuf::from(std::env::var("HOME")?).join("models/Qwen3.8-27B-QUASAR-NVFP4");
    let mut prompt: Vec<u32> = vec![9707];
    let mut max_new = 64usize;
    let mut depth = 2usize;
    let mut context = 2048usize;
    let mut verify = false;
    let mut profile = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model = PathBuf::from(args.next().ok_or("--model")?),
            "--prompt-ids" => {
                prompt = args
                    .next()
                    .ok_or("--prompt-ids")?
                    .split(',')
                    .map(|id| id.trim().parse::<u32>())
                    .collect::<Result<_, _>>()?;
            }
            "--max-new" => max_new = args.next().ok_or("--max-new")?.parse()?,
            "--draft" => depth = args.next().ok_or("--draft")?.parse()?,
            "--context" => context = args.next().ok_or("--context")?.parse()?,
            "--verify" => verify = true,
            "--profile" => profile = true,
            other => return Err(format!("неизвестный аргумент: {other}").into()),
        }
    }
    assert!((1..=7).contains(&depth), "глубина черновика 1..7");

    let checkpoint = Checkpoint::open(&model)?;
    let weights = ModelWeights::load(&checkpoint)?;
    let mut executor = Executor::new(ExecutorConfig {
        max_batch: 1,
        max_context: context,
    })?;
    let mut speculator = Speculator::new(&checkpoint, context)?;

    // Спекулятивный прогон.
    executor.prefill_sequence(&weights, &prompt, 0, 0)?;
    executor.copy_prefill_hidden_row(prompt.len() - 1, &mut speculator.hidden)?;
    let mut next = executor.argmax_to_host(1)?[0];
    let mut generated: Vec<u32> = Vec::with_capacity(max_new);
    let mut position = prompt.len();
    let mut accepted_total = 0usize;
    let mut steps = 0usize;
    let started = Instant::now();

    let mut draft_seconds = 0.0;
    let mut verify_seconds = 0.0;
    let mut commit_seconds = 0.0;
    while generated.len() < max_new {
        let phase = Instant::now();
        let drafts = speculator.draft(&weights, next, position, depth)?;
        draft_seconds += phase.elapsed().as_secs_f64();
        let phase = Instant::now();
        let mut batch = Vec::with_capacity(depth + 1);
        batch.push(next);
        batch.extend_from_slice(&drafts);

        let truth = executor.verify_speculation(&weights, &batch, position - 1, 0)?;
        verify_seconds += phase.elapsed().as_secs_f64();
        let phase = Instant::now();
        // Принимается самый длинный совпавший префикс; за ним идёт токен,
        // который модель выдала на первой расходящейся строке — он верен по
        // построению, поэтому шаг всегда даёт хотя бы один токен.
        let mut accepted = 0usize;
        while accepted < drafts.len() && truth[accepted] == drafts[accepted] {
            accepted += 1;
        }
        let rows = accepted + 1;
        executor.commit_speculation(&weights, batch.len(), rows, 0)?;
        commit_seconds += phase.elapsed().as_secs_f64();

        generated.push(next);
        for index in 0..accepted {
            generated.push(drafts[index]);
        }
        next = truth[accepted];
        position += rows;
        accepted_total += accepted;
        steps += 1;
        executor.copy_prefill_hidden_row(rows - 1, &mut speculator.hidden)?;
    }
    let elapsed = started.elapsed().as_secs_f64();
    generated.truncate(max_new);

    println!("промпт: {prompt:?}");
    println!("выдано: {generated:?}");
    println!(
        "\n{} токенов за {:.3} с = {:.1} tok/s, шагов {steps}, принято черновиков {accepted_total} ({:.2} на шаг)",
        generated.len(),
        elapsed,
        generated.len() as f64 / elapsed,
        accepted_total as f64 / steps as f64
    );
    println!(
        "токенов за шаг: {:.2} (глубина черновика {depth})",
        (accepted_total + steps) as f64 / steps as f64
    );
    if profile {
        let phases = executor.profile_totals()?;
        let total: f32 = phases.iter().map(|(_, ms)| ms).sum();
        println!("\nразметка шагов проверки (сумма по всем шагам, {total:.1} мс):");
        for (name, ms) in phases.iter().take(10) {
            println!("  {name:<24} {ms:8.2} мс  {:5.1}%", ms / total * 100.0);
        }
    }
    println!(
        "на шаг: черновики {:.2} мс, проверка {:.2} мс, фиксация {:.2} мс, прочее {:.2} мс",
        draft_seconds * 1e3 / steps as f64,
        verify_seconds * 1e3 / steps as f64,
        commit_seconds * 1e3 / steps as f64,
        (elapsed - draft_seconds - verify_seconds - commit_seconds) * 1e3 / steps as f64
    );

    if verify {
        let mut plain = Executor::new(ExecutorConfig {
            max_batch: 1,
            max_context: context,
        })?;
        plain.prefill_sequence(&weights, &prompt, 0, 0)?;
        let mut token = plain.argmax_to_host(1)?[0];
        let mut reference = Vec::with_capacity(max_new);
        let started = Instant::now();
        for step in 0..max_new {
            reference.push(token);
            plain.decode(&weights, &[token], &[(prompt.len() + step) as u32])?;
            token = plain.argmax_to_host(1)?[0];
        }
        let plain_seconds = started.elapsed().as_secs_f64();
        println!(
            "\nобычный greedy: {:.1} tok/s, ускорение {:.2}x",
            max_new as f64 / plain_seconds,
            (max_new as f64 / elapsed) / (max_new as f64 / plain_seconds)
        );
        assert_eq!(
            generated, reference,
            "спекуляция изменила выход — это ошибка, а не оптимизация"
        );
        println!("выход совпал с обычным greedy токен в токен");
    }
    Ok(())
}

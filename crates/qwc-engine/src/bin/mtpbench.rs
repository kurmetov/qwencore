//! Acceptance и цена чернового шага MTP-головы на GPU.
//!
//! `cargo run --release -p qwc-engine --bin mtpbench -- --corpus bench/corpus/core.jsonl --max-new 128`
//!
//! Тот же замер, что оффлайн в `bench/mtp_acceptance.py`, но головой, которая
//! реально считается нашими ядрами: если числа сходятся, реализация верна.
//! Заодно меряется время чернового шага — из него и складывается вся выгода.

use qwc_core::arch::{HIDDEN_SIZE, VOCAB_SIZE};
use qwc_cuda::paged_attention::KvCacheDtype;
use qwc_cuda::{DeviceBuffer, Event, Stream};
use qwc_engine::mtp::{self, MtpScratch};
use qwc_engine::{Executor, ExecutorConfig, ModelWeights, PREFILL_CHUNK_SIZE};
use qwc_model::Checkpoint;
use std::path::PathBuf;

fn parse_cases(text: &str) -> Vec<(String, Vec<u32>)> {
    let mut cases = Vec::new();
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        let id = line
            .split("\"id\":\"")
            .nth(1)
            .and_then(|rest| rest.split('"').next())
            .unwrap_or("case")
            .to_string();
        let explicit = list(line, "\"prompt_token_ids\":");
        let prompt = match explicit {
            Some(ids) => ids,
            None => {
                let unit = list(line, "\"token_ids\":").expect("нет токенов промпта");
                let length: usize = line
                    .split("\"length\":")
                    .nth(1)
                    .and_then(|rest| rest.split([',', '}']).next())
                    .and_then(|value| value.trim().parse().ok())
                    .expect("нет length");
                unit.iter().cycle().take(length).copied().collect()
            }
        };
        cases.push((id, prompt));
    }
    cases
}

fn list(line: &str, key: &str) -> Option<Vec<u32>> {
    let rest = line.split(key).nth(1)?;
    let inside = rest.trim_start().strip_prefix('[')?.split(']').next()?;
    inside.split(',').map(|x| x.trim().parse().ok()).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model = PathBuf::from(std::env::var("HOME")?).join("models/Qwen3.8-27B-QUASAR-NVFP4");
    let mut corpus = PathBuf::from("bench/corpus/core.jsonl");
    let mut max_new = 128usize;
    let mut context = 4096usize;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model = PathBuf::from(args.next().ok_or("--model")?),
            "--corpus" => corpus = PathBuf::from(args.next().ok_or("--corpus")?),
            "--max-new" => max_new = args.next().ok_or("--max-new")?.parse()?,
            "--context" => context = args.next().ok_or("--context")?.parse()?,
            other => return Err(format!("неизвестный аргумент: {other}").into()),
        }
    }

    let cases = parse_cases(&std::fs::read_to_string(&corpus)?);
    let checkpoint = Checkpoint::open(&model)?;
    let weights = ModelWeights::load(&checkpoint)?;
    let head = mtp::load(&checkpoint)?;
    let mut executor = Executor::new(ExecutorConfig {
        max_batch: 1,
        max_context: context,
    })?;
    let stream = Stream::new()?;
    let mut scratch = MtpScratch::new(1, context, KvCacheDtype::Fp8)?;
    let mut hidden = DeviceBuffer::<u16>::zeroed(HIDDEN_SIZE)?;
    let mut draft_hidden = DeviceBuffer::<u16>::zeroed(HIDDEN_SIZE)?;
    let mut draft_logits = DeviceBuffer::<f32>::zeroed(VOCAB_SIZE)?;
    let mut argmax = qwc_cuda::sampling::Argmax::new(1, VOCAB_SIZE)?;
    println!(
        "голова: {:.0} МБ весов, KV одного слоя: {:.0} МБ\n",
        head.resident_bytes() as f64 / 1e6,
        scratch.resident_bytes() as f64 / 1e6
    );

    println!("{:>12} {:>9} {:>12}", "случай", "черновиков", "acceptance");
    println!("{:->12} {:->9} {:->12}", "", "", "");
    let (mut hits, mut total) = (0usize, 0usize);
    let mut draft_times = Vec::new();
    let mut head_times = Vec::new();
    for (id, prompt) in &cases {
        assert!(prompt.len() <= PREFILL_CHUNK_SIZE, "{id}: промпт длиннее чанка");
        assert!(prompt.len() + max_new <= context);
        // Новый прогон — своя история у головы: KV-кэш черновика обнуляется.
        scratch = MtpScratch::new(1, context, KvCacheDtype::Fp8)?;
        executor.prefill_sequence(&weights, prompt, 0, 0)?;
        let mut next = executor.argmax_to_host(1)?;
        let (mut case_hits, mut case_total) = (0usize, 0usize);
        let mut predicted: Option<u32> = None;

        for step in 0..max_new {
            let position = (prompt.len() + step - 1) as u32;
            if step == 0 {
                executor.copy_prefill_hidden_row(prompt.len() - 1, &mut hidden)?;
            } else {
                executor.copy_decode_hidden_row(0, &mut hidden)?;
            }
            // Черновик: голова видит h_t и токен, который модель только что
            // выдала, и предсказывает следующий.
            // Голова и словарная проекция меряются порознь: lm_head читает
            // 1.27 ГБ, то есть может стоить дороже самой головы.
            let started = Event::new()?;
            let after_head = Event::new()?;
            let finished = Event::new()?;
            started.record(&stream)?;
            mtp::draft(&head, &weights, &mut scratch, &hidden, &next, &[position],
                       &mut draft_hidden, &stream)?;
            after_head.record(&stream)?;
            weights.lm_head.logits(&draft_hidden, &mut draft_logits, 1, &stream)?;
            argmax.sample(&draft_logits, 1, &stream)?;
            finished.record(&stream)?;
            finished.synchronize()?;
            draft_times.push(Event::elapsed_ms(&started, &after_head)? as f64);
            head_times.push(Event::elapsed_ms(&after_head, &finished)? as f64);
            let draft = argmax.to_host(1)?[0];

            if let Some(guess) = predicted.replace(draft) {
                case_total += 1;
                if guess == next[0] {
                    case_hits += 1;
                }
            }

            executor.decode(&weights, &next, &[(prompt.len() + step) as u32])?;
            next = executor.argmax_to_host(1)?;
        }
        hits += case_hits;
        total += case_total;
        println!(
            "{id:>12} {case_total:>9} {:>12.3}",
            case_hits as f64 / case_total.max(1) as f64
        );
    }

    draft_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    head_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "\nacceptance всего: {:.3} ({total} черновиков)",
        hits as f64 / total.max(1) as f64
    );
    let head_median = draft_times[draft_times.len() / 2];
    let vocab_median = head_times[head_times.len() / 2];
    println!(
        "черновой шаг: голова {head_median:.3} мс + lm_head {vocab_median:.3} мс = {:.3} мс",
        head_median + vocab_median
    );
    println!(
        "  голова: {:.0} ГБ/с из 1790 возможных",
        head.resident_bytes() as f64 / 1e9 / (head_median * 1e-3)
    );
    Ok(())
}

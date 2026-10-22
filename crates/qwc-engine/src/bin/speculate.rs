//! Speculative decode on the MTP head: drafts by the head, verification by the model.
//!
//! `cargo run --release -p qwc-engine --bin speculate -- --prompt-ids 9707 --max-new 64 --draft 2`
//!
//! A step works like this: the head proposes k tokens in a chain (each costs
//! reading its 849 MB plus the vocab projection), the model verifies them in one
//! pass over the weights, and the longest matching prefix plus one
//! of its own tokens on top is accepted. The DeltaNet state meanwhile lives in the draft slot:
//! until it's known how many were accepted, the real one can't be touched.
//!
//! The `--verify` flag runs the same thing with plain greedy and compares tokens:
//! speculation has no right to change the output, only the speed.

use qwc_core::arch::{HIDDEN_SIZE, VOCAB_SIZE};
use qwc_cuda::paged_attention::KvCacheDtype;
use qwc_cuda::sampling::Argmax;
use qwc_cuda::{DeviceBuffer, Stream};
use qwc_engine::mtp::{self, MtpHead, MtpScratch};
use qwc_engine::{Executor, ExecutorConfig, ModelWeights};
use qwc_model::Checkpoint;
use std::path::PathBuf;
use std::time::Instant;

/// Verification goes through the prefill path, regular decode through its own, and their logits
/// don't match bit for bit.
///
/// Measured on twelve prompt lengths: max|dlogit| of row 0 lies in
/// 0.076..0.318 and depends on context, not draft depth — on
/// one prompt the value is the same at depth 1, 2 and 3. For comparison:
/// a position shift by one gave 1.24..1.36, and W4A4 with quantized
/// activations 0.64. A tolerance of 0.5 separates rounding from both with margin.
const ROW_ZERO_TOLERANCE: f32 = 0.50;

/// Tolerance for a tie in logits. Observed argmax flips had a gap of
/// at most 0.023 — an order of magnitude below the tolerance. This isn't proof of
/// no error, but a bound "beyond what rounding gave":
/// a position shift flipped the argmax with a gap of 0.66.
const TIE_TOLERANCE: f32 = 0.25;

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

    /// A draft chain: the first goes off the model's hidden state, then
    /// the head continues off its own output and its own prediction.
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
            // SAFETY: the buffers live in self and don't overlap with the output.
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
    let mut diagnose = false;
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
            "--diagnose" => diagnose = true,
            "--profile" => profile = true,
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }
    assert!((1..=7).contains(&depth), "draft depth 1..7");

    let checkpoint = Checkpoint::open(&model)?;
    let weights = ModelWeights::load(&checkpoint)?;
    let mut executor = Executor::new(ExecutorConfig {
        max_batch: 1,
        max_context: context,
    })?;
    let mut speculator = Speculator::new(&checkpoint, context)?;

    // The speculative run.
    executor.prefill_sequence(&weights, &prompt, 0, 0)?;
    executor.copy_prefill_hidden_row(prompt.len() - 1, &mut speculator.hidden)?;
    let mut next = executor.argmax_to_host(1)?[0];
    let mut generated: Vec<u32> = Vec::with_capacity(max_new);
    // Where each emitted token came from: a model row or an accepted draft.
    let mut origin: Vec<&'static str> = Vec::with_capacity(max_new);
    let pool_blocks: Vec<u32> = (0..executor.max_blocks() as u32).collect();
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

        // The bench drives one sequence and owns the whole pool,
        // so its pages are simply the first max_blocks of the pool.
        let truth = executor.verify_speculation(&weights, &batch, position, 0, &pool_blocks)?;
        verify_seconds += phase.elapsed().as_secs_f64();
        let phase = Instant::now();
        // The longest matching prefix is accepted; after it comes the token
        // the model emitted on the first diverging row — it's correct by
        // construction, so a step always gives at least one token.
        let mut accepted = 0usize;
        while accepted < drafts.len() && truth[accepted] == drafts[accepted] {
            accepted += 1;
        }
        let rows = accepted + 1;
        executor.commit_speculation(&weights, batch.len(), rows, 0)?;
        commit_seconds += phase.elapsed().as_secs_f64();

        generated.push(next);
        origin.push("model");
        for &draft in &drafts[..accepted] {
            generated.push(draft);
            origin.push("draft");
        }
        next = truth[accepted];
        position += rows;
        accepted_total += accepted;
        steps += 1;
        executor.copy_prefill_hidden_row(rows - 1, &mut speculator.hidden)?;
    }
    let elapsed = started.elapsed().as_secs_f64();
    generated.truncate(max_new);
    origin.truncate(max_new);

    println!("prompt: {prompt:?}");
    println!("generated: {generated:?}");
    println!(
        "\n{} tokens in {:.3} s = {:.1} tok/s, {steps} steps, {accepted_total} drafts accepted ({:.2} per step)",
        generated.len(),
        elapsed,
        generated.len() as f64 / elapsed,
        accepted_total as f64 / steps as f64
    );
    println!(
        "tokens per step: {:.2} (draft depth {depth})",
        (accepted_total + steps) as f64 / steps as f64
    );
    if profile {
        let phases = executor.profile_totals()?;
        let total: f32 = phases.iter().map(|(_, ms)| ms).sum();
        println!("\nverification step breakdown (sum over all steps, {total:.1} ms):");
        for (name, ms) in phases.iter().take(10) {
            println!("  {name:<24} {ms:8.2} ms  {:5.1}%", ms / total * 100.0);
        }
    }
    println!(
        "per step: drafts {:.2} ms, verification {:.2} ms, commit {:.2} ms, other {:.2} ms",
        draft_seconds * 1e3 / steps as f64,
        verify_seconds * 1e3 / steps as f64,
        commit_seconds * 1e3 / steps as f64,
        (elapsed - draft_seconds - verify_seconds - commit_seconds) * 1e3 / steps as f64
    );

    if diagnose {
        let worst = row_zero_logit_gap(&checkpoint, &weights, &prompt, context, depth)?;
        println!("verification row 0 vs regular decode: max|dlogit| {worst:.5}");
    }

    if verify {
        // Verification row 0 computes exactly the same as regular decode: the same
        // token, the same position, the same input state. A position shift shows
        // here immediately, but in tokens far from always: with a shift by one
        // the logit divergence reached 1.36, while the argmax most often survived.
        let worst = row_zero_logit_gap(&checkpoint, &weights, &prompt, context, depth)?;
        assert!(
            worst <= ROW_ZERO_TOLERANCE,
            "verification row 0 diverged from decode by {worst:.4} with tolerance \
             {ROW_ZERO_TOLERANCE} — that's not rounding, it's a different computation"
        );
        println!("\nverification row 0 vs decode: max|dlogit| {worst:.5}");

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
            "plain greedy: {:.1} tok/s, speedup {:.2}x",
            max_new as f64 / plain_seconds,
            (max_new as f64 / elapsed) / (max_new as f64 / plain_seconds)
        );

        let first = generated
            .iter()
            .zip(reference.iter())
            .position(|(left, right)| left != right);
        let Some(index) = first else {
            println!("output matched plain greedy token for token");
            return Ok(());
        };

        // Verification and decode take different paths and don't match bit for bit.
        // As long as the model really doesn't care which of two tokens to take,
        // the divergence is a property, not an error; the logit gap tells them apart.
        let mut probe = Executor::new(ExecutorConfig {
            max_batch: 1,
            max_context: context,
        })?;
        probe.prefill_sequence(&weights, &prompt, 0, 0)?;
        let mut probe_token = probe.argmax_to_host(1)?[0];
        for step in 0..index {
            probe.decode(&weights, &[probe_token], &[(prompt.len() + step) as u32])?;
            probe_token = probe.argmax_to_host(1)?[0];
        }
        let logits = probe.logits_to_host(0)?;
        let gap = logits[reference[index] as usize] - logits[generated[index] as usize];
        assert!(
            gap <= TIE_TOLERANCE,
            "speculation changed the output not on a tie: token {index}, greedy {} vs \
             {} for speculation, logit gap {gap:.4} with tolerance {TIE_TOLERANCE} — \
             this is a bug, not an optimization",
            reference[index],
            generated[index]
        );
        // The source matters: a model row points at the verification pass itself,
        // an accepted draft also at the acceptance check.
        println!(
            "divergence on a tie: token {index} (position {}, source {}), greedy {} \
             vs {} for speculation, logit gap {gap:.5}",
            prompt.len() + index,
            origin[index],
            reference[index],
            generated[index]
        );
    }
    Ok(())
}

/// How much row 0 of speculative verification diverges from regular decode.
///
/// Their input is the same — the same token at the same position from the same
/// state — so anything showing up here beyond rounding means
/// verification computes the wrong thing.
fn row_zero_logit_gap(
    checkpoint: &Checkpoint,
    weights: &ModelWeights,
    prompt: &[u32],
    context: usize,
    depth: usize,
) -> Result<f32, Box<dyn std::error::Error>> {
    let config = || ExecutorConfig {
        max_batch: 1,
        max_context: context,
    };

    let mut spec = Executor::new(config())?;
    spec.prefill_sequence(weights, prompt, 0, 0)?;
    let first = spec.argmax_to_host(1)?[0];
    let mut speculator = Speculator::new(checkpoint, context)?;
    spec.copy_prefill_hidden_row(prompt.len() - 1, &mut speculator.hidden)?;
    let drafts = speculator.draft(weights, first, prompt.len(), depth)?;
    let mut batch = vec![first];
    batch.extend_from_slice(&drafts);
    let blocks: Vec<u32> = (0..spec.max_blocks() as u32).collect();
    spec.verify_speculation(weights, &batch, prompt.len(), 0, &blocks)?;
    let spec_logits = spec.logits_to_host(0)?;

    let mut plain = Executor::new(config())?;
    plain.prefill_sequence(weights, prompt, 0, 0)?;
    let plain_first = plain.argmax_to_host(1)?[0];
    assert_eq!(first, plain_first, "prefill diverged from itself");
    plain.decode(weights, &[plain_first], &[prompt.len() as u32])?;
    let plain_logits = plain.logits_to_host(0)?;

    Ok(spec_logits
        .iter()
        .zip(plain_logits.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max))
}

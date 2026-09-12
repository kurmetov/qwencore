//! Steady-state decode latency for one persistent sequence.
//!
//! Emits JSON so `bench/latency.py` can score qwc and the reference engines
//! with one methodology. Two token counts are generated and the per-token cost
//! is taken from the slope between them, which cancels prefill, the first-token
//! cost and every fixed per-request overhead. The in-process median of the
//! decode step is reported alongside it as a cross-check.
//!
//! ```
//! cargo run --release -p qwc-engine --bin latency -- --model PATH --repeats 5
//! ```

use qwc_core::arch::VOCAB_SIZE;
use qwc_engine::{Executor, ExecutorConfig, ModelWeights};
use qwc_model::Checkpoint;
use std::path::PathBuf;
use std::time::Instant;

struct Args {
    model: PathBuf,
    prompt: Vec<u32>,
    short: usize,
    long: usize,
    repeats: usize,
    warmup: usize,
    batch: usize,
    context: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse()?;
    qwc_cuda::Device::init(0)?;
    qwc_cuda::set_memory_limit(24 * 1_000_000_000)?;
    let checkpoint = Checkpoint::open(&args.model)?;
    let load_started = Instant::now();
    let weights = ModelWeights::load(&checkpoint)?;
    let load_seconds = load_started.elapsed().as_secs_f64();
    let mut executor = Executor::new(ExecutorConfig {
        max_batch: args.batch,
        max_context: args.context,
    })?;

    // The first pass captures the CUDA graph and touches every resident
    // allocation; timing it would measure setup, not the steady state.
    for _ in 0..args.warmup {
        run_once(&mut executor, &weights, &args, args.long)?;
    }

    let mut short_ms = Vec::with_capacity(args.repeats);
    let mut long_ms = Vec::with_capacity(args.repeats);
    let mut step_medians = Vec::with_capacity(args.repeats);
    for _ in 0..args.repeats {
        let (elapsed, _) = run_once(&mut executor, &weights, &args, args.short)?;
        short_ms.push(elapsed);
        let (elapsed, median) = run_once(&mut executor, &weights, &args, args.long)?;
        long_ms.push(elapsed);
        step_medians.push(median);
    }

    let short = median(&mut short_ms);
    let long = median(&mut long_ms);
    let slope = (long - short) / (args.long - args.short) as f64;
    let step = median(&mut step_medians);
    println!(
        concat!(
            "{{\"engine\":\"qwc\",\"version\":\"{}\",\"batch\":{},",
            "\"prompt_tokens\":{},\"short_tokens\":{},\"long_tokens\":{},",
            "\"repeats\":{},\"load_seconds\":{:.3},",
            "\"short_ms\":{:.4},\"long_ms\":{:.4},",
            "\"decode_ms_per_token\":{:.4},\"tokens_per_second\":{:.2},",
            "\"in_process_step_ms\":{:.4}}}"
        ),
        env!("CARGO_PKG_VERSION"),
        args.batch,
        args.prompt.len(),
        args.short,
        args.long,
        args.repeats,
        load_seconds,
        short,
        long,
        slope,
        args.batch as f64 * 1000.0 / slope,
        step,
    );
    Ok(())
}

/// Returns the wall time of prefill plus `tokens` decode steps, and the median
/// of the decode steps inside that run.
fn run_once(
    executor: &mut Executor,
    weights: &ModelWeights,
    args: &Args,
    tokens: usize,
) -> Result<(f64, f64), Box<dyn std::error::Error>> {
    let started = Instant::now();
    for slot in 0..args.batch {
        executor.prefill_sequence(weights, &args.prompt, 0, slot)?;
    }
    let mut next = executor.argmax_to_host(args.batch)?;
    let mut steps = Vec::with_capacity(tokens);
    for step in 0..tokens {
        let positions = vec![(args.prompt.len() + step) as u32; args.batch];
        let step_started = Instant::now();
        executor.decode(weights, &next, &positions)?;
        next = executor.argmax_to_host(args.batch)?;
        steps.push(step_started.elapsed().as_secs_f64() * 1e3);
    }
    let elapsed = started.elapsed().as_secs_f64() * 1e3;
    Ok((elapsed, median(&mut steps)))
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn parse() -> Result<Args, Box<dyn std::error::Error>> {
    let mut model = home().join("models/Qwen3.8-27B-QUASAR-NVFP4");
    let mut prompt = vec![9707u32, 11, 1879, 374, 264, 1273, 315, 279, 1849];
    let mut short = 8usize;
    let mut long = 72usize;
    let mut repeats = 5usize;
    let mut warmup = 2usize;
    let mut batch = 1usize;
    let mut context = 2048usize;
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
            "--short" => short = args.next().ok_or("--short")?.parse()?,
            "--long" => long = args.next().ok_or("--long")?.parse()?,
            "--repeats" => repeats = args.next().ok_or("--repeats")?.parse()?,
            "--warmup" => warmup = args.next().ok_or("--warmup")?.parse()?,
            "--batch" => batch = args.next().ok_or("--batch")?.parse()?,
            "--context" => context = args.next().ok_or("--context")?.parse()?,
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }
    if long <= short {
        return Err("--long must exceed --short".into());
    }
    if prompt.is_empty() || prompt.iter().any(|&id| (id as usize) >= VOCAB_SIZE) {
        return Err("invalid prompt".into());
    }
    if prompt.len() + long > context {
        return Err("prompt + --long exceeds --context".into());
    }
    Ok(Args {
        model,
        prompt,
        short,
        long,
        repeats,
        warmup,
        batch,
        context,
    })
}

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/root".into()))
}

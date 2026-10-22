//! Token loop: tensor-core prefill, then greedy generation.
//! `cargo run --release -p qwc-engine --bin generate -- --prompt-ids 9707,11 --max-new 32`

use qwc_core::arch::VOCAB_SIZE;
use qwc_engine::{Executor, ExecutorConfig, ModelWeights};
use qwc_model::Checkpoint;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut dir: Option<PathBuf> = None;
    let mut prompt: Vec<u32> = vec![9707];
    let mut max_new = 32usize;
    let mut context = 2048usize;
    let mut batch = 1usize;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--prompt-ids" => {
                prompt = args
                    .next()
                    .ok_or("--prompt-ids needs a list")?
                    .split(',')
                    .map(|id| id.trim().parse::<u32>())
                    .collect::<Result<_, _>>()?;
            }
            "--max-new" => max_new = args.next().ok_or("--max-new")?.parse()?,
            "--context" => context = args.next().ok_or("--context")?.parse()?,
            "--batch" => batch = args.next().ok_or("--batch")?.parse()?,
            other => dir = Some(PathBuf::from(other)),
        }
    }
    let dir = dir.unwrap_or_else(|| home().join("models/Qwen3.8-27B-QUASAR-NVFP4"));
    assert!(!prompt.is_empty(), "prompt is empty");
    assert!((1..=qwc_engine::executor::MAX_BATCH).contains(&batch));
    assert!(prompt.iter().all(|&id| (id as usize) < VOCAB_SIZE));
    assert!(
        prompt.len() + max_new <= context,
        "prompt + generation > context"
    );

    qwc_cuda::Device::init(0)?;
    qwc_cuda::set_memory_limit(24 * 1_000_000_000)?;

    let checkpoint = Checkpoint::open(&dir)?;
    let started = Instant::now();
    let weights = ModelWeights::load(&checkpoint)?;
    println!(
        "weights: {:.2} GB in {:.1} s",
        weights.stats().resident_bytes() as f64 / 1e9,
        started.elapsed().as_secs_f64()
    );

    let mut executor = Executor::new(ExecutorConfig {
        max_batch: batch,
        max_context: context,
    })?;
    println!(
        "cache and state: {:.2} GB for {batch} x {context} tokens\n",
        executor.cache_bytes() as f64 / 1e9
    );

    let prompt_started = Instant::now();
    for sequence in 0..batch {
        executor.prefill_sequence(&weights, &prompt, 0, sequence)?;
    }
    let prompt_seconds = prompt_started.elapsed().as_secs_f64();

    let mut generated = vec![Vec::with_capacity(max_new); batch];
    let mut latencies = Vec::with_capacity(max_new);
    let mut sampling = Vec::with_capacity(max_new);
    let mut next = executor.argmax_to_host(batch)?;
    for step in 0..max_new {
        for sequence in 0..batch {
            generated[sequence].push(next[sequence]);
        }
        let position = (prompt.len() + step) as u32;
        let positions = vec![position; batch];
        let step_started = Instant::now();
        executor.decode(&weights, &next, &positions)?;
        latencies.push(step_started.elapsed().as_secs_f64() * 1e3);
        let sample_started = Instant::now();
        next = executor.argmax_to_host(batch)?;
        sampling.push(sample_started.elapsed().as_secs_f64() * 1e3);
    }

    println!("prompt: {prompt:?}");
    println!("output (sequence 0): {:?}", generated[0]);
    if batch > 1 {
        println!(
            "all {batch} identical prompts matched: {}",
            generated.windows(2).all(|pair| pair[0] == pair[1])
        );
    }
    println!(
        "\nprompt {} tokens in {:.1} ms ({:.3} ms/token)",
        prompt.len() * batch,
        prompt_seconds * 1e3,
        prompt_seconds * 1e3 / (prompt.len() * batch) as f64
    );
    latencies.sort_by(f64::total_cmp);
    sampling.sort_by(f64::total_cmp);
    let median = latencies[latencies.len() / 2];
    let sample_median = sampling[sampling.len() / 2];
    println!(
        "decode step: median {:.2} ms ({:.1} tok/s goodput), min {:.2}, max {:.2}",
        median,
        batch as f64 * 1000.0 / median,
        latencies[0],
        latencies[latencies.len() - 1]
    );
    println!(
        "GPU argmax: median {:.2} ms  =>  full cycle {:.2} ms ({:.1} tok/s goodput)",
        sample_median,
        median + sample_median,
        batch as f64 * 1000.0 / (median + sample_median)
    );

    let logits = executor.logits_to_host(0)?;
    let finite = logits.iter().filter(|value| value.is_finite()).count();
    println!(
        "\nlogits: {finite} finite of {VOCAB_SIZE}, range [{:.2}, {:.2}]",
        logits.iter().cloned().fold(f32::INFINITY, f32::min),
        logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );
    let mut top: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
    top.sort_by(|a, b| b.1.total_cmp(&a.1));
    println!("top-5 of the last step: {:?}", &top[..5]);
    Ok(())
}

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/root".into()))
}

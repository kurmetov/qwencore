//! End-to-end continuous-batching loop backed by `qwc-runtime`.

use qwc_core::arch::VOCAB_SIZE;
use qwc_engine::{Executor, ExecutorConfig, ModelWeights, PREFILL_CHUNK_SIZE};
use qwc_model::Checkpoint;
use qwc_runtime::{BatchLayout, CacheManager, Request, Scheduler, SchedulerConfig, SeqId};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut dir: Option<PathBuf> = None;
    let mut prompt = vec![760, 6511, 314, 9338, 369];
    let mut requests = 8usize;
    let mut max_new = 8usize;
    let mut context = 2048usize;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--prompt-ids" => {
                prompt = args
                    .next()
                    .ok_or("--prompt-ids requires a list")?
                    .split(',')
                    .map(|id| id.trim().parse::<u32>())
                    .collect::<Result<_, _>>()?;
            }
            "--requests" => requests = args.next().ok_or("--requests")?.parse()?,
            "--max-new" => max_new = args.next().ok_or("--max-new")?.parse()?,
            "--context" => context = args.next().ok_or("--context")?.parse()?,
            other => dir = Some(PathBuf::from(other)),
        }
    }
    assert!(!prompt.is_empty());
    assert!(prompt.iter().all(|&token| (token as usize) < VOCAB_SIZE));
    assert!(requests > 0 && requests <= qwc_engine::executor::MAX_BATCH);
    assert!(prompt.len() + max_new <= context);

    qwc_cuda::Device::init(0)?;
    qwc_cuda::set_memory_limit(24 * 1_000_000_000)?;
    let dir = dir.unwrap_or_else(|| home().join("models/Qwen3.8-27B-QUASAR-NVFP4"));
    let checkpoint = Checkpoint::open(&dir)?;
    let weights = ModelWeights::load(&checkpoint)?;
    let max_blocks = context.div_ceil(qwc_cuda::paged_attention::PAGE_SIZE);
    let mut executor = Executor::new(ExecutorConfig {
        max_batch: requests,
        max_context: context,
    })?;
    let cache = CacheManager::new(
        requests,
        requests * max_blocks,
        qwc_cuda::paged_attention::PAGE_SIZE,
    );
    let mut scheduler = Scheduler::new(SchedulerConfig::new(requests, PREFILL_CHUNK_SIZE), cache);
    let prompts: HashMap<SeqId, Vec<u32>> = (1..=requests as u32)
        .map(|id| (id, prompt.clone()))
        .collect();
    let mut generated: HashMap<SeqId, Vec<u32>> = (1..=requests as u32)
        .map(|id| (id, Vec::with_capacity(max_new)))
        .collect();
    let mut pending = HashMap::<SeqId, u32>::new();
    for id in 1..=requests as u32 {
        scheduler.submit(Request {
            id,
            prompt_tokens: prompt.len(),
            max_new_tokens: max_new,
        })?;
    }

    let started = Instant::now();
    let mut completed = 0usize;
    while completed < requests {
        let batch = scheduler
            .next_batch()?
            .ok_or("scheduler made no progress")?;
        let layout = BatchLayout::build(&batch, scheduler.cache())?;
        let mut input = Vec::with_capacity(layout.num_tokens());
        for &id in &batch.decode {
            input.push(*pending.get(&id).ok_or("decode token is missing")?);
        }
        for chunk in &batch.prefill {
            input.extend_from_slice(&prompts[&chunk.id][chunk.offset..chunk.offset + chunk.tokens]);
        }

        for &id in &batch.decode {
            generated.get_mut(&id).unwrap().push(pending[&id]);
        }
        for (id, token) in executor.execute_layout(&weights, &layout, &input)? {
            pending.insert(id, token);
        }
        completed += scheduler.complete_batch(&[])?.len();
    }

    let elapsed = started.elapsed().as_secs_f64();
    let first = &generated[&1];
    println!("sequence 1: {first:?}");
    println!(
        "all {requests} sequences match: {}",
        generated.values().all(|tokens| tokens == first)
    );
    println!(
        "{} generated tokens in {:.1} ms ({:.1} tok/s goodput)",
        requests * max_new,
        elapsed * 1e3,
        (requests * max_new) as f64 / elapsed
    );
    Ok(())
}

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/root".into()))
}

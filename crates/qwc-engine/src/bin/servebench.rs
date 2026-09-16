//! Serving throughput under concurrency: N requests, at most C in flight.
//!
//! Every request is submitted at t=0, so the engine runs saturated and the
//! number this reports is steady-state throughput, not a latency figure. TTFT
//! therefore includes queueing for everything past the first C requests, which
//! is what a real backlog looks like.
//!
//! Emits JSON for `bench/servebench.py` to score against the reference engines.

use qwc_core::arch::{KV_ELEMS_PER_TOKEN, VOCAB_SIZE};
use qwc_cuda::delta_net::DeltaStateMode;
use qwc_cuda::paged_attention::KvCacheDtype;
use qwc_engine::DecodeLinearMode;
use qwc_engine::{Executor, ExecutorConfig, ModelWeights, PREFILL_CHUNK_SIZE};
use qwc_model::Checkpoint;
use qwc_runtime::{BatchLayout, CacheManager, Request, Scheduler, SchedulerConfig, SeqId};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

struct Args {
    model: PathBuf,
    requests: usize,
    concurrency: usize,
    prompt_tokens: usize,
    max_new: usize,
    context: usize,
    prefill_chunk: usize,
    memory_limit: usize,
    kv_cache_bytes: usize,
    kv_cache: KvCacheDtype,
    delta_state: DeltaStateMode,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse()?;
    qwc_cuda::Device::init(0)?;
    qwc_cuda::set_memory_limit(args.memory_limit)?;
    let checkpoint = Checkpoint::open(&args.model)?;
    let load_started = Instant::now();
    let weights = ModelWeights::load(&checkpoint)?;
    let load_seconds = load_started.elapsed().as_secs_f64();

    let max_blocks = args.context.div_ceil(qwc_cuda::paged_attention::PAGE_SIZE);
    let bytes_per_block = qwc_cuda::paged_attention::PAGE_SIZE
        * KV_ELEMS_PER_TOKEN
        * args.kv_cache.bytes_per_element();
    let kv_pool_blocks = args.kv_cache_bytes / bytes_per_block;
    if kv_pool_blocks < max_blocks {
        return Err(format!(
            "KV pool has {kv_pool_blocks} blocks, but --context {} needs at least {max_blocks}; increase --kv-cache-gb",
            args.context
        )
        .into());
    }
    let mut executor = Executor::new_with_pool_options(
        ExecutorConfig {
            max_batch: args.concurrency,
            max_context: args.context,
        },
        args.kv_cache,
        DecodeLinearMode::Auto,
        args.delta_state,
        Some(kv_pool_blocks),
    )?;
    let cache_gb = executor.cache_bytes() as f64 / 1e9;
    let cache = CacheManager::new(
        args.concurrency,
        executor.kv_pool_blocks(),
        qwc_cuda::paged_attention::PAGE_SIZE,
    );
    let mut scheduler = Scheduler::new(
        SchedulerConfig::new(args.concurrency, args.prefill_chunk),
        cache,
    );

    // Same length for every request, but distinct content: identical prompts
    // would let a prefix-caching engine skip prefill entirely and report a
    // throughput no engine can actually sustain.
    let prompts: HashMap<SeqId, Vec<u32>> = (1..=args.requests as u32)
        .map(|id| (id, prompt_for(id, args.prompt_tokens)))
        .collect();

    for id in 1..=args.requests as u32 {
        scheduler.submit(Request {
            id,
            prompt_tokens: args.prompt_tokens,
            max_new_tokens: args.max_new,
        })?;
    }

    let mut pending = HashMap::<SeqId, u32>::new();
    let mut produced = HashMap::<SeqId, usize>::new();
    let mut first_token_ms = HashMap::<SeqId, f64>::new();
    let mut last_token_at = HashMap::<SeqId, Instant>::new();
    let mut inter_token_ms: Vec<f64> = Vec::with_capacity(args.requests * args.max_new);

    let started = Instant::now();
    let mut completed = 0usize;
    let mut steps = 0usize;
    // One full pass over the weights per step.
    let mut passes = 0usize;
    while completed < args.requests {
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
            *produced.entry(id).or_insert(0) += 1;
        }

        for (id, token) in executor.execute_layout(&weights, &layout, &input)? {
            let now = Instant::now();
            first_token_ms
                .entry(id)
                .or_insert_with(|| (now - started).as_secs_f64() * 1e3);
            if let Some(previous) = last_token_at.insert(id, now) {
                inter_token_ms.push((now - previous).as_secs_f64() * 1e3);
            }
            pending.insert(id, token);
        }
        // A step is one forward pass now, mixed or not.
        passes += 1;
        completed += scheduler.complete_batch(&[])?.len();
        steps += 1;
    }
    let elapsed = started.elapsed().as_secs_f64();

    let output_tokens: usize = produced.values().sum();
    let mut ttft: Vec<f64> = first_token_ms.values().copied().collect();
    println!(
        concat!(
            "{{\"engine\":\"qwc\",\"version\":\"{}\",\"requests\":{},\"concurrency\":{},",
            "\"prompt_tokens\":{},\"max_new_tokens\":{},\"context\":{},\"kv_cache\":\"{}\",",
            "\"load_seconds\":{:.3},\"cache_gb\":{:.2},\"kv_pool_blocks\":{},\"engine_steps\":{},\"weight_passes\":{},",
            "\"wall_seconds\":{:.4},\"output_tokens\":{},",
            "\"output_tokens_per_second\":{:.2},\"requests_per_second\":{:.3},",
            "\"ttft_ms_p50\":{:.2},\"ttft_ms_p95\":{:.2},",
            "\"itl_ms_p50\":{:.3},\"itl_ms_p95\":{:.3}}}"
        ),
        env!("CARGO_PKG_VERSION"),
        args.requests,
        args.concurrency,
        args.prompt_tokens,
        args.max_new,
        args.context,
        args.kv_cache.as_str(),
        load_seconds,
        cache_gb,
        executor.kv_pool_blocks(),
        steps,
        passes,
        elapsed,
        output_tokens,
        output_tokens as f64 / elapsed,
        args.requests as f64 / elapsed,
        percentile(&mut ttft, 0.50),
        percentile(&mut ttft, 0.95),
        percentile(&mut inter_token_ms, 0.50),
        percentile(&mut inter_token_ms, 0.95),
    );
    Ok(())
}

/// Distinct token ids per request; mirrors `prompt_ids` in bench/servebench.py
/// so every engine sees byte-identical prompts.
fn prompt_for(id: u32, length: usize) -> Vec<u32> {
    (0..length)
        .map(|index| 1000 + ((id.wrapping_mul(7919) + index as u32) % 20000))
        .collect()
}

fn percentile(values: &mut [f64], fraction: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(f64::total_cmp);
    let index = ((values.len() - 1) as f64 * fraction).round() as usize;
    values[index]
}

fn parse() -> Result<Args, Box<dyn std::error::Error>> {
    let mut model = home().join("models/Qwen3.8-27B-QUASAR-NVFP4");
    let mut requests = 200usize;
    let mut concurrency = 32usize;
    let mut prompt_tokens = 256usize;
    let mut max_new = 128usize;
    let mut context = 2048usize;
    // Бюджет токенов на шаг; потолок — ёмкость арены префилла.
    let mut prefill_chunk = PREFILL_CHUNK_SIZE;
    let mut memory_limit = 28_000_000_000usize;
    // Physical pages are shared. Five GB is enough for four full 32K
    // sequences or many short requests without reserving 32K for each slot.
    let mut kv_cache_bytes = 5_000_000_000usize;
    let mut kv_cache = KvCacheDtype::Fp8;
    let mut delta_state = DeltaStateMode::Wy;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model = PathBuf::from(args.next().ok_or("--model")?),
            "--requests" => requests = args.next().ok_or("--requests")?.parse()?,
            "--concurrency" => concurrency = args.next().ok_or("--concurrency")?.parse()?,
            "--prompt-tokens" => prompt_tokens = args.next().ok_or("--prompt-tokens")?.parse()?,
            "--max-new" => max_new = args.next().ok_or("--max-new")?.parse()?,
            "--context" => context = args.next().ok_or("--context")?.parse()?,
            "--prefill-chunk" => {
                prefill_chunk = args.next().ok_or("--prefill-chunk")?.parse()?
            }
            "--delta-state" => {
                delta_state = match args
                    .next()
                    .ok_or("--delta-state needs bf16, fp32 or wy")?
                    .as_str()
                {
                    "bf16" => DeltaStateMode::Bf16,
                    "fp32" => DeltaStateMode::Fp32,
                    "wy" => DeltaStateMode::Wy,
                    other => {
                        return Err(format!("неизвестное значение --delta-state: {other}").into());
                    }
                };
            }
            "--kv-cache" => {
                kv_cache = match args.next().ok_or("--kv-cache needs fp8 or bf16")?.as_str() {
                    "fp8" => KvCacheDtype::Fp8,
                    "bf16" => KvCacheDtype::Bf16,
                    other => return Err(format!("unsupported --kv-cache value: {other}").into()),
                };
            }
            "--memory-limit-gb" => {
                let gb: f64 = args.next().ok_or("--memory-limit-gb")?.parse()?;
                memory_limit = (gb * 1e9) as usize;
            }
            "--kv-cache-gb" => {
                let gb: f64 = args.next().ok_or("--kv-cache-gb")?.parse()?;
                if !gb.is_finite() || gb <= 0.0 {
                    return Err("--kv-cache-gb must be positive".into());
                }
                kv_cache_bytes = (gb * 1e9) as usize;
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }
    if concurrency == 0 || concurrency > qwc_engine::executor::MAX_BATCH {
        return Err(format!(
            "--concurrency must be 1..={}",
            qwc_engine::executor::MAX_BATCH
        )
        .into());
    }
    if requests < concurrency {
        return Err("--requests must be at least --concurrency".into());
    }
    if !(1..=PREFILL_CHUNK_SIZE).contains(&prefill_chunk) {
        return Err(format!("--prefill-chunk must be 1..={PREFILL_CHUNK_SIZE}").into());
    }
    if prompt_tokens == 0 || prompt_tokens + max_new > context {
        return Err("prompt + generation must fit in --context".into());
    }
    if prompt_tokens >= VOCAB_SIZE {
        return Err("prompt too long".into());
    }
    Ok(Args {
        model,
        requests,
        concurrency,
        prompt_tokens,
        max_new,
        context,
        prefill_chunk,
        memory_limit,
        kv_cache_bytes,
        kv_cache,
        delta_state,
    })
}

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/root".into()))
}

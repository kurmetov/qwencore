//! Streaming benchmark for Qwen3.8 GQA paged attention.
//! `cargo run --release -p qwc-cuda --bin attentionbench`

use qwc_core::arch::{ATTN_HEAD_DIM, NUM_ATTN_HEADS, NUM_FULL_LAYERS, NUM_KV_HEADS};
use qwc_cuda::paged_attention::{self, DecodeKernel, PAGE_SIZE, PagedAttentionWorkspace};
use qwc_cuda::{Device, DeviceBuffer, Event, Stream, bf16};

#[allow(clippy::too_many_arguments)]
fn measure(
    kernel: DecodeKernel,
    partitions: usize,
    caches: &[(DeviceBuffer<u8>, DeviceBuffer<u8>)],
    query: &DeviceBuffer<u16>,
    tables: &DeviceBuffer<u32>,
    lengths: &DeviceBuffer<u32>,
    num_blocks: usize,
    max_blocks: usize,
    batch: usize,
    context: usize,
    output: &mut DeviceBuffer<u16>,
    stream: &Stream,
) -> Result<(f64, usize), Box<dyn std::error::Error>> {
    let mut workspace =
        PagedAttentionWorkspace::with_partitions(kernel, batch, context, partitions)?;
    let partitions = workspace.partitions();
    let mut enqueue_all = || -> qwc_cuda::Result<()> {
        for (key, value) in caches {
            paged_attention::decode_fp8(
                query,
                key,
                value,
                num_blocks,
                tables,
                lengths,
                max_blocks,
                output,
                &mut workspace,
                batch,
                context,
                stream,
            )?;
        }
        Ok(())
    };
    for _ in 0..3 {
        enqueue_all()?;
    }
    stream.synchronize()?;

    let iterations = if batch * context >= 32 * 2_048 { 3 } else { 7 };
    let mut samples = [0.0f32; 7];
    for sample in &mut samples {
        let (start, end) = (Event::new()?, Event::new()?);
        start.record(stream)?;
        for _ in 0..iterations {
            enqueue_all()?;
        }
        end.record(stream)?;
        end.synchronize()?;
        *sample = Event::elapsed_ms(&start, &end)?;
    }
    samples.sort_by(f32::total_cmp);
    Ok((
        samples[3] as f64 * 1e3 / iterations as f64 / NUM_FULL_LAYERS as f64,
        partitions,
    ))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::init(0)?;
    let stream = Stream::new()?;
    let requested: Vec<usize> = std::env::args()
        .skip(1)
        .map(|value| value.parse())
        .collect::<Result<_, _>>()?;
    assert!(requested.is_empty() || requested.len() == 2);
    println!(
        "FP8 paged GQA decode, RTX 5090 ({} SM), {} distinct layer caches",
        device.sm_count, NUM_FULL_LAYERS
    );
    println!(
        "  {:>5} | {:>7} | {:>14} | {:>14} | {:>9}",
        "batch", "context", "query-head", "shared-KV", "winner"
    );
    println!(
        "  {:->5}-+-{:->7}-+-{:->14}-+-{:->14}-+-{:->9}",
        "", "", "", "", ""
    );

    for (batch, context) in [
        (1usize, 512usize),
        (1, 2_048),
        (1, 8_192),
        (1, 32_768),
        (2, 2_048),
        (2, 8_192),
        (2, 32_768),
        (4, 2_048),
        (4, 8_192),
        (8, 2_048),
        (8, 8_192),
        (16, 2_048),
        (16, 8_192),
        (32, 512),
        (32, 2_048),
        (32, 8_192),
    ] {
        if requested.len() == 2 && (batch != requested[0] || context != requested[1]) {
            continue;
        }
        let blocks_per_sequence = context.div_ceil(PAGE_SIZE);
        let num_blocks = batch * blocks_per_sequence;
        let cache_elements = num_blocks * NUM_KV_HEADS * PAGE_SIZE * ATTN_HEAD_DIM;
        let caches: Vec<(DeviceBuffer<u8>, DeviceBuffer<u8>)> = (0..NUM_FULL_LAYERS)
            .map(|_| {
                Ok((
                    DeviceBuffer::zeroed(cache_elements)?,
                    DeviceBuffer::zeroed(cache_elements)?,
                ))
            })
            .collect::<qwc_cuda::Result<_>>()?;
        let tables: Vec<u32> = (0..batch)
            .flat_map(|sequence| {
                (0..blocks_per_sequence)
                    .map(move |block| (sequence * blocks_per_sequence + block) as u32)
            })
            .collect();
        let lengths = vec![context as u32; batch];
        let device_tables = DeviceBuffer::from_slice(&tables)?;
        let device_lengths = DeviceBuffer::from_slice(&lengths)?;
        let query = DeviceBuffer::from_slice(&vec![
            bf16::from_f32(0.5);
            batch * NUM_ATTN_HEADS * ATTN_HEAD_DIM
        ])?;
        let mut output = DeviceBuffer::<u16>::zeroed(query.len())?;
        let query_default =
            paged_attention::partition_count(DecodeKernel::QueryHead, batch, context);
        let shared_default =
            paged_attention::partition_count(DecodeKernel::SharedKv, batch, context);
        let mut query_candidates = vec![1, query_default / 2, query_default];
        let mut shared_candidates = vec![1, shared_default / 2, shared_default];
        query_candidates.retain(|&parts| parts > 0);
        shared_candidates.retain(|&parts| parts > 0);
        query_candidates.sort_unstable();
        shared_candidates.sort_unstable();
        query_candidates.dedup();
        shared_candidates.dedup();

        let mut query_best = (f64::INFINITY, 0usize);
        for parts in query_candidates {
            let measured = measure(
                DecodeKernel::QueryHead,
                parts,
                &caches,
                &query,
                &device_tables,
                &device_lengths,
                num_blocks,
                blocks_per_sequence,
                batch,
                context,
                &mut output,
                &stream,
            )?;
            if measured.0 < query_best.0 {
                query_best = measured;
            }
        }
        let mut shared_best = (f64::INFINITY, 0usize);
        for parts in shared_candidates {
            let measured = measure(
                DecodeKernel::SharedKv,
                parts,
                &caches,
                &query,
                &device_tables,
                &device_lengths,
                num_blocks,
                blocks_per_sequence,
                batch,
                context,
                &mut output,
                &stream,
            )?;
            if measured.0 < shared_best.0 {
                shared_best = measured;
            }
        }
        let (query_us, query_parts) = query_best;
        let (shared_us, shared_parts) = shared_best;
        let winner = if query_us < shared_us {
            "query"
        } else {
            "shared"
        };
        println!(
            "  {batch:>5} | {context:>7} | {query_us:>8.2} us/{query_parts:<2} | \
             {shared_us:>8.2} us/{shared_parts:<2} | {winner:>9}",
        );
    }
    Ok(())
}

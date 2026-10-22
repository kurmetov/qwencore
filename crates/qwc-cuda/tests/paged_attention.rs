//! Correctness of direct and split-context FP8 paged attention.

use qwc_core::arch::{ATTN_HEAD_DIM, NUM_ATTN_HEADS, NUM_KV_HEADS};
use qwc_cuda::paged_attention::{
    self, DecodeKernel, KvCacheDtype, PAGE_SIZE, PackedAttentionWorkspace, PackedShape,
    PagedAttentionWorkspace,
};
use qwc_cuda::{DeviceBuffer, Stream, bf16};

fn run_case(context: usize, expect_split: bool) {
    const BATCH: usize = 1;
    let max_blocks = context.div_ceil(PAGE_SIZE);
    let num_blocks = max_blocks;
    let cache_elements = num_blocks * NUM_KV_HEADS * PAGE_SIZE * ATTN_HEAD_DIM;
    let key_codes = [0x18u8, 0x98, 0x20, 0xa0, 0x28, 0xa8, 0x00];
    let value_codes = [0x30u8, 0xb0, 0x28, 0xa8, 0x20, 0xa0, 0x00];
    let key_cache: Vec<u8> = (0..cache_elements)
        .map(|index| key_codes[(index * 5 + index / ATTN_HEAD_DIM) % key_codes.len()])
        .collect();
    let value_cache: Vec<u8> = (0..cache_elements)
        .map(|index| value_codes[(index * 3 + index / 17) % value_codes.len()])
        .collect();
    // Reverse physical pages so the test cannot accidentally ignore the table.
    let block_tables: Vec<u32> = (0..max_blocks as u32).rev().collect();
    let context_lengths = vec![context as u32];
    let query: Vec<u16> = (0..NUM_ATTN_HEADS * ATTN_HEAD_DIM)
        .map(|index| {
            let value = ((index * 13 % 29) as f32 - 14.0) / 28.0;
            bf16::from_f32(value)
        })
        .collect();

    let mut expected = vec![0.0f32; query.len()];
    paged_attention::reference::decode_fp8(
        &query,
        &key_cache,
        &value_cache,
        &block_tables,
        &context_lengths,
        max_blocks,
        &mut expected,
        BATCH,
    );

    let stream = Stream::new().unwrap();
    let device_query = DeviceBuffer::from_slice(&query).unwrap();
    let device_key = DeviceBuffer::from_slice(&key_cache).unwrap();
    let device_value = DeviceBuffer::from_slice(&value_cache).unwrap();
    let device_tables = DeviceBuffer::from_slice(&block_tables).unwrap();
    let device_lengths = DeviceBuffer::from_slice(&context_lengths).unwrap();
    let mut query_gate = vec![0u16; BATCH * NUM_ATTN_HEADS * ATTN_HEAD_DIM * 2];
    for head in 0..NUM_ATTN_HEADS {
        let base = head * ATTN_HEAD_DIM * 2 + ATTN_HEAD_DIM;
        for dimension in 0..ATTN_HEAD_DIM {
            query_gate[base + dimension] = bf16::from_f32((dimension % 17) as f32 * 0.1 - 0.8);
        }
    }
    let device_query_gate = DeviceBuffer::from_slice(&query_gate).unwrap();
    for kernel in [DecodeKernel::QueryHead, DecodeKernel::SharedKv] {
        for gated in [false, true] {
            let mut output = DeviceBuffer::<u16>::zeroed(query.len()).unwrap();
            let mut workspace =
                PagedAttentionWorkspace::with_kernel(kernel, BATCH, context).unwrap();
            assert_eq!(workspace.partitions() > 1, expect_split);
            if gated {
                paged_attention::decode_fp8_gated(
                    &device_query,
                    &device_query_gate,
                    &device_key,
                    &device_value,
                    num_blocks,
                    &device_tables,
                    &device_lengths,
                    max_blocks,
                    &mut output,
                    &mut workspace,
                    BATCH,
                    context,
                    &stream,
                )
            } else {
                paged_attention::decode_fp8(
                    &device_query,
                    &device_key,
                    &device_value,
                    num_blocks,
                    &device_tables,
                    &device_lengths,
                    max_blocks,
                    &mut output,
                    &mut workspace,
                    BATCH,
                    context,
                    &stream,
                )
            }
            .unwrap();
            stream.synchronize().unwrap();

            for (index, (&actual, &expected)) in
                output.to_vec().unwrap().iter().zip(&expected).enumerate()
            {
                let actual = bf16::to_f32(actual);
                let expected = if gated {
                    let head = index / ATTN_HEAD_DIM;
                    let dimension = index % ATTN_HEAD_DIM;
                    let raw = bf16::to_f32(
                        query_gate[head * ATTN_HEAD_DIM * 2 + ATTN_HEAD_DIM + dimension],
                    );
                    expected / (1.0 + (-raw).exp())
                } else {
                    expected
                };
                assert!(
                    (actual - expected).abs() <= 0.004 + expected.abs() * 0.004,
                    "kernel={kernel:?}, gated={gated}, context={context}, index={index}: GPU={actual}, CPU={expected}"
                );
            }
        }
    }
}

#[test]
fn direct_partition_matches_cpu() {
    run_case(23, false);
}

#[test]
fn split_context_and_paging_match_cpu() {
    run_case(257, true);
}

#[test]
fn bf16_cache_matches_cpu_for_direct_and_split_contexts() {
    const BATCH: usize = 1;
    for context in [23usize, 257] {
        let max_blocks = context.div_ceil(PAGE_SIZE);
        let cache_elements = max_blocks * NUM_KV_HEADS * PAGE_SIZE * ATTN_HEAD_DIM;
        let key: Vec<u16> = (0..cache_elements)
            .map(|index| bf16::from_f32((index % 31) as f32 / 20.0 - 0.75))
            .collect();
        let value: Vec<u16> = (0..cache_elements)
            .map(|index| bf16::from_f32((index % 23) as f32 / 16.0 - 0.6))
            .collect();
        let key_bytes: Vec<u8> = key.iter().flat_map(|item| item.to_ne_bytes()).collect();
        let value_bytes: Vec<u8> = value.iter().flat_map(|item| item.to_ne_bytes()).collect();
        let tables: Vec<u32> = (0..max_blocks as u32).rev().collect();
        let lengths = [context as u32];
        let query: Vec<u16> = (0..NUM_ATTN_HEADS * ATTN_HEAD_DIM)
            .map(|index| bf16::from_f32((index % 29) as f32 / 24.0 - 0.6))
            .collect();
        let mut expected = vec![0.0f32; query.len()];
        paged_attention::reference::decode_bf16(
            &query,
            &key,
            &value,
            &tables,
            &lengths,
            max_blocks,
            &mut expected,
            BATCH,
        );

        let stream = Stream::new().unwrap();
        let device_query = DeviceBuffer::from_slice(&query).unwrap();
        let device_key = DeviceBuffer::from_slice(&key_bytes).unwrap();
        let device_value = DeviceBuffer::from_slice(&value_bytes).unwrap();
        let device_tables = DeviceBuffer::from_slice(&tables).unwrap();
        let device_lengths = DeviceBuffer::from_slice(&lengths).unwrap();
        let gate =
            DeviceBuffer::from_slice(&vec![0u16; NUM_ATTN_HEADS * ATTN_HEAD_DIM * 2]).unwrap();
        let mut output = DeviceBuffer::<u16>::zeroed(query.len()).unwrap();
        let mut workspace = PagedAttentionWorkspace::new(BATCH, context).unwrap();
        paged_attention::decode_gated(
            &device_query,
            &gate,
            &device_key,
            &device_value,
            max_blocks,
            &device_tables,
            &device_lengths,
            max_blocks,
            &mut output,
            &mut workspace,
            BATCH,
            context,
            KvCacheDtype::Bf16,
            &stream,
        )
        .unwrap();
        stream.synchronize().unwrap();
        for (actual, expected) in output.to_vec().unwrap().iter().zip(expected) {
            // Zero gate means sigmoid(0) = 0.5.
            let actual = bf16::to_f32(*actual);
            let expected = expected * 0.5;
            assert!((actual - expected).abs() <= 0.004 + expected.abs() * 0.004);
        }
    }
}

/// A tile of prefill rows must match row-wise decode: each row
/// sees exactly its own causal prefix, and the shared KV pass must not
/// mix rows with each other.
#[test]
fn prefill_tile_matches_rowwise_decode() {
    // More rows than warps in a block, to check both an incomplete tail
    // tile and a page boundary inside one tile.
    for rows in [19usize, 127, 200] {
        prefill_case(rows, PrefillKernel::Tiled, 0);
    }
}

/// The same for the tensor-core kernel: causality, page order, an incomplete
/// tail tile and a page boundary inside a tile.
#[test]
fn prefill_mma_matches_rowwise_decode() {
    // 601 rows is 19 KV tiles and a context longer than the prefill chunk: exactly
    // the shape on which the end-to-end eval diverged.
    for rows in [19usize, 127, 200, 601] {
        prefill_case(rows, PrefillKernel::Mma, 0);
    }
}

/// A chunk filling the whole arena: 2048 rows is 128 query tiles and a context twice
/// as long as anything the cases above check. A CPU reference on this shape would take
/// minutes, so the tiled kernel acts as the arbiter here: it's checked against
/// CPU on small shapes, and here only the row computation is checked.
#[test]
fn prefill_mma_matches_the_tiled_kernel_on_a_full_arena_chunk() {
    const ROWS: usize = 2048;
    let contexts: Vec<u32> = (0..ROWS).map(|row| (1 + row) as u32).collect();
    let max_context = *contexts.iter().max().unwrap() as usize;
    let max_blocks = max_context.div_ceil(PAGE_SIZE).max(1);
    let num_blocks = max_blocks;
    let cache_elements = num_blocks * NUM_KV_HEADS * PAGE_SIZE * ATTN_HEAD_DIM;
    let key_codes = [0x18u8, 0x98, 0x20, 0xa0, 0x28, 0xa8, 0x00];
    let value_codes = [0x30u8, 0xb0, 0x28, 0xa8, 0x20, 0xa0, 0x00];
    let key_cache: Vec<u8> = (0..cache_elements)
        .map(|index| key_codes[(index * 5 + index / ATTN_HEAD_DIM) % key_codes.len()])
        .collect();
    let value_cache: Vec<u8> = (0..cache_elements)
        .map(|index| value_codes[(index * 3 + index / 17) % value_codes.len()])
        .collect();
    let one_table: Vec<u32> = (0..max_blocks as u32).rev().collect();
    let block_tables: Vec<u32> = (0..ROWS).flat_map(|_| one_table.clone()).collect();
    let query: Vec<u16> = (0..ROWS * NUM_ATTN_HEADS * ATTN_HEAD_DIM)
        .map(|index| bf16::from_f32(((index * 13 % 29) as f32 - 14.0) / 28.0))
        .collect();
    let gate = vec![0u16; query.len() * 2];

    let stream = Stream::new().unwrap();
    let device_query = DeviceBuffer::from_slice(&query).unwrap();
    let device_gate = DeviceBuffer::from_slice(&gate).unwrap();
    let device_key = DeviceBuffer::from_slice(&key_cache).unwrap();
    let device_value = DeviceBuffer::from_slice(&value_cache).unwrap();
    let device_tables = DeviceBuffer::from_slice(&block_tables).unwrap();
    let device_lengths = DeviceBuffer::from_slice(&contexts).unwrap();

    let mut mma = DeviceBuffer::<u16>::zeroed(query.len()).unwrap();
    paged_attention::prefill_gated_mma(
        &device_query, &device_gate, &device_key, &device_value, num_blocks,
        &device_tables, &device_lengths, max_blocks, &mut mma, ROWS, 0,
        KvCacheDtype::Fp8, &stream,
    )
    .unwrap();
    let mut tiled = DeviceBuffer::<u16>::zeroed(query.len()).unwrap();
    paged_attention::prefill_gated(
        &device_query, &device_gate, &device_key, &device_value, num_blocks,
        &device_tables, &device_lengths, max_blocks, &mut tiled, ROWS, 0,
        KvCacheDtype::Fp8, &stream,
    )
    .unwrap();
    stream.synchronize().unwrap();

    for (index, (&a, &b)) in mma
        .to_vec()
        .unwrap()
        .iter()
        .zip(&tiled.to_vec().unwrap())
        .enumerate()
    {
        let (a, b) = (bf16::to_f32(a), bf16::to_f32(b));
        assert!(
            (a - b).abs() <= 0.004 + b.abs() * 0.02,
            "row {}, index {index}: MMA={a}, tiled={b}",
            index / (NUM_ATTN_HEADS * ATTN_HEAD_DIM)
        );
    }
}

#[derive(Clone, Copy, PartialEq)]
enum PrefillKernel {
    Tiled,
    Mma,
}

/// The second and following prompt chunks: the rows are the same, but their causal prefix
/// doesn't start at one. In the engine this shape occurs on any
/// prompt longer than a chunk, and no test covered it before.
#[test]
fn prefill_mma_matches_rowwise_decode_on_a_later_chunk() {
    for (rows, start) in [(89usize, 512usize), (16, 512), (200, 1024)] {
        prefill_case(rows, PrefillKernel::Mma, start);
    }
}

fn prefill_case(rows: usize, kernel: PrefillKernel, context_start: usize) {
    #[allow(non_snake_case)]
    let ROWS = rows;
    let first_context = context_start + 1;
    let contexts: Vec<u32> = (0..ROWS).map(|row| (first_context + row) as u32).collect();
    let max_context = *contexts.iter().max().unwrap() as usize;
    let max_blocks = max_context.div_ceil(PAGE_SIZE);
    let num_blocks = max_blocks;
    let cache_elements = num_blocks * NUM_KV_HEADS * PAGE_SIZE * ATTN_HEAD_DIM;

    let key_codes = [0x18u8, 0x98, 0x20, 0xa0, 0x28, 0xa8, 0x00];
    let value_codes = [0x30u8, 0xb0, 0x28, 0xa8, 0x20, 0xa0, 0x00];
    let key_cache: Vec<u8> = (0..cache_elements)
        .map(|index| key_codes[(index * 5 + index / ATTN_HEAD_DIM) % key_codes.len()])
        .collect();
    let value_cache: Vec<u8> = (0..cache_elements)
        .map(|index| value_codes[(index * 3 + index / 17) % value_codes.len()])
        .collect();

    // One sequence: all rows share the same block table.
    // The page order is reversed so the test can't be passed by ignoring it.
    let one_table: Vec<u32> = (0..max_blocks as u32).rev().collect();
    let block_tables: Vec<u32> = (0..ROWS).flat_map(|_| one_table.clone()).collect();

    let query: Vec<u16> = (0..ROWS * NUM_ATTN_HEADS * ATTN_HEAD_DIM)
        .map(|index| bf16::from_f32(((index * 13 % 29) as f32 - 14.0) / 28.0))
        .collect();
    let mut query_gate = vec![0u16; ROWS * NUM_ATTN_HEADS * ATTN_HEAD_DIM * 2];
    for row in 0..ROWS {
        for head in 0..NUM_ATTN_HEADS {
            let base = (row * NUM_ATTN_HEADS + head) * ATTN_HEAD_DIM * 2 + ATTN_HEAD_DIM;
            for dimension in 0..ATTN_HEAD_DIM {
                query_gate[base + dimension] =
                    bf16::from_f32(((dimension + row) % 17) as f32 * 0.1 - 0.8);
            }
        }
    }

    let mut expected = vec![0.0f32; query.len()];
    paged_attention::reference::decode_fp8(
        &query,
        &key_cache,
        &value_cache,
        &block_tables,
        &contexts,
        max_blocks,
        &mut expected,
        ROWS,
    );

    let stream = Stream::new().unwrap();
    let device_query = DeviceBuffer::from_slice(&query).unwrap();
    let device_gate = DeviceBuffer::from_slice(&query_gate).unwrap();
    let device_key = DeviceBuffer::from_slice(&key_cache).unwrap();
    let device_value = DeviceBuffer::from_slice(&value_cache).unwrap();
    let device_tables = DeviceBuffer::from_slice(&block_tables).unwrap();
    let device_lengths = DeviceBuffer::from_slice(&contexts).unwrap();
    let mut output = DeviceBuffer::<u16>::zeroed(query.len()).unwrap();

    let launch = match kernel {
        PrefillKernel::Tiled => paged_attention::prefill_gated,
        PrefillKernel::Mma => paged_attention::prefill_gated_mma,
    };
    launch(
        &device_query,
        &device_gate,
        &device_key,
        &device_value,
        num_blocks,
        &device_tables,
        &device_lengths,
        max_blocks,
        &mut output,
        ROWS,
        0,
        KvCacheDtype::Fp8,
        &stream,
    )
    .unwrap();
    stream.synchronize().unwrap();

    for (index, (&actual, &reference)) in output.to_vec().unwrap().iter().zip(&expected).enumerate()
    {
        let actual = bf16::to_f32(actual);
        let raw = bf16::to_f32(query_gate[(index / ATTN_HEAD_DIM) * ATTN_HEAD_DIM * 2
            + ATTN_HEAD_DIM
            + index % ATTN_HEAD_DIM]);
        let expected = reference / (1.0 + (-raw).exp());
        assert!(
            (actual - expected).abs() <= 0.004 + expected.abs() * 0.004,
            "row {}, index {index}: GPU={actual}, CPU={expected}",
            index / (NUM_ATTN_HEADS * ATTN_HEAD_DIM)
        );
    }
}

/// The same input, but via the old path: the decode kernel on prefill rows. If it
/// doesn't match the reference, then the divergence in eval is not a regression of the new
/// kernel but a fix of the old one.
#[test]
fn rowwise_decode_on_prefill_rows_matches_cpu() {
    const ROWS: usize = 19;
    let contexts: Vec<u32> = (0..ROWS).map(|row| (1 + row) as u32).collect();
    let max_context = *contexts.iter().max().unwrap() as usize;
    let max_blocks = max_context.div_ceil(PAGE_SIZE).max(1);
    let num_blocks = max_blocks;
    let cache_elements = num_blocks * NUM_KV_HEADS * PAGE_SIZE * ATTN_HEAD_DIM;

    let key_codes = [0x18u8, 0x98, 0x20, 0xa0, 0x28, 0xa8, 0x00];
    let value_codes = [0x30u8, 0xb0, 0x28, 0xa8, 0x20, 0xa0, 0x00];
    let key_cache: Vec<u8> = (0..cache_elements)
        .map(|index| key_codes[(index * 5 + index / ATTN_HEAD_DIM) % key_codes.len()])
        .collect();
    let value_cache: Vec<u8> = (0..cache_elements)
        .map(|index| value_codes[(index * 3 + index / 17) % value_codes.len()])
        .collect();
    let one_table: Vec<u32> = (0..max_blocks as u32).rev().collect();
    let block_tables: Vec<u32> = (0..ROWS).flat_map(|_| one_table.clone()).collect();
    let query: Vec<u16> = (0..ROWS * NUM_ATTN_HEADS * ATTN_HEAD_DIM)
        .map(|index| bf16::from_f32(((index * 13 % 29) as f32 - 14.0) / 28.0))
        .collect();

    let mut expected = vec![0.0f32; query.len()];
    paged_attention::reference::decode_fp8(
        &query,
        &key_cache,
        &value_cache,
        &block_tables,
        &contexts,
        max_blocks,
        &mut expected,
        ROWS,
    );

    let stream = Stream::new().unwrap();
    let device_query = DeviceBuffer::from_slice(&query).unwrap();
    let device_key = DeviceBuffer::from_slice(&key_cache).unwrap();
    let device_value = DeviceBuffer::from_slice(&value_cache).unwrap();
    let device_tables = DeviceBuffer::from_slice(&block_tables).unwrap();
    let device_lengths = DeviceBuffer::from_slice(&contexts).unwrap();
    let mut output = DeviceBuffer::<u16>::zeroed(query.len()).unwrap();
    let mut workspace = PagedAttentionWorkspace::new(ROWS, max_context).unwrap();

    paged_attention::decode_fp8(
        &device_query,
        &device_key,
        &device_value,
        num_blocks,
        &device_tables,
        &device_lengths,
        max_blocks,
        &mut output,
        &mut workspace,
        ROWS,
        max_context,
        &stream,
    )
    .unwrap();
    stream.synchronize().unwrap();

    let mut worst = 0.0f32;
    let mut worst_index = 0usize;
    for (index, (&actual, &reference)) in output.to_vec().unwrap().iter().zip(&expected).enumerate()
    {
        let difference = (bf16::to_f32(actual) - reference).abs();
        if difference > worst {
            worst = difference;
            worst_index = index;
        }
    }
    let reference = expected[worst_index];
    assert!(
        worst <= 0.004 + reference.abs() * 0.004,
        "row {}, index {worst_index}: divergence {worst}, reference {reference}",
        worst_index / (NUM_ATTN_HEADS * ATTN_HEAD_DIM)
    );
}

/// A direct comparison of the paths with each other: the reference accepts both within tolerance,
/// but the engine runs them through 64 layers, and what matters is how much they diverge from each
/// other, not how close each is to the CPU.
#[test]
fn prefill_tile_and_rowwise_decode_agree_within_bf16() {
    for bf16_cache in [false, true] {
        agreement_case(bf16_cache);
    }
}

fn agreement_case(bf16_cache: bool) {
    const ROWS: usize = 127;
    let contexts: Vec<u32> = (0..ROWS).map(|row| (1 + row) as u32).collect();
    let max_context = *contexts.iter().max().unwrap() as usize;
    let max_blocks = max_context.div_ceil(PAGE_SIZE).max(1);
    let num_blocks = max_blocks;
    let cache_elements = num_blocks * NUM_KV_HEADS * PAGE_SIZE * ATTN_HEAD_DIM;

    let key_codes = [0x18u8, 0x98, 0x20, 0xa0, 0x28, 0xa8, 0x00];
    let value_codes = [0x30u8, 0xb0, 0x28, 0xa8, 0x20, 0xa0, 0x00];
    let bytes_per_element = if bf16_cache { 2 } else { 1 };
    let key_cache: Vec<u8> = (0..cache_elements * bytes_per_element)
        .map(|index| key_codes[(index * 5 + index / ATTN_HEAD_DIM) % key_codes.len()])
        .collect();
    let value_cache: Vec<u8> = (0..cache_elements * bytes_per_element)
        .map(|index| value_codes[(index * 3 + index / 17) % value_codes.len()])
        .collect();
    let one_table: Vec<u32> = (0..max_blocks as u32).rev().collect();
    let block_tables: Vec<u32> = (0..ROWS).flat_map(|_| one_table.clone()).collect();
    let query: Vec<u16> = (0..ROWS * NUM_ATTN_HEADS * ATTN_HEAD_DIM)
        .map(|index| bf16::from_f32(((index * 13 % 29) as f32 - 14.0) / 28.0))
        .collect();

    let stream = Stream::new().unwrap();
    let device_query = DeviceBuffer::from_slice(&query).unwrap();
    let device_key = DeviceBuffer::from_slice(&key_cache).unwrap();
    let device_value = DeviceBuffer::from_slice(&value_cache).unwrap();
    let device_tables = DeviceBuffer::from_slice(&block_tables).unwrap();
    let device_lengths = DeviceBuffer::from_slice(&contexts).unwrap();
    let gate = vec![0u16; ROWS * NUM_ATTN_HEADS * ATTN_HEAD_DIM * 2];
    let device_gate = DeviceBuffer::from_slice(&gate).unwrap();

    let dtype = if bf16_cache { KvCacheDtype::Bf16 } else { KvCacheDtype::Fp8 };
    let mut tiled = DeviceBuffer::<u16>::zeroed(query.len()).unwrap();
    paged_attention::prefill_gated(
        &device_query, &device_gate, &device_key, &device_value, num_blocks,
        &device_tables, &device_lengths, max_blocks, &mut tiled, ROWS, 0,
        dtype, &stream,
    )
    .unwrap();

    let mut rowwise = DeviceBuffer::<u16>::zeroed(query.len()).unwrap();
    // The engine creates the workspace for the configured max context, not
    // the actual one, so on a short prompt the old path still went
    // through the partitioned kernel. We reproduce exactly that.
    let mut workspace = PagedAttentionWorkspace::new(ROWS, 4096).unwrap();
    paged_attention::decode_gated(
        &device_query, &device_gate, &device_key, &device_value, num_blocks,
        &device_tables, &device_lengths, max_blocks, &mut rowwise, &mut workspace,
        ROWS, max_context, dtype, &stream,
    )
    .unwrap();
    stream.synchronize().unwrap();

    let (a, b) = (tiled.to_vec().unwrap(), rowwise.to_vec().unwrap());
    let mut worst = 0.0f32;
    let mut worst_index = 0usize;
    for (index, (&x, &y)) in a.iter().zip(&b).enumerate() {
        let difference = (bf16::to_f32(x) - bf16::to_f32(y)).abs();
        if difference > worst {
            worst = difference;
            worst_index = index;
        }
    }
    println!(
        "bf16_cache={bf16_cache}: max path divergence {worst:.3e}, row {}, channel {}",
        worst_index / (NUM_ATTN_HEADS * ATTN_HEAD_DIM),
        worst_index % ATTN_HEAD_DIM
    );
    assert!(worst < 0.02, "paths diverged by {worst}");
}

/// Who is closer to the reference on the shape where the engine showed a divergence: few rows,
/// short context, and the workspace created for the configured max,
/// so the old path splits the context into partitions and merges them.
/// Accuracy of the tensor-core kernel vs row-wise decode against a shared
/// fp32 reference. P is rounded to bf16 before PV, so what matters is not "does it
/// match" but "is it no worse" — on long context too.
#[test]
fn prefill_mma_is_not_less_accurate_than_rowwise_decode() {
    for rows in [63usize, 200, 601] {
        let contexts: Vec<u32> = (0..rows).map(|row| (1 + row) as u32).collect();
        let max_context = *contexts.iter().max().unwrap() as usize;
        let max_blocks = max_context.div_ceil(PAGE_SIZE).max(1);
        let num_blocks = max_blocks;
        let cache_elements = num_blocks * NUM_KV_HEADS * PAGE_SIZE * ATTN_HEAD_DIM;
        let key_codes = [0x18u8, 0x98, 0x20, 0xa0, 0x28, 0xa8, 0x00];
        let value_codes = [0x30u8, 0xb0, 0x28, 0xa8, 0x20, 0xa0, 0x00];
        let key_cache: Vec<u8> = (0..cache_elements)
            .map(|index| key_codes[(index * 5 + index / ATTN_HEAD_DIM) % key_codes.len()])
            .collect();
        let value_cache: Vec<u8> = (0..cache_elements)
            .map(|index| value_codes[(index * 3 + index / 17) % value_codes.len()])
            .collect();
        let one_table: Vec<u32> = (0..max_blocks as u32).rev().collect();
        let block_tables: Vec<u32> = (0..rows).flat_map(|_| one_table.clone()).collect();
        let query: Vec<u16> = (0..rows * NUM_ATTN_HEADS * ATTN_HEAD_DIM)
            .map(|index| bf16::from_f32(((index * 13 % 29) as f32 - 14.0) / 28.0))
            .collect();

        let mut expected = vec![0.0f32; query.len()];
        paged_attention::reference::decode_fp8(
            &query, &key_cache, &value_cache, &block_tables, &contexts,
            max_blocks, &mut expected, rows,
        );

        let stream = Stream::new().unwrap();
        let device_query = DeviceBuffer::from_slice(&query).unwrap();
        let device_key = DeviceBuffer::from_slice(&key_cache).unwrap();
        let device_value = DeviceBuffer::from_slice(&value_cache).unwrap();
        let device_tables = DeviceBuffer::from_slice(&block_tables).unwrap();
        let device_lengths = DeviceBuffer::from_slice(&contexts).unwrap();
        let gate = vec![0u16; rows * NUM_ATTN_HEADS * ATTN_HEAD_DIM * 2];
        let device_gate = DeviceBuffer::from_slice(&gate).unwrap();

        let mut mma = DeviceBuffer::<u16>::zeroed(query.len()).unwrap();
        paged_attention::prefill_gated_mma(
            &device_query, &device_gate, &device_key, &device_value, num_blocks,
            &device_tables, &device_lengths, max_blocks, &mut mma, rows, 0,
            KvCacheDtype::Fp8, &stream,
        )
        .unwrap();

        let mut rowwise = DeviceBuffer::<u16>::zeroed(query.len()).unwrap();
        let mut workspace = PagedAttentionWorkspace::new(rows, 4096).unwrap();
        paged_attention::decode_fp8_gated(
            &device_query, &device_gate, &device_key, &device_value, num_blocks,
            &device_tables, &device_lengths, max_blocks, &mut rowwise, &mut workspace,
            rows, max_context, &stream,
        )
        .unwrap();
        stream.synchronize().unwrap();

        let (a, b) = (mma.to_vec().unwrap(), rowwise.to_vec().unwrap());
        let mut mma_error = 0.0f32;
        let mut row_error = 0.0f32;
        for index in 0..a.len() {
            let (x, y) = (bf16::to_f32(a[index]), bf16::to_f32(b[index]));
            mma_error = mma_error.max((x - expected[index]).abs());
            row_error = row_error.max((y - expected[index]).abs());
        }
        println!("rows {rows:3}: MMA error {mma_error:.3e}, row-wise {row_error:.3e}");
        assert!(
            mma_error <= row_error * 4.0 + 1e-3,
            "rows {rows}: MMA {mma_error:.3e} vs row-wise {row_error:.3e}"
        );
    }
}

#[test]
fn prefill_tile_is_not_less_accurate_than_rowwise_decode() {
    for rows in [5usize, 14, 63] {
        let contexts: Vec<u32> = (0..rows).map(|row| (1 + row) as u32).collect();
        let max_context = *contexts.iter().max().unwrap() as usize;
        let max_blocks = max_context.div_ceil(PAGE_SIZE).max(1);
        let num_blocks = max_blocks;
        let cache_elements = num_blocks * NUM_KV_HEADS * PAGE_SIZE * ATTN_HEAD_DIM;
        let key_codes = [0x18u8, 0x98, 0x20, 0xa0, 0x28, 0xa8, 0x00];
        let value_codes = [0x30u8, 0xb0, 0x28, 0xa8, 0x20, 0xa0, 0x00];
        let key_cache: Vec<u8> = (0..cache_elements)
            .map(|index| key_codes[(index * 5 + index / ATTN_HEAD_DIM) % key_codes.len()])
            .collect();
        let value_cache: Vec<u8> = (0..cache_elements)
            .map(|index| value_codes[(index * 3 + index / 17) % value_codes.len()])
            .collect();
        let one_table: Vec<u32> = (0..max_blocks as u32).rev().collect();
        let block_tables: Vec<u32> = (0..rows).flat_map(|_| one_table.clone()).collect();
        let query: Vec<u16> = (0..rows * NUM_ATTN_HEADS * ATTN_HEAD_DIM)
            .map(|index| bf16::from_f32(((index * 13 % 29) as f32 - 14.0) / 28.0))
            .collect();

        let mut expected = vec![0.0f32; query.len()];
        paged_attention::reference::decode_fp8(
            &query, &key_cache, &value_cache, &block_tables, &contexts,
            max_blocks, &mut expected, rows,
        );

        let stream = Stream::new().unwrap();
        let device_query = DeviceBuffer::from_slice(&query).unwrap();
        let device_key = DeviceBuffer::from_slice(&key_cache).unwrap();
        let device_value = DeviceBuffer::from_slice(&value_cache).unwrap();
        let device_tables = DeviceBuffer::from_slice(&block_tables).unwrap();
        let device_lengths = DeviceBuffer::from_slice(&contexts).unwrap();
        let gate = vec![0u16; rows * NUM_ATTN_HEADS * ATTN_HEAD_DIM * 2];
        let device_gate = DeviceBuffer::from_slice(&gate).unwrap();

        let mut tiled = DeviceBuffer::<u16>::zeroed(query.len()).unwrap();
        paged_attention::prefill_gated(
            &device_query, &device_gate, &device_key, &device_value, num_blocks,
            &device_tables, &device_lengths, max_blocks, &mut tiled, rows, 0,
            KvCacheDtype::Fp8, &stream,
        )
        .unwrap();

        let mut rowwise = DeviceBuffer::<u16>::zeroed(query.len()).unwrap();
        let mut workspace = PagedAttentionWorkspace::new(rows, 4096).unwrap();
        paged_attention::decode_fp8_gated(
            &device_query, &device_gate, &device_key, &device_value, num_blocks,
            &device_tables, &device_lengths, max_blocks, &mut rowwise, &mut workspace,
            rows, max_context, &stream,
        )
        .unwrap();
        stream.synchronize().unwrap();

        let (a, b) = (tiled.to_vec().unwrap(), rowwise.to_vec().unwrap());
        let mut tile_error = 0.0f32;
        let mut row_error = 0.0f32;
        let mut between = 0.0f32;
        for index in 0..a.len() {
            let (x, y) = (bf16::to_f32(a[index]), bf16::to_f32(b[index]));
            tile_error = tile_error.max((x - expected[index]).abs());
            row_error = row_error.max((y - expected[index]).abs());
            between = between.max((x - y).abs());
        }
        println!(
            "rows {rows:3}: partitions {}, tile error {tile_error:.3e}, \
             row-wise error {row_error:.3e}, between them {between:.3e}",
            workspace.partitions()
        );
        assert!(
            tile_error <= row_error * 1.5 + 1e-6,
            "the tiled path is worse than the reference: {tile_error} vs {row_error}"
        );
    }
}

#[test]
fn workspace_partitions_follow_actual_rows() {
    // A 32-slot, 32K executor: a single decode must split
    // the context into partitions rather than go with one, like a full batch.
    let workspace = PagedAttentionWorkspace::new(32, 32_768).unwrap();
    let (_, single) = workspace.plan_for(1);
    let (_, full) = workspace.plan_for(32);
    assert!(single > 1, "batch 1 runs with {single} partition");
    assert_eq!(full, 1);
    let alone = PagedAttentionWorkspace::new(1, 32_768).unwrap();
    assert_eq!(workspace.plan_for(1), alone.plan_for(1));
}

#[test]
fn oversized_workspace_matches_exact_one_bit_for_bit() {
    // Sizing by actual rows: the answer doesn't depend on workspace capacity.
    const CONTEXT: usize = 4_096;
    let max_blocks = CONTEXT.div_ceil(PAGE_SIZE);
    let cache_elements = max_blocks * NUM_KV_HEADS * PAGE_SIZE * ATTN_HEAD_DIM;
    let key: Vec<u8> = (0..cache_elements).map(|index| (index * 7 % 113) as u8 & 0x3f).collect();
    let value: Vec<u8> = (0..cache_elements).map(|index| (index * 5 % 97) as u8 & 0x3f).collect();
    let tables: Vec<u32> = (0..max_blocks as u32).collect();
    let query: Vec<u16> = (0..NUM_ATTN_HEADS * ATTN_HEAD_DIM)
        .map(|index| bf16::from_f32(((index * 11 % 31) as f32 - 15.0) / 30.0))
        .collect();
    let stream = Stream::new().unwrap();
    let device_query = DeviceBuffer::from_slice(&query).unwrap();
    let device_key = DeviceBuffer::from_slice(&key).unwrap();
    let device_value = DeviceBuffer::from_slice(&value).unwrap();
    let device_tables = DeviceBuffer::from_slice(&tables).unwrap();
    let device_lengths = DeviceBuffer::from_slice(&[CONTEXT as u32]).unwrap();
    let mut run = |capacity: usize| {
        let mut workspace = PagedAttentionWorkspace::new(capacity, CONTEXT).unwrap();
        let mut output = DeviceBuffer::<u16>::zeroed(query.len()).unwrap();
        paged_attention::decode_fp8(
            &device_query, &device_key, &device_value, max_blocks, &device_tables,
            &device_lengths, max_blocks, &mut output, &mut workspace, 1, CONTEXT, &stream,
        )
        .unwrap();
        stream.synchronize().unwrap();
        output.to_vec().unwrap()
    };
    assert_eq!(run(1), run(32));
}

/// Packed kernel inputs on an arena of `arena` rows.
///
/// V depends on the physical page number, K is pseudorandom: weights are
/// non-uniform, and a lost partition or a foreign page table shifts
/// the output noticeably, not by noise.
struct PackedInputs {
    query: Vec<u16>,
    query_gate: Vec<u16>,
    key_cache: Vec<u8>,
    value_cache: Vec<u8>,
}

fn packed_inputs(arena: usize, num_blocks: usize) -> PackedInputs {
    let cache_elements = num_blocks * NUM_KV_HEADS * PAGE_SIZE * ATTN_HEAD_DIM;
    let key_codes = [0x18u8, 0x98, 0x20, 0xa0, 0x28, 0xa8, 0x30, 0xb0];
    let value_codes = [0xb8u8, 0xb0, 0xa8, 0x00, 0x28, 0x30, 0x38];
    let key_cache = (0..cache_elements)
        .map(|index| {
            let hash = (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 59;
            key_codes[hash as usize % key_codes.len()]
        })
        .collect();
    let value_cache = (0..cache_elements)
        .map(|index| {
            let physical = index / (NUM_KV_HEADS * PAGE_SIZE * ATTN_HEAD_DIM);
            let level = physical * value_codes.len() / num_blocks;
            value_codes[(level + index % 2 + index / ATTN_HEAD_DIM % 3) % value_codes.len()]
        })
        .collect();
    let query = (0..arena * NUM_ATTN_HEADS * ATTN_HEAD_DIM)
        .map(|index| bf16::from_f32(((index * 13 % 29) as f32 - 14.0) / 28.0))
        .collect();
    let mut query_gate = vec![0u16; arena * NUM_ATTN_HEADS * ATTN_HEAD_DIM * 2];
    for row in 0..arena {
        for head in 0..NUM_ATTN_HEADS {
            let base = (row * NUM_ATTN_HEADS + head) * ATTN_HEAD_DIM * 2 + ATTN_HEAD_DIM;
            for dimension in 0..ATTN_HEAD_DIM {
                query_gate[base + dimension] =
                    bf16::from_f32(((dimension + row + head) % 17) as f32 * 0.1 - 0.8);
            }
        }
    }
    PackedInputs { query, query_gate, key_cache, value_cache }
}

/// The packed kernel on rows `row_base..row_base + rows` of the arena. Returns
/// the output of the whole arena and a CPU reference with the gate.
#[allow(clippy::too_many_arguments)]
fn run_packed(
    inputs: &PackedInputs,
    contexts: &[u32],
    tables: &[u32],
    num_blocks: usize,
    max_blocks: usize,
    shape: PackedShape,
    rows: usize,
    row_base: usize,
    partitions: Option<usize>,
    host_context: usize,
) -> Vec<u16> {
    let arena = contexts.len();
    let stream = Stream::new().unwrap();
    let device_query = DeviceBuffer::from_slice(&inputs.query).unwrap();
    let device_gate = DeviceBuffer::from_slice(&inputs.query_gate).unwrap();
    let device_key = DeviceBuffer::from_slice(&inputs.key_cache).unwrap();
    let device_value = DeviceBuffer::from_slice(&inputs.value_cache).unwrap();
    let device_tables = DeviceBuffer::from_slice(tables).unwrap();
    let device_lengths = DeviceBuffer::from_slice(contexts).unwrap();
    let max_context = *contexts.iter().max().unwrap() as usize;
    let mut workspace = match partitions {
        Some(partitions) => {
            PackedAttentionWorkspace::with_partitions(rows, max_context, partitions).unwrap()
        }
        None => PackedAttentionWorkspace::new(&[shape], rows, max_context).unwrap(),
    };
    let mut output = DeviceBuffer::<u16>::zeroed(arena * NUM_ATTN_HEADS * ATTN_HEAD_DIM).unwrap();
    paged_attention::gated_packed(
        &device_query, &device_gate, &device_key, &device_value, num_blocks,
        &device_tables, &device_lengths, max_blocks, &mut output, shape, rows, row_base,
        host_context, &mut workspace, &stream,
    )
    .unwrap();
    stream.synchronize().unwrap();
    output.to_vec().unwrap()
}

/// Checks rows `row_base..row_base + rows` against the CPU reference; rows outside
/// the segment must stay zero.
#[allow(clippy::too_many_arguments)]
fn check_packed_against_cpu(
    label: &str,
    inputs: &PackedInputs,
    contexts: &[u32],
    tables: &[u32],
    max_blocks: usize,
    rows: usize,
    row_base: usize,
    actual: &[u16],
) {
    let arena = contexts.len();
    let width = NUM_ATTN_HEADS * ATTN_HEAD_DIM;
    let mut expected = vec![0.0f32; arena * width];
    paged_attention::reference::decode_fp8(
        &inputs.query, &inputs.key_cache, &inputs.value_cache, tables, contexts, max_blocks,
        &mut expected, arena,
    );
    let mut worst = 0.0f32;
    for (index, &value) in actual.iter().enumerate() {
        let row = index / width;
        let actual = bf16::to_f32(value);
        if row < row_base || row >= row_base + rows {
            assert_eq!(value, 0, "{label}: row {row} outside the segment was touched");
            continue;
        }
        let raw = bf16::to_f32(
            inputs.query_gate[(index / ATTN_HEAD_DIM) * ATTN_HEAD_DIM * 2
                + ATTN_HEAD_DIM
                + index % ATTN_HEAD_DIM],
        );
        let expected = expected[index] / (1.0 + (-raw).exp());
        worst = worst.max((actual - expected).abs());
        assert!(
            (actual - expected).abs() <= 0.004 + expected.abs() * 0.004,
            "{label}: row {row}, head {}, channel {}: GPU={actual}, CPU={expected}",
            index / ATTN_HEAD_DIM % NUM_ATTN_HEADS,
            index % ATTN_HEAD_DIM
        );
    }
    println!("{label}: max error {worst:.3e}");
}

/// A segment of one sequence after a `start` history, after `row_base`
/// foreign single rows.
fn packed_segment_case(
    rows: usize,
    start: usize,
    row_base: usize,
    partitions: Option<usize>,
    host_context: Option<usize>,
) {
    let arena = row_base + rows;
    let contexts: Vec<u32> = (0..arena)
        .map(|row| if row < row_base { 1 } else { (start + row - row_base + 1) as u32 })
        .collect();
    let max_context = *contexts.iter().max().unwrap() as usize;
    let max_blocks = max_context.div_ceil(PAGE_SIZE);
    let num_blocks = max_blocks;
    let one_table: Vec<u32> = (0..max_blocks as u32).rev().collect();
    let tables: Vec<u32> = (0..arena).flat_map(|_| one_table.clone()).collect();
    let inputs = packed_inputs(arena, num_blocks);
    let actual = run_packed(
        &inputs, &contexts, &tables, num_blocks, max_blocks, PackedShape::Segment, rows,
        row_base, partitions, host_context.unwrap_or(max_context),
    );
    let label = format!("segment of {rows} rows at {start}, P={partitions:?}");
    check_packed_against_cpu(&label, &inputs, &contexts, &tables, max_blocks, rows, row_base, &actual);
}

#[test]
fn packed_segment_matches_cpu() {
    // Verifying three drafts after a long history, after the arena's decode rows.
    packed_segment_case(4, 6_000, 3, None, None);
    // One row, the number of partitions doesn't divide the context.
    packed_segment_case(1, 3_000, 0, Some(7), None);
    // More than 32 partitions: the reduction takes weights with a second batch of lanes.
    packed_segment_case(1, 3_000, 0, Some(45), None);
    // Two and five rows: tiles of 16 and 32 packed rows.
    packed_segment_case(2, 700, 1, None, None);
    packed_segment_case(5, 1_000, 0, Some(3), None);
    // Prompt tail: tiles of ten rows, the first rows have empty late partitions,
    // and the last tile is incomplete.
    packed_segment_case(203, 150, 5, Some(4), None);
    // No splitting, the causal edge inside a tile and inside a page.
    packed_segment_case(130, 0, 0, Some(1), None);
}

#[test]
fn packed_segment_survives_an_underestimated_context() {
    // The host underestimated the context: the last partition still reaches
    // the end of the causal prefix.
    packed_segment_case(4, 4_000, 0, Some(4), Some(2_000));
}

#[test]
fn packed_decode_rows_use_their_own_tables_and_contexts() {
    // Decode rows of different sequences: each has its own page permutation
    // and its own context, including incomplete pages and a single row of
    // one token.
    let contexts: Vec<u32> = vec![1, 63, 64, 65, 3_000, 777, 4_096];
    let arena = contexts.len();
    let max_context = *contexts.iter().max().unwrap() as usize;
    let max_blocks = max_context.div_ceil(PAGE_SIZE);
    let num_blocks = max_blocks + 3;
    let tables: Vec<u32> = (0..arena)
        .flat_map(|row| {
            (0..max_blocks).map(move |block| ((block * 7 + row * 5) % num_blocks) as u32)
        })
        .collect();
    let inputs = packed_inputs(arena, num_blocks);
    for partitions in [None, Some(1), Some(5)] {
        let actual = run_packed(
            &inputs, &contexts, &tables, num_blocks, max_blocks, PackedShape::Rows, arena, 0,
            partitions, max_context,
        );
        let label = format!("decode {arena} rows, P={partitions:?}");
        check_packed_against_cpu(&label, &inputs, &contexts, &tables, max_blocks, arena, 0, &actual);
    }
}

/// The old MMA kernel and the packed one on a common shape; a CPU reference here is
/// too expensive.
fn packed_matches_unsplit(rows: usize, start: usize) {
    let contexts: Vec<u32> = (0..rows).map(|row| (start + row + 1) as u32).collect();
    let max_context = *contexts.iter().max().unwrap() as usize;
    let max_blocks = max_context.div_ceil(PAGE_SIZE);
    let num_blocks = max_blocks;
    let one_table: Vec<u32> = (0..max_blocks as u32).rev().collect();
    let tables: Vec<u32> = (0..rows).flat_map(|_| one_table.clone()).collect();
    let inputs = packed_inputs(rows, num_blocks);
    let packed = run_packed(
        &inputs, &contexts, &tables, num_blocks, max_blocks, PackedShape::Segment, rows, 0,
        None, max_context,
    );

    let stream = Stream::new().unwrap();
    let mut unsplit = DeviceBuffer::<u16>::zeroed(packed.len()).unwrap();
    paged_attention::prefill_gated_mma(
        &DeviceBuffer::from_slice(&inputs.query).unwrap(),
        &DeviceBuffer::from_slice(&inputs.query_gate).unwrap(),
        &DeviceBuffer::from_slice(&inputs.key_cache).unwrap(),
        &DeviceBuffer::from_slice(&inputs.value_cache).unwrap(),
        num_blocks,
        &DeviceBuffer::from_slice(&tables).unwrap(),
        &DeviceBuffer::from_slice(&contexts).unwrap(),
        max_blocks,
        &mut unsplit,
        rows,
        0,
        KvCacheDtype::Fp8,
        &stream,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let unsplit = unsplit.to_vec().unwrap();
    let mut worst = 0.0f32;
    for (index, (&a, &b)) in packed.iter().zip(&unsplit).enumerate() {
        let (a, b) = (bf16::to_f32(a), bf16::to_f32(b));
        worst = worst.max((a - b).abs());
        // The old kernel rounds P to bf16, the packed one to f16.
        assert!(
            (a - b).abs() <= 0.004 + b.abs() * 0.012,
            "{rows} rows at {start}, index {index}: packed={a}, old={b}"
        );
    }
    println!("{rows} rows at {start}: max divergence from the old kernel {worst:.3e}");
}

#[test]
fn packed_matches_unsplit_on_a_long_context() {
    packed_matches_unsplit(4, 30_000);
}

#[test]
fn packed_matches_unsplit_on_a_full_chunk() {
    packed_matches_unsplit(2_048, 1_000);
}

/// Sharp softmax: K up to ±4, q up to ±2, scores of order one. On the test caches
/// above scores are hundredths, softmax is nearly uniform, and the output is the mean of V for any
/// QK error: a mixed-up dim layout wouldn't fail there. Tolerance is
/// a fraction of the reference's largest element, not absolute.
#[test]
fn packed_matches_cpu_on_a_sharp_softmax() {
    for (rows, start, partitions) in [
        (1usize, 200usize, Some(1)),
        (1, 3_000, None),
        (4, 200, Some(1)),
        (4, 3_000, None),
        (8, 200, Some(1)),
        (130, 0, Some(1)),
        (130, 500, Some(3)),
        // The engine's split shapes: partial outputs in f16 are merged by the reduction,
        // and partitions of a sharp softmax have very different maxima.
        (8, 3_000, Some(40)),
        (64, 3_000, Some(7)),
    ] {
        let contexts: Vec<u32> = (0..rows).map(|row| (start + row + 1) as u32).collect();
        let max_context = *contexts.iter().max().unwrap() as usize;
        let max_blocks = max_context.div_ceil(PAGE_SIZE);
        let num_blocks = max_blocks;
        let mut inputs = packed_inputs(rows, num_blocks);
        let codes = [0x40u8, 0xc0, 0x48, 0xc8, 0x38, 0xb8, 0x50, 0xd0, 0x30, 0x00];
        for (index, byte) in inputs.key_cache.iter_mut().enumerate() {
            let hash = (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 58;
            *byte = codes[hash as usize % codes.len()];
        }
        for (index, value) in inputs.query.iter_mut().enumerate() {
            let hash = ((index as u64 + 7).wrapping_mul(0xd6e8_feb8_6659_fd93) >> 40) % 1000;
            *value = bf16::from_f32(hash as f32 / 250.0 - 2.0);
        }
        let one_table: Vec<u32> = (0..max_blocks as u32).rev().collect();
        let tables: Vec<u32> = (0..rows).flat_map(|_| one_table.clone()).collect();
        let actual = run_packed(
            &inputs, &contexts, &tables, num_blocks, max_blocks, PackedShape::Segment, rows, 0,
            partitions, max_context,
        );
        let mut expected = vec![0.0f32; actual.len()];
        paged_attention::reference::decode_fp8(
            &inputs.query, &inputs.key_cache, &inputs.value_cache, &tables, &contexts, max_blocks,
            &mut expected, rows,
        );
        let gated: Vec<f32> = expected
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                let raw = bf16::to_f32(
                    inputs.query_gate[(index / ATTN_HEAD_DIM) * ATTN_HEAD_DIM * 2
                        + ATTN_HEAD_DIM
                        + index % ATTN_HEAD_DIM],
                );
                value / (1.0 + (-raw).exp())
            })
            .collect();
        let scale = gated.iter().fold(0.0f32, |acc, value| acc.max(value.abs()));
        let worst = actual
            .iter()
            .zip(&gated)
            .map(|(&value, &expected)| (bf16::to_f32(value) - expected).abs())
            .fold(0.0f32, f32::max);
        println!("{rows} rows at {start}, P={partitions:?}: error {worst:.2e} at scale {scale:.3}");
        // Rounding the output to bf16 is 0.4% of an element; observed up to 0.3% of scale.
        assert!(worst <= scale * 0.006, "{rows} rows at {start}: {worst} at scale {scale}");
    }
}

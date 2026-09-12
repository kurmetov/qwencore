//! Correctness of direct and split-context FP8 paged attention.

use qwc_core::arch::{ATTN_HEAD_DIM, NUM_ATTN_HEADS, NUM_KV_HEADS};
use qwc_cuda::paged_attention::{
    self, DecodeKernel, KvCacheDtype, PAGE_SIZE, PagedAttentionWorkspace,
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

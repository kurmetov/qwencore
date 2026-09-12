//! Correctness of fused Q/K normalization, partial RoPE and paged FP8 write.

use qwc_core::arch::{ATTN_HEAD_DIM, NUM_ATTN_HEADS, NUM_KV_HEADS, ROPE_DIM};
use qwc_cuda::attention_prepare::AttentionPreprocessor;
use qwc_cuda::nvfp4;
use qwc_cuda::paged_attention::{KvCacheDtype, PAGE_SIZE};
use qwc_cuda::{DeviceBuffer, Stream, bf16};

fn normalized(input: &[u16], weight: &[u16], epsilon: f32) -> Vec<u16> {
    let sum: f32 = input.iter().map(|&value| bf16::to_f32(value).powi(2)).sum();
    let inverse = 1.0 / (sum / input.len() as f32 + epsilon).sqrt();
    input
        .iter()
        .zip(weight)
        .map(|(&value, &weight)| {
            bf16::from_f32(bf16::to_f32(value) * inverse * (1.0 + bf16::to_f32(weight)))
        })
        .collect()
}

fn rotate(value: &[u16], cosine: &[u16], sine: &[u16], dimension: usize) -> f32 {
    let x = bf16::to_f32(value[dimension]);
    if dimension >= ROPE_DIM {
        return x;
    }
    let half = ROPE_DIM / 2;
    let paired = if dimension < half {
        -bf16::to_f32(value[dimension + half])
    } else {
        bf16::to_f32(value[dimension - half])
    };
    x * bf16::to_f32(cosine[dimension]) + paired * bf16::to_f32(sine[dimension])
}

#[test]
fn preprocesses_interleaved_query_gate_and_writes_selected_pages() {
    const BATCH: usize = 2;
    const BLOCKS: usize = 4;
    const EPSILON: f32 = 1e-6;
    let q_weight: Vec<u16> = (0..ATTN_HEAD_DIM)
        .map(|i| bf16::from_f32((i % 11) as f32 * 0.002 - 0.01))
        .collect();
    let k_weight: Vec<u16> = (0..ATTN_HEAD_DIM)
        .map(|i| bf16::from_f32((i % 7) as f32 * 0.003 - 0.009))
        .collect();
    let mut q_gate = vec![0u16; BATCH * NUM_ATTN_HEADS * ATTN_HEAD_DIM * 2];
    for b in 0..BATCH {
        for head in 0..NUM_ATTN_HEADS {
            let base = (b * NUM_ATTN_HEADS + head) * ATTN_HEAD_DIM * 2;
            for dimension in 0..ATTN_HEAD_DIM {
                let q = ((b * 17 + head * 7 + dimension * 3) % 41) as f32 / 20.0 - 1.0;
                q_gate[base + dimension] = bf16::from_f32(q);
                // Deliberately unrelated: catches a global-half instead of
                // per-head [query, gate] split.
                q_gate[base + ATTN_HEAD_DIM + dimension] = bf16::from_f32(20.0 + q);
            }
        }
    }
    let kv_elements = BATCH * NUM_KV_HEADS * ATTN_HEAD_DIM;
    let key: Vec<u16> = (0..kv_elements)
        .map(|i| bf16::from_f32((i % 37) as f32 / 18.0 - 1.0))
        .collect();
    let value: Vec<u16> = (0..kv_elements)
        .map(|i| bf16::from_f32((i % 23) as f32 / 16.0 - 0.7))
        .collect();
    let mut cosine = vec![0u16; BATCH * ROPE_DIM];
    let mut sine = vec![0u16; BATCH * ROPE_DIM];
    for b in 0..BATCH {
        for dimension in 0..ROPE_DIM {
            let angle = (b + 3) as f32 * (dimension % (ROPE_DIM / 2) + 1) as f32 * 0.003;
            cosine[b * ROPE_DIM + dimension] = bf16::from_f32(angle.cos());
            sine[b * ROPE_DIM + dimension] = bf16::from_f32(angle.sin());
        }
    }
    let physical_blocks = vec![3u32, 1];
    let block_offsets = vec![7u32, 63];
    let cache_elements = BLOCKS * NUM_KV_HEADS * PAGE_SIZE * ATTN_HEAD_DIM;

    let preprocessor = AttentionPreprocessor::from_host(&q_weight, &k_weight, EPSILON).unwrap();
    let stream = Stream::new().unwrap();
    let device_q_gate = DeviceBuffer::from_slice(&q_gate).unwrap();
    let device_key = DeviceBuffer::from_slice(&key).unwrap();
    let device_value = DeviceBuffer::from_slice(&value).unwrap();
    let device_cosine = DeviceBuffer::from_slice(&cosine).unwrap();
    let device_sine = DeviceBuffer::from_slice(&sine).unwrap();
    let device_blocks = DeviceBuffer::from_slice(&physical_blocks).unwrap();
    let device_offsets = DeviceBuffer::from_slice(&block_offsets).unwrap();
    let mut query = DeviceBuffer::<u16>::zeroed(BATCH * NUM_ATTN_HEADS * ATTN_HEAD_DIM).unwrap();
    let mut key_cache = DeviceBuffer::<u8>::zeroed(cache_elements).unwrap();
    let mut value_cache = DeviceBuffer::<u8>::zeroed(cache_elements).unwrap();
    preprocessor
        .prepare_decode_fp8(
            &device_q_gate,
            &device_key,
            &device_value,
            &device_cosine,
            &device_sine,
            &device_blocks,
            &device_offsets,
            &mut query,
            &mut key_cache,
            &mut value_cache,
            BLOCKS,
            BATCH,
            &stream,
        )
        .unwrap();
    stream.synchronize().unwrap();

    let actual_query = query.to_vec().unwrap();
    let actual_key = key_cache.to_vec().unwrap();
    let actual_value = value_cache.to_vec().unwrap();
    for b in 0..BATCH {
        let row_cosine = &cosine[b * ROPE_DIM..][..ROPE_DIM];
        let row_sine = &sine[b * ROPE_DIM..][..ROPE_DIM];
        for head in 0..NUM_ATTN_HEADS {
            let projection_base = (b * NUM_ATTN_HEADS + head) * ATTN_HEAD_DIM * 2;
            let projection = &q_gate[projection_base..][..ATTN_HEAD_DIM];
            let norm = normalized(projection, &q_weight, EPSILON);
            let output_base = (b * NUM_ATTN_HEADS + head) * ATTN_HEAD_DIM;
            for dimension in 0..ATTN_HEAD_DIM {
                let expected = rotate(&norm, row_cosine, row_sine, dimension);
                let actual = bf16::to_f32(actual_query[output_base + dimension]);
                assert!(
                    (actual - expected).abs() <= 0.008,
                    "query b={b}, h={head}, d={dimension}: GPU={actual}, CPU={expected}"
                );
            }
        }

        for head in 0..NUM_KV_HEADS {
            let projection_base = (b * NUM_KV_HEADS + head) * ATTN_HEAD_DIM;
            let norm = normalized(&key[projection_base..][..ATTN_HEAD_DIM], &k_weight, EPSILON);
            let cache_base = (((physical_blocks[b] as usize * NUM_KV_HEADS + head) * PAGE_SIZE
                + block_offsets[b] as usize)
                * ATTN_HEAD_DIM) as usize;
            for dimension in 0..ATTN_HEAD_DIM {
                let expected_key = rotate(&norm, row_cosine, row_sine, dimension);
                let decoded_key = nvfp4::reference::e4m3(actual_key[cache_base + dimension]);
                assert!(
                    (decoded_key - expected_key).abs() <= 0.015 + expected_key.abs() * 0.07,
                    "key b={b}, h={head}, d={dimension}: FP8={decoded_key}, CPU={expected_key}"
                );
                let expected_value = bf16::to_f32(value[projection_base + dimension]);
                let decoded_value = nvfp4::reference::e4m3(actual_value[cache_base + dimension]);
                assert!(
                    (decoded_value - expected_value).abs() <= 0.015 + expected_value.abs() * 0.07,
                    "value b={b}, h={head}, d={dimension}: FP8={decoded_value}, CPU={expected_value}"
                );
            }
        }
    }

    // A page not selected by either sequence remains untouched.
    let unused_base = 2 * NUM_KV_HEADS * PAGE_SIZE * ATTN_HEAD_DIM;
    assert!(
        actual_key[unused_base..unused_base + 1024]
            .iter()
            .all(|&value| value == 0)
    );

    // The BF16 diagnostic mode writes the same selected cache cells without
    // E4M3 quantization. Buffers stay byte-typed so the executor can choose
    // their element width at construction time.
    let mut bf16_key_cache = DeviceBuffer::<u8>::zeroed(cache_elements * 2).unwrap();
    let mut bf16_value_cache = DeviceBuffer::<u8>::zeroed(cache_elements * 2).unwrap();
    preprocessor
        .prepare_decode(
            &device_q_gate,
            &device_key,
            &device_value,
            &device_cosine,
            &device_sine,
            &device_blocks,
            &device_offsets,
            &mut query,
            &mut bf16_key_cache,
            &mut bf16_value_cache,
            BLOCKS,
            BATCH,
            KvCacheDtype::Bf16,
            &stream,
        )
        .unwrap();
    stream.synchronize().unwrap();
    let actual_key = bf16_key_cache.to_vec().unwrap();
    let actual_value = bf16_value_cache.to_vec().unwrap();
    for b in 0..BATCH {
        let row_cosine = &cosine[b * ROPE_DIM..][..ROPE_DIM];
        let row_sine = &sine[b * ROPE_DIM..][..ROPE_DIM];
        for head in 0..NUM_KV_HEADS {
            let projection_base = (b * NUM_KV_HEADS + head) * ATTN_HEAD_DIM;
            let norm = normalized(&key[projection_base..][..ATTN_HEAD_DIM], &k_weight, EPSILON);
            let cache_base = ((physical_blocks[b] as usize * NUM_KV_HEADS + head) * PAGE_SIZE
                + block_offsets[b] as usize)
                * ATTN_HEAD_DIM;
            for dimension in 0..ATTN_HEAD_DIM {
                let offset = (cache_base + dimension) * 2;
                let actual_key = bf16::to_f32(u16::from_ne_bytes([
                    actual_key[offset],
                    actual_key[offset + 1],
                ]));
                let actual_value = bf16::to_f32(u16::from_ne_bytes([
                    actual_value[offset],
                    actual_value[offset + 1],
                ]));
                let expected_key = rotate(&norm, row_cosine, row_sine, dimension);
                let expected_value = bf16::to_f32(value[projection_base + dimension]);
                assert!((actual_key - expected_key).abs() <= 0.008);
                assert_eq!(actual_value, expected_value);
            }
        }
    }
}

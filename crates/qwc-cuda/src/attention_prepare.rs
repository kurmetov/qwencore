//! Q/K normalization, partial RoPE and FP8 paged-cache write for decode.

use crate::error::{Result, check};
use crate::paged_attention::KvCacheDtype;
use crate::{DeviceBuffer, Stream, ffi};
use qwc_core::arch::{ATTN_HEAD_DIM, NUM_ATTN_HEADS, NUM_KV_HEADS, ROPE_DIM};

pub struct AttentionPreprocessor {
    query_norm_weight: DeviceBuffer<u16>,
    key_norm_weight: DeviceBuffer<u16>,
    epsilon: f32,
}

impl AttentionPreprocessor {
    /// Both checkpoint weights are zero-centered: the actual multiplier is
    /// `1 + weight`, as required by Qwen3.5 RMSNorm.
    pub fn from_host(
        query_norm_weight: &[u16],
        key_norm_weight: &[u16],
        epsilon: f32,
    ) -> Result<Self> {
        assert_eq!(query_norm_weight.len(), ATTN_HEAD_DIM);
        assert_eq!(key_norm_weight.len(), ATTN_HEAD_DIM);
        assert!(epsilon.is_finite() && epsilon > 0.0);
        Ok(Self {
            query_norm_weight: DeviceBuffer::from_slice(query_norm_weight)?,
            key_norm_weight: DeviceBuffer::from_slice(key_norm_weight)?,
            epsilon,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prepare_decode_fp8(
        &self,
        query_gate_projection: &DeviceBuffer<u16>,
        key_projection: &DeviceBuffer<u16>,
        value_projection: &DeviceBuffer<u16>,
        cosine: &DeviceBuffer<u16>,
        sine: &DeviceBuffer<u16>,
        physical_blocks: &DeviceBuffer<u32>,
        block_offsets: &DeviceBuffer<u32>,
        query: &mut DeviceBuffer<u16>,
        key_cache: &mut DeviceBuffer<u8>,
        value_cache: &mut DeviceBuffer<u8>,
        num_blocks: usize,
        batch: usize,
        stream: &Stream,
    ) -> Result<()> {
        self.prepare_decode(
            query_gate_projection,
            key_projection,
            value_projection,
            cosine,
            sine,
            physical_blocks,
            block_offsets,
            query,
            key_cache,
            value_cache,
            num_blocks,
            batch,
            KvCacheDtype::Fp8,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prepare_decode(
        &self,
        query_gate_projection: &DeviceBuffer<u16>,
        key_projection: &DeviceBuffer<u16>,
        value_projection: &DeviceBuffer<u16>,
        cosine: &DeviceBuffer<u16>,
        sine: &DeviceBuffer<u16>,
        physical_blocks: &DeviceBuffer<u32>,
        block_offsets: &DeviceBuffer<u32>,
        query: &mut DeviceBuffer<u16>,
        key_cache: &mut DeviceBuffer<u8>,
        value_cache: &mut DeviceBuffer<u8>,
        num_blocks: usize,
        batch: usize,
        cache_dtype: KvCacheDtype,
        stream: &Stream,
    ) -> Result<()> {
        assert!((1..=crate::MAX_STEP_ROWS).contains(&batch));
        assert!(query_gate_projection.len() >= batch * NUM_ATTN_HEADS * ATTN_HEAD_DIM * 2);
        assert!(key_projection.len() >= batch * NUM_KV_HEADS * ATTN_HEAD_DIM);
        assert!(value_projection.len() >= batch * NUM_KV_HEADS * ATTN_HEAD_DIM);
        assert!(cosine.len() >= batch * ROPE_DIM);
        assert!(sine.len() >= batch * ROPE_DIM);
        assert!(physical_blocks.len() >= batch);
        assert!(block_offsets.len() >= batch);
        assert!(query.len() >= batch * NUM_ATTN_HEADS * ATTN_HEAD_DIM);
        let cache_elements =
            num_blocks * NUM_KV_HEADS * super::paged_attention::PAGE_SIZE * ATTN_HEAD_DIM;
        let cache_bytes = cache_elements * cache_dtype.bytes_per_element();
        assert_eq!(key_cache.len(), cache_bytes);
        assert_eq!(value_cache.len(), cache_bytes);

        let launch = match cache_dtype {
            KvCacheDtype::Fp8 => ffi::qwc_prepare_attention_decode_fp8,
            KvCacheDtype::Bf16 => ffi::qwc_prepare_attention_decode_bf16,
        };
        check(unsafe {
            launch(
                query_gate_projection.as_ptr(),
                key_projection.as_ptr(),
                value_projection.as_ptr(),
                self.query_norm_weight.as_ptr(),
                self.key_norm_weight.as_ptr(),
                cosine.as_ptr(),
                sine.as_ptr(),
                physical_blocks.as_ptr(),
                block_offsets.as_ptr(),
                query.as_mut_ptr(),
                key_cache.as_mut_ptr(),
                value_cache.as_mut_ptr(),
                batch as i32,
                self.epsilon,
                stream.raw(),
            )
        })
    }
}

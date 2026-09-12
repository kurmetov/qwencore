// Fused decode preprocessing for a Qwen3.5 full-attention layer:
//   - split per-head [query, gate] projection layout;
//   - zero-centered Q/K RMSNorm;
//   - partial RoPE over the first 64 of 256 dimensions;
//   - write the current K/V token directly into paged FP8 cache.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int kQueryHeads = 24;
constexpr int kKvHeads = 4;
constexpr int kHeadDim = 256;
constexpr int kQGateHeadDim = 2 * kHeadDim;
constexpr int kRopeDim = 64;
constexpr int kPageSize = 64;
constexpr int kThreads = 256;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ float block_inverse_rms(float value, float epsilon) {
  __shared__ float warp_sums[kThreads / 32];
  __shared__ float inverse;
  value = warp_sum(value);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads();
  if (warp == 0) {
    float sum = lane < kThreads / 32 ? warp_sums[lane] : 0.0f;
    sum = warp_sum(sum);
    if (lane == 0) {
      inverse = rsqrtf(sum / static_cast<float>(kHeadDim) + epsilon);
    }
  }
  __syncthreads();
  return inverse;
}

__device__ __forceinline__ float rotate(
    const __nv_bfloat16* normalized,
    const __nv_bfloat16* cosine,
    const __nv_bfloat16* sine,
    int dimension) {
  float value = __bfloat162float(normalized[dimension]);
  if (dimension >= kRopeDim) {
    return value;
  }
  const int half = kRopeDim / 2;
  const int paired = dimension < half ? dimension + half : dimension - half;
  const float rotated = __bfloat162float(normalized[paired]) *
                        (dimension < half ? -1.0f : 1.0f);
  return value * __bfloat162float(cosine[dimension]) +
         rotated * __bfloat162float(sine[dimension]);
}

__device__ __forceinline__ uint8_t to_fp8(float value) {
  __nv_fp8_e4m3 converted(value);
  return reinterpret_cast<uint8_t&>(converted);
}

__global__ __launch_bounds__(kThreads) void prepare_query_kernel(
    const __nv_bfloat16* __restrict__ query_gate_projection,
    const __nv_bfloat16* __restrict__ norm_weight,
    const __nv_bfloat16* __restrict__ cosine,
    const __nv_bfloat16* __restrict__ sine,
    __nv_bfloat16* __restrict__ query,
    float epsilon) {
  const int head = blockIdx.x;
  const int batch = blockIdx.y;
  const int dimension = threadIdx.x;
  const size_t projection_base =
      (static_cast<size_t>(batch) * kQueryHeads + head) * kQGateHeadDim;
  const float raw = __bfloat162float(query_gate_projection[projection_base + dimension]);
  const float inverse = block_inverse_rms(raw * raw, epsilon);

  __shared__ __nv_bfloat16 normalized[kHeadDim];
  normalized[dimension] = __float2bfloat16(
      raw * inverse * (1.0f + __bfloat162float(norm_weight[dimension])));
  __syncthreads();
  const __nv_bfloat16* row_cosine = cosine + static_cast<size_t>(batch) * kRopeDim;
  const __nv_bfloat16* row_sine = sine + static_cast<size_t>(batch) * kRopeDim;
  const size_t output_base =
      (static_cast<size_t>(batch) * kQueryHeads + head) * kHeadDim;
  query[output_base + dimension] =
      __float2bfloat16(rotate(normalized, row_cosine, row_sine, dimension));
}

template <bool Bf16Cache>
__global__ __launch_bounds__(kThreads) void prepare_key_value_kernel(
    const __nv_bfloat16* __restrict__ key_projection,
    const __nv_bfloat16* __restrict__ value_projection,
    const __nv_bfloat16* __restrict__ norm_weight,
    const __nv_bfloat16* __restrict__ cosine,
    const __nv_bfloat16* __restrict__ sine,
    const uint32_t* __restrict__ physical_blocks,
    const uint32_t* __restrict__ block_offsets,
    void* __restrict__ key_cache,
    void* __restrict__ value_cache,
    float epsilon) {
  const int head = blockIdx.x;
  const int batch = blockIdx.y;
  const int dimension = threadIdx.x;
  const size_t projection_base =
      (static_cast<size_t>(batch) * kKvHeads + head) * kHeadDim;
  const float raw = __bfloat162float(key_projection[projection_base + dimension]);
  const float inverse = block_inverse_rms(raw * raw, epsilon);

  __shared__ __nv_bfloat16 normalized[kHeadDim];
  normalized[dimension] = __float2bfloat16(
      raw * inverse * (1.0f + __bfloat162float(norm_weight[dimension])));
  __syncthreads();
  const __nv_bfloat16* row_cosine = cosine + static_cast<size_t>(batch) * kRopeDim;
  const __nv_bfloat16* row_sine = sine + static_cast<size_t>(batch) * kRopeDim;
  const size_t cache_base =
      (((static_cast<size_t>(physical_blocks[batch]) * kKvHeads + head) *
             kPageSize +
         block_offsets[batch]) *
        kHeadDim);
  const float key = rotate(normalized, row_cosine, row_sine, dimension);
  const float value = __bfloat162float(value_projection[projection_base + dimension]);
  if constexpr (Bf16Cache) {
    static_cast<__nv_bfloat16*>(key_cache)[cache_base + dimension] =
        __float2bfloat16(key);
    static_cast<__nv_bfloat16*>(value_cache)[cache_base + dimension] =
        __float2bfloat16(value);
  } else {
    static_cast<uint8_t*>(key_cache)[cache_base + dimension] = to_fp8(key);
    static_cast<uint8_t*>(value_cache)[cache_base + dimension] = to_fp8(value);
  }
}

template <bool Bf16Cache>
cudaError_t prepare_attention(
    const void* query_gate_projection,
    const void* key_projection,
    const void* value_projection,
    const void* query_norm_weight,
    const void* key_norm_weight,
    const void* cosine,
    const void* sine,
    const void* physical_blocks,
    const void* block_offsets,
    void* query,
    void* key_cache,
    void* value_cache,
    int batch,
    float epsilon,
    cudaStream_t stream) {
  if (query_gate_projection == nullptr || key_projection == nullptr ||
      value_projection == nullptr || query_norm_weight == nullptr ||
      key_norm_weight == nullptr || cosine == nullptr || sine == nullptr ||
      physical_blocks == nullptr || block_offsets == nullptr || query == nullptr ||
      key_cache == nullptr || value_cache == nullptr || batch <= 0 || batch > 1024 ||
      !isfinite(epsilon) || epsilon <= 0.0f) {
    return cudaErrorInvalidValue;
  }
  dim3 query_grid(kQueryHeads, batch);
  prepare_query_kernel<<<query_grid, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(query_gate_projection),
      static_cast<const __nv_bfloat16*>(query_norm_weight),
      static_cast<const __nv_bfloat16*>(cosine),
      static_cast<const __nv_bfloat16*>(sine),
      static_cast<__nv_bfloat16*>(query),
      epsilon);
  cudaError_t launched = cudaGetLastError();
  if (launched != cudaSuccess) {
    return launched;
  }
  dim3 key_grid(kKvHeads, batch);
  prepare_key_value_kernel<Bf16Cache><<<key_grid, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(key_projection),
      static_cast<const __nv_bfloat16*>(value_projection),
      static_cast<const __nv_bfloat16*>(key_norm_weight),
      static_cast<const __nv_bfloat16*>(cosine),
      static_cast<const __nv_bfloat16*>(sine),
      static_cast<const uint32_t*>(physical_blocks),
      static_cast<const uint32_t*>(block_offsets),
      key_cache,
      value_cache,
      epsilon);
  return cudaGetLastError();
}

}  // namespace

extern "C" cudaError_t qwc_prepare_attention_decode_fp8(
    const void* query_gate_projection,
    const void* key_projection,
    const void* value_projection,
    const void* query_norm_weight,
    const void* key_norm_weight,
    const void* cosine,
    const void* sine,
    const void* physical_blocks,
    const void* block_offsets,
    void* query,
    void* key_cache,
    void* value_cache,
    int batch,
    float epsilon,
    cudaStream_t stream) {
  return prepare_attention<false>(
      query_gate_projection, key_projection, value_projection,
      query_norm_weight, key_norm_weight, cosine, sine,
      physical_blocks, block_offsets, query, key_cache, value_cache,
      batch, epsilon, stream);
}

extern "C" cudaError_t qwc_prepare_attention_decode_bf16(
    const void* query_gate_projection,
    const void* key_projection,
    const void* value_projection,
    const void* query_norm_weight,
    const void* key_norm_weight,
    const void* cosine,
    const void* sine,
    const void* physical_blocks,
    const void* block_offsets,
    void* query,
    void* key_cache,
    void* value_cache,
    int batch,
    float epsilon,
    cudaStream_t stream) {
  return prepare_attention<true>(
      query_gate_projection, key_projection, value_projection,
      query_norm_weight, key_norm_weight, cosine, sine,
      physical_blocks, block_offsets, query, key_cache, value_cache,
      batch, epsilon, stream);
}

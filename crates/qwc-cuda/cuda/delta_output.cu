// Qwen3.5 RMSNormGated after the recurrent DeltaNet update.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "limits.cuh"

namespace {

constexpr int kHeads = 48;
constexpr int kHeadDim = 128;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__global__ __launch_bounds__(kHeadDim) void gated_rmsnorm_kernel(
    const float* __restrict__ input,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    float epsilon,
    int gate_stride) {
  const int head = blockIdx.x;
  const int batch = blockIdx.y;
  const int dimension = threadIdx.x;
  const size_t offset =
      (static_cast<size_t>(batch) * kHeads + head) * kHeadDim + dimension;
  // Гейт может лежать срезом слитой арены, вход и выход — всегда плотно.
  const size_t gate_offset =
      static_cast<size_t>(batch) * gate_stride + head * kHeadDim + dimension;

  // The recurrent implementation returns to its original BF16 dtype before
  // Qwen3_5RMSNormGated converts it back to FP32 for the variance.
  const __nv_bfloat16 rounded = __float2bfloat16(input[offset]);
  const float value = __bfloat162float(rounded);
  float sum = warp_sum(value * value);
  __shared__ float warp_sums[4];
  __shared__ float inverse;
  const int lane = dimension & 31;
  const int warp = dimension >> 5;
  if (lane == 0) {
    warp_sums[warp] = sum;
  }
  __syncthreads();
  if (warp == 0) {
    sum = lane < 4 ? warp_sums[lane] : 0.0f;
    sum = warp_sum(sum);
    if (lane == 0) {
      inverse = rsqrtf(sum / static_cast<float>(kHeadDim) + epsilon);
    }
  }
  __syncthreads();

  const __nv_bfloat16 normalized = __float2bfloat16(value * inverse);
  const __nv_bfloat16 weighted = __float2bfloat16(
      __bfloat162float(weight[dimension]) * __bfloat162float(normalized));
  const float gate_value = __bfloat162float(gate[gate_offset]);
  const float silu = gate_value / (1.0f + __expf(-gate_value));
  output[offset] = __float2bfloat16(__bfloat162float(weighted) * silu);
}

}  // namespace

extern "C" cudaError_t qwc_delta_gated_rmsnorm(
    const float* input,
    const void* gate,
    const void* weight,
    void* output,
    int batch,
    float epsilon,
    int gate_stride,
    cudaStream_t stream) {
  if (input == nullptr || gate == nullptr || weight == nullptr || output == nullptr ||
      batch <= 0 || batch > qwc::kMaxStepRows || !isfinite(epsilon) || epsilon <= 0.0f ||
      gate_stride < kHeads * kHeadDim) {
    return cudaErrorInvalidValue;
  }
  dim3 grid(kHeads, batch);
  gated_rmsnorm_kernel<<<grid, kHeadDim, 0, stream>>>(
      input,
      static_cast<const __nv_bfloat16*>(gate),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<__nv_bfloat16*>(output),
      epsilon,
      gate_stride);
  return cudaGetLastError();
}

// Decode RMSNorm for Qwen hidden states.
//
// One CTA owns one sequence row. It optionally performs residual += input,
// keeps the rounded BF16 residual in shared memory, computes RMS in FP32, and
// either writes BF16 or quantizes directly into the SM120 NVFP4 input layout.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

struct Packed16 {
  uint32_t lo;
  uint32_t hi;
};

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ float reciprocal_approximate_ftz(float value) {
  float result;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
  return result;
}

__device__ __forceinline__ uint32_t encode_e2m1(float value) {
  const uint32_t bits = __float_as_uint(value);
  const uint32_t absolute_bits = bits & 0x7fffffffu;
  if (absolute_bits > 0x7f800000u) {
    return 7u;
  }
  const float x = __uint_as_float(absolute_bits);
  uint32_t magnitude;
  if (x < 0.75f) {
    magnitude = x > 0.25f ? 1u : 0u;
  } else if (x <= 1.25f) {
    magnitude = 2u;
  } else if (x < 1.75f) {
    magnitude = 3u;
  } else if (x <= 2.5f) {
    magnitude = 4u;
  } else if (x < 3.5f) {
    magnitude = 5u;
  } else if (x <= 5.0f) {
    magnitude = 6u;
  } else {
    magnitude = 7u;
  }
  return ((bits >> 31) << 3) | magnitude;
}

__device__ __forceinline__ Packed16 pack_e2m1(float (&values)[16]) {
  Packed16 output{0, 0};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    output.lo |= encode_e2m1(values[i]) << (4 * i);
    output.hi |= encode_e2m1(values[i + 8]) << (4 * i);
  }
  return output;
}

__device__ __forceinline__ size_t scale_offset(
    int row, int group, int group_blocks) {
  const int row_block = row >> 7;
  const int row_in_quad = row & 31;
  const int row_quad = (row >> 5) & 3;
  const int group_block = group >> 2;
  const int group_in_block = group & 3;
  return (((static_cast<size_t>(row_block) * group_blocks + group_block) * 32 +
           row_in_quad) *
              4 +
          row_quad) *
             4 +
         group_in_block;
}

__device__ __forceinline__ float stage_and_inverse_rms(
    const __nv_bfloat16* input,
    __nv_bfloat16* residual,
    __nv_bfloat16* staged,
    float* warp_sums,
    float* inverse_rms,
    int row,
    int hidden,
    float epsilon) {
  float sum = 0.0f;
  const size_t base = static_cast<size_t>(row) * hidden;
  for (int column = threadIdx.x; column < hidden; column += blockDim.x) {
    float value = __bfloat162float(input[base + column]);
    if (residual != nullptr) {
      value += __bfloat162float(residual[base + column]);
      const __nv_bfloat16 rounded = __float2bfloat16(value);
      residual[base + column] = rounded;
      staged[column] = rounded;
      value = __bfloat162float(rounded);
    } else {
      staged[column] = input[base + column];
    }
    sum = fmaf(value, value, sum);
  }

  sum = warp_sum(sum);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) {
    warp_sums[warp] = sum;
  }
  __syncthreads();

  if (warp == 0) {
    float block_sum = lane < blockDim.x / 32 ? warp_sums[lane] : 0.0f;
    block_sum = warp_sum(block_sum);
    if (lane == 0) {
      *inverse_rms = rsqrtf(block_sum / static_cast<float>(hidden) + epsilon);
    }
  }
  __syncthreads();
  return *inverse_rms;
}

__global__ void rmsnorm_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    int hidden,
    float epsilon) {
  extern __shared__ __align__(16) unsigned char shared_raw[];
  auto* staged = reinterpret_cast<__nv_bfloat16*>(shared_raw);
  auto* warp_sums = reinterpret_cast<float*>(staged + hidden);
  auto* inverse_rms = warp_sums + 8;
  const int row = blockIdx.x;
  const float inverse = stage_and_inverse_rms(
      input, residual, staged, warp_sums, inverse_rms, row, hidden, epsilon);
  const size_t base = static_cast<size_t>(row) * hidden;
  for (int column = threadIdx.x; column < hidden; column += blockDim.x) {
    const float normalized = __bfloat162float(staged[column]) * inverse *
                             (1.0f + __bfloat162float(weight[column]));
    output[base + column] = __float2bfloat16(normalized);
  }
}

__global__ void rmsnorm_nvfp4_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    uint8_t* __restrict__ packed,
    uint8_t* __restrict__ scales,
    int hidden,
    float epsilon,
    float global_scale) {
  extern __shared__ __align__(16) unsigned char shared_raw[];
  auto* staged = reinterpret_cast<__nv_bfloat16*>(shared_raw);
  auto* warp_sums = reinterpret_cast<float*>(staged + hidden);
  auto* inverse_rms = warp_sums + 8;
  const int row = blockIdx.x;
  const float inverse = stage_and_inverse_rms(
      input, residual, staged, warp_sums, inverse_rms, row, hidden, epsilon);

  const int groups = hidden / 16;
  const int group_blocks = groups / 4;
  for (int group = threadIdx.x; group < groups; group += blockDim.x) {
    float values[16];
    float maximum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int column = group * 16 + i;
      const float value = __bfloat162float(staged[column]) * inverse *
                          (1.0f + __bfloat162float(weight[column]));
      values[i] = value;
      maximum = fmaxf(maximum, fabsf(value));
    }

    __nv_fp8_e4m3 fp8_scale(global_scale * (maximum * (1.0f / 6.0f)));
    const uint8_t raw_scale = reinterpret_cast<uint8_t&>(fp8_scale);
    const float rounded_scale = static_cast<float>(fp8_scale);
    const float quant_multiplier =
        rounded_scale == 0.0f
            ? 0.0f
            : reciprocal_approximate_ftz(
                  rounded_scale * reciprocal_approximate_ftz(global_scale));
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      values[i] *= quant_multiplier;
    }

    const Packed16 result = pack_e2m1(values);
    const size_t packed_offset =
        static_cast<size_t>(row) * (hidden / 2) + group * 8;
    *reinterpret_cast<Packed16*>(packed + packed_offset) = result;
    scales[scale_offset(row, group, group_blocks)] = raw_scale;
  }
}

cudaError_t validate(
    const void* input,
    const void* weight,
    const void* output,
    int batch,
    int hidden,
    float epsilon) {
  if (input == nullptr || weight == nullptr || output == nullptr || batch <= 0 ||
      batch > 1024 || hidden <= 0 || hidden % 256 != 0 || !isfinite(epsilon) ||
      epsilon <= 0.0f) {
    return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

}  // namespace

extern "C" cudaError_t qwc_rmsnorm_bf16(
    const void* input,
    void* residual,
    const void* weight,
    void* output,
    int batch,
    int hidden,
    float epsilon,
    cudaStream_t stream) {
  const cudaError_t valid = validate(input, weight, output, batch, hidden, epsilon);
  if (valid != cudaSuccess) {
    return valid;
  }
  constexpr int threads = 256;
  const size_t shared_bytes = hidden * sizeof(__nv_bfloat16) + 9 * sizeof(float);
  rmsnorm_bf16_kernel<<<batch, threads, shared_bytes, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<__nv_bfloat16*>(residual),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<__nv_bfloat16*>(output),
      hidden,
      epsilon);
  return cudaGetLastError();
}

extern "C" cudaError_t qwc_rmsnorm_nvfp4(
    const void* input,
    void* residual,
    const void* weight,
    void* packed,
    void* scales,
    int batch,
    int hidden,
    float epsilon,
    float global_scale,
    cudaStream_t stream) {
  const cudaError_t valid = validate(input, weight, packed, batch, hidden, epsilon);
  if (valid != cudaSuccess || scales == nullptr || !isfinite(global_scale) ||
      global_scale <= 0.0f) {
    return cudaErrorInvalidValue;
  }
  constexpr int threads = 256;
  const size_t shared_bytes = hidden * sizeof(__nv_bfloat16) + 9 * sizeof(float);
  rmsnorm_nvfp4_kernel<<<batch, threads, shared_bytes, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<__nv_bfloat16*>(residual),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<uint8_t*>(packed),
      static_cast<uint8_t*>(scales),
      hidden,
      epsilon,
      global_scale);
  return cudaGetLastError();
}

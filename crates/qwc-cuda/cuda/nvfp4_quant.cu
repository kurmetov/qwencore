// BF16 -> NVFP4 dynamic activation quantization for decode.
// One thread owns one contiguous group of 16 values and writes the CUTLASS
// 128x4 scale layout directly. Padded scale slots are zeroed once by the
// owning Rust buffer, so the hot path only touches live rows.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

struct Packed16 {
  uint32_t lo;
  uint32_t hi;
};

__device__ __forceinline__ float reciprocal_approximate_ftz(float value) {
  float result;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
  return result;
}

// SM120 exposes block-scaled E2M1 MMA but CUDA 13.1 does not expose the
// cvt.e2m1x2 instruction for this target. This is the same RNE mapping as the
// CUTLASS software fallback. Midpoints choose the even E2M1 code.
__device__ __forceinline__ uint32_t encode_e2m1(float value) {
  const uint32_t bits = __float_as_uint(value);
  const uint32_t absolute_bits = bits & 0x7fffffffu;
  if (absolute_bits > 0x7f800000u) {
    return 7u;  // NaN -> positive max, matching saturating CUDA conversion.
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

__device__ __forceinline__ Packed16 pack_e2m1(float (&v)[16]) {
  Packed16 out{0, 0};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out.lo |= encode_e2m1(v[i]) << (4 * i);
    out.hi |= encode_e2m1(v[i + 8]) << (4 * i);
  }
  return out;
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

__global__ void quantize_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,
    uint8_t* __restrict__ packed,
    uint8_t* __restrict__ scales,
    int in_features,
    float global_scale) {
  const int row = blockIdx.y;
  const int groups = in_features / 16;
  const int group = blockIdx.x * blockDim.x + threadIdx.x;
  if (group >= groups) {
    return;
  }

  float values[16];
  float maximum = 0.0f;
  const size_t input_base = static_cast<size_t>(row) * in_features + group * 16;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const float value = __bfloat162float(input[input_base + i]);
    values[i] = value;
    maximum = fmaxf(maximum, fabsf(value));
  }

  const float unrounded_scale = global_scale * (maximum * (1.0f / 6.0f));
  __nv_fp8_e4m3 fp8_scale(unrounded_scale);
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
      static_cast<size_t>(row) * (in_features / 2) + group * 8;
  *reinterpret_cast<Packed16*>(packed + packed_offset) = result;

  const int group_blocks = (groups + 3) / 4;
  scales[scale_offset(row, group, group_blocks)] = raw_scale;
}

}  // namespace

extern "C" cudaError_t qwc_nvfp4_quantize_bf16(
    const void* input,
    void* packed,
    void* scales,
    int batch,
    int in_features,
    float global_scale,
    cudaStream_t stream) {
  if (input == nullptr || packed == nullptr || scales == nullptr || batch <= 0 ||
      batch > 1024 || in_features <= 0 || in_features % 256 != 0 ||
      !isfinite(global_scale) || global_scale <= 0.0f) {
    return cudaErrorInvalidValue;
  }
  constexpr int threads = 256;
  const int groups = in_features / 16;
  const dim3 grid((groups + threads - 1) / threads, batch);
  quantize_bf16_kernel<<<grid, threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<uint8_t*>(packed),
      static_cast<uint8_t*>(scales),
      in_features,
      global_scale);
  return cudaGetLastError();
}

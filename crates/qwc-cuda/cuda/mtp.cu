// MTP draft-голова: линейные слои в BF16.
//
// Голова лежит в чекпоинте неквантованной (`ignored: .*mtp.*`), поэтому все
// её матрицы идут мимо NVFP4-пути. Черновой шаг делает восемь таких умножений
// при M <= 8 строк — задача чисто полосовая: 849 МБ весов за проход, то есть
// 475 мкс на потолке карты. Значит и форма кернела нужна полосовая: варп
// владеет строкой весов и читает её подряд, вход лежит в shared.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int kWarpSize = 32;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / kWarpSize;
// Ширина куска входа в shared. 1024 float на строку — это 4 КБ, при восьми
// строках 32 КБ, то есть два блока на SM.
constexpr int kChunk = 1024;   // кратен вектору из восьми

constexpr int kMaxRows = 8;

__device__ __forceinline__ float warp_sum(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  return value;
}

template <int ROWS>
__global__ __launch_bounds__(kThreads) void bf16_linear_kernel(
    const __nv_bfloat16* __restrict__ weights,   // [n, k], row-major
    const __nv_bfloat16* __restrict__ input,     // [ROWS, k]
    __nv_bfloat16* __restrict__ output,          // [ROWS, n]
    int k,
    int n) {
  __shared__ float staged[ROWS][kChunk];

  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int column = blockIdx.x * kWarps + warp;
  const bool active = column < n;
  const size_t weight_base = static_cast<size_t>(column) * k;

  float accumulator[ROWS];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    accumulator[r] = 0.0f;
  }

  for (int chunk = 0; chunk < k; chunk += kChunk) {
    const int width = min(kChunk, k - chunk);
    __syncthreads();
    for (int index = threadIdx.x; index < ROWS * width; index += kThreads) {
      const int r = index / width;
      const int position = index - r * width;
      staged[r][position] =
          __bfloat162float(input[static_cast<size_t>(r) * k + chunk + position]);
    }
    __syncthreads();
    if (!active) {
      continue;
    }
    // Восемь элементов на дорожку: варп за раз забирает 512 байт вместо 64.
    // Поэлементное чтение давало около 575 ГБ/с из 1790 возможных.
    constexpr int kVector = 8;
    const int vectors = width / kVector;
    for (int index = lane; index < vectors; index += kWarpSize) {
      const int position = index * kVector;
      const float4 packed = *reinterpret_cast<const float4*>(
          weights + weight_base + chunk + position);
      const __nv_bfloat16* values = reinterpret_cast<const __nv_bfloat16*>(&packed);
#pragma unroll
      for (int element = 0; element < kVector; ++element) {
        const float weight = __bfloat162float(values[element]);
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
          accumulator[r] += weight * staged[r][position + element];
        }
      }
    }
    for (int position = vectors * kVector + lane; position < width; position += kWarpSize) {
      const float weight = __bfloat162float(weights[weight_base + chunk + position]);
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        accumulator[r] += weight * staged[r][position];
      }
    }
  }

  if (!active) {
    return;
  }
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    const float total = warp_sum(accumulator[r]);
    if (lane == 0) {
      output[static_cast<size_t>(r) * n + column] = __float2bfloat16(total);
    }
  }
}

// SwiGLU по месту: down_proj читает произведение silu(gate) * up.
__global__ void swiglu_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ out,
    int elements) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  const float g = __bfloat162float(gate[index]);
  const float u = __bfloat162float(up[index]);
  out[index] = __float2bfloat16(g / (1.0f + __expf(-g)) * u);
}

// Вход fc: две нормированные половины подряд. Норму считает rmsnorm, здесь
// только склейка — отдельный кернел дешевле лишнего прохода по памяти.
__global__ void concat_kernel(
    const __nv_bfloat16* __restrict__ left,
    const __nv_bfloat16* __restrict__ right,
    __nv_bfloat16* __restrict__ out,
    int rows,
    int width) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= rows * 2 * width) {
    return;
  }
  const int column = index % (2 * width);
  const int row = index / (2 * width);
  out[index] = column < width
      ? left[static_cast<size_t>(row) * width + column]
      : right[static_cast<size_t>(row) * width + column - width];
}

}  // namespace

extern "C" cudaError_t qwc_bf16_linear(
    const void* weights,
    const void* input,
    void* output,
    int rows,
    int k,
    int n,
    cudaStream_t stream) {
  if (weights == nullptr || input == nullptr || output == nullptr || rows <= 0 ||
      rows > kMaxRows || k <= 0 || n <= 0) {
    return cudaErrorInvalidValue;
  }
  const int blocks = (n + kWarps - 1) / kWarps;
  const __nv_bfloat16* w = static_cast<const __nv_bfloat16*>(weights);
  const __nv_bfloat16* in = static_cast<const __nv_bfloat16*>(input);
  __nv_bfloat16* out = static_cast<__nv_bfloat16*>(output);
  switch (rows) {
#define QWC_BF16_LINEAR_CASE(r)                                              \
  case r:                                                                    \
    bf16_linear_kernel<r><<<blocks, kThreads, 0, stream>>>(w, in, out, k, n); \
    break;
    QWC_BF16_LINEAR_CASE(1)
    QWC_BF16_LINEAR_CASE(2)
    QWC_BF16_LINEAR_CASE(3)
    QWC_BF16_LINEAR_CASE(4)
    QWC_BF16_LINEAR_CASE(5)
    QWC_BF16_LINEAR_CASE(6)
    QWC_BF16_LINEAR_CASE(7)
    QWC_BF16_LINEAR_CASE(8)
#undef QWC_BF16_LINEAR_CASE
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

extern "C" cudaError_t qwc_bf16_swiglu(
    const void* gate,
    const void* up,
    void* out,
    int elements,
    cudaStream_t stream) {
  if (gate == nullptr || up == nullptr || out == nullptr || elements <= 0) {
    return cudaErrorInvalidValue;
  }
  const int threads = 256;
  swiglu_kernel<<<(elements + threads - 1) / threads, threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(gate),
      static_cast<const __nv_bfloat16*>(up),
      static_cast<__nv_bfloat16*>(out),
      elements);
  return cudaGetLastError();
}

extern "C" cudaError_t qwc_bf16_concat(
    const void* left,
    const void* right,
    void* out,
    int rows,
    int width,
    cudaStream_t stream) {
  if (left == nullptr || right == nullptr || out == nullptr || rows <= 0 || width <= 0) {
    return cudaErrorInvalidValue;
  }
  const int threads = 256;
  const int elements = rows * 2 * width;
  concat_kernel<<<(elements + threads - 1) / threads, threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(left),
      static_cast<const __nv_bfloat16*>(right),
      static_cast<__nv_bfloat16*>(out),
      rows,
      width);
  return cudaGetLastError();
}

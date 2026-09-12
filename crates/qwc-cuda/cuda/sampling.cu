// Greedy sampling without copying the vocabulary logits to the host.

#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>

namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ bool better(
    float value, uint32_t index, float best, uint32_t best_index) {
  return value > best || (value == best && index < best_index);
}

__device__ void reduce_candidate(
    float& value,
    uint32_t& index,
    float* shared_values,
    uint32_t* shared_indices) {
  shared_values[threadIdx.x] = value;
  shared_indices[threadIdx.x] = index;
  __syncthreads();
  for (int offset = kThreads / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      const float other = shared_values[threadIdx.x + offset];
      const uint32_t other_index = shared_indices[threadIdx.x + offset];
      if (better(other, other_index, shared_values[threadIdx.x], shared_indices[threadIdx.x])) {
        shared_values[threadIdx.x] = other;
        shared_indices[threadIdx.x] = other_index;
      }
    }
    __syncthreads();
  }
  value = shared_values[0];
  index = shared_indices[0];
}

__global__ void argmax_parts_kernel(
    const float* __restrict__ logits,
    float* __restrict__ partial_values,
    uint32_t* __restrict__ partial_indices,
    int vocab) {
  const int batch = blockIdx.y;
  const float* row = logits + static_cast<size_t>(batch) * vocab;
  float best = -CUDART_INF_F;
  uint32_t best_index = UINT32_MAX;
  for (int column = blockIdx.x * kThreads + threadIdx.x;
       column < vocab;
       column += gridDim.x * kThreads) {
    const float value = row[column];
    if (better(value, static_cast<uint32_t>(column), best, best_index)) {
      best = value;
      best_index = static_cast<uint32_t>(column);
    }
  }
  __shared__ float shared_values[kThreads];
  __shared__ uint32_t shared_indices[kThreads];
  reduce_candidate(best, best_index, shared_values, shared_indices);
  if (threadIdx.x == 0) {
    const size_t offset = static_cast<size_t>(batch) * gridDim.x + blockIdx.x;
    partial_values[offset] = best;
    partial_indices[offset] = best_index;
  }
}

__global__ void argmax_reduce_kernel(
    const float* __restrict__ partial_values,
    const uint32_t* __restrict__ partial_indices,
    uint32_t* __restrict__ output,
    int parts) {
  const int batch = blockIdx.x;
  float best = -CUDART_INF_F;
  uint32_t best_index = UINT32_MAX;
  for (int part = threadIdx.x; part < parts; part += kThreads) {
    const size_t offset = static_cast<size_t>(batch) * parts + part;
    const float value = partial_values[offset];
    const uint32_t index = partial_indices[offset];
    if (better(value, index, best, best_index)) {
      best = value;
      best_index = index;
    }
  }
  __shared__ float shared_values[kThreads];
  __shared__ uint32_t shared_indices[kThreads];
  reduce_candidate(best, best_index, shared_values, shared_indices);
  if (threadIdx.x == 0) {
    output[batch] = best_index;
  }
}

}  // namespace

extern "C" cudaError_t qwc_argmax(
    const float* logits,
    float* partial_values,
    uint32_t* partial_indices,
    uint32_t* output,
    int vocab,
    int batch,
    int parts,
    cudaStream_t stream) {
  if (logits == nullptr || partial_values == nullptr || partial_indices == nullptr ||
      output == nullptr || vocab <= 0 || batch <= 0 || batch > 128 ||
      parts <= 0 || parts > kThreads) {
    return cudaErrorInvalidValue;
  }
  dim3 grid(parts, batch);
  argmax_parts_kernel<<<grid, kThreads, 0, stream>>>(
      logits, partial_values, partial_indices, vocab);
  cudaError_t launched = cudaGetLastError();
  if (launched != cudaSuccess) {
    return launched;
  }
  argmax_reduce_kernel<<<batch, kThreads, 0, stream>>>(
      partial_values, partial_indices, output, parts);
  return cudaGetLastError();
}

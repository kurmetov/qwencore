// Small activation kernels shared by tensor-core projection paths.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

__global__ void swiglu_bf16_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ output,
    int elements) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  const float g = __bfloat162float(gate[index]);
  const float u = __bfloat162float(up[index]);
  output[index] = __float2bfloat16((g / (1.0f + __expf(-g))) * u);
}

}  // namespace

extern "C" cudaError_t qwc_swiglu_bf16(
    const void* gate,
    const void* up,
    void* output,
    int elements,
    cudaStream_t stream) {
  if (gate == nullptr || up == nullptr || output == nullptr || elements <= 0) {
    return cudaErrorInvalidValue;
  }
  constexpr int threads = 256;
  const int blocks = (elements + threads - 1) / threads;
  swiglu_bf16_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(gate),
      static_cast<const __nv_bfloat16*>(up),
      static_cast<__nv_bfloat16*>(output),
      elements);
  return cudaGetLastError();
}

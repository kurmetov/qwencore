// Compacts selected rows of a BF16 [rows, cols] arena into a dense buffer.
//
// A fused multi-sequence step needs the logits of the last row of every
// sequence. Calling the vocabulary projection once per row would re-read the
// whole 1.27 GB lm_head each time, so the rows are gathered first and the
// projection runs once over the compact result.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "limits.cuh"

namespace {

__global__ void gather_rows_kernel(
    const __nv_bfloat16* __restrict__ source,
    const uint32_t* __restrict__ row_indices,
    __nv_bfloat16* __restrict__ destination,
    int cols) {
  const int row = blockIdx.y;
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  if (column >= cols) {
    return;
  }
  const size_t from = (size_t)row_indices[row] * cols + column;
  destination[(size_t)row * cols + column] = source[from];
}

}  // namespace

extern "C" cudaError_t qwc_gather_rows_bf16(
    const void* source,
    const void* row_indices,
    void* destination,
    int rows,
    int cols,
    cudaStream_t stream) {
  if (source == nullptr || row_indices == nullptr || destination == nullptr ||
      rows <= 0 || rows > qwc::kMaxStepRows || cols <= 0 || cols % 8 != 0) {
    return cudaErrorInvalidValue;
  }
  constexpr int threads = 256;
  const dim3 grid((cols + threads - 1) / threads, rows);
  gather_rows_kernel<<<grid, threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(source),
      static_cast<const uint32_t*>(row_indices),
      static_cast<__nv_bfloat16*>(destination),
      cols);
  return cudaGetLastError();
}

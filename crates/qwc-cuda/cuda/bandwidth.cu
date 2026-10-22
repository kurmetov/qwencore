// Measures actually achievable memory bandwidth.
//
// The main conclusion of the roofline analysis: the whole target concurrency range (1..32)
// is memory-bound, so the right progress metric is
// the share of achieved bandwidth, not tensor core utilization.
// For that share to mean anything it needs an honest denominator: not the spec
// 1792 GB/s, but what this particular card delivers on pure reads.

#include <cuda_runtime.h>

namespace {

// Reads dominate decode: each step reads all model weights.
// The reduction is only there so the compiler doesn't drop the loads.
__global__ void bw_read_kernel(const float4* __restrict__ src, size_t n4, float* __restrict__ out) {
    float acc = 0.0f;
    size_t stride = size_t(gridDim.x) * blockDim.x;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n4; i += stride) {
        float4 v = src[i];
        acc += v.x + v.y + v.z + v.w;
    }
    // Intra-warp reduction, then one atomic access per warp.
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, off);
    }
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(out, acc);
    }
}

__global__ void bw_copy_kernel(const float4* __restrict__ src, float4* __restrict__ dst, size_t n4) {
    size_t stride = size_t(gridDim.x) * blockDim.x;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n4; i += stride) {
        dst[i] = src[i];
    }
}

// 170 SMs; take extra so the scheduler always has work.
constexpr int kBlock = 256;
constexpr int kBlocksPerSm = 8;

int grid_for(size_t n4) {
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    long long want = (long long)sms * kBlocksPerSm;
    long long need = (long long)((n4 + kBlock - 1) / kBlock);
    return (int)(want < need ? want : (need > 0 ? need : 1));
}

} // namespace

extern "C" {

// Traffic: bytes (read only).
cudaError_t qwc_bw_read(const void* src, size_t bytes, float* out, cudaStream_t stream) {
    size_t n4 = bytes / sizeof(float4);
    bw_read_kernel<<<grid_for(n4), kBlock, 0, stream>>>(
        static_cast<const float4*>(src), n4, out);
    return cudaGetLastError();
}

// Traffic: 2 * bytes (read and write).
cudaError_t qwc_bw_copy(const void* src, void* dst, size_t bytes, cudaStream_t stream) {
    size_t n4 = bytes / sizeof(float4);
    bw_copy_kernel<<<grid_for(n4), kBlock, 0, stream>>>(
        static_cast<const float4*>(src), static_cast<float4*>(dst), n4);
    return cudaGetLastError();
}

} // extern "C"

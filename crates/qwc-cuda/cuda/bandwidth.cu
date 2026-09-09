// Замер реально достижимой пропускной способности памяти.
//
// Главный вывод roofline-анализа: весь целевой диапазон concurrency (1..32)
// находится в memory-bound режиме, поэтому правильная метрика прогресса —
// доля достигнутой пропускной способности, а не загрузка тензорных ядер.
// Чтобы эта доля что-то значила, нужен честный знаменатель: не паспортные
// 1792 GB/s, а то, что выдаёт эта конкретная карта на чистом чтении.

#include <cuda_runtime.h>

namespace {

// Чтение доминирует в decode: за шаг вычитываются все веса модели.
// Редукция нужна только чтобы компилятор не выбросил загрузки.
__global__ void bw_read_kernel(const float4* __restrict__ src, size_t n4, float* __restrict__ out) {
    float acc = 0.0f;
    size_t stride = size_t(gridDim.x) * blockDim.x;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n4; i += stride) {
        float4 v = src[i];
        acc += v.x + v.y + v.z + v.w;
    }
    // Внутриварповая редукция, затем один атомарный доступ на варп.
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

// 170 SM; берём с запасом, чтобы планировщик всегда имел работу.
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

// Трафик: bytes (только чтение).
cudaError_t qwc_bw_read(const void* src, size_t bytes, float* out, cudaStream_t stream) {
    size_t n4 = bytes / sizeof(float4);
    bw_read_kernel<<<grid_for(n4), kBlock, 0, stream>>>(
        static_cast<const float4*>(src), n4, out);
    return cudaGetLastError();
}

// Трафик: 2 * bytes (чтение и запись).
cudaError_t qwc_bw_copy(const void* src, void* dst, size_t bytes, cudaStream_t stream) {
    size_t n4 = bytes / sizeof(float4);
    bw_copy_kernel<<<grid_for(n4), kBlock, 0, stream>>>(
        static_cast<const float4*>(src), static_cast<float4*>(dst), n4);
    return cudaGetLastError();
}

} // extern "C"

// Gated DeltaNet: рекуррентный шаг decode.
//
// 48 из 64 слоёв Qwen3.8 — линейное внимание с рекуррентным состоянием
// S размера [48 голов, 128, 128]. На каждый токен состояние читается и
// переписывается целиком: при batch=32 это 21% всего трафика decode-шага,
// вдвое больше KV-кэша.
//
// Рекуррентность:
//     S_t = a * S (I - b k k^T) + b v k^T = a*S + b(v - a*(S k)) k^T
//     o_t = S_t q = a*(S q) + b(v - a*(S k)) * (k . q)
//
// Отсюда главное свойство реализации: S k и S q считаются за ОДИН проход,
// после чего строка состояния остаётся в регистрах и переписывается на месте.
// Одно чтение и одна запись вместо трёх проходов по памяти.
//
// Раскладка блока: варп владеет строкой состояния. 32 потока x 4 элемента =
// ровно 128, доступ вектором по 8 байт даёт полностью слитые 256-байтные
// транзакции на варп.
//
// Строки состояния дополнительно разбиваются между блоками по gridDim.z.
// Без этого при batch=1 запускается всего 48 блоков (по одному на v-голову)
// на 170 SM, и три четверти чипа простаивает: замер давал 24% пропускной
// способности вместо 95%. Разбиение независимо, потому что строки состояния
// обновляются независимо друг от друга — связывает их только скаляр k.q.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kDk = 128;        // linear_key_head_dim
constexpr int kDv = 128;        // linear_value_head_dim
constexpr int kHv = 48;         // linear_num_value_heads
constexpr int kHk = 16;         // linear_num_key_heads
constexpr int kRatio = kHv / kHk;  // k/q броадкастятся на v-головы 1:3
constexpr int kPerThread = kDk / 32;
constexpr int kBlock = 256;     // 8 варпов -> 8 строк состояния за итерацию

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, off);
    }
    return v;
}

__global__ __launch_bounds__(kBlock) void delta_decode_kernel(
    __nv_bfloat16* __restrict__ state,   // [B, kHv, kDv, kDk]
    const float* __restrict__ q,         // [B, kHk, kDk], L2-нормирован
    const float* __restrict__ k,         // [B, kHk, kDk], L2-нормирован
    const float* __restrict__ v,         // [B, kHv, kDv]
    const float* __restrict__ alpha,     // [B, kHv]
    const float* __restrict__ beta,      // [B, kHv]
    float* __restrict__ out)             // [B, kHv, kDv]
{
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    const int hk = h / kRatio;

    // Диапазон строк, за который отвечает этот блок.
    const int rows_per_block = kDv / gridDim.z;
    const int row_begin = blockIdx.z * rows_per_block;
    const int row_end = row_begin + rows_per_block;

    __shared__ float sk[kDk];
    __shared__ float sq[kDk];
    __shared__ float s_kq;

    const float* kp = k + (size_t)(b * kHk + hk) * kDk;
    const float* qp = q + (size_t)(b * kHk + hk) * kDk;
    for (int i = threadIdx.x; i < kDk; i += kBlock) {
        sk[i] = kp[i];
        sq[i] = qp[i];
    }
    __syncthreads();

    // Скаляр k . q нужен всем строкам, считаем один раз на блок.
    if (threadIdx.x < 32) {
        float acc = 0.0f;
        for (int i = threadIdx.x; i < kDk; i += 32) {
            acc += sk[i] * sq[i];
        }
        acc = warp_sum(acc);
        if (threadIdx.x == 0) {
            s_kq = acc;
        }
    }
    __syncthreads();

    const float a = alpha[b * kHv + h];
    const float bt = beta[b * kHv + h];
    const float kq = s_kq;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    constexpr int kWarps = kBlock / 32;

    __nv_bfloat16* S = state + (size_t)(b * kHv + h) * kDv * kDk;
    const float* vp = v + (size_t)(b * kHv + h) * kDv;
    float* op = out + (size_t)(b * kHv + h) * kDv;

    for (int row = row_begin + warp; row < row_end; row += kWarps) {
        float2* srow = reinterpret_cast<float2*>(S + (size_t)row * kDk);

        // Одно 8-байтовое чтение на поток: 4 bf16, слитые 256 байт на варп.
        float2 packed = srow[lane];
        __nv_bfloat162 p0 = *reinterpret_cast<__nv_bfloat162*>(&packed.x);
        __nv_bfloat162 p1 = *reinterpret_cast<__nv_bfloat162*>(&packed.y);

        float s[kPerThread];
        s[0] = __bfloat162float(p0.x);
        s[1] = __bfloat162float(p0.y);
        s[2] = __bfloat162float(p1.x);
        s[3] = __bfloat162float(p1.y);

        const int base = lane * kPerThread;

        // Оба матвека за один проход, пока строка в регистрах.
        float u = 0.0f, w = 0.0f;
        #pragma unroll
        for (int t = 0; t < kPerThread; ++t) {
            u += s[t] * sk[base + t];
            w += s[t] * sq[base + t];
        }
        u = warp_sum(u);
        w = warp_sum(w);
        u = __shfl_sync(0xffffffff, u, 0);
        w = __shfl_sync(0xffffffff, w, 0);

        const float c = bt * (vp[row] - a * u);

        if (lane == 0) {
            op[row] = a * w + c * kq;
        }

        // Обновление на месте: строка уже в регистрах, повторное чтение не нужно.
        #pragma unroll
        for (int t = 0; t < kPerThread; ++t) {
            s[t] = a * s[t] + c * sk[base + t];
        }
        p0 = __nv_bfloat162(__float2bfloat16(s[0]), __float2bfloat16(s[1]));
        p1 = __nv_bfloat162(__float2bfloat16(s[2]), __float2bfloat16(s[3]));
        packed.x = *reinterpret_cast<float*>(&p0);
        packed.y = *reinterpret_cast<float*>(&p1);
        srow[lane] = packed;
    }
}

} // namespace

// Число блоков на голову подбирается так, чтобы занять все SM.
// Ограничения: делитель kDv, и на блок должно приходиться не меньше строк,
// чем в нём варпов, иначе часть варпов простаивает.
static int splits_for(int batch) {
    constexpr int kWarps = kBlock / 32;
    constexpr int kMaxSplit = kDv / kWarps;   // 128 / 8 = 16

    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    const int base = kHv * batch;
    if (base >= sms * 2) {
        return 1;   // блоков и так достаточно
    }
    int want = (sms * 2 + base - 1) / base;
    int split = 1;
    while (split * 2 <= kMaxSplit && split < want) {
        split *= 2;   // держим делителем kDv
    }
    return split;
}

extern "C" cudaError_t qwc_delta_decode(
    void* state, const float* q, const float* k, const float* v,
    const float* alpha, const float* beta, float* out,
    int batch, cudaStream_t stream)
{
    dim3 grid(kHv, batch, splits_for(batch));
    delta_decode_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<__nv_bfloat16*>(state), q, k, v, alpha, beta, out);
    return cudaGetLastError();
}

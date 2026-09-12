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
#include <stdint.h>

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

// Две редукции за один проход по бабочке: шагов столько же, сколько у одной,
// а результат остаётся во всех дорожках — броадкаст с нулевой дорожки не нужен.
//
// Третьей здесь была k.q. Она одна на голову и не зависит от строки
// состояния, поэтому её считает prepare, а не 128 строк каждой из трёх
// v-голов по разу на токен.
__device__ __forceinline__ void warp_sum2(float& a, float& b) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        a += __shfl_xor_sync(0xffffffff, a, off);
        b += __shfl_xor_sync(0xffffffff, b, off);
    }
}

__global__ __launch_bounds__(kBlock) void delta_decode_kernel(
    __nv_bfloat16* __restrict__ state,   // [B, kHv, kDv, kDk]
    const uint32_t* __restrict__ state_slots,
    const float* __restrict__ q,         // [B, kHk, kDk], L2-нормирован
    const float* __restrict__ k,         // [B, kHk, kDk], L2-нормирован
    const float* __restrict__ v,         // [B, kHv, kDv]
    const float* __restrict__ alpha,     // [B, kHv]
    const float* __restrict__ beta,      // [B, kHv]
    const float* __restrict__ kq,        // [B, kHk], скаляр k.q из prepare
    float* __restrict__ out)             // [B, kHv, kDv]
{
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    const int state_slot = state_slots == nullptr ? b : state_slots[b];
    const int hk = h / kRatio;

    // Диапазон строк, за который отвечает этот блок.
    const int rows_per_block = kDv / gridDim.z;
    const int row_begin = blockIdx.z * rows_per_block;
    const int row_end = row_begin + rows_per_block;

    __shared__ float sk[kDk];
    __shared__ float sq[kDk];

    const float* kp = k + (size_t)(b * kHk + hk) * kDk;
    const float* qp = q + (size_t)(b * kHk + hk) * kDk;
    for (int i = threadIdx.x; i < kDk; i += kBlock) {
        sk[i] = kp[i];
        sq[i] = qp[i];
    }
    __syncthreads();

    const float a = alpha[b * kHv + h];
    const float bt = beta[b * kHv + h];
    // Скаляр k . q общий для всех строк и обеих веток: его считает prepare,
    // поэтому decode и повторный prefill читают ровно одно и то же число.
    const float kq_token = kq[b * kHk + hk];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    constexpr int kWarps = kBlock / 32;

    __nv_bfloat16* S = state + (size_t)(state_slot * kHv + h) * kDv * kDk;
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
            op[row] = a * w + c * kq_token;
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

// Chunk scan for one persistent sequence. Every warp keeps one state row in
// registers across all time steps.
//
// kRoundPerToken selects the rounding policy of the recurrent state inside a
// chunk. With rounding on, every step goes through BF16 and the chunk result
// matches repeated decode exactly. With it off, the row stays in FP32 registers
// for the whole chunk and only the write-back at the chunk boundary rounds —
// this is what the reference implementation does during prefill, so the switch
// isolates per-token rounding as a source of divergence.
template <bool kRoundPerToken>
__global__ __launch_bounds__(kBlock) void delta_prefill_kernel(
    __nv_bfloat16* __restrict__ state,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ alpha,
    const float* __restrict__ beta,
    const float* __restrict__ kq,
    float* __restrict__ out,
    int state_slot,
    int tokens) {
    // Варп владеет строкой состояния и идёт по токенам сам. Прежняя раскладка
    // ставила на токен три __syncthreads() и считала k.q одним варпом из
    // восьми, пока остальные стояли на барьере: рекуррентность и так
    // последовательна, а барьеры добавляли к ней блочную синхронизацию.
    // Строки состояния независимы — связывает их только скаляр k.q, который
    // варп теперь считает себе сам, в той же бабочке, что u и w.
    constexpr int kWarps = kBlock / 32;
    static_assert(kDv % kWarps == 0, "строки состояния делятся между варпами");

    const int h = blockIdx.x;
    const int hk = h / kRatio;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.z * kWarps + warp;
    const int base = lane * kPerThread;

    __nv_bfloat16* S = state + (size_t)(state_slot * kHv + h) * kDv * kDk;
    float2* srow = reinterpret_cast<float2*>(S + (size_t)row * kDk);
    float2 packed = srow[lane];
    __nv_bfloat162 p0 = *reinterpret_cast<__nv_bfloat162*>(&packed.x);
    __nv_bfloat162 p1 = *reinterpret_cast<__nv_bfloat162*>(&packed.y);
    float s[kPerThread];
    s[0] = __bfloat162float(p0.x);
    s[1] = __bfloat162float(p0.y);
    s[2] = __bfloat162float(p1.x);
    s[3] = __bfloat162float(p1.y);

    // Вход токена не зависит от состояния, поэтому загрузка следующего токена
    // выдаётся до того, как посчитан текущий: рекуррентность последовательна,
    // а её ожидание памяти — нет.
    const float* krow = k + (size_t)hk * kDk + base;
    const float* qrow = q + (size_t)hk * kDk + base;
    float4 kv = *reinterpret_cast<const float4*>(krow);
    float4 qv = *reinterpret_cast<const float4*>(qrow);

    for (int token = 0; token < tokens; ++token) {
        const float4 kv_cur = kv;
        const float4 qv_cur = qv;
        if (token + 1 < tokens) {
            kv = *reinterpret_cast<const float4*>(krow + (size_t)(token + 1) * kHk * kDk);
            qv = *reinterpret_cast<const float4*>(qrow + (size_t)(token + 1) * kHk * kDk);
        }

        float u = s[0] * kv_cur.x + s[1] * kv_cur.y + s[2] * kv_cur.z + s[3] * kv_cur.w;
        float w = s[0] * qv_cur.x + s[1] * qv_cur.y + s[2] * qv_cur.z + s[3] * qv_cur.w;
        warp_sum2(u, w);

        const float kq_token = kq[token * kHk + hk];
        const float a = alpha[token * kHv + h];
        const float bt = beta[token * kHv + h];
        const float value = v[(size_t)(token * kHv + h) * kDv + row];
        const float c = bt * (value - a * u);
        if (lane == 0) {
            out[(size_t)(token * kHv + h) * kDv + row] = a * w + c * kq_token;
        }

        const float update[kPerThread] = {
            a * s[0] + c * kv_cur.x,
            a * s[1] + c * kv_cur.y,
            a * s[2] + c * kv_cur.z,
            a * s[3] + c * kv_cur.w,
        };
        #pragma unroll
        for (int element = 0; element < kPerThread; ++element) {
            s[element] = kRoundPerToken
                ? __bfloat162float(__float2bfloat16(update[element]))
                : update[element];
        }
    }

    p0 = __nv_bfloat162(__float2bfloat16(s[0]), __float2bfloat16(s[1]));
    p1 = __nv_bfloat162(__float2bfloat16(s[2]), __float2bfloat16(s[3]));
    packed.x = *reinterpret_cast<float*>(&p0);
    packed.y = *reinterpret_cast<float*>(&p1);
    srow[lane] = packed;
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
    void* state, const uint32_t* state_slots,
    const float* q, const float* k, const float* v,
    const float* alpha, const float* beta, const float* kq, float* out,
    int state_capacity, int batch, cudaStream_t stream)
{
    if (state == nullptr || q == nullptr || k == nullptr || v == nullptr ||
        alpha == nullptr || beta == nullptr || kq == nullptr || out == nullptr ||
        state_capacity < batch || batch <= 0 || batch > 128) {
        return cudaErrorInvalidValue;
    }
    dim3 grid(kHv, batch, splits_for(batch));
    delta_decode_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<__nv_bfloat16*>(state), state_slots,
        q, k, v, alpha, beta, kq, out);
    return cudaGetLastError();
}

extern "C" cudaError_t qwc_delta_prefill(
    void* state,
    const float* q,
    const float* k,
    const float* v,
    const float* alpha,
    const float* beta,
    const float* kq,
    float* out,
    int state_capacity,
    int state_slot,
    int tokens,
    int round_state_per_token,
    int row_offset,
    cudaStream_t stream) {
    if (state == nullptr || q == nullptr || k == nullptr || v == nullptr ||
        alpha == nullptr || beta == nullptr || kq == nullptr || out == nullptr ||
        state_capacity <= 0 || state_slot < 0 || state_slot >= state_capacity ||
        tokens <= 0 || tokens > 1024 || round_state_per_token < 0 ||
        round_state_per_token > 1 || row_offset < 0) {
        return cudaErrorInvalidValue;
    }
    // The caller may hand in one slice of a fused multi-sequence token arena.
    // Advancing the bases here keeps the kernel indexing from row zero.
    q += (size_t)row_offset * kHk * kDk;
    k += (size_t)row_offset * kHk * kDk;
    v += (size_t)row_offset * kHv * kDv;
    alpha += (size_t)row_offset * kHv;
    beta += (size_t)row_offset * kHv;
    kq += (size_t)row_offset * kHk;
    out += (size_t)row_offset * kHv * kDv;
    // Одна строка состояния на варп: 48 голов x 16 блоков = 768 блоков на
    // 170 SM. splits_for здесь не нужен — prefill всегда идёт одной
    // последовательностью, и делить нечего, кроме строк.
    dim3 grid(kHv, 1, kDv / (kBlock / 32));
    if (round_state_per_token) {
        delta_prefill_kernel<true><<<grid, kBlock, 0, stream>>>(
            static_cast<__nv_bfloat16*>(state), q, k, v, alpha, beta, kq, out,
            state_slot, tokens);
    } else {
        delta_prefill_kernel<false><<<grid, kBlock, 0, stream>>>(
            static_cast<__nv_bfloat16*>(state), q, k, v, alpha, beta, kq, out,
            state_slot, tokens);
    }
    return cudaGetLastError();
}

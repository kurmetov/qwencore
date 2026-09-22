// Gated DeltaNet: рекуррентный шаг decode и чанковый скан prefill.
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
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "limits.cuh"

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

// Та же бабочка на N независимых сумм: шагов столько же, сколько у одной, а
// цепочек — N, и латентность shfl перекрывается ими, а не простоем.
template <int N>
__device__ __forceinline__ void warp_sum_n(float (&vals)[N]) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        #pragma unroll
        for (int index = 0; index < N; ++index) {
            vals[index] += __shfl_xor_sync(0xffffffff, vals[index], off);
        }
    }
}

// Строк состояния на варп в prefill. Одна строка означала, что все восемь
// варпов блока грузят одну и ту же строку k и q (адрес зависит только от
// дорожки), и так же делают все 16 блоков по z: k и q читались по 128 раз.
// Здесь же лежит и вторая беда — бабочка на двух значениях упиралась в
// латентность shfl. Обе лечатся одним: варп ведёт kRows строк сразу.
constexpr int kPrefillRows = 4;
// Prefill берёт свой размер блока: строк на голову 128, и при kPrefillRows их
// хватает либо на много мелких блоков, либо на мало крупных. Мелкие ровнее
// ложатся на 170 SM.
constexpr int kPrefillBlock = 64;

// Чанк матричной (WY) формы: она меняет проход по токенам на пять матричных
// умножений на тайл. 64 — компромисс: меньше, и подготовка (её стоимость на
// токен не зависит от длины чанка) перестаёт окупаться; больше, и матрица
// 64x64 на голову с её обратной перестаёт помещаться в shared блока.
constexpr int kWyChunk = 64;

// Шаг рекуррентности для одной строки состояния. Вынесен из кернелов, потому
// что строку decode читает из двух разных хранений (bf16 и int8), а
// расходиться этой математике между ними нельзя.
//
// Возвращает вклад строки в выход; состояние `s` обновляется на месте.
__device__ __forceinline__ float delta_row_step(
    float (&s)[kPerThread], const float* sk, const float* sq,
    int base, float a, float bt, float v_row, float kq_token) {
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

    const float c = bt * (v_row - a * u);
    #pragma unroll
    for (int t = 0; t < kPerThread; ++t) {
        s[t] = a * s[t] + c * sk[base + t];
    }
    return a * w + c * kq_token;
}

// ---------------------------------------------------------------------------
// Восьмибитное хранение состояния
// ---------------------------------------------------------------------------
//
// Строка состояния (128 элементов по dk) хранится как int8 плюс один
// масштаб f32 на строку. Масштаб именно построчный: вдоль строки идут оба
// скалярных произведения S k и S q, так что вся сумма делит один масштаб.
//
// int8, а не e4m3: строки плоские (пик близок к RMS), и четыре бита
// экспоненты им не нужны — int8 отдаёт их мантиссе и вчетверо точнее при том
// же байте (`bench/results/delta-state-8bit.md`).
//
// Варп владеет строкой, поэтому на поток приходится ровно kPerThread = 4
// байта: 32 дорожки дают слитую 128-байтовую транзакцию, как и bf16-путь со
// своими 256 байтами.

__device__ __forceinline__ void state_load_int8(
    const int8_t* __restrict__ row, float scale, int lane, float (&s)[kPerThread]) {
    const char4 packed = *reinterpret_cast<const char4*>(row + lane * kPerThread);
    s[0] = (float)packed.x * scale;
    s[1] = (float)packed.y * scale;
    s[2] = (float)packed.z * scale;
    s[3] = (float)packed.w * scale;
}

// Пишет строку и возвращает её масштаб. Масштаб считается по максимуму строки,
// поэтому перед записью нужна редукция по варпу — она же единственная цена
// 8-битного хранения сверх самой упаковки.
__device__ __forceinline__ float state_store_int8(
    int8_t* __restrict__ row, int lane, const float (&s)[kPerThread]) {
    float peak = 0.0f;
    #pragma unroll
    for (int t = 0; t < kPerThread; ++t) {
        peak = fmaxf(peak, fabsf(s[t]));
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        peak = fmaxf(peak, __shfl_xor_sync(0xffffffff, peak, off));
    }

    char4 packed;
    if (peak == 0.0f) {
        packed = make_char4(0, 0, 0, 0);
    } else {
        const float inverse = 127.0f / peak;
        int8_t level[kPerThread];
        #pragma unroll
        for (int t = 0; t < kPerThread; ++t) {
            level[t] = (int8_t)(int)fminf(fmaxf(rintf(s[t] * inverse), -127.0f), 127.0f);
        }
        packed = make_char4(level[0], level[1], level[2], level[3]);
    }
    *reinterpret_cast<char4*>(row + lane * kPerThread) = packed;
    return peak * (1.0f / 127.0f);
}

__global__ __launch_bounds__(kBlock) void delta_decode_int8_kernel(
    int8_t* __restrict__ state,          // [B, kHv, kDv, kDk]
    float* __restrict__ state_scales,    // [B, kHv, kDv]
    const uint32_t* __restrict__ state_slots,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ alpha,
    const float* __restrict__ beta,
    const float* __restrict__ kq,
    float* __restrict__ out)
{
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    const int state_slot = state_slots == nullptr ? b : (int)state_slots[b];
    const int hk = h / kRatio;

    const int rows_per_block = kDv / gridDim.z;
    const int row_begin = blockIdx.z * rows_per_block;
    const int row_end = row_begin + rows_per_block;

    __shared__ float sk[kDk];
    __shared__ float sq[kDk];
    // Масштабы блока проходят через shared, а не через глобальную память
    // построчно. Запись четырёх байт одной дорожкой — это частичная
    // транзакция на строку состояния, и она стоила 2.97 мс из 3.3 мс
    // накладных 8-битного хранения при c=32: дороже всей квантизации и всего
    // трафика самого состояния вместе взятых.
    __shared__ float srow_scale[kDv];

    const float* kp = k + (size_t)(b * kHk + hk) * kDk;
    const float* qp = q + (size_t)(b * kHk + hk) * kDk;
    for (int i = threadIdx.x; i < kDk; i += kBlock) {
        sk[i] = kp[i];
        sq[i] = qp[i];
    }

    const float a = alpha[b * kHv + h];
    const float bt = beta[b * kHv + h];
    const float kq_token = kq[b * kHk + hk];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    constexpr int kWarps = kBlock / 32;

    int8_t* S = state + (size_t)(state_slot * kHv + h) * kDv * kDk;
    float* scales = state_scales + (size_t)(state_slot * kHv + h) * kDv;
    const float* vp = v + (size_t)(b * kHv + h) * kDv;
    float* op = out + (size_t)(b * kHv + h) * kDv;

    // Чтение масштабов — одним слитным обращением на блок, в той же
    // загрузке, что k и q.
    const int rows_here = row_end - row_begin;
    for (int i = threadIdx.x; i < rows_here; i += kBlock) {
        srow_scale[i] = scales[row_begin + i];
    }
    __syncthreads();

    for (int row = row_begin + warp; row < row_end; row += kWarps) {
        int8_t* srow = S + (size_t)row * kDk;
        float s[kPerThread];
        state_load_int8(srow, srow_scale[row - row_begin], lane, s);

        const float contribution = delta_row_step(
            s, sk, sq, lane * kPerThread, a, bt, vp[row], kq_token);
        if (lane == 0) {
            op[row] = contribution;
        }

        const float scale = state_store_int8(srow, lane, s);
        if (lane == 0) {
            srow_scale[row - row_begin] = scale;
        }
    }

    __syncthreads();
    for (int i = threadIdx.x; i < rows_here; i += kBlock) {
        scales[row_begin + i] = srow_scale[i];
    }
}

// Перекладка слота между хранениями. Префилл последовательности идёт по
// расквантованной bf16-копии и квантуется один раз в конце промпта: если
// его сегменты перезапускать с 8-битного состояния, весь остаток промпта
// считается с испорченного S_0, и MAE подскакивает в 11 раз на границе
// `PREFILL_CHUNK_SIZE` (`bench/results/delta-state-8bit.md`).
__global__ __launch_bounds__(kBlock) void delta_state_pack_kernel(
    const __nv_bfloat16* __restrict__ source,
    int8_t* __restrict__ packed,
    float* __restrict__ scales,
    int source_slot, int packed_slot)
{
    const int h = blockIdx.x;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    constexpr int kWarps = kBlock / 32;

    const __nv_bfloat16* S = source + (size_t)(source_slot * kHv + h) * kDv * kDk;
    int8_t* D = packed + (size_t)(packed_slot * kHv + h) * kDv * kDk;
    float* scale_row = scales + (size_t)(packed_slot * kHv + h) * kDv;

    for (int row = blockIdx.z * (kDv / gridDim.z) + warp;
         row < (blockIdx.z + 1) * (kDv / gridDim.z); row += kWarps) {
        const float2 raw = reinterpret_cast<const float2*>(S + (size_t)row * kDk)[lane];
        const __nv_bfloat162 p0 = *reinterpret_cast<const __nv_bfloat162*>(&raw.x);
        const __nv_bfloat162 p1 = *reinterpret_cast<const __nv_bfloat162*>(&raw.y);
        float s[kPerThread] = {
            __bfloat162float(p0.x), __bfloat162float(p0.y),
            __bfloat162float(p1.x), __bfloat162float(p1.y)};

        const float scale = state_store_int8(D + (size_t)row * kDk, lane, s);
        if (lane == 0) {
            scale_row[row] = scale;
        }
    }
}

__global__ __launch_bounds__(kBlock) void delta_state_unpack_kernel(
    const int8_t* __restrict__ packed,
    const float* __restrict__ scales,
    __nv_bfloat16* __restrict__ destination,
    int packed_slot, int destination_slot)
{
    const int h = blockIdx.x;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    constexpr int kWarps = kBlock / 32;

    const int8_t* S = packed + (size_t)(packed_slot * kHv + h) * kDv * kDk;
    const float* scale_row = scales + (size_t)(packed_slot * kHv + h) * kDv;
    __nv_bfloat16* D = destination + (size_t)(destination_slot * kHv + h) * kDv * kDk;

    for (int row = blockIdx.z * (kDv / gridDim.z) + warp;
         row < (blockIdx.z + 1) * (kDv / gridDim.z); row += kWarps) {
        float s[kPerThread];
        state_load_int8(S + (size_t)row * kDk, scale_row[row], lane, s);

        const __nv_bfloat162 p0 =
            __nv_bfloat162(__float2bfloat16(s[0]), __float2bfloat16(s[1]));
        const __nv_bfloat162 p1 =
            __nv_bfloat162(__float2bfloat16(s[2]), __float2bfloat16(s[3]));
        float2 raw;
        raw.x = *reinterpret_cast<const float*>(&p0);
        raw.y = *reinterpret_cast<const float*>(&p1);
        reinterpret_cast<float2*>(D + (size_t)row * kDk)[lane] = raw;
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

        // Оба матвека за один проход, пока строка в регистрах, и обновление
        // на месте: повторное чтение не нужно.
        const float contribution = delta_row_step(
            s, sk, sq, lane * kPerThread, a, bt, vp[row], kq_token);

        if (lane == 0) {
            op[row] = contribution;
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
__global__ __launch_bounds__(kPrefillBlock) void delta_prefill_kernel(
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
    constexpr int kWarps = kPrefillBlock / 32;
    constexpr int kRows = kPrefillRows;
    static_assert(kDv % (kWarps * kRows) == 0, "строки состояния делятся между варпами");

    const int h = blockIdx.x;
    const int hk = h / kRatio;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row0 = (blockIdx.z * kWarps + warp) * kRows;
    const int base = lane * kPerThread;

    __nv_bfloat16* S = state + (size_t)(state_slot * kHv + h) * kDv * kDk;
    float s[kRows][kPerThread];
    #pragma unroll
    for (int r = 0; r < kRows; ++r) {
        float2* srow = reinterpret_cast<float2*>(S + (size_t)(row0 + r) * kDk);
        float2 packed = srow[lane];
        __nv_bfloat162 p0 = *reinterpret_cast<__nv_bfloat162*>(&packed.x);
        __nv_bfloat162 p1 = *reinterpret_cast<__nv_bfloat162*>(&packed.y);
        s[r][0] = __bfloat162float(p0.x);
        s[r][1] = __bfloat162float(p0.y);
        s[r][2] = __bfloat162float(p1.x);
        s[r][3] = __bfloat162float(p1.y);
    }

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

        // Первые kRows сумм — u каждой строки, вторые — её w. Одна бабочка на
        // все: k и q при этом прочитаны один раз на kRows строк, а не на одну.
        float red[2 * kRows];
        #pragma unroll
        for (int r = 0; r < kRows; ++r) {
            red[r] = s[r][0] * kv_cur.x + s[r][1] * kv_cur.y
                   + s[r][2] * kv_cur.z + s[r][3] * kv_cur.w;
            red[kRows + r] = s[r][0] * qv_cur.x + s[r][1] * qv_cur.y
                           + s[r][2] * qv_cur.z + s[r][3] * qv_cur.w;
        }
        warp_sum_n<2 * kRows>(red);

        const float kq_token = kq[token * kHk + hk];
        const float a = alpha[token * kHv + h];
        const float bt = beta[token * kHv + h];
        const size_t row_base = (size_t)(token * kHv + h) * kDv + row0;
        float result[kRows];
        #pragma unroll
        for (int r = 0; r < kRows; ++r) {
            const float c = bt * (v[row_base + r] - a * red[r]);
            result[r] = a * red[kRows + r] + c * kq_token;
            const float update[kPerThread] = {
                a * s[r][0] + c * kv_cur.x,
                a * s[r][1] + c * kv_cur.y,
                a * s[r][2] + c * kv_cur.z,
                a * s[r][3] + c * kv_cur.w,
            };
            #pragma unroll
            for (int element = 0; element < kPerThread; ++element) {
                s[r][element] = kRoundPerToken
                    ? __bfloat162float(__float2bfloat16(update[element]))
                    : update[element];
            }
        }
        if (lane == 0) {
            #pragma unroll
            for (int r = 0; r < kRows; ++r) {
                out[row_base + r] = result[r];
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < kRows; ++r) {
        __nv_bfloat162 p0 = __nv_bfloat162(__float2bfloat16(s[r][0]), __float2bfloat16(s[r][1]));
        __nv_bfloat162 p1 = __nv_bfloat162(__float2bfloat16(s[r][2]), __float2bfloat16(s[r][3]));
        float2 packed;
        packed.x = *reinterpret_cast<float*>(&p0);
        packed.y = *reinterpret_cast<float*>(&p1);
        reinterpret_cast<float2*>(S + (size_t)(row0 + r) * kDk)[lane] = packed;
    }
}

__device__ __forceinline__ uint32_t pack_bf16_pair(
    const __nv_bfloat16* source) {
    return *reinterpret_cast<const uint32_t*>(source);
}

__device__ __forceinline__ void wy_mma_m16n8k16(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
}

// Fixed-shape strided-batched GEMM used by all WY phases.  A is row-major;
// B is logical column-major.  `a_group`/`b_group` express the 1:3 broadcast
// from key heads to value heads without copying either operand.
__global__ __launch_bounds__(32) void wy_gemm_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    float* __restrict__ destination,
    const float* __restrict__ addend,
    const float* __restrict__ output_row_scale,
    const float* __restrict__ addend_batch_scale,
    int m, int n, int inner, int batches,
    size_t a_batch_stride, size_t b_batch_stride,
    size_t destination_batch_stride, size_t addend_batch_stride,
    int a_row_stride, int b_column_stride,
    int destination_row_stride, int addend_row_stride,
    int a_group, int b_group,
    int output_scale_batch_stride,
    int addend_scale_batch_stride, int addend_scale_offset,
    float addend_scale) {
    const int batch = blockIdx.z;
    if (batch >= batches) {
        return;
    }
    const int lane = threadIdx.x;
    const int group = lane >> 2;
    const int position = lane & 3;
    const int row0 = blockIdx.y * 16;
    const int column0 = blockIdx.x * 8;
    const __nv_bfloat16* aa = a + (size_t)(batch / a_group) * a_batch_stride;
    const __nv_bfloat16* bb = b + (size_t)(batch / b_group) * b_batch_stride;
    float values[4] = {};
    for (int k0 = 0; k0 < inner; k0 += 16) {
        uint32_t af[4];
        uint32_t bf[2];
        af[0] = pack_bf16_pair(aa + (size_t)(row0 + group) * a_row_stride + k0 + position * 2);
        af[1] = pack_bf16_pair(aa + (size_t)(row0 + group + 8) * a_row_stride + k0 + position * 2);
        af[2] = pack_bf16_pair(aa + (size_t)(row0 + group) * a_row_stride + k0 + position * 2 + 8);
        af[3] = pack_bf16_pair(aa + (size_t)(row0 + group + 8) * a_row_stride + k0 + position * 2 + 8);
        // B[k, column] is physically a row belonging to `column`.
        bf[0] = pack_bf16_pair(bb + (size_t)(column0 + group) * b_column_stride + k0 + position * 2);
        bf[1] = pack_bf16_pair(bb + (size_t)(column0 + group) * b_column_stride + k0 + position * 2 + 8);
        wy_mma_m16n8k16(values, af, bf);
    }

    float batch_addend_scale = addend_scale;
    if (addend_batch_scale != nullptr) {
        batch_addend_scale *= addend_batch_scale[
            batch * addend_scale_batch_stride + addend_scale_offset];
    }
    #pragma unroll
    for (int index = 0; index < 4; ++index) {
        const int row = row0 + group + (index >= 2 ? 8 : 0);
        const int column = column0 + position * 2 + (index & 1);
        if (row >= m || column >= n) {
            continue;
        }
        float value = values[index];
        if (output_row_scale != nullptr) {
            value *= output_row_scale[batch * output_scale_batch_stride + row];
        }
        const size_t destination_offset = (size_t)batch * destination_batch_stride
            + (size_t)row * destination_row_stride + column;
        if (addend != nullptr) {
            const size_t addend_offset = (size_t)batch * addend_batch_stride
                + (size_t)row * addend_row_stride + column;
            value += batch_addend_scale * addend[addend_offset];
        }
        destination[destination_offset] = value;
    }
}

// Тайлы q и k лежат как [чанк][голова][токен][измерение]: индекс батча в
// `wy_gemm_kernel` — это chunk * головы + голова, поэтому раздача одной
// k-головы на три v-головы (`b_group`) остаётся делением индекса.
//
// Ни одна фаза до слитого ядра не зависит от состояния, поэтому все они
// считаются сразу по всем чанкам сегмента — по одному запуску на слой.
__global__ void wy_prepare_qk_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    __nv_bfloat16* __restrict__ query_tile,
    __nv_bfloat16* __restrict__ key_tile,
    int tokens, int chunks) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr int kPerChunk = kHk * kWyChunk * kDk;
    if (index >= chunks * kPerChunk) {
        return;
    }
    const int dimension = index % kDk;
    const int token = (index / kDk) % kWyChunk;
    const int head = (index / (kWyChunk * kDk)) % kHk;
    const int chunk = index / kPerChunk;
    const int row = chunk * kWyChunk + token;
    __nv_bfloat16 qv = __float2bfloat16(0.0f);
    __nv_bfloat16 kv = __float2bfloat16(0.0f);
    if (row < tokens) {
        const size_t source = (size_t)row * kHk * kDk
            + (size_t)head * kDk + dimension;
        qv = __float2bfloat16(q[source]);
        kv = __float2bfloat16(k[source]);
    }
    query_tile[index] = qv;
    key_tile[index] = kv;
}

// Затухание копится в логарифмах, а не произведением.
//
// Произведение alpha по чанку — это то самое gamma_t, на которое домножается
// вклад входного состояния. На 64 токенах оно легко уходит под 1e-38: при
// alpha = 0.2 уже к концу чанка. Дальше отношения gamma_t / gamma_r
// превращаются в 0/0, и скан выдаёт NaN. Разность логарифмов всегда конечна,
// а exp от неё либо честный ноль, либо число.
__global__ void wy_decay_kernel(
    const float* __restrict__ alpha,
    float* __restrict__ log_decay,
    int tokens, int chunks) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= chunks * kHv) {
        return;
    }
    const int head = index % kHv;
    const int start = (index / kHv) * kWyChunk;
    float* destination = log_decay + (size_t)index * kWyChunk;
    float sum = 0.0f;
    for (int token = 0; token < kWyChunk; ++token) {
        if (start + token < tokens) {
            // Нижняя граница держит логарифм конечным: alpha = 0 дало бы
            // -inf в обеих частях разности и NaN вместо нуля.
            sum += __logf(fmaxf(alpha[(size_t)(start + token) * kHv + head],
                                1.0e-38f));
            destination[token] = sum;
        } else {
            destination[token] = -INFINITY;
        }
    }
}

// Множители причинной части и обратная к единично-нижнетреугольной A = I + M.
//
// Обе строятся из одних и тех же граммов и gamma, поэтому считаются одним
// блоком: M не выходит за пределы shared и в глобальную память не попадает.
// Прямая подстановка идёт по столбцам, а столбцы независимы — поток владеет
// столбцом и читает только его, так что барьер нужен один, после сборки M.
// Раньше подстановка шла в пространстве значений (128 измерений на токен) и
// перечитывала коэффициенты из глобальной памяти: 50 МБ на чанк слоя и две
// трети времени всего скана.
__global__ __launch_bounds__(kWyChunk) void wy_factor_kernel(
    const float* __restrict__ beta,
    const float* __restrict__ log_decay,
    const float* __restrict__ gram_kk,
    const float* __restrict__ gram_qk,
    __nv_bfloat16* __restrict__ inverse,
    __nv_bfloat16* __restrict__ output_factor,
    int tokens) {
    __shared__ float m[kWyChunk][kWyChunk];
    __shared__ float t[kWyChunk][kWyChunk];
    const int head = blockIdx.x;
    const int chunk = blockIdx.y;
    const int column = threadIdx.x;
    const int start = chunk * kWyChunk;
    const int length = min(kWyChunk, tokens - start);
    const float* chunk_decay =
        log_decay + (size_t)(chunk * kHv + head) * kWyChunk;
    const size_t gram_base =
        (size_t)(chunk * kHk + head / kRatio) * kWyChunk * kWyChunk;
    const size_t head_base = (size_t)(chunk * kHv + head) * kWyChunk * kWyChunk;
    const float column_decay = chunk_decay[column];

    for (int row = 0; row < kWyChunk; ++row) {
        float a = 0.0f;
        float f = 0.0f;
        if (row < length && column <= row) {
            const float ratio = __expf(chunk_decay[row] - column_decay);
            const size_t offset = gram_base + (size_t)row * kWyChunk + column;
            f = ratio * gram_qk[offset];
            if (column < row) {
                a = beta[(size_t)(start + row) * kHv + head] * ratio
                    * gram_kk[offset];
            }
        }
        m[row][column] = a;
        output_factor[head_base + (size_t)row * kWyChunk + column] =
            __float2bfloat16(f);
    }
    __syncthreads();

    // Строки за длиной чанка остаются единичными: правая часть там занулена,
    // так что нули доходят до конца сами, без масок в следующих фазах.
    //
    // Подстановка идёт панелями по 16 строк. Наивная форма читала из shared
    // дважды на каждую FMA — и множитель, и уже посчитанное значение, — и
    // упиралась в это, а не в арифметику. В панельной форме значение
    // предыдущей строки читается один раз на шестнадцать FMA, а внутри
    // панели вообще живёт в регистрах: обращений к shared вдвое меньше.
    constexpr int kPanel = 16;
    for (int panel = 0; panel < kWyChunk / kPanel; ++panel) {
        const int base = panel * kPanel;
        float accumulator[kPanel];
        #pragma unroll
        for (int index = 0; index < kPanel; ++index) {
            accumulator[index] = (base + index == column) ? 1.0f : 0.0f;
        }
        for (int previous = 0; previous < base; ++previous) {
            const float value = t[previous][column];
            #pragma unroll
            for (int index = 0; index < kPanel; ++index) {
                accumulator[index] -= m[base + index][previous] * value;
            }
        }
        float solved[kPanel];
        #pragma unroll
        for (int index = 0; index < kPanel; ++index) {
            float value = accumulator[index];
            #pragma unroll
            for (int inner = 0; inner < index; ++inner) {
                value -= m[base + index][base + inner] * solved[inner];
            }
            solved[index] = value;
            t[base + index][column] = value;
            inverse[head_base + (size_t)(base + index) * kWyChunk + column] =
                __float2bfloat16(value);
        }
    }
}

// Правая часть системы, сразу транспонированная: [голова][измерение][токен].
// P на входе — это S_0 K^T в той же раскладке, поэтому транспонирования между
// фазами не нужно ни одного.
// ------------------------------------------------------------------ //
// Слитый последовательный скан: один запуск на слой вместо восьми на чанк.
//
// Последовательность чанков разорвать нельзя — состояние конца чанка нужно
// следующему. Но всё, что между ними, помещается в один варп: варп владеет
// 16 строками состояния (dv) на всю глубину dk и делает над ними пять
// матричных умножений подряд, ни разу не выкладывая состояние в память.
// Раньше те же пять умножений были пятью запусками, и состояние ходило в
// глобальную память и обратно на каждом чанке.
//
// Раскладка фрагментов m16n8k16 одна и та же для трёх ролей состояния:
// аккумулятор (строка dv, столбец dk), операнд A в P = S K^T и операнд B в
// Q S^T. Во всех трёх дорожка держит строки grp и grp+8 и пару столбцов с
// pos*2 — поэтому упакованная BF16-копия аккумулятора годится как есть и
// перекладка не нужна.
constexpr int kFusedWarps = 4;
constexpr int kFusedThreads = kFusedWarps * 32;
constexpr int kFusedRows = 16;                       // dv-строк на варп
constexpr int kFusedSlice = kFusedWarps * kFusedRows;
constexpr int kFusedSlices = kDv / kFusedSlice;
constexpr int kDkTiles = kDk / 8;                    // n-тайлов состояния
constexpr int kChunkTiles = kWyChunk / 8;            // n-тайлов по токенам
constexpr int kKeyPad = kDk + 8;

struct WyFusedShared {
    __nv_bfloat16 key[kWyChunk * kKeyPad];
    float gamma[kWyChunk];
    float weight[kWyChunk];   // gamma_C / gamma_t — вес чанковой границы
    float decay[kWyChunk];    // beta_t * gamma_t — вклад входного состояния
};

constexpr size_t wy_fused_shared_bytes() { return sizeof(WyFusedShared); }

__device__ __forceinline__ uint32_t wy_pack(float low, float high) {
    const __nv_bfloat16 a = __float2bfloat16(low);
    const __nv_bfloat16 b = __float2bfloat16(high);
    return static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(&a))
        | (static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(&b)) << 16);
}

__device__ __forceinline__ uint32_t wy_pack_shared(
    const __nv_bfloat16* source) {
    return *reinterpret_cast<const uint32_t*>(source);
}

__device__ __forceinline__ uint32_t wy_pack_strided(
    const __nv_bfloat16* low, const __nv_bfloat16* high) {
    return static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(low))
        | (static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(high)) << 16);
}

// Копия строк в shared по 8 элементов (16 байт) за раз. Набивка выбрана так,
// что шаг ряда кратен 16 байтам, иначе векторная запись развалится.
template <int kRow, int kPad>
__device__ __forceinline__ void wy_stage_rows(
    const __nv_bfloat16* __restrict__ source, __nv_bfloat16* destination,
    int rows, int thread, int threads) {
    constexpr int kVector = 8;
    constexpr int kPerRow = kRow / kVector;
    for (int index = thread; index < rows * kPerRow; index += threads) {
        const int row = index / kPerRow;
        const int column = (index % kPerRow) * kVector;
        *reinterpret_cast<float4*>(destination + row * kPad + column) =
            *reinterpret_cast<const float4*>(source + (size_t)row * kRow + column);
    }
}

__global__ __launch_bounds__(kFusedThreads, 1) void wy_fused_scan_kernel(
    const __nv_bfloat16* __restrict__ key_tile,
    const __nv_bfloat16* __restrict__ query_tile,
    const __nv_bfloat16* __restrict__ inverse,
    const __nv_bfloat16* __restrict__ factor,
    const __nv_bfloat16* __restrict__ value_tile,
    const float* __restrict__ log_decay,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ state,
    float* __restrict__ out,
    int tokens, int chunks) {
    extern __shared__ __align__(16) char wy_fused_raw[];
    WyFusedShared& shared = *reinterpret_cast<WyFusedShared*>(wy_fused_raw);

    const int head = blockIdx.x;
    const int key_head = head / kRatio;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int grp = lane >> 2;
    const int pos = lane & 3;
    const int row0 = blockIdx.y * kFusedSlice + warp * kFusedRows;

    // Состояние: строки row0+grp и row0+grp+8, столбцы tile*8+pos*2 и +1.
    float state_acc[kDkTiles][4];
    {
        const __nv_bfloat16* source = state + (size_t)head * kDv * kDk;
        #pragma unroll
        for (int tile = 0; tile < kDkTiles; ++tile) {
            const int column = tile * 8 + pos * 2;
            const uint32_t low = *reinterpret_cast<const uint32_t*>(
                source + (size_t)(row0 + grp) * kDk + column);
            const uint32_t high = *reinterpret_cast<const uint32_t*>(
                source + (size_t)(row0 + grp + 8) * kDk + column);
            const __nv_bfloat16* lowp = reinterpret_cast<const __nv_bfloat16*>(&low);
            const __nv_bfloat16* highp = reinterpret_cast<const __nv_bfloat16*>(&high);
            state_acc[tile][0] = __bfloat162float(lowp[0]);
            state_acc[tile][1] = __bfloat162float(lowp[1]);
            state_acc[tile][2] = __bfloat162float(highp[0]);
            state_acc[tile][3] = __bfloat162float(highp[1]);
        }
    }

    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int start = chunk * kWyChunk;
        const int length = min(kWyChunk, tokens - start);
        const size_t qk_base =
            ((size_t)chunk * kHk + key_head) * kWyChunk * kDk;
        const size_t head_base =
            ((size_t)chunk * kHv + head) * kWyChunk * kWyChunk;
        const float* chunk_decay =
            log_decay + (size_t)(chunk * kHv + head) * kWyChunk;

        __syncthreads();
        wy_stage_rows<kDk, kKeyPad>(
            key_tile + qk_base, shared.key, kWyChunk, threadIdx.x, kFusedThreads);
        for (int token = threadIdx.x; token < kWyChunk; token += kFusedThreads) {
            const float cumulative = chunk_decay[token];
            const float value = __expf(cumulative);
            shared.gamma[token] = value;
            if (token < length) {
                // Вес границы чанка — отношение затуханий, то есть exp от
                // разности логарифмов: оно не больше единицы при любом alpha.
                shared.weight[token] =
                    __expf(chunk_decay[length - 1] - cumulative);
                shared.decay[token] =
                    beta[(size_t)(start + token) * kHv + head] * value;
            } else {
                shared.weight[token] = 0.0f;
                shared.decay[token] = 0.0f;
            }
        }
        __syncthreads();

        // BF16-копия состояния: пара (строка, строка+8) на каждый столбцовый
        // тайл. Она же операнд A для S K^T и операнд B для Q S^T.
        uint32_t state_pack[kDkTiles][2];
        #pragma unroll
        for (int tile = 0; tile < kDkTiles; ++tile) {
            state_pack[tile][0] = wy_pack(state_acc[tile][0], state_acc[tile][1]);
            state_pack[tile][1] = wy_pack(state_acc[tile][2], state_acc[tile][3]);
        }

        // P = S_0 K^T: [16 строк dv] x [64 токена].
        float product[kChunkTiles][4] = {};
        #pragma unroll
        for (int group = 0; group < kDk / 16; ++group) {
            const uint32_t a[4] = {
                state_pack[2 * group][0], state_pack[2 * group][1],
                state_pack[2 * group + 1][0], state_pack[2 * group + 1][1]};
            #pragma unroll
            for (int tile = 0; tile < kChunkTiles; ++tile) {
                const __nv_bfloat16* row =
                    shared.key + (size_t)(tile * 8 + grp) * kKeyPad + group * 16;
                const uint32_t b[2] = {
                    wy_pack_shared(row + pos * 2),
                    wy_pack_shared(row + pos * 2 + 8)};
                wy_mma_m16n8k16(product[tile], a, b);
            }
        }

        // Правая часть: b_t v_t - b_t gamma_t S_0 k_t, сразу в BF16-фрагменты
        // операнда A для умножения на T^T.
        uint32_t right[kChunkTiles][2];
        {
            const __nv_bfloat16* values =
                value_tile + ((size_t)chunk * kHv + head) * kDv * kWyChunk;
            #pragma unroll
            for (int tile = 0; tile < kChunkTiles; ++tile) {
                const int token = tile * 8 + pos * 2;
                const float d0 = shared.decay[token];
                const float d1 = shared.decay[token + 1];
                const uint32_t low = *reinterpret_cast<const uint32_t*>(
                    values + (size_t)(row0 + grp) * kWyChunk + token);
                const uint32_t high = *reinterpret_cast<const uint32_t*>(
                    values + (size_t)(row0 + grp + 8) * kWyChunk + token);
                const __nv_bfloat16* lowp = reinterpret_cast<const __nv_bfloat16*>(&low);
                const __nv_bfloat16* highp = reinterpret_cast<const __nv_bfloat16*>(&high);
                right[tile][0] = wy_pack(
                    __bfloat162float(lowp[0]) - d0 * product[tile][0],
                    __bfloat162float(lowp[1]) - d1 * product[tile][1]);
                right[tile][1] = wy_pack(
                    __bfloat162float(highp[0]) - d0 * product[tile][2],
                    __bfloat162float(highp[1]) - d1 * product[tile][3]);
            }
        }

        // C^T = RHS^T T^T.
        float coefficient[kChunkTiles][4] = {};
        #pragma unroll
        for (int group = 0; group < kWyChunk / 16; ++group) {
            const uint32_t a[4] = {
                right[2 * group][0], right[2 * group][1],
                right[2 * group + 1][0], right[2 * group + 1][1]};
            #pragma unroll
            for (int tile = 0; tile < kChunkTiles; ++tile) {
                const __nv_bfloat16* row = inverse + head_base
                    + (size_t)(tile * 8 + grp) * kWyChunk + group * 16;
                const uint32_t b[2] = {
                    wy_pack_shared(row + pos * 2),
                    wy_pack_shared(row + pos * 2 + 8)};
                wy_mma_m16n8k16(coefficient[tile], a, b);
            }
        }

        // O = diag(gamma) Q S_0^T + F C. Строки здесь — токены, столбцы —
        // 16 dv-строк варпа, поэтому m-тайлов четыре, а n-тайлов два.
        constexpr int kOutRowTiles = kWyChunk / 16;
        float output[kOutRowTiles][2][4] = {};
        #pragma unroll
        for (int group = 0; group < kDk / 16; ++group) {
            #pragma unroll
            for (int mt = 0; mt < kOutRowTiles; ++mt) {
                // Q читается фрагментами прямо из глобальной памяти: все
                // варпы CTA берут одни и те же элементы, их держит L1, а
                // освободившиеся 17 КБ shared дороже этих обращений.
                const __nv_bfloat16* row = query_tile + qk_base
                    + (size_t)(mt * 16 + grp) * kDk + group * 16;
                const __nv_bfloat16* row8 = query_tile + qk_base
                    + (size_t)(mt * 16 + grp + 8) * kDk + group * 16;
                const uint32_t a[4] = {
                    wy_pack_shared(row + pos * 2),
                    wy_pack_shared(row8 + pos * 2),
                    wy_pack_shared(row + pos * 2 + 8),
                    wy_pack_shared(row8 + pos * 2 + 8)};
                #pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    const uint32_t b[2] = {
                        state_pack[2 * group][nt], state_pack[2 * group + 1][nt]};
                    wy_mma_m16n8k16(output[mt][nt], a, b);
                }
            }
        }
        #pragma unroll
        for (int mt = 0; mt < kOutRowTiles; ++mt) {
            const float g0 = shared.gamma[mt * 16 + grp];
            const float g8 = shared.gamma[mt * 16 + grp + 8];
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                output[mt][nt][0] *= g0;
                output[mt][nt][1] *= g0;
                output[mt][nt][2] *= g8;
                output[mt][nt][3] *= g8;
            }
        }
        // C в роли операнда B: n — это dv-строка, k — токен.
        uint32_t coefficient_pack[kChunkTiles][2];
        #pragma unroll
        for (int tile = 0; tile < kChunkTiles; ++tile) {
            coefficient_pack[tile][0] =
                wy_pack(coefficient[tile][0], coefficient[tile][1]);
            coefficient_pack[tile][1] =
                wy_pack(coefficient[tile][2], coefficient[tile][3]);
        }
        #pragma unroll
        for (int group = 0; group < kWyChunk / 16; ++group) {
            #pragma unroll
            for (int mt = 0; mt < kOutRowTiles; ++mt) {
                const __nv_bfloat16* row = factor + head_base
                    + (size_t)(mt * 16 + grp) * kWyChunk + group * 16;
                const __nv_bfloat16* row8 = factor + head_base
                    + (size_t)(mt * 16 + grp + 8) * kWyChunk + group * 16;
                const uint32_t a[4] = {
                    wy_pack_shared(row + pos * 2),
                    wy_pack_shared(row8 + pos * 2),
                    wy_pack_shared(row + pos * 2 + 8),
                    wy_pack_shared(row8 + pos * 2 + 8)};
                #pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    const uint32_t b[2] = {
                        coefficient_pack[2 * group][nt],
                        coefficient_pack[2 * group + 1][nt]};
                    wy_mma_m16n8k16(output[mt][nt], a, b);
                }
            }
        }
        #pragma unroll
        for (int mt = 0; mt < kOutRowTiles; ++mt) {
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const int column = row0 + nt * 8 + pos * 2;
                const int token0 = mt * 16 + grp;
                const int token8 = token0 + 8;
                float* base = out + ((size_t)(start + token0) * kHv + head) * kDv;
                if (token0 < length) {
                    base[column] = output[mt][nt][0];
                    base[column + 1] = output[mt][nt][1];
                }
                if (token8 < length) {
                    float* base8 = out + ((size_t)(start + token8) * kHv + head) * kDv;
                    base8[column] = output[mt][nt][2];
                    base8[column + 1] = output[mt][nt][3];
                }
            }
        }

        // S_C = gamma_C S_0 + C_weighted^T K.
        const float last = shared.gamma[length - 1];
        #pragma unroll
        for (int tile = 0; tile < kDkTiles; ++tile) {
            #pragma unroll
            for (int index = 0; index < 4; ++index) {
                state_acc[tile][index] *= last;
            }
        }
        #pragma unroll
        for (int tile = 0; tile < kChunkTiles; ++tile) {
            const int token = tile * 8 + pos * 2;
            const float w0 = shared.weight[token];
            const float w1 = shared.weight[token + 1];
            coefficient_pack[tile][0] = wy_pack(
                coefficient[tile][0] * w0, coefficient[tile][1] * w1);
            coefficient_pack[tile][1] = wy_pack(
                coefficient[tile][2] * w0, coefficient[tile][3] * w1);
        }
        #pragma unroll
        for (int group = 0; group < kWyChunk / 16; ++group) {
            const uint32_t a[4] = {
                coefficient_pack[2 * group][0], coefficient_pack[2 * group][1],
                coefficient_pack[2 * group + 1][0], coefficient_pack[2 * group + 1][1]};
            #pragma unroll
            for (int tile = 0; tile < kDkTiles; ++tile) {
                const int column = tile * 8 + grp;
                const __nv_bfloat16* base = shared.key + column;
                const int token = group * 16 + pos * 2;
                const uint32_t b[2] = {
                    wy_pack_strided(base + (size_t)token * kKeyPad,
                                    base + (size_t)(token + 1) * kKeyPad),
                    wy_pack_strided(base + (size_t)(token + 8) * kKeyPad,
                                    base + (size_t)(token + 9) * kKeyPad)};
                wy_mma_m16n8k16(state_acc[tile], a, b);
            }
        }
    }

    // Состояние последовательности переживает шаг, поэтому пишется обратно
    // один раз, в конце.
    {
        __nv_bfloat16* destination = state + (size_t)head * kDv * kDk;
        #pragma unroll
        for (int tile = 0; tile < kDkTiles; ++tile) {
            const int column = tile * 8 + pos * 2;
            const uint32_t low = wy_pack(state_acc[tile][0], state_acc[tile][1]);
            const uint32_t high = wy_pack(state_acc[tile][2], state_acc[tile][3]);
            *reinterpret_cast<uint32_t*>(
                destination + (size_t)(row0 + grp) * kDk + column) = low;
            *reinterpret_cast<uint32_t*>(
                destination + (size_t)(row0 + grp + 8) * kDk + column) = high;
        }
    }
}

// Тайл значений в BF16 и транспонированный: [чанк][голова][измерение][токен].
// Слитое ядро читает его в раскладке аккумулятора, где строка — измерение, а
// столбец — токен; транспонировать там было бы негде.
__global__ __launch_bounds__(256) void wy_prepare_value_kernel(
    const float* __restrict__ v,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ value_tile,
    int tokens) {
    __shared__ float tile[kWyChunk][kDv + 1];
    const int head = blockIdx.x;
    const int chunk = blockIdx.y;
    const int start = chunk * kWyChunk;
    const int length = min(kWyChunk, tokens - start);
    for (int index = threadIdx.x; index < kWyChunk * kDv; index += 256) {
        const int dimension = index % kDv;
        const int token = index / kDv;
        float value = 0.0f;
        if (token < length) {
            const size_t row = (size_t)(start + token) * kHv + head;
            value = beta[row] * v[row * kDv + dimension];
        }
        tile[token][dimension] = value;
    }
    __syncthreads();
    __nv_bfloat16* destination =
        value_tile + ((size_t)chunk * kHv + head) * kDv * kWyChunk;
    for (int index = threadIdx.x; index < kDv * kWyChunk; index += 256) {
        const int token = index % kWyChunk;
        const int dimension = index / kWyChunk;
        destination[index] = __float2bfloat16(tile[token][dimension]);
    }
}

static void launch_wy_gemm(
    const __nv_bfloat16* a, const __nv_bfloat16* b,
    float* destination, const float* addend,
    const float* output_row_scale, const float* addend_batch_scale,
    int m, int n, int inner, int batches,
    size_t a_batch_stride, size_t b_batch_stride,
    size_t destination_batch_stride, size_t addend_batch_stride,
    int a_row_stride, int b_column_stride,
    int destination_row_stride, int addend_row_stride,
    int a_group, int b_group, int output_scale_batch_stride,
    int addend_scale_batch_stride, int addend_scale_offset,
    float addend_scale, cudaStream_t stream) {
    dim3 grid((n + 7) / 8, (m + 15) / 16, batches);
    wy_gemm_kernel<<<grid, 32, 0, stream>>>(
        a, b, destination, addend, output_row_scale, addend_batch_scale,
        m, n, inner, batches, a_batch_stride, b_batch_stride,
        destination_batch_stride, addend_batch_stride,
        a_row_stride, b_column_stride, destination_row_stride,
        addend_row_stride, a_group, b_group, output_scale_batch_stride,
        addend_scale_batch_stride, addend_scale_offset, addend_scale);
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

extern "C" cudaError_t qwc_delta_decode_int8(
    void* state, float* state_scales, const uint32_t* state_slots,
    const float* q, const float* k, const float* v,
    const float* alpha, const float* beta, const float* kq, float* out,
    int state_capacity, int batch, cudaStream_t stream)
{
    if (state == nullptr || state_scales == nullptr || q == nullptr ||
        k == nullptr || v == nullptr || alpha == nullptr || beta == nullptr ||
        kq == nullptr || out == nullptr ||
        state_capacity < batch || batch <= 0 || batch > 128) {
        return cudaErrorInvalidValue;
    }
    dim3 grid(kHv, batch, splits_for(batch));
    delta_decode_int8_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<int8_t*>(state), state_scales, state_slots,
        q, k, v, alpha, beta, kq, out);
    return cudaGetLastError();
}

// Один слот bf16 -> один слот int8 и обратно. Оба конца адресуются слотами,
// потому что bf16-копия у движка одна, а 8-битных слотов столько же, сколько
// последовательностей.
extern "C" cudaError_t qwc_delta_state_pack(
    const void* source, void* packed, float* scales,
    int source_capacity, int source_slot,
    int packed_capacity, int packed_slot, cudaStream_t stream)
{
    if (source == nullptr || packed == nullptr || scales == nullptr ||
        source_slot < 0 || source_slot >= source_capacity ||
        packed_slot < 0 || packed_slot >= packed_capacity) {
        return cudaErrorInvalidValue;
    }
    dim3 grid(kHv, 1, splits_for(1));
    delta_state_pack_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(source), static_cast<int8_t*>(packed),
        scales, source_slot, packed_slot);
    return cudaGetLastError();
}

extern "C" cudaError_t qwc_delta_state_unpack(
    const void* packed, const float* scales, void* destination,
    int packed_capacity, int packed_slot,
    int destination_capacity, int destination_slot, cudaStream_t stream)
{
    if (packed == nullptr || scales == nullptr || destination == nullptr ||
        packed_slot < 0 || packed_slot >= packed_capacity ||
        destination_slot < 0 || destination_slot >= destination_capacity) {
        return cudaErrorInvalidValue;
    }
    dim3 grid(kHv, 1, splits_for(1));
    delta_state_unpack_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const int8_t*>(packed), scales,
        static_cast<__nv_bfloat16*>(destination), packed_slot, destination_slot);
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
        tokens <= 0 || tokens > qwc::kMaxStepRows || round_state_per_token < 0 ||
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
    // kPrefillRows строк на варп, kPrefillBlock/32 варпа на блок: 48 голов x
    // 16 блоков = 768 блоков на 170 SM. splits_for здесь не нужен — prefill
    // всегда идёт одной последовательностью, и делить нечего, кроме строк.
    dim3 grid(kHv, 1, kDv / ((kPrefillBlock / 32) * kPrefillRows));
    if (round_state_per_token) {
        delta_prefill_kernel<true><<<grid, kPrefillBlock, 0, stream>>>(
            static_cast<__nv_bfloat16*>(state), q, k, v, alpha, beta, kq, out,
            state_slot, tokens);
    } else {
        delta_prefill_kernel<false><<<grid, kPrefillBlock, 0, stream>>>(
            static_cast<__nv_bfloat16*>(state), q, k, v, alpha, beta, kq, out,
            state_slot, tokens);
    }
    return cudaGetLastError();
}

extern "C" cudaError_t qwc_delta_prefill_wy(
    void* state,
    const float* q,
    const float* k,
    const float* v,
    const float* alpha,
    const float* beta,
    float* out,
    void* query_tile_pointer,
    void* key_tile_pointer,
    void* gram_kk_pointer,
    void* gram_qk_pointer,
    void* inverse_pointer,
    void* output_factor_pointer,
    void* value_tile_pointer,
    void* log_decay_pointer,
    int state_capacity,
    int state_slot,
    int tokens,
    int row_offset,
    cudaStream_t stream) {
    if (state == nullptr || q == nullptr || k == nullptr || v == nullptr ||
        alpha == nullptr || beta == nullptr || out == nullptr ||
        query_tile_pointer == nullptr || key_tile_pointer == nullptr ||
        gram_kk_pointer == nullptr || gram_qk_pointer == nullptr ||
        inverse_pointer == nullptr || output_factor_pointer == nullptr ||
        value_tile_pointer == nullptr || log_decay_pointer == nullptr ||
        state_capacity <= 0 || state_slot < 0 ||
        state_slot >= state_capacity || tokens <= 0 ||
        tokens > qwc::kMaxStepRows || row_offset < 0) {
        return cudaErrorInvalidValue;
    }

    auto* state_pool = static_cast<__nv_bfloat16*>(state);
    auto* query_tile = static_cast<__nv_bfloat16*>(query_tile_pointer);
    auto* key_tile = static_cast<__nv_bfloat16*>(key_tile_pointer);
    auto* gram_kk = static_cast<float*>(gram_kk_pointer);
    auto* gram_qk = static_cast<float*>(gram_qk_pointer);
    auto* inverse = static_cast<__nv_bfloat16*>(inverse_pointer);
    auto* output_factor = static_cast<__nv_bfloat16*>(output_factor_pointer);
    auto* value_tile = static_cast<__nv_bfloat16*>(value_tile_pointer);
    auto* log_decay = static_cast<float*>(log_decay_pointer);

    // Дальше все координаты строк — внутри выбранного сегмента арены.
    q += (size_t)row_offset * kHk * kDk;
    k += (size_t)row_offset * kHk * kDk;
    v += (size_t)row_offset * kHv * kDv;
    alpha += (size_t)row_offset * kHv;
    beta += (size_t)row_offset * kHv;
    out += (size_t)row_offset * kHv * kDv;

    constexpr int kThreads = 256;
    constexpr int kStateElements = kHv * kDv * kDk;
    constexpr int kQkTileElements = kHk * kWyChunk * kDk;
    const int chunks = (tokens + kWyChunk - 1) / kWyChunk;
    const int qk_blocks = (chunks * kQkTileElements + kThreads - 1) / kThreads;
    const int decay_blocks = (chunks * kHv + kThreads - 1) / kThreads;
    __nv_bfloat16* selected_state =
        state_pool + (size_t)state_slot * kStateElements;

    // Подготовка не зависит от состояния и считается сразу по всем чанкам —
    // по одному запуску на слой. Когда каждая из этих фаз запускалась на
    // каждый чанк, пусковая задержка стоила больше самой работы.
    wy_prepare_qk_kernel<<<qk_blocks, kThreads, 0, stream>>>(
        q, k, query_tile, key_tile, tokens, chunks);
    wy_decay_kernel<<<decay_blocks, kThreads, 0, stream>>>(
        alpha, log_decay, tokens, chunks);
    // K K^T и Q K^T общие для трёх v-голов одной k-головы.
    launch_wy_gemm(
        key_tile, key_tile, gram_kk, nullptr, nullptr, nullptr,
        kWyChunk, kWyChunk, kDk, kHk * chunks,
        kWyChunk * kDk, kWyChunk * kDk,
        kWyChunk * kWyChunk, 0,
        kDk, kDk, kWyChunk, 0,
        1, 1, 0, 0, 0, 0.0f, stream);
    launch_wy_gemm(
        query_tile, key_tile, gram_qk, nullptr, nullptr, nullptr,
        kWyChunk, kWyChunk, kDk, kHk * chunks,
        kWyChunk * kDk, kWyChunk * kDk,
        kWyChunk * kWyChunk, 0,
        kDk, kDk, kWyChunk, 0,
        1, 1, 0, 0, 0, 0.0f, stream);
    wy_factor_kernel<<<dim3(kHv, chunks), kWyChunk, 0, stream>>>(
        beta, log_decay, gram_kk, gram_qk, inverse, output_factor, tokens);
    wy_prepare_value_kernel<<<dim3(kHv, chunks), 256, 0, stream>>>(
        v, beta, value_tile, tokens);

    // Цепочка чанков разорвана быть не может: состояние конца чанка нужно
    // следующему. Слитое ядро проходит её целиком, держа состояние в
    // регистрах варпа.
    static const cudaError_t opted = cudaFuncSetAttribute(
        wy_fused_scan_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(wy_fused_shared_bytes()));
    if (opted != cudaSuccess) {
        return opted;
    }
    wy_fused_scan_kernel<<<dim3(kHv, kFusedSlices), kFusedThreads,
        wy_fused_shared_bytes(), stream>>>(
        key_tile, query_tile, inverse, output_factor, value_tile,
        log_decay, beta, selected_state, out, tokens, chunks);
    return cudaGetLastError();
}

// ---------------------------------------------------------------------------
// Перекладка состояния на 8-битную сетку
// ---------------------------------------------------------------------------
//
// Состояние остаётся в bf16: кернел только сажает значения на ту сетку, на
// которой они оказались бы при 8-битном хранении, и возвращает обратно.
// Биты теряются ровно те же, поэтому diff-eval даёт настоящий ответ по
// точности, пока раскладка, пулы и бюджет ещё не тронуты.
//
// Масштаб — на строку состояния (128 элементов по dk): именно вдоль строки
// идут оба скалярных произведения S k и S q, так что вся сумма делит один
// масштаб. Строки плоские (пик близок к RMS), поэтому int8 с масштабом по
// максимуму на строке бьёт e4m3 вчетверо при том же байте: e4m3 тратит
// четыре бита на экспоненту, которая строке не нужна.
//
// Обратный ход пишет bf16, то есть добавляет собственное округление bf16
// поверх 8-битного. Настоящее хранение деквантует в fp32-регистры и этой
// добавки не имеет, так что замер через этот кернел — оценка сверху.

namespace {

constexpr int kRequantBlock = 256;

// 1 — int8 с построчным масштабом, 2 — e4m3 с построчным масштабом.
template <int kMode>
__global__ __launch_bounds__(kRequantBlock) void delta_state_requantize_kernel(
    __nv_bfloat16* __restrict__ state,
    const uint32_t* __restrict__ state_slots,
    int first_slot)
{
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    const int slot = state_slots == nullptr ? first_slot + b : (int)state_slots[b];

    const int rows_per_block = kDv / gridDim.z;
    const int row_begin = blockIdx.z * rows_per_block;
    const int row_end = row_begin + rows_per_block;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    constexpr int kWarps = kRequantBlock / 32;

    __nv_bfloat16* S = state + (size_t)(slot * kHv + h) * kDv * kDk;

    for (int row = row_begin + warp; row < row_end; row += kWarps) {
        float2* srow = reinterpret_cast<float2*>(S + (size_t)row * kDk);

        float2 packed = srow[lane];
        __nv_bfloat162 p0 = *reinterpret_cast<__nv_bfloat162*>(&packed.x);
        __nv_bfloat162 p1 = *reinterpret_cast<__nv_bfloat162*>(&packed.y);

        float s[kPerThread];
        s[0] = __bfloat162float(p0.x);
        s[1] = __bfloat162float(p0.y);
        s[2] = __bfloat162float(p1.x);
        s[3] = __bfloat162float(p1.y);

        float peak = 0.0f;
        #pragma unroll
        for (int t = 0; t < kPerThread; ++t) {
            peak = fmaxf(peak, fabsf(s[t]));
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            peak = fmaxf(peak, __shfl_xor_sync(0xffffffff, peak, off));
        }
        if (peak == 0.0f) {
            continue;
        }

        if (kMode == 1) {
            const float step = peak * (1.0f / 127.0f);
            const float inverse = 1.0f / step;
            #pragma unroll
            for (int t = 0; t < kPerThread; ++t) {
                const float level = fminf(fmaxf(rintf(s[t] * inverse), -127.0f), 127.0f);
                s[t] = level * step;
            }
        } else {
            const float scale = peak * (1.0f / 448.0f);
            const float inverse = 1.0f / scale;
            #pragma unroll
            for (int t = 0; t < kPerThread; ++t) {
                const __nv_fp8_e4m3 quantized = __nv_fp8_e4m3(s[t] * inverse);
                s[t] = (float)quantized * scale;
            }
        }

        p0 = __nv_bfloat162(__float2bfloat16(s[0]), __float2bfloat16(s[1]));
        p1 = __nv_bfloat162(__float2bfloat16(s[2]), __float2bfloat16(s[3]));
        packed.x = *reinterpret_cast<float*>(&p0);
        packed.y = *reinterpret_cast<float*>(&p1);
        srow[lane] = packed;
    }
}

}  // namespace

// `state_slots` задаёт слоты поимённо (decode); при nullptr берётся
// `count` слотов подряд с `first_slot` (один сегмент prefill).
extern "C" cudaError_t qwc_delta_state_requantize(
    void* state, const uint32_t* state_slots,
    int first_slot, int state_capacity, int count, int mode,
    cudaStream_t stream)
{
    if (state == nullptr || count <= 0 || state_capacity <= 0 ||
        first_slot < 0 || first_slot + count > state_capacity ||
        (mode != 1 && mode != 2)) {
        return cudaErrorInvalidValue;
    }
    dim3 grid(kHv, count, splits_for(count));
    if (mode == 1) {
        delta_state_requantize_kernel<1><<<grid, kRequantBlock, 0, stream>>>(
            static_cast<__nv_bfloat16*>(state), state_slots, first_slot);
    } else {
        delta_state_requantize_kernel<2><<<grid, kRequantBlock, 0, stream>>>(
            static_cast<__nv_bfloat16*>(state), state_slots, first_slot);
    }
    return cudaGetLastError();
}

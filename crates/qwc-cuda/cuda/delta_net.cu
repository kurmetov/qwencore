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

// The WY path trades the token-by-token state recurrence for four small
// matrix products per tile.  Sixty-four is large enough to amortize launches
// and small enough that all per-head triangular factors stay resident.
constexpr int kWyChunk = 64;

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

__global__ void wy_initialize_state_kernel(
    const __nv_bfloat16* __restrict__ source,
    float* __restrict__ state_fp32,
    __nv_bfloat16* __restrict__ state_bf16) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int elements = kHv * kDv * kDk;
    if (index < elements) {
        const __nv_bfloat16 value = source[index];
        state_bf16[index] = value;
        state_fp32[index] = __bfloat162float(value);
    }
}

__global__ void wy_prepare_qk_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    __nv_bfloat16* __restrict__ query_tile,
    __nv_bfloat16* __restrict__ key_tile,
    __nv_bfloat16* __restrict__ key_transposed,
    int start, int length) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int elements = kHk * kWyChunk * kDk;
    if (index >= elements) {
        return;
    }
    const int dimension = index % kDk;
    const int token = (index / kDk) % kWyChunk;
    const int head = index / (kWyChunk * kDk);
    __nv_bfloat16 qv = __float2bfloat16(0.0f);
    __nv_bfloat16 kv = __float2bfloat16(0.0f);
    if (token < length) {
        const size_t source = (size_t)(start + token) * kHk * kDk
            + (size_t)head * kDk + dimension;
        qv = __float2bfloat16(q[source]);
        kv = __float2bfloat16(k[source]);
    }
    query_tile[index] = qv;
    key_tile[index] = kv;
    key_transposed[(size_t)(head * kDk + dimension) * kWyChunk + token] = kv;
}

__global__ void wy_gamma_kernel(
    const float* __restrict__ alpha,
    float* __restrict__ gamma,
    int start, int length) {
    const int head = blockIdx.x;
    if (threadIdx.x != 0 || head >= kHv) {
        return;
    }
    float product = 1.0f;
    for (int token = 0; token < kWyChunk; ++token) {
        if (token < length) {
            product *= alpha[(size_t)(start + token) * kHv + head];
            gamma[head * kWyChunk + token] = product;
        } else {
            gamma[head * kWyChunk + token] = 0.0f;
        }
    }
}

__global__ void wy_transform_kernel(
    const float* __restrict__ beta,
    const float* __restrict__ gamma,
    const float* __restrict__ gram_kk,
    const float* __restrict__ gram_qk,
    float* __restrict__ triangular,
    __nv_bfloat16* __restrict__ output_factor,
    int start, int length) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int elements = kHv * kWyChunk * kWyChunk;
    if (index >= elements) {
        return;
    }
    const int previous = index % kWyChunk;
    const int token = (index / kWyChunk) % kWyChunk;
    const int head = index / (kWyChunk * kWyChunk);
    float a = 0.0f;
    float f = 0.0f;
    if (token < length && previous <= token) {
        const float ratio = gamma[head * kWyChunk + token]
            / gamma[head * kWyChunk + previous];
        const int key_head = head / kRatio;
        const size_t gram_offset = (size_t)(key_head * kWyChunk + token)
            * kWyChunk + previous;
        f = ratio * gram_qk[gram_offset];
        if (previous < token) {
            a = beta[(size_t)(start + token) * kHv + head]
                * ratio * gram_kk[gram_offset];
        }
    }
    triangular[index] = a;
    output_factor[index] = __float2bfloat16(f);
}

__global__ __launch_bounds__(128) void wy_solve_kernel(
    const float* __restrict__ v,
    const float* __restrict__ beta,
    const float* __restrict__ gamma,
    const float* __restrict__ triangular,
    float* __restrict__ coefficients,
    __nv_bfloat16* __restrict__ coefficients_transposed,
    __nv_bfloat16* __restrict__ weighted_coefficients_transposed,
    int start, int length) {
    const int head = blockIdx.x;
    const int value_dimension = threadIdx.x;
    const size_t matrix_base = (size_t)head * kWyChunk * kDv;
    const size_t triangle_base = (size_t)head * kWyChunk * kWyChunk;
    for (int token = 0; token < length; ++token) {
        const float bt = beta[(size_t)(start + token) * kHv + head];
        const size_t slot = matrix_base + (size_t)token * kDv + value_dimension;
        float c = bt * (
            v[((size_t)(start + token) * kHv + head) * kDv + value_dimension]
            - gamma[head * kWyChunk + token] * coefficients[slot]);
        for (int previous = 0; previous < token; ++previous) {
            c -= triangular[triangle_base + (size_t)token * kWyChunk + previous]
                * coefficients[matrix_base + (size_t)previous * kDv + value_dimension];
        }
        coefficients[slot] = c;
    }
    const float last = gamma[head * kWyChunk + length - 1];
    for (int token = 0; token < kWyChunk; ++token) {
        float c = 0.0f;
        float weighted = 0.0f;
        if (token < length) {
            c = coefficients[matrix_base + (size_t)token * kDv + value_dimension];
            weighted = c * last / gamma[head * kWyChunk + token];
        }
        const size_t transposed = (size_t)(head * kDv + value_dimension)
            * kWyChunk + token;
        coefficients_transposed[transposed] = __float2bfloat16(c);
        weighted_coefficients_transposed[transposed] = __float2bfloat16(weighted);
    }
}

__global__ void wy_refresh_state_kernel(
    const float* __restrict__ state_fp32,
    __nv_bfloat16* __restrict__ state_bf16,
    __nv_bfloat16* __restrict__ final_state) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int elements = kHv * kDv * kDk;
    if (index < elements) {
        const __nv_bfloat16 value = __float2bfloat16(state_fp32[index]);
        state_bf16[index] = value;
        if (final_state != nullptr) {
            final_state[index] = value;
        }
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
    void* state_fp32_pointer,
    void* state_bf16_pointer,
    void* query_tile_pointer,
    void* key_tile_pointer,
    void* key_transposed_pointer,
    void* gram_kk_pointer,
    void* gram_qk_pointer,
    void* triangular_pointer,
    void* output_factor_pointer,
    void* coefficients_pointer,
    void* coefficients_transposed_pointer,
    void* weighted_coefficients_transposed_pointer,
    void* gamma_pointer,
    int state_capacity,
    int state_slot,
    int tokens,
    int row_offset,
    cudaStream_t stream) {
    if (state == nullptr || q == nullptr || k == nullptr || v == nullptr ||
        alpha == nullptr || beta == nullptr || out == nullptr ||
        state_fp32_pointer == nullptr || state_bf16_pointer == nullptr ||
        query_tile_pointer == nullptr || key_tile_pointer == nullptr ||
        key_transposed_pointer == nullptr || gram_kk_pointer == nullptr ||
        gram_qk_pointer == nullptr || triangular_pointer == nullptr ||
        output_factor_pointer == nullptr || coefficients_pointer == nullptr ||
        coefficients_transposed_pointer == nullptr ||
        weighted_coefficients_transposed_pointer == nullptr ||
        gamma_pointer == nullptr || state_capacity <= 0 || state_slot < 0 ||
        state_slot >= state_capacity || tokens <= 0 ||
        tokens > qwc::kMaxStepRows || row_offset < 0) {
        return cudaErrorInvalidValue;
    }

    auto* state_pool = static_cast<__nv_bfloat16*>(state);
    auto* state_fp32 = static_cast<float*>(state_fp32_pointer);
    auto* state_bf16 = static_cast<__nv_bfloat16*>(state_bf16_pointer);
    auto* query_tile = static_cast<__nv_bfloat16*>(query_tile_pointer);
    auto* key_tile = static_cast<__nv_bfloat16*>(key_tile_pointer);
    auto* key_transposed = static_cast<__nv_bfloat16*>(key_transposed_pointer);
    auto* gram_kk = static_cast<float*>(gram_kk_pointer);
    auto* gram_qk = static_cast<float*>(gram_qk_pointer);
    auto* triangular = static_cast<float*>(triangular_pointer);
    auto* output_factor = static_cast<__nv_bfloat16*>(output_factor_pointer);
    auto* coefficients = static_cast<float*>(coefficients_pointer);
    auto* coefficients_transposed =
        static_cast<__nv_bfloat16*>(coefficients_transposed_pointer);
    auto* weighted_coefficients_transposed =
        static_cast<__nv_bfloat16*>(weighted_coefficients_transposed_pointer);
    auto* gamma = static_cast<float*>(gamma_pointer);

    // The Rust side validates the full fused arena.  From here on all row
    // coordinates are relative to the selected sequence segment.
    q += (size_t)row_offset * kHk * kDk;
    k += (size_t)row_offset * kHk * kDk;
    v += (size_t)row_offset * kHv * kDv;
    alpha += (size_t)row_offset * kHv;
    beta += (size_t)row_offset * kHv;
    out += (size_t)row_offset * kHv * kDv;

    constexpr int kThreads = 256;
    constexpr int kStateElements = kHv * kDv * kDk;
    constexpr int kQkTileElements = kHk * kWyChunk * kDk;
    constexpr int kHeadMatrixElements = kHv * kWyChunk * kWyChunk;
    const int state_blocks = (kStateElements + kThreads - 1) / kThreads;
    const int qk_blocks = (kQkTileElements + kThreads - 1) / kThreads;
    const int matrix_blocks = (kHeadMatrixElements + kThreads - 1) / kThreads;
    __nv_bfloat16* selected_state =
        state_pool + (size_t)state_slot * kStateElements;
    wy_initialize_state_kernel<<<state_blocks, kThreads, 0, stream>>>(
        selected_state, state_fp32, state_bf16);

    for (int start = 0; start < tokens; start += kWyChunk) {
        const int length = min(kWyChunk, tokens - start);
        wy_prepare_qk_kernel<<<qk_blocks, kThreads, 0, stream>>>(
            q, k, query_tile, key_tile, key_transposed, start, length);
        wy_gamma_kernel<<<kHv, 1, 0, stream>>>(
            alpha, gamma, start, length);

        // K K^T and Q K^T are shared by the three value heads in a key-head
        // group.  The following scalar transforms add beta and gamma ratios.
        launch_wy_gemm(
            key_tile, key_tile, gram_kk, nullptr, nullptr, nullptr,
            kWyChunk, kWyChunk, kDk, kHk,
            kWyChunk * kDk, kWyChunk * kDk,
            kWyChunk * kWyChunk, 0,
            kDk, kDk, kWyChunk, 0,
            1, 1, 0, 0, 0, 0.0f, stream);
        launch_wy_gemm(
            query_tile, key_tile, gram_qk, nullptr, nullptr, nullptr,
            kWyChunk, kWyChunk, kDk, kHk,
            kWyChunk * kDk, kWyChunk * kDk,
            kWyChunk * kWyChunk, 0,
            kDk, kDk, kWyChunk, 0,
            1, 1, 0, 0, 0, 0.0f, stream);
        wy_transform_kernel<<<matrix_blocks, kThreads, 0, stream>>>(
            beta, gamma, gram_kk, gram_qk, triangular, output_factor,
            start, length);

        // P = K S_0^T.  `wy_solve_kernel` turns P into the right-hand side
        // and performs the forward substitution independently for 128 values.
        launch_wy_gemm(
            key_tile, state_bf16, coefficients, nullptr, nullptr, nullptr,
            kWyChunk, kDv, kDk, kHv,
            kWyChunk * kDk, kDv * kDk,
            kWyChunk * kDv, 0,
            kDk, kDk, kDv, 0,
            kRatio, 1, 0, 0, 0, 0.0f, stream);
        wy_solve_kernel<<<kHv, kDv, 0, stream>>>(
            v, beta, gamma, triangular, coefficients,
            coefficients_transposed, weighted_coefficients_transposed,
            start, length);

        // O_0 = diag(gamma) Q S_0^T, directly into the fused row-major
        // output arena.  Then add F C using the causal factor matrix.
        launch_wy_gemm(
            query_tile, state_bf16, out + (size_t)start * kHv * kDv,
            nullptr, gamma, nullptr,
            kWyChunk, kDv, kDk, kHv,
            kWyChunk * kDk, kDv * kDk,
            kDv, 0,
            kDk, kDk, kHv * kDv, 0,
            kRatio, 1, kWyChunk, 0, 0, 0.0f, stream);
        launch_wy_gemm(
            output_factor, coefficients_transposed,
            out + (size_t)start * kHv * kDv,
            out + (size_t)start * kHv * kDv,
            nullptr, nullptr,
            kWyChunk, kDv, kWyChunk, kHv,
            kWyChunk * kWyChunk, kDv * kWyChunk,
            kDv, kDv,
            kWyChunk, kWyChunk, kHv * kDv, kHv * kDv,
            1, 1, 0, 0, 0, 1.0f, stream);

        // S_C = gamma_C S_0 + C_weighted^T K.  Keep the accumulator in
        // FP32 across tiles, but materialize BF16 once for the next MMA.
        launch_wy_gemm(
            weighted_coefficients_transposed, key_transposed,
            state_fp32, state_fp32, nullptr, gamma,
            kDv, kDk, kWyChunk, kHv,
            kDv * kWyChunk, kDk * kWyChunk,
            kDv * kDk, kDv * kDk,
            kWyChunk, kWyChunk, kDk, kDk,
            1, kRatio, 0, kWyChunk, length - 1, 1.0f, stream);
        const bool final_tile = start + length == tokens;
        wy_refresh_state_kernel<<<state_blocks, kThreads, 0, stream>>>(
            state_fp32, state_bf16, final_tile ? selected_state : nullptr);
    }
    return cudaGetLastError();
}

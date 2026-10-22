// Gated DeltaNet: recurrent decode step and chunked prefill scan.
//
// 48 of Qwen3.8's 64 layers are linear attention with a recurrent state
// S of size [48 heads, 128, 128]. For every token the state is read and
// rewritten in full: at batch=32 that's 21% of all decode-step traffic,
// twice the KV cache.
//
// Recurrence:
//     S_t = a * S (I - b k k^T) + b v k^T = a*S + b(v - a*(S k)) k^T
//     o_t = S_t q = a*(S q) + b(v - a*(S k)) * (k . q)
//
// Hence the key property of the implementation: S k and S q are computed in ONE pass,
// after which the state row stays in registers and is rewritten in place.
// One read and one write instead of three passes over memory.
//
// Block layout: a warp owns a state row. 32 threads x 4 elements =
// exactly 128, 8-byte vector access gives fully coalesced 256-byte
// transactions per warp.
//
// State rows are additionally split across blocks via gridDim.z.
// Without this, batch=1 launches only 48 blocks (one per v head)
// on 170 SMs, and three quarters of the chip sit idle: measured 24% of
// bandwidth instead of 95%. The split is independent because state rows
// are updated independently of each other — only the scalar k.q links them.

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
constexpr int kRatio = kHv / kHk;  // k/q are broadcast onto v heads 1:3
constexpr int kPerThread = kDk / 32;
constexpr int kBlock = 256;     // 8 warps -> 8 state rows per iteration

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, off);
    }
    return v;
}

// Two reductions in one butterfly pass: as many steps as for one,
// and the result stays in all lanes — no broadcast from lane zero needed.
//
// The third one here used to be k.q. There's one per head and it doesn't depend on the state
// row, so prepare computes it, rather than the 128 rows of each of three
// v heads once per token.
__device__ __forceinline__ void warp_sum2(float& a, float& b) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        a += __shfl_xor_sync(0xffffffff, a, off);
        b += __shfl_xor_sync(0xffffffff, b, off);
    }
}

// The same butterfly over N independent sums: as many steps as for one, but
// N chains, and shfl latency is hidden by them rather than by idling.
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

// State rows per warp in prefill. One row meant that all eight
// warps of a block load the same k and q row (the address depends only on the
// lane), and so do all 16 blocks along z: k and q were read 128 times.
// The second problem lies here too — a butterfly over two values was bound by
// shfl latency. Both are cured by one thing: a warp handles kRows rows at once.
constexpr int kPrefillRows = 4;
// Prefill takes its own block size: 128 rows per head, and with kPrefillRows they
// suffice either for many small blocks or for few large ones. Small ones spread
// more evenly over 170 SMs.
constexpr int kPrefillBlock = 64;

// Chunk of the matrix (WY) form: it replaces the pass over tokens with five matrix
// multiplies per tile. 64 is a compromise: smaller, and preparation (whose per-token
// cost doesn't depend on chunk length) stops paying off; larger, and the
// 64x64 per-head matrix with its inverse stops fitting in a block's shared memory.
constexpr int kWyChunk = 64;

// Recurrence step for one state row. Pulled out of the kernels because
// decode reads the row from two different storages (bf16 and int8), and
// this math must not diverge between them.
//
// Returns the row's contribution to the output; the state `s` is updated in place.
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
// Eight-bit state storage
// ---------------------------------------------------------------------------
//
// A state row (128 elements along dk) is stored as int8 plus one
// f32 scale per row. The scale is per row specifically: both dot products
// S k and S q run along a row, so the whole sum shares one scale.
//
// int8, not e4m3: rows are flat (the peak is close to RMS), and they don't need four
// exponent bits — int8 gives them to the mantissa and is four times more accurate at the same
// byte (`bench/results/delta-state-8bit.md`).
//
// A warp owns a row, so each thread gets exactly kPerThread = 4
// bytes: 32 lanes give a coalesced 128-byte transaction, like the bf16 path with
// its 256 bytes.

__device__ __forceinline__ void state_load_int8(
    const int8_t* __restrict__ row, float scale, int lane, float (&s)[kPerThread]) {
    const char4 packed = *reinterpret_cast<const char4*>(row + lane * kPerThread);
    s[0] = (float)packed.x * scale;
    s[1] = (float)packed.y * scale;
    s[2] = (float)packed.z * scale;
    s[3] = (float)packed.w * scale;
}

// Writes a row and returns its scale. The scale is computed from the row max,
// so a warp reduction is needed before the write — it's also the only cost of
// 8-bit storage beyond the packing itself.
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
    // Block scales go through shared rather than global memory
    // row by row. A four-byte write from one lane is a partial
    // transaction per state row, and it cost 2.97 ms of the 3.3 ms
    // 8-bit storage overhead at c=32: more than all the quantization and all
    // the traffic of the state itself combined.
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

    // Scales are read with one coalesced access per block, in the same
    // load as k and q.
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

// Moving a slot between storages. A sequence's prefill runs on a
// dequantized bf16 copy and is quantized once at the end of the prompt: if
// its segments were restarted from the 8-bit state, the whole rest of the prompt
// would be computed from a spoiled S_0, and MAE jumps 11x at the boundary
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
    const float* __restrict__ q,         // [B, kHk, kDk], L2-normalized
    const float* __restrict__ k,         // [B, kHk, kDk], L2-normalized
    const float* __restrict__ v,         // [B, kHv, kDv]
    const float* __restrict__ alpha,     // [B, kHv]
    const float* __restrict__ beta,      // [B, kHv]
    const float* __restrict__ kq,        // [B, kHk], scalar k.q from prepare
    float* __restrict__ out)             // [B, kHv, kDv]
{
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    const int state_slot = state_slots == nullptr ? b : state_slots[b];
    const int hk = h / kRatio;

    // The range of rows this block is responsible for.
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
    // The scalar k . q is shared by all rows and both branches: prepare computes it,
    // so decode and a repeated prefill read exactly the same number.
    const float kq_token = kq[b * kHk + hk];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    constexpr int kWarps = kBlock / 32;

    __nv_bfloat16* S = state + (size_t)(state_slot * kHv + h) * kDv * kDk;
    const float* vp = v + (size_t)(b * kHv + h) * kDv;
    float* op = out + (size_t)(b * kHv + h) * kDv;

    for (int row = row_begin + warp; row < row_end; row += kWarps) {
        float2* srow = reinterpret_cast<float2*>(S + (size_t)row * kDk);

        // One 8-byte read per thread: 4 bf16, coalesced 256 bytes per warp.
        float2 packed = srow[lane];
        __nv_bfloat162 p0 = *reinterpret_cast<__nv_bfloat162*>(&packed.x);
        __nv_bfloat162 p1 = *reinterpret_cast<__nv_bfloat162*>(&packed.y);

        float s[kPerThread];
        s[0] = __bfloat162float(p0.x);
        s[1] = __bfloat162float(p0.y);
        s[2] = __bfloat162float(p1.x);
        s[3] = __bfloat162float(p1.y);

        // Both matvecs in one pass while the row is in registers, and the update
        // in place: no re-read needed.
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
    // A warp owns a state row and walks the tokens by itself. The old layout
    // put three __syncthreads() per token and computed k.q with one warp out of
    // eight while the others waited at the barrier: the recurrence is sequential
    // anyway, and the barriers added block synchronization on top of it.
    // State rows are independent — only the scalar k.q links them, which
    // the warp now computes for itself, in the same butterfly as u and w.
    constexpr int kWarps = kPrefillBlock / 32;
    constexpr int kRows = kPrefillRows;
    static_assert(kDv % (kWarps * kRows) == 0, "state rows are split among warps");

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

    // A token's input doesn't depend on the state, so the next token's load
    // is issued before the current one is computed: the recurrence is sequential,
    // but its waiting on memory isn't.
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

        // The first kRows sums are each row's u, the second its w. One butterfly for
        // all: k and q are read once per kRows rows, not per one.
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

// The q and k tiles are laid out as [chunk][head][token][dim]: the batch index in
// `wy_gemm_kernel` is chunk * heads + head, so handing one
// k head to three v heads (`b_group`) remains an index division.
//
// No phase before the fused kernel depends on the state, so all of them
// are computed for all chunks of the segment at once — one launch per layer.
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

// Decay is accumulated in logarithms, not as a product.
//
// The product of alpha over a chunk is exactly the gamma_t that multiplies
// the input state's contribution. Over 64 tokens it easily drops below 1e-38: at
// alpha = 0.2 already by the end of the chunk. Then the ratios gamma_t / gamma_r
// turn into 0/0, and the scan outputs NaN. A difference of logarithms is always finite,
// and exp of it is either an honest zero or a number.
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
            // The lower bound keeps the logarithm finite: alpha = 0 would give
            // -inf on both sides of the difference and NaN instead of zero.
            sum += __logf(fmaxf(alpha[(size_t)(start + token) * kHv + head],
                                1.0e-38f));
            destination[token] = sum;
        } else {
            destination[token] = -INFINITY;
        }
    }
}

// Causal-part multipliers and the inverse of the unit lower-triangular A = I + M.
//
// Both are built from the same grams and gamma, so they're computed in one
// block: M doesn't leave shared and never reaches global memory.
// Forward substitution goes by columns, and columns are independent — a thread owns
// a column and reads only it, so one barrier is needed, after M is assembled.
// Substitution used to run in value space (128 dims per token) and
// re-read coefficients from global memory: 50 MB per layer chunk and two
// thirds of the whole scan's time.
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

    // Rows beyond the chunk length stay unit: the right-hand side is zeroed there,
    // so zeros propagate to the end on their own, without masks in later phases.
    //
    // Substitution goes in 16-row panels. The naive form read shared
    // twice per FMA — both the multiplier and the already computed value — and
    // was bound by that, not by arithmetic. In the panel form the value of
    // the previous row is read once per sixteen FMAs, and within the
    // panel it lives in registers entirely: half the shared accesses.
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

// The system's right-hand side, already transposed: [head][dim][token].
// P on input is S_0 K^T in the same layout, so not a single transposition
// is needed between phases.
// ------------------------------------------------------------------ //
// Fused sequential scan: one launch per layer instead of eight per chunk.
//
// The sequence of chunks can't be broken — a chunk's final state is needed by
// the next one. But everything between them fits in one warp: a warp owns
// 16 state rows (dv) over the full dk depth and does five
// matrix multiplies over them in a row, never spilling the state to memory.
// Before, the same five multiplies were five launches, and the state went to
// global memory and back on every chunk.
//
// The m16n8k16 fragment layout is the same for three roles of the state:
// accumulator (row dv, column dk), operand A in P = S K^T and operand B in
// Q S^T. In all three a lane holds rows grp and grp+8 and a pair of columns at
// pos*2 — so the packed BF16 copy of the accumulator works as is and
// no reshuffle is needed.
constexpr int kFusedWarps = 4;
constexpr int kFusedThreads = kFusedWarps * 32;
constexpr int kFusedRows = 16;                       // dv rows per warp
constexpr int kFusedSlice = kFusedWarps * kFusedRows;
constexpr int kFusedSlices = kDv / kFusedSlice;
constexpr int kDkTiles = kDk / 8;                    // state n-tiles
constexpr int kChunkTiles = kWyChunk / 8;            // n-tiles along tokens
constexpr int kKeyPad = kDk + 8;

struct WyFusedShared {
    __nv_bfloat16 key[kWyChunk * kKeyPad];
    float gamma[kWyChunk];
    float weight[kWyChunk];   // gamma_C / gamma_t — chunk boundary weight
    float decay[kWyChunk];    // beta_t * gamma_t — input state contribution
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

// Copies rows into shared 8 elements (16 bytes) at a time. Padding is chosen so
// that the row stride is a multiple of 16 bytes, otherwise the vector store falls apart.
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

    // State: rows row0+grp and row0+grp+8, columns tile*8+pos*2 and +1.
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
                // The chunk boundary weight is a ratio of decays, i.e. exp of
                // a difference of logarithms: it's at most one for any alpha.
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

        // BF16 copy of the state: a (row, row+8) pair for each column
        // tile. It's also operand A for S K^T and operand B for Q S^T.
        uint32_t state_pack[kDkTiles][2];
        #pragma unroll
        for (int tile = 0; tile < kDkTiles; ++tile) {
            state_pack[tile][0] = wy_pack(state_acc[tile][0], state_acc[tile][1]);
            state_pack[tile][1] = wy_pack(state_acc[tile][2], state_acc[tile][3]);
        }

        // P = S_0 K^T: [16 dv rows] x [64 tokens].
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

        // Right-hand side: b_t v_t - b_t gamma_t S_0 k_t, straight into BF16 fragments
        // of operand A for multiplying by T^T.
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

        // O = diag(gamma) Q S_0^T + F C. Rows here are tokens, columns are
        // the warp's 16 dv rows, so there are four m-tiles and two n-tiles.
        constexpr int kOutRowTiles = kWyChunk / 16;
        float output[kOutRowTiles][2][4] = {};
        #pragma unroll
        for (int group = 0; group < kDk / 16; ++group) {
            #pragma unroll
            for (int mt = 0; mt < kOutRowTiles; ++mt) {
                // Q is read as fragments straight from global memory: all
                // warps of a CTA take the same elements, L1 holds them, and
                // the 17 KB of shared freed up is worth more than these accesses.
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
        // C as operand B: n is a dv row, k is a token.
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

    // The sequence state outlives the step, so it's written back
    // once, at the end.
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

// Value tile in BF16, transposed: [chunk][head][dim][token].
// The fused kernel reads it in accumulator layout, where a row is a dim and
// a column is a token; there would be nowhere to transpose there.
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

// The number of blocks per head is chosen to occupy all SMs.
// Constraints: a divisor of kDv, and each block must get at least as many rows
// as it has warps, otherwise some warps sit idle.
static int splits_for(int batch) {
    constexpr int kWarps = kBlock / 32;
    constexpr int kMaxSplit = kDv / kWarps;   // 128 / 8 = 16

    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    const int base = kHv * batch;
    if (base >= sms * 2) {
        return 1;   // already enough blocks
    }
    int want = (sms * 2 + base - 1) / base;
    int split = 1;
    while (split * 2 <= kMaxSplit && split < want) {
        split *= 2;   // keep it a divisor of kDv
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

// One bf16 slot -> one int8 slot and back. Both ends are addressed by slots,
// because the engine has one bf16 copy, and as many 8-bit slots as there are
// sequences.
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
    // kPrefillRows rows per warp, kPrefillBlock/32 warps per block: 48 heads x
    // 16 blocks = 768 blocks on 170 SMs. splits_for isn't needed here — prefill
    // always runs one sequence, and there's nothing to split except rows.
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

    // From here on all row coordinates are within the chosen arena segment.
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

    // Preparation doesn't depend on the state and is computed for all chunks at once —
    // one launch per layer. When each of these phases was launched per
    // chunk, the launch latency cost more than the work itself.
    wy_prepare_qk_kernel<<<qk_blocks, kThreads, 0, stream>>>(
        q, k, query_tile, key_tile, tokens, chunks);
    wy_decay_kernel<<<decay_blocks, kThreads, 0, stream>>>(
        alpha, log_decay, tokens, chunks);
    // K K^T and Q K^T are shared by the three v heads of one k head.
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

    // The chain of chunks can't be broken: a chunk's final state is needed by
    // the next one. The fused kernel walks it whole, keeping the state in
    // warp registers.
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
// Snapping the state onto an 8-bit grid
// ---------------------------------------------------------------------------
//
// The state stays in bf16: the kernel only snaps values onto the grid
// they would land on with 8-bit storage, and writes them back.
// Exactly the same bits are lost, so diff-eval gives the real accuracy answer
// while the layout, pools and budget are still untouched.
//
// The scale is per state row (128 elements along dk): it's along a row that
// both dot products S k and S q run, so the whole sum shares one
// scale. Rows are flat (the peak is close to RMS), so int8 with a per-row max
// scale beats e4m3 fourfold at the same byte: e4m3 spends
// four bits on an exponent the row doesn't need.
//
// The reverse pass writes bf16, i.e. adds its own bf16 rounding
// on top of the 8-bit one. Real storage dequantizes into fp32 registers and doesn't
// have this extra, so a measurement through this kernel is an upper bound.

namespace {

constexpr int kRequantBlock = 256;

// 1 — int8 with per-row scale, 2 — e4m3 with per-row scale.
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

// `state_slots` gives slots by name (decode); with nullptr,
// `count` consecutive slots starting at `first_slot` are taken (one prefill segment).
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

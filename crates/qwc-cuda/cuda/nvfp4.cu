// Decode-проекция NVFP4 x BF16 для batch 1..4.
//
// Чекпоинт хранит W построчно: два E2M1 в байте и одну E4M3-шкалу
// на 16 весов. В отличие от W4A4 GEMM, этот путь не квантует активации:
// для малого decode-batch это убирает отдельный kernel, сохраняет качество
// и почти не меняет DRAM-трафик, потому что доминируют веса.
//
// Блок считает 16 выходных строк. BF16-тайл X[B, 256] загружается в shared
// один раз, а не по разу на строку; каждый 128-битный фрагмент W читается
// ровно один раз и используется для всех B последовательностей.

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kMaxBatch = 4;

__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t raw) {
    __nv_fp8_e4m3 value;
    value.__x = raw;
    return static_cast<float>(value);
}

__device__ __forceinline__ size_t block_scale_offset(
    int row, int group, int group_blocks) {
    const size_t tile =
        static_cast<size_t>(row >> 7) * group_blocks + (group >> 2);
    const int row_bits = ((row & 31) << 4) | (((row >> 5) & 3) << 2);
    return (tile << 9) | row_bits | (group & 3);
}

template <int kRows, int kKThreads, int kKTile, int kBatch>
__global__ __launch_bounds__(128, 8)
void nvfp4_w4a16_kernel(
    const uint8_t* __restrict__ packed,       // [out, in / 2]
    const uint8_t* __restrict__ scales,       // CUTLASS 128x4 block-scale layout
    const __nv_bfloat16* __restrict__ input,  // [batch, in]
    __nv_bfloat16* __restrict__ output,       // [batch, out]
    int out_features,
    int in_features,
    float inverse_weight_global_scale)
{
    static_assert(kRows * kKThreads == 128);
    static_assert(kKTile % kKThreads == 0);
    constexpr int kElemsPerThread = kKTile / kKThreads;
    constexpr int kBytesPerThread = kElemsPerThread / 2;
    constexpr int kGroupsPerThread = (kElemsPerThread + 15) / 16;
    constexpr int kPairsPerGroup = (kElemsPerThread < 16 ? kElemsPerThread : 16) / 2;
    static_assert(kBytesPerThread == 4 || kBytesPerThread == 8 || kBytesPerThread == 16);

    __shared__ __align__(16) __nv_bfloat16 x_tile[kMaxBatch][kKTile];

    const int k_lane = threadIdx.x;  // 0..7
    const int row_lane = threadIdx.y; // 0..15
    const int linear_thread = row_lane * kKThreads + k_lane;
    const int row = blockIdx.x * kRows + row_lane;
    const bool valid_row = row < out_features;

    float accum[kBatch] = {};
    const int packed_stride = in_features / 2;
    const int scale_group_blocks = in_features / 64;

    for (int k0 = 0; k0 < in_features; k0 += kKTile) {
        // batch * 256 bf16 = batch * 32 vectorных 128-битных загрузок.
        const int vectors = kBatch * (kKTile * int(sizeof(__nv_bfloat16)) / int(sizeof(uint4)));
        for (int v = linear_thread; v < vectors; v += kRows * kKThreads) {
            const int vectors_per_batch = kKTile * int(sizeof(__nv_bfloat16)) / int(sizeof(uint4));
            const int b = v / vectors_per_batch;
            const int vi = v - b * vectors_per_batch;
            reinterpret_cast<uint4*>(x_tile[b])[vi] =
                reinterpret_cast<const uint4*>(input + static_cast<size_t>(b) * in_features + k0)[vi];
        }
        __syncthreads();

        if (valid_row) {
            const uint8_t* weight_row = packed + static_cast<size_t>(row) * packed_stride;

            // Слитая 64- или 128-битная загрузка: вся строка блока даёт 128 B.
            uint32_t words[4] = {0, 0, 0, 0};
            if constexpr (kBytesPerThread == 16) {
                const uint4 fragment =
                    reinterpret_cast<const uint4*>(weight_row + k0 / 2)[k_lane];
                words[0] = fragment.x;
                words[1] = fragment.y;
                words[2] = fragment.z;
                words[3] = fragment.w;
            } else if constexpr (kBytesPerThread == 8) {
                const uint2 fragment =
                    reinterpret_cast<const uint2*>(weight_row + k0 / 2)[k_lane];
                words[0] = fragment.x;
                words[1] = fragment.y;
            } else {
                words[0] = reinterpret_cast<const uint32_t*>(weight_row + k0 / 2)[k_lane];
            }

            #pragma unroll
            for (int group = 0; group < kGroupsPerThread; ++group) {
                int scale_group;
                if constexpr (kElemsPerThread >= 16) {
                    scale_group = k_lane * kGroupsPerThread + group;
                } else {
                    // При 8 элементах два соседних потока делят scale-блок из 16.
                    scale_group = k_lane / (16 / kElemsPerThread);
                }
                const int global_group = k0 / 16 + scale_group;
                const uint8_t scale_raw =
                    scales[block_scale_offset(row, global_group, scale_group_blocks)];
                const float scale = fp8_e4m3_to_float(scale_raw) * inverse_weight_global_scale;
                float partial[kBatch] = {};

                #pragma unroll
                for (int pair = 0; pair < kPairsPerGroup; ++pair) {
                    const int byte_index = group * kPairsPerGroup + pair;
                    const uint8_t fp4x2 = static_cast<uint8_t>(
                        words[byte_index >> 2] >> ((byte_index & 3) * 8));
                    const __half2_raw half_raw =
                        __nv_cvt_fp4x2_to_halfraw2(fp4x2, __NV_E2M1);
                    const __half2 weight = *reinterpret_cast<const __half2*>(&half_raw);
                    const float2 wf = __half22float2(weight);
                    const int x_index = k_lane * kElemsPerThread + group * 16 + pair * 2;

                    #pragma unroll
                    for (int b = 0; b < kBatch; ++b) {
                        const __nv_bfloat162 xv =
                            *reinterpret_cast<const __nv_bfloat162*>(&x_tile[b][x_index]);
                        const float2 xf = __bfloat1622float2(xv);
                        partial[b] = fmaf(wf.x, xf.x, partial[b]);
                        partial[b] = fmaf(wf.y, xf.y, partial[b]);
                    }
                }

                #pragma unroll
                for (int b = 0; b < kBatch; ++b) {
                    accum[b] = fmaf(partial[b], scale, accum[b]);
                }
            }
        }
        __syncthreads();
    }

    // В каждом варпе лежат четыре независимые группы по 8 k-потоков.
    #pragma unroll
    for (int offset = kKThreads / 2; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int b = 0; b < kBatch; ++b) {
            accum[b] += __shfl_down_sync(0xffffffff, accum[b], offset, kKThreads);
        }
    }

    if (k_lane == 0 && valid_row) {
        #pragma unroll
        for (int b = 0; b < kBatch; ++b) {
            output[static_cast<size_t>(b) * out_features + row] =
                __float2bfloat16(accum[b]);
        }
    }
}

// MLP decode: gate_proj + up_proj + silu(gate) * up в одном kernel.
// Обе матрицы читаются один раз, а activation tile загружается в shared один раз
// на пару проекций. Выход сразу имеет форму входа down_proj.
template <int kBatch>
__global__ __launch_bounds__(128, 8)
void nvfp4_swiglu_w4a16_kernel(
    const uint8_t* __restrict__ gate_packed,
    const uint8_t* __restrict__ gate_scales,
    const uint8_t* __restrict__ up_packed,
    const uint8_t* __restrict__ up_scales,
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int out_features,
    int in_features,
    float inverse_gate_global_scale,
    float inverse_up_global_scale)
{
    constexpr int kRows = 16;
    constexpr int kKThreads = 8;
    constexpr int kKTile = 256;
    constexpr int kElemsPerThread = 32;

    __shared__ __align__(16) __nv_bfloat16 x_tile[kMaxBatch][kKTile];

    const int k_lane = threadIdx.x;
    const int row_lane = threadIdx.y;
    const int linear_thread = row_lane * kKThreads + k_lane;
    const int row = blockIdx.x * kRows + row_lane;
    const bool valid_row = row < out_features;
    const int packed_stride = in_features / 2;
    const int scale_group_blocks = in_features / 64;
    float gate_accum[kBatch] = {};
    float up_accum[kBatch] = {};

    for (int k0 = 0; k0 < in_features; k0 += kKTile) {
        constexpr int kVectorsPerBatch = kKTile * int(sizeof(__nv_bfloat16)) / int(sizeof(uint4));
        for (int v = linear_thread; v < kBatch * kVectorsPerBatch; v += 128) {
            const int b = v / kVectorsPerBatch;
            const int vi = v - b * kVectorsPerBatch;
            reinterpret_cast<uint4*>(x_tile[b])[vi] =
                reinterpret_cast<const uint4*>(input + static_cast<size_t>(b) * in_features + k0)[vi];
        }
        __syncthreads();

        if (valid_row) {
            const uint8_t* gate_weight_row =
                gate_packed + static_cast<size_t>(row) * packed_stride;
            const uint8_t* up_weight_row =
                up_packed + static_cast<size_t>(row) * packed_stride;
            const uint4 gate_fragment =
                reinterpret_cast<const uint4*>(gate_weight_row + k0 / 2)[k_lane];
            const uint4 up_fragment =
                reinterpret_cast<const uint4*>(up_weight_row + k0 / 2)[k_lane];
            const uint32_t gate_words[4] = {
                gate_fragment.x, gate_fragment.y, gate_fragment.z, gate_fragment.w};
            const uint32_t up_words[4] = {
                up_fragment.x, up_fragment.y, up_fragment.z, up_fragment.w};

            #pragma unroll
            for (int group = 0; group < 2; ++group) {
                const int scale_group = k0 / 16 + k_lane * 2 + group;
                const size_t scale_index =
                    block_scale_offset(row, scale_group, scale_group_blocks);
                const float gate_scale = fp8_e4m3_to_float(gate_scales[scale_index]) *
                    inverse_gate_global_scale;
                const float up_scale = fp8_e4m3_to_float(up_scales[scale_index]) *
                    inverse_up_global_scale;
                float gate_partial[kBatch] = {};
                float up_partial[kBatch] = {};

                #pragma unroll
                for (int pair = 0; pair < 8; ++pair) {
                    const int byte_index = group * 8 + pair;
                    const int word = byte_index >> 2;
                    const int shift = (byte_index & 3) * 8;
                    const uint8_t gate_fp4x2 = static_cast<uint8_t>(gate_words[word] >> shift);
                    const uint8_t up_fp4x2 = static_cast<uint8_t>(up_words[word] >> shift);
                    const __half2_raw gate_raw =
                        __nv_cvt_fp4x2_to_halfraw2(gate_fp4x2, __NV_E2M1);
                    const __half2_raw up_raw =
                        __nv_cvt_fp4x2_to_halfraw2(up_fp4x2, __NV_E2M1);
                    const float2 gate_weight =
                        __half22float2(*reinterpret_cast<const __half2*>(&gate_raw));
                    const float2 up_weight =
                        __half22float2(*reinterpret_cast<const __half2*>(&up_raw));
                    const int x_index = k_lane * kElemsPerThread + group * 16 + pair * 2;

                    #pragma unroll
                    for (int b = 0; b < kBatch; ++b) {
                        const __nv_bfloat162 xv =
                            *reinterpret_cast<const __nv_bfloat162*>(&x_tile[b][x_index]);
                        const float2 xf = __bfloat1622float2(xv);
                        gate_partial[b] = fmaf(gate_weight.x, xf.x, gate_partial[b]);
                        gate_partial[b] = fmaf(gate_weight.y, xf.y, gate_partial[b]);
                        up_partial[b] = fmaf(up_weight.x, xf.x, up_partial[b]);
                        up_partial[b] = fmaf(up_weight.y, xf.y, up_partial[b]);
                    }
                }

                #pragma unroll
                for (int b = 0; b < kBatch; ++b) {
                    gate_accum[b] = fmaf(gate_partial[b], gate_scale, gate_accum[b]);
                    up_accum[b] = fmaf(up_partial[b], up_scale, up_accum[b]);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int b = 0; b < kBatch; ++b) {
            gate_accum[b] += __shfl_down_sync(0xffffffff, gate_accum[b], offset, kKThreads);
            up_accum[b] += __shfl_down_sync(0xffffffff, up_accum[b], offset, kKThreads);
        }
    }

    if (k_lane == 0 && valid_row) {
        #pragma unroll
        for (int b = 0; b < kBatch; ++b) {
            const float gate = gate_accum[b];
            const float silu = gate / (1.0f + __expf(-gate));
            output[static_cast<size_t>(b) * out_features + row] =
                __float2bfloat16(silu * up_accum[b]);
        }
    }
}

template <int kRows, int kKThreads, int kKTile, int kBatch>
cudaError_t launch_w4a16(
    const void* packed,
    const void* scales,
    const void* input,
    void* output,
    int out_features,
    int in_features,
    float inverse_weight_global_scale,
    cudaStream_t stream)
{
    const dim3 block(kKThreads, kRows);
    const dim3 grid((out_features + kRows - 1) / kRows);
    nvfp4_w4a16_kernel<kRows, kKThreads, kKTile, kBatch><<<grid, block, 0, stream>>>(
        static_cast<const uint8_t*>(packed),
        static_cast<const uint8_t*>(scales),
        static_cast<const __nv_bfloat16*>(input),
        static_cast<__nv_bfloat16*>(output),
        out_features,
        in_features,
        inverse_weight_global_scale);
    return cudaGetLastError();
}

template <int kRows, int kKThreads, int kKTile>
cudaError_t launch_batch(
    const void* packed,
    const void* scales,
    const void* input,
    void* output,
    int out_features,
    int in_features,
    int batch,
    float inverse_weight_global_scale,
    cudaStream_t stream)
{
    switch (batch) {
        case 1:
            return launch_w4a16<kRows, kKThreads, kKTile, 1>(packed, scales, input, output,
                out_features, in_features, inverse_weight_global_scale, stream);
        case 2:
            return launch_w4a16<kRows, kKThreads, kKTile, 2>(packed, scales, input, output,
                out_features, in_features, inverse_weight_global_scale, stream);
        case 3:
            return launch_w4a16<kRows, kKThreads, kKTile, 3>(packed, scales, input, output,
                out_features, in_features, inverse_weight_global_scale, stream);
        case 4:
            return launch_w4a16<kRows, kKThreads, kKTile, 4>(packed, scales, input, output,
                out_features, in_features, inverse_weight_global_scale, stream);
        default:
            return cudaErrorInvalidValue;
    }
}

template <int kBatch>
cudaError_t launch_swiglu(
    const void* gate_packed,
    const void* gate_scales,
    const void* up_packed,
    const void* up_scales,
    const void* input,
    void* output,
    int out_features,
    int in_features,
    float inverse_gate_global_scale,
    float inverse_up_global_scale,
    cudaStream_t stream)
{
    constexpr int kRows = 16;
    constexpr int kKThreads = 8;
    const dim3 block(kKThreads, kRows);
    const dim3 grid((out_features + kRows - 1) / kRows);
    nvfp4_swiglu_w4a16_kernel<kBatch><<<grid, block, 0, stream>>>(
        static_cast<const uint8_t*>(gate_packed),
        static_cast<const uint8_t*>(gate_scales),
        static_cast<const uint8_t*>(up_packed),
        static_cast<const uint8_t*>(up_scales),
        static_cast<const __nv_bfloat16*>(input),
        static_cast<__nv_bfloat16*>(output),
        out_features,
        in_features,
        inverse_gate_global_scale,
        inverse_up_global_scale);
    return cudaGetLastError();
}

} // namespace

extern "C" cudaError_t qwc_nvfp4_w4a16(
    const void* packed,
    const void* scales,
    const void* input,
    void* output,
    int out_features,
    int in_features,
    int batch,
    float weight_global_scale,
    cudaStream_t stream)
{
    if (packed == nullptr || scales == nullptr || input == nullptr || output == nullptr ||
        out_features <= 0 || in_features <= 0 || in_features % 256 != 0 ||
        batch <= 0 || batch > kMaxBatch || !(weight_global_scale > 0.0f)) {
        return cudaErrorInvalidValue;
    }

    if (out_features < 8192 && in_features % 512 == 0) {
        constexpr int kRows = 4;
        constexpr int kKThreads = 32;
        constexpr int kKTile = 512;
        return launch_batch<kRows, kKThreads, kKTile>(packed, scales, input, output,
            out_features, in_features, batch, 1.0f / weight_global_scale, stream);
    } else {
        constexpr int kRows = 16;
        constexpr int kKThreads = 8;
        constexpr int kKTile = 256;
        return launch_batch<kRows, kKThreads, kKTile>(packed, scales, input, output,
            out_features, in_features, batch, 1.0f / weight_global_scale, stream);
    }
}

extern "C" cudaError_t qwc_nvfp4_swiglu_w4a16(
    const void* gate_packed,
    const void* gate_scales,
    const void* up_packed,
    const void* up_scales,
    const void* input,
    void* output,
    int out_features,
    int in_features,
    int batch,
    float gate_global_scale,
    float up_global_scale,
    cudaStream_t stream)
{
    if (gate_packed == nullptr || gate_scales == nullptr ||
        up_packed == nullptr || up_scales == nullptr || input == nullptr || output == nullptr ||
        out_features <= 0 || in_features <= 0 || in_features % 256 != 0 ||
        batch <= 0 || batch > kMaxBatch ||
        !(gate_global_scale > 0.0f) || !(up_global_scale > 0.0f)) {
        return cudaErrorInvalidValue;
    }

    switch (batch) {
        case 1:
            return launch_swiglu<1>(gate_packed, gate_scales, up_packed, up_scales,
                input, output, out_features, in_features,
                1.0f / gate_global_scale, 1.0f / up_global_scale, stream);
        case 2:
            return launch_swiglu<2>(gate_packed, gate_scales, up_packed, up_scales,
                input, output, out_features, in_features,
                1.0f / gate_global_scale, 1.0f / up_global_scale, stream);
        case 3:
            return launch_swiglu<3>(gate_packed, gate_scales, up_packed, up_scales,
                input, output, out_features, in_features,
                1.0f / gate_global_scale, 1.0f / up_global_scale, stream);
        case 4:
            return launch_swiglu<4>(gate_packed, gate_scales, up_packed, up_scales,
                input, output, out_features, in_features,
                1.0f / gate_global_scale, 1.0f / up_global_scale, stream);
        default:
            return cudaErrorInvalidValue;
    }
}

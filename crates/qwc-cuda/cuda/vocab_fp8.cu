// Vocabulary matrices in FP8: embedding table and lm_head.
//
// Both matrices are [vocab, hidden] BF16 in the checkpoint and cost 2.54 GB
// each. Per-row (per-token) E4M3 with an FP32 scale halves that and keeps the
// error local: one badly scaled row cannot spoil the rest of the vocabulary.
// The scale factors out of the dot product, so lm_head applies it once per
// row instead of per element.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

// Largest finite E4M3 magnitude.
constexpr float kFp8Max = 448.0f;
constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kRowsPerBlock = kThreads / kWarpSize;
// Batch limit of one logits launch: the hidden tile lives in static shared
// memory, and larger batches are split by the host wrapper.
constexpr int kMaxLogitsBatch = 8;
constexpr int kChunk = 512;

__device__ __forceinline__ float fp8_to_float(uint8_t raw) {
  __nv_fp8_e4m3 value;
  value.__x = raw;
  return static_cast<float>(value);
}

__device__ __forceinline__ uint8_t to_fp8(float value) {
  __nv_fp8_e4m3 converted(value);
  return converted.__x;
}

// One hidden column against ROWS weights: fully unrolled, so the
// byte shift within the packed word stays a constant.
template <int ROWS, int COLUMNS>
__device__ __forceinline__ void accumulate(
    const uint32_t (&packed)[ROWS],
    int shift,
    const float* __restrict__ hidden_column,
    int stride,
    float (&accumulator)[ROWS][COLUMNS]) {
  float x[COLUMNS];
#pragma unroll
  for (int i = 0; i < COLUMNS; ++i) {
    x[i] = hidden_column[i * stride];
  }
#pragma unroll
  for (int j = 0; j < ROWS; ++j) {
    const float weight = fp8_to_float(static_cast<uint8_t>(packed[j] >> (8 * shift)));
#pragma unroll
    for (int i = 0; i < COLUMNS; ++i) {
      accumulator[j][i] += weight * x[i];
    }
  }
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__global__ void quantize_rows_kernel(
    const __nv_bfloat16* __restrict__ input,
    uint8_t* __restrict__ output,
    float* __restrict__ row_scales,
    int cols,
    int first_row) {
  const int local_row = blockIdx.x;
  const int row = first_row + local_row;
  const size_t input_base = static_cast<size_t>(local_row) * cols;
  const size_t output_base = static_cast<size_t>(row) * cols;

  __shared__ float partial[kThreads / kWarpSize];
  float absolute_max = 0.0f;
  for (int column = threadIdx.x; column < cols; column += kThreads) {
    absolute_max = fmaxf(absolute_max, fabsf(__bfloat162float(input[input_base + column])));
  }
  absolute_max = fmaxf(absolute_max, __shfl_xor_sync(0xffffffffu, absolute_max, 16));
  absolute_max = fmaxf(absolute_max, __shfl_xor_sync(0xffffffffu, absolute_max, 8));
  absolute_max = fmaxf(absolute_max, __shfl_xor_sync(0xffffffffu, absolute_max, 4));
  absolute_max = fmaxf(absolute_max, __shfl_xor_sync(0xffffffffu, absolute_max, 2));
  absolute_max = fmaxf(absolute_max, __shfl_xor_sync(0xffffffffu, absolute_max, 1));
  if ((threadIdx.x & (kWarpSize - 1)) == 0) {
    partial[threadIdx.x / kWarpSize] = absolute_max;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float row_max = 0.0f;
    for (int i = 0; i < kThreads / kWarpSize; ++i) {
      row_max = fmaxf(row_max, partial[i]);
    }
    // An empty row must not cause division by zero and carries no information.
    partial[0] = row_max > 0.0f ? row_max / kFp8Max : 1.0f;
    row_scales[row] = partial[0];
  }
  __syncthreads();

  const float inverse = 1.0f / partial[0];
  for (int column = threadIdx.x; column < cols; column += kThreads) {
    const float value = __bfloat162float(input[input_base + column]) * inverse;
    output[output_base + column] = to_fp8(value);
  }
}

__global__ void embedding_gather_kernel(
    const uint8_t* __restrict__ table,
    const float* __restrict__ row_scales,
    const uint32_t* __restrict__ token_ids,
    __nv_bfloat16* __restrict__ output,
    int cols,
    int vocab) {
  const uint32_t token = token_ids[blockIdx.x];
  const size_t output_base = static_cast<size_t>(blockIdx.x) * cols;
  if (token >= static_cast<uint32_t>(vocab)) {
    // A token outside the vocab is an error further up the stack; here what matters is not reading
    // foreign memory and producing a distinguishable zero.
    for (int column = threadIdx.x; column < cols; column += kThreads) {
      output[output_base + column] = __float2bfloat16(0.0f);
    }
    return;
  }
  const float scale = row_scales[token];
  const size_t table_base = static_cast<size_t>(token) * cols;
  for (int column = threadIdx.x; column < cols; column += kThreads) {
    output[output_base + column] =
        __float2bfloat16(fp8_to_float(table[table_base + column]) * scale);
  }
}

__global__ void bf16_embedding_gather_kernel(
    const __nv_bfloat16* __restrict__ table,
    const uint32_t* __restrict__ token_ids,
    __nv_bfloat16* __restrict__ output,
    int cols,
    int vocab) {
  const uint32_t token = token_ids[blockIdx.x];
  const size_t output_base = static_cast<size_t>(blockIdx.x) * cols;
  if (token >= static_cast<uint32_t>(vocab)) {
    for (int column = threadIdx.x; column < cols; column += kThreads) {
      output[output_base + column] = __float2bfloat16(0.0f);
    }
    return;
  }
  const size_t table_base = static_cast<size_t>(token) * cols;
  for (int column = threadIdx.x; column < cols; column += kThreads) {
    output[output_base + column] = table[table_base + column];
  }
}

// One warp per vocabulary row, one CTA per kRowsPerBlock rows. The hidden
// states are staged in shared memory once per CTA: without that every warp
// would re-read them from L2, and at batch 8 that traffic already exceeds the
// weights themselves.
template <int BATCH>
__global__ void lm_head_kernel(
    const uint8_t* __restrict__ weights,
    const float* __restrict__ row_scales,
    const __nv_bfloat16* __restrict__ hidden,
    float* __restrict__ logits,
    int hidden_size,
    int vocab) {
  __shared__ float staged[BATCH][kChunk];

  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int row = blockIdx.x * kRowsPerBlock + warp;
  const bool active = row < vocab;
  const size_t weight_base = static_cast<size_t>(row) * hidden_size;

  float accumulator[BATCH];
#pragma unroll
  for (int b = 0; b < BATCH; ++b) {
    accumulator[b] = 0.0f;
  }

  for (int chunk = 0; chunk < hidden_size; chunk += kChunk) {
    const int width = min(kChunk, hidden_size - chunk);
    __syncthreads();
    for (int index = threadIdx.x; index < BATCH * width; index += kThreads) {
      const int b = index / width;
      const int column = index - b * width;
      staged[b][column] =
          __bfloat162float(hidden[static_cast<size_t>(b) * hidden_size + chunk + column]);
    }
    __syncthreads();
    if (!active) {
      continue;
    }
    // Each lane takes four weights at a time: fp8 is read as uint32 words,
    // otherwise a row takes 5120 single-byte accesses.
    for (int base = lane * 4; base < width; base += kWarpSize * 4) {
      const int remaining = width - base;
      uint32_t packed = 0;
      if (remaining >= 4) {
        packed = *reinterpret_cast<const uint32_t*>(weights + weight_base + chunk + base);
      } else {
        for (int i = 0; i < remaining; ++i) {
          packed |= static_cast<uint32_t>(weights[weight_base + chunk + base + i]) << (8 * i);
        }
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        if (i >= remaining) {
          break;
        }
        const float weight = fp8_to_float(static_cast<uint8_t>(packed >> (8 * i)));
#pragma unroll
        for (int b = 0; b < BATCH; ++b) {
          accumulator[b] += weight * staged[b][base + i];
        }
      }
    }
  }

  if (!active) {
    return;
  }
  const float scale = row_scales[row];
#pragma unroll
  for (int b = 0; b < BATCH; ++b) {
    const float total = warp_sum(accumulator[b]);
    if (lane == 0) {
      logits[static_cast<size_t>(b) * vocab + row] = total * scale;
    }
  }
}

// Batched pass over the vocab: a tile of [kRowTile rows × BATCH_TILE columns].
//
// The old kernel keeps the hidden tile in static shared and is capped at batch 8,
// while larger batches re-read all 1.27 GB of the table for each group: at
// concurrency 64 that's eight passes and a third of the step. Here the table is read
// once. The cost of staging hidden in shared is amortized by the number of rows per block:
// at 128 rows it's half the weights themselves, whereas at eight it would be
// an order of magnitude more.
constexpr int kRowTile = 128;
constexpr int kBatchThreads = 16;
constexpr int kRowThreads = kThreads / kBatchThreads;
constexpr int kRowsPerThread = kRowTile / kRowThreads;
constexpr int kLogitsChunk = 64;
// Tile step over batch: accumulators live in registers, so the
// smallest fitting tile is taken, not the largest.
constexpr int kBatchTileStep = kBatchThreads;
constexpr int kMaxBatchedLogits = 96;

template <int BATCH_TILE>
__global__ __launch_bounds__(kThreads) void lm_head_batched_kernel(
    const uint8_t* __restrict__ weights,
    const float* __restrict__ row_scales,
    const __nv_bfloat16* __restrict__ hidden,
    float* __restrict__ logits,
    int hidden_size,
    int vocab,
    int batch) {
  constexpr int kBatchPerThread = BATCH_TILE / kBatchThreads;
  __shared__ uint8_t weight_tile[kRowTile][kLogitsChunk];
  // An extra column spreads the tile reads in the inner loop across banks.
  __shared__ float hidden_tile[BATCH_TILE][kLogitsChunk + 1];

  const int row_base = blockIdx.x * kRowTile;
  const int row_group = threadIdx.x / kBatchThreads;
  const int batch_group = threadIdx.x % kBatchThreads;

  float accumulator[kRowsPerThread][kBatchPerThread];
#pragma unroll
  for (int j = 0; j < kRowsPerThread; ++j) {
#pragma unroll
    for (int i = 0; i < kBatchPerThread; ++i) {
      accumulator[j][i] = 0.0f;
    }
  }

  for (int base = 0; base < hidden_size; base += kLogitsChunk) {
    const int width = min(kLogitsChunk, hidden_size - base);
    __syncthreads();
    // Weights: four bytes per thread, sixteen threads per row.
    for (int index = threadIdx.x; index < kRowTile * (kLogitsChunk / 4);
         index += kThreads) {
      const int row = index / (kLogitsChunk / 4);
      const int column = (index % (kLogitsChunk / 4)) * 4;
      uint32_t packed = 0;
      if (row_base + row < vocab && column < width) {
        const size_t offset =
            static_cast<size_t>(row_base + row) * hidden_size + base + column;
        if (column + 4 <= width) {
          packed = *reinterpret_cast<const uint32_t*>(weights + offset);
        } else {
          for (int i = 0; i < width - column; ++i) {
            packed |= static_cast<uint32_t>(weights[offset + i]) << (8 * i);
          }
        }
      }
      *reinterpret_cast<uint32_t*>(&weight_tile[row][column]) = packed;
    }
    // hidden: a row per sequence, columns go consecutively.
    for (int index = threadIdx.x; index < BATCH_TILE * kLogitsChunk;
         index += kThreads) {
      const int b = index / kLogitsChunk;
      const int column = index - b * kLogitsChunk;
      float value = 0.0f;
      if (b < batch && column < width) {
        value = __bfloat162float(
            hidden[static_cast<size_t>(b) * hidden_size + base + column]);
      }
      hidden_tile[b][column] = value;
    }
    __syncthreads();

    for (int k = 0; k < width; k += 4) {
      uint32_t packed[kRowsPerThread];
#pragma unroll
      for (int j = 0; j < kRowsPerThread; ++j) {
        packed[j] = *reinterpret_cast<const uint32_t*>(
            &weight_tile[row_group * kRowsPerThread + j][k]);
      }
      if (k + 4 <= width) {
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          accumulate<kRowsPerThread, kBatchPerThread>(
              packed, s, &hidden_tile[batch_group * kBatchPerThread][k + s],
              kLogitsChunk + 1, accumulator);
        }
      } else {
        for (int s = 0; s < width - k; ++s) {
          accumulate<kRowsPerThread, kBatchPerThread>(
              packed, s, &hidden_tile[batch_group * kBatchPerThread][k + s],
              kLogitsChunk + 1, accumulator);
        }
      }
    }
  }

#pragma unroll
  for (int j = 0; j < kRowsPerThread; ++j) {
    const int row = row_base + row_group * kRowsPerThread + j;
    if (row >= vocab) {
      continue;
    }
    const float scale = row_scales[row];
#pragma unroll
    for (int i = 0; i < kBatchPerThread; ++i) {
      const int b = batch_group * kBatchPerThread + i;
      if (b < batch) {
        logits[static_cast<size_t>(b) * vocab + row] = accumulator[j][i] * scale;
      }
    }
  }
}

// A wide batch goes through tensor cores, and it's bound not by them but by
// how many bytes of the table are in flight. Weights are loaded straight into fragment
// registers 16 bytes per thread and prefetched one chunk ahead while
// the current one is computed; only hidden goes through shared. The old variant put
// a 128x32 tile into shared byte by byte, with two barriers per tile, and held
// 380-450 GB/s at any batch.
//
// Within a chunk of kMmaK columns the K order is permuted: thread (group,
// position) takes 16 consecutive bytes of a row, and step s hands out of them
// columns 4s..4s+3 onto logical k = 2p, 2p+1, 2p+8, 2p+9. The B fragment reads
// hidden with the same permutation, so the dot product doesn't change.
constexpr int kMmaK = 64;
constexpr int kMmaRowTiles = 1;
constexpr int kMmaWarps = kThreads / kWarpSize;
constexpr int kMmaRows = kMmaWarps * 16 * kMmaRowTiles;
constexpr int kHiddenChunk = 128;
// Four extra bf16 shift a hidden row by two banks: reading the B fragment
// eight bytes at a time goes without conflicts.
constexpr int kHiddenStride = kHiddenChunk + 4;

__device__ __forceinline__ void mma_m16n8k16(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
        "r"(b[0]), "r"(b[1]));
}

// Two E4M3 from the low 16 bits into a BF16 pair. Both conversions are exact: E4M3
// fits in both f16 and bf16 without rounding.
__device__ __forceinline__ uint32_t e4m3x2_to_bf16x2(uint32_t pair) {
  const __half2_raw raw = __nv_cvt_fp8x2_to_halfraw2(
      static_cast<__nv_fp8x2_storage_t>(pair & 0xffffu), __NV_E4M3);
  const float2 value = __half22float2(*reinterpret_cast<const __half2*>(&raw));
  const __nv_bfloat162 packed = __float22bfloat162_rn(value);
  return *reinterpret_cast<const uint32_t*>(&packed);
}

template <int BATCH_TILE>
__global__ __launch_bounds__(kThreads) void lm_head_batched_mma_kernel(
    const uint8_t* __restrict__ weights,
    const float* __restrict__ row_scales,
    const __nv_bfloat16* __restrict__ hidden,
    float* __restrict__ logits,
    int hidden_size,
    int vocab,
    int batch) {
  static_assert(BATCH_TILE % 8 == 0);
  constexpr int kBatchTiles = BATCH_TILE / 8;
  __shared__ __align__(16) __nv_bfloat16 hidden_tile[BATCH_TILE][kHiddenStride];

  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int group = lane / 4;
  const int position = lane % 4;
  const int warp_row = blockIdx.x * kMmaRows + warp * 16 * kMmaRowTiles;

  // Row r = warp_row + 16t + group + 8h; beyond the vocab — zeros.
  const uint8_t* row_base[kMmaRowTiles][2];
  bool row_valid[kMmaRowTiles][2];
#pragma unroll
  for (int t = 0; t < kMmaRowTiles; ++t) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int row = warp_row + t * 16 + group + h * 8;
      row_valid[t][h] = row < vocab;
      row_base[t][h] = weights + static_cast<size_t>(row_valid[t][h] ? row : 0) *
                                     hidden_size + position * 16;
    }
  }

  // A loop step is kHiddenChunk columns, kMmaSteps chunks of kMmaK: the next
  // step is fully in flight while the current one is computed.
  constexpr int kMmaSteps = kHiddenChunk / kMmaK;
  uint4 current[kMmaSteps][kMmaRowTiles][2];
  uint4 next[kMmaSteps][kMmaRowTiles][2];
  auto load = [&](uint4 (&into)[kMmaSteps][kMmaRowTiles][2], int k) {
#pragma unroll
    for (int u = 0; u < kMmaSteps; ++u) {
#pragma unroll
      for (int t = 0; t < kMmaRowTiles; ++t) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          into[u][t][h] =
              row_valid[t][h]
                  ? __ldcs(reinterpret_cast<const uint4*>(row_base[t][h] + k + u * kMmaK))
                  : make_uint4(0, 0, 0, 0);
        }
      }
    }
  };

  float accumulator[kMmaRowTiles][kBatchTiles][4] = {};
  load(current, 0);

  for (int k = 0; k < hidden_size; k += kHiddenChunk) {
    // The next step goes into flight before the barrier and before the current one's math.
    if (k + kHiddenChunk < hidden_size) {
      load(next, k + kHiddenChunk);
    }
    __syncthreads();
    constexpr int kVectors = kHiddenChunk / 4;
    for (int index = threadIdx.x; index < BATCH_TILE * kVectors;
         index += kThreads) {
      const int b = index / kVectors;
      const int column = (index % kVectors) * 4;
      uint2 value = make_uint2(0, 0);
      if (b < batch) {
        value = *reinterpret_cast<const uint2*>(
            hidden + static_cast<size_t>(b) * hidden_size + k + column);
      }
      *reinterpret_cast<uint2*>(&hidden_tile[b][column]) = value;
    }
    __syncthreads();

#pragma unroll
    for (int u = 0; u < kMmaSteps; ++u) {
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        uint32_t af[kMmaRowTiles][4];
#pragma unroll
        for (int t = 0; t < kMmaRowTiles; ++t) {
          const uint32_t low = (&current[u][t][0].x)[s];
          const uint32_t high = (&current[u][t][1].x)[s];
          af[t][0] = e4m3x2_to_bf16x2(low);
          af[t][1] = e4m3x2_to_bf16x2(high);
          af[t][2] = e4m3x2_to_bf16x2(low >> 16);
          af[t][3] = e4m3x2_to_bf16x2(high >> 16);
        }
#pragma unroll
        for (int tile = 0; tile < kBatchTiles; ++tile) {
          const uint2 pair = *reinterpret_cast<const uint2*>(
              &hidden_tile[tile * 8 + group][u * kMmaK + position * 16 + s * 4]);
          const uint32_t bf[2] = {pair.x, pair.y};
#pragma unroll
          for (int t = 0; t < kMmaRowTiles; ++t) {
            mma_m16n8k16(accumulator[t][tile], af[t], bf);
          }
        }
      }
    }

#pragma unroll
    for (int u = 0; u < kMmaSteps; ++u) {
#pragma unroll
      for (int t = 0; t < kMmaRowTiles; ++t) {
        current[u][t][0] = next[u][t][0];
        current[u][t][1] = next[u][t][1];
      }
    }
  }

#pragma unroll
  for (int t = 0; t < kMmaRowTiles; ++t) {
#pragma unroll
    for (int index = 0; index < 4; ++index) {
      const int row = warp_row + t * 16 + group + (index >= 2 ? 8 : 0);
      if (row >= vocab) {
        continue;
      }
      const float scale = row_scales[row];
#pragma unroll
      for (int tile = 0; tile < kBatchTiles; ++tile) {
        const int column_batch = tile * 8 + position * 2 + (index & 1);
        if (column_batch < batch) {
          logits[static_cast<size_t>(column_batch) * vocab + row] =
              accumulator[t][tile][index] * scale;
        }
      }
    }
  }
}

// Diagnostic path over the checkpoint's original BF16 lm_head. It deliberately
// mirrors the FP8 launch geometry so an A/B run changes weight storage and
// dequantization, not the surrounding executor.
template <int BATCH>
__global__ void bf16_lm_head_kernel(
    const __nv_bfloat16* __restrict__ weights,
    const __nv_bfloat16* __restrict__ hidden,
    float* __restrict__ logits,
    int hidden_size,
    int vocab) {
  __shared__ float staged[BATCH][kChunk];

  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int row = blockIdx.x * kRowsPerBlock + warp;
  const bool active = row < vocab;
  const size_t weight_base = static_cast<size_t>(row) * hidden_size;

  float accumulator[BATCH];
#pragma unroll
  for (int b = 0; b < BATCH; ++b) {
    accumulator[b] = 0.0f;
  }

  for (int chunk = 0; chunk < hidden_size; chunk += kChunk) {
    const int width = min(kChunk, hidden_size - chunk);
    __syncthreads();
    for (int index = threadIdx.x; index < BATCH * width; index += kThreads) {
      const int b = index / width;
      const int column = index - b * width;
      staged[b][column] =
          __bfloat162float(hidden[static_cast<size_t>(b) * hidden_size + chunk + column]);
    }
    __syncthreads();
    if (!active) {
      continue;
    }
    for (int column = lane; column < width; column += kWarpSize) {
      const float weight = __bfloat162float(weights[weight_base + chunk + column]);
#pragma unroll
      for (int b = 0; b < BATCH; ++b) {
        accumulator[b] += weight * staged[b][column];
      }
    }
  }

  if (!active) {
    return;
  }
#pragma unroll
  for (int b = 0; b < BATCH; ++b) {
    const float total = warp_sum(accumulator[b]);
    if (lane == 0) {
      logits[static_cast<size_t>(b) * vocab + row] = total;
    }
  }
}

// Logits only over a list of vocab rows.
//
// The draft head doesn't need an exact argmax over 248 320 rows: its proposal
// is verified by the main model anyway, and a shortlist miss costs a rejected
// draft, not a wrong output. The row list is given from outside — a frequency
// prefix of the vocab plus context tokens.
//
// A warp owns a row: a row is 5120 bytes, the strip is read four bytes per
// thread, the activation lies in shared and is read by all warps of the block.
__global__ __launch_bounds__(kThreads) void lm_head_subset_kernel(
    const uint8_t* __restrict__ weights,
    const float* __restrict__ row_scales,
    const __nv_bfloat16* __restrict__ hidden,
    const uint32_t* __restrict__ row_ids,
    float* __restrict__ logits,
    int hidden_size,
    int count,
    int vocab) {
  extern __shared__ float staged[];
  for (int index = threadIdx.x; index < hidden_size; index += kThreads) {
    staged[index] = __bfloat162float(hidden[index]);
  }
  __syncthreads();

  constexpr int kWarpsPerBlock = kThreads / kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int slot = blockIdx.x * kWarpsPerBlock + warp;
  if (slot >= count) {
    return;
  }
  const int row = static_cast<int>(row_ids[slot]);
  if (row < 0 || row >= vocab) {
    if (lane == 0) {
      logits[slot] = -INFINITY;
    }
    return;
  }

  const size_t base = static_cast<size_t>(row) * hidden_size;
  float accumulator = 0.0f;
  for (int column = lane * 4; column < hidden_size; column += kWarpSize * 4) {
    const uint32_t packed =
        *reinterpret_cast<const uint32_t*>(weights + base + column);
#pragma unroll
    for (int byte = 0; byte < 4; ++byte) {
      accumulator = fmaf(
          fp8_to_float(static_cast<uint8_t>(packed >> (8 * byte))),
          staged[column + byte],
          accumulator);
    }
  }
  const float total = warp_sum(accumulator);
  if (lane == 0) {
    logits[slot] = total * row_scales[row];
  }
}

}  // namespace

extern "C" cudaError_t qwc_fp8_lm_head_subset(
    const void* weights,
    const void* row_scales,
    const void* hidden,
    const void* row_ids,
    void* logits,
    int count,
    int hidden_size,
    int vocab,
    cudaStream_t stream) {
  if (weights == nullptr || row_scales == nullptr || hidden == nullptr ||
      row_ids == nullptr || logits == nullptr || count <= 0 || hidden_size <= 0 ||
      hidden_size % 4 != 0 || vocab <= 0) {
    return cudaErrorInvalidValue;
  }
  constexpr int kWarpsPerBlock = kThreads / kWarpSize;
  const int blocks = (count + kWarpsPerBlock - 1) / kWarpsPerBlock;
  const size_t shared = static_cast<size_t>(hidden_size) * sizeof(float);
  lm_head_subset_kernel<<<blocks, kThreads, shared, stream>>>(
      static_cast<const uint8_t*>(weights),
      static_cast<const float*>(row_scales),
      static_cast<const __nv_bfloat16*>(hidden),
      static_cast<const uint32_t*>(row_ids),
      static_cast<float*>(logits),
      hidden_size,
      count,
      vocab);
  return cudaGetLastError();
}

extern "C" cudaError_t qwc_fp8_quantize_rows(
    const void* input,
    void* output,
    void* row_scales,
    int rows,
    int cols,
    int first_row,
    cudaStream_t stream) {
  if (input == nullptr || output == nullptr || row_scales == nullptr || rows <= 0 ||
      cols <= 0 || first_row < 0) {
    return cudaErrorInvalidValue;
  }
  quantize_rows_kernel<<<rows, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<uint8_t*>(output),
      static_cast<float*>(row_scales),
      cols,
      first_row);
  return cudaGetLastError();
}

extern "C" cudaError_t qwc_fp8_embedding_gather(
    const void* table,
    const void* row_scales,
    const void* token_ids,
    void* output,
    int batch,
    int cols,
    int vocab,
    cudaStream_t stream) {
  if (table == nullptr || row_scales == nullptr || token_ids == nullptr ||
      output == nullptr || batch <= 0 || cols <= 0 || vocab <= 0) {
    return cudaErrorInvalidValue;
  }
  embedding_gather_kernel<<<batch, kThreads, 0, stream>>>(
      static_cast<const uint8_t*>(table),
      static_cast<const float*>(row_scales),
      static_cast<const uint32_t*>(token_ids),
      static_cast<__nv_bfloat16*>(output),
      cols,
      vocab);
  return cudaGetLastError();
}

extern "C" cudaError_t qwc_bf16_embedding_gather(
    const void* table,
    const void* token_ids,
    void* output,
    int batch,
    int cols,
    int vocab,
    cudaStream_t stream) {
  if (table == nullptr || token_ids == nullptr || output == nullptr || batch <= 0 ||
      cols <= 0 || vocab <= 0) {
    return cudaErrorInvalidValue;
  }
  bf16_embedding_gather_kernel<<<batch, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(table),
      static_cast<const uint32_t*>(token_ids),
      static_cast<__nv_bfloat16*>(output),
      cols,
      vocab);
  return cudaGetLastError();
}

extern "C" cudaError_t qwc_fp8_lm_head(
    const void* weights,
    const void* row_scales,
    const void* hidden,
    void* logits,
    int batch,
    int hidden_size,
    int vocab,
    cudaStream_t stream) {
  if (weights == nullptr || row_scales == nullptr || hidden == nullptr ||
      logits == nullptr || batch <= 0 || batch > kMaxLogitsBatch || hidden_size <= 0 ||
      vocab <= 0) {
    return cudaErrorInvalidValue;
  }
  const int blocks = (vocab + kRowsPerBlock - 1) / kRowsPerBlock;
  const uint8_t* w = static_cast<const uint8_t*>(weights);
  const float* s = static_cast<const float*>(row_scales);
  const __nv_bfloat16* h = static_cast<const __nv_bfloat16*>(hidden);
  float* out = static_cast<float*>(logits);
  switch (batch) {
#define QWC_LM_HEAD_CASE(n)                                                  \
  case n:                                                                    \
    lm_head_kernel<n><<<blocks, kThreads, 0, stream>>>(                      \
        w, s, h, out, hidden_size, vocab);                                   \
    break;
    QWC_LM_HEAD_CASE(1)
    QWC_LM_HEAD_CASE(2)
    QWC_LM_HEAD_CASE(3)
    QWC_LM_HEAD_CASE(4)
    QWC_LM_HEAD_CASE(5)
    QWC_LM_HEAD_CASE(6)
    QWC_LM_HEAD_CASE(7)
    QWC_LM_HEAD_CASE(8)
#undef QWC_LM_HEAD_CASE
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

extern "C" cudaError_t qwc_bf16_lm_head(
    const void* weights,
    const void* hidden,
    void* logits,
    int batch,
    int hidden_size,
    int vocab,
    cudaStream_t stream) {
  if (weights == nullptr || hidden == nullptr || logits == nullptr || batch <= 0 ||
      batch > kMaxLogitsBatch || hidden_size <= 0 || vocab <= 0) {
    return cudaErrorInvalidValue;
  }
  const int blocks = (vocab + kRowsPerBlock - 1) / kRowsPerBlock;
  const __nv_bfloat16* w = static_cast<const __nv_bfloat16*>(weights);
  const __nv_bfloat16* h = static_cast<const __nv_bfloat16*>(hidden);
  float* out = static_cast<float*>(logits);
  switch (batch) {
#define QWC_BF16_LM_HEAD_CASE(n)                                             \
  case n:                                                                    \
    bf16_lm_head_kernel<n><<<blocks, kThreads, 0, stream>>>(                 \
        w, h, out, hidden_size, vocab);                                      \
    break;
    QWC_BF16_LM_HEAD_CASE(1)
    QWC_BF16_LM_HEAD_CASE(2)
    QWC_BF16_LM_HEAD_CASE(3)
    QWC_BF16_LM_HEAD_CASE(4)
    QWC_BF16_LM_HEAD_CASE(5)
    QWC_BF16_LM_HEAD_CASE(6)
    QWC_BF16_LM_HEAD_CASE(7)
    QWC_BF16_LM_HEAD_CASE(8)
#undef QWC_BF16_LM_HEAD_CASE
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

extern "C" cudaError_t qwc_fp8_lm_head_batched(
    const void* weights,
    const void* row_scales,
    const void* hidden,
    void* logits,
    int batch,
    int hidden_size,
    int vocab,
    cudaStream_t stream) {
  if (weights == nullptr || row_scales == nullptr || hidden == nullptr ||
      logits == nullptr || batch <= 0 || batch > kMaxBatchedLogits ||
      hidden_size <= 0 || hidden_size % 4 != 0 || vocab <= 0) {
    return cudaErrorInvalidValue;
  }
  const uint8_t* w = static_cast<const uint8_t*>(weights);
  const float* s = static_cast<const float*>(row_scales);
  const __nv_bfloat16* h = static_cast<const __nv_bfloat16*>(hidden);
  float* out = static_cast<float*>(logits);
  // The MMA path takes hidden in chunks of kHiddenChunk; other widths go through
  // the scalar kernel.
  const bool mma = hidden_size % kHiddenChunk == 0;
  const int blocks = mma ? (vocab + kMmaRows - 1) / kMmaRows
                         : (vocab + kRowTile - 1) / kRowTile;
  switch ((batch + kBatchTileStep - 1) / kBatchTileStep) {
#define QWC_LM_HEAD_TILE(n)                                                  \
  case (n) / kBatchTileStep:                                                 \
    if (mma) {                                                               \
      lm_head_batched_mma_kernel<n><<<blocks, kThreads, 0, stream>>>(        \
          w, s, h, out, hidden_size, vocab, batch);                          \
    } else {                                                                 \
      lm_head_batched_kernel<n><<<blocks, kThreads, 0, stream>>>(            \
          w, s, h, out, hidden_size, vocab, batch);                          \
    }                                                                        \
    break;
    QWC_LM_HEAD_TILE(16)
    QWC_LM_HEAD_TILE(32)
    QWC_LM_HEAD_TILE(48)
    QWC_LM_HEAD_TILE(64)
    QWC_LM_HEAD_TILE(80)
    QWC_LM_HEAD_TILE(96)
#undef QWC_LM_HEAD_TILE
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

extern "C" int qwc_fp8_max_batched_logits() { return kMaxBatchedLogits; }

extern "C" int qwc_fp8_max_logits_batch() { return kMaxLogitsBatch; }

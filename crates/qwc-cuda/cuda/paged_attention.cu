// FP8 paged-attention decode for the 16 full-attention layers of Qwen3.8.
//
// One CTA owns one KV head and all six query heads in its GQA group. Each
// cached K/V element is therefore loaded once instead of six times. Long
// contexts are partitioned across grid.z and merged by a second kernel.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>
#include "limits.cuh"

namespace {

constexpr int kQueryHeads = 24;
constexpr int kKvHeads = 4;
constexpr int kGroup = kQueryHeads / kKvHeads;
constexpr int kHeadDim = 256;
constexpr int kPageSize = 64;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kValuesPerLane = kHeadDim / 32;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ float fp8_to_float(uint8_t raw) {
  __nv_fp8_e4m3 value;
  reinterpret_cast<uint8_t&>(value) = raw;
  return static_cast<float>(value);
}

template <bool Bf16Cache>
__device__ __forceinline__ float cache_to_float(const void* cache, size_t offset) {
  if constexpr (Bf16Cache) {
    return __bfloat162float(static_cast<const __nv_bfloat16*>(cache)[offset]);
  } else {
    return fp8_to_float(static_cast<const uint8_t*>(cache)[offset]);
  }
}

// How many elements one thread puts into shared at a time. Element-wise staging
// of KV cost more instructions than all the tile's MMAs combined.
constexpr int kStageVector = 8;

// Eight cache elements per access.
//
// Element-wise reads turned tile staging into 16384 two-byte loads
// per CTA. It's always aligned: the dimension is a multiple of eight, and the row stride in shared
// is a multiple of 16 bytes.
template <bool Bf16Cache>
__device__ __forceinline__ void stage_eight(
    const void* cache, size_t offset, __nv_bfloat16* destination) {
  __nv_bfloat16 staged[kStageVector];
  if constexpr (Bf16Cache) {
    const float4 raw = *reinterpret_cast<const float4*>(
        static_cast<const __nv_bfloat16*>(cache) + offset);
    *reinterpret_cast<float4*>(destination) = raw;
    return;
  } else {
    const uint2 raw = *reinterpret_cast<const uint2*>(
        static_cast<const uint8_t*>(cache) + offset);
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&raw);
#pragma unroll
    for (int e = 0; e < kStageVector; ++e) {
      staged[e] = __float2bfloat16(fp8_to_float(bytes[e]));
    }
  }
  *reinterpret_cast<float4*>(destination) =
      *reinterpret_cast<const float4*>(staged);
}

__device__ __forceinline__ void stage_eight_zero(__nv_bfloat16* destination) {
  *reinterpret_cast<float4*>(destination) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

__device__ __forceinline__ size_t cache_offset(
    uint32_t block, int kv_head, int token_in_block, int dimension) {
  return (((static_cast<size_t>(block) * kKvHeads + kv_head) * kPageSize +
           token_in_block) *
              kHeadDim +
          dimension);
}

__device__ __forceinline__ float output_gate(
    const __nv_bfloat16* query_gate_projection,
    int batch,
    int query_head,
    int dimension) {
  if (query_gate_projection == nullptr) {
    return 1.0f;
  }
  const size_t offset =
      (static_cast<size_t>(batch) * kQueryHeads + query_head) *
          (2 * kHeadDim) +
      kHeadDim + dimension;
  const float gate = __bfloat162float(query_gate_projection[offset]);
  return 1.0f / (1.0f + __expf(-gate));
}

template <bool Bf16Cache, bool Direct>
__global__ __launch_bounds__(kThreads, 1) void paged_attention_parts_kernel(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ query_gate_projection,
    const void* __restrict__ key_cache,
    const void* __restrict__ value_cache,
    const uint32_t* __restrict__ block_tables,
    const uint32_t* __restrict__ context_lengths,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    float* __restrict__ partial_output,
    int max_blocks,
    int partitions,
    float softmax_scale) {
  const int kv_head = blockIdx.x;
  const int batch = blockIdx.y;
  const int partition = blockIdx.z;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;

  __shared__ float shared_query[kGroup][kHeadDim];
  __shared__ float shared_output[kWarps][kGroup][kHeadDim];
  __shared__ float shared_max[kWarps][kGroup];
  __shared__ float shared_sum[kWarps][kGroup];
  __shared__ float merged_max[kGroup];
  __shared__ float merged_sum[kGroup];

#pragma unroll
  for (int q = 0; q < kGroup; ++q) {
    const int query_head = kv_head * kGroup + q;
    const size_t offset =
        (static_cast<size_t>(batch) * kQueryHeads + query_head) * kHeadDim;
    shared_query[q][threadIdx.x] = __bfloat162float(query[offset + threadIdx.x]);
  }
  __syncthreads();

  const int context = static_cast<int>(context_lengths[batch]);
  const int tokens_per_partition = (context + partitions - 1) / partitions;
  const int begin = partition * tokens_per_partition;
  const int end = min(context, begin + tokens_per_partition);

  float local_max[kGroup];
  float local_sum[kGroup];
  float accumulator[kGroup][kValuesPerLane];
#pragma unroll
  for (int q = 0; q < kGroup; ++q) {
    local_max[q] = -CUDART_INF_F;
    local_sum[q] = 0.0f;
#pragma unroll
    for (int i = 0; i < kValuesPerLane; ++i) {
      accumulator[q][i] = 0.0f;
    }
  }

  const int dimension_base = lane * kValuesPerLane;
  for (int token = begin + warp; token < end; token += kWarps) {
    const int logical_block = token / kPageSize;
    const uint32_t physical_block =
        block_tables[static_cast<size_t>(batch) * max_blocks + logical_block];
    const size_t base = cache_offset(
        physical_block, kv_head, token % kPageSize, dimension_base);

    float key[kValuesPerLane];
#pragma unroll
    for (int i = 0; i < kValuesPerLane; ++i) {
      key[i] = cache_to_float<Bf16Cache>(key_cache, base + i);
    }

    float scores[kGroup];
#pragma unroll
    for (int q = 0; q < kGroup; ++q) {
      float score = 0.0f;
#pragma unroll
      for (int i = 0; i < kValuesPerLane; ++i) {
        score = fmaf(key[i], shared_query[q][dimension_base + i], score);
      }
      score = warp_sum(score);
      scores[q] = __shfl_sync(0xffffffffu, score, 0) * softmax_scale;
    }

    float value[kValuesPerLane];
#pragma unroll
    for (int i = 0; i < kValuesPerLane; ++i) {
      value[i] = cache_to_float<Bf16Cache>(value_cache, base + i);
    }

#pragma unroll
    for (int q = 0; q < kGroup; ++q) {
      const float next_max = fmaxf(local_max[q], scores[q]);
      const float previous_weight = __expf(local_max[q] - next_max);
      const float token_weight = __expf(scores[q] - next_max);
      local_sum[q] = local_sum[q] * previous_weight + token_weight;
#pragma unroll
      for (int i = 0; i < kValuesPerLane; ++i) {
        accumulator[q][i] =
            accumulator[q][i] * previous_weight + token_weight * value[i];
      }
      local_max[q] = next_max;
    }
  }

#pragma unroll
  for (int q = 0; q < kGroup; ++q) {
    if (lane == 0) {
      shared_max[warp][q] = local_max[q];
      shared_sum[warp][q] = local_sum[q];
    }
#pragma unroll
    for (int i = 0; i < kValuesPerLane; ++i) {
      shared_output[warp][q][dimension_base + i] = accumulator[q][i];
    }
  }
  __syncthreads();

  if (threadIdx.x < kGroup) {
    const int q = threadIdx.x;
    float maximum = -CUDART_INF_F;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) {
      maximum = fmaxf(maximum, shared_max[w][q]);
    }
    float sum = 0.0f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) {
      if (shared_sum[w][q] > 0.0f) {
        sum += shared_sum[w][q] * __expf(shared_max[w][q] - maximum);
      }
    }
    merged_max[q] = maximum;
    merged_sum[q] = sum;
  }
  __syncthreads();

  if (warp != 0) {
    return;
  }

#pragma unroll
  for (int q = 0; q < kGroup; ++q) {
    float numerator[kValuesPerLane] = {};
#pragma unroll
    for (int w = 0; w < kWarps; ++w) {
      if (shared_sum[w][q] > 0.0f) {
        const float weight = __expf(shared_max[w][q] - merged_max[q]);
#pragma unroll
        for (int i = 0; i < kValuesPerLane; ++i) {
          numerator[i] +=
              shared_output[w][q][dimension_base + i] * weight;
        }
      }
    }

    const int query_head = kv_head * kGroup + q;
    if constexpr (Direct) {
      const size_t output_base =
          (static_cast<size_t>(batch) * kQueryHeads + query_head) * kHeadDim;
#pragma unroll
      for (int i = 0; i < kValuesPerLane; ++i) {
        const float value =
            merged_sum[q] > 0.0f ? numerator[i] / merged_sum[q] : 0.0f;
        output[output_base + dimension_base + i] = __float2bfloat16(
            value * output_gate(
                        query_gate_projection,
                        batch,
                        query_head,
                        dimension_base + i));
      }
    } else {
      const size_t partial =
          (static_cast<size_t>(batch) * kQueryHeads + query_head) * partitions +
          partition;
      if (lane == 0) {
        partial_max[partial] = merged_max[q];
        partial_sum[partial] = merged_sum[q];
      }
      const size_t output_base = partial * kHeadDim;
#pragma unroll
      for (int i = 0; i < kValuesPerLane; ++i) {
        partial_output[output_base + dimension_base + i] = numerator[i];
      }
    }
  }
}

// Causal prefill: one CTA owns a tile of query rows of one sequence,
// a warp per row. The decode kernel splits tokens between warps, so it serves
// one row per pass and on a prefill chunk re-reads the same KV pages
// as many times as there are tokens in the chunk. Here there's one KV pass per tile, and
// the tile's rows reuse the already read K and V: traffic drops kWarps times
// with the same register budget, because a warp still owns
// exactly one online softmax.
//
// No inter-warp merge is needed: a warp's accumulator is complete for its row, and
// the old kernel's shared_output (48 KiB) doesn't exist here.
template <bool Bf16Cache>
__global__ __launch_bounds__(kThreads) void paged_attention_prefill_kernel(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ query_gate_projection,
    const void* __restrict__ key_cache,
    const void* __restrict__ value_cache,
    const uint32_t* __restrict__ block_tables,
    const uint32_t* __restrict__ context_lengths,
    __nv_bfloat16* __restrict__ output,
    int max_blocks,
    int rows,
    int row_base,
    float softmax_scale) {
  const int kv_head = blockIdx.x;
  const int tile = blockIdx.y;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;

  // The tile's queries are kept in bf16: in float that's 48 KiB per block, and conversion
  // on read costs one instruction.
  __shared__ __nv_bfloat16 shared_query[kWarps][kGroup][kHeadDim];

  const int tile_first = tile * kWarps;
  const int tile_rows = min(kWarps, rows - tile_first);
  for (int slot = 0; slot < tile_rows; ++slot) {
    const int batch_row = row_base + tile_first + slot;
    for (int i = threadIdx.x; i < kGroup * kHeadDim; i += kThreads) {
      const int q = i / kHeadDim;
      const int dimension = i - q * kHeadDim;
      const int query_head = kv_head * kGroup + q;
      const size_t offset =
          (static_cast<size_t>(batch_row) * kQueryHeads + query_head) * kHeadDim;
      shared_query[slot][q][dimension] = query[offset + dimension];
    }
  }
  __syncthreads();

  // There are no more barriers after this, so extra warps simply leave.
  if (warp >= tile_rows) {
    return;
  }
  const int batch = row_base + tile_first + warp;
  const int context = static_cast<int>(context_lengths[batch]);

  float local_max[kGroup];
  float local_sum[kGroup];
  float accumulator[kGroup][kValuesPerLane];
#pragma unroll
  for (int q = 0; q < kGroup; ++q) {
    local_max[q] = -CUDART_INF_F;
    local_sum[q] = 0.0f;
#pragma unroll
    for (int i = 0; i < kValuesPerLane; ++i) {
      accumulator[q][i] = 0.0f;
    }
  }

  const int dimension_base = lane * kValuesPerLane;
  for (int token = 0; token < context; ++token) {
    const int logical_block = token / kPageSize;
    const uint32_t physical_block =
        block_tables[static_cast<size_t>(batch) * max_blocks + logical_block];
    const size_t base = cache_offset(
        physical_block, kv_head, token % kPageSize, dimension_base);

    float key[kValuesPerLane];
#pragma unroll
    for (int i = 0; i < kValuesPerLane; ++i) {
      key[i] = cache_to_float<Bf16Cache>(key_cache, base + i);
    }

    float scores[kGroup];
#pragma unroll
    for (int q = 0; q < kGroup; ++q) {
      float score = 0.0f;
#pragma unroll
      for (int i = 0; i < kValuesPerLane; ++i) {
        score = fmaf(
            key[i],
            __bfloat162float(shared_query[warp][q][dimension_base + i]),
            score);
      }
      score = warp_sum(score);
      scores[q] = __shfl_sync(0xffffffffu, score, 0) * softmax_scale;
    }

    float value[kValuesPerLane];
#pragma unroll
    for (int i = 0; i < kValuesPerLane; ++i) {
      value[i] = cache_to_float<Bf16Cache>(value_cache, base + i);
    }

#pragma unroll
    for (int q = 0; q < kGroup; ++q) {
      const float next_max = fmaxf(local_max[q], scores[q]);
      const float previous_weight = __expf(local_max[q] - next_max);
      const float token_weight = __expf(scores[q] - next_max);
      local_sum[q] = local_sum[q] * previous_weight + token_weight;
#pragma unroll
      for (int i = 0; i < kValuesPerLane; ++i) {
        accumulator[q][i] =
            accumulator[q][i] * previous_weight + token_weight * value[i];
      }
      local_max[q] = next_max;
    }
  }

#pragma unroll
  for (int q = 0; q < kGroup; ++q) {
    const int query_head = kv_head * kGroup + q;
    const size_t output_base =
        (static_cast<size_t>(batch) * kQueryHeads + query_head) * kHeadDim;
#pragma unroll
    for (int i = 0; i < kValuesPerLane; ++i) {
      const float value =
          local_sum[q] > 0.0f ? accumulator[q][i] / local_sum[q] : 0.0f;
      output[output_base + dimension_base + i] = __float2bfloat16(
          value * output_gate(
                      query_gate_projection,
                      batch,
                      query_head,
                      dimension_base + i));
    }
  }
}

// Causal prefill on tensor cores.
//
// The row-wise kernels — both the decode one and the tiled one — compute QK^T and PV with warp
// reductions: each (row, token) pair takes six butterfly passes, and
// the phase is bound by instruction issue, not memory. Here both matrices go
// through mma.m16n8k16, i.e. one instruction per 16x8x16.
//
// A tile of rows must belong to one sequence: the page table
// is taken from the tile's first row. For a prefill chunk this holds by construction.

// A lane reads a fragment at address (grp * row + pos * 2): without padding
// grp steps by multiples of 32 banks and the whole warp lands on four banks. Padding
// makes the row stride odd in banks and spreads the lanes over all 32.
constexpr int kRowPad = kHeadDim + 8;

__device__ __forceinline__ uint32_t pack_pair(
    __nv_bfloat16 low, __nv_bfloat16 high) {
  return static_cast<uint32_t>(*reinterpret_cast<uint16_t*>(&low)) |
         (static_cast<uint32_t>(*reinterpret_cast<uint16_t*>(&high)) << 16);
}

__device__ __forceinline__ void mma_m16n8k16(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Loading fragments with one instruction instead of four scalar reads.
//
// `ldmatrix` reads eight 8x8 rows at once and distributes them across lanes
// exactly as `mma.m16n8k16` expects them. Each lane provides its own address: for
// .x4 lane L gives row L%8 of matrix L/8, for .x2 the same among the first
// sixteen. The .trans variant transposes the tile on the fly, and operand V,
// stored in shared as [key][dim], needs neither reshuffling
// nor assembling two bytes at a time.
__device__ __forceinline__ uint32_t shared_address(const void* pointer) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void ldmatrix_x4(
    uint32_t (&fragment)[4], const __nv_bfloat16* source) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(fragment[0]), "=r"(fragment[1]), "=r"(fragment[2]),
        "=r"(fragment[3])
      : "r"(shared_address(source)));
}

__device__ __forceinline__ void ldmatrix_x2(
    uint32_t (&fragment)[2], const __nv_bfloat16* source) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
      : "=r"(fragment[0]), "=r"(fragment[1])
      : "r"(shared_address(source)));
}

__device__ __forceinline__ void ldmatrix_x2_trans(
    uint32_t (&fragment)[2], const __nv_bfloat16* source) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
      : "=r"(fragment[0]), "=r"(fragment[1])
      : "r"(shared_address(source)));
}

// A 64-query-row tile: a warp owns its sixteen rows and
// all of the tile's keys.
//
// The old form split the work crosswise: 16 rows per CTA, four warps cut
// along keys in QK^T and along dimensions in PV. Because of this the compute went through shared —
// S there, back for softmax, P there again — and every 32 keys
// had two barriers. Worse: a staged KV tile served 16
// query rows, and the whole context was re-read for every 16 rows.
//
// Here the KV tile is shared by 64 rows: the staging is the same, but there are four times
// as many MMAs on it. Softmax lives in the warp's registers (a butterfly over four lanes with
// the same grp), P goes from the QK^T accumulator into operand A without a single
// reshuffle, and barriers remain only for staging.
constexpr int kFlashWarps = 4;
constexpr int kFlashThreads = kFlashWarps * 32;
constexpr int kFlashRows = kFlashWarps * 16;
constexpr int kFlashKeys = 32;
constexpr int kFlashKeyTiles = kFlashKeys / 8;      // n-tiles in QK^T
constexpr int kFlashKeyGroups = kFlashKeys / 16;    // k-groups in PV
constexpr int kFlashDimTiles = kHeadDim / 8;        // accumulator n-tiles

struct FlashShared {
  __nv_bfloat16 query[kFlashRows * kRowPad];
  __nv_bfloat16 key[kFlashKeys * kRowPad];
  __nv_bfloat16 value[kFlashKeys * kRowPad];
  int row_context[kFlashRows];
};

constexpr size_t flash_shared_bytes() { return sizeof(FlashShared); }

// Row max and sum: a row lives on four lanes with the same grp,
// i.e. neighbors in the lowest two bits.
__device__ __forceinline__ float flash_row_max(float value) {
  value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 1));
  return fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 2));
}

__device__ __forceinline__ float flash_row_sum(float value) {
  value += __shfl_xor_sync(0xffffffffu, value, 1);
  return value + __shfl_xor_sync(0xffffffffu, value, 2);
}

template <bool Bf16Cache>
__global__ __launch_bounds__(kFlashThreads, 1) void paged_attention_prefill_flash_kernel(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ query_gate_projection,
    const void* __restrict__ key_cache,
    const void* __restrict__ value_cache,
    const uint32_t* __restrict__ block_tables,
    const uint32_t* __restrict__ context_lengths,
    __nv_bfloat16* __restrict__ output,
    int max_blocks,
    int rows,
    int row_base,
    float softmax_scale) {
  const int kv_head = blockIdx.x;
  const int tile = blockIdx.y;
  const int query_head = kv_head * kGroup + blockIdx.z;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int grp = lane >> 2;
  const int pos = lane & 3;

  const int tile_first = tile * kFlashRows;
  const int tile_rows = min(kFlashRows, rows - tile_first);
  if (tile_rows <= 0) {
    return;
  }
  const int row0 = warp * 16;

  extern __shared__ __align__(16) char flash_shared_raw[];
  FlashShared& shared = *reinterpret_cast<FlashShared*>(flash_shared_raw);

  for (int i = threadIdx.x; i < kFlashRows * kHeadDim; i += kFlashThreads) {
    const int dimension = i % kHeadDim;
    const int r = i / kHeadDim;
    __nv_bfloat16 value = __float2bfloat16(0.0f);
    if (r < tile_rows) {
      const size_t base =
          (static_cast<size_t>(row_base + tile_first + r) * kQueryHeads +
           query_head) * kHeadDim;
      value = query[base + dimension];
    }
    shared.query[r * kRowPad + dimension] = value;
  }
  for (int r = threadIdx.x; r < kFlashRows; r += kFlashThreads) {
    shared.row_context[r] = r < tile_rows
        ? static_cast<int>(context_lengths[row_base + tile_first + r])
        : 0;
  }
  __syncthreads();

  int max_context = 0;
#pragma unroll
  for (int r = 0; r < kFlashRows; ++r) {
    max_context = max(max_context, shared.row_context[r]);
  }
  // The page table is shared by the tile: the chunk's rows belong to one
  // sequence.
  const uint32_t* table =
      block_tables + static_cast<size_t>(row_base + tile_first) * max_blocks;
  // Causal prefixes of the warp's rows: beyond them it has nothing to do in the tile.
  const int warp_context =
      max(shared.row_context[row0 + 15], shared.row_context[row0]);

  float accumulator[kFlashDimTiles][4] = {};
  float running_max[2] = {-CUDART_INF_F, -CUDART_INF_F};
  float running_sum[2] = {0.0f, 0.0f};

  for (int key_base = 0; key_base < max_context; key_base += kFlashKeys) {
    __syncthreads();
    for (int i = threadIdx.x * kStageVector; i < kFlashKeys * kHeadDim;
         i += kFlashThreads * kStageVector) {
      const int key = i / kHeadDim;
      const int dimension = i - key * kHeadDim;
      const int token = key_base + key;
      if (token < max_context) {
        const uint32_t block = table[token / kPageSize];
        const size_t offset =
            cache_offset(block, kv_head, token % kPageSize, dimension);
        stage_eight<Bf16Cache>(
            key_cache, offset, &shared.key[key * kRowPad + dimension]);
        stage_eight<Bf16Cache>(
            value_cache, offset, &shared.value[key * kRowPad + dimension]);
      } else {
        stage_eight_zero(&shared.key[key * kRowPad + dimension]);
        stage_eight_zero(&shared.value[key * kRowPad + dimension]);
      }
    }
    __syncthreads();
    if (key_base >= warp_context) {
      continue;   // the warp has done its barriers, nothing for it to compute
    }

    // QK^T: the warp's rows against all keys of the tile.
    float scores[kFlashKeyTiles][4] = {};
    // Addresses for ldmatrix: a lane gives a row of its 8x8 matrix.
    const int matrix = lane >> 3;
    const int matrix_row = lane & 7;
    const __nv_bfloat16* query_row =
        &shared.query[(row0 + matrix_row + 8 * (matrix & 1)) * kRowPad
                      + 8 * (matrix >> 1)];
    const __nv_bfloat16* key_row =
        &shared.key[(size_t)matrix_row * kRowPad + 8 * (matrix & 1)];
    for (int k0 = 0; k0 < kHeadDim; k0 += 16) {
      uint32_t a[4];
      ldmatrix_x4(a, query_row + k0);
#pragma unroll
      for (int t = 0; t < kFlashKeyTiles; ++t) {
        uint32_t b[2];
        ldmatrix_x2(b, key_row + (size_t)(t * 8) * kRowPad + k0);
        mma_m16n8k16(scores[t], a, b);
      }
    }

    // Scale and causal mask. A lane holds rows grp and grp+8.
    const int context_low = shared.row_context[row0 + grp];
    const int context_high = shared.row_context[row0 + grp + 8];
#pragma unroll
    for (int t = 0; t < kFlashKeyTiles; ++t) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int token = key_base + t * 8 + pos * 2 + (i & 1);
        const int limit = (i >= 2) ? context_high : context_low;
        scores[t][i] =
            token < limit ? scores[t][i] * softmax_scale : -CUDART_INF_F;
      }
    }

    // Online softmax: all in warp registers, not a single shared access.
    float tile_max[2] = {-CUDART_INF_F, -CUDART_INF_F};
#pragma unroll
    for (int t = 0; t < kFlashKeyTiles; ++t) {
      tile_max[0] = fmaxf(tile_max[0], fmaxf(scores[t][0], scores[t][1]));
      tile_max[1] = fmaxf(tile_max[1], fmaxf(scores[t][2], scores[t][3]));
    }
    tile_max[0] = flash_row_max(tile_max[0]);
    tile_max[1] = flash_row_max(tile_max[1]);

    float correction[2];
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const float merged = fmaxf(running_max[half], tile_max[half]);
      // A row with no allowed key in the tile: subtracting
      // from -inf is not allowed, it's NaN.
      const bool empty = merged == -CUDART_INF_F;
      correction[half] = (empty || running_max[half] == -CUDART_INF_F)
          ? (empty ? 1.0f : 0.0f)
          : __expf(running_max[half] - merged);
      float sum = 0.0f;
#pragma unroll
      for (int t = 0; t < kFlashKeyTiles; ++t) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          float& slot = scores[t][half * 2 + i];
          const float weight =
              (empty || slot == -CUDART_INF_F) ? 0.0f : __expf(slot - merged);
          slot = weight;
          sum += weight;
        }
      }
      sum = flash_row_sum(sum);
      if (!empty) {
        running_sum[half] = running_sum[half] * correction[half] + sum;
        running_max[half] = merged;
      }
    }
#pragma unroll
    for (int t = 0; t < kFlashDimTiles; ++t) {
      accumulator[t][0] *= correction[0];
      accumulator[t][1] *= correction[0];
      accumulator[t][2] *= correction[1];
      accumulator[t][3] *= correction[1];
    }

    // P from the QK^T accumulator straight into operand A: the m16n8k16 layout matches
    // if the key column is treated as the k index.
    uint32_t probability[kFlashKeyGroups][4];
#pragma unroll
    for (int g = 0; g < kFlashKeyGroups; ++g) {
      probability[g][0] = pack_pair(__float2bfloat16(scores[2 * g][0]),
                                    __float2bfloat16(scores[2 * g][1]));
      probability[g][1] = pack_pair(__float2bfloat16(scores[2 * g][2]),
                                    __float2bfloat16(scores[2 * g][3]));
      probability[g][2] = pack_pair(__float2bfloat16(scores[2 * g + 1][0]),
                                    __float2bfloat16(scores[2 * g + 1][1]));
      probability[g][3] = pack_pair(__float2bfloat16(scores[2 * g + 1][2]),
                                    __float2bfloat16(scores[2 * g + 1][3]));
    }
    // V is stored as [key][dim], while operand B needs the key along k: the
    // instruction itself transposes the tile.
    const __nv_bfloat16* value_row =
        &shared.value[(size_t)(matrix_row + 8 * matrix) * kRowPad];
#pragma unroll
    for (int g = 0; g < kFlashKeyGroups; ++g) {
#pragma unroll
      for (int t = 0; t < kFlashDimTiles; ++t) {
        uint32_t b[2];
        ldmatrix_x2_trans(
            b, value_row + (size_t)(g * 16) * kRowPad + t * 8);
        mma_m16n8k16(accumulator[t], probability[g], b);
      }
    }
  }

#pragma unroll
  for (int t = 0; t < kFlashDimTiles; ++t) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int r = row0 + grp + (i >= 2 ? 8 : 0);
      if (r >= tile_rows) {
        continue;
      }
      const int dimension = t * 8 + pos * 2 + (i & 1);
      const int batch_row = row_base + tile_first + r;
      const float denominator = running_sum[i >= 2 ? 1 : 0];
      const float value =
          denominator > 0.0f ? accumulator[t][i] / denominator : 0.0f;
      const size_t offset =
          (static_cast<size_t>(batch_row) * kQueryHeads + query_head) *
              kHeadDim + dimension;
      output[offset] = __float2bfloat16(
          value * output_gate(
                      query_gate_projection, batch_row, query_head, dimension));
    }
  }
}

// Higher-occupancy alternative: one CTA per query head. It rereads the KV
// head for every member of a GQA group, but uses far fewer registers and only
// ~9 KiB shared memory. This wins for shapes where parallelism is scarcer than
// DRAM bandwidth; the host dispatcher chooses from measured crossover data.
template <bool Bf16Cache, bool Direct>
__global__ __launch_bounds__(kThreads) void paged_attention_query_head_kernel(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ query_gate_projection,
    const void* __restrict__ key_cache,
    const void* __restrict__ value_cache,
    const uint32_t* __restrict__ block_tables,
    const uint32_t* __restrict__ context_lengths,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    float* __restrict__ partial_output,
    int max_blocks,
    int partitions,
    float softmax_scale) {
  const int query_head = blockIdx.x;
  const int kv_head = query_head / kGroup;
  const int batch = blockIdx.y;
  const int partition = blockIdx.z;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;

  __shared__ float shared_query[kHeadDim];
  __shared__ float shared_output[kWarps][kHeadDim];
  __shared__ float shared_max[kWarps];
  __shared__ float shared_sum[kWarps];
  __shared__ float merged_max;
  __shared__ float merged_sum;

  const size_t query_offset =
      (static_cast<size_t>(batch) * kQueryHeads + query_head) * kHeadDim;
  shared_query[threadIdx.x] = __bfloat162float(query[query_offset + threadIdx.x]);
  __syncthreads();

  const int context = static_cast<int>(context_lengths[batch]);
  const int tokens_per_partition = (context + partitions - 1) / partitions;
  const int begin = partition * tokens_per_partition;
  const int end = min(context, begin + tokens_per_partition);
  const int dimension_base = lane * kValuesPerLane;

  float local_max = -CUDART_INF_F;
  float local_sum = 0.0f;
  float accumulator[kValuesPerLane] = {};
  for (int token = begin + warp; token < end; token += kWarps) {
    const uint32_t physical_block = block_tables[
        static_cast<size_t>(batch) * max_blocks + token / kPageSize];
    const size_t base = cache_offset(
        physical_block, kv_head, token % kPageSize, dimension_base);
    float score = 0.0f;
#pragma unroll
    for (int i = 0; i < kValuesPerLane; ++i) {
      score = fmaf(
          cache_to_float<Bf16Cache>(key_cache, base + i),
          shared_query[dimension_base + i],
          score);
    }
    score = __shfl_sync(0xffffffffu, warp_sum(score), 0) * softmax_scale;
    const float next_max = fmaxf(local_max, score);
    const float previous_weight = __expf(local_max - next_max);
    const float token_weight = __expf(score - next_max);
    local_sum = local_sum * previous_weight + token_weight;
#pragma unroll
    for (int i = 0; i < kValuesPerLane; ++i) {
      accumulator[i] = accumulator[i] * previous_weight +
                       token_weight * cache_to_float<Bf16Cache>(value_cache, base + i);
    }
    local_max = next_max;
  }

  if (lane == 0) {
    shared_max[warp] = local_max;
    shared_sum[warp] = local_sum;
  }
#pragma unroll
  for (int i = 0; i < kValuesPerLane; ++i) {
    shared_output[warp][dimension_base + i] = accumulator[i];
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float maximum = -CUDART_INF_F;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) {
      maximum = fmaxf(maximum, shared_max[w]);
    }
    float sum = 0.0f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) {
      if (shared_sum[w] > 0.0f) {
        sum += shared_sum[w] * __expf(shared_max[w] - maximum);
      }
    }
    merged_max = maximum;
    merged_sum = sum;
  }
  __syncthreads();

  if (warp != 0) {
    return;
  }
  float numerator[kValuesPerLane] = {};
#pragma unroll
  for (int w = 0; w < kWarps; ++w) {
    if (shared_sum[w] > 0.0f) {
      const float weight = __expf(shared_max[w] - merged_max);
#pragma unroll
      for (int i = 0; i < kValuesPerLane; ++i) {
        numerator[i] += shared_output[w][dimension_base + i] * weight;
      }
    }
  }

  if constexpr (Direct) {
#pragma unroll
    for (int i = 0; i < kValuesPerLane; ++i) {
      const float value =
          merged_sum > 0.0f ? numerator[i] / merged_sum : 0.0f;
      output[query_offset + dimension_base + i] = __float2bfloat16(
          value * output_gate(
                      query_gate_projection,
                      batch,
                      query_head,
                      dimension_base + i));
    }
  } else {
    const size_t partial =
        (static_cast<size_t>(batch) * kQueryHeads + query_head) * partitions +
        partition;
    if (lane == 0) {
      partial_max[partial] = merged_max;
      partial_sum[partial] = merged_sum;
    }
    const size_t output_base = partial * kHeadDim;
#pragma unroll
    for (int i = 0; i < kValuesPerLane; ++i) {
      partial_output[output_base + dimension_base + i] = numerator[i];
    }
  }
}

__global__ __launch_bounds__(kThreads) void reduce_partitions_kernel(
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum,
    const float* __restrict__ partial_output,
    const __nv_bfloat16* __restrict__ query_gate_projection,
    __nv_bfloat16* __restrict__ output,
    int partitions) {
  const int query_head = blockIdx.x;
  const int batch = blockIdx.y;
  const size_t first =
      (static_cast<size_t>(batch) * kQueryHeads + query_head) * partitions;

  __shared__ float maximum;
  __shared__ float denominator;
  if (threadIdx.x == 0) {
    float value = -CUDART_INF_F;
    for (int part = 0; part < partitions; ++part) {
      if (partial_sum[first + part] > 0.0f) {
        value = fmaxf(value, partial_max[first + part]);
      }
    }
    maximum = value;

    float sum = 0.0f;
    for (int part = 0; part < partitions; ++part) {
      if (partial_sum[first + part] > 0.0f) {
        sum += partial_sum[first + part] *
               __expf(partial_max[first + part] - value);
      }
    }
    denominator = sum;
  }
  __syncthreads();

  float numerator = 0.0f;
  for (int part = 0; part < partitions; ++part) {
    if (partial_sum[first + part] > 0.0f) {
      const float weight = __expf(partial_max[first + part] - maximum);
      numerator += partial_output[(first + part) * kHeadDim + threadIdx.x] * weight;
    }
  }
  const size_t output_offset =
      (static_cast<size_t>(batch) * kQueryHeads + query_head) * kHeadDim +
      threadIdx.x;
  const float value = denominator > 0.0f ? numerator / denominator : 0.0f;
  output[output_offset] = __float2bfloat16(
      value * output_gate(
                  query_gate_projection,
                  batch,
                  query_head,
                  threadIdx.x));
}

cudaError_t validate(
    const void* query,
    const void* key_cache,
    const void* value_cache,
    const void* block_tables,
    const void* context_lengths,
    const void* output,
    int batch,
    int max_blocks,
    int partitions,
    float softmax_scale) {
  if (query == nullptr || key_cache == nullptr || value_cache == nullptr ||
      block_tables == nullptr || context_lengths == nullptr || output == nullptr ||
      batch <= 0 || batch > qwc::kMaxDecodeRows || max_blocks <= 0 || partitions <= 0 ||
      !isfinite(softmax_scale) || softmax_scale <= 0.0f) {
    return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

template <bool Bf16Cache>
cudaError_t launch_paged_attention(
    const void* query,
    const void* query_gate_projection,
    const void* key_cache,
    const void* value_cache,
    const void* block_tables,
    const void* context_lengths,
    void* output,
    void* workspace,
    size_t workspace_bytes,
    int batch,
    int max_blocks,
    int partitions,
    int share_kv,
    float softmax_scale,
    cudaStream_t stream) {
  const cudaError_t valid = validate(
      query,
      key_cache,
      value_cache,
      block_tables,
      context_lengths,
      output,
      batch,
      max_blocks,
      partitions,
      softmax_scale);
  if (valid != cudaSuccess) {
    return valid;
  }

  dim3 grid(share_kv ? kKvHeads : kQueryHeads, batch, partitions);
  if (partitions == 1) {
    if (share_kv) {
      paged_attention_parts_kernel<Bf16Cache, true><<<grid, kThreads, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(query),
          static_cast<const __nv_bfloat16*>(query_gate_projection),
          key_cache,
          value_cache,
          static_cast<const uint32_t*>(block_tables),
          static_cast<const uint32_t*>(context_lengths),
          static_cast<__nv_bfloat16*>(output),
          nullptr, nullptr, nullptr, max_blocks, partitions, softmax_scale);
    } else {
      paged_attention_query_head_kernel<Bf16Cache, true><<<grid, kThreads, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(query),
          static_cast<const __nv_bfloat16*>(query_gate_projection),
          key_cache,
          value_cache,
          static_cast<const uint32_t*>(block_tables),
          static_cast<const uint32_t*>(context_lengths),
          static_cast<__nv_bfloat16*>(output),
          nullptr, nullptr, nullptr, max_blocks, partitions, softmax_scale);
    }
    return cudaGetLastError();
  }

  const size_t partials =
      static_cast<size_t>(batch) * kQueryHeads * partitions;
  const size_t required = partials * (kHeadDim + 2) * sizeof(float);
  if (workspace == nullptr || workspace_bytes < required) {
    return cudaErrorInvalidValue;
  }
  auto* partial_max = static_cast<float*>(workspace);
  auto* partial_sum = partial_max + partials;
  auto* partial_output = partial_sum + partials;
  if (share_kv) {
    paged_attention_parts_kernel<Bf16Cache, false><<<grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(query),
        static_cast<const __nv_bfloat16*>(query_gate_projection),
        key_cache,
        value_cache,
        static_cast<const uint32_t*>(block_tables),
        static_cast<const uint32_t*>(context_lengths),
        static_cast<__nv_bfloat16*>(output),
        partial_max, partial_sum, partial_output,
        max_blocks, partitions, softmax_scale);
  } else {
    paged_attention_query_head_kernel<Bf16Cache, false><<<grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(query),
        static_cast<const __nv_bfloat16*>(query_gate_projection),
        key_cache,
        value_cache,
        static_cast<const uint32_t*>(block_tables),
        static_cast<const uint32_t*>(context_lengths),
        static_cast<__nv_bfloat16*>(output),
        partial_max, partial_sum, partial_output,
        max_blocks, partitions, softmax_scale);
  }
  cudaError_t launched = cudaGetLastError();
  if (launched != cudaSuccess) {
    return launched;
  }
  dim3 reduction_grid(kQueryHeads, batch);
  reduce_partitions_kernel<<<reduction_grid, kThreads, 0, stream>>>(
      partial_max, partial_sum, partial_output,
      static_cast<const __nv_bfloat16*>(query_gate_projection),
      static_cast<__nv_bfloat16*>(output), partitions);
  return cudaGetLastError();
}

}  // namespace

namespace {

template <bool Bf16Cache>
cudaError_t launch_prefill_attention(
    const void* query,
    const void* query_gate_projection,
    const void* key_cache,
    const void* value_cache,
    const void* block_tables,
    const void* context_lengths,
    void* output,
    int rows,
    int row_base,
    int max_blocks,
    float softmax_scale,
    cudaStream_t stream) {
  if (query == nullptr || key_cache == nullptr || value_cache == nullptr ||
      block_tables == nullptr || context_lengths == nullptr ||
      output == nullptr || rows <= 0 || row_base < 0 || max_blocks <= 0) {
    return cudaErrorInvalidValue;
  }
  const dim3 grid(kKvHeads, (rows + kWarps - 1) / kWarps, 1);
  paged_attention_prefill_kernel<Bf16Cache><<<grid, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(query),
      static_cast<const __nv_bfloat16*>(query_gate_projection),
      key_cache,
      value_cache,
      static_cast<const uint32_t*>(block_tables),
      static_cast<const uint32_t*>(context_lengths),
      static_cast<__nv_bfloat16*>(output),
      max_blocks,
      rows,
      row_base,
      softmax_scale);
  return cudaGetLastError();
}

template <bool Bf16Cache>
cudaError_t launch_prefill_attention_mma(
    const void* query,
    const void* query_gate_projection,
    const void* key_cache,
    const void* value_cache,
    const void* block_tables,
    const void* context_lengths,
    void* output,
    int rows,
    int row_base,
    int max_blocks,
    float softmax_scale,
    cudaStream_t stream) {
  if (query == nullptr || key_cache == nullptr || value_cache == nullptr ||
      block_tables == nullptr || context_lengths == nullptr ||
      output == nullptr || rows <= 0 || row_base < 0 || max_blocks <= 0) {
    return cudaErrorInvalidValue;
  }
  const size_t shared = flash_shared_bytes();
  static const cudaError_t opted = cudaFuncSetAttribute(
      paged_attention_prefill_flash_kernel<Bf16Cache>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(shared));
  if (opted != cudaSuccess) {
    return opted;
  }
  const dim3 grid(kKvHeads, (rows + kFlashRows - 1) / kFlashRows, kGroup);
  paged_attention_prefill_flash_kernel<Bf16Cache>
      <<<grid, kFlashThreads, shared, stream>>>(
          static_cast<const __nv_bfloat16*>(query),
          static_cast<const __nv_bfloat16*>(query_gate_projection),
          key_cache,
          value_cache,
          static_cast<const uint32_t*>(block_tables),
          static_cast<const uint32_t*>(context_lengths),
          static_cast<__nv_bfloat16*>(output),
          max_blocks,
          rows,
          row_base,
          softmax_scale);
  return cudaGetLastError();
}

// Attention with a packed GQA group: a CTA per KV head, MMA tile rows are
// (query row, group head) pairs.
//
// The flash kernel above gives each query head its own CTA. KV is read six
// times there, and with four draft verification rows only four of the tile's 64 rows
// are occupied: 4 rows at 60k cost 1.47 ms per layer vs 90 µs for FlashInfer's
// xqa. Here all six heads of a group go as rows of one tile:
// packed row p is query row p / 6, head p % 6. Decode is
// 6 rows of 16, verifying three drafts is 24 of 32, and every KV byte
// is read once.
//
// KV sits in shared as is, in fp8. Before, the tile was converted to bf16 during
// staging, and the load ran synchronously with compute. Now a page (64
// keys) arrives via cp.async into a double buffer while the previous one is
// computed, and bytes are converted to f16 already in fragments.
//
// If there are fewer than four m-tiles, the extra warps split the stage's keys: each
// SMSP has its own tensor cores, and decode with one warp per SM would be bound by one of
// the four. The warps' partial sums are reduced in shared at the end.
// Warps per CTA: on a 64-row tile, two per m-tile. One CTA lives on an
// SM alone, and with one warp per scheduler the tensor cores sat idle while
// the warp computed softmax and converted fp8; the second warp fills these gaps.
template <int MTiles>
__host__ __device__ constexpr int packed_warps() {
  return MTiles == 4 ? 8 : 4;
}
constexpr int kPackedStageKeys = kPageSize;
constexpr int kPackedStageBytes = kPackedStageKeys * kHeadDim;
constexpr int kPackedChunks = kHeadDim / 16;   // 16-byte chunks in an fp8 row
constexpr int kPackedMaxTileSize = 64;

template <int MTiles, int Warps = packed_warps<MTiles>()>
struct PackedShared {
  union {
    struct {
      uint8_t key[2][kPackedStageBytes];
      uint8_t value[2][kPackedStageBytes];
    } ring;
    // Accumulators of the warps that split keys — after the last stage.
    float merge[(Warps / MTiles - 1 > 0 ? Warps - MTiles : 1) *
                32 * kFlashDimTiles * 4];
  };
  __half query[MTiles * 16 * kRowPad];
  int row_context[MTiles * 16];
  float merge_max[Warps][16];
  float merge_sum[Warps][16];
};

// A key row is 256 bytes, 16-byte chunks are permuted by the key's three low
// bits. ldmatrix reads eight rows at one chunk position, and without
// the permutation all eight would land in the same banks.
__device__ __forceinline__ int packed_swizzle(int key, int chunk) {
  return key * kHeadDim + ((chunk ^ (key & 7)) << 4);
}

__device__ __forceinline__ void cp_async_16(
    void* destination, const void* source, bool valid) {
  // An invalid chunk is filled with zeros: the tail of the last page may
  // hold anything, including fp8 NaN, and 0 * NaN in PV is NaN.
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
               :
               : "r"(shared_address(destination)), "l"(source),
                 "r"(valid ? 16 : 0));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int Pending>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(Pending) : "memory");
}

__device__ __forceinline__ void ldmatrix_x4_raw(
    uint32_t (&fragment)[4], const void* source) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(fragment[0]), "=r"(fragment[1]), "=r"(fragment[2]),
        "=r"(fragment[3])
      : "r"(shared_address(source)));
}

__device__ __forceinline__ void ldmatrix_x4_trans_raw(
    uint32_t (&fragment)[4], const void* source) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(fragment[0]), "=r"(fragment[1]), "=r"(fragment[2]),
        "=r"(fragment[3])
      : "r"(shared_address(source)));
}

// Two e4m3 from the low 16 bits into f16x2; the low byte goes to the low half.
// Any e4m3 is exactly representable in f16.
__device__ __forceinline__ uint32_t fp8x2_to_half2(uint32_t pair) {
  uint32_t result;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;\n"
      : "=r"(result)
      : "h"(static_cast<uint16_t>(pair)));
  return result;
}

__device__ __forceinline__ void mma_f16_m16n8k16(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ uint32_t pack_half2(float low, float high) {
  const __half2 value = __floats2half2_rn(low, high);
  return *reinterpret_cast<const uint32_t*>(&value);
}

// A tile is tile_size consecutive packed rows, at most MTiles * 16.
// A tile has one page table, that of the row of its first packed row,
// so a tile must not cross a sequence boundary. For decode
// tile_size = 6, one query row; a single-sequence segment
// is cut into 64 packed rows regardless of row boundaries: with
// row alignment 64 rows would give 7 tiles of 60 instead of 6 of 64.
//
// Fragment layouts.
// QK^T: operand B is K rows. ldmatrix over fp8 read as b16 gives
// a lane four consecutive bytes: dims 4pos..4pos+3 of key grp. The order of
// summation over dims is free if A is permuted the same way, so
// within a 16 of dims the logical k {2pos, 2pos+1, 2pos+8, 2pos+9} are
// physical 4pos..4pos+3. The query is put into shared already permuted.
// PV: operand B is V with the key along k. ldmatrix.trans over b16 gives a lane
// a key pair 2pos, 2pos+1 for a dim pair 2grp, 2grp+1; prmt splits
// the bytes into two n-tiles: even dims of the 16 and odd ones. As a result a lane
// holds four consecutive dims in the accumulator, 16c + 4pos..+3.
template <int MTiles, bool Direct>
__global__ __launch_bounds__(packed_warps<MTiles>() * 32, 1) void paged_attention_packed_kernel(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ query_gate_projection,
    const uint8_t* __restrict__ key_cache,
    const uint8_t* __restrict__ value_cache,
    const uint32_t* __restrict__ block_tables,
    const uint32_t* __restrict__ context_lengths,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    __half* __restrict__ partial_output,
    int max_blocks,
    int rows,
    int row_base,
    int tile_size,
    float softmax_scale) {
  constexpr int kPackedThreads = packed_warps<MTiles>() * 32;
  constexpr int kSlices = packed_warps<MTiles>() / MTiles;
  constexpr int kSliceKeys = kPackedStageKeys / kSlices;
  static_assert(kSliceKeys >= 16, "PV takes keys in groups of 16");
  constexpr int kChunkKeys = kSliceKeys < 32 ? kSliceKeys : 32;
  constexpr int kKeyTiles = kChunkKeys / 8;
  constexpr int kKeyGroups = kChunkKeys / 16;
  constexpr int kPackedRows = MTiles * 16;

  const int kv_head = blockIdx.x;
  const int tile = blockIdx.y;
  const int partition = blockIdx.z;
  const int partitions = gridDim.z;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int grp = lane >> 2;
  const int pos = lane & 3;
  const int m_tile = warp % MTiles;
  const int slice = warp / MTiles;
  const int row0 = m_tile * 16;

  // Packed row g of a segment is query row g / 6, head g % 6.
  const int tile_first = tile * tile_size;
  const int tile_count = min(tile_size, rows * kGroup - tile_first);

  extern __shared__ __align__(16) char packed_shared_raw[];
  auto& shared = *reinterpret_cast<PackedShared<MTiles>*>(packed_shared_raw);

  // The query in f16 with dims permuted within a 16. Eight physical
  // dims 4pp..4pp+3 for two pp are the logical pairs (2pp, 2pp+1) and
  // (2pp+8, 2pp+9).
  for (int i = threadIdx.x; i < kPackedRows * (kHeadDim / 8);
       i += kPackedThreads) {
    const int p = i / (kHeadDim / 8);
    const int octet = i % (kHeadDim / 8);
    const int packed = tile_first + p;
    uint4 raw = make_uint4(0u, 0u, 0u, 0u);
    if (p < tile_count) {
      const size_t base =
          (static_cast<size_t>(row_base + packed / kGroup) * kQueryHeads +
           kv_head * kGroup + packed % kGroup) *
              kHeadDim +
          octet * 8;
      raw = *reinterpret_cast<const uint4*>(query + base);
    }
    const __nv_bfloat16* values = reinterpret_cast<const __nv_bfloat16*>(&raw);
    __half* chunk = &shared.query[p * kRowPad + (octet >> 1) * 16];
#pragma unroll
    for (int q = 0; q < 2; ++q) {
      const int pp = 2 * (octet & 1) + q;
      *reinterpret_cast<uint32_t*>(chunk + 2 * pp) =
          pack_half2(__bfloat162float(values[4 * q]),
                     __bfloat162float(values[4 * q + 1]));
      *reinterpret_cast<uint32_t*>(chunk + 8 + 2 * pp) =
          pack_half2(__bfloat162float(values[4 * q + 2]),
                     __bfloat162float(values[4 * q + 3]));
    }
  }
  for (int p = threadIdx.x; p < kPackedRows; p += kPackedThreads) {
    shared.row_context[p] = p < tile_count
        ? static_cast<int>(context_lengths[row_base + (tile_first + p) / kGroup])
        : 0;
  }
  __syncthreads();

  int max_context = 0;
  int warp_context = 0;
#pragma unroll
  for (int p = 0; p < kPackedRows; ++p) {
    max_context = max(max_context, shared.row_context[p]);
    if (p >= row0 && p < row0 + 16) {
      warp_context = max(warp_context, shared.row_context[p]);
    }
  }
  const uint32_t* table =
      block_tables + static_cast<size_t>(row_base + tile_first / kGroup) * max_blocks;

  // The partition span is computed here, from the tile's actual context, and
  // is a multiple of a page. The host picks only the number of partitions: the captured
  // CUDA graph doesn't depend on context length, and decode rows of different lengths
  // each split their own.
  const int span =
      (max_context + partitions * kPageSize - 1) / (partitions * kPageSize) * kPageSize;
  const int key_begin = Direct ? 0 : partition * span;
  const int key_end = Direct ? max_context : min(max_context, key_begin + span);
  const int stages = key_end > key_begin
      ? (key_end - key_begin + kPackedStageKeys - 1) / kPackedStageKeys
      : 0;
  const int first_page = key_begin / kPageSize;

  auto issue = [&](int stage, uint32_t block) {
    const int buffer = stage & 1;
    const int key_base = key_begin + stage * kPackedStageKeys;
    const size_t slab =
        (static_cast<size_t>(block) * kKvHeads + kv_head) * kPackedStageBytes;
#pragma unroll
    for (int i = threadIdx.x; i < kPackedStageKeys * kPackedChunks;
         i += kPackedThreads) {
      const int key = i / kPackedChunks;
      const int at = packed_swizzle(key, i % kPackedChunks);
      const bool valid = key_base + key < key_end;
      cp_async_16(&shared.ring.key[buffer][at], key_cache + slab + i * 16, valid);
      cp_async_16(&shared.ring.value[buffer][at], value_cache + slab + i * 16, valid);
    }
    cp_async_commit();
  };

  // The next stage's page number is read one stage ahead: otherwise the dependent
  // load from the table would stand before every cp.async issue.
  if (stages > 0) {
    issue(0, table[first_page]);
  }
  uint32_t next_block = stages > 1 ? table[first_page + 1] : 0u;

  float accumulator[kFlashDimTiles][4] = {};
  float running_max[2] = {-CUDART_INF_F, -CUDART_INF_F};
  float running_sum[2] = {0.0f, 0.0f};

  const int matrix = lane >> 3;
  const int matrix_row = lane & 7;
  const __half* query_row =
      &shared.query[(row0 + matrix_row + 8 * (matrix & 1)) * kRowPad +
                    8 * (matrix >> 1)];

  for (int stage = 0; stage < stages; ++stage) {
    // The next stage's buffer was read in the previous stage: all warps must
    // finish it before cp.async overwrites it.
    __syncthreads();
    if (stage + 1 < stages) {
      issue(stage + 1, next_block);
      next_block = stage + 2 < stages ? table[first_page + stage + 2] : 0u;
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();

    const uint8_t* keys = shared.ring.key[stage & 1];
    const uint8_t* values = shared.ring.value[stage & 1];
    const int stage_base = key_begin + stage * kPackedStageKeys;
#pragma unroll
    for (int chunk_key = slice * kSliceKeys;
         chunk_key < (slice + 1) * kSliceKeys; chunk_key += kChunkKeys) {
      const int key_base = stage_base + chunk_key;
      if (key_base >= warp_context) {
        break;   // the warp will do its barriers, nothing for it to compute
      }

      float scores[kKeyTiles][4] = {};
#pragma unroll
      for (int k4 = 0; k4 < kPackedChunks; k4 += 4) {
        uint32_t a[4][4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          ldmatrix_x4_raw(a[j], query_row + (k4 + j) * 16);
        }
#pragma unroll
        for (int t = 0; t < kKeyTiles; ++t) {
          // Matrix j is the 16 of dims k4 + j for eight keys of tile t.
          uint32_t packed[4];
          ldmatrix_x4_raw(
              packed,
              keys + packed_swizzle(chunk_key + t * 8 + matrix_row, k4 + matrix));
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const uint32_t b[2] = {fp8x2_to_half2(packed[j]),
                                   fp8x2_to_half2(packed[j] >> 16)};
            mma_f16_m16n8k16(scores[t], a[j], b);
          }
        }
      }

      // Scale and causal mask. A lane holds rows grp and grp+8.
      const int context_low = shared.row_context[row0 + grp];
      const int context_high = shared.row_context[row0 + grp + 8];
#pragma unroll
      for (int t = 0; t < kKeyTiles; ++t) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int token = key_base + t * 8 + pos * 2 + (i & 1);
          const int limit = (i >= 2) ? context_high : context_low;
          scores[t][i] =
              token < limit ? scores[t][i] * softmax_scale : -CUDART_INF_F;
        }
      }

      float tile_max[2] = {-CUDART_INF_F, -CUDART_INF_F};
#pragma unroll
      for (int t = 0; t < kKeyTiles; ++t) {
        tile_max[0] = fmaxf(tile_max[0], fmaxf(scores[t][0], scores[t][1]));
        tile_max[1] = fmaxf(tile_max[1], fmaxf(scores[t][2], scores[t][3]));
      }
      tile_max[0] = flash_row_max(tile_max[0]);
      tile_max[1] = flash_row_max(tile_max[1]);

      float correction[2];
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const float merged = fmaxf(running_max[half], tile_max[half]);
        const bool empty = merged == -CUDART_INF_F;
        correction[half] = (empty || running_max[half] == -CUDART_INF_F)
            ? (empty ? 1.0f : 0.0f)
            : __expf(running_max[half] - merged);
        float sum = 0.0f;
#pragma unroll
        for (int t = 0; t < kKeyTiles; ++t) {
#pragma unroll
          for (int i = 0; i < 2; ++i) {
            float& slot = scores[t][half * 2 + i];
            const float weight =
                (empty || slot == -CUDART_INF_F) ? 0.0f : __expf(slot - merged);
            slot = weight;
            sum += weight;
          }
        }
        sum = flash_row_sum(sum);
        if (!empty) {
          running_sum[half] = running_sum[half] * correction[half] + sum;
          running_max[half] = merged;
        }
      }
#pragma unroll
      for (int t = 0; t < kFlashDimTiles; ++t) {
        accumulator[t][0] *= correction[0];
        accumulator[t][1] *= correction[0];
        accumulator[t][2] *= correction[1];
        accumulator[t][3] *= correction[1];
      }

      uint32_t probability[kKeyGroups][4];
#pragma unroll
      for (int g = 0; g < kKeyGroups; ++g) {
        probability[g][0] = pack_half2(scores[2 * g][0], scores[2 * g][1]);
        probability[g][1] = pack_half2(scores[2 * g][2], scores[2 * g][3]);
        probability[g][2] = pack_half2(scores[2 * g + 1][0], scores[2 * g + 1][1]);
        probability[g][3] = pack_half2(scores[2 * g + 1][2], scores[2 * g + 1][3]);
      }

#pragma unroll
      for (int g = 0; g < kKeyGroups; ++g) {
        // Matrices: keys 0-7 and 8-15 of the group for dim 16s c and c+1.
        const int key = chunk_key + g * 16 + matrix_row + 8 * (matrix & 1);
#pragma unroll
        for (int c = 0; c < kPackedChunks; c += 2) {
          uint32_t packed[4];
          ldmatrix_x4_trans_raw(packed, values + packed_swizzle(key, c + (matrix >> 1)));
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const uint32_t low = packed[2 * h];
            const uint32_t high = packed[2 * h + 1];
            const uint32_t even[2] = {fp8x2_to_half2(__byte_perm(low, 0u, 0x20)),
                                      fp8x2_to_half2(__byte_perm(high, 0u, 0x20))};
            const uint32_t odd[2] = {fp8x2_to_half2(__byte_perm(low, 0u, 0x31)),
                                     fp8x2_to_half2(__byte_perm(high, 0u, 0x31))};
            mma_f16_m16n8k16(accumulator[2 * (c + h)], probability[g], even);
            mma_f16_m16n8k16(accumulator[2 * (c + h) + 1], probability[g], odd);
          }
        }
      }
    }
  }

  if constexpr (kSlices > 1) {
    // The ring is free: all stages are computed, cp.async has been waited on. Warps
    // of slices 1.. put their sums, the slice-0 warp reduces them into its own.
    __syncthreads();
    if (slice > 0) {
      float* dump =
          shared.merge + ((slice - 1) * MTiles + m_tile) * 32 * kFlashDimTiles * 4;
#pragma unroll
      for (int t = 0; t < kFlashDimTiles; ++t) {
        *reinterpret_cast<float4*>(&dump[(t * 32 + lane) * 4]) = make_float4(
            accumulator[t][0], accumulator[t][1], accumulator[t][2],
            accumulator[t][3]);
      }
      if (pos == 0) {
        shared.merge_max[warp][grp] = running_max[0];
        shared.merge_max[warp][grp + 8] = running_max[1];
        shared.merge_sum[warp][grp] = running_sum[0];
        shared.merge_sum[warp][grp + 8] = running_sum[1];
      }
    }
    __syncthreads();
    if (slice > 0) {
      return;
    }
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int r = grp + 8 * half;
      float merged = running_sum[half] > 0.0f ? running_max[half] : -CUDART_INF_F;
#pragma unroll
      for (int s = 1; s < kSlices; ++s) {
        const int other = s * MTiles + m_tile;
        if (shared.merge_sum[other][r] > 0.0f) {
          merged = fmaxf(merged, shared.merge_max[other][r]);
        }
      }
      if (merged == -CUDART_INF_F) {
        continue;   // the row is empty in all slices
      }
      const float own =
          running_sum[half] > 0.0f ? __expf(running_max[half] - merged) : 0.0f;
      float sum = running_sum[half] * own;
#pragma unroll
      for (int t = 0; t < kFlashDimTiles; ++t) {
        accumulator[t][2 * half] *= own;
        accumulator[t][2 * half + 1] *= own;
      }
#pragma unroll
      for (int s = 1; s < kSlices; ++s) {
        const int other = s * MTiles + m_tile;
        const float other_sum = shared.merge_sum[other][r];
        if (other_sum <= 0.0f) {
          continue;
        }
        const float scale = __expf(shared.merge_max[other][r] - merged);
        sum += other_sum * scale;
        const float* dump =
            shared.merge + ((s - 1) * MTiles + m_tile) * 32 * kFlashDimTiles * 4;
#pragma unroll
        for (int t = 0; t < kFlashDimTiles; ++t) {
          const float2 value = *reinterpret_cast<const float2*>(
              &dump[(t * 32 + lane) * 4 + 2 * half]);
          accumulator[t][2 * half] += value.x * scale;
          accumulator[t][2 * half + 1] += value.y * scale;
        }
      }
      running_max[half] = merged;
      running_sum[half] = sum;
    }
  }

#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const int r = row0 + grp + 8 * half;
    if (r >= tile_count) {
      continue;
    }
    const int local = (tile_first + r) / kGroup;
    const int query_head = kv_head * kGroup + (tile_first + r) % kGroup;
    if constexpr (!Direct) {
      // Partition rows are numbered from the segment start: row_base is added by
      // the reduction when it writes the output. The partition output goes out already
      // normalized, in f16: it's a convex combination of V, at most 448 in magnitude,
      // and f16 holds it with range to spare and three bits more precisely than the
      // bf16 final output. Sum and max stay in fp32.
      const size_t slot =
          (static_cast<size_t>(local) * kQueryHeads + query_head) *
              partitions +
          partition;
      if (pos == 0) {
        partial_max[slot] = running_max[half];
        partial_sum[slot] = running_sum[half];
      }
      const float inverse =
          running_sum[half] > 0.0f ? 1.0f / running_sum[half] : 0.0f;
#pragma unroll
      for (int c = 0; c < kPackedChunks; ++c) {
        *reinterpret_cast<uint2*>(
            &partial_output[slot * kHeadDim + c * 16 + pos * 4]) =
            make_uint2(pack_half2(accumulator[2 * c][2 * half] * inverse,
                                  accumulator[2 * c + 1][2 * half] * inverse),
                       pack_half2(accumulator[2 * c][2 * half + 1] * inverse,
                                  accumulator[2 * c + 1][2 * half + 1] * inverse));
      }
    } else {
      const int batch_row = row_base + local;
      const float inverse =
          running_sum[half] > 0.0f ? 1.0f / running_sum[half] : 0.0f;
      const size_t row_offset =
          (static_cast<size_t>(batch_row) * kQueryHeads + query_head) * kHeadDim;
#pragma unroll
      for (int c = 0; c < kPackedChunks; ++c) {
        const int dimension = c * 16 + pos * 4;
        float value[4] = {accumulator[2 * c][2 * half],
                          accumulator[2 * c + 1][2 * half],
                          accumulator[2 * c][2 * half + 1],
                          accumulator[2 * c + 1][2 * half + 1]};
        if (query_gate_projection != nullptr) {
          const uint2 raw = *reinterpret_cast<const uint2*>(
              query_gate_projection + row_offset * 2 + kHeadDim + dimension);
          const __nv_bfloat16* gate = reinterpret_cast<const __nv_bfloat16*>(&raw);
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            value[e] *= 1.0f / (1.0f + __expf(-__bfloat162float(gate[e])));
          }
        }
        __nv_bfloat16 packed[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          packed[e] = __float2bfloat16(value[e] * inverse);
        }
        *reinterpret_cast<uint2*>(output + row_offset + dimension) =
            *reinterpret_cast<const uint2*>(packed);
      }
    }
  }
}

__device__ __forceinline__ float warp_all_max(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
  }
  return value;
}

__device__ __forceinline__ float warp_all_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  return value;
}

// Partition reduction of the packed kernel. A partition's output is stored normalized in
// f16, so the partition weight is its share of the denominator, sum * exp(max - M) / D.
// A warp per (row, head) pair, a lane handles eight consecutive dims: one
// 16-byte read per partition. The shared reduce_partitions_kernel has a thread per
// dim and a sequential pass of thread zero over the weights.
__global__ __launch_bounds__(kThreads) void reduce_packed_partitions_kernel(
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum,
    const __half* __restrict__ partial_output,
    const __nv_bfloat16* __restrict__ query_gate_projection,
    __nv_bfloat16* __restrict__ output,
    int pairs,
    int partitions,
    int row_base) {
  const int pair = blockIdx.x * kWarps + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (pair >= pairs) {
    return;
  }
  const size_t first = static_cast<size_t>(pair) * partitions;
  const int output_row = row_base + pair / kQueryHeads;
  const int query_head = pair % kQueryHeads;
  const size_t row_offset =
      (static_cast<size_t>(output_row) * kQueryHeads + query_head) * kHeadDim;
  // The gate doesn't depend on partitions, and its read goes along with the first ones: on
  // decode the reduction is a chain of a couple of L2 accesses, and a third at the end
  // would cost a noticeable share of its time.
  uint4 gate_raw = make_uint4(0u, 0u, 0u, 0u);
  if (query_gate_projection != nullptr) {
    gate_raw = *reinterpret_cast<const uint4*>(
        query_gate_projection + row_offset * 2 + kHeadDim + lane * 8);
  }

  float maximum = -CUDART_INF_F;
  for (int part = lane; part < partitions; part += 32) {
    if (partial_sum[first + part] > 0.0f) {
      maximum = fmaxf(maximum, partial_max[first + part]);
    }
  }
  maximum = warp_all_max(maximum);
  float denominator = 0.0f;
  for (int part = lane; part < partitions; part += 32) {
    const float sum = partial_sum[first + part];
    if (sum > 0.0f) {
      denominator += sum * __expf(partial_max[first + part] - maximum);
    }
  }
  denominator = warp_all_sum(denominator);
  const float inverse = denominator > 0.0f ? 1.0f / denominator : 0.0f;

  float value[8] = {};
  for (int base = 0; base < partitions; base += 32) {
    float weight = 0.0f;
    if (base + lane < partitions) {
      const float sum = partial_sum[first + base + lane];
      weight = sum > 0.0f
          ? sum * __expf(partial_max[first + base + lane] - maximum) * inverse
          : 0.0f;
    }
    // No branch for an empty partition: it writes exact zeros, and the branch prevents
    // the compiler from issuing the unrolled loop's loads at once, and draft
    // verification at P=32 waited for 32 chained reads.
    const int count = min(32, partitions - base);
    const __half* source = partial_output + (first + base) * kHeadDim + lane * 8;
#pragma unroll 16
    for (int j = 0; j < count; ++j) {
      const float share = __shfl_sync(0xffffffffu, weight, j);
      const uint4 raw = *reinterpret_cast<const uint4*>(source + j * kHeadDim);
      const __half2* halves = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 pairwise = __half22float2(halves[e]);
        value[2 * e] += pairwise.x * share;
        value[2 * e + 1] += pairwise.y * share;
      }
    }
  }

  if (query_gate_projection != nullptr) {
    const __nv_bfloat16* gate = reinterpret_cast<const __nv_bfloat16*>(&gate_raw);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      value[e] *= 1.0f / (1.0f + __expf(-__bfloat162float(gate[e])));
    }
  }
  __nv_bfloat16 packed[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    packed[e] = __float2bfloat16(value[e]);
  }
  *reinterpret_cast<uint4*>(output + row_offset + lane * 8) =
      *reinterpret_cast<const uint4*>(packed);
}

// Partial-sum bytes per (query row x head, partition) pair: max and sum
// in fp32 and the normalized output in f16.
constexpr size_t kPackedPartialBytes = 2 * sizeof(float) + kHeadDim * sizeof(__half);

template <int MTiles>
cudaError_t launch_packed_attention(
    const void* query,
    const void* query_gate_projection,
    const void* key_cache,
    const void* value_cache,
    const void* block_tables,
    const void* context_lengths,
    void* output,
    void* workspace,
    size_t workspace_bytes,
    int rows,
    int row_base,
    int max_blocks,
    int tile_size,
    int partitions,
    float softmax_scale,
    cudaStream_t stream) {
  const size_t shared = sizeof(PackedShared<MTiles>);
  static const cudaError_t opted_direct = cudaFuncSetAttribute(
      paged_attention_packed_kernel<MTiles, true>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared));
  static const cudaError_t opted_split = cudaFuncSetAttribute(
      paged_attention_packed_kernel<MTiles, false>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared));
  if (opted_direct != cudaSuccess) {
    return opted_direct;
  }
  if (opted_split != cudaSuccess) {
    return opted_split;
  }
  const dim3 grid(kKvHeads, (rows * kGroup + tile_size - 1) / tile_size, partitions);
  if (partitions == 1) {
    paged_attention_packed_kernel<MTiles, true>
        <<<grid, packed_warps<MTiles>() * 32, shared, stream>>>(
            static_cast<const __nv_bfloat16*>(query),
            static_cast<const __nv_bfloat16*>(query_gate_projection),
            static_cast<const uint8_t*>(key_cache),
            static_cast<const uint8_t*>(value_cache),
            static_cast<const uint32_t*>(block_tables),
            static_cast<const uint32_t*>(context_lengths),
            static_cast<__nv_bfloat16*>(output),
            nullptr, nullptr, nullptr,
            max_blocks, rows, row_base, tile_size, softmax_scale);
    return cudaGetLastError();
  }

  const size_t partials = static_cast<size_t>(rows) * kQueryHeads * partitions;
  const size_t required = partials * kPackedPartialBytes;
  if (workspace == nullptr || workspace_bytes < required) {
    return cudaErrorInvalidValue;
  }
  // partials is a multiple of 24, and the f16 output starts at a 16-byte boundary.
  auto* partial_max = static_cast<float*>(workspace);
  auto* partial_sum = partial_max + partials;
  auto* partial_output = reinterpret_cast<__half*>(partial_sum + partials);
  paged_attention_packed_kernel<MTiles, false>
      <<<grid, packed_warps<MTiles>() * 32, shared, stream>>>(
          static_cast<const __nv_bfloat16*>(query),
          static_cast<const __nv_bfloat16*>(query_gate_projection),
          static_cast<const uint8_t*>(key_cache),
          static_cast<const uint8_t*>(value_cache),
          static_cast<const uint32_t*>(block_tables),
          static_cast<const uint32_t*>(context_lengths),
          static_cast<__nv_bfloat16*>(output),
          partial_max, partial_sum, partial_output,
          max_blocks, rows, row_base, tile_size, softmax_scale);
  const cudaError_t launched = cudaGetLastError();
  if (launched != cudaSuccess) {
    return launched;
  }
  const int pairs = rows * kQueryHeads;
  reduce_packed_partitions_kernel<<<(pairs + kWarps - 1) / kWarps, kThreads, 0, stream>>>(
      partial_max, partial_sum, partial_output,
      static_cast<const __nv_bfloat16*>(query_gate_projection),
      static_cast<__nv_bfloat16*>(output), pairs, partitions, row_base);
  return cudaGetLastError();
}

}  // namespace

// Packed attention over the fp8 cache. tile_size is the number of packed rows (query
// row x group head) in a tile: 6 for decode rows of different
// sequences, up to 64 for a segment of one.
extern "C" cudaError_t qwc_paged_attention_packed_fp8(
    const void* query,
    const void* query_gate_projection,
    const void* key_cache,
    const void* value_cache,
    const void* block_tables,
    const void* context_lengths,
    void* output,
    void* workspace,
    size_t workspace_bytes,
    int rows,
    int row_base,
    int max_blocks,
    int tile_size,
    int partitions,
    float softmax_scale,
    cudaStream_t stream) {
  if (query == nullptr || key_cache == nullptr || value_cache == nullptr ||
      block_tables == nullptr || context_lengths == nullptr ||
      output == nullptr || rows <= 0 || rows > qwc::kMaxStepRows ||
      row_base < 0 || max_blocks <= 0 || tile_size <= 0 ||
      tile_size > kPackedMaxTileSize || partitions <= 0 ||
      partitions > 65535 || !isfinite(softmax_scale) ||
      softmax_scale <= 0.0f) {
    return cudaErrorInvalidValue;
  }
  if (tile_size <= 16) {
    return launch_packed_attention<1>(
        query, query_gate_projection, key_cache, value_cache, block_tables,
        context_lengths, output, workspace, workspace_bytes, rows, row_base,
        max_blocks, tile_size, partitions, softmax_scale,
        stream);
  }
  if (tile_size <= 32) {
    return launch_packed_attention<2>(
        query, query_gate_projection, key_cache, value_cache, block_tables,
        context_lengths, output, workspace, workspace_bytes, rows, row_base,
        max_blocks, tile_size, partitions, softmax_scale,
        stream);
  }
  return launch_packed_attention<4>(
      query, query_gate_projection, key_cache, value_cache, block_tables,
      context_lengths, output, workspace, workspace_bytes, rows, row_base,
      max_blocks, tile_size, partitions, softmax_scale,
      stream);
}

extern "C" cudaError_t qwc_paged_attention_prefill_mma_bf16(
    const void* query,
    const void* query_gate_projection,
    const void* key_cache,
    const void* value_cache,
    const void* block_tables,
    const void* context_lengths,
    void* output,
    int rows,
    int row_base,
    int max_blocks,
    float softmax_scale,
    cudaStream_t stream) {
  return launch_prefill_attention_mma<true>(
      query, query_gate_projection, key_cache, value_cache, block_tables,
      context_lengths, output, rows, row_base, max_blocks, softmax_scale,
      stream);
}

extern "C" cudaError_t qwc_paged_attention_prefill_mma_fp8(
    const void* query,
    const void* query_gate_projection,
    const void* key_cache,
    const void* value_cache,
    const void* block_tables,
    const void* context_lengths,
    void* output,
    int rows,
    int row_base,
    int max_blocks,
    float softmax_scale,
    cudaStream_t stream) {
  return launch_prefill_attention_mma<false>(
      query, query_gate_projection, key_cache, value_cache, block_tables,
      context_lengths, output, rows, row_base, max_blocks, softmax_scale,
      stream);
}

extern "C" cudaError_t qwc_paged_attention_prefill_bf16(
    const void* query,
    const void* query_gate_projection,
    const void* key_cache,
    const void* value_cache,
    const void* block_tables,
    const void* context_lengths,
    void* output,
    int rows,
    int row_base,
    int max_blocks,
    float softmax_scale,
    cudaStream_t stream) {
  return launch_prefill_attention<true>(
      query, query_gate_projection, key_cache, value_cache, block_tables,
      context_lengths, output, rows, row_base, max_blocks, softmax_scale,
      stream);
}

extern "C" cudaError_t qwc_paged_attention_prefill_fp8(
    const void* query,
    const void* query_gate_projection,
    const void* key_cache,
    const void* value_cache,
    const void* block_tables,
    const void* context_lengths,
    void* output,
    int rows,
    int row_base,
    int max_blocks,
    float softmax_scale,
    cudaStream_t stream) {
  return launch_prefill_attention<false>(
      query, query_gate_projection, key_cache, value_cache, block_tables,
      context_lengths, output, rows, row_base, max_blocks, softmax_scale,
      stream);
}

extern "C" cudaError_t qwc_paged_attention_fp8(
    const void* query,
    const void* query_gate_projection,
    const void* key_cache,
    const void* value_cache,
    const void* block_tables,
    const void* context_lengths,
    void* output,
    void* workspace,
    size_t workspace_bytes,
    int batch,
    int max_blocks,
    int partitions,
    int share_kv,
    float softmax_scale,
    cudaStream_t stream) {
  return launch_paged_attention<false>(
      query, query_gate_projection, key_cache, value_cache, block_tables,
      context_lengths, output, workspace, workspace_bytes, batch, max_blocks,
      partitions, share_kv, softmax_scale, stream);
}

extern "C" cudaError_t qwc_paged_attention_bf16(
    const void* query,
    const void* query_gate_projection,
    const void* key_cache,
    const void* value_cache,
    const void* block_tables,
    const void* context_lengths,
    void* output,
    void* workspace,
    size_t workspace_bytes,
    int batch,
    int max_blocks,
    int partitions,
    int share_kv,
    float softmax_scale,
    cudaStream_t stream) {
  return launch_paged_attention<true>(
      query, query_gate_projection, key_cache, value_cache, block_tables,
      context_lengths, output, workspace, workspace_bytes, batch, max_blocks,
      partitions, share_kv, softmax_scale, stream);
}

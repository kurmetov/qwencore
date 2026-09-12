// FP8 paged-attention decode for the 16 full-attention layers of Qwen3.8.
//
// One CTA owns one KV head and all six query heads in its GQA group. Each
// cached K/V element is therefore loaded once instead of six times. Long
// contexts are partitioned across grid.z and merged by a second kernel.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>

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

// Causal prefill: одно CTA владеет тайлом строк запроса одной последовательности,
// варп на строку. Decode-ядро делит токены между варпами, поэтому обслуживает
// одну строку за проход и на чанке префилла перечитывает те же страницы KV
// столько раз, сколько в чанке токенов. Здесь проход по KV один на тайл, а
// строки тайла переиспользуют уже прочитанные K и V: трафик падает в kWarps раз
// при том же регистровом бюджете, потому что варпу по-прежнему принадлежит
// ровно один онлайн-softmax.
//
// Межварповое слияние не нужно: аккумулятор варпа полон для своей строки, и
// shared_output прежнего ядра (48 KiB) здесь не существует.
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

  // Запросы тайла держим в bf16: во float это 48 KiB на блок, а конвертация
  // при чтении стоит одну инструкцию.
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

  // Дальше барьеров нет, поэтому лишние варпы просто уходят.
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
      batch <= 0 || batch > 1024 || max_blocks <= 0 || partitions <= 0 ||
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

}  // namespace

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

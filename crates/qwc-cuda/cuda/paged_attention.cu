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

// Causal prefill на тензорных ядрах.
//
// Построчные ядра — и decode-, и тайловое — считают QK^T и PV варповыми
// редукциями: на пару (строка, токен) приходится шесть проходов по бабочке, и
// фаза упирается в выдачу инструкций, а не в память. Здесь обе матрицы идут
// через mma.m16n8k16, то есть одна инструкция на 16x8x16.
//
// CTA — четыре варпа, тайл из kMmaRows строк запроса одной головы группы.
// KV-тайл кладётся в shared: K как [токен][dim], V транспонированно
// [dim][токен]. Так у обоих операндов B соседние по k элементы лежат рядом и
// фрагмент читается одним 32-битным словом.
//
// Тайл строк обязан принадлежать одной последовательности: таблица страниц
// берётся у первой строки тайла. Для чанка префилла это так по построению.
constexpr int kMmaWarps = 4;
constexpr int kMmaThreads = kMmaWarps * 32;
constexpr int kMmaRows = 16;                              // M
constexpr int kMmaKeys = 32;                              // KV-тайл
constexpr int kMmaDimPerWarp = kHeadDim / kMmaWarps;      // 64
constexpr int kMmaTilesPerWarp = kMmaDimPerWarp / 8;      // 8 n-тайлов
// Фрагмент читается дорожкой по адресу (grp * ряд + pos * 2): без набивки
// grp уходит кратно 32 банкам и весь варп садится на четыре банка. Набивка
// делает шаг ряда нечётным в банках и разводит дорожки по всем 32.
constexpr int kRowPad = kHeadDim + 8;     // ряд q_tile, k_tile и v_tile
constexpr int kProbPad = kMmaKeys + 8;    // ряд p_tile
// Сколько элементов кладёт в shared один поток за раз. Поэлементная укладка
// KV стоила больше инструкций, чем все MMA тайла вместе взятые.
constexpr int kStageVector = 8;
// Сколько голов группы обслуживает одна CTA на общем KV-тайле. kGroup = 6,
// поэтому делители: 1, 2, 3, 6.
constexpr int kMmaGroups = 1;
// Блоков на SM, которые ptxas обязан уместить. Без этого числа он при
// динамической shared считает, что блоков влезет много, и экономит регистры
// в ущерб параллелизму инструкций.
constexpr int kMmaBlocksPerSm = 2;

// Раскладка shared одним типом: смещения полей — константы, и доступ
// компилируется так же, как к статическим массивам.
struct MmaShared {
  __nv_bfloat16 query[kMmaGroups * kMmaRows * kRowPad];
  __nv_bfloat16 key[kMmaKeys * kRowPad];
  __nv_bfloat16 value[kMmaKeys * kRowPad];
  __nv_bfloat16 probability[kMmaGroups * kMmaRows * kProbPad];
  float score[kMmaGroups * kMmaRows * kMmaKeys];
  float row_max[kMmaGroups * kMmaRows];
  float row_sum[kMmaGroups * kMmaRows];
  float row_scale[kMmaGroups * kMmaRows];
  int row_context[kMmaRows];
};

constexpr size_t mma_shared_bytes() {
  return sizeof(MmaShared);
}

__device__ __forceinline__ uint32_t pack_bf16(const __nv_bfloat16* source) {
  return *reinterpret_cast<const uint32_t*>(source);
}

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

template <bool Bf16Cache>
// Второй аргумент обязателен: shared здесь динамическая, её размер ptxas на
// этапе компиляции не видит, решает что блоков влезет много и экономит
// регистры — теряя параллелизм инструкций. Блоков на SM всё равно не больше
// двух-трёх, и бюджет регистров надо назвать явно.
__global__ __launch_bounds__(kMmaThreads, kMmaBlocksPerSm) void paged_attention_prefill_mma_kernel(
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
  const int head_base = blockIdx.z * kMmaGroups;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  // Раскладка фрагментов m16n8k16: дорожка держит строки grp и grp+8, а по
  // столбцам — пару, начинающуюся с pos*2.
  const int grp = lane >> 2;
  const int pos = lane & 3;

  const int tile_first = tile * kMmaRows;
  const int tile_rows = min(kMmaRows, rows - tile_first);
  if (tile_rows <= 0) {
    return;
  }

  // Shared динамическая: при kMmaGroups > 1 раскладка уже не влезает в
  // статические 48 КБ на блок. Разложена структурой, а не указательной
  // арифметикой: так смещения остаются константами времени компиляции, а
  // выравнивание — видимым компилятору.
  extern __shared__ __align__(16) char mma_shared[];
  MmaShared& shared = *reinterpret_cast<MmaShared*>(mma_shared);
  __nv_bfloat16* const q_tile = shared.query;
  __nv_bfloat16* const k_tile = shared.key;
  __nv_bfloat16* const v_tile = shared.value;
  __nv_bfloat16* const p_tile = shared.probability;
  float* const s_tile = shared.score;
  float* const row_max = shared.row_max;
  float* const row_sum = shared.row_sum;
  float* const row_scale = shared.row_scale;
  int* const row_context = shared.row_context;

  for (int i = threadIdx.x; i < kMmaGroups * kMmaRows * kHeadDim;
       i += kMmaThreads) {
    const int dimension = i % kHeadDim;
    const int slot = i / kHeadDim;
    const int r = slot % kMmaRows;
    const int g = slot / kMmaRows;
    __nv_bfloat16 value = __float2bfloat16(0.0f);
    if (r < tile_rows) {
      const size_t base =
          (static_cast<size_t>(row_base + tile_first + r) * kQueryHeads +
           kv_head * kGroup + head_base + g) * kHeadDim;
      value = query[base + dimension];
    }
    q_tile[slot * kRowPad + dimension] = value;
  }
  for (int i = threadIdx.x; i < kMmaGroups * kMmaRows; i += kMmaThreads) {
    row_max[i] = -CUDART_INF_F;
    row_sum[i] = 0.0f;
  }
  if (threadIdx.x < kMmaRows) {
    const int r = threadIdx.x;
    row_context[r] = r < tile_rows
        ? static_cast<int>(context_lengths[row_base + tile_first + r])
        : 0;
  }
  __syncthreads();

  int max_context = 0;
#pragma unroll
  for (int r = 0; r < kMmaRows; ++r) {
    max_context = max(max_context, row_context[r]);
  }

  // Таблица страниц общая на тайл: строки чанка принадлежат одной
  // последовательности.
  const uint32_t* table =
      block_tables + static_cast<size_t>(row_base + tile_first) * max_blocks;

  float accumulator[kMmaGroups][kMmaTilesPerWarp][4];
#pragma unroll
  for (int g = 0; g < kMmaGroups; ++g) {
#pragma unroll
    for (int t = 0; t < kMmaTilesPerWarp; ++t) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        accumulator[g][t][i] = 0.0f;
      }
    }
  }

  for (int key_base = 0; key_base < max_context; key_base += kMmaKeys) {
    __syncthreads();
    // Уложенный тайл обслуживает все kMmaGroups голов группы: раньше его
    // грузила каждая из шести CTA по отдельности, и это и была основная
    // статья расхода ядра.
    for (int i = threadIdx.x * kStageVector; i < kMmaKeys * kHeadDim;
         i += kMmaThreads * kStageVector) {
      const int key = i / kHeadDim;
      const int dimension = i - key * kHeadDim;
      const int token = key_base + key;
      if (token < max_context) {
        const uint32_t block = table[token / kPageSize];
        const size_t offset =
            cache_offset(block, kv_head, token % kPageSize, dimension);
#pragma unroll
        for (int e = 0; e < kStageVector; ++e) {
          k_tile[key * kRowPad + dimension + e] = __float2bfloat16(
              cache_to_float<Bf16Cache>(key_cache, offset + e));
          v_tile[key * kRowPad + dimension + e] = __float2bfloat16(
              cache_to_float<Bf16Cache>(value_cache, offset + e));
        }
      } else {
#pragma unroll
        for (int e = 0; e < kStageVector; ++e) {
          k_tile[key * kRowPad + dimension + e] = __float2bfloat16(0.0f);
          v_tile[key * kRowPad + dimension + e] = __float2bfloat16(0.0f);
        }
      }
    }
    __syncthreads();

    // QK^T: варп берёт свои восемь ключей тайла и проходит весь head_dim.
    const int key_column = warp * 8;
#pragma unroll
    for (int g = 0; g < kMmaGroups; ++g) {
      float scores[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      // Смещения в shared считаются 32-битными: size_t уводит адресацию в
      // обобщённое пространство, а это уже не дешёвый shared-доступ.
      const int q_base = g * kMmaRows * kRowPad;
      for (int k0 = 0; k0 < kHeadDim; k0 += 16) {
        uint32_t a[4];
        uint32_t b[2];
        a[0] = pack_bf16(&q_tile[q_base + grp * kRowPad + k0 + pos * 2]);
        a[1] = pack_bf16(&q_tile[q_base + (grp + 8) * kRowPad + k0 + pos * 2]);
        a[2] = pack_bf16(&q_tile[q_base + grp * kRowPad + k0 + pos * 2 + 8]);
        a[3] = pack_bf16(&q_tile[q_base + (grp + 8) * kRowPad + k0 + pos * 2 + 8]);
        b[0] = pack_bf16(&k_tile[(key_column + grp) * kRowPad + k0 + pos * 2]);
        b[1] = pack_bf16(&k_tile[(key_column + grp) * kRowPad + k0 + pos * 2 + 8]);
        mma_m16n8k16(scores, a, b);
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int r = grp + (i >= 2 ? 8 : 0);
        const int column = key_column + pos * 2 + (i & 1);
        const int token = key_base + column;
        s_tile[(g * kMmaRows + r) * kMmaKeys + column] =
            token < row_context[r] ? scores[i] * softmax_scale : -CUDART_INF_F;
      }
    }
    __syncthreads();

    // Строка достаётся восьми соседним дорожкам: максимум и сумма сходятся
    // бабочкой внутри восьмёрки, а не последовательным проходом одного потока.
    for (int slot = threadIdx.x >> 3; slot < kMmaGroups * kMmaRows;
         slot += kMmaThreads / 8) {
      const int part = threadIdx.x & 7;
      const int score_base = slot * kMmaKeys;
      float local_max = -CUDART_INF_F;
#pragma unroll
      for (int c = part; c < kMmaKeys; c += 8) {
        local_max = fmaxf(local_max, s_tile[score_base + c]);
      }
#pragma unroll
      for (int offset = 4; offset > 0; offset >>= 1) {
        local_max =
            fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, offset));
      }
      const float merged = fmaxf(row_max[slot], local_max);
      // Строки внутри варпа расходятся по этому условию, поэтому ветка не
      // должна содержать шаффл: маска 0xffffffff требует всех дорожек.
      const bool empty = merged == -CUDART_INF_F;
      float local_sum = 0.0f;
#pragma unroll
      for (int c = part; c < kMmaKeys; c += 8) {
        float weight = 0.0f;
        if (!empty) {
          const float score = s_tile[score_base + c];
          weight = score == -CUDART_INF_F ? 0.0f : __expf(score - merged);
        }
        p_tile[slot * kProbPad + c] = __float2bfloat16(weight);
        local_sum += weight;
      }
#pragma unroll
      for (int offset = 4; offset > 0; offset >>= 1) {
        local_sum += __shfl_xor_sync(0xffffffffu, local_sum, offset);
      }
      if (part == 0) {
        if (empty) {
          row_scale[slot] = 1.0f;
        } else {
          const float correction = row_max[slot] == -CUDART_INF_F
              ? 0.0f
              : __expf(row_max[slot] - merged);
          row_sum[slot] = row_sum[slot] * correction + local_sum;
          row_max[slot] = merged;
          row_scale[slot] = correction;
        }
      }
    }
    __syncthreads();

#pragma unroll
    for (int g = 0; g < kMmaGroups; ++g) {
      // Перенос онлайн-softmax на накопитель: дорожка держит grp и grp+8.
      const float scale_low = row_scale[g * kMmaRows + grp];
      const float scale_high = row_scale[g * kMmaRows + grp + 8];
#pragma unroll
      for (int t = 0; t < kMmaTilesPerWarp; ++t) {
        accumulator[g][t][0] *= scale_low;
        accumulator[g][t][1] *= scale_low;
        accumulator[g][t][2] *= scale_high;
        accumulator[g][t][3] *= scale_high;
      }
      // PV: варп владеет своей четвертью head_dim.
      const int p_base = g * kMmaRows * kProbPad;
#pragma unroll
      for (int t = 0; t < kMmaTilesPerWarp; ++t) {
        const int dimension_column = warp * kMmaDimPerWarp + t * 8;
        for (int k0 = 0; k0 < kMmaKeys; k0 += 16) {
          uint32_t a[4];
          uint32_t b[2];
          a[0] = pack_bf16(&p_tile[p_base + grp * kProbPad + k0 + pos * 2]);
          a[1] = pack_bf16(&p_tile[p_base + (grp + 8) * kProbPad + k0 + pos * 2]);
          a[2] = pack_bf16(&p_tile[p_base + grp * kProbPad + k0 + pos * 2 + 8]);
          a[3] = pack_bf16(&p_tile[p_base + (grp + 8) * kProbPad + k0 + pos * 2 + 8]);
          const int dimension = dimension_column + grp;
          b[0] = pack_pair(v_tile[(k0 + pos * 2) * kRowPad + dimension],
                           v_tile[(k0 + pos * 2 + 1) * kRowPad + dimension]);
          b[1] = pack_pair(v_tile[(k0 + pos * 2 + 8) * kRowPad + dimension],
                           v_tile[(k0 + pos * 2 + 9) * kRowPad + dimension]);
          mma_m16n8k16(accumulator[g][t], a, b);
        }
      }
    }
  }

#pragma unroll
  for (int g = 0; g < kMmaGroups; ++g) {
    const int query_head = kv_head * kGroup + head_base + g;
#pragma unroll
    for (int t = 0; t < kMmaTilesPerWarp; ++t) {
      const int dimension_base = warp * kMmaDimPerWarp + t * 8 + pos * 2;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int r = grp + (i >= 2 ? 8 : 0);
        if (r >= tile_rows) {
          continue;
        }
        const int dimension = dimension_base + (i & 1);
        const int batch_row = row_base + tile_first + r;
        const float denominator = row_sum[g * kMmaRows + r];
        const float value =
            denominator > 0.0f ? accumulator[g][t][i] / denominator : 0.0f;
        const size_t offset =
            (static_cast<size_t>(batch_row) * kQueryHeads + query_head) *
                kHeadDim + dimension;
        output[offset] = __float2bfloat16(
            value * output_gate(
                        query_gate_projection, batch_row, query_head, dimension));
      }
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
  static_assert(kGroup % kMmaGroups == 0, "головы группы делятся между CTA");
  const size_t shared = mma_shared_bytes();
  if (shared > 48u * 1024u) {
    // Динамическая shared сверх 48 КБ доступна только по явному запросу.
    const cudaError_t opted = cudaFuncSetAttribute(
        paged_attention_prefill_mma_kernel<Bf16Cache>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared));
    if (opted != cudaSuccess) {
      return opted;
    }
  }
  const dim3 grid(
      kKvHeads, (rows + kMmaRows - 1) / kMmaRows, kGroup / kMmaGroups);
  paged_attention_prefill_mma_kernel<Bf16Cache>
      <<<grid, kMmaThreads, shared, stream>>>(
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

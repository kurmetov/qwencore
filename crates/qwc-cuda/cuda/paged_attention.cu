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

// Сколько элементов кладёт в shared один поток за раз. Поэлементная укладка
// KV стоила больше инструкций, чем все MMA тайла вместе взятые.
constexpr int kStageVector = 8;

// Восемь элементов кэша за одно обращение.
//
// Поэлементное чтение превращало укладку тайла в 16384 двухбайтовых загрузки
// на CTA. Выровнено оно всегда: измерение кратно восьми, а шаг ряда в shared
// кратен 16 байтам.
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
// Тайл строк обязан принадлежать одной последовательности: таблица страниц
// берётся у первой строки тайла. Для чанка префилла это так по построению.

// Фрагмент читается дорожкой по адресу (grp * ряд + pos * 2): без набивки
// grp уходит кратно 32 банкам и весь варп садится на четыре банка. Набивка
// делает шаг ряда нечётным в банках и разводит дорожки по всем 32.
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

// Загрузка фрагментов одной инструкцией вместо четырёх скалярных чтений.
//
// `ldmatrix` читает восемь строк 8x8 сразу и раскладывает их по дорожкам
// ровно так, как их ждёт `mma.m16n8k16`. Адрес даёт каждая дорожка своя: для
// .x4 дорожка L задаёт строку L%8 матрицы L/8, для .x2 — то же среди первых
// шестнадцати. Вариант .trans разворачивает тайл на лету, и операнд V,
// лежащий в shared как [ключ][измерение], не приходится ни перекладывать,
// ни собирать по два байта.
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

// Тайл в 64 строки запроса: варп владеет своими шестнадцатью строками и
// всеми ключами тайла.
//
// Прежняя форма делила работу поперёк: 16 строк на CTA, четыре варпа резали
// по ключам в QK^T и по измерениям в PV. Из-за этого счёт шёл через shared —
// S туда, обратно под softmax, P снова туда, — и на каждые 32 ключа
// приходилось два барьера. Хуже другое: уложенный тайл KV обслуживал 16
// строк запроса, и весь контекст перечитывался на каждые 16 строк.
//
// Здесь тайл KV общий на 64 строки: укладка та же, а MMA на ней вчетверо
// больше. Softmax живёт в регистрах варпа (бабочка по четырём дорожкам с
// одинаковым grp), P из аккумулятора QK^T идёт в операнд A без единой
// перекладки, и барьеры остались только у стейджинга.
constexpr int kFlashWarps = 4;
constexpr int kFlashThreads = kFlashWarps * 32;
constexpr int kFlashRows = kFlashWarps * 16;
constexpr int kFlashKeys = 32;
constexpr int kFlashKeyTiles = kFlashKeys / 8;      // n-тайлов в QK^T
constexpr int kFlashKeyGroups = kFlashKeys / 16;    // k-групп в PV
constexpr int kFlashDimTiles = kHeadDim / 8;        // n-тайлов аккумулятора

struct FlashShared {
  __nv_bfloat16 query[kFlashRows * kRowPad];
  __nv_bfloat16 key[kFlashKeys * kRowPad];
  __nv_bfloat16 value[kFlashKeys * kRowPad];
  int row_context[kFlashRows];
};

constexpr size_t flash_shared_bytes() { return sizeof(FlashShared); }

// Максимум и сумма строки: строка живёт на четырёх дорожках с одинаковым grp,
// то есть на соседних по младшим двум битам.
__device__ __forceinline__ float flash_row_max(float value) {
  value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 1));
  return fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 2));
}

__device__ __forceinline__ float flash_row_sum(float value) {
  value += __shfl_xor_sync(0xffffffffu, value, 1);
  return value + __shfl_xor_sync(0xffffffffu, value, 2);
}

// Разбиение по контексту: grid.z — член GQA-группы и партиция ключей.
//
// На малом числе строк сетка без разбиения — 24 CTA на тайл, и каждый читает
// весь контекст в одиночку: четыре строки проверки черновиков на 30k стоили
// 5.3 мс на слой против 0.2 мс у decode-строки. С партициями каждый CTA
// считает свой отрезок ключей и пишет ненормированный числитель, максимум и
// сумму строки; сводит их reduce_partitions_kernel, как у decode. Direct —
// одна партиция, сразу в выход: этот путь не менялся.
template <bool Bf16Cache, bool Direct>
__global__ __launch_bounds__(kFlashThreads, 1) void paged_attention_prefill_flash_kernel(
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
    int rows,
    int row_base,
    int partition_tokens,
    float softmax_scale) {
  const int kv_head = blockIdx.x;
  const int tile = blockIdx.y;
  const int query_head = kv_head * kGroup + blockIdx.z % kGroup;
  const int partition = blockIdx.z / kGroup;
  const int partitions = gridDim.z / kGroup;
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
  // Таблица страниц общая на тайл: строки чанка принадлежат одной
  // последовательности.
  const uint32_t* table =
      block_tables + static_cast<size_t>(row_base + tile_first) * max_blocks;
  // Причинные префиксы строк варпа: за ними ему в тайле делать нечего.
  const int warp_context =
      max(shared.row_context[row0 + 15], shared.row_context[row0]);

  float accumulator[kFlashDimTiles][4] = {};
  float running_max[2] = {-CUDART_INF_F, -CUDART_INF_F};
  float running_sum[2] = {0.0f, 0.0f};

  // Отрезок ключей партиции. partition_tokens кратен тайлу ключей, поэтому
  // тайл никогда не пересекает границу отрезка: за ней лежит только конец
  // контекста, а его маскирует причинный предел строки. Последняя партиция
  // идёт до конца контекста тайла, даже если хост его недооценил: тогда
  // пострадает только скорость, а не ответ.
  const int key_begin = Direct ? 0 : partition * partition_tokens;
  const int key_end = (Direct || partition == partitions - 1)
      ? max_context
      : min(max_context, key_begin + partition_tokens);

  for (int key_base = key_begin; key_base < key_end; key_base += kFlashKeys) {
    __syncthreads();
    for (int i = threadIdx.x * kStageVector; i < kFlashKeys * kHeadDim;
         i += kFlashThreads * kStageVector) {
      const int key = i / kHeadDim;
      const int dimension = i - key * kHeadDim;
      const int token = key_base + key;
      if (token < key_end) {
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
      continue;   // барьеры варп отработал, считать ему нечего
    }

    // QK^T: строки варпа против всех ключей тайла.
    float scores[kFlashKeyTiles][4] = {};
    // Адреса для ldmatrix: дорожка задаёт строку своей 8x8 матрицы.
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

    // Масштаб и причинная маска. Дорожка держит строки grp и grp+8.
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

    // Онлайн-softmax: всё в регистрах варпа, ни одного обращения к shared.
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
      // Строка, у которой в тайле нет ни одного разрешённого ключа: вычитать
      // из -inf нельзя, это NaN.
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

    // P из аккумулятора QK^T прямо в операнд A: раскладка m16n8k16 совпадает,
    // если считать столбец ключа индексом k.
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
    // V лежит как [ключ][измерение], а операнду B нужен ключ по k: тайл
    // разворачивает сама инструкция.
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

  if constexpr (!Direct) {
    // Строки партиций нумеруются от начала сегмента: row_base добавит
    // редукция, когда будет писать выход.
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int r = row0 + grp + half * 8;
      if (pos == 0 && r < tile_rows) {
        const size_t slot =
            (static_cast<size_t>(tile_first + r) * kQueryHeads + query_head) *
                partitions + partition;
        partial_max[slot] = running_max[half];
        partial_sum[slot] = running_sum[half];
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
        const size_t slot =
            (static_cast<size_t>(tile_first + r) * kQueryHeads + query_head) *
                partitions + partition;
        partial_output[slot * kHeadDim + dimension] = accumulator[t][i];
      }
    }
    return;
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
    int partitions,
    int row_base) {
  const int query_head = blockIdx.x;
  const int batch = blockIdx.y;
  const int output_row = row_base + batch;
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
      (static_cast<size_t>(output_row) * kQueryHeads + query_head) * kHeadDim +
      threadIdx.x;
  const float value = denominator > 0.0f ? numerator / denominator : 0.0f;
  output[output_offset] = __float2bfloat16(
      value * output_gate(
                  query_gate_projection,
                  output_row,
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
      static_cast<__nv_bfloat16*>(output), partitions, 0);
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
    void* workspace,
    size_t workspace_bytes,
    int rows,
    int row_base,
    int max_blocks,
    int partitions,
    int partition_tokens,
    float softmax_scale,
    cudaStream_t stream) {
  if (query == nullptr || key_cache == nullptr || value_cache == nullptr ||
      block_tables == nullptr || context_lengths == nullptr ||
      output == nullptr || rows <= 0 || row_base < 0 || max_blocks <= 0 ||
      partitions <= 0 || partitions * kGroup > 65535 ||
      (partitions > 1 &&
       (partition_tokens <= 0 || partition_tokens % kFlashKeys != 0))) {
    return cudaErrorInvalidValue;
  }
  const size_t shared = flash_shared_bytes();
  static const cudaError_t opted_direct = cudaFuncSetAttribute(
      paged_attention_prefill_flash_kernel<Bf16Cache, true>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(shared));
  static const cudaError_t opted_split = cudaFuncSetAttribute(
      paged_attention_prefill_flash_kernel<Bf16Cache, false>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(shared));
  if (opted_direct != cudaSuccess) {
    return opted_direct;
  }
  if (opted_split != cudaSuccess) {
    return opted_split;
  }
  const dim3 grid(kKvHeads, (rows + kFlashRows - 1) / kFlashRows, kGroup * partitions);
  if (partitions == 1) {
    paged_attention_prefill_flash_kernel<Bf16Cache, true>
        <<<grid, kFlashThreads, shared, stream>>>(
            static_cast<const __nv_bfloat16*>(query),
            static_cast<const __nv_bfloat16*>(query_gate_projection),
            key_cache,
            value_cache,
            static_cast<const uint32_t*>(block_tables),
            static_cast<const uint32_t*>(context_lengths),
            static_cast<__nv_bfloat16*>(output),
            nullptr, nullptr, nullptr,
            max_blocks,
            rows,
            row_base,
            0,
            softmax_scale);
    return cudaGetLastError();
  }

  const size_t partials = static_cast<size_t>(rows) * kQueryHeads * partitions;
  const size_t required = partials * (kHeadDim + 2) * sizeof(float);
  if (workspace == nullptr || workspace_bytes < required) {
    return cudaErrorInvalidValue;
  }
  auto* partial_max = static_cast<float*>(workspace);
  auto* partial_sum = partial_max + partials;
  auto* partial_output = partial_sum + partials;
  paged_attention_prefill_flash_kernel<Bf16Cache, false>
      <<<grid, kFlashThreads, shared, stream>>>(
          static_cast<const __nv_bfloat16*>(query),
          static_cast<const __nv_bfloat16*>(query_gate_projection),
          key_cache,
          value_cache,
          static_cast<const uint32_t*>(block_tables),
          static_cast<const uint32_t*>(context_lengths),
          static_cast<__nv_bfloat16*>(output),
          partial_max, partial_sum, partial_output,
          max_blocks,
          rows,
          row_base,
          partition_tokens,
          softmax_scale);
  const cudaError_t launched = cudaGetLastError();
  if (launched != cudaSuccess) {
    return launched;
  }
  const dim3 reduction_grid(kQueryHeads, rows);
  reduce_partitions_kernel<<<reduction_grid, kThreads, 0, stream>>>(
      partial_max, partial_sum, partial_output,
      static_cast<const __nv_bfloat16*>(query_gate_projection),
      static_cast<__nv_bfloat16*>(output), partitions, row_base);
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
    void* workspace,
    size_t workspace_bytes,
    int rows,
    int row_base,
    int max_blocks,
    int partitions,
    int partition_tokens,
    float softmax_scale,
    cudaStream_t stream) {
  return launch_prefill_attention_mma<true>(
      query, query_gate_projection, key_cache, value_cache, block_tables,
      context_lengths, output, workspace, workspace_bytes, rows, row_base,
      max_blocks, partitions, partition_tokens, softmax_scale, stream);
}

extern "C" cudaError_t qwc_paged_attention_prefill_mma_fp8(
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
    int partitions,
    int partition_tokens,
    float softmax_scale,
    cudaStream_t stream) {
  return launch_prefill_attention_mma<false>(
      query, query_gate_projection, key_cache, value_cache, block_tables,
      context_lengths, output, workspace, workspace_bytes, rows, row_base,
      max_blocks, partitions, partition_tokens, softmax_scale, stream);
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

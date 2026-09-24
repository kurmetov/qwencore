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

// Внимание с упакованной GQA-группой: CTA на KV-голову, строки MMA-тайла —
// пары (строка запроса, голова группы).
//
// flash-ядро выше даёт каждой голове запроса свой CTA. KV там читается шесть
// раз, а на четырёх строках проверки черновиков из 64 строк тайла заняты
// четыре: 4 строки на 60k стоили 1.47 мс на слой против 90 мкс у xqa
// FlashInfer. Здесь все шесть голов группы идут строками одного тайла:
// упакованная строка p — строка запроса p / 6, голова p % 6. Decode —
// 6 строк из 16, проверка трёх черновиков — 24 из 32, и каждый байт KV
// читается один раз.
//
// KV лежит в shared как есть, в fp8. Раньше тайл переводился в bf16 при
// укладке, и загрузка шла синхронно с вычислениями. Теперь страница (64
// ключа) приезжает через cp.async в двойной буфер, пока считается
// предыдущая, а в f16 байты переводятся уже во фрагментах.
//
// Если m-тайлов меньше четырёх, лишние варпы делят ключи стадии: у каждого
// SMSP свои тензорные ядра, и decode одним варпом на SM упёрся бы в один из
// четырёх. Частичные суммы варпов сводятся в shared в конце.
constexpr int kPackedWarps = 4;
constexpr int kPackedThreads = kPackedWarps * 32;
constexpr int kPackedStageKeys = kPageSize;
constexpr int kPackedStageBytes = kPackedStageKeys * kHeadDim;
constexpr int kPackedChunks = kHeadDim / 16;   // 16-байтных кусков в ряду fp8
constexpr int kPackedMaxTileRows = 64 / kGroup;

template <int MTiles>
struct PackedShared {
  union {
    struct {
      uint8_t key[2][kPackedStageBytes];
      uint8_t value[2][kPackedStageBytes];
    } ring;
    // Аккумуляторы варпов, деливших ключи, — после последней стадии.
    float merge[(kPackedWarps / MTiles - 1 > 0 ? kPackedWarps - MTiles : 1) *
                32 * kFlashDimTiles * 4];
  };
  __half query[MTiles * 16 * kRowPad];
  int row_context[MTiles * 16];
  float merge_max[kPackedWarps][16];
  float merge_sum[kPackedWarps][16];
};

// Ряд ключа — 256 байт, 16-байтные куски переставлены по трём младшим битам
// ключа. ldmatrix читает восемь рядов на одной позиции куска, и без
// перестановки все восемь легли бы в одни банки.
__device__ __forceinline__ int packed_swizzle(int key, int chunk) {
  return key * kHeadDim + ((chunk ^ (key & 7)) << 4);
}

__device__ __forceinline__ void cp_async_16(
    void* destination, const void* source, bool valid) {
  // Недействительный кусок заполняется нулями: хвост последней страницы может
  // хранить что угодно, в том числе fp8 NaN, а 0 * NaN в PV — NaN.
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

// Два e4m3 из младших 16 бит в f16x2; младший байт — в младшую половину.
// Любое e4m3 точно представимо в f16.
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

// Тайл в MTiles * 16 упакованных строк, tile_rows целых строк запроса одной
// последовательности (таблица страниц — у первой строки тайла). Для decode
// tile_rows = 1, и каждая строка — своя последовательность.
//
// Раскладки фрагментов.
// QK^T: operand B — ряды K. ldmatrix над fp8, прочитанным как b16, отдаёт
// дорожке четыре байта подряд: измерения 4pos..4pos+3 ключа grp. Порядок
// суммирования по измерениям свободен, если A переставлен так же, поэтому
// в 16-ке измерений логические k {2pos, 2pos+1, 2pos+8, 2pos+9} — это
// физические 4pos..4pos+3. Запрос кладётся в shared уже переставленным.
// PV: operand B — V с ключом по k. ldmatrix.trans над b16 отдаёт дорожке
// пару ключей 2pos, 2pos+1 для пары измерений 2grp, 2grp+1; prmt разводит
// байты на два n-тайла: чётные измерения 16-ки и нечётные. В итоге дорожка
// держит в аккумуляторе четыре измерения подряд, 16c + 4pos..+3.
template <int MTiles, bool Direct>
__global__ __launch_bounds__(kPackedThreads, 1) void paged_attention_packed_kernel(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ query_gate_projection,
    const uint8_t* __restrict__ key_cache,
    const uint8_t* __restrict__ value_cache,
    const uint32_t* __restrict__ block_tables,
    const uint32_t* __restrict__ context_lengths,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    float* __restrict__ partial_output,
    int max_blocks,
    int rows,
    int row_base,
    int tile_rows,
    float softmax_scale) {
  constexpr int kSlices = kPackedWarps / MTiles;
  constexpr int kSliceKeys = kPackedStageKeys / kSlices;
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

  const int tile_first = tile * tile_rows;
  const int tile_count = min(tile_rows, rows - tile_first);

  extern __shared__ __align__(16) char packed_shared_raw[];
  auto& shared = *reinterpret_cast<PackedShared<MTiles>*>(packed_shared_raw);

  // Запрос в f16 с перестановкой измерений внутри 16-ки. Восемь физических
  // измерений 4pp..4pp+3 для двух pp — это логические пары (2pp, 2pp+1) и
  // (2pp+8, 2pp+9).
  for (int i = threadIdx.x; i < kPackedRows * (kHeadDim / 8);
       i += kPackedThreads) {
    const int p = i / (kHeadDim / 8);
    const int octet = i % (kHeadDim / 8);
    const int local = p / kGroup;
    uint4 raw = make_uint4(0u, 0u, 0u, 0u);
    if (local < tile_count) {
      const size_t base =
          (static_cast<size_t>(row_base + tile_first + local) * kQueryHeads +
           kv_head * kGroup + p % kGroup) *
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
    const int local = p / kGroup;
    shared.row_context[p] = local < tile_count
        ? static_cast<int>(context_lengths[row_base + tile_first + local])
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
      block_tables + static_cast<size_t>(row_base + tile_first) * max_blocks;

  // Отрезок партиции считается здесь, по фактическому контексту тайла, и
  // кратен странице. Хост выбирает только число партиций: захваченный
  // CUDA-граф не зависит от длины контекста, а decode-строки разной длины
  // делят каждая свой.
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

  // Номер страницы следующей стадии читается на стадию вперёд: зависимая
  // загрузка из таблицы иначе стояла бы перед каждой выдачей cp.async.
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
    // Буфер следующей стадии читали на предыдущей: все варпы должны её
    // досчитать, прежде чем его перепишет cp.async.
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
        break;   // барьеры варп отработает, считать ему нечего
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
          // Матрица j — 16-ка измерений k4 + j восьми ключей тайла t.
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

      // Масштаб и причинная маска. Дорожка держит строки grp и grp+8.
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
        // Матрицы: ключи 0-7 и 8-15 группы для 16-к измерений c и c+1.
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
    // Кольцо свободно: все стадии досчитаны, cp.async дождались. Варпы
    // срезов 1.. кладут свои суммы, варп среза 0 сводит их к своим.
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
        continue;   // строка пуста у всех срезов
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
    const int local = r / kGroup;
    if (local >= tile_count) {
      continue;
    }
    const int query_head = kv_head * kGroup + r % kGroup;
    if constexpr (!Direct) {
      // Строки партиций нумеруются от начала сегмента: row_base добавит
      // редукция, когда будет писать выход.
      const size_t slot =
          (static_cast<size_t>(tile_first + local) * kQueryHeads + query_head) *
              partitions +
          partition;
      if (pos == 0) {
        partial_max[slot] = running_max[half];
        partial_sum[slot] = running_sum[half];
      }
#pragma unroll
      for (int c = 0; c < kPackedChunks; ++c) {
        *reinterpret_cast<float4*>(
            &partial_output[slot * kHeadDim + c * 16 + pos * 4]) =
            make_float4(accumulator[2 * c][2 * half],
                        accumulator[2 * c + 1][2 * half],
                        accumulator[2 * c][2 * half + 1],
                        accumulator[2 * c + 1][2 * half + 1]);
      }
    } else {
      const int batch_row = row_base + tile_first + local;
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
    int tile_rows,
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
  const dim3 grid(kKvHeads, (rows + tile_rows - 1) / tile_rows, partitions);
  if (partitions == 1) {
    paged_attention_packed_kernel<MTiles, true>
        <<<grid, kPackedThreads, shared, stream>>>(
            static_cast<const __nv_bfloat16*>(query),
            static_cast<const __nv_bfloat16*>(query_gate_projection),
            static_cast<const uint8_t*>(key_cache),
            static_cast<const uint8_t*>(value_cache),
            static_cast<const uint32_t*>(block_tables),
            static_cast<const uint32_t*>(context_lengths),
            static_cast<__nv_bfloat16*>(output),
            nullptr, nullptr, nullptr,
            max_blocks, rows, row_base, tile_rows, softmax_scale);
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
  paged_attention_packed_kernel<MTiles, false>
      <<<grid, kPackedThreads, shared, stream>>>(
          static_cast<const __nv_bfloat16*>(query),
          static_cast<const __nv_bfloat16*>(query_gate_projection),
          static_cast<const uint8_t*>(key_cache),
          static_cast<const uint8_t*>(value_cache),
          static_cast<const uint32_t*>(block_tables),
          static_cast<const uint32_t*>(context_lengths),
          static_cast<__nv_bfloat16*>(output),
          partial_max, partial_sum, partial_output,
          max_blocks, rows, row_base, tile_rows, softmax_scale);
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

// Упакованное внимание по fp8-кэшу. tile_rows — строк запроса одной
// последовательности в тайле: 1 для decode-строк разных последовательностей,
// до 10 для сегмента одной.
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
    int tile_rows,
    int partitions,
    float softmax_scale,
    cudaStream_t stream) {
  if (query == nullptr || key_cache == nullptr || value_cache == nullptr ||
      block_tables == nullptr || context_lengths == nullptr ||
      output == nullptr || rows <= 0 || rows > qwc::kMaxStepRows ||
      row_base < 0 || max_blocks <= 0 || tile_rows <= 0 ||
      tile_rows > kPackedMaxTileRows || partitions <= 0 ||
      partitions > 65535 || !isfinite(softmax_scale) ||
      softmax_scale <= 0.0f) {
    return cudaErrorInvalidValue;
  }
  const int packed = tile_rows * kGroup;
  if (packed <= 16) {
    return launch_packed_attention<1>(
        query, query_gate_projection, key_cache, value_cache, block_tables,
        context_lengths, output, workspace, workspace_bytes, rows, row_base,
        max_blocks, tile_rows, partitions, softmax_scale,
        stream);
  }
  if (packed <= 32) {
    return launch_packed_attention<2>(
        query, query_gate_projection, key_cache, value_cache, block_tables,
        context_lengths, output, workspace, workspace_bytes, rows, row_base,
        max_blocks, tile_rows, partitions, softmax_scale,
        stream);
  }
  return launch_packed_attention<4>(
      query, query_gate_projection, key_cache, value_cache, block_tables,
      context_lengths, output, workspace, workspace_bytes, rows, row_base,
      max_blocks, tile_rows, partitions, softmax_scale,
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

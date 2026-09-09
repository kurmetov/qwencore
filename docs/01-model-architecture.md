# Qwen3.8-27B — архитектура и следствия для inference engine

Источник: `Qwen/Qwen3.8-27B/config.json` (`Qwen3_5ForConditionalGeneration`, transformers 5.8.0.dev0).

## 1. Что это за модель

Dense 27B, **нативно vision-language**, с **гибридным attention** и **встроенной MTP-головой**.

```
model_type            qwen3_5
num_hidden_layers     64
hidden_size           5120
intermediate_size     17408      (SwiGLU, silu)
vocab_size            248320
rms_norm_eps          1e-6
tie_word_embeddings   false      <- lm_head отдельный, 1.27B параметров
max_position_embeddings 262144
```

### 1.1 Гибридный стек слоёв (3:1)

`full_attention_interval = 4` → паттерн `[L, L, L, F] × 16`:

* **48 слоёв** — `linear_attention` = **Gated DeltaNet** (рекуррентное состояние константного размера)
* **16 слоёв** — `full_attention` (индексы 3, 7, 11, ..., 63)

### 1.2 Full attention (16 слоёв)

```
num_attention_heads     24
num_key_value_heads     4       (GQA, группа 6)
head_dim                256     <- НЕ 128
attn_output_gate        true    <- дополнительный gate-проекшн на выходе
partial_rotary_factor   0.25    <- RoPE только на первых 64 из 256 dims
rope_theta              1e7
rope: mrope_interleaved, mrope_section [11, 11, 10]
```

### 1.3 Linear attention / Gated DeltaNet (48 слоёв)

```
linear_num_key_heads    16   (q, k)
linear_key_head_dim     128
linear_num_value_heads  48   (v)   <- broadcast k/q 1:3 на v-головы
linear_value_head_dim   128
linear_conv_kernel_dim  4         <- короткая causal conv1d перед рекуррентностью
output_gate_type        swish
mamba_ssm_dtype         float32   <- состояние в fp32
```

### 1.4 MTP

```
mtp_num_hidden_layers        1
mtp_use_dedicated_embeddings false
```
Draft-голова **уже в чекпоинте**. Speculative decoding не требует отдельной draft-модели.
Замеренный acceptance rate в vLLM: **0.77–0.90** при `num_speculative_tokens=3`.

### 1.5 Vision tower (выбрасываем)

27 слоёв, hidden 1152, patch 16, spatial_merge 2, out_hidden 5120 (~0.5B параметров).
В text-only режиме **не грузится вообще** — экономия ~0.5–1 GB VRAM и весь препроцессинг.

---

## 2. Разбор параметров (26.90B)

| Компонент | Параметров |
|---|---:|
| MLP × 64 (3 × 5120 × 17408) | 17.11 B |
| Linear-attn × 48 (q,k,v,out,gate) | 5.56 B |
| Full-attn × 16 (q,k,v,o,gate) | 1.68 B |
| embed_tokens (248320 × 5120) | 1.27 B |
| lm_head (untied) | 1.27 B |
| **Итого** | **26.90 B** |

MLP = 64% всех весов → именно они дают выигрыш от NVFP4.
Эмбеддинги + lm_head = 2.54B (9.4%) — заметная доля, их точность/формат надо решать отдельно.

---

## 3. Память: главное открытие проекта

### 3.1 Веса (NVFP4 = 4 бита + FP8 e4m3 scale на блок 16 → 4.5 бита эфф.)

| Чекпоинт | Рецепт | Размер |
|---|---|---:|
| `nvidia/Qwen3.8-27B-NVFP4` | NVFP4: MLP + lm_head; **FP8: attn + linear-attn**; BF16 embed | **≈ 20.1 GB** |
| `QUASAR-QAT/...-QUASAR-NVFP4` | NVFP4: все linear-слои; BF16 embed | **≈ 17.0 GB** |
| то же + FP8 embed | | **≈ 15.7 GB** |

На 31.4 GiB usable это разница между **11 GB** и **15.7 GB** свободных под кэш — т.е. напрямую задаёт max concurrency.

### 3.2 Per-sequence состояние (константное, НЕ растёт с контекстом)

Рекуррентное состояние DeltaNet: `48 слоёв × 48 v-голов × 128 × 128`

```
= 37.75 M элементов на последовательность
= 151 MB  @ fp32   (как в конфиге)
=  75 MB  @ bf16   (наш ablation-эксперимент)
```
+ conv-состояние (kernel 4): ~6 MB fp32 — пренебрежимо.

### 3.3 KV cache (только 16 full-attn слоёв)

`16 × 4 kv-головы × 256 × 2 (K,V)` = 32 768 элементов на токен:

```
BF16   64 KB / токен
FP8    32 KB / токен
NVFP4  18 KB / токен
```

### 3.4 Точка пересечения — то, чего нет ни в одном обычном движке

При FP8 KV состояние DeltaNet «стоит» столько же, сколько KV-кэш при:

```
bf16 state (75 MB)  ->  ~2 360 токенов контекста
fp32 state (151 MB) ->  ~4 700 токенов контекста
```

**Ниже ~2.4K–4.7K контекста доминирует не KV-кэш, а рекуррентное состояние.**
Т.е. на коротких запросах concurrency упирается в фиксированные слоты состояний,
а не в paged KV. Это принципиально другой аллокатор и другой scheduler.

### 3.5 Бюджет 32 GB (считается кодом)

`cargo run -p qwc-core --bin plan` — единственный источник истины по этим числам.
Валидация модели памяти: расчёт «чекпоинт как есть» даёт **20.50 GB** против
реальных **20.6 GB** файлов в репозитории QUASAR, т.е. разбор параметров по слоям верен.

```
Веса
  чекпоинт как есть (грузит vLLM):            20.50 GB
  наша загрузка (text-only, fp8 head/embed):  17.08 GB
  фора:                                        3.42 GB

Кэш (kv fp8_e4m3, state bf16)
  KV на токен (16 слоёв):        32768 B
  состояние на слот (48 слоёв):   81.4 MB
  точка пересечения:              2484 токенов

Бюджет
  всего VRAM       34.19 GB   (32607 MiB)
  зарезервировано   2.20 GB   (десктоп gnome-shell ~1.2 GB + CUDA-контекст)
  веса             17.08 GB
  workspace         2.00 GB
  под кэш          12.91 GB
```

Максимальный concurrency:

| контекст | fp8 KV | nvfp4 KV | vLLM-like (fp8 KV, веса как на диске) |
|---:|---:|---:|---:|
| 4K | 59 | 82 | 44 |
| 8K | 36 | 55 | 27 |
| 16K | 20 | 33 | 15 |
| 32K | **11** | **18** | **8** |
| 64K | 5 | 10 | 4 |
| 128K | 2 | 5 | 2 |

Вся целевая матрица бенчмарков (вплоть до 32 seq x 2.5K = 5.29 GB) помещается с запасом
в 12.91 GB — упор в производительность, а не в память. Память становится узким местом
начиная с 16K контекста.

Два вывода:
* **NVFP4 KV-кэш даёт +64% concurrency на 32K** — это отдельная крупная оптимизация,
  которой нет в baseline на потребительском Blackwell.
* **3.42 GB экономии на весах = +3 последовательности на 32K** (11 против 8),
  то есть +37% ещё до единой строчки кернелов.

---

## 4. Следствия для дизайна движка

1. **KV-кэш ≠ вся память.** Нужны две подсистемы: paged KV (16 слоёв) + пул фиксированных
   state-слотов (48 слоёв). Admission control должен считать обе.
2. **Prefix caching почти не работает «как обычно».** Для linear-слоёв нельзя переиспользовать
   часть префикса — только снапшот состояния на границе, а это 75–151 MB на снапшот.
   Отдельный исследовательский вопрос: окупается ли snapshot-кэш префиксов.
3. **head_dim = 256** — почти все готовые paged-attention кернелы заточены под ≤128.
   На SM120 всего **99 KB shared memory на SM** (против 228 KB на SM100) → тайлинг придётся
   проектировать вручную, это главный кернельный риск.
4. **attn_output_gate** — лишний GEMM 5120×6144 на каждом full-attn слое, фьюзится с o_proj.
5. **partial RoPE 25%** — RoPE только на 64 из 256 dims; остальное passthrough.
   В **text-only** mrope вырождается в обычный RoPE (все три позиции равны) → выкидываем mrope целиком.
6. **MTP встроен** → адаптивный speculative decoding доступен «из коробки», без второй модели.
   Верификация дешёвая: draft = 1 слой.
7. **Chunked prefill естественен** для DeltaNet (состояние переносится между чанками),
   но требует chunk-wise parallel scan, а не наивной рекуррентности.

---

## 5. Целевое железо: RTX 5090 (GB202, SM120)

```
VRAM             32 GB GDDR7, ~1792 GB/s
Compute cap      12.0  -> nvcc -arch=sm_120a (или compute_120f, CUDA 13.x)
Driver (тут)     595.84, CUDA 13.2
Shared mem/SM    99 KB   (SM100: 228 KB)
```

**Критично:** SM120 использует SM80-style `mma.sync` с block-scaling
(`mma.sync.aligned.m16n8k64...kind::mxf4nvf4.block_scale`), а **не** `tcgen05.mma` датацентрового
Blackwell. Всё, что построено на tcgen05 (DeepGEMM, часть CUTLASS SM100 collective builders),
на 5090 не собирается и не работает. Нужны CUTLASS-схемы под SM120 с ручным тайлингом под 99 KB.

## 6. Известная слабость baseline (наш целевой выигрыш)

vLLM на одной RTX 5090 с NVFP4 требует `--enforce-eager` — при захвате CUDA-графов случается OOM.
То есть **baseline работает без CUDA Graphs**. Статическое планирование памяти в
специализированном рантайме позволяет графы захватить — это прямой, измеримый выигрыш по ITL.

---

## 7. Целевой чекпоинт: QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4

```
format          nvfp4-pack-quantized  (compressed-tensors)
group_size      16
weight scale    float8_e4m3   (per-16 блок)
act scale       float8_e4m3, local dynamic  -> W4A4, квантизация активаций на лету
symmetric       true
ignored         lm_head, .*visual.*, .*mtp.*, .*embed_vision.*, .*embed_audio.*
```

Шарды: 5 × safetensors, **20.6 GB на диске**. Раскладка:

| | размер |
|---|---:|
| Все linear-слои (24.35B @ 4.5 бит) | 13.7 GB |
| `lm_head` BF16 (не квантован) | 2.54 GB |
| `embed_tokens` BF16 | 2.54 GB |
| vision tower BF16 | ~1.0 GB |
| MTP-голова BF16 | ~0.8 GB |
| **итого** | **20.6 GB** |

### Наши преобразования при загрузке (то, чего vLLM не делает)

| Шаг | Экономия |
|---|---:|
| Не грузим vision tower (text-only) | −1.0 GB |
| `lm_head` → FP8 | −1.27 GB |
| `embed_tokens` → FP8 (lookup, точность некритична) | −1.27 GB |
| **VRAM под веса** | **≈ 17.0 GB** |

vLLM грузит все 20.6 GB. **3.6 GB форы = +3-4 последовательности на 32K контекста.**

### Следствие для кернелов

`local dynamic activation quantization` означает, что перед каждым GEMM нужно квантовать
активации в FP4 с FP8-шкалой на блок 16 **в рантайме, на каждом токене**.
Фьюз `RMSNorm -> amax(block16) -> quant_fp4` обязателен, иначе три лишних прохода по VRAM
на каждом из 64 слоёв на каждом токене.

MTP-голова и `lm_head` остаются в BF16/FP8 -> нужен второй GEMM-путь помимо NVFP4.

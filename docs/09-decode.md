# Token loop: prefill, CUDA graphs и continuous batching

```bash
IDS=$(scripts/tok.py encode "The capital of France is")
cargo run --release -p qwc-engine --bin generate -- --prompt-ids "$IDS" --max-new 24
scripts/tok.py decode "<выданные id>"
```

## Полный тракт

`Executor` держит адреса decode-буферов постоянными и захватывает отдельный
CUDA graph для каждого встретившегося размера batch. Один шаг проходит все 64
слоя:

* FP8 embedding сразу выдаёт BF16;
* 48 Gated DeltaNet слоёв обновляют conv-history и рекуррентный state slot;
* 16 full-attention слоёв записывают FP8 paged KV и считают gated GQA;
* MLP выполняет SwiGLU и down projection;
* residual-add совмещён с RMSNorm следующего блока;
* FP8 `lm_head` выдаёт логиты, двухступенчатый GPU argmax возвращает только
  четырёхбайтовый token ID.

Batch 1..4 использует W4A16 и fused gate/up. Batch 5..32 автоматически
переходит на W4A4 tensor-core GEMM; для него gate и up считаются двумя GEMM с
небольшим BF16 SwiGLU-кернелом между ними.

## Causal prefill

Промпт разбивается на чанки с фиксированным `M=128`. Все NVFP4-проекции и MLP
считают 128 строк одним W4A4 GEMM. В неполном последнем чанке лишние строки
являются padding и не попадают в причинные операции.

Рекуррентную ось нельзя трактовать как batch. Поэтому prefill имеет два
специализированных скана:

* один CUDA thread ведёт трёхэлементную conv-history одного канала через весь
  чанк;
* один warp держит строку DeltaNet-state в регистрах через все токены чанка.
  После каждого шага state округляется в BF16, поэтому результат совпадает с
  последовательным decode, но 1.5 MB состояния слоя читаются и пишутся один
  раз на чанк, а не 128 раз.

Full attention сначала параллельно записывает K/V всех токенов чанка, затем
использует общую физическую page table и отдельную `context_len` для каждой
query-строки. Это даёт обычную causal mask без отдельного dense attention
буфера. Логиты считаются только для последнего живого токена.

На RTX 5090 601-токенный prompt прошёл за **171.9 ms**, или **0.286
ms/token**. Старый token-by-token путь занимал около 8 s. После prefill модель
корректно продолжила повторяющуюся панграмму (`…The` → ` quick`) и затем начала
её разбор.

## Decode и sampling, batch 1

| | |
|---|---:|
| шаг decode с CUDA graph, медиана | **14.17 ms (70.5 tok/s)** |
| GPU argmax | **0.03 ms** |
| полный такт | **14.20 ms (70.4 tok/s)** |
| прежний host argmax | 0.38 ms и 1 MB PCIe readback |

Первый шаг конкретного batch-size захватывает graph и поэтому не входит в
steady-state медиану. Позиции, длины, state slots и page tables находятся в
резидентных device buffers: перед повторным launch меняется их содержимое, но
не адреса.

Контекст 30 и 601 токен по-прежнему имеет одинаковый ITL: на batch 1 тракт
упирается в 15 GB весов и состояния за шаг, а не в десятки мегабайт KV.

## W4A4 decode, batch 8

Восемь одинаковых промптов после независимого prefill дали одинаковые токены:

```text
 Paris.
The capital of Germany is Berlin.
```

Медиана decode — **24.90 ms**, то есть **321.3 tok/s goodput**; вместе с GPU
argmax — **319.3 tok/s**. Это сквозная проверка dispatcher, а не изолированный
projection benchmark.

## Связка с runtime

`Executor::execute_layout` принимает `qwc_runtime::BatchLayout`. Decode-часть
batch получает реальные `state_slots`, позиции, длины контекста и физические
KV page IDs. Prefill-чанки используют те же произвольные slots/pages, после
чего GPU argmax возвращает `(SeqId, token)` для каждого элемента batch.

Пример полного цикла admission → prefill → decode → completion:

```bash
cargo run --release -p qwc-engine --bin scheduled -- --requests 8 --max-new 8
```

Проверочный запуск завершил восемь запросов, все результаты совпали. При новом
prompt с offset 0 переиспользованный DeltaNet/conv slot обнуляется на GPU;
страницы KV не требуют очистки, потому что видимость задаётся `context_len`.

## Проверка

```bash
cargo test --workspace -- --test-threads=1
cargo clippy --workspace --all-targets -- -D warnings
```

Сейчас проходят 68 тестов, включая GPU-oracle для chunked DeltaNet scan и
детерминированного argmax. Отдельно выполнены end-to-end запуски 27B checkpoint
для batch 1, batch 8 и scheduler-driven page tables. Пошаговое сравнение
token IDs и top-k log-probabilities с внешними runtime описано в
[`10-differential-eval.md`](10-differential-eval.md).

## Ограничения decode-пути

* несколько prefill-последовательностей смешанного batch исполняются
  последовательно, хотя каждый их чанк использует tensor cores;
* W4A4 gate/up перечитывают нормализованную активацию: общего fused mainloop
  у них нет;
* нет preemption, prefix snapshots и corpus-level perplexity eval;
* `scripts/tok.py` использует ASCII-приближение pre-tokenizer; для Unicode
  нужен полный tokenizer.

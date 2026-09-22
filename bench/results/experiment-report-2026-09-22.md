# Экспериментальный контур — отчёт 2026-09-22

## Итог

Проект получил воспроизводимый контур для абляций и честного сравнения с vLLM:

- каждый trial запускается в отдельном процессе;
- порядок вариантов меняется между trials;
- сохраняются raw JSONL, полная команда, commit, dirty-state, hash исходников и
  состояние GPU до/после запуска;
- отчёт считает bootstrap 95% CI среднего и paired 95% CI относительного
  ускорения;
- publishable запуск блокируется, если до старта занято больше 8 GiB VRAM;
- vLLM запускается с CUDA graphs, явно выключенным prefix caching, тем же
  checkpoint, corpus, KV dtype, context, длинами и общим VRAM budget.

Реализация: `bench/experiments.py`. Методика: `docs/11-experiments.md`.

## Исправленный roofline

Старая модель смешивала resident footprint и трафик decode-шага.

| величина | значение |
|---|---:|
| resident weights | 17.08 GB |
| веса, реально читаемые ordinary decode | **14.96 GB** |
| измеренная устойчивая полоса DRAM | **1.605 TB/s** |
| идеальный ITL, batch=1, ctx=2048 | **9.42 ms** |
| физический потолок | **106 tok/s** |

MTP-голова не читается обычным шагом, а embedding делает lookup одной строки,
поэтому обе величины исключены из полного весового прохода. Расчёт теперь
живёт в `WeightPlan::decode_weight_bytes` и используется `qwc-core plan`.

Последний ранее измеренный ordinary decode 79.2 tok/s соответствует примерно
75% исправленного roofline. Спекулятивный decode нельзя делить на этот потолок:
один проход основной модели подтверждает несколько токенов.

## Матрица сравнений

Comparison suite заранее фиксирует четыре режима:

| сценарий | QwenCore | vLLM |
|---|---|---|
| batch-1 без MTP | ordinary decode | ordinary decode |
| batch-1 с MTP | k=3 + shortlist 32K | Qwen3.5 MTP k=3 |
| serving c=32 | chunk 2048 | max batched tokens 2048 |
| prefill 4096/8 | chunk 2048 | max batched tokens 4096 |

Во всех строках используются реальные неповторяющиеся prompt token IDs.

## Матрица абляций

| переход | изолированный фактор |
|---|---|
| base -> MTP k=3 | speculative decoding |
| MTP k=3 -> shortlist 32K | сокращённый draft `lm_head` |
| recurrent BF16 -> recurrent FP32 | округление состояния внутри prefill |
| recurrent FP32 -> WY | алгоритм DeltaNet scan |
| prefill chunk 512 -> 2048 | token budget шага |
| BF16 KV -> FP8 KV | формат KV-кэша |

Int8 DeltaNet state пока остаётся cross-build абляцией: runtime-переключателя
назад на BF16 нет. Существующий измеренный результат — 25.72 -> 24.67 ms при
batch=32 и −1.13 GB VRAM; он не смешивается с одно-бинарными абляциями.

## Проверки

- Python unit tests: **7/7 passed**.
- `cargo test -p qwc-core`: **16/16 passed**.
- `cargo check -p qwc-engine --bin servebench`: **passed** в отдельном target.
- `git diff --check`: **passed**.
- Обе experiment suites прошли dry-run; все команды и корпуса разрешаются.
- Общий `cargo fmt --check` остаётся красным на существующих незакоммиченных
  файлах вне этого изменения; автоматическое переформатирование чужого дерева
  не выполнялось.

## Почему нет новой CI-таблицы

Во время работы RTX 5090 была занята на 28.9 из 32.6 GiB. Основные потребители:

- `VLLM::EngineCore`: 24.2 GiB;
- отдельный Python-процесс: 3.1 GiB.

Запуск 26-GB сравнения в таком состоянии был бы либо OOM, либо измерением
конкуренции с чужим workload. Старые одиночные точки нельзя превратить в
доверительные интервалы постфактум, поэтому новые цифры не выдумывались.

## Команды для финального прогона

После освобождения GPU:

```bash
~/.venvs/baseline/bin/python bench/experiments.py run \
  --suite comparison --repeats 7 \
  --output bench/results/comparison-trials.jsonl

python3 bench/experiments.py report \
  --input bench/results/comparison-trials.jsonl \
  --output bench/results/comparison-with-ci.md

~/.venvs/baseline/bin/python bench/experiments.py run \
  --suite ablations --repeats 7 \
  --output bench/results/ablation-trials.jsonl

python3 bench/experiments.py report \
  --input bench/results/ablation-trials.jsonl \
  --output bench/results/ablations-with-ci.md
```

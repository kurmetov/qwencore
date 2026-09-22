# Экспериментальный протокол: абляции, roofline и сравнение с vLLM

Цель этого протокола — отделить инженерный результат от случайной удачной
цифры. Публикуется не лучший прогон, а все независимые trials, точная команда,
состояние GPU до и после прогона и 95% доверительный интервал.

## Что именно утверждаем

У проекта три разных утверждения, и их нельзя смешивать в один множитель:

1. **Batch-1 decode:** насколько близко обычный шаг подходит к bandwidth
   roofline и выигрывает ли он у vLLM без спекуляции.
2. **Batch-1 MTP:** сколько accepted tokens получается на один проход весов;
   сравниваются лучшие MTP-конфигурации обоих движков.
3. **Serving под бюджетом:** aggregate throughput, TTFT и ITL при одинаковом
   общем VRAM budget. Здесь меньший resident footprint является частью
   результата, но не называется преимуществом CUDA-ядер.

Prefill публикуется отдельно: он compute-bound и не подчиняется decode
roofline.

## Roofline

Знаменатель decode — байты, реально прочитанные шагом, а не вся занятая VRAM:

```text
resident weights                           17.08 GB
ordinary decode streamed weights           14.96 GB
batch=1 state read+write + KV(ctx=2048)    ~0.16 GB
measured sustainable DRAM read bandwidth    1.605 TB/s
ideal ITL                                  ~9.42 ms
ideal throughput                            ~106 tok/s
```

MTP имеет другой знаменатель: один основной проход может подтвердить больше
одного токена. Для него рядом с tok/s обязательно публикуются accepted
tokens/step, draft time и verification time; делить MTP tok/s на обычный
single-token roofline нельзя.

Числа генерируются из кода:

```bash
cargo run -q -p qwc-core --bin plan
```

## Повторные прогоны и интервалы

`bench/experiments.py` запускает каждый trial в новом процессе. Порядок
вариантов циклически меняется между trials, чтобы прогрев карты и thermal drift
не доставались всегда одному движку. Сырой JSONL дописывается после каждого
прогона с `fsync`, поэтому длинный эксперимент можно продолжить `--resume`.

Отчёт показывает percentile-bootstrap 95% CI среднего по независимым
процессным прогонам, 20 000 resamples с фиксированным seed. Для относительного
ускорения применяется paired bootstrap по одинаковым номерам trials. Минимум —
3 повтора, нормальный публикуемый запуск — 7 или больше.

По умолчанию runner отказывается стартовать, если до эксперимента занято больше
8 GiB VRAM. Это защита от случайного сравнения рядом с чужим inference job.
Порог меняется через `--max-background-memory-gb`; флаг `--allow-busy-gpu`
оставлен только для диагностических, заведомо загрязнённых прогонов и записывается
в artifact.

```bash
# Сначала проверить точные команды.
python3 bench/experiments.py run --suite comparison --dry-run

# Честное сравнение. Нужен Python из vLLM venv.
~/.venvs/baseline/bin/python bench/experiments.py run \
  --suite comparison --repeats 7 \
  --output bench/results/comparison-trials.jsonl

python3 bench/experiments.py report \
  --input bench/results/comparison-trials.jsonl \
  --output bench/results/comparison-with-ci.md
```

Comparison suite фиксирует одинаковые checkpoint, corpus, KV dtype, context,
число запросов, длины и общий memory budget. CUDA graphs включены, prefix
caching явно выключен, перед записью есть warmup. Для c=32 используются
одинаковые token budgets 2048. Для чистого prefill публикуются заранее выбранные
лучшие значения из свипа: chunk 2048 у QwenCore и
`max_num_batched_tokens=4096` у vLLM.

## Абляции

```bash
~/.venvs/baseline/bin/python bench/experiments.py run \
  --suite ablations --repeats 7 \
  --output bench/results/ablation-trials.jsonl

python3 bench/experiments.py report \
  --input bench/results/ablation-trials.jsonl \
  --output bench/results/ablations-with-ci.md
```

Матрица устроена так, чтобы соседние строки меняли один фактор:

| сценарий | переход | что изолируется |
|---|---|---|
| `ablate-speculation` | base -> MTP k=3 | speculative decoding |
| | MTP k=3 -> + shortlist 32K | неполный draft `lm_head` |
| `ablate-prefill-scan` | recurrent BF16 -> recurrent FP32 -> WY | округление и затем алгоритм DeltaNet scan |
| `ablate-prefill-chunk` | 512 -> 2048 | token budget шага |
| `ablate-kv` | BF16 -> FP8 | формат KV при неизменном workload |

Int8 DeltaNet state пока не входит в автоматическую матрицу: в текущем дереве
нет runtime-переключателя обратно на BF16. Его существующая A/B сделана двумя
сборками. Публиковать её следует отдельной строкой с двумя commit/diff hashes,
а не делать вид, что это тот же исполняемый файл.

## Правила публикации

* Не удалять failed trials из JSONL. Отчёт считает только успешные, но явно
  показывает число исключённых ошибок.
* Не запускать publishable серию при посторонней вычислительной нагрузке на
  GPU. Desktop footprint допустим, если он одинаков и записан в artifact.
* Не выбирать лучший из повторов. Публикуются среднее, CI и все raw trials.
* Любой post-hoc выбор batch/token budget сначала уходит в tuning sweep, затем
  подтверждается новой серией trials.
* Speed ablation сопровождается quality A/B теми же token IDs. Для
  приближённых форматов одного совпадения greedy недостаточно: сохраняются
  top-k overlap и logprob error.

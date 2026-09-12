# Differential eval

Осмысленное продолжение smoke-теста — сравнение не строк, а следующего
распределения токенов на одном checkpoint. Контур находится в
`bench/diff_eval.py`, замороженный корпус — в `bench/corpus/core.jsonl`.

## Что сравнивается

Каждый backend получает **те же token IDs**, без chat template, BOS и скрытой
ретокенизации. Для каждого шага сохраняются greedy token ID и top-20 raw
log-probabilities. Отчёт считает:

* длину общего greedy-префикса;
* top-1 agreement;
* Jaccard overlap множеств top-k;
* mean/max absolute delta общих log-probabilities.

После первой greedy-развилки распределения больше не сравниваются: у моделей
уже разные контексты. Сама длина общего префикса продолжает отражать расхождение
генерации.

Корпус включает английский текст, Python, русский, китайский, арабский,
математику и whitespace, а также синтетические длины 63/64/65 (граница
64-токенной KV-страницы), 127/128/129 (граница prefill-чанка) и 601 токен.

## Быстрый запуск

```bash
PY=$HOME/.venvs/baseline/bin/python
MODEL=$HOME/models/Qwen3.8-27B-QUASAR-NVFP4

$PY bench/diff_eval.py validate-corpus --model "$MODEL"
$PY bench/diff_eval.py probe --model "$MODEL"

$PY bench/diff_eval.py run --engine qwc --batch 1 \
  --model "$MODEL" --output bench/results/qwc-b1.jsonl
$PY bench/diff_eval.py run --engine qwc --batch 8 \
  --model "$MODEL" --output bench/results/qwc-b8.jsonl
$PY bench/diff_eval.py run --engine qwc --batch 1 --lm-head bf16 \
  --model "$MODEL" --output bench/results/qwc-b1-bf16-head.jsonl
$PY bench/diff_eval.py run --engine qwc --batch 1 --embedding bf16 \
  --model "$MODEL" --output bench/results/qwc-b1-bf16-embedding.jsonl
$PY bench/diff_eval.py run --engine qwc --batch 1 --qwc-kv-cache-dtype bf16 \
  --model "$MODEL" --output bench/results/qwc-b1-bf16-kv.jsonl
$PY bench/diff_eval.py run --engine qwc --batch 1 --qwc-decode-linear w4a4 \
  --model "$MODEL" --output bench/results/qwc-b1-w4a4-decode.jsonl
$PY bench/diff_eval.py run --engine vllm \
  --model "$MODEL" --output bench/results/vllm.jsonl
$PY bench/diff_eval.py run --engine vllm --vllm-kv-cache-dtype fp8 \
  --model "$MODEL" --output bench/results/vllm-fp8-kv.jsonl

$PY bench/diff_eval.py compare \
  --reference bench/results/vllm.jsonl \
  bench/results/qwc-b1.jsonl bench/results/qwc-b8.jsonl \
  --output bench/results/report.md \
  --json-output bench/results/report.json \
  --fail-on-threshold
```

Пороги по умолчанию: top-1 не ниже 0.95, top-k Jaccard не ниже 0.70,
log-prob MAE не выше 0.25. Их можно менять аргументами `compare`, но baseline
следует хранить со строгими значениями, а не подгонять после результата.

`--lm-head bf16` оставляет финальную матрицу в исходном типе checkpoint
и добавляет 1.27 GB VRAM. Это A/B-диагностика: если MAE почти не изменится,
ошибка приходит в `lm_head` уже из hidden state; если упадёт ниже gate,
FP8-квантизацию головы нельзя оставлять в quality-конфигурации.
`--embedding bf16` проводит тот же A/B-замер для входной таблицы.
`--vllm-kv-cache-dtype fp8` квантует только full-attention KV у reference;
Gated DeltaNet state остаётся в dtype модели. Это изолирует вклад FP8 KV
без смены checkpoint или весов.

## Backend-адаптеры

### Transformers

```bash
$PY bench/diff_eval.py run --engine transformers \
  --model "$MODEL" --output bench/results/transformers.jsonl
```

Адаптер читает полные logits непосредственно из `forward`, поэтому это самый
сильный эталон. Для `device_map` требуется `accelerate`. Важно: backend обязан
уметь исполнять именно `compressed-tensors` NVFP4 checkpoint; деквантизация
27B в BF16 не помещается на 32-GB GPU и не считается эквивалентным запуском.

### vLLM

Offline API получает `prompt_token_ids`, `temperature=0`, отключённые penalties
и top-k raw log-probabilities. EOS игнорируется, чтобы все backend вернули одно
число шагов.

### SGLang

Запускается внешний server, после чего используется нативный `/generate`: он,
в отличие от OpenAI chat API, принимает `input_ids` и возвращает numeric token
IDs.

```bash
python -m sglang.launch_server --model-path "$MODEL" --port 30000
$PY bench/diff_eval.py run --engine sglang --url http://127.0.0.1:30000 \
  --model "$MODEL" --output bench/results/sglang.jsonl
```

### llama.cpp

Нужен GGUF, полученный из **этого же** checkpoint без смены квантования или
токенизатора. Нативный `/completion` принимает массив token IDs и с
`n_probs=20` возвращает IDs и log-probabilities.

```bash
llama-server -m /models/qwen3.8-27b-nvfp4.gguf -c 1024 -ngl 999 --port 8080
$PY bench/diff_eval.py run --engine llamacpp --url http://127.0.0.1:8080 \
  --model "$MODEL" --output bench/results/llamacpp.jsonl
```

Если конвертер не поддерживает `Qwen3_5ForConditionalGeneration` и
`nvfp4-pack-quantized`, такой backend помечается несовместимым. Сравнивать с
другим GGUF-квантом можно как quality experiment, но не как проверку qwc.

### TensorRT-LLM, LMDeploy, MLC LLM, Ollama и другие OpenAI server

Для них есть fallback `--engine openai`. Следует указать имя отдельно, чтобы
оно не потерялось в отчёте:

```bash
$PY bench/diff_eval.py run --engine openai \
  --name tensorrt-llm --url http://127.0.0.1:8000 \
  --served-model-name qwen --model "$MODEL" \
  --output bench/results/tensorrt-llm.jsonl
```

OpenAI schema не возвращает numeric token IDs. Адаптер принимает позицию лишь
когда строка токена однозначно round-trip'ится официальным tokenizer; иначе
ставит `unsupported_exact_token_output`. Это слабее нативных адаптеров, но не
подменяет идентичность токенов догадкой.

### TGI

TGI custom API возвращает numeric IDs и top tokens, но принимает только текст.
Поэтому адаптер запускает лишь семантические cases, для которых официальный
tokenizer подтверждает точное совпадение с замороженными IDs. Синтетические
границы честно получают `unsupported_exact_token_input`.

ExLlamaV2 не имеет пути для этого NVFP4 checkpoint. Ollama и MLC требуют
конвертированную модель, поэтому их результаты являются отдельным сравнением
формата, а не строгим differential test исходных весов.

## Артефакт и CI

JSONL начинается с run-header, затем содержит по записи на case. Артефакты
самодостаточны: model path, version backend, SHA-256 корпуса, режим logits и
top-k. Их удобно сохранять между коммитами и сравнивать тем же `compare`.

CPU-тесты harness:

```bash
python3 -m unittest discover -s bench/tests -v
cargo test -p qwc-engine --bin eval
```

`--fail-on-threshold` возвращает ненулевой exit status и предназначен для CI.

## Зафиксированный прогон 2026-09-11

Полный отчёт лежит в `bench/results/report.md`, исходные JSONL — рядом. На
RTX 5090 32 GB практическим reference стал vLLM 0.28.0: Transformers 5.16.1
не смог исполнить packed checkpoint в заданной памяти.

| Кандидат | Покрытие | Greedy prefix | Top-1 | Top-20 Jaccard | Log-prob MAE | Результат |
|---|---:|---:|---:|---:|---:|---|
| qwc batch 1 | 14/14 | 1.000 | 1.000 | 0.782 | 0.364 | все 120 token IDs совпали; строгий calibration gate не пройден |
| qwc batch 8 | 14/14 | 0.967 | 0.991 | 0.772 | 0.436 | одна развилка в `math`, шаг 8; строгий gate не пройден |

У batch 1 генерация совпала с vLLM на каждом шаге, включая длины 63/64/65,
127/128/129 и 601. Это сильная проверка вычислительного тракта, но не
доказательство численной эквивалентности: MAE общих top-20 log-probabilities
выше порога 0.25, а максимальная дельта 3.868 пришлась на `long-601`.

Batch 8 отличается только в `math`: первые восемь токенов совпадают, затем
vLLM выбирает token 6558, а qwc — 271. Остальные 13 cases совпадают целиком.
Вероятные места накопления ошибки — FP8 embedding/lm_head, FP8 KV и точность
DeltaNet state; этот прогон их не локализует, поэтому причина пока не
приписывается одному кернелу.

Статус остальных runtime и причины отсутствия строгого результата записаны в
`bench/results/README.md`. В частности, SGLang 0.5.19 загрузил исходный NVFP4
checkpoint, но его первый FP4 warmup потребовал FlashInfer JIT и полный CUDA
SDK. На этой рабочей станции прогон был остановлен после потери отзывчивости
хоста; corpus result не публикуется.

# qwencore

Inference-движок для `Qwen3.8-27B-QUASAR-NVFP4` — dense 27B с гибридным
attention и NVFP4-весами — на **одной RTX 5090 32 GB**: Rust, собственные
CUDA-ядра под SM120, text-only.

Предмет проекта — не отдельное ядро, а методика. Потолки считаются от
**измеренных** пределов этой карты, а не паспортных; каждая оптимизация
проверяется однофакторной абляцией; публикуются все прогоны, включая неудачные.

## Результаты

«Потолок» ниже — расчёт из измеренной полосы и посчитанного трафика шага.
«Измерено» — прогон движка. Эти две категории нигде не смешиваются.

### Batch-1 decode

| | значение | откуда |
|---|---:|---|
| устойчивое чтение DRAM, измерено | **1.605 TB/s** | `gpuinfo`, буфер 4 GiB; паспортные 1792 GB/s недостижимы |
| веса, реально читаемые обычным шагом | **14.96 GB** | `plan`; resident footprint — 17.08 GB, это другая величина |
| + состояние DeltaNet и KV при ctx=2048 | 0.16 GB | `plan` |
| **потолок ITL** | **9.42 мс** | 15.12 GB / 1.605 TB/s |
| **потолок пропускной** | **106 tok/s** | то же |
| измерено, обычный decode | 79.2 tok/s | **75% потолка** |
| измерено, спекуляция k=3 + шортлист 32K | 138.5 tok/s | знаменатель другой, см. ниже |

Спекулятивный decode **нельзя** делить на этот потолок: один проход по весам
подтверждает несколько токенов, и 138.5 tok/s формально дают 130% — это
признак неверного знаменателя, а не нарушенной физики. Для спекуляции рядом с
tok/s публикуются accepted tokens/step, время черновика и время проверки.

Обе строки «измерено» — медианы прогонов **одной сессии**
([`speculation-corpus-2026-09-17.md`](bench/results/speculation-corpus-2026-09-17.md)),
без доверительных интервалов: карта в тот момент была частично занята, и по
правилам ниже это не publishable-серия. Интервалы появятся вместе с прогоном
suite, и до тех пор эти два числа — ориентир, а не результат.

### Эффективность GEMM

| режим | доля | чем измерено |
|---|---:|---|
| prefill, CUTLASS NVFP4 GEMM, M=2048 | **84% пика FP4** | 1402 TFLOPS из 1676, форма `down` |
| prefill, шаг целиком | 37.8% пика FP4 | 11 780 tok/s = 633 TFLOP/s |
| decode, CUTLASS, M=1..32 | **53-64% достижимой полосы** | 47-57% паспортной |
| decode, свой W4A16, batch 1, широкие формы | 87-94% достижимой полосы | `mlp_gate_up` 91%, `lm_head` 94% |
| decode, шаг batch 1, взвешенно по фазам | 68% достижимой полосы | вниз тянут `attn_qkv` (61%) и `la_out` (67%) |

Две последние строки — причина, по которой на batch 1 движок не использует
CUTLASS: разрыв узкого GEMM закрыт своим кернелом. Разбор с формами, временами
и знаменателями — [`docs/03-roofline.md`](docs/03-roofline.md).

### Абляция: состояние DeltaNet из bf16 в int8

| c=32, ctx=2048 | bf16 | int8 |
|---|---:|---:|
| шаг с CUDA-графом | 25.72 мс | **24.67 мс** |
| кэш и состояние | 4.92 GB | **3.79 GB** (−1.13 GB) |

int8, а не fp8, — по замеру: при том же одном байте int8 вчетверо точнее,
потому что строки состояния плоские и экспонента им не нужна. На batch 1
выигрыша нет и быть не может — расквантованная рабочая копия стоит
фиксированные 75.5 MB, окупаемость начинается с трёх слотов. Точность
проверена дифф-eval на шести длинах контекста; greedy-выход не меняется нигде.
Детали — [`bench/results/delta-state-8bit.md`](bench/results/delta-state-8bit.md).

Оговорка: это A/B **между двумя сборками**, а не переключатель в рантайме,
поэтому в автоматическую матрицу абляций строка не входит и публикуется
отдельно, с двумя хэшами коммитов.

### Что ещё НЕ измерено

* **Сравнение с vLLM по этому протоколу не опубликовано.** Comparison suite
  написан и проходит dry-run, но ни одной серии с доверительными интервалами
  не снято: карту занимал чужой workload. Одиночные точки прошлых сессий
  нельзя превратить в интервалы постфактум, поэтому ни одного множителя против
  vLLM в этом README нет и не будет до того, как suite отработает.
* **Качество модели.** Дифф-eval сверяет распределения токен в токен, но
  перплексии и task-eval нет. Влияние W4A4 и FP8-словаря на качество — открыто.
* Спекуляция при batch > 1, префикс-кэш гибридной модели, FP8-внимание на
  префилле, конфигурация CUTLASS W4A4 на малом M.

## Почему memory-bound

Переход из memory-bound в compute-bound для NVFP4 наступает при **batch ~294**
(`compute_bound_batch(Dtype::Nvfp4)`): machine balance равен
1676 TOPS / 1605 GB/s = 1044 FLOP/byte, а NVFP4-вес занимает 4.5 бита с учётом
FP8-шкал блоков. Целевой диапазон проекта — concurrency 1..32 — лежит глубоко
в memory-bound режиме.

Отсюда приоритеты, и они не интуитивные:

* вычислительное преимущество FP4 (1676 TOPS) в этом сценарии почти не
  работает; FP4 ценен тем, что делает **веса меньше**;
* оптимизировать надо **байты, а не FLOPы** — фьюзинг, отказ от промежуточных
  буферов, формат кэша и состояния, GEMV вместо GEMM;
* загрузка тензорных ядер — плохая метрика прогресса для decode, а доля
  достигнутой **пропускной способности памяти** — хорошая.

Ровно поэтому int8-состояние и FP8-словарь стоят в одном ряду с ядрами: это
тот же рычаг.

## Как мы меряем

Протокол целиком — [`docs/11-experiments.md`](docs/11-experiments.md),
реализация — `bench/experiments.py`.

* Каждый trial — **отдельный процесс**; порядок вариантов циклически меняется
  между trials, чтобы прогрев и thermal drift не доставались всегда одному.
* Отчёт даёт **percentile-bootstrap 95% CI** среднего (20 000 resamples,
  фиксированный seed); для относительного ускорения — paired bootstrap по
  одинаковым номерам trials. Минимум 3 повтора, публикуемый запуск — 7+.
* **Публикуются все trials.** Failed не удаляются из JSONL; отчёт считает
  успешные и явно печатает число исключённых.
* **Лучший прогон не выбирается** — публикуются среднее, CI и сырые данные.
  Любой post-hoc выбор batch или token budget сначала уходит в отдельный
  tuning sweep.
* Runner отказывается стартовать, если до эксперимента на карте занято больше
  8 GiB: это защита от сравнения рядом с чужим inference job.

Что каждая цифра означает и чем подтверждена — в `bench/results/`, по файлу на
замер, с датой и условиями. Числа из разных сессий несопоставимы: машина
дрейфует до 5-6% за день, и в файлах это оговорено.

## Воспроизведение

С нуля. CUTLASS и cuda-shim в гит не входят, без них `qwc-cuda` не собирается;
подробности и обоснование версий — [`docs/02-toolchain.md`](docs/02-toolchain.md).

```bash
git clone --depth 1 --branch v4.8.0 https://github.com/NVIDIA/cutlass third_party/cutlass
```

```bash
./scripts/gen-cuda-shim.sh third_party/cuda-shim
```

```bash
export PATH=/usr/local/cuda/bin:$PATH
```

Чекпоинт — `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` с Hugging Face, 20.6 GB в
пяти шардах; скрипт поддерживает докачку. Раскладка и её сверка с архитектурой —
[`docs/05-checkpoint.md`](docs/05-checkpoint.md).

```bash
./scripts/fetch-checkpoint.sh
```

Дальше — без GPU и без чекпоинта:

```bash
cargo run -q -p qwc-core --bin plan
```

```bash
python3 -m unittest discover -s bench/tests -v
```

```bash
python3 bench/experiments.py run --suite comparison --dry-run
```

```bash
python3 bench/experiments.py run --suite ablations --dry-run
```

Нужна карта:

```bash
cargo run --release -p qwc-cuda --bin gpuinfo
```

```bash
cargo test --workspace -- --test-threads=1
```

Полные серии с доверительными интервалами (нужен Python из venv с vLLM):

```bash
~/.venvs/baseline/bin/python bench/experiments.py run --suite ablations --repeats 7 --output bench/results/ablation-trials.jsonl
```

```bash
python3 bench/experiments.py report --input bench/results/ablation-trials.jsonl --output bench/results/ablations-with-ci.md
```

## Документация

| | |
|---|---|
| [`docs/03-roofline.md`](docs/03-roofline.md) | измеренная полоса, потолки decode, эффективность GEMM |
| [`docs/11-experiments.md`](docs/11-experiments.md) | протокол экспериментов: что утверждаем, как мерим, что запрещено |
| [`docs/01-model-architecture.md`](docs/01-model-architecture.md) | архитектура Qwen3.8 и следствия для движка |
| [`docs/02-toolchain.md`](docs/02-toolchain.md) | CUDA, SM120, CUTLASS, известные грабли тулчейна |
| [`docs/06-runtime.md`](docs/06-runtime.md) | двухресурсный кэш и continuous batching |
| [`docs/10-differential-eval.md`](docs/10-differential-eval.md) | сверка распределений с внешними runtime |
| `bench/results/` | по файлу на замер, с датой и условиями |

## Лицензия

MIT, см. [`LICENSE`](LICENSE). Чекпоинт и CUTLASS сюда не входят и живут под
своими лицензиями.

## OpenAI-compatible server

Сервер по умолчанию поднимает честный лимит контекста 32K, FP8 KV и общий
пул физических KV-страниц. `--context` задаёт лимит одной последовательности,
а `--kv-cache-gb` — память всего пула; поэтому 32K не резервируются заранее
для каждого из `--max-seqs` слотов.

```bash
cargo run --release -p qwc-engine --bin serve -- \
  --model ~/models/Qwen3.8-27B-QUASAR-NVFP4 \
  --context 32768 \
  --max-seqs 32 \
  --kv-cache fp8 \
  --kv-cache-gb 5 \
  --memory-limit-gb 28
```

Endpoints:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions` (обычный JSON и SSE streaming)

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model":"Qwen3.8-27B-QUASAR-NVFP4",
    "messages":[{"role":"user","content":"Ответь одним словом: работает?"}],
    "temperature":0,
    "max_tokens":32,
    "stream":true,
    "enable_thinking":false
  }'
```

Для OpenAI-клиента или агентного harness укажите:

```text
base_url = http://127.0.0.1:8000/v1
api_key  = local
model    = Qwen3.8-27B-QUASAR-NVFP4
```

Unicode tokenizer читается из `tokenizer.json`; text-only ветка штатного Qwen
chat template воспроизведена сервером. OpenAI `tools` преобразуются в родной
Qwen XML prompt, а XML-вызов модели обратно в структурированный `tool_calls`.
Текущий sampler — greedy: параметры
`temperature`/`top_p` пока принимаются для совместимости, но не меняют выбор
токена; top-k/top-p будет отдельным GPU path.

### Pi

Добавьте provider в `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "qwencore": {
      "baseUrl": "http://127.0.0.1:8000/v1",
      "api": "openai-completions",
      "apiKey": "local",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false
      },
      "models": [{
        "id": "Qwen3.8-27B-QUASAR-NVFP4",
        "name": "QwenCore local",
        "reasoning": false,
        "contextWindow": 32768,
        "maxTokens": 4096,
        "samplingParams": {
          "temperature": 0,
          "enable_thinking": false
        }
      }]
    }
  }
}
```

Формат custom provider описан в
[документации Pi](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/models.md).

### Hermes Agent

Можно запустить `hermes model` и выбрать `Custom endpoint`, либо добавить в
`~/.hermes/config.yaml`:

```yaml
model:
  default: Qwen3.8-27B-QUASAR-NVFP4
  provider: custom
  base_url: http://127.0.0.1:8000/v1
  api_key: local
  context_length: 32768
```

Это штатный путь Hermes для self-hosted `/v1/chat/completions`; см.
[документацию custom providers](https://github.com/NousResearch/hermes-agent/blob/main/website/docs/integrations/providers.md#custom--self-hosted-llm-providers).

### Размер FP8 KV-пула

Одна KV-страница содержит 64 токена и занимает 2 MiB по всем 16
full-attention слоям. Полная последовательность 32K занимает 512 страниц,
то есть примерно 1.07 GB. Пул 5 GB держит около четырёх полностью заполненных
32K последовательностей либо существенно больше коротких запросов. Admission
control резервирует lifetime запроса (`prompt + max_tokens`), поэтому GPU OOM
во время decode не возникает.

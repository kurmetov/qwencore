# qwencore

Локальный text-only runtime для `Qwen3.8-27B-QUASAR-NVFP4` на RTX 5090.

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

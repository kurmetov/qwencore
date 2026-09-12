# Backend coverage — 2026-09-11

Hardware: NVIDIA GeForce RTX 5090, 32 GB. Model:
`Qwen3.8-27B-QUASAR-NVFP4`, architecture
`Qwen3_5ForConditionalGeneration`, packed `compressed-tensors` NVFP4.

## Completed strict runs

| Runtime | Version | Exact frozen token IDs | Native packed checkpoint | Result |
|---|---:|---:|---:|---|
| qwc, batch 1 | 0.1.0 | yes | yes | 120/120 greedy tokens equal to vLLM |
| qwc, batch 8 | 0.1.0 | yes | yes | one divergence: `math`, generation step 8 |
| vLLM | 0.28.0 | yes | yes | reference |

See `report.md` for aggregate and per-case scores and `report.json` for the
machine-readable comparison. The source distributions are in `qwc-b1.jsonl`,
`qwc-b8.jsonl`, and `vllm.jsonl`.

## Attempted but not published as comparisons

| Runtime | Version | Observed status | Why there is no score |
|---|---:|---|---|
| Transformers | 5.16.1 | checkpoint recognized; forward tried to decompress 496 linear layers to BF16 and exhausted the 32-GB GPU at about 26% | not an executable packed-NVFP4 reference on this GPU |
| SGLang | 0.5.19 | loaded all five shards as `Qwen3_5ForConditionalGeneration`, `quant=compressed-tensors`, using 18.81 GB; FP4 warmup entered FlashInfer JIT | the host became unresponsive during JIT and was rebooted; no corpus response was recorded |

The SGLang launch first exposed a missing `ninja` on PATH and then a missing
`curand_kernel.h` in `/usr/local/cuda`. Supplying SGLang's venv `ninja` and a
narrow cuRAND include path got the process into FP4 JIT, but the experiment was
stopped for host safety. This is a launch/toolchain result, not evidence about
SGLang output quality.

## Adapter present, strict run unavailable

| Runtime | Harness route | Limitation for this checkpoint |
|---|---|---|
| llama.cpp | native `/completion` | requires a tokenizer-compatible GGUF conversion; no equivalent NVFP4 GGUF is present |
| TensorRT-LLM | OpenAI-compatible server | runtime/engine not installed locally |
| LMDeploy | OpenAI-compatible server | runtime not installed locally |
| TGI | native text API | runtime not installed; only text cases can preserve exact input IDs |
| MLC LLM | OpenAI-compatible server | requires a converted model |
| Ollama | OpenAI-compatible server | requires a converted model |
| ExLlamaV2 | none | no path for this compressed-tensors NVFP4 checkpoint |

Changing the quantization to make one of these runtimes load would measure a
different model representation. The harness therefore reports such cases as
unsupported instead of presenting them as an apples-to-apples qwc regression.

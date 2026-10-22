# Differential eval

The meaningful continuation of the smoke test is comparing not strings but the next-token
distribution on the same checkpoint. The harness is in
`bench/diff_eval.py`, the frozen corpus in `bench/corpus/core.jsonl`.

## What is compared

Each backend gets **the same token IDs**, without chat template, BOS or hidden
retokenization. For each step the greedy token ID and top-20 raw
log-probabilities are saved. The report computes:

* the length of the common greedy prefix;
* top-1 agreement;
* Jaccard overlap of the top-k sets;
* mean/max absolute delta of the shared log-probabilities.

After the first greedy divergence distributions are no longer compared: the models
already have different contexts. The common prefix length itself keeps reflecting the generation
divergence.

The corpus includes English text, Python, Russian, Chinese, Arabic,
math and whitespace, plus synthetic lengths 63/64/65 (the
64-token KV page boundary), 127/128/129 (the prefill chunk boundary) and 601 tokens.

## Quick start

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

Default thresholds: top-1 at least 0.95, top-k Jaccard at least 0.70,
log-prob MAE at most 0.25. They can be changed with `compare` arguments, but the baseline
should be kept with strict values, not tuned after seeing the result.

`--lm-head bf16` keeps the final matrix in the checkpoint's original type
and adds 1.27 GB of VRAM. This is an A/B diagnostic: if MAE barely changes,
the error reaches `lm_head` already from the hidden state; if it drops below the gate,
FP8 quantization of the head can't stay in the quality configuration.
`--embedding bf16` does the same A/B for the input table.
`--vllm-kv-cache-dtype fp8` quantizes only the reference's full-attention KV;
the Gated DeltaNet state stays in the model dtype. This isolates the FP8 KV contribution
without changing the checkpoint or weights.

## Backend adapters

### Transformers

```bash
$PY bench/diff_eval.py run --engine transformers \
  --model "$MODEL" --output bench/results/transformers.jsonl
```

The adapter reads full logits directly from `forward`, so it's the strongest
reference. `device_map` requires `accelerate`. Important: the backend must
be able to run exactly the `compressed-tensors` NVFP4 checkpoint; dequantizing
27B to BF16 doesn't fit on a 32 GB GPU and doesn't count as an equivalent run.

### vLLM

The offline API gets `prompt_token_ids`, `temperature=0`, disabled penalties
and top-k raw log-probabilities. EOS is ignored so all backends return the same
number of steps.

### SGLang

An external server is started, then the native `/generate` is used: unlike
the OpenAI chat API, it accepts `input_ids` and returns numeric token
IDs.

```bash
python -m sglang.launch_server --model-path "$MODEL" --port 30000
$PY bench/diff_eval.py run --engine sglang --url http://127.0.0.1:30000 \
  --model "$MODEL" --output bench/results/sglang.jsonl
```

### llama.cpp

You need a GGUF produced from **this same** checkpoint without changing quantization or
tokenizer. The native `/completion` accepts an array of token IDs and with
`n_probs=20` returns IDs and log-probabilities.

```bash
llama-server -m /models/qwen3.8-27b-nvfp4.gguf -c 1024 -ngl 999 --port 8080
$PY bench/diff_eval.py run --engine llamacpp --url http://127.0.0.1:8080 \
  --model "$MODEL" --output bench/results/llamacpp.jsonl
```

If the converter doesn't support `Qwen3_5ForConditionalGeneration` and
`nvfp4-pack-quantized`, the backend is marked incompatible. Comparing against
another GGUF quant is possible as a quality experiment, but not as a qwc check.

### TensorRT-LLM, LMDeploy, MLC LLM, Ollama and other OpenAI servers

For these there is the `--engine openai` fallback. Give the name separately so
it doesn't get lost in the report:

```bash
$PY bench/diff_eval.py run --engine openai \
  --name tensorrt-llm --url http://127.0.0.1:8000 \
  --served-model-name qwen --model "$MODEL" \
  --output bench/results/tensorrt-llm.jsonl
```

The OpenAI schema doesn't return numeric token IDs. The adapter accepts a position only
when the token string round-trips unambiguously through the official tokenizer; otherwise
it sets `unsupported_exact_token_output`. This is weaker than the native adapters, but it doesn't
replace token identity with a guess.

### TGI

The TGI custom API returns numeric IDs and top tokens, but accepts only text.
So the adapter runs only the semantic cases for which the official
tokenizer confirms an exact match with the frozen IDs. Synthetic
boundaries honestly get `unsupported_exact_token_input`.

ExLlamaV2 has no path for this NVFP4 checkpoint. Ollama and MLC need
a converted model, so their results are a separate format comparison,
not a strict differential test of the original weights.

## Artifact and CI

The JSONL starts with a run header, then has one record per case. Artifacts are
self-contained: model path, backend version, corpus SHA-256, logits mode and
top-k. They are convenient to keep between commits and compare with the same `compare`.

Harness CPU tests:

```bash
python3 -m unittest discover -s bench/tests -v
cargo test -p qwc-engine --bin eval
```

`--fail-on-threshold` returns a nonzero exit status and is meant for CI.

## Recorded run 2026-09-11

The full report is in `bench/results/report.md`, the source JSONL next to it. On the
RTX 5090 32 GB the practical reference became vLLM 0.28.0: Transformers 5.16.1
couldn't run the packed checkpoint within the given memory.

| Candidate | Coverage | Greedy prefix | Top-1 | Top-20 Jaccard | Log-prob MAE | Result |
|---|---:|---:|---:|---:|---:|---|
| qwc batch 1 | 14/14 | 1.000 | 1.000 | 0.782 | 0.364 | all 120 token IDs matched; strict calibration gate not passed |
| qwc batch 8 | 14/14 | 0.967 | 0.991 | 0.772 | 0.436 | one divergence in `math`, step 8; strict gate not passed |

At batch 1 generation matched vLLM at every step, including lengths 63/64/65,
127/128/129 and 601. That's a strong check of the compute path, but not
proof of numerical equivalence: the MAE of shared top-20 log-probabilities
is above the 0.25 threshold, and the max delta of 3.868 fell on `long-601`.

Batch 8 differs only in `math`: the first eight tokens match, then
vLLM picks token 6558 and qwc picks 271. The other 13 cases match entirely.
Likely places of error accumulation are FP8 embedding/lm_head, FP8 KV and DeltaNet state
precision; this run doesn't localize them, so the cause is not yet
attributed to a single kernel.

The status of the other runtimes and the reasons for no strict result are recorded in
`bench/results/README.md`. In particular, SGLang 0.5.19 loaded the original NVFP4
checkpoint, but its first FP4 warmup required FlashInfer JIT and the full CUDA
SDK. On this workstation the run was stopped after the host became
unresponsive; no corpus result is published.

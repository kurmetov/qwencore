# Batch-1 decode latency — 2026-09-11

> **Single-session points, no confidence intervals.** The vLLM comparison here
> predates the protocol in
> [`docs/11-experiments.md`](../../docs/11-experiments.md): one or two runs per
> point, no variant-order rotation, no bootstrap CI, on a partially occupied
> card. These multipliers are superseded by the comparison suite and must not be
> quoted as a project result. Per-point conditions are given below.

Hardware: RTX 5090, 32 GB. Model: `Qwen3.8-27B-QUASAR-NVFP4`, packed
compressed-tensors NVFP4, identical files for both engines.

## Method

`bench/latency.py` takes the per-token cost from the **slope** between a short
and a long completion from the same prompt:

    decode_ms_per_token = (T(72 tokens) - T(8 tokens)) / 64

The slope cancels prefill, the first-token cost and every fixed per-request
overhead, so engines with very different request paths stay comparable. Both
sides: greedy, `detokenize=False`, CUDA graphs on, warmup before any timing,
median of 5 repeats. The prompt is passed as token IDs, so no tokenizer
difference enters the measurement. vLLM is given `max_num_seqs`/
`max_num_batched_tokens` matched to what qwc is configured for, rather than the
8192-token default it would never use here.

## Result

| Engine | ms/token | tok/s | runs | % of roofline |
|---|---:|---:|---|---:|
| **qwc 0.1.0** | **13.11** | **76.3** | 13.10 / 13.11 / 13.13 | **82%** |
| vLLM 0.28.0 | 19.58 | 51.1 | 19.06 / 19.58 / 19.93 | 55% |

**qwc is 1.49× faster per decoded token.** The spread across independent runs is
0.03 ms for qwc and 0.9 ms for vLLM — the gap is far outside both.

The roofline is the analytic memory-bound ceiling from `qwc-core --bin plan`:
10.78 ms/token at batch 1, ctx 2048, because a decode step reads every weight.
qwc leaves 18% on the table; vLLM leaves 45%.

## What this does and does not say

It is a single-request, batch-1, short-context latency number — the thing a
user feels typing at one session. It is not throughput: vLLM's per-token cost
includes a scheduler step that exists to serve many concurrent requests, and
that cost amortizes under load where qwc's fixed batch-1 path has nothing to
amortize. At batch 32 the roofline itself moves to 15.22 ms/token and the
comparison has to be redone.

## The other two engines

| Engine | Status | What blocks a number |
|---|---|---|
| SGLang 0.5.19 | installed, own venv | the previous launch entered FlashInfer FP4 JIT and made the host unresponsive; it was rebooted. Re-running is a risk to the machine, not a scripting problem. |
| llama.cpp | built (`llama-server`), Qwen3-Next/gated-delta present in `llama-arch.cpp` | no GGUF of this checkpoint exists. Producing one means dequantizing NVFP4 to BF16 (~54 GB) and requantizing, which measures a different numeric representation — legitimate for a pure speed comparison only if labelled as such. |

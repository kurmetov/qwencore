<div align="center">

# qwencore

**A single-GPU inference engine for Qwen3.8-27B NVFP4 on the RTX 5090**

![Rust 2024](https://img.shields.io/badge/Rust-2024-b7410e?logo=rust)
![CUDA SM120](https://img.shields.io/badge/CUDA-sm__120a-76b900?logo=nvidia)
![GPU RTX 5090](https://img.shields.io/badge/GPU-RTX%205090%2032%20GB-333)
![License MIT](https://img.shields.io/badge/license-MIT-blue)

**75.8 tok/s** batch-1 decode (72% of the bandwidth ceiling) ·
**159 tok/s** with MTP speculation · **1161 tok/s** at c=32 (vLLM: 963)

</div>

An inference engine for `Qwen3.8-27B-QUASAR-NVFP4` — a dense 27B model with
hybrid attention and NVFP4 weights — on a **single RTX 5090 32 GB**: Rust,
custom CUDA kernels for SM120, text-only.

- **Kernels**: own W4A16 GEMV, Gated DeltaNet scan (recurrent and WY), paged
  and packed attention, FP8 vocab; CUTLASS NVFP4 GEMM for prefill.
- **Runtime**: paged FP8 KV cache, int8 DeltaNet state, continuous batching,
  CUDA graphs, MTP speculation, prefix cache.
- **Server**: OpenAI-compatible `/v1/chat/completions` with SSE and tool calls.
- **Measurement**: every claim has a dated report in `bench/results/`, and a
  differential eval checks token distributions against external runtimes.

## Results

"Ceiling" is computed from the card's measured bandwidth and per-step traffic;
"measured" is an engine run.

### Batch-1 decode

| | value | source |
|---|---:|---|
| sustained DRAM read, measured | **1.605 TB/s** | `gpuinfo`, 4 GiB buffer; the spec 1792 GB/s is unreachable |
| weights actually read per regular step | **14.96 GB** | `plan`; resident footprint is 17.08 GB, a different quantity |
| + DeltaNet state and KV at ctx=2048 | 0.16 GB | `plan` |
| **ITL ceiling** | **9.42 ms** | 15.12 GB / 1.605 TB/s |
| **throughput ceiling** | **106 tok/s** | same |
| measured, regular decode | 75.8 tok/s | **72% of ceiling** |
| measured, speculation k=3 + 32K shortlist | 159.1 tok/s | 3.36 tokens per pass, different denominator, see below |

Speculation has a different denominator: one pass over the weights confirms
several tokens, so tok/s is published together with accepted tokens/step,
draft time and verification time.

Both "measured" rows are medians of three runs in one session on 2026-09-24 on
a partially busy card, without confidence intervals; methodology —
[`speculation-corpus-2026-09-17.md`](bench/results/speculation-corpus-2026-09-17.md),
measurement — [`packed-attention-2026-09-24.md`](bench/results/packed-attention-2026-09-24.md),
section "Continuation". The earlier 79.2 and 138.5 were taken on 09-17: regular
decode of the same build before that day's changes gives 76.4 in this session,
so the gap to 79.2 is the card, not the code; speculation improved thanks to
packed attention and head priming.

### GEMM efficiency

| mode | share | measured by |
|---|---:|---|
| prefill, CUTLASS NVFP4 GEMM, M=2048 | **84% of FP4 peak** | 1402 of 1676 TFLOPS, `down` shape |
| prefill, whole step | 37.8% of FP4 peak | 11 780 tok/s = 633 TFLOP/s |
| decode, CUTLASS GEMM (79b, W4A4), M=1..32 | **53-64% of achievable bandwidth** | 47-57% of spec |
| decode, own W4A16, batch 1, wide shapes | 87-94% of achievable bandwidth | `mlp_gate_up` 91%, `lm_head` 94% |
| decode, CUTLASS GEMV (`91_fp4_gemv`), batch 1 | on par with own W4A16 | 0-2.5% in CUTLASS's favor, one series |
| decode, batch-1 step, phase-weighted | 68% of achievable bandwidth | dragged down by `attn_qkv` (61%) and `la_out` (67%) |

At M=1 the GEMM scheme is latency-bound, so at batch 1 the engine uses its own
W4A16: on `gate/up` it is 1.8x faster than the CUTLASS GEMM path (33.0 vs
59.3 µs). That is a lead over GEMM, not over CUTLASS in general: the
specialized CUTLASS GEMV, in the same series as our kernel, does the same
([`cutlass-gemv-2026-10-09.md`](bench/results/cutlass-gemv-2026-10-09.md)).
Our path exists not for speed but for format: bf16 in and out, and one weight
matrix shared across several tokens. Breakdown with shapes, timings and
denominators — [`docs/03-roofline.md`](docs/03-roofline.md).

### Ablation: DeltaNet state bf16 → int8

| c=32, ctx=2048 | bf16 | int8 |
|---|---:|---:|
| step with CUDA graph | 25.72 ms | **24.67 ms** |
| cache and state | 4.92 GB | **3.79 GB** (−1.13 GB) |

int8 rather than fp8 is a measured choice: at the same single byte, int8 is
four times more accurate because state rows are flat and need no exponent. At
batch 1 there is no gain and can't be — the dequantized working copy costs a
fixed 75.5 MB, break-even starts at three slots. Greedy output is unchanged —
verified by diff-eval at six context lengths. The A/B was taken with two
builds, not a runtime switch; details and hashes —
[`bench/results/delta-state-8bit.md`](bench/results/delta-state-8bit.md).

### qwc vs vLLM

Same card, same 25 GB budget per engine, KV fp8, two interleaved rounds on
2026-09-25 —
[`latency-2026-09-25.md`](bench/results/latency-2026-09-25.md).

| | qwc | vLLM |
|---|---:|---:|
| batch 1, TTFT of a 256-token prompt | **28 ms** | 65 ms |
| batch 1, prefill of an 8k prompt | 647 ms | **584 ms** |
| batch 1 + MTP k=3, short prompt | **159.7 tok/s** | 144.4 tok/s |
| batch 1 + MTP, 8k and 30k prompts | TPOT 5.4-5.6 ms | doesn't start |
| c=32, throughput | **1161 tok/s** | 963 tok/s |
| c=32, per-request TPOT | 26.1 ms | **17.4 ms** |

Without speculation qwc's TPOT is 2-3.5% lower at every prompt length.

### What's not here

Numbers apply to batch 1..32, ctx up to 32K, greedy and a single card. The
vLLM comparison is two rounds in one session on a shared card, without
confidence intervals; the full comparison suite is written but hasn't been run
as a series. Model quality was not measured —
diff-eval matches distributions token by token; no perplexity or task-eval.
Diff-eval runs the `eval` binary, which drives the executor directly, without
the scheduler, prefix cache or speculation; its conclusions don't apply to
`serve` with default settings, see
[below](#speculation-and-prefix-cache). There is no speculation at batch > 1 at
all: the server enables it only when generating a single request. The prefix
cache was measured in one session, without intervals. Not measured: FP8
attention in prefill and CUTLASS W4A4 at small M.

## Why memory-bound

The transition from memory-bound to compute-bound for NVFP4 happens at
**batch ~294** (`compute_bound_batch(Dtype::Nvfp4)`): machine balance is
1676 TOPS / 1605 GB/s = 1044 FLOP/byte, and an NVFP4 weight takes 4.5 bits
including FP8 block scales. The project's target range — concurrency 1..32 —
is deep in the memory-bound regime.

Hence the priorities, which are not intuitive:

* FP4's compute advantage (1676 TOPS) barely matters here; FP4 is valuable
  because it makes **weights smaller**;
* optimize **bytes, not FLOPs** — fusion, no intermediate buffers, cache and
  state format, GEMV instead of GEMM;
* tensor core utilization is a bad progress metric for decode; achieved
  **memory bandwidth** share is a good one.

That is exactly why int8 state and the FP8 vocab sit alongside the kernels:
it's the same lever.

## OpenAI-compatible server

By default the server sets an honest 32K context limit, FP8 KV and a shared
pool of physical KV pages. `--context` is the per-sequence limit and
`--kv-cache-gb` is the memory of the whole pool, so 32K is not reserved
upfront for each of the `--max-seqs` slots.

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
- `POST /v1/chat/completions` (plain JSON and SSE streaming)
- `POST /v1/completions` (no chat template, no streaming; for matching against `eval`)

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model":"Qwen3.8-27B-QUASAR-NVFP4",
    "messages":[{"role":"user","content":"Answer in one word: does it work?"}],
    "temperature":0,
    "max_tokens":32,
    "stream":true,
    "enable_thinking":false
  }'
```

For an OpenAI client or agent harness:

```text
base_url = http://127.0.0.1:8000/v1
api_key  = local
model    = Qwen3.8-27B-QUASAR-NVFP4
```

The Unicode tokenizer is read from `tokenizer.json`; the text-only branch of
the stock Qwen chat template is reproduced by the server. OpenAI `tools` are
converted to the native Qwen XML prompt, and the model's XML call back into
structured `tool_calls`. The sampler is greedy. `temperature`/`top_p` are
accepted for OpenAI client compatibility but don't affect token choice: with
`temperature > 0` the response is still deterministic.

### FP8 KV pool size

One KV page holds 64 tokens and takes 2 MiB across all 16 full-attention
layers. A full 32K sequence takes 512 pages, about 1.07 GB. A 5 GB pool holds
about four fully filled 32K sequences, or many more short requests. Admission
control reserves the request's lifetime (`prompt + max_tokens`), so there is no
GPU OOM during decode.

### Speculation and prefix cache

Both are on by default.

| flag | default | what it does |
|---|---|---|
| `--speculative K` | 3 | the MTP head proposes K drafts, the model verifies them in one pass. Works only when a single request is generating; the head takes another 0.85 GB |
| `--shortlist N` | 32768 | the draft computes logits over the N most frequent tokens plus context tokens, not the whole vocab |
| `--mtp-prime N` | 2048 | the head is primed on the last N prompt tokens if the server has one sequence |
| `--prefix-cache N` | 8 | DeltaNet state snapshots at the history boundary (~81 MB each); the next dialog turn doesn't recompute history |

Both change greedy output on near-tied tokens. Draft verification goes through
the prefill path, and its logits don't match decode bit for bit. The cache
cuts every prompt at the last `<|im_start|>`, even on a miss, and the short
tail after the cut is computed by a recurrent scan, not a chunked one.
Diff-eval's numeric claims apply to the `eval` path at batch 1. With
`--prefix-cache 0 --speculative 0` the server on a single request matches
`eval` bit for bit if both use the same `--context`: attention partitioning
depends on it. Speculation adds decode-level noise; tokens on the diff-eval
corpora all match. The prefix cache on long prompts fails the diff-eval
threshold (MAE 0.41), cause not yet investigated. For comparison: a single
reordering of fp32 addition in prefill attention gives 0.11–0.13 against the
previous build, so the 0.25 threshold for prefill changes is strict. With
several concurrent requests there is no match in any case: at batch 8 diff-eval
fails the strict threshold. Breakdown —
[`experiment-report-2026-09-24.md`](bench/results/experiment-report-2026-09-24.md).

Draft verification uses a packed attention kernel (all six heads of a GQA
group are rows of one tile), and on long context speculation wins: at 30k
tokens 165 tok/s vs 73.5 for regular decode, at 8k — 187 vs 76.5 (single
request). The prompts here are this repo's sources, and the continuation copies
code from context, so draft acceptance is high; on other text the gain will be
smaller. Breakdown —
[`packed-attention-2026-09-24.md`](bench/results/packed-attention-2026-09-24.md).

`POST /v1/completions` accepts a prompt as text or token ids and returns
`logprobs` (up to 20); `ignore_eos` and `return_tokens_as_token_ids` work as in
vLLM. Used by `bench/diff_eval.py --engine openai --openai-token-ids`.

## Reproducing

From scratch. CUTLASS and cuda-shim are not in git, and `qwc-cuda` won't build
without them; details and version rationale —
[`docs/02-toolchain.md`](docs/02-toolchain.md).

```bash
git clone --depth 1 --branch v4.8.0 https://github.com/NVIDIA/cutlass third_party/cutlass
./scripts/gen-cuda-shim.sh third_party/cuda-shim
export PATH=/usr/local/cuda/bin:$PATH
```

Checkpoint — `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` from Hugging Face, 20.6 GB
in five shards; the script supports resuming. Layout and its check against the
architecture — [`docs/05-checkpoint.md`](docs/05-checkpoint.md).

```bash
./scripts/fetch-checkpoint.sh
```

No GPU or checkpoint needed:

```bash
cargo run -q -p qwc-core --bin plan
python3 -m unittest discover -s bench/tests -v
python3 bench/experiments.py run --suite comparison --dry-run
python3 bench/experiments.py run --suite ablations --dry-run
```

GPU required:

```bash
cargo run --release -p qwc-cuda --bin gpuinfo
cargo test --workspace -- --test-threads=1
```

Full series with confidence intervals. Requires the interpreter from an
environment with vLLM installed — it brings up the reference runtime for the
comparison suite:

```bash
/path/to/vllm-venv/bin/python bench/experiments.py run --suite ablations --repeats 7 --output bench/results/ablation-trials.jsonl
python3 bench/experiments.py report --input bench/results/ablation-trials.jsonl --output bench/results/ablations-with-ci.md
```

## Measurements

Protocol and its rationale — [`docs/11-experiments.md`](docs/11-experiments.md),
implementation — `bench/experiments.py`: a separate process per trial,
bootstrap CI, all trials written to JSONL. Raw measurements live in
`bench/results/`, one file per run, with date and conditions; numbers from
different sessions are not comparable — the machine drifts up to 5-6% per day.

## Repository layout

| path | what's there |
|---|---|
| `crates/qwc-core` | architecture as compile-time constants, VRAM plan, roofline model; CPU only |
| `crates/qwc-model` | safetensors via mmap, checkpoint check against the architecture |
| `crates/qwc-runtime` | KV/state cache, scheduler, batch layout; CPU-testable |
| `crates/qwc-cuda` | FFI, device memory, CUDA kernels (`cuda/*.cu`) and microbenchmarks |
| `crates/qwc-engine` | weight loading, step executor, MTP, `serve`, `eval` and bench binaries |
| `bench/` | experiment harness, diff-eval, vLLM baselines, corpora and `results/` |
| `scripts/` | checkpoint fetch, toolchain shim, baseline venv, CLI clients |
| `docs/` | design notes, in reading order |

## Documentation

| | |
|---|---|
| [`01-model-architecture.md`](docs/01-model-architecture.md) | Qwen3.8 architecture and implications for the engine |
| [`02-toolchain.md`](docs/02-toolchain.md) | CUDA, SM120, CUTLASS, known toolchain pitfalls |
| [`03-roofline.md`](docs/03-roofline.md) | measured bandwidth, decode ceilings, GEMM efficiency |
| [`04-deltanet.md`](docs/04-deltanet.md) | Gated DeltaNet: the recurrent decode step |
| [`05-checkpoint.md`](docs/05-checkpoint.md) | QUASAR-NVFP4 checkpoint layout |
| [`06-runtime.md`](docs/06-runtime.md) | two-resource cache and continuous batching |
| [`07-nvfp4-decode.md`](docs/07-nvfp4-decode.md) | NVFP4 decode: W4A16, W4A4 and fused MLP |
| [`08-weights.md`](docs/08-weights.md) | loading weights into VRAM |
| [`09-decode.md`](docs/09-decode.md) | token loop: prefill, CUDA graphs, continuous batching |
| [`10-differential-eval.md`](docs/10-differential-eval.md) | distribution matching against external runtimes |
| [`11-experiments.md`](docs/11-experiments.md) | experiment protocol: what we claim, how we measure, what's forbidden |
| `bench/results/` | one file per measurement, with date and conditions |

## License

MIT, see [`LICENSE`](LICENSE). The checkpoint and CUTLASS are not included and
are under their own licenses.

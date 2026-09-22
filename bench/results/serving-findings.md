# Serving throughput under concurrency — 2026-09-11

> **Single-session points, no confidence intervals.** The vLLM comparison here
> predates the protocol in
> [`docs/11-experiments.md`](../../docs/11-experiments.md): one or two runs per
> point, no variant-order rotation, no bootstrap CI, on a partially occupied
> card. These multipliers are superseded by the comparison suite and must not be
> quoted as a project result. Per-point conditions are given below.

200 requests, 256 prompt tokens, 128 generated tokens each, all submitted at
t=0 so both engines run saturated. Same model files, same GPU, greedy,
`detokenize=False`, CUDA graphs on, warmup before timing.
`bench/servebench.py`.

## Result

200 requests, 256 prompt tokens, 128 generated each, submitted at t=0.

| Concurrency | qwc tok/s | vLLM tok/s | ratio |
|---:|---:|---:|---:|
| 32 | 750 | 983 | 0.76× |
| 48 | 872 | — | — |
| 64 | 953 | 1002 | 0.95× |
| **80** | **1026** | 985 | **1.05×** |

Three independent runs at concurrency 80: qwc 1033.1 / 1026.6 / 1025.6, vLLM
973.3 / 980.6 / 976.7. The ~50 tok/s gap is well outside both spreads.

vLLM is saturated from concurrency 32 upward — 983 / 1002 / 985 is flat. qwc
keeps scaling, so the crossover is at concurrency ~72. Below that vLLM still
wins; the win is at high concurrency, which is what the target was.

Concurrency 96 does not fit: 96 sequences cost 14.2 GB of state and KV on top
of 16.25 GB of weights, against ~29.6 GB usable.

## Where it came from

Starting point this session was 699 tok/s at concurrency 64.

| Change | c=64 tok/s |
|---|---:|
| starting point | 699 |
| one forward pass per step instead of one per sequence | 886 |
| projections sized to live rows, not the whole arena | 894 |
| one recurrent launch for the whole decode group | 942 |
| `MAX_BATCH` 64 → 96, enabling concurrency 80 | 1026 (at c=80) |

**The fusion was the big one.** `execute_layout` used to launch a full 64-layer
pass for the decode batch and then one more *per prefill sequence* — at
concurrency 64 that was 521 steps but **821 weight passes**. Now a step is one
pass, mixed or not. Decode-only steps still take the CUDA-graph path, so the
1.49× batch-1 latency win is untouched.

Two details mattered as much as the idea:

- Only the DeltaNet scan is per sequence, because it touches recurrent state
  rather than weights. It gets its slice of the fused arena through a row
  offset applied in the `extern "C"` entry points, so the kernels still index
  from row zero.
- The one-token rows are contiguous from row zero and share a single recurrent
  launch. Routing each of them through its own scan first cost ~3000 kernel
  launches per step and gave back 48 tok/s when fixed.

The vocabulary projection runs once per step: the last row of every sequence is
gathered into a dense buffer first, instead of re-reading the 1.27 GB lm_head
per sequence.

## Correctness

Unchanged throughout. The eval corpus still reproduces 120/120 greedy tokens
against vLLM with MAE 0.364 and Jaccard 0.782 — identical to before any of this
work. The fused continuous-batching path produces the same tokens as the
single-sequence path for the same prompt, and 8 concurrent copies of a prompt
all agree. A GPU test covers the row offset: the same tokens scanned at offset
zero and at an offset give the same state and output, and leave neighbouring
rows untouched.

## What is left

Per-step efficiency is still the lever. At concurrency 64 a step moves ~27 GB
(16.25 GB weights, 9.6 GB DeltaNet state read+write, KV) which is 15 ms at the
card's bandwidth; measured steps are ~52 ms, so ~29% of roofline. The known
cause is the W4A4 GEMM: from `nvfp4bench` it reads weights at 793–1055 GB/s
while our own W4A16 kernel reaches 1379–1479 GB/s. Closing that is the next
large win and would lift every concurrency at once, including the ones where
vLLM still leads.

## Landed earlier this session

- Batch ceiling 32 → 96 (`MAX_BATCH`); the decode kernels accept 128, the
  binding constraint is memory at 0.148 GB/sequence at ctx 2048.
- `PREFILL_CHUNK_SIZE` 128 → 1024, with the kernel guards raised to match.

## Method

`bench/servebench.py`. Same model files, same GPU, greedy, `detokenize=False`,
CUDA graphs on, warmup before timing. vLLM gets `max_num_seqs` matched to the
concurrency under test rather than its 8192-token default batch shape.

# Score against vLLM under a shared memory budget — 2026-09-17

> **Single points from one session, without confidence intervals.** The comparison with
> vLLM here was taken before the protocol
> [`docs/11-experiments.md`](../../docs/11-experiments.md) existed: one or two runs per
> point, no variant order rotation and no bootstrap CI, on a partially busy
> card. These multipliers are superseded by a comparison suite run and must not
> be quoted as a project result. The conditions of each point are given below.

## Why the methodology changed

The previous comparison (`where-we-stand-2026-09-16.md`) was taken at one point and
carried a caveat: the card was partially held by someone else's process, vLLM lived under 24.5 GB and
starved for cache on mixed points, so part of our lead was its
starvation, not our kernels. "Re-measure on a free card" suggested itself.

That's the wrong conclusion. On the target server, rerankers,
indexing and whisper live next to the engine — **a cramped card is the working condition**, not
a measurement nuisance. Measuring on an empty card means measuring a mode that won't exist.

So instead of one point — a sweep over the **total per-engine budget**:
`bench/servebench.py --engine-budget-gb B`. For us that's an absolute
`--memory-limit-gb`, for vLLM it's converted to `gpu_memory_utilization` of
card capacity; both values are written to the artifact and verifiable.

And the earlier caveat turns around: our weight headroom (17.08 GB vs 20.5
for vLLM) is not "the unfair part of the lead" but the most effective part of the product
exactly in this mode.

## Conditions

RTX 5090, 34.19 GB. **The card was not freed**: ~4.7 GB held by the desktop and
someone else's service, the same for both engines across all runs.
Context 2048, KV fp8 for both, a unique prompt per request, everything submitted at t=0.
vLLM 0.28.0 with FlashInfer brought up (see below) — that's its best available
configuration. Our engine is **without speculation**: MTP lives in `bin/speculate.rs`
and isn't wired into the server yet.

## Viability boundary

8 requests, c=4, 256/32 — the numbers here only show whether the engine starts.

| budget | qwencore | vLLM | vLLM + MTP |
|---|---|---|---|
| 19 GB | 221.9 tok/s | — | — |
| 22 GB | 221.1 | **doesn't start** | **doesn't start** |
| 24 GB | 221.9 | **doesn't start** | **doesn't start** |
| 26 GB | 221.7 | 260.6 | 142.4 |
| 28 GB | 221.0 | 258.0 | 236.5 |

Minimum: **19 GB for us, 26 for vLLM**. The 7 GB difference is exactly what
remains for neighbors on the card.

## Chat point

160 requests, c=32, 256/128. All runs produced the full 20 480 tokens.

| budget | qwencore | vLLM | vLLM + MTP |
|---|---|---|---|
| 22 GB | **1173.6** tok/s, TTFT 7.2 s | doesn't start | doesn't start |
| 24 GB | 1166.4, TTFT 7.4 s | doesn't start | doesn't start |
| 26 GB | **1167.5**, TTFT 7.4 s | 632.2, TTFT 15.3 s | doesn't start |
| 28 GB | error, see below | 1083.2, TTFT 8.5 s | 515.8, TTFT 19.2 s |

**At 26 GB — the only budget where both work — we're 1.85x in
throughput and 2.1x in TTFT.**

The main thing in the table isn't the multiplier but the shape of the curve: our throughput is **flat**
from 22 to 26 GB (1173 / 1166 / 1167), while vLLM's drops from 1083 to 632 when
going from 28 to 26. We're insensitive to the budget in this range, it
isn't.

A counter-observation the table hides: vLLM's ITL p50 is **better**
than ours (15.0 vs 22.7 ms). It holds fewer sequences at once,
so each goes faster, while end-to-end throughput is half.

## Batch 1: here we lose

8 requests, c=1, 256/128, 26 GB budget for both.

| | tok/s | ITL p50 |
|---|---:|---:|
| qwencore (server, no speculation) | 78.3 | 12.50 ms |
| vLLM without MTP | 72.3 | 13.38 ms |
| vLLM + MTP, k=2 | 127.5 | 7.18 ms |
| **vLLM + MTP, k=3** | **133.9** | **5.97 ms** |

**A properly configured vLLM beats our server by 1.71x.** Without MTP our lead
at this point is 1.08x, not 1.57x as in the previous analysis. Part of the difference
is explained by context (8192 then, 2048 here), but the main thing is different: back then
vLLM was measured without FlashInfer and without speculation, i.e. in a configuration knowingly worse
for it.

Our speculation closes the gap with room to spare: the bench 2.04x (k=2) and 2.20x
(k=3) over 78.3 give 160-172 tok/s vs their 133.9. But it lives in
`bin/speculate.rs` and isn't wired into the server, so **as of today this table row
doesn't count as ours**.

## vLLM's MTP: doesn't improve throughput

At c=4 — 142 vs 260 at 26 GB; at c=32 — 516 vs 1083 at 28 GB, and at
26 GB it doesn't start at all: `max_num_seqs (32) exceeds available Mamba cache
blocks (19)`. As expected: at wide batch the step already amortizes the weights, while
speculation adds work and eats cache.

Hence the important point: the thesis "vLLM will enable MTP and cut our lead" is **not confirmed**
and doesn't apply to throughput at all. It has to be checked on batch-1 latency,
where speculation is supposed to help. That point hasn't been measured yet.

## What's unfair in this table

- **Our 28 GB run crashed** (`CudaError(1)`). With 4.7 GB taken by others,
  a 28 GB budget physically doesn't fit: we try to pick the limit, while vLLM
  profiles free memory and takes less than nominal. So the 28 GB row is
  **not comparable**; there is one comparable row — 26 GB.
- The first sweep gave us a minimum of 24 GB, and that was **a bench bug against us**:
  `servebench.rs` asked for a fixed 5 GB of KV on top of the weights, whereas
  vLLM takes the rest of the budget. Fixed: `--kv-cache-gb 0` means "all
  the rest", and that's substituted with a total budget. After the fix the minimum dropped
  from 24 to 19 GB.
- Our side is **without speculation**. The bench 2.04x (depth 2) and 2.20x
  (depth 3) at batch 1 aren't part of this table.

## How to bring up vLLM with MTP

Four layers, each silently breaking the previous conclusion:

1. MTP requires FlashInfer; 0.6.16.post3 is installed and imports;
2. `has_flashinfer()` considers it available only with `nvcc` on PATH —
   `/usr/local/cuda/bin` wasn't there;
3. JIT requires `ninja` — it's in `~/.venvs/baseline/bin`, but the venv was invoked by
   absolute path, so its bin didn't get onto PATH;
4. the build failed on `curand_kernel.h`: in CUDA 13 the curand headers moved out of
   the toolkit into the pip package `nvidia/cu13`.

At the fourth step substituting the whole `nvidia/cu13/include` gives "CUDA compiler
and CUDA toolkit headers are incompatible" — it's 13.0 there vs nvcc 13.1.
The solution is the same trick as in our `crates/qwc-cuda/build.rs`: a shadow include
tree of symlinks to the real toolkit, a patched
`crt/math_functions.h` for the glibc `rsqrt` conflict, plus symlinks only to
`curand*.h`.

```
export PATH=/usr/local/cuda/bin:$HOME/.venvs/baseline/bin:$PATH
export NVCC_APPEND_FLAGS="-I<shim>"
export MAX_JOBS=4
```

`MAX_JOBS` is mandatory: without it JIT launches nvcc per core, and on this
machine the CUTLASS build took all of host RAM — the OOM killer fired. Runs
should be kept under `systemd-run --user --scope -p MemoryMax=36G`.

**This also changes the quality measurements.** With FlashInfer vLLM takes `FLASHINFER`
attention instead of `TRITON_ATTN` and `FlashInferCutlassNvFp4LinearKernel` instead of
`CutlassNvFp4LinearKernel`. So `numerics-findings.md` from 11.09 and the conclusion
"we're 70 times closer to the truth" were taken in a configuration that no longer exists, and they
need re-measuring.

## What's next

1. **Speculation in our server** — it went from "nice to have" to mandatory: without
   it batch 1 is lost to a properly configured rival by 1.71x.
2. FP8 DeltaNet state: state pools at c=32 take 2.7 GB and raise
   our floor from 19 to 22 GB. A slot half the size lowers the floor and raises the step
   ceiling.
3. Re-measure numerics in the FlashInfer configuration.

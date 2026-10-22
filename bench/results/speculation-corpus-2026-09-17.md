# Speculation on an honest corpus and where the step really goes — 2026-09-17

> **Single points from one session, without confidence intervals.** The comparison with
> vLLM here was taken before the protocol
> [`docs/11-experiments.md`](../../docs/11-experiments.md) existed: one or two runs per
> point, no variant order rotation and no bootstrap CI, on a partially busy
> card. These multipliers are superseded by a comparison suite run and must not
> be quoted as a project result. The conditions of each point are given below.

All numbers in this file were taken in one session back to back, RTX 5090, ~4.6 GB of the card
held by the desktop and someone else's service, 26 GB budget for both engines, context 2048,
KV fp8. Numbers from other `bench/results` files are not comparable with these.

## What was broken

`servebench.rs` reset `hidden_for` based on the cumulative `completed`: after
the first request finished the condition was true forever, and only
every other step stayed speculative. The reset now uses the return value of
`complete_batch_multi`. On synthetic data, 8 requests, c=1, k=3: **106.9 -> 128.3
tok/s**, speculative steps 295/562 -> 335/353.

## A corpus instead of a progression

The synthetic `prompt_ids` is an arithmetic progression, and acceptance on it is
inflated. `bench/make_corpus.py` cuts prompts from real repository text
(documentation prose and sources — exactly what an agent server sees), the chunks
don't overlap, both engines read the same file via `--corpus`.

| 8 requests, c=1, 26 GB | synthetic | corpus |
|---|---:|---:|
| qwc, no speculation | 77.8 | 78.2 |
| qwc, k=3 | 128.3 | **119.2** |
| vLLM, no MTP | 72.3 | 70.5 |
| vLLM + MTP k=3 | 133.9 | **146.7** |

**The conclusion flips.** On synthetic data we looked ahead; on real
text their head guesses better and ours worse, and at batch 1 with speculation we
lose by 1.23x rather than win by 1.27x. Without speculation we're ahead by
1.11x.

Separately: the earlier "1.27x" figure compared our run on **one** request
(169.9) with a vLLM run on **eight** (133.9). At the same point
the difference is different.

The chat point had no inflation: 160 requests, c=32 — **1165.8 vs 618.5
(1.88x)**, TTFT 7.35 s vs 15.68 s. On synthetic data it was 1167.5 vs 632.2.

## What a speculative step consists of

4 requests, c=1, corpus. Draft and verification are timed with clocks (both phases
end with a copy to host, no separate sync needed).

| verification rows | verification, ms | draft, ms | step, ms | tokens/step | tok/s |
|---:|---:|---:|---:|---:|---:|
| 2 (k=1) | 14.82 | 1.49 | 16.31 | 1.826 | 107.1 |
| 3 (k=2) | 16.01 | 2.98 | 18.99 | 2.438 | 122.4 |
| 4 (k=3) | 17.85 | 4.49 | 22.33 | 2.891 | **123.4** |
| 5 (k=4) | 21.24 | 5.93 | 27.17 | 2.903 | 102.8 |
| 6 (k=5) | 22.21 | 7.38 | 29.59 | 3.030 | 98.6 |
| 8 (k=7) | 22.07 | 10.27 | 32.34 | 3.091 | 92.3 |

A regular decode step for comparison is 12.46 ms.

Two things that weren't visible before:

1. **Verification plateaus.** From six rows on it costs the same ~22 ms.
   Beyond the fifth row verification is almost free, and only the draft grows —
   linearly, 1.49 ms per depth. That's exactly why the k=3 optimum holds.
2. **There is no acceptance cliff.** The accepted-length histogram (`accept_histogram` in
   the artifact) gives per depth 0.834 / 0.740 / 0.713 at k=3 and 0.826 / 0.580 /
   0.662 / 0.755 / 0.700 at k=5. Acceptance is flat up to the fifth depth.

## Where the tax really sits

`stepprofile`, pure decode, the same card:

| phase | batch 1 | batch 4 | GB/s: 1 -> 4 |
|---|---:|---:|---|
| gemm.mlp_gate_up | 4.375 | 4.792 | 1467 -> 1339 |
| gemm.mlp_down | 2.255 | 3.035 | 1423 -> 1057 |
| gemm.la_in | 1.630 | 2.186 | 1398 -> 1042 |
| lm_head | 0.840 | 1.268 | 1514 -> 1004 |
| gemm.la_out | 0.785 | 1.008 | 1082 -> 843 |
| gemm.attn_qkv | 0.678 | 0.988 | 974 -> 669 |
| **sum of phases** | **13.90** | **16.96** | 1088 -> 919 |

Weight traffic meanwhile **doesn't change** — the same 15.1 GB. Four rows instead of
one cost 3.06 ms of pure lost bandwidth.

This is also the verification tax: verification on 4 rows costs 17.85 ms vs
13.90 for decode on one row, and 3.06 of the 3.95 ms difference is explained by exactly
this, not by the prefill route. Moving verification to the decode path would save
~0.9 ms, not 5 — this idea can be closed.

## Reprioritizing by measurement

**A cheap draft (shortlist) is not the first item.** The vocab projection is
0.82 of the draft's 1.49 ms, i.e. 2.46 of the step's 22.33 ms at k=3. Shortlist coverage
was measured on 25 600 actually generated tokens:

| shortlist | coverage |
|---|---:|
| context only (prompt + already emitted) | 0.814 |
| id < 8192 | 0.784 |
| id < 8192 ∪ context | 0.919 |
| id < 32768 ∪ context | 0.958 |

A shortlist miss is a rejected draft, so acceptance is multiplied by
coverage. Computing from the measured chains: shortlist 8k ∪ context gives
**+6.7%** (123.4 -> 131.7), 32k ∪ context — **+10.3%** (-> 136.1). Useful,
but not "+13% and close to 192 tok/s": that estimate kept acceptance
unchanged.

**W4A16 bandwidth at batch 2..4 is the first item.** If batch 4 held the bandwidth
of batch 1, the decode step would be 13.90 instead of 16.96, k=3 verification about 14.5 ms,
the step 19 ms, and batch 1 would reach ~152 tok/s (+23%). The same tax sits on
the chat point too, and exactly on the c=2..8 range where vLLM falls short.

The kernel already has `rows_per_thread=2` for batch>=3 (the narrow-batch drop sat
in shared activation loads, not in DRAM; there is no separate note on that sweep in
`bench/results`), and a microbenchmark then gave gate/up 1411 GB/s. The real shapes
at batch 4 give 669..1339, with `attn_qkv` and `la_out` the worst — they weren't in that
sweep.
That's the next job: a layout sweep over these shapes, not over gate/up.

## A small thing that no longer gets in the way

The memory limit is now clamped to the actually free VRAM
(`set_memory_limit` returns the real limit), and the KV pool also by
`cudaMemGetInfo` at capture time. A 40 GB request on a busy card shrinks to
27.55 and **works**, instead of `CudaError(1)` in the middle of loading. The effective
limit is written to the artifact as the `memory_limit_gb` field.

## What remained unchecked

- vLLM prefill was not re-measured with FlashInfer: the "1.27x" lag still
  relies on a configuration that no longer exists.
- Speculation at batch 2..8 was not tried: `Speculator` holds the KV of one
  sequence, the condition can't be lifted with a single line.
- FP8 DeltaNet state and the hybrid prefix cache were not touched.

---

# Continuation: what was done from the list and how it ended

Second half of the day, same card, same corpus, 26 GB budget for both.

## 1. W4A16 bandwidth at narrow batch

Geometry sweep over **all** decode shapes (`cargo run --release -p qwc-cuda
--bin nvfp4narrow`) — the previous sweep covered gate/up, but other shapes were the ones that dropped:

| shape | before, b=4 | after, b=4 |
|---|---:|---:|
| q+gate [12288, 5120] | 994 | **1254** GB/s |
| la_in [16480, 5120] | 1135 | **1319** |
| k or v [1024, 5120] | 254 | **285** |
| la_out, attn_out, down | — | already at their best |

The selection policy was rewritten by measurement: wide shapes go to 8x16 instead of
16x8 (half the warps per row — half the repeated activation reads
from shared), a narrow output takes the longest k tile that divides in_features.

Step result: pure decode b=4 16.96 -> 16.24 ms, b=1 13.90 -> 13.73.
End-to-end batch 1 with speculation 119.2 -> 124.3 tok/s (median of three runs).

**The "+23%" estimate was not confirmed: it came out +4.5%.** It assumed the
narrow-batch tax was untuned configuration. The sweep showed it's not: the kernel
is close to its ceiling on every shape, and the gap between b=1 and b=4 is held
by the kernel's design itself.

## 2. The W4A16 boundary was in the wrong place

This turned out bigger than the retune. `MAX_W4A16_BATCH` was 4, and from the fifth row everything
went to W4A4 — but above four the kernel simply **wasn't instantiated**, rather than
losing. I instantiated it and measured (ms per pure decode step):

| concurrency | W4A16 | W4A4 |
|---:|---:|---:|
| 5 | **18.63** | 21.67 |
| 6 | **20.48** | 21.80 |
| 7 | 24.94 | **22.39** |
| 8 | 30.52 | **22.27** |

The boundary was moved to 6. Before that **the fifth sequence reduced
throughput**: 4 rows in 16.3 ms is 245 tok/s, while 5 in 21.7 is 231.

As a side effect this also fixes speculation at depth 4: verifying five rows cost 21.0
ms, now 18.5, and k=4 rose from 112.9 to 131.2 tok/s (k=3 still
remains the optimum).

## 3. Vocab shortlist — done, +9%

Kernel `qwc_fp8_lm_head_subset` (a warp owns a row, activation in shared),
list = frequency prefix of the vocab ∪ context tokens, flag `--shortlist N`.
Checked against the full pass by the `subset_logits_match_full_pass_rows` test.

| shortlist | tok/s | draft, ms | tokens/step |
|---:|---:|---:|---:|
| off | 126.6 | 4.43 | 2.784 |
| 8 192 | 137.7 | 2.06 | 2.695 |
| **32 768** | **138.0** | 2.29 | 2.743 |
| 65 536 | 137.0 | 2.58 | 2.758 |

Acceptance dropped by only 1.5%, not 4% as estimated: the context part of
the list covers almost everything. Median of three runs: 126.6 -> 138.0, **+9.0%**.

Output unchanged: `speculate --verify` — "matched regular greedy token
for token", max|dlogit| of row 0 is 0.127 with a tolerance of 0.5.

## 4. Prefill re-measured with FlashInfer

Point 4096/8, c=16, context 8192, KV bf16, step width swept on both
sides.

| | best width | prefill |
|---|---|---:|
| qwencore | chunk 2048 | 11 780 tok/s |
| vLLM 0.28 + FlashInfer | max_batched_tokens 4096 | **13 655** |

The lag is **1.15x**, not 1.27x: the old number 14 802 was taken in a
configuration that no longer exists.

The sweep over our chunk didn't plateau at the 2048 cap, so the cap was
raised to 4096 and measured: 12 126 tok/s (+2.3%) at the cost of **0.75 GB** of arena. With
our viability boundary at 19 GB that's a bad deal — reverted to 2048.

## 5. An honest score against vLLM across the range

Corpus, 26 GB, 256/128, one session.

| point | qwencore | vLLM 0.28 | |
|---|---:|---:|---|
| c=1, no speculation | **79.2** | 70.5 | us 1.12x |
| c=1, speculation k=3 + shortlist | 138.5 | **146.7** (MTP k=3) | vLLM 1.06x |
| c=4 | 252.2 | **282.8** | vLLM 1.12x |
| c=8 | 369.7 | **529.9** | vLLM 1.43x |
| c=32 | **1168.8** | 618.5 | us 1.89x |
| prefill 4096/8, c=16 | 11 780 | **13 655** | vLLM 1.15x |

**The hypothesis about "vLLM failing at c=2..8" was not confirmed.** It fails
there only with MTP on (c=4: 260.4 vs 282.8 without it; c=8: 138.7
vs 529.9). Without MTP the middle of the range is its territory, and we're behind there.

Our only big win is high concurrency under a tight budget,
where it starves for cache. That's a real advantage for exactly the server
all this is for, but it can't be called general superiority.

## What's the biggest now

The middle of the range, c=4..16, where we're behind by 1.1..1.4x. W4A4 runs there, and
by measurement it gives 732 GB/s of effective weight bandwidth at batch 8 vs 1400+
achievable — i.e. roughly 2x headroom. It's the same lever that moves
the chat point.

## Left untouched

FP8 DeltaNet state, a per-step token budget for ITL p95, the hybrid prefix
cache, speculation at batch > 1.

---

# Middle of the range c=4..16: what was found and what wasn't

The goal was to close the 1.1..1.4x gap at c=4..8, where W4A4 gives 930..999 GB/s out of
an achievable 1605. Three hypotheses checked, two came up empty.

## Empty: W4A16 occupancy at wide batch

W4A16 at batch 7..8 collapses (a c=8 step takes 30.5 ms vs 22.3 for W4A4), and
the explanation seemed obvious: `__launch_bounds__(128, 8)` requires 32 registers per
thread, and the accumulators at kBatch=8 don't fit there. Measured by blocks per
SM (ms per pure decode step):

| blocks per SM | c=5 | c=6 | c=7 | c=8 |
|---:|---:|---:|---:|---:|
| **8** | **18.44** | **20.43** | 24.94 | 30.52 |
| 3 | 19.33 | 22.03 | 24.16 | 25.65 |
| 2 | 19.30 | 22.01 | 24.09 | 25.65 |

Occupancy does hurt at 7..8, but even the best variant (25.6 ms) is worse than
W4A4 (22.3). No reason to loosen the boundary — kept at 8.

## Empty: stream-K and M padding

The W4A4 GEMM is CUTLASS SM120, and stream-K turns on only when there are half as many tiles
as SMs (`is_narrow`). For `gate_up` (136 tiles on 170 SMs) and `la_in`
it never turned on. I widened the boundary to `tiles <= sm_count` — **no
difference**: 1014 -> 1009 GB/s at batch 8, i.e. within noise. Reverted.

The second idea came from `nvfp4bench`: the same `gate/up` matrix takes 49.5 µs
at M=8 and 36.0 at M=64, although the M tile is 128 in both cases and the traffic
is the same. If so, it pays to pad M to 64 with idle rows. I checked with a direct
measurement — **doesn't reproduce**: on la_in at batch 8 padding gives 1021 GB/s
vs 1065 without it. The discrepancy with `nvfp4bench` remains unexplained, but
there's no lever there.

## Found: kernel choice depends on shape, not just batch

`select_decode_kernel` took the shape as an argument and ignored it. A sweep of both
paths over all shapes showed each has its own boundary:

| shape | W4A16 wins up to |
|---|---|
| la_in [16480, 5120] | 6 rows |
| q+gate [12288, 5120] | 6 |
| la_out, attn_out [5120, 6144] | 6 |
| **down [5120, 17408]** | **4** (long K) |
| **k and v [1024, 5120]** | **always** (W4A4 holds 157 GB/s at any batch) |

The policy was rewritten by this table. Pure decode step:

| | c=6 | c=8 | c=16 |
|---|---:|---:|---:|
| before | 20.43 | 22.10 | 23.46 ms |
| after | **19.99** | **21.87** | **23.18** |

Not much, but free and backed by measurement. End-to-end c=6: 303.8 tok/s.

## Conclusion on the middle of the range

There are no cheap levers left there. The remaining gap is the CUTLASS configuration of
W4A4 at small M: 128x128x256 tile, ping-pong, one cluster. Moving it requires
a sweep of tiles, clusters and epilogue schedules, and that's a separate job of
a day or two, not an add-on for an evening.

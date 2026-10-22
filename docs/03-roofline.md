# Roofline: where exactly there is something to win

Baseline performance values are measurements on this RTX 5090;
ceilings and crossover are computed from them. The model lives in
[`crates/qwc-core/src/roofline.rs`](../crates/qwc-core/src/roofline.rs),
is printed by `cargo run -p qwc-core --bin plan`.

## CUTLASS NVFP4 GEMM measurement (example 79b, W4A4)

The shapes are real Qwen3.8 layers. The `% BW` columns are the bandwidth share
computed from the kernel's weight traffic: against the spec 1792 GB/s and against the measured
1605 GB/s. The denominator in all conclusions is **1605** — the rationale is in the section
"Measured bandwidth" below; the spec column is kept only
for cross-checking against other people's measurements.

| operation | M | time | TFLOPS | % of FP4 peak | % BW (1792, spec) | % BW (1605, measured) |
|---|---:|---:|---:|---:|---:|---:|
| gate/up | 1 | 59.3 us | 3 | 0% | 47% | **53%** |
| gate/up | 8 | 56.4 us | 25 | 2% | 50% | **55%** |
| gate/up | 32 | 48.7 us | 117 | 7% | 57% | **64%** |
| gate/up | 128 | 24.1 us | 946 | 56% | 116%* | 130%* |
| gate/up | 512 | 78.6 us | 1161 | 69% | 36% | 40% |
| gate/up | 2048 | 261.2 us | **1398** | **83%** | 11% | 12% |
| down † | 16 | 43.9 us | 65 | 4% | 64% | **71%** |
| down | 2048 | 260.3 us | **1402** | **84%** | 11% | 12% |
| qkv+gate | 2048 | 201.5 us | 1279 | 76% | 10% | 11% |

\* above 100% — the 50 MB matrix settled in the 96 MiB L2 between iterations. In real
inference the model weights are 17 GB and L2 doesn't help like that, so the decode numbers here
are rather optimistic.

† A different session and a different kernel instance: measured on 2026-09-12, cooperative +
stream-K scheduler for narrow shapes —
[`nvfp4-narrow-gemm.md`](../bench/results/nvfp4-narrow-gemm.md). Not strictly comparable
with the other rows of the table: the same HEAD dropped about
5% within a day, as the source itself warns. Bandwidth is recomputed from time with the same
4.5-bit traffic as the whole table; the source document gives 1016 GB/s for this
point because it counts 4 bits per weight and ignores the FP8 block
scales. Before stream-K the same point cost 57.0 us — 880 GB/s, i.e. 49%
of spec and 55% of achievable.

### Two conclusions

**Prefill: CUTLASS squeezes out 84% of FP4 peak.** Our own GEMM won't beat that —
this confirms the chosen strategy "CUTLASS for GEMM, our own kernels around it".

**Decode: at M = 1..32 only 53-64% of achievable bandwidth is reached**
(47-57% of spec) — the range bounds come from the M=1 and
M=32 rows of the same `gate/up` measurement. That's roughly a 2x gap. The kernel doesn't
even saturate L2, i.e. it is bound not by bandwidth but by latency and occupancy —
typical for a GEMM scheme applied to a GEMV-shaped problem. This is where
the main ITL win lay — and it has since been taken by a GEMV kernel, not by the CUTLASS
GEMM path (see below).

The best measured at narrow M for CUTLASS itself is `down` with the
stream-K scheduler: 71% of achievable bandwidth (footnote †, different session). So
picking the scheduler per shape closes the gap partially, but not fully.

### The gap is closed by a GEMV kernel, not GEMM

The table above is entirely about the CUTLASS GEMM path, and **the engine doesn't use it at batch 1**:
there our own W4A16 runs (`nvfp4::Linear::forward_w4a16`, `swiglu_w4a16`).
The forecast "a specialized FP4 GEMV should reach 85-90% of achievable bandwidth"
stood here since 09.09 as an expectation — on wide shapes it has been **reached**:

| measurement | GB/s | % of 1605 |
|---|---:|---:|
| `nvfp4bench`, W4A16, one row | 1513 | 94%‡ |
| batch-1 step: `lm_head` | 1514 | 94% |
| batch-1 step: `gemm.mlp_gate_up` | 1467 | 91% |
| batch-1 step: `gemm.mlp_down` | 1423 | 89% |
| batch-1 step: `gemm.la_in` | 1398 | 87% |
| batch-1 step: `gemm.la_out` | 1082 | 67% |
| batch-1 step: `gemm.attn_qkv` | 974 | 61% |
| **batch-1 step: phase-weighted** | **1088** | **68%** |

Sources: [`speculation-corpus-2026-09-17.md`](../bench/results/speculation-corpus-2026-09-17.md)
(per-phase `stepprofile` breakdown) and
[`speculative-decode-2026-09-16.md`](../bench/results/speculative-decode-2026-09-16.md)
(microbenchmark). Both are different sessions from the CUTLASS table above.

‡ The microbenchmark percentage is inflated by about 3 points. The `nvfp4bench` working set
is 4 matrices, 141-200 MB, and part of it stays in the 96 MiB L2. For
that size `gpuinfo` gives 1656 GB/s (256 MiB buffer), not 1605, and the CUTLASS
GEMV in one batched launch reaches 1654 on it. Against 1656 the same 1513 is
91%. This doesn't affect the "batch-1 step" rows: there a step reads all 15 GB
of weights.

**The specialized CUTLASS GEMV is on par.** `91_fp4_gemv`, run in
the same series as `nvfp4bench` and with one launch per matrix like ours, gives
0-2.5% in its favor: gate/up 33.00 vs 32.96 µs, `down` 32.25 vs
33.07, q+gate 24.18 vs 24.34
([`cutlass-gemv-2026-10-09.md`](../bench/results/cutlass-gemv-2026-10-09.md)).
So the 1.8x lead here is over a GEMM scheme on a GEMV problem, not over CUTLASS
in general. We keep our own W4A16 for the format: the CUTLASS GEMV takes and returns FP4,
and its `batch` means different matrices, not several tokens on one.
The same measurement showed that 4 matrices go 6-12% faster in one batched launch
than in four launches. The tail of each launch is a separate decode lever.

The rest of the gap moved to two other places, and they're worth naming separately:

* **narrow projections** — `attn_qkv` 61% and `la_out` 67% pull the weighted step from
  90 down to 68%;
* **W4A4 at batch >= 6**, where the concurrency range goes: it holds about
  1000-1014 GB/s (62-63%) at practically any width.

Still unmeasured: our W4A16 on the `attn_qkv`/`la_out` shapes
after the layout sweep, and CUTLASS W4A4 configuration at small M.

## Decode ceiling for the whole model

You can't substitute the resident footprint for step traffic here. **17.08 GB** is resident
on the GPU, but regular decode doesn't read the MTP head and does a lookup
of one embedding row instead of a pass over the whole table. A step actually streams
**14.96 GB** of weights: the NVFP4 body and the FP8 `lm_head`. The formula separately adds
DeltaNet state read/write and KV reads.

At the measured 1605 GB/s (rationale below) the current roofline is:

| batch | ctx | ITL at 100% achievable BW | ITL at 53% | tok/s ceiling |
|---:|---:|---:|---:|---:|
| 1 | 2K | **9.42 ms** | 17.77 ms | **106** |
| 1 | 32K | 10.04 ms | 18.95 ms | 100 |
| 8 | 2K | 10.10 ms | 19.06 ms | 792 |
| 16 | 2K | 10.88 ms | 20.53 ms | 1470 |
| 32 | 2K | 12.44 ms | 23.48 ms | **2572** |
| 32 | 8K | 16.46 ms | 31.05 ms | 1944 |

The 53% value is the currently achieved share of achievable bandwidth for CUTLASS
`gate/up` at M=1: 50.14 MB of weights in 59.3 us — that's 845 GB/s, i.e. 47%
of the spec 1792 GB/s and 53% of the measured 1605 GB/s (the M=1 row of the table above).

**Caveat on the "ITL at 53%" column.** The 53% share was measured on one kernel —
`gate/up` at M=1 — and is extrapolated here to the whole step, including
DeltaNet state read/write and KV cache reads. Those two streams don't go through
the CUTLASS GEMM, and their own efficiency wasn't measured, so
the column reads not as an ITL prediction but as an estimate "if all of the step's
traffic ran at narrow-GEMM efficiency". The real lower bound is the
"ITL at 100% achievable BW" column: it relies only on measured bandwidth
and computed traffic.

Note: context barely affects ITL at batch=1 (9.42 vs 10.04 ms
between 2K and 32K). The hybrid architecture works as advertised here — only 16 layers
of 64 read the growing KV cache.

## What a decode step consists of at batch=32, ctx=2048

| | traffic | share |
|---|---:|---:|
| weights actually read by the step | 14.96 GB | 75% |
| **DeltaNet state int8** | **2.87 GB** | **14%** |
| KV cache | 2.15 GB | 11% |

The recurrent state is read and rewritten in full for every token of every
sequence. The current int8 format with a per-row scale is half the cost of
BF16; the measured ablation gave 25.72 -> 24.67 ms on a batch=32 step and saved
1.13 GB of VRAM. Details and accuracy check — in
[`bench/results/delta-state-8bit.md`](../bench/results/delta-state-8bit.md).

## The main strategic conclusion

The transition from memory-bound to compute-bound for NVFP4 happens at **batch ~294**.
That value is printed by `compute_bound_batch(Dtype::Nvfp4)` from
`crates/qwc-core/src/roofline.rs`: machine balance is 1676 TOPS / 1605 GB/s
= 1044 FLOP/byte, and actual NVFP4 traffic is 4.5 bits per weight
including FP8 block scales.

The project's whole target range — concurrency 1..32 — is **deep in the
memory-bound regime**. Therefore:

* FP4's compute advantage (1676 TOPS) barely matters in our scenario;
* FP4 is valuable exactly because it makes weights smaller — that's traffic, not arithmetic;
* optimize **bytes, not FLOPs**: fusion, no intermediate
  buffers, cache format, GEMV instead of GEMM;
* hence tensor core utilization is a bad progress metric for this project,
  and the achieved share of memory bandwidth is a good one.

---

## Measured bandwidth

`cargo run -p qwc-cuda --bin gpuinfo` on this card:

| buffer | GB/s | % of spec | |
|---:|---:|---:|---|
| 32 MiB | 1930 | 108% | fits in L2 |
| 64 MiB | 3247 | 181% | fits in L2 |
| 96 MiB | 3886 | 217% | exactly the L2 size |
| 128 MiB | 1502 | 84% | |
| 256 MiB | 1656 | 92% | |
| 1 GiB | 1602 | 89% | |
| 4 GiB | **1605** | **90%** | steady-state DRAM |

Copy (read + write): 1408 GB/s, 79% of spec.

Conclusions:

* **Achievable DRAM read is 1605 GB/s, not 1792.** All estimates in
  `roofline.rs` use it (`ACHIEVABLE_BANDWIDTH`); the spec number is
  unreachable in principle and useless as a denominator.
* **L2 delivers 3886 GB/s — 2.4x DRAM.** But 96 MiB is incomparable with
  14.96 GB of weight traffic. L2 can only be exploited within one layer,
  not across steps.
* Writes are noticeably more expensive than reads, which matters for DeltaNet: its state
  is both read and rewritten in full on every step.

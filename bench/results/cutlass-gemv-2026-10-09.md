# Our W4A16 vs CUTLASS GEMV at batch 1 — 2026-10-09

## Summary

- **We're at parity with the CUTLASS GEMV, not ahead.** With the same launch
  scheme (4 matrices — 4 launches) `91_fp4_gemv` and our W4A16 differ by
  0-2.5%: gate/up even, q+gate −0.7%, `down` −2.5% in CUTLASS's favor.
  The 1.8x lead (gate/up, 33.0 vs 59.3 µs) is a lead over the CUTLASS GEMM path
  (example 79b, W4A4, 53-64% of bandwidth), not over CUTLASS in general.
- **The tail of each launch costs 6-12%.** The same 4 matrices in one batched launch
  CUTLASS runs at 1640-1650 GB/s instead of 1460-1555. The difference is about
  2.7 µs per GEMV: the wave tail and the gap between kernels. The kernel has nothing to do with it
  here; the lever is PDL or fusing adjacent projections.
- **1605 GB/s is not the ceiling for this bench.** Batched CUTLASS gives 1654,
  exactly `gpuinfo` on a 256 MiB buffer (1656): the bench's working set
  (141-200 MB) partially stays in the 96 MiB L2. The "of 1605" percentages for
  `nvfp4bench` are inflated by about 3 points. This doesn't extend to the per-phase step
  breakdown: there the working set is all 15 GB of weights.

## Conditions

- RTX 5090, CUDA 13.1, CUTLASS 4.8.0, qwc — `9cee687`. Desktop open,
  no other GPU compute; load 1.7-3.1, `next-build` finished before
  the series started.
- Script — [`bench/cutlass/gemv-series.sh`](../cutlass/gemv-series.sh),
  7 ABBA rounds: odd rounds run `nvfp4bench` first, even ones CUTLASS.
  Raw output — [`cutlass-gemv-2026-10-09.log`](cutlass-gemv-2026-10-09.log).
- `91_fp4_gemv` is patched
  ([`91_fp4_gemv_bench.patch`](../cutlass/91_fp4_gemv_bench.patch)): the
  stock example has one warmup and 10 iterations. The patch measures like
  `nvfp4bench` — 20 warmups, 7 series of 40 passes, median — and adds
  `--per_matrix`, where 4 matrices go as four launches with `batch_count=1`.
  The check against the CUTLASS host reference passes (`test pass`) once before
  the series.
- Both kernels have 4 different matrices, cycled through. GB/s is weight traffic only
  (FP4 plus an E4M3 scale per 16 elements), decimal, one definition for
  both.

## Result

Median over 7 rounds, µs per matrix (weight GB/s):

| shape | qwc W4A16 | CUTLASS, per launch | CUTLASS, batched | qwc vs CUTLASS |
|---|---:|---:|---:|---:|
| gate/up 17408×5120 | 32.96 (1521) | 33.00 (1519) | 30.31 (1654) | 1.00x |
| down 5120×17408 | 33.07 (1516) | 32.25 (1555) | 30.49 (1644) | 0.975x |
| q+gate 12288×5120 | 24.34 (1454) | 24.18 (1464) | 21.61 (1638) | 0.993x |

Spread across rounds (min-max): qwc gate/up 32.92-34.71, CUTLASS 32.95-35.98.
Both benches are bimodal: some rounds have a slow mode at ~5%. The median
over rounds cuts it off, a single run doesn't.

A preliminary series in the same session, the same way but outside the script, gave
the same within a percent: qwc/CUTLASS 0.993, 0.971, 0.997.

## Why the CUTLASS GEMV doesn't replace our path

- Its input and output are FP4 with scale recomputation in the epilogue. Our input and output
  bf16.
- Its `batch` means different matrices, each with its own vector. We need one
  matrix for several tokens: decode batch and speculation verification rows.
- Weight scales are in its swizzled layout, not the checkpoint layout.

So here it's a reference for comparison, not a replacement candidate.

## Caveats

- Weights in `nvfp4bench` are zero (`Linear::zeroed`), in CUTLASS random.
  Read volume doesn't change because of that. DRAM power changes, but on a
  memory-bound kernel the card doesn't hit the power cap.
- CUTLASS's activation is FP4 (~2.5 KB at `K=5120`), ours bf16 (~10 KB). That's
  0.02% of weight traffic.
- The comparison is at the kernel level. The end-to-end step is measured separately: there narrow
  projections have their own efficiency (`attn_qkv` 61%, `la_out` 67%), and the CUTLASS GEMV
  wasn't checked on them.

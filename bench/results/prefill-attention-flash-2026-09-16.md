# Prefill attention: a 64-row tile — 2026-09-16

After the WY scan, attention became the main cost item of prefill: 36.8% of the step
vs 9.6% for the scan. The MMA kernel of 14.09 split the work crosswise — 16 query
rows per CTA, four warps cut along keys in QK^T and along dimensions in PV.
Because of this the computation went through shared (S there, back for softmax, P there
again), every 32 keys had two barriers, and a staged KV tile
served only 16 query rows.

The new form: **a 64-row tile, a warp owns its 16 rows and all of the
tile's keys**. Softmax lives in the warp's registers (a butterfly over four
lanes with the same grp), P goes from the QK^T accumulator into operand A without
reshuffling, barriers remain only for staging. The same staged KV tile
serves four times as many MMAs.

## Kernel, 2048-row chunk, 16 layers, bf16 KV

| position | 16-row tile | 64-row tile | + ldmatrix and vector staging | |
|---:|---:|---:|---:|---|
| 0 | 23.36 ms | 17.64 | **13.78** | 1.69x |
| 1024 | 42.99 | 30.85 | **24.17** | 1.78x |
| 3584 | 93.75 | 61.46 | **49.66** | 1.89x |
| 7680 | 173.18 | 109.61 | **91.50** | 1.89x |

On fp8 KV: 76.58 -> 49.61 ms at position 3584 (1.54x). The gap between fp8 and
bf16 closed — the kernel is no longer bound by cache conversion.

Two changes after the form change:

1. **`ldmatrix` instead of scalar reads.** Fragments were assembled two bytes at a time:
   128 MMAs came with about 450 shared accesses. `ldmatrix` reads
   eight 8x8 rows with one instruction and lays them out exactly as
   `mma.m16n8k16` expects; the `.trans` variant transposes V on the fly, so
   it no longer needs transposing during staging. 61.5 -> 55.7 ms.
2. **Vector KV staging.** `cache_to_float` read the cache element by element — 16384
   two-byte loads per CTA per tile. Now eight elements per
   access (float4 for bf16, uint2 for fp8). 55.7 -> 49.7 ms.

Checked and rejected by measurement: `ldmatrix.x4` instead of `.x2` (four
matrices per instruction) — 49.7 vs 50.1 ms, i.e. nothing; the kernel is no longer
bound by issuing load instructions. A 16-key KV tile instead of 32 —
53.2 ms, the extra barriers cost more than the occupancy gains.

## Engine step, chunk 2046 at position 3584

| phase | before | after |
|---|---:|---:|
| attn.prefill | 79.86 ms (36.8%) | **49.63 ms (27.0%)** |
| gemm.mlp_gate_up | 38.95 (18.0%) | 38.19 (20.8%) |
| delta.scan_prefill | 20.91 (9.6%) | 20.45 (11.1%) |
| whole step | 212.71 ms | **183.26 ms** |

## End-to-end prefill (4096/8, c=16, chunk 2048)

| | 16-row tile | 64-row tile |
|---|---:|---:|
| bf16 KV | 10 837 / 10 736 tok/s | **11 719 / 11 685** |
| TTFT p50 | 6112 ms | **5685** |

fp8 KV gives 11 748 tok/s — the same.

Over the day (WY scan plus this): prefill 8559 -> 11 719 tok/s, +37%.

Numerically the kernel is equivalent to the old one: diff-eval against a saved
vLLM 0.28 run — top-1 1.000 on all 14 cases, Jaccard 0.789 vs 0.782,
logprob MAE 0.374 vs 0.364.

## What's left

The kernel runs at ~75 TFLOP/s against a peak of about 210. It's now bound by
occupancy: 67.8 KB of shared per CTA (query tile 33.8 + K and V 16.9 each) — that's
one block per SM, four warps. Two blocks would give twice as many warps for
latency hiding, but for that the layout must move from padding to an
xor-swizzle: without padding the tiles take 49.2 KB, with it 50.7, and that's right
across the threshold.

The second untouched thing is `cp.async` with double buffering: right now a warp
waits for staging at a barrier instead of computing the previous tile.

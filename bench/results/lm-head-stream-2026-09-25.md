# Batched `lm_head` at DRAM bandwidth — 2026-09-25

## Summary

- **Kernel: 3.17 → 0.85 ms at batch 32 (3.75x), 4.42 → 1.12 ms at batch 64.**
  At batch 32 that's 1.50 TB/s — the card's bandwidth ceiling.
- **Pure decode step c=32: 23.68 → 20.85 ms** (`stepprofile`), the
  `lm_head` phase 3.33 → 1.05 ms, its share of the step 13% → 4.6%.
- **servebench c=32: 1139 → 1242 tok/s (+9%)**, ITL p50 23.2 → 20.9 ms,
  TPOT p50 26.5 → 24.3 ms. Three interleaved rounds, one session.
- Logits unchanged: diff_eval at `--batch 32`, old kernel vs new —
  MAE 0.000, max Δ 0.000, top-1 1.000.

## What was wrong

The old MMA kernel held 380-450 GB/s and barely depended on batch
(`lmheadbench`: 2.79 ms at 16, 3.17 at 32). So the bottleneck wasn't the tensor
cores but latency: a 128×32 weight tile was put into shared with byte
loads (16 byte loads per thread), with conversion to bf16 and two
barriers per tile, without prefetching the next one. In flight there was ~1.5 KB per SM
vs the ~6 KB needed by Little's law.

## What was done

- Weights go from global memory straight into A fragment registers, 16 bytes
  per load, without shared. For this the K order is permuted within a 64-column
  chunk: a thread takes 16 consecutive bytes of a row, the B fragment reads
  hidden with the same permutation.
- Prefetching a whole hidden chunk (128 columns, 4 loads of 16 bytes per
  thread) ahead: the next chunk is in flight while the barrier and math of the
  current one run.
- Only hidden goes through shared, with a row offset of 4 bf16 — reading
  the B fragment without bank conflicts.
- A hidden width not divisible by 128 goes to the old scalar batched kernel.

## Geometry sweep (`lmheadbench`, ms)

| variant | b=16 | b=32 | b=64 | b=80 |
|---|---:|---:|---:|---:|
| before | 2.790 | 3.169 | 4.422 | 4.551 |
| registers, 64-column prefetch, 2 m-tiles per warp | 1.308 | 1.330 | 1.413 | 1.572 |
| same, 1 m-tile | 1.246 | 1.279 | 1.409 | 1.472 |
| same, 4 m-tiles | 1.250 | 1.248 | 1.386 | 1.631 |
| 128-column prefetch, 2 m-tiles | 0.869 | 0.934 | 1.163 | 1.352 |
| **128 prefetch, 1 m-tile** | **0.834** | **0.845** | **1.137** | **1.219** |
| 256 prefetch, 1 m-tile | 0.835 | 0.866 | 1.056 | 1.405 |

Tile geometry decides almost nothing; prefetch depth decides.

## servebench c=32 (160 requests, 256/128, KV fp8, `--kv-cache-gb 3`)

| round | before tok/s | after tok/s | ITL p50 before / after |
|---|---:|---:|---:|
| 2 | 1123 | 1227 | 23.45 / 21.15 |
| 3 | 1157 | 1257 | 22.93 / 20.77 |
| 4 | 1136 | 1243 | 23.16 / 20.97 |

The first round didn't come up for either: OBS started (+0.8 GB), the budget was cut
to 23 GB, and the default KV didn't fit. `--kv-cache-gb 3` covers
32 × 384 tokens with room to spare.

## What else

`stepprofile` c=32 after the change: sum of phases 22.76 ms, `norm` 1.99 ms (8.7%) —
now the third-largest phase after `mlp_gate_up` and `mlp_down`.

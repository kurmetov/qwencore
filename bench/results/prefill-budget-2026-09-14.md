# Prefill budget: where 118 µs per token go — 2026-09-14

> **Single points from one session, without confidence intervals.** The comparison with
> vLLM here was taken before the protocol
> [`docs/11-experiments.md`](../../docs/11-experiments.md) existed: one or two runs per
> point, no variant order rotation and no bootstrap CI, on a partially busy
> card. These multipliers are superseded by a comparison suite run and must not
> be quoted as a project result. The conditions of each point are given below.

After MMA attention, end-to-end prefill ran at 7488 tok/s vs 14656 for vLLM 0.28 —
a 1.96x lag. Moving on blindly isn't possible: we need to know what
a token consists of.

## Step breakdown

`stepprofile --concurrency 2 --prefill-tokens 2046 --prefill-start 3584
--context 8192`, i.e. a full arena chunk at a late prompt position.
A step is 241.2 ms for 2046 tokens — **117.9 µs/token**.

| phase | ms | share | µs/token |
|---|---:|---:|---:|
| attn.prefill | 78.27 | 32.4% | 38.3 |
| delta.scan_prefill | 48.79 | 20.2% | 23.8 |
| gemm.mlp_gate_up | 38.20 | 15.8% | 18.7 |
| gemm.mlp_down | 20.61 | 8.5% | 10.1 |
| gemm.la_in | 13.12 | 5.4% | 6.4 |
| swiglu | 8.70 | 3.6% | 4.3 |
| norm | 7.91 | 3.3% | 3.9 |
| other 14 phases | 25.62 | 10.8% | 12.5 |

There is no "unexplained remainder": projections together are 41.2 µs/token,
attention 38.3, scan 23.8, everything else 14.6. The earlier estimate of a 43 µs remainder
was an artifact of comparing the end-to-end figure (averaged over all prompt
positions) with per-phase measurements at one position — you can't count that way.

The average token of a 4096 prompt with chunk 2048 costs less than the profile above,
because the first chunk sees an empty context: attention averages to
17.2 µs/token, and the budget becomes **~97 µs/token** — projections 41.2,
scan 23.8, attention 17.2, other 14.6. Measured end to end — 9313 tok/s
(107 µs/token), the difference goes to the scheduler, host and decode tail.

Two phases run at the ceiling and there's nothing to touch: `swiglu` — 13.6 GB in 8.7 ms,
i.e. 1.57 TB/s, `lm_head` — 1378 GB/s.

## What chunk size changes

No phase except the scan is indifferent to M. Projections at M=511 run at
1097 TFLOP/s, at M=2048 at 1350 (NVFP4 peak on the 5090 is 1676). Attention on
a large tile re-reads KV less often.

| chunk | projections, µs/token | attention, µs/token | sum |
|---:|---:|---:|---:|
| 512 | 44.1 | 23.3 | 67.4 |
| 1024 | 38.8 | 19.7 | 58.5 |
| **2048** | **35.8** | **17.2** | **53.0** |

Attention is recomputed over the whole 4096 prompt: chunk 512 is eight launches at
positions 0..3584, chunk 2048 is two. Launch cost is linear in position, so
only full prompts can be compared, not a single position.

A measurement on the engine confirmed the microbenchmarks. Same position 3584, prompt 4096,
32 requests, KV bf16:

| | chunk 512 | chunk 1024 | chunk 2048 |
|---|---:|---:|---:|
| end-to-end prefill | 7603 tok/s | 8772 | **9188** |
| engine steps | 265 | 137 | 73 |
| TTFT p50 | 8.59 s | 7.48 | 7.23 |
| ITL p50 | 66.2 ms | 111.9 | **207.9** |
| projections (breakdown) | 52.0 µs/token | | **41.2** |
| scan (breakdown) | 23.6 µs/token | | 23.8 |

**+21% prefill — and 3x worse ITL.** The price is fair: a decode row waits for the whole
chunk. So the scheduler budget is exposed as `--prefill-chunk` (the cap is
arena capacity, `PREFILL_CHUNK_SIZE`), rather than nailed to it.

On chat the budget means nothing: 256-token prompts, c=32, 128 requests —
1123.8 tok/s with chunk 512 vs 1113.8 with 2048, same ITL. Short
prefills were already packed several per step.

Done: `PREFILL_CHUNK_SIZE = MAX_STEP_ROWS = 2048`, row caps consolidated into
one `limits.cuh` (`kMaxStepRows`) instead of seven `1024` literals across kernels,
partition_count sizing left on decode rows (`kMaxDecodeRows`).
Shapes beyond the old cap are covered by the tests
`quantizer_and_w4a4_match_oracle_past_the_old_row_cap` and
`prefill_mma_matches_the_tiled_kernel_on_a_full_arena_chunk`.

Merging `gate`+`up` into one matrix at M=2048 isn't needed: 482.3 µs vs
244.3 + 243.7 = 488.0, i.e. 1%. At M=511 it gave 12% (135.3 vs
154.1) — the effect was not the merge but the narrow M.

## What big items remain

1. **DeltaNet scan — 23.1 µs/token and independent of chunk.** The recurrent
   form does 4.7 MFLOP per token per layer and delivers about 10 TFLOP/s: it's
   sequential over tokens and doesn't touch tensor cores. The chunked form
   (WY representation, as in FLA and vLLM) computes a chunk of 64 tokens with
   matrices: roughly 1.6x more FLOPs, but they run on tensor
   cores. Even at 100 TFLOP/s that's 3-4 µs/token. Potential: **-19 µs/token**.
2. **Attention — 17.2 µs/token after chunk 2048.** Its useful work is
   3.3 TFLOP per prompt (the causal half), i.e. the kernel runs at ~47 TFLOP/s
   against a bf16 peak of about 200. Causality is already respected per tile: the KV loop is
   bounded by the max context of the tile's rows.

   I checked and rejected the hypothesis "bound by KV re-reads": the kernel can
   serve several heads of a group with one KV tile (`kMmaGroups`), and that
   changes traffic by a multiple. Measured at chunk 2048:

   | kMmaGroups | position 0 | 1024 | 3584 | 7680 |
   |---:|---:|---:|---:|---:|
   | 1 | 18.83 ms | 34.82 | 77.46 | 142.32 |
   | 2 | 21.37 | 39.62 | 86.27 | 161.16 |
   | 3 | 18.59 | 34.25 | 73.31 | 138.98 |

   Three times fewer KV reads give 3-5% on long context and zero on short:
   register pressure eats the gain. So it's not traffic but the
   plumbing: a tile is 16 rows and 32 keys, each such tile has two
   `__syncthreads`, a pass of S through shared, a read back for softmax and a write of
   P to shared again. The right form is a 64-row tile where a warp owns
   its 16 rows and all keys: then softmax lives in the warp's registers,
   P doesn't go through shared, and barriers remain only for KV staging.
   Potential: **-8 µs/token**, a rough estimate.
3. **Remainder.** Not analyzed yet; see above.

Projections are 41.2 µs/token, the biggest row, but also the closest to the ceiling:
1350 TFLOP/s of the 1676 peak. About fifteen percent is left there, and it's expensive.

The work order is determined by this list, not by convenience: the scan first.
The chunked form's math has already been checked against the recurrent one on CPU —
`reference::prefill_chunked_fp32`, test `chunked_wy_form_matches_the_recurrent_scan`.

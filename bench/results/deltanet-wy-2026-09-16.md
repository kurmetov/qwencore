# Chunked (WY) DeltaNet scan in prefill — 2026-09-16

The recurrent scan goes over tokens sequentially and doesn't touch the tensor
cores: 4.7 MFLOP per token and layer, about 10 TFLOP/s. The chunked form computes
the same 64 tokens with matrices — roughly one and a half times more FLOPs, but they
land on MMA. The prefill budget (`prefill-budget-2026-09-14.md`) put the scan
first in the queue: 23.8 µs/token of ~97.

## Summary

Microbenchmark `deltaprefillbench`, one sequence, 48 layers, measurements
back to back in one session:

| tokens | recurrent | WY | |
|---:|---:|---:|---|
| 128 | 23.13 µs/token | 23.42 | 0.99x |
| 256 | 22.74 | 13.93 | 1.63x |
| 512 | 22.34 | **10.19** | 2.19x |
| 1024 | 21.69 | **10.48** | 2.07x |

At 128 tokens there's no gain: that's two chunks, and six prep launches per
layer don't pay off. From 512 tokens on the form reaches its 1.9x.

Engine step, `stepprofile --concurrency 2 --prefill-tokens 2046
--prefill-start 3584 --context 8192`:

| | recurrent | WY |
|---|---:|---:|
| delta.scan_prefill | 50.04 ms (20.3%) | **20.96 ms (9.7%)** |
| whole step | 245.97 ms | **213.86 ms** |

End-to-end prefill, `servebench --prompt-tokens 4096 --max-new 8
--concurrency 16 --prefill-chunk 2048`, bf16 KV, median of three runs:

| | recurrent | WY | |
|---|---:|---:|---|
| prefill | 9195 tok/s | **10627 tok/s** | +15.6% |
| TTFT p50 | 7262 ms | **6251 ms** | −14% |

Chat point 256/128 c=80 (prompt shorter than two chunks): 1643 vs 1669 tok/s,
ITL 33.0 vs 32.8 — within noise, no regression.

Numerically the path is equivalent to the old one: diff-eval against a saved
vLLM 0.28 run gives top-1 1.000 on all 14 cases, Jaccard 0.790 vs 0.782,
logprob MAE 0.367 vs 0.364. The verdict there is FAIL for both, for a known
reason — the reference itself quantizes activations to FP4
(`numerics-findings.md`).

## How it came together

The first working version was **4.6x slower** than the recurrent one (100
µs/token). Three steps driven by measurement, not intuition:

1. **Forward substitution in value space — 66 of 102 µs/token.**
   It solved the system for all 128 value dimensions at once, re-reading coefficients
   from global memory: 50 MB per layer chunk. Replaced with an explicit inverse of the
   unit lower-triangular matrix (64x64, entirely in shared) plus an MMA with
   it. 102 -> 55.
2. **Launches.** State-independent phases were computed per chunk:
   14 launches per chunk, i.e. 7680 per 1024-token step at a launch
   latency of 1.3-3.1 µs (measured separately). Moved to one launch per layer
   for all chunks at once. 55 -> 40.
3. **A fused kernel for the sequential part.** The chain of chunks can't be broken,
   but everything inside it fits in one warp: a warp owns 16 state rows
   over the full dk depth and does five matrix multiplies in a row without
   spilling the state to memory. 40 -> 12.
4. **Two more small things by measurement.** Q is read as fragments directly from global
   memory rather than shared: all warps of a CTA take the same elements, L1
   holds them, and 17 KB of shared costs more (11.8 -> 11.2). Substitution in
   `wy_factor` switched to 16-row panels: the previous row's value
   is read once per sixteen FMAs, within a panel it lives in registers
   (11.2 -> 10.2).

In the fused kernel what decided it was not so much the fusion itself as the layout:
the m16n8k16 accumulator (row dv, column dk) works as operand A for
`S K^T` and as operand B for `Q S^T` without a single reshuffle — the BF16 copy of
the state is packed once per chunk and serves both roles.

Sweeps on the fused kernel: 4 warps per CTA (13.0 µs/token) vs 2 (14.4) and 8
(16.0); T and F are read from global memory, not shared — this freed
18 KB and gave 13.5 -> 13.0.

Remainder by phase (measured by disabling phases, 512 tokens, at that time still 11.9
µs/token; after panel substitution `wy_factor` costs about half
as much):

| phase | µs/token |
|---|---:|
| fused kernel + value tile | 5.7 |
| inverse matrix + multipliers (`wy_factor`) | 3.0 -> ~1.5 |
| grams K K^T and Q K^T | 1.2 |
| q and k tiles | 0.4 |
| decay | 0.2 |

There's little left down this path and it's expensive. The scan is now 9.7% of the step,
while attention at the same position is 36.7%: the next prefill work is there, not
here (a 64-query-row tile where a warp owns its 16 rows and all
keys — see `prefill-budget-2026-09-14.md`).

### What was measured and rejected

Transposing values inside the fused kernel (instead of a separate kernel and a
25 MB buffer) — **slower**: 11.2 -> 11.8 µs/token. The separate kernel is
more parallel, while in the fused one value staging lands on the same chain as
the compute, with two to four warps per SM.

## A bug caught by the bench, not by a test

Diff-eval failed on the `ru` case (14 prompt tokens): the GPU argmax returned
0xFFFFFFFF, i.e. NaN in the logits. The cause was decay.

gamma_t (the product of alpha over a chunk) over 64 tokens underflows
fp32 already at alpha around 0.2, and the whole chunked form is built on the ratios
gamma_t / gamma_r. Once both are zeroed the ratio becomes 0/0. Such real
alpha values do occur in the model, but not in the tests: they had 0.92..0.96, where
gamma per chunk drops only to 0.005.

Fixed by moving decay to logarithms: the sum of `log(alpha)` is accumulated,
the ratio is taken as `exp` of the difference (always at most one), and
gamma_t itself as `exp` of the sum (on underflow it honestly gives zero). The same
was done in the CPU reference. The test now passes for decays of 0.94, 0.5, 0.15 and
0.05 per token.

## Cost

The WY scan scratch is about 84 MB per engine (q/k tiles, grams, inverse
matrices, multipliers, value tile; sized for the full 2048-row arena).
It's allocated only in `--delta-state wy` mode. At c=80 with context 2048 this is
already noticeable: one of the chat-point runs didn't start for lack of memory.

The staged (non-fused) path was removed along with its kernels: it was scaffolding for
verification and 3.4x slower than the fused one.

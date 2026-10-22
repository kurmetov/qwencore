# Eight-bit DeltaNet state: choosing the format — 2026-09-18

The DeltaNet state — 48 layers of [48 heads, 128, 128] — costs 75.5 MB per
sequence in bf16 and is 21% of decode-step traffic at c=32. One byte
per element lowers the memory floor from 19 to ~17.8 GB and removes half of that traffic.
There is exactly one question: how many bits survive the recurrence.

## Format: int8, not fp8

A simulation of one head on CPU, 512 tokens, the state is rounded after
every token, the reference is the same recurrence in fp32. Relative L2 of the
output:

| alpha | bf16 | e4m3/row | int8/row | int8/head |
|---|---:|---:|---:|---:|
| 0.20..0.60 | 7.4e-4 | 1.2e-2 | **2.9e-3** | 7.9e-3 |
| 0.70..0.95 | 2.4e-3 | 3.8e-2 | **9.4e-3** | 2.1e-2 |
| 0.90..0.999 | 4.8e-3 | 7.7e-2 | **1.9e-2** | 3.3e-2 |
| 0.995..1.0 | 1.4e-2 | 2.2e-1 | **5.5e-2** | 8.5e-2 |

Three conclusions, each changes the decision:

1. **int8 beats e4m3 fourfold at the same byte.** Per-row and per-head
   scales gave e4m3 the same result — so state rows are
   flat, and they don't need four exponent bits. int8 gives them to the mantissa:
   7 bits vs 3.
2. **Scaling by the row max is nearly optimal.** Clipping at 3, 4 and 5 RMS
   turned out worse than the max everywhere (e.g. 2.6e-2 vs 1.9e-2 at 4 RMS on
   the 0.90..0.999 row). A row's peak sits closer to RMS than a Gaussian's, and
   there's nothing to clip.
3. **Stochastic rounding doesn't help, it hurts by √2** (1.1e-1 vs
   7.7e-2 for e4m3). The error is already independent across steps, there's nothing to average:
   the recurrence doesn't accumulate many small increments into one cell, it
   rescales the whole row.

The scale is per state row (128 elements along dk): both dot products
S k and S q run along a row, so the whole sum shares one scale.

The error grows with alpha: both signal and error accumulate as 1/(1-a²), so
the relative value holds, but long retention is the worst case.
The remaining gap to bf16 is exactly 4x, and there's nothing to close it with one byte.

## What's already done

`qwc_delta_state_requantize` (`cuda/delta_net.cu`) snaps the state onto
an 8-bit grid right in the bf16 pool, without touching the layout: the same bits are lost.
This decouples the accuracy question from reworking three kernels, the pools and the
budget — diff-eval answers before that, not after.

Writing back to bf16 adds its own 2^-9 rounding on top of the
8-bit one, so a measurement through this kernel is **an upper bound**: real
storage dequantizes into fp32 registers and doesn't have this extra.

The kernel is checked element-wise against a CPU reference
(`state_requantize_matches_host_and_beats_e4m3`), on the test's live data:
int8 4.4e-3 vs e4m3 2.5e-2 relative error per application.

The key to matching the reference: the kernel multiplies by the inverse step and rounds
to even (`rintf`). Division with round-half-away-from-zero (`f32::round` in Rust)
gives a different step at the half point — the reference must repeat the operations, not
the formula.

## Diff-eval: 8 bits pass, but not head-on

The reference is our own run with bf16 state: comparison with vLLM is useless here,
the reference's own error dominates there (all three candidates gave MAE 0.38-0.39,
indistinguishable).

Corpus `bench/corpus/state-length.jsonl`: **the same text**, truncated
at each length. This matters — the first attempt took different lengths from different
documents, and a jump in the curve could have been the text, not the length.

| context | prefill segments | int8 everywhere | int8 on decode only |
|---:|---:|---|---|
| | | MAE / Jaccard / greedy | MAE / Jaccard / greedy |
| 1024 | 1 | 0.041 / 0.961 / 1.00 | 0.039 / 0.957 / 1.00 |
| 2048 | 1 | 0.042 / 0.961 / 1.00 | 0.027 / 0.992 / 1.00 |
| 3072 | 2 | **0.480** / 0.808 / 1.00 | 0.036 / 0.984 / 1.00 |
| 4096 | 2 | **0.663** / 0.816 / **0.71** | 0.031 / 0.980 / 1.00 |
| 8192 | 4 | 0.676 / 0.706 / 1.00 | 0.034 / 0.972 / 1.00 |
| 16384 | 8 | 0.702 / 0.666 / 1.00 | 0.099 / 0.934 / 1.00 |

**The damage is done not by decode but by the prefill segment boundary.** The 11x jump
between 2048 and 3072 coincides exactly with `PREFILL_CHUNK_SIZE = 2048`: a prompt
longer than one segment goes into a second pass already from the 8-bit state, and
the whole rest of the prompt — thousands of tokens — is computed from a spoiled S_0. Their
hidden states also go into the KV of the sixteen full-attention layers, so
one state requantization dirties the representation of the whole prompt tail.

The control that separates this: 2048 prompt tokens, but 64 generated tokens
instead of 8. If decode were accumulating, the error would grow with the step number. It
plateaus:

| decode steps | 0..7 | 8..15 | 16..23 | 24..31 | 32..39 | 40..47 | 48..55 | 56..63 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MAE | 0.037 | 0.084 | 0.136 | 0.074 | 0.118 | 0.100 | 0.084 | 0.125 |

Decode reads and writes the 8-bit state sixty-four times in a row and
stays at MAE ~0.1. It's not the number of requantizations, but **how many tokens
read the spoiled state after it**.

One more consequence: end-to-end int8 and e4m3 nearly converged (0.70 vs
0.73 at 16k) with a fivefold difference on the kernel itself. So on long
context we're bound not by the grid step but by saturation — the model
diverges the same way from any perturbation above some threshold. Choosing
int8 is still right: on short context and in the tail (max Δ 2.69 vs
5.11 at 4k) it's better, and costs the same.

## What this means for the layout

8 bits for the state pass with margin — MAE 0.03..0.10 and Jaccard >= 0.93 at
all lengths up to 16k, greedy output doesn't change anywhere — **provided that
a single sequence's prefill is not restarted from a quantized
state**.

Hence a layout that wasn't obvious before measuring:

* 8 bits is the **storage format for resting slots**. That's where the gain lives: there are 33
  slots at batch 32, and each gets cheaper from 75.5 MB to 37.8 MB plus 1.2 MB
  of per-row scales.
* A single sequence's prefill runs on a **dequantized working copy**
  in bf16 and is quantized once at the end of the prompt. One copy is needed per engine
  (75.5 MB), not per slot: a step may contain several segments, but
  at most one stays unfinished. A chunk comes out partial only
  when it has exhausted the step's token budget (`tokens = remaining.min(budget)`), and that
  immediately breaks the gathering loop — so at any moment exactly one
  sequence can be partially prefilled (`scheduler.rs:251`).
* Decode reads and writes 8 bits on every step — that is the traffic saving,
  and it costs MAE ~0.1.

Memory result: 33 slots from 2.57 GB to 1.28 GB plus a 75.5 MB working copy,
i.e. minus ~1.2 GB, floor 19 -> ~17.8 GB. In decode-step traffic the state's
share at c=32 drops from 21% to ~10.5%.

## Layout: what came out

Implemented per the scheme above. Measurements — `stepprofile`, the same machine
back to back, the baseline built from a worktree at HEAD with the same compiler.

| c=32, context 2048 | bf16 | int8 |
|---|---:|---:|
| cache and state | 4.92 GB | **3.79 GB** |
| `delta.scan` | 1.963 ms | **1.786 ms** |
| step with graph | 25.721 ms | **24.667 ms** |

| c=1 | bf16 | int8 |
|---|---:|---:|
| cache and state | 0.31 GB | 0.32 GB |
| `delta.scan` | 0.261 ms | 0.295 ms |
| step with graph | 12.507 ms | 12.500 ms |

The 1.13 GB saving at c=32 agrees with the calculation: 33 slots of 37.2 MB minus
the 75.5 MB working copy. **At batch 1 there's no gain and can't be**: the copy
costs a fixed 75.5 MB, and two slots save 75 MB. The break-even point is
about three slots.

The accuracy of the real layout matched the model exactly (the reference is a run
with bf16 state on the same corpus):

| context | 1024 | 2048 | 3072 | 4096 | 8192 | 16384 |
|---|---:|---:|---:|---:|---:|---:|
| MAE | 0.044 | 0.043 | 0.050 | 0.029 | 0.044 | 0.110 |
| Jaccard | 0.952 | 0.964 | 0.980 | 0.984 | 0.965 | 0.947 |
| greedy | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

There's no step at 2048: the working copy does exactly what it was introduced for.

## Scales cost more than everything else combined

The first version of 8-bit decode turned out **2.3x slower** than bf16
(4.529 vs 1.963 ms) — with half the traffic, the same 40 registers and
not a single spill to local memory. Breakdown by parts:

| variant | `delta.scan` |
|---|---:|
| access layout only, no scales or quantization | 1.208 ms |
| + all quantization (division, `rintf`, max reduction) | 1.563 ms |
| + reading and writing the scale array per row | 4.529 ms |
| + scales via shared, one coalesced access per block | **1.786 ms** |

The cost came from the write `if (lane == 0) scales[row] = scale;`: four bytes
from one lane is a partial transaction for every state row,
196608 of them per layer. 2.97 ms of the 3.3 ms overhead, i.e. more than all
the quantization arithmetic and all the traffic of the state itself.

The first hypothesis was different — the latency of the warp max reduction before
the write. It wasn't confirmed: without the reduction it's the same 4.538 ms. The breakdown by parts
took three builds and immediately showed the culprit; guessing from one number was
pointless.

The fix is to collect the block's scales in shared and write them in one coalesced
access after the loop. The same 512 bytes per block, but in one transaction instead of
128.

## What's missing

* **A runtime bf16/int8 switch.** The pools are now unconditionally 8-bit,
  and the bf16 baseline can only be measured with a build from git. That was enough for the A/B, but not for
  a rollback in production.
* **A server measurement.** `servebench` works (speculation k=3 gives 1.556
  tokens per step), but the `--corpus` and `--speculative` flags appeared in this same
  uncommitted tree, so it can't be used to compare with HEAD. It needs
  a separate session on a quiet card: right now other processes hold 4.9 GB, and with
  a 19 GB budget both builds fail the same way.

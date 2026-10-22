# Speculative decode on the MTP head — 2026-09-16

> **Single points from one session, without confidence intervals.** The comparison with
> vLLM here was taken before the protocol
> [`docs/11-experiments.md`](../../docs/11-experiments.md) existed: one or two runs per
> point, no variant order rotation and no bootstrap CI, on a partially busy
> card. These multipliers are superseded by a comparison suite run and must not
> be quoted as a project result. The conditions of each point are given below.

Drafts are proposed by the head from the checkpoint, and the model verifies them in one pass
over the weights. The output must match regular greedy — this is checked by the
`--verify` flag and holds on all runs.

## Summary

`speculate --max-new 96`, batch 1, a 9-token prompt:

| draft depth | tok/s | accepted per step | tokens per step |
|---:|---:|---:|---:|
| regular greedy | 82.3 | — | 1.00 |
| 1 | 122.0 | 0.88 | 1.88 |
| **2** | **124.2** | **1.58** | **2.58** |
| 3 | 106.2 | 1.53 | 2.53 |
| 4 | 91.2 | 1.43 | 2.43 |

The best point is two drafts: **1.51x** over regular decode. On other prompts
1.55-1.60x. Against vLLM 0.28 on the same card (50.4 tok/s) that's **2.5x**.

## What a step consists of (depth 2)

| | ms |
|---|---:|
| drafts (2 x 1.48) | 2.96 |
| model verification, 3 rows | 17.14 |
| state commit | 0.23 |
| **total** | **20.3** |

Draft: 0.64 ms for the head itself (849 MB BF16, 1320 of 1790 GB/s) plus 0.82 ms
for the vocab projection (1.27 GB). The vocab costs more than the head.

## Why going deeper than three doesn't pay off

Verifying k+1 rows does **not cost like a decode step**: 13.75 ms for two rows,
17.14 for three, 19.06 for four, vs 12.2 ms for a regular step. The reason is
narrow NVFP4 shapes: `nvfp4bench` measures W4A16 at 1513 GB/s for one row,
1078 for three and 840 for four, whereas W4A4 runs steadily around 1000 at
any width. So each added verification row costs about 1.8 ms,
even though the weights are read once.

**This is the next lever.** If verifying three rows cost like a decode
step, depth 2 would give 2.58 / (12.2 + 2.96 + 0.2) = **168 tok/s**, i.e.
2.05x instead of 1.51x. We need a batch-insensitive W4A16 for 2-8 rows.

Second biggest is the draft's vocab projection: 0.82 ms of 1.48. In NVFP4
the vocab would take 0.64 GB instead of 1.27.

## How verification works

The DeltaNet state is recurrent, so it can't be rolled back — it can only be left
untouched. Verification runs on a **draft slot**: before it the sequence's state
is copied there (81.4 MB, 0.05 ms), the k+1 rows go through the
stack on it, and the real one stays in place. Then:

- all drafts accepted — the draft state is moved back with a copy;
- some accepted — conv and scan are replayed on the real slot from the saved
  mixer inputs (48 layers x up to 8 rows x 16480 values, 12.7 MB), and this doesn't cost
  a second pass over the weights.

The commit totals 0.23 ms per step, i.e. a bit over one percent.

KV pages are written for all k+1 rows at once: rejected positions are overwritten by
the next step, the context length grows only by accepted tokens.

## Side fixes

- Narrow prefill (up to 128 rows) now picks the projection kernel by the same rule
  as decode. Before, it always went W4A4, and a verification step cost 24 ms instead of 17.
- A short scan segment (under 192 tokens) is computed in the recurrent form:
  the chunked one has six prep launches per layer, and on three rows they
  cost more than the scan itself.

## Rejected by measurement

Extending W4A16 to 5-8 rows: per `nvfp4bench` at batch 4 W4A16 gives 840
GB/s vs 1000 for W4A4, and further the gap only grows — no gain. The change
managed to break 5-8 token prompts (the fused swiglu branch accepts
only four rows) and was reverted.

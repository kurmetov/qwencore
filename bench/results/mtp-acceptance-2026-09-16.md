# MTP head acceptance — 2026-09-16

> **Single points from one session, without confidence intervals.** The comparison with
> vLLM here was taken before the protocol
> [`docs/11-experiments.md`](../../docs/11-experiments.md) existed: one or two runs per
> point, no variant order rotation and no bootstrap CI, on a partially busy
> card. These multipliers are superseded by a comparison suite run and must not
> be quoted as a project result. The conditions of each point are given below.

A check before building speculative decode: is the draft head usable,
which sits in the checkpoint and hasn't been loaded so far. Measured offline, without
verification, rollback or scheduler.

## How it was measured

`qwc-engine --bin mtpdump` captures from the engine the final hidden states of all
positions (the very tensor that goes into `lm_head`) and the tokens themselves:
14 cases of the `bench/corpus/core.jsonl` corpus, 128 generated tokens
per case, 3042 positions.

`bench/mtp_acceptance.py` runs the head (torch, weights as is) and checks its
argmax against what the real model produced. Position t gets h_t and the embedding of
token t+1 and must predict t+2 — exactly the pairing used in speculation.

Dump self-check: `lm_head(h_t)` reproduces token t+1 at 130 of
132 positions, i.e. the captured states really are the right ones.

## Result

| | acceptance |
|---|---:|
| over generated positions, whole corpus | **0.929** |
| same, without synthetic repeats | **0.897** |
| over prompt positions | 0.891 |

Draft chain: the second and further drafts go off the head's own output, and the
probability is conditional — accepting the k-th given the previous ones were accepted.

| k | p (whole corpus) | E[tokens per step] | p (meaningful) | E |
|---:|---:|---:|---:|---:|
| 1 | 0.929 | 1.93 | 0.898 | 1.90 |
| 2 | 0.881 | 2.75 | 0.799 | 2.62 |
| 3 | 0.858 | 3.45 | 0.718 | 3.13 |

The architecture reference (`docs/01-model-architecture.md`) gives 0.77-0.90 at three tokens —
the measurement confirms it and gives 0.72-0.90 on meaningful texts.

## What this gives

A draft step is reading 849 MB BF16 = 475 µs at 1.79 TB/s, a DeltaNet state
snapshot for rollback is 91 µs. The batch-1 baseline today: 12.66 ms, 79 tok/s.

| k | tok/s (meaningful) | vs us | vs vLLM 0.28 (50.4) |
|---:|---:|---:|---:|
| 1 | 144 | 1.82x | 2.85x |
| 2 | **191** | **2.42x** | **3.79x** |
| 3 | 221 | 2.80x | 4.38x |

The head in FP8 (425 MB) saves another 237 µs per draft: k=3 gives 232 tok/s.
That's already a second iteration, not needed for the first.

## Caveats

- The measurement is an upper bound for two reasons: the head sees only accepted
  tokens (that's how it is in speculation), but its own draft positions don't
  get into the KV of neighboring drafts, i.e. the chain is slightly understated, and
  the rest is overstated to the extent the corpus is synthetic.
- Half the corpus is repetitive text (`page-*`, `chunk-*`, `long-601`),
  where acceptance is naturally higher. Rely on the
  "meaningful" column: 0.898 / 0.799 / 0.718.
- The corpus is short (128 tokens per case) and not agentic. A real
  estimate needs a run on dialogs with tools — there the share of repeats from
  context is higher, and acceptance should grow.

## A pitfall that cost an hour

All norms in this model are the **Gemma variant**: scale `(1 + weight)`, not
`weight`. In `rmsnorm.cu` it's like that from the very start, but in the Python reference I
wrote the usual form — and got **exactly zero matches** on all 3042
positions, including synthetic repeats where the head must hit almost
always. The diagnostic that led to the cause: if you drop the attention layer
from the head, its output through `lm_head` started predicting the *input*
token (63 of 132) — so the path is alive, and something breaks inside the layer.
A flat zero over the whole corpus isn't "a weak head", it's a structural bug.

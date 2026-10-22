# Where we stand and where to hit — 2026-09-16

> **Single points from one session, without confidence intervals.** The comparison with
> vLLM here was taken before the protocol
> [`docs/11-experiments.md`](../../docs/11-experiments.md) existed: one or two runs per
> point, no variant order rotation and no bootstrap CI, on a partially busy
> card. These multipliers are superseded by a comparison suite run and must not
> be quoted as a project result. The conditions of each point are given below.

All numbers below were taken in one session, back to back, on one card. Between sessions
the machine drifts by 5%, so they can't be compared with tables from previous days.

## 1. Score against vLLM 0.28

Conditions are the same for both: context 8192, **bf16 KV**, a unique prompt per
request, everything submitted at t=0, `bench/servebench.py`. Step width swept on both
sides (`--prefill-chunk` for us, `--max-batched-tokens` for vLLM), the table shows
the best. Median of two runs.

| point | qwencore | vLLM 0.28 | |
|---|---:|---:|---|
| batch-1 decode, ITL | **12.66 ms** (79.0 tok/s) | 19.83 ms (50.4 tok/s) | **us 1.57x** |
| 256/128, c=32, 160 requests | **1154 tok/s** | 496 tok/s | **us 2.33x** |
| 4096/512, c=16 | **414 tok/s** (39.6 s) | 176 tok/s (92.9 s) | **us 2.34x** |
| 4096/8, c=16 — pure prefill | 11 731 tok/s | **13 100 tok/s** | vLLM 1.12x |

**A caveat without which the table lies.** The card is partially held by someone else's service,
about 27 GB of 32.6 is free. We lived under `--memory-limit-gb 26`, vLLM under
`--gpu-memory-utilization 0.75` (24.5 GB). vLLM loads 20.5 GB of weights vs
our 17.08 (text-only, fp8 head and embed), so it had about 4 GB left for KV,
and on the long point it starves for cache: its ITL is better than ours (21.7
vs 28.6 ms), but end-to-end throughput is half — it simply holds fewer
sequences at once. Part of the 2.3x gap on mixed points is our
weight headroom, not kernel speed. The headroom is real and won't go away on a
32 GB card, but calling it a kernel advantage is dishonest.

There is only one clean point — prefill: memory is barely involved there, and there we are **behind
by 12%**. On the morning of 12.09 the lag was 3.9x.

## 2. What we covered in four days

| | 11.09 | 12.09 | 14.09 | 16.09 |
|---|---:|---:|---:|---:|
| batch-1 ITL | 13.11 ms | | | **12.66 ms** |
| c=64/80, chat | 1026 tok/s | 1498 | | **1154 @ c=32** |
| pure prefill | | 3763 tok/s | 7488 | **11 731** |

The big items in order: a fused step instead of a pass per sequence (+27%),
batched `lm_head` (eight passes over the vocab -> one, +20% at c=64),
recurrent DeltaNet decode at 84-95% of bandwidth, stream-K for narrow NVFP4 shapes,
fused mixer input projection (+9%), MMA attention in prefill (2x),
prefill chunk 512 -> 2048 (+21%), chunked DeltaNet WY scan (2.19x on the phase),
64-row attention tile (1.89x on the kernel).

## 3. What was tried and rejected by measurement

So we don't go there twice:

- tiled prefill attention (8 rows per tile, warp per row) — **slower**
  than the row-wise decode path: 13.2 vs 9.3 ms;
- `kMmaGroups` — serving several heads of a group with one KV tile: 3-5% on
  long context and zero on short, register pressure eats the traffic gain;
- shared for alpha/beta/k.q in the scan, 8 state rows per warp, blocks of 256
  threads — all three slower than the baseline;
- merging gate+up into one matrix at M=2048 — 1% (at M=511 it gave 12%, but
  what won there was the narrow M, not the merge);
- `ldmatrix.x4` instead of `.x2` in attention — nothing (49.7 vs 50.1 ms);
- a 16-key KV tile instead of 32 — worse, the extra barriers cost more than occupancy gains;
- transposing values inside the fused scan instead of a separate kernel —
  11.2 -> 11.8 µs/token;
- forced W4A4 at batch 1 — worse in both speed and accuracy
  (top-1 0.991 vs 1.000).

## 4. Physics: how much is left in each mode

**Decode is memory-bound.** A step reads all the weights (17.08 GB) plus the DeltaNet
state plus KV. The analytical batch-1 ceiling is 10.78 ms; we're at 12.66, i.e.
**84% of the ceiling**, vLLM at 54%. Kernel changes have 18% left here, and
none of them will give more than a couple of percent.

**Prefill is tensor-core-bound.** 2 x 26.87e9 FLOP per token: our
11 731 tok/s is 630 TFLOP/s, **37.6% of NVFP4 peak** (1676). vLLM at 13 100 —
42.1%. Breakdown of our step at position 3584: attention 27.0%, `gemm.mlp_gate_up`
20.8%, `mlp_down` 11.3%, scan 11.1%. Even zeroing attention entirely
would give +37%; the realistic kernel remainder is 15-25%.

The conclusion that defines the plan: **with kernel changes we'll pull prefill up to
parity, but we won't crush it. The gap is created elsewhere.**

## 5. Where to hit

### 5.1 MTP speculation — the main thing

The checkpoint contains a draft head: `mtp.*`, one attention layer plus fc, **849 MB
in BF16**. We don't load it — `checkpoint.rs` only counts its bytes and puts them in
`unused`. `docs/01-model-architecture.md` recorded from the very start: the acceptance
rate of this head, measured in vLLM, is **0.77-0.90 with three speculative
tokens**.

Economics on our numbers (bandwidth 1.79 TB/s):

- a draft step = reading 849 MB = **475 µs**, that's 5% of the main step's weight;
- rolling back DeltaNet state = an 81.4 MB snapshot there and back = **91 µs**;
- verifying k+1 tokens is one pass of the main model, exactly the same path
  the prefill chunk already has.

| k | acceptance | E[tokens] | step | tok/s | vs us | vs vLLM |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.77 | 1.77 | 13.2 ms | 134 | 1.69x | 2.65x |
| 2 | 0.77 | 2.36 | 13.7 ms | 172 | 2.18x | 3.42x |
| 3 | 0.77 | 2.82 | 14.2 ms | 199 | 2.52x | 3.94x |
| 3 | 0.90 | 3.44 | 14.2 ms | 243 | 3.07x | 4.81x |

So batch 1 goes from 79 to 170-240 tok/s. This doesn't violate the memory ceiling (92.8 tok/s)
— it stops being a ceiling, because one pass over the weights
yields more than one token.

**Why this is specifically our move, not a general one.** A hybrid model: 48 of 64 layers
carry recurrent state, and speculation requires rolling it back. For us rollback
costs 91 µs per sequence, because the state slot is our
own pool, and a snapshot is a single copy. Plus our step runs at
84% of the ceiling vs 54% for vLLM: even if they enable MTP, the multiplier lands on
a worse base.

**The first step costs half a day and answers the whole question.** Load the head and
run it offline on the `bench/corpus/core.jsonl` corpus: count how often the
MTP argmax matches the main model's next token. No verification, no
rollback, no scheduler — only acceptance. If it confirms 0.77+, build
all the machinery next; if not, we've saved a week.

### 5.2 DeltaNet state in FP8 — the second biggest

A state slot is 81.4 MB per sequence, read and rewritten
every step. Its share of step traffic: 21% at c=32, 33% at c=64, 37% at c=80.

| c | step traffic now | in fp8 | tok/s ceiling now | in fp8 |
|---:|---:|---:|---:|---:|
| 32 | 24.4 GB | 21.8 GB | 2344 | 2624 (+12%) |
| 64 | 31.8 GB | 26.6 GB | 3603 | 4309 (+20%) |
| 80 | 35.5 GB | 29.0 GB | 4037 | 4945 (+23%) |

The second effect matters more than the first: a slot half the size means **twice as many
slots**. At context 2048 the bottleneck right now is exactly the state slot, not
KV (the crossover point is 2484 tokens): 87 sequences vs 120.

The risk is accuracy: the state is recurrent, the error accumulates over the whole length.
Checked with exactly the same diff-eval as all previous per-type A/Bs; the
`--delta-state` switch already exists.

### 5.3 Prefix snapshots — for the agent scenario

The README targets Pi and Hermes, i.e. a dialog where every turn resends the whole
history. Right now a turn with an 8k-token history pays full prefill (0.7 s);
with a live session it would pay only for the new tokens (~0.03 s).

For the 16 attention layers it's an ordinary prefix cache on top of our page pool.
For the 48 linear layers part of a prefix can't be reused — only a snapshot of
state at the boundary, 81.4 MB per snapshot (`docs/01-model-architecture.md` calls
it "a separate research question"). For one or two live dialogs that's
pennies; for a hundred it won't fit. So the right form is not a general prefix
cache but **session continuation**: keep the sequence's slot alive between
turns and check the prefix.

### 5.4 Remaining prefill kernel work — to reach parity

- **FP8 attention.** The kernel runs at 75 TFLOP/s against a bf16 peak of 210; fp8 doubles
  the peak, and our KV is already in fp8 — right now we expand it to bf16 when
  staging. Estimate: phase −30%, step −8%.
- **Attention occupancy.** 67.8 KB of shared per CTA — one block and four warps per
  SM. An XOR swizzle instead of padding brings tiles under 50 KB and gives two blocks.
- **`wy_factor`** — a 16x16 block inversion on MMA instead of panel substitution.
- **`cp.async`** with double buffering in both tiled kernels.

## 6. Order

1. MTP acceptance offline — half a day, decides the fate of the whole direction.
2. Full MTP speculation, if acceptance is confirmed.
3. FP8 DeltaNet state (A/B via diff-eval) — both throughput and capacity.
4. Remaining prefill kernel work up to parity with vLLM.
5. Session continuation — once an agent scenario becomes a measurement target.

And one note on methodology: the comparison with vLLM must be repeated on a completely free
card. Today it lived under 24.5 GB; at its favorite 0.93 (30.3 GB) it
looks different on mixed points.

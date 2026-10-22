# Packed GQA attention vs FlashInfer — report 2026-09-24

## Summary

- **Attention rewritten: one CTA per KV head, the group's six heads are rows
  of one MMA tile.** The old kernels used a CTA per query head and read KV
  six times. With four draft-verification rows, four of the tile's 64 rows were
  occupied.
- **Against FlashInfer on the same shapes:** decode and verification of three drafts —
  parity with xqa (both are memory-bandwidth bound, ~1.45 TB/s at 60k). Segments of
  32–192 rows are 8–25% faster than FA2, 512 rows 1.6x. That 1.6x is not
  the kernel's merit: FA2's plan at 512 rows disables split-KV and underloads the card.
- **End to end, speculation on long context started paying off:** `serve`, k=3,
  c=1: 30k — 33.7 → 165 tok/s, 8k — 87 → 187 tok/s. Without speculation 30k c=2 —
  19.84 → 14.79 ms/step. TTFT at 30k — 4.96 → 3.37 s.
- **Comparing against the previous build with the MAE 0.25 threshold doesn't work for prefill
  changes.** The old kernel with only the fp32 addition order changed by itself
  gives MAE 0.11–0.13 and max Δ up to 5.9 nats against itself. Against vLLM all
  variants, including the packed one, are indistinguishable (0.387–0.412).

Commits: `c4f4a22` (split-K in the old kernel, didn't make it into the engine), `cafbdc8`
(FlashInfer bench), `844ed9a` (packed kernel), `8fb6ad5` (engine),
`d181413` (tiles, 8 warps, partition choice). RTX 5090, driver 595.91.07. An
unrelated process was on the card (uvicorn, 3 GB, ~3% load). All numbers are from
one session, without confidence intervals; treat differences under ~5% as noise.

## 1. FlashInfer bench

`bench/flashinfer_attention.py`: xqa (decode and `q_len_per_req=4`, the way vLLM
verifies MTP drafts) and FA2 batch prefill. Shapes are the same as in `packedbench`:
a segment of one sequence after a `start` history, KV fp8 e4m3 HND,
page 64, 16 layers with separate caches, a shuffled page table. Each
kernel is checked against a torch reference by relative error (≤0.66%).

Launch pitfalls:

- `FLASHINFER_DISABLE_JIT=1` is mandatory: otherwise a cache miss launches JIT, and
  that hangs the desktop. The flag also forbids loading a ready `.so` from the JIT cache
  (without its own build FlashInfer trusts only AOT), so the bench patches
  `JitSpecNvcc.try_load` and loads `jit_library_path` itself.
- xqa and FA2 need different buffers. xqa keeps semaphores at the start of the workspace and expects
  them zeroed; FA2's partial sums overwrite them, and the next xqa call silently
  computes wrong results (6% relative error at 30k).

## 2. Kernel

`gated_packed` in `crates/qwc-cuda/cuda/paged_attention.cu`.

- Packed row p of a segment is query row p / 6, head p % 6. A tile is
  up to 64 consecutive packed rows: a decode row (its own page
  table) is 6 of 16, verification of up to five rows is up to 30 of 32, a segment is 64 each
  regardless of row boundaries (segment rows share the table).
- KV stays in fp8. A page (64 keys, 32 KB for K+V) goes via `cp.async` into
  a double buffer while the previous one is computed; 16-byte chunks are permuted
  by XOR on the key's three low bits so `ldmatrix` doesn't hit the same banks.
- K fragments are taken by `ldmatrix` over bytes read as b16; they are converted to f16
  by `cvt.rn.f16x2.e4m3x2` right in registers. Dimensions within
  a 16 are permuted so that four consecutive K bytes are b0 and b1 of one
  lane; the query is put into shared with the same permutation. V is
  transposed by `ldmatrix.trans` over byte pairs, `prmt` splits the bytes
  into n-tiles of even and odd dimensions.
- MMA in f16 with an fp32 accumulator: e4m3 is exact in f16, P in f16 is more precise than
  the old kernel's bf16.
- 4 warps, 8 on a 64-row tile (two per m-tile). The extra warps split
  the stage's keys, partial sums are reduced in shared. One CTA per SM (shared
  72–100 KB), and with one warp per scheduler the tensor cores sat idle
  while the warp computed softmax and converted fp8.
- The kernel cuts the partition span itself from the tile's actual context, the host
  picks only the number of partitions. The decode CUDA graph doesn't depend on context
  length, decode rows of different lengths each split their own context.

### Choosing the number of partitions

Time is decided by wave quantization: with one CTA per SM every started
wave costs a full one. Modes (`packed_partition_count`):

- one or two m-tiles (decode, verification) are memory-bandwidth bound, which
  64 CTAs per m-tile saturate;
- a 64-row tile uses a "waves × (span + 64 keys of CTA overhead)" model,
  fitted on a dense sweep (`packedbench --dense`: 9 sizes × 4
  contexts × P from 1 to 48). It guesses the best P on 28 of 36 shapes,
  missing by 1.8% on average, 14% at worst; the old "3.5–6 waves" rule
  missed by 14% on average and up to 53%. A 64-row tile is 24 CTAs per
  partition, and almost everywhere P=7 comes out: 168 CTAs, one nearly full wave.
  Raw data — `packed-dense-sweep-2026-09-24.txt` (the "packed" column
  there is computed with the old rule), fit —
  `bench/fit_packed_partitions.py`;
- at most 2048 (row, partition) pairs: partial sums up to 51 MB.

## 3. Kernel vs old paths and FlashInfer

`packedbench` and `bench/flashinfer_attention.py`, µs per layer, 8k / 30k / 60k.
"Old path" is what the engine used: `prefill_gated_mma` without splitting for
segments and verification, `decode_gated` for decode.

| shape | old path | FlashInfer | packed |
|---|---|---|---|
| decode, 1 row | 1394.5 / 5094.0 / 10219.4 | 18.9 / 45.6 / 83.5 (xqa) | 18.4 / 44.7 / 84.6 |
| verification, 4 rows | 1397.8 / 5095.5 / 10222.5 | 23.9 / 51.0 / 90.2 (xqa) | 22.3 / 49.9 / 91.7 |
| 32 rows | — | 58.2 / 161.5 / 306.9 | 47.8 / 137.5 / 261.8 |
| 64 rows | 1438.7 / 5171.1 / 10336.8 | 101.9 / 309.5 / 613.4 | 83.4 / 265.9 / 517.2 |
| 96 rows | — | 144.2 / 465.5 / 922.4 | 132.8 / 410.6 / 789.9 |
| 128 rows | 1446.7 / 5195.1 / 10360.0 | 177.9 / 595.1 / 1178.2 | 163.6 / 530.6 / 1034.6 |
| 192 rows | — | 298.5 / 1025.5 / 2039.8 | 242.3 / 790.4 / 1536.4 |
| 256 rows | 1483.3 / 5257.8 / 10495.3 | 397.8 / 1379.0 / 2730.2 | 316.3 / 1060.9 / 2062.6 |
| 512 rows | 2486.9 / 9513.9 / 20831.8 | 1167.6 / 4074.5 / 8075.5 | 659.6 / 2273.2 / 4461.4 |
| 2048-row chunk | 6289.1 / 23974.7 / 52404.0 | not measured | 2849.1 / 9209.0 / 17810.0 |

The decode row here is a one-row segment, as with xqa on the bench; the old
path for it is `prefill_gated_mma`. The engine itself ran decode through
`decode_gated`; against that (rows from different sequences): batch 1 —
62.0 → 17.5 at 8k, 223.7 → 47.5 at 32k, 413.6 → 82.2 at 60k; batch 2 at 32k —
439.8 → 88.2; batch 8 at 8k — 403.2 → 87.8; batch 32 at 8k — 937.7 → 345.6.
FlashInfer was not measured for batch > 1.

### Why it looks like this against FA2

FA2's plan (`wrapper._plan_info`) at 60k:

| rows | CTA | split-KV |
|---:|---:|---|
| 4 | 316 | yes |
| 64 | 336 | yes |
| 128 | 336 | yes |
| 256 | 288 | yes |
| 512 | 192 | no |

At 512 rows FA2 disables KV splitting and gives 192 CTAs on 170 SMs.
The packed kernel with one partition on this shape gave 7652 µs — almost the same as
FA2 (8076). The 4461 µs comes only from the partition choice. Where FA2
splits KV, its inner loop is no worse than ours: before the `d181413` changes it was
faster at 64–128 rows. FA2 also packs heads into tile rows
(a tile of 64 packed rows), so against FlashInfer this is not an advantage.

## 4. End to end

`serve --prefix-cache 0`, defaults `--max-seqs 32 --context 32768`.
"Before" is build `cafbdc8`, built in a separate worktree and measured in the same
session; it agreed with the morning report within 3%. `bench/serve_decode.py`
— the slope between 1-token and 129-token requests.

| mode | prompt | c | before, ms/token | after, ms/token |
|---|---:|---:|---:|---:|
| k=3 | 8 192 | 1 | 11.46 | 5.34 |
| k=3 | 30 000 | 1 | 29.71 | 6.05 |
| no speculation | 8 192 | 1 | 13.70 | 13.08 |
| no speculation | 8 192 | 2 | 15.12 | 13.72 |
| no speculation | 30 000 | 1 | 16.16 | 13.61 |
| no speculation | 30 000 | 2 | 19.84 | 14.79 |

The no-speculation rows were taken on `8fb6ad5`; `d181413` didn't touch decode.

TTFT (one-token request, minimum of three): 8k — 1079 → 915 ms, 30k —
4960 → 3368 ms.

Where the speculation speedup comes from — `servebench`, k=3, c=1, 3 requests of 128
tokens. Draft acceptance is the same (80–85% of steps accept all three), but
verification time per step dropped: 8k — 40 → 16.4 ms, 30k — 98 → 17 ms. Verification
used to run the whole `prefill_gated_mma`: 16 layers at 1.40 ms at 8k and 5.09 ms at 30k.

The `serve_decode` corpus is the repository's own sources, and the continuation copies
code from context. So draft acceptance is high, but it's the same for both
builds, and the comparison is fair.

## 5. Correctness

- `tests/paged_attention.rs`, 23 tests. Checked against CPU on segments of 1–203
  rows (a non-dividing partition count, empty late partitions, a segment in
  the middle of the arena, a tail tile of two packed rows) and on
  decode rows with their own tables and contexts (1, 63, 64, 65, …). Checked
  against the old kernel on 4 rows after 30k and on a 2048-row chunk.
- `packed_matches_cpu_on_a_sharp_softmax`: in the old tests scores are hundredths,
  softmax is nearly uniform, and the output is the mean of V for any QK error.
  Here K is up to ±4, q up to ±2, tolerance is a fraction of the reference's largest element.
  Error up to 0.3% of scale, which is the bf16 rounding of the output.

Diff-eval, `core` corpus, `--context 17408`:

| variant | vs previous build | vs vLLM | max Δ vs vLLM |
|---|---:|---:|---:|
| previous build | 0.000 | 0.389 | 3.84 |
| old kernel, split-K P=2 (addition order only) | 0.107 | 0.387 | 4.24 |
| old kernel, split-K P=4 | 0.132 | 0.406 | 5.47 |
| packed, `8fb6ad5` | 0.266 | 0.396 | 3.60 |
| packed, `d181413` | 0.254 | 0.412 | 3.95 |

- Per-layer A/B in the engine on the same inputs: the packed kernel differs from
  the old one by at most a bf16 ulp (layer 14: 0.5 at a scale of 73).
- Packed decode alone (segments on the old kernel) — MAE 0.031
  vs the previous build, which is decode noise level.
- `d181413` vs `8fb6ad5` — MAE 0.133, same as the control run.
- The only greedy-path divergence is the `zh` case, step 2: 561 (−1.271)
  vs 328 (−1.289), almost a tie. At steps 0–1 the packed distribution is
  closer to vLLM than the old one.
- On `state-length` vs the previous build MAE is 0.480. There is no vLLM reference for this
  corpus: without it long prompts are checked only this way.

## What's next

1. **Partial sums in f16.** The (row, partition) pair cap holds 384–512
   rows at P=1–4, although the best is P=7: 384 rows at 60k — 3521 µs vs 3093.
   A normalized partial output in f16 doubles the cap within the same 51 MB and
   cuts reduction traffic.
2. **Priming the MTP head as a segment.** Now it runs in the decode-row shape,
   a tile per row: at 30k about 330 ms per request (`servebench`,
   `prime_ms_total`). The priming rows are one sequence, so the
   `Segment` shape applies.
3. **Revisit the prefix cache "FAIL"** (MAE 0.41 in the morning report) with the same
   method: a control run with changed addition order and a check
   against vLLM. Probably the same prefill sensitivity to ulp.
4. The dead split-K path (`prefill_gated_mma_split`,
   `PrefillAttentionWorkspace`, `prefillsplitbench`) — delete.
5. End-to-end comparison with vLLM at 8k/30k with MTP. Launching vLLM may trigger
   FlashInfer JIT: ask the user first.

---

# Continuation: items 1-3 of "What's next" and README revision

Evening of the same day. The card was shared with unrelated processes: uvicorn (3 GB) and
from 16:39 `python -m index_worker` (2.4 GB, load spiking up to 25%). All
timings were taken A/B interleaved with `a18925a` in the same session; SM clocks
were lower than in the morning, and absolute numbers are 5-7% worse than the morning ones for both builds.
Commits: `08cfcb5` (f16), `81ef357` (workspace), `7a60b18` (MTP).

## 1. Partial sums in f16

The partition output goes out normalized in f16, max and sum in fp32: 520 bytes
per (row x head, partition) pair instead of 1032. The pair cap is 4096 within the
same 51 MB. The reduction is our own (`reduce_packed_partitions_kernel`): a warp per
pair, a lane handles eight dimensions. The shared `reduce_partitions_kernel` is
untouched. A branch for an empty partition in the reduction cost verification of 4 rows at
30k 50 -> 53 µs: the unrolled loop's loads were chained. Removed.

`packedbench`, µs per layer:

| shape | `a18925a` | f16 | |
|---|---:|---:|---|
| 384 rows, 60k | 3641 (P=1) | 3081 (P=7) | quiet card, −15% |
| 384 rows, 60k | 3682 | 3299 | medians of three interleaved, −10.4% |
| 512 rows, 60k | 4575 (P=4) | 4136 (P=7) | quiet card, −9.6% |
| 512 rows, 60k | 4719 | 4386 | interleaved, −7.1% |
| 384 rows, 30k | 1900 | 1660 | −12.6% |
| 256 rows, 8k | 335.5 | 300.3 | same P=7, less traffic |
| decode and verification | | | within 0.2 µs, noise |

**The "−10% at 512 rows" criterion is the limit of equal partitions, not of the kernel.**
512 rows is 192 CTAs per partition with one CTA per SM. Dense sweep: P=1 —
two waves of 3720 µs; P=4 — 5 waves of a quarter each (model 4650, measured
4677); P=7 — 8 waves of a seventh each (4252, measured 4334). Ideal packing of
192 CTAs on 170 SMs with no overhead is 4202, i.e. −10.2% from P=4. Beyond that —
only making the key pass itself cheaper.

The dense sweep was redone (`packed-dense-sweep-f16-2026-09-24.txt`). The model
with k0 = 64 remained the best, and the constants didn't change: miss of 0.3% on
average and 2.4% at worst (on fp32 data it was 1.8% and 14%).

Accuracy. Kernel A/B on the same inputs (sharp softmax): f16 changes 11-14%
of elements, more than 1 ulp — 0.01-0.13%, max — an ulp at scale, same as the
control "same HEAD at a neighboring P" (7% of elements, 1 ulp). Diff-eval,
`--context 17408`:

| variant | core vs vLLM | core vs `a18925a` | state-length vs `a18925a` |
|---|---:|---:|---:|
| `a18925a`, repeat run | 0.412 | 0.000 | — |
| control: fp32, normalized, new reduction | 0.416 | 0.048 | 0.142 |
| f16 | 0.425 | 0.063 | 0.156 |
| f16, partitions summed in reverse order | 0.411 | — | 0.166 |

Exact prefix vs vLLM is 0.858 for all, top-1 0.981. The only
f16 outlier is `long-601` (0.808 vs 0.564): the variant with reverse
addition order on the same f16 bytes gets 0.552 there. That's addition order,
not f16 precision.

## 2. Priming the MTP head as a segment

The head's call rows are always one sequence: the scratch has a single KV,
`upload` gives all rows one table. The `Segment` shape applies to all
`draft` calls, not just priming.

Along the way a workspace sizing bug turned up: it was sized by P at the context
cap, but the wave model isn't monotonic in context. 6-8 rows at 32768
take P=40, at 32256 — P=42, and priming failed the size check. Now
the size is by the largest allowed P; the executor gets exactly 4096 pairs,
the head's scratch at 32k is 12.8 MB instead of 0.2.

| 30k, k=3, `--mtp-prime 2048` | Rows | Segment |
|---|---:|---:|
| `prime_ms_total`, servebench, 3 requests | 1044 / 1049 | 858 / 870 |
| TTFT serve, 30k, minimum of three | 3627 / 3617 | 3594 / 3562 |
| TTFT serve, 8k | 977 / 978 | 962 / 970 |

Priming −60 ms per request, TTFT −1…−1.5%. Priming is not bound by
attention: 2048 positions go in batches of eight rows (`mtp::MAX_ROWS`) with
a sync after each, that's 256 passes over 849 MB of head weights,
~140 ms of one weight read out of ~290. The next lever is the head's linear layers
over the whole priming batch at once.

## 3. The README top table

Methodology of 09-17: `servebench --corpus bench/corpus/serve.jsonl --requests 8
--concurrency 1 --prompt-tokens 256 --max-new 128 --context 2048
--memory-limit-gb 26`, median of three runs (4-6; runs 1-2 ran under
`index_worker` load and were discarded).

| | runs | median |
|---|---|---:|
| regular decode | 75.75 / 76.27 / 74.49 | 75.8 tok/s |
| k=3 + 32K shortlist | 160.64 / 158.19 / 159.07 | 159.1 tok/s |

Speculation: 3.36 tokens per pass (histogram 29 / 37 / 32 / 204),
draft 0.73 s, verification 4.97 s for 8 requests. Regular decode of `a18925a` in
the same session — 76.44 / 76.46 vs 76.09 / 76.29 for the new build: the 79.2
from the README isn't reproduced on today's card by either build.

## 4. The prefix cache "FAIL" is the chunk boundary

`serve --context 17408 --kv-cache-gb 2 --max-seqs 4 --speculative 0`
vs `eval` of the same build, state-length (18 cases up to 16k):

| serve | MAE | max Δ | exact prefix |
|---|---:|---:|---:|
| cache off, chunk 2048 | 0.000 | 0 | 1.000 |
| cache off, chunk 512 | 0.556 | 5.62 | 0.799 |
| cache off, chunk 1024 | 0.515 | 7.66 | 0.958 |
| cache off, chunk 1536 | 0.490 | 6.15 | 0.958 |
| cache 8, chunk 2048 (morning FAIL) | 0.499 | 5.47 | 0.958 |
| cache 8, chunk 1024 | 0.515 | 7.66 | 0.958 |
| cache 8 vs cache off, both chunk 1024 | **0.000** | 0 | 1.000 |

- **Restoring a snapshot is bit-exact.** Snapshots on this corpus sit at
  1024, 2048, 3072, 4096, 8192; with chunk 1024 the boundaries of the path with the cache and
  without it coincide, and the outputs match completely.
- The whole divergence is an extra chunk boundary at the snapshot position. The WY scan
  accumulates DeltaNet state in fp32 within a chunk and writes it in bf16 at
  the boundary; a boundary is one more rounding of the 48 layers' state, the same thing
  decode does on every token. Any shift of boundaries gives 0.49-0.56 on this corpus,
  the cache gives 0.50.
- On core, where a vLLM reference exists, the cache is neutral: eval 0.425, serve with cache
  0.433; per case the shifts go both ways (`chunk-127` +0.30, `chunk-129`
  −0.08), like the "order only" variants.
- For comparison on state-length: permuting addition in attention gives
  0.14-0.17, the morning attention kernel change 0.48. The 0.25 threshold doesn't fit this
  corpus.

What's missing: a vLLM reference on state-length. Without it, it's not proven that the boundary
doesn't lead away from the truth — only that it's indistinguishable from changing the chunk size.

## What's next

1. ~~vLLM reference on state-length~~ — below: the boundary leads away from the reference by
   0.03-0.04 with chunk 512-1536, and with the default chunk 2048 it changes
   nothing.
2. Priming the head in one batch: the head's linear layers over 2048 rows
   instead of 256 batches of eight, down to ~250 ms TTFT on a long prompt.
3. ~~Dead split-K path~~ — removed by reverting `c4f4a22`: it never made it into
   the engine, the packed kernel covered all its shapes.

---

# vLLM reference on state-length

17:30-17:50, card shared with uvicorn (3 GB) and `index_worker` (2.4 GB).
vLLM 0.28.0 via `bench/diff_eval.py`: `CutlassNvFp4LinearKernel`,
`FLASH_ATTN`, GDN prefill on Triton/FLA, DeltaNet state in fp32
(`mamba_ssm_dtype`), KV in bf16. FlashInfer not involved: nvcc not on
PATH, JIT blocked by `FLASHINFER_DISABLE_JIT=1`, run under `nice -n 19`,
`MAX_JOBS=4`. The reference is step 16392: the whole prompt in one step, no chunk
boundaries (`vllm-state-length.jsonl`). The qwc candidates are the artifacts from item 4:
eval of the f16 build and serve `7a60b18`; serve with chunk 2048 matches eval
bit for bit.

## First, about the reference itself

- **vLLM is nondeterministic on long prompts.** Five runs in one
  configuration follow the same greedy path, but logprobs differ pairwise by
  0.343 ± 0.097 (0.13-0.43). The noise grows with length: for the first pair 0.17 at
  1024 tokens and 0.62 at 16384. On core a repeat gives 0.045.
- **`vllm.jsonl` from 09-12 was taken in a different vLLM configuration.** Today's
  core against it is 0.413 with exact prefix 0.917; even the
  five-token `en-fact` diverges (−0.455 vs −0.385 at step 0), while a repeat
  today is 0.045. "qwc vs vLLM" on core (0.39-0.42) is of the same order
  as vLLM vs vLLM in another configuration. Core in today's
  configuration is `vllm-core-2026-09-24.jsonl`; `vllm.jsonl` is untouched,
  earlier reports were checked against it.
- **A vLLM step size is a different configuration, not just boundaries.** vLLM
  compiles the graph for the step range (`compile range (1, N)` in the log). With
  step 2048 a 1024-token prompt goes without a boundary, same as with 16392, yet
  diverges from it by 0.640 vs 0.165 for a repeat. So vLLM runs
  with step 2048 and 512 are not a boundary control. The boundary is measured with qwc variants:
  qwc is deterministic, and eval and serve have no other difference.

## Chunk boundary vs vLLM

MAE to each of the five whole-prompt vLLM runs, mean ± sd over references.
The difference is with eval on the same reference, paired.

| variant | MAE to whole-prompt vLLM | difference from eval |
|---|---:|---:|
| qwc eval (chunk 2048) | 0.610 ± 0.020 | 0 |
| qwc serve, chunk 1536 | 0.635 ± 0.007 | +0.025 ± 0.015 |
| qwc serve, chunk 1024 | 0.641 ± 0.009 | +0.031 ± 0.020 |
| qwc serve, chunk 512 | 0.653 ± 0.015 | +0.043 ± 0.013 |
| qwc serve, cache 8, chunk 2048 | 0.616 ± 0.013 | +0.005 ± 0.019 |
| qwc serve, cache 8, chunk 1024 | 0.641 ± 0.009 | +0.031 ± 0.020 |
| qwc eval `a18925a` (order only) | 0.616 ± 0.021 | +0.006 ± 0.003 |
| qwc eval, fp32 control (order only) | 0.607 ± 0.021 | −0.003 ± 0.006 |
| qwc eval, reverse partition order | 0.617 ± 0.017 | +0.007 ± 0.007 |
| vLLM, step 2048 | 0.689 ± 0.012 | +0.078 ± 0.021 |
| vLLM, step 512 | 0.669 ± 0.008 | +0.058 ± 0.016 |
| whole-prompt vLLM vs whole-prompt vLLM | 0.343 ± 0.097 | |

By length, against the first run:

| variant | 1024 | 2048 | 3072 | 4096 | 8192 | 16384 |
|---|---:|---:|---:|---:|---:|---:|
| vLLM, repeat | 0.165 | 0.225 | 0.418 | 0.452 | 0.626 | 0.623 |
| qwc eval | 0.507 | 0.409 | 0.723 | 0.546 | 0.783 | 0.916 |
| qwc serve, chunk 512 | 0.577 | 0.460 | 0.688 | 0.606 | 0.747 | 0.896 |

- **An extra boundary leads away from the reference, but only slightly** — and monotonically in
  the number of boundaries: chunk 1536 gives +0.025, 1024 — +0.031, 512 —
  +0.043 ± 0.013. Permutations of addition in attention stay within
  ±0.007. This is the cost of rounding state to bf16 at the boundary discussed
  in item 4. But it's 12-20 times less than serve vs eval's 0.49-0.56:
  the rest is a sideways shift, at the same distance from the reference.
- With chunk 2048 the cache adds boundaries only at snapshot positions not
  divisible by 2048 (1024, 3072): +0.005 ± 0.019, indistinguishable from zero.
- At 1024 tokens eval and chunks of 1024 and up have no boundary at all, serve 512
  has one, at 512: 0.577 vs 0.507. This is the cleanest "one
  boundary" case: +0.07.
- The farthest from whole-prompt vLLM is vLLM itself with a different step, farther than any
  qwc variant.

**Item 4 is closed.** With the default chunk 2048 serve matches eval bit
for bit. A smaller chunk costs 0.03-0.04 to the reference, not 0.5. The 0.25 threshold for
state-length doesn't fit either against qwc or against vLLM: a single vLLM run
is itself noisy at 0.34, and differences in the hundredths are distinguishable only paired, over
several reference runs.

## Bench changes

`bench/diff_eval.py`, vLLM adapter:

- `enable_prefix_caching=False` explicitly. In state-length a short prompt is
  a prefix of a longer one from the same document, and with the cache vLLM would restore its KV and
  state instead of prefilling.
- `language_model_only` by default (`--vllm-keep-vision` — as before).
  The vision tower (0.92 GB) is not needed for a text corpus, and on a shared card
  step 16392 starts only at util 0.76 even without it.
- `--vllm-max-num-seqs`, default 4. Before capturing graphs vLLM
  allocates KV for `min(max_num_seqs, 512)` blocks, and a block of a hybrid model
  is the DeltaNet state of one sequence. At 256 that came to 784 MiB
  for each of the 64 layers, and even core failed. Cases go one at a time, decode
  is a size-1 graph at any value.
- `--vllm-max-batched-tokens` — prefill step; the default on the 5090 is
  8192.
- The artifact header gets a configuration summary: step, `max_num_seqs`, prefix
  cache, state format, `language_model_only`.

Step 16392 doesn't start at util 0.72 (KV 0.27 GiB with 1.15 needed), at
0.76 — 2.51 GiB. Per-case breakdown for all "variant × reference" pairs and
vLLM run pairs — `vllm-state-length-scores.json`.

```
nice -n 19 env MAX_JOBS=4 FLASHINFER_DISABLE_JIT=1 ~/.venvs/baseline/bin/python \
  bench/diff_eval.py run --engine vllm --gpu-memory-utilization 0.76 \
  --vllm-max-batched-tokens 16392 --corpus bench/corpus/state-length.jsonl \
  --output bench/results/vllm-state-length.jsonl
```

## What's next

1. Priming the head in one batch — item 2 above, unchanged.
2. If even these 0.03-0.04 at a small chunk become needed: keep the state of the
   active prefill between chunks in fp32 rather than bf16. One copy per
   engine, 151 MB instead of 75.5. Verify with the same paired check over
   several vLLM runs.
3. Run core comparisons with vLLM against `vllm-core-2026-09-24.jsonl`, not
   `vllm.jsonl`, and remember: against vLLM only what exceeds its own noise
   is distinguishable.

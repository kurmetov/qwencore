# Server vs eval and decode attention — report 2026-09-24

## Summary

- **Decode attention in `serve` chose the context split by executor
  capacity, not by the actual batch.** With the default `--max-seqs 32`
  the kernel ran with one partition even for a single request. The choice is now made on
  every call by the number of rows. Ablation on `serve` without speculation: **4.07x** on
  the decode step at 30k with c=1 (66.49 → 16.34 ms), 3.67x at c=2, 1.96x at 8k.
- **Speculation on long context currently loses to regular decode**: at 30k
  with c=1 33.5 tok/s vs 61.2. The likely cause is that draft verification goes through
  the prefill MMA kernel, which for 4 rows has 24 CTAs without splitting by
  context. Not measured per phase.
- **`serve` matches `eval` bit for bit** with `--prefix-cache 0
  --speculative 0` and the same `--context`, and after the fix — at any
  `--max-seqs`. Speculation adds decode-level noise (MAE 0.04). **The prefix
  cache fails the diff-eval threshold on long prompts**: MAE 0.41, one
  divergence in 18 cases. That's an order of magnitude more than decode noise, the cause is not
  investigated.

Commit `32aa5c0` plus uncommitted changes (dirty), RTX 5090, driver
595.91.07. An unrelated process was on the card the whole time (python, 3 GB,
6–8% load). All numbers are from one session, without confidence intervals.

## 1. Ablation: splitting decode attention by the actual batch

### Mechanism

`PagedAttentionWorkspace::new(batch, max_context)` chose the kernel
(`select_decode_kernel`) and the number of partitions (`partition_count`) once, by capacity. For
the `serve` executor the capacity is `--max-seqs`, 32 by default:
`partition_count(QueryHead, 32, 32768) = ⌈340 / (32·24)⌉ = 1`. With one
request decode attention launched 24 CTAs on 170 SMs, and each CTA read the whole
context alone.

The same bug was in two more places:
- the decode rows of a mixed step: their workspace is sized for 1024 rows;
- the MTP head draft: sized for 8 rows with an actual one.

The fix: the workspace is allocated for the worst case over all row counts
up to capacity, and the kernel and partitions are chosen in `decode_impl` by the actual
`batch` (`PagedAttentionWorkspace::plan_for`). The choice depends only on the number of
rows and `max_context`, so the CUDA graph for each batch stays
correct. The `with_kernel`/`with_partitions` path for benches is unchanged.

### Kernel, `attentionbench` (FP8 KV, 16 layers, µs per layer)

| batch | context | partitions as in `serve` | µs | partitions after | µs | difference |
|---:|---:|---|---:|---|---:|---:|
| 1 | 32 768 | QueryHead / 1 | 3606.45 | SharedKv / 43 | 230.05 | 15.7x |
| 2 | 32 768 | QueryHead / 1 | 3598.04 | SharedKv / 22 | 444.37 | 8.1x |
| 1 | 8 192 | QueryHead / 1 | 902.94 | QueryHead / 15 | 66.50 | 13.6x |
| 4 | 8 192 | QueryHead / 1 | 894.14 | QueryHead / 4 | 236.41 | 3.8x |

### End to end, `serve --prefix-cache 0 --speculative 0`

Server defaults: `--max-seqs 32 --context 32768`, `--kv-cache-gb 3`.
Measured with `bench/serve_decode.py`: the slope between 1-token and 129-token requests,
minimum of two repeats, `ignore_eos`. "Before" is the same build with the choice by
capacity: one line was toggled (`adaptive`), everything else the same.

| prompt | c | before, ms/step | after, ms/step | before, tok/s | after, tok/s | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 8 192 | 1 | 27.72 | 14.14 | 36.1 | 70.7 | 1.96x |
| 8 192 | 2 | 28.70 | 15.64 | 69.7 | 127.9 | 1.83x |
| 30 000 | 1 | 66.49 | 16.34 | 15.0 | 61.2 | **4.07x** |
| 30 000 | 2 | 72.65 | 19.77 | 27.5 | 101.2 | **3.67x** |

This is exactly the agent mode: several dialogs on long context, each
generating without speculation. `servebench` didn't see the bug: there the executor
is created with `--concurrency` slots, and capacity matches the actual
batch.

### With defaults (speculation k=3, shortlist 32K), c=1

| prompt | before, ms/token | after, ms/token | before, tok/s | after, tok/s |
|---:|---:|---:|---:|---:|
| 8 192 | 12.11 | 11.89 | 82.6 | 84.1 |
| 30 000 | 34.13 | 29.83 | 29.3 | 33.5 |

The head draft got faster, but speculation at 30k is slower than regular decode
after the fix: 33.5 vs 61.2 tok/s. At 8k the gain is 1.19x instead of 1.58x on
short prompts (commit `32aa5c0`).

The likely cause is draft verification. It goes through
`prefill_gated_mma`, whose grid is `(4 KV heads, ⌈rows/64⌉, 6)`: for 4
verification rows that's 24 CTAs, each reading the whole context without splitting. It's the same
disease as above, but in the prefill kernel. There's no per-phase step breakdown, this is
an inference from the launch grid.

### Correctness

- `tests/paged_attention.rs`: new `workspace_partitions_follow_actual_rows`
  and `oversized_workspace_matches_exact_one_bit_for_bit` (a workspace for 32
  rows at batch 1 gives the same answer as one for 1). 13/13.
- Diff-eval: the output of single-slot `eval` after the fix is bit for bit the same
  (`core` and `state-length`, MAE 0.000); `serve` with 4 slots matches it bit
  for bit (section 2).

## 2. `serve` vs `eval`

`serve` got `/v1/completions`: a prompt as token ids, top-k `logprobs`,
the vLLM extensions `ignore_eos` and `return_tokens_as_token_ids`. Log-probabilities
are computed as in `eval`. The logits row is checked against the card's argmax; on
mismatch the request fails rather than returning someone else's numbers. This way `bench/diff_eval.py
--engine openai --openai-token-ids` checks the server with the same pipeline as
the other engines.

The reference is `eval --context 17408`. Server: `--context 17408 --kv-cache-gb 2`,
corpora `core` (14 cases) and `state-length` (18 cases up to 16k), requests
one at a time.

| `serve` configuration | corpus | exact prefix | top-1 | Jaccard | MAE | max Δ | threshold |
|---|---|---:|---:|---:|---:|---:|---|
| 1 slot, cache and speculation off | core / len | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0.000 / 0.000 | 0 / 0 | PASS |
| 4 slots, off, **before** the fix | core / len | 1.000 / 1.000 | 1.000 / 1.000 | 0.983 / 0.973 | 0.024 / 0.042 | 0.72 / 3.12 | PASS |
| 4 slots, off, **after** | core / len | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0.000 / 0.000 | 0 / 0 | PASS |
| 4 slots, speculation k=3, before | core / len | 1.000 / 1.000 | 1.000 / 1.000 | 0.981 / 0.971 | 0.035 / 0.043 | 1.44 / 0.94 | PASS |
| 4 slots, cache 8, before | core / len | 1.000 / **0.951** | 1.000 / 0.993 | 0.918 / **0.793** | 0.107 / **0.411** | 3.86 / 4.45 | **FAIL** |
| defaults (cache + k=3), before | core / len | 1.000 / 0.951 | 1.000 / 0.993 | 0.920 / 0.792 | 0.115 / 0.412 | 4.18 / 4.16 | FAIL |

What follows from this:

1. **The server matches `eval` bit for bit only with the same `--context`.**
   `max_context` affects the kernel choice itself (`batch == 1 && max_context >=
   16384` — SharedKv) and the partition cap. `eval` with a corpus-sized context (609
   for `core`) and with 17408 diverge: MAE 0.019.
2. **Speculation adds decode-level noise**: MAE 0.035–0.043 vs 0.024–0.042
   from attention splitting alone. All tokens matched.
3. **The prefix cache is the main source of divergence.** In these corpora cases
   continue each other (`doc0-2048` starts with `doc0-1024`), and the server
   restores a snapshot at 1024…8192 tokens and then plays out up to 8192 tokens of
   tail. Already at step 0 the top-20 tail drifts by 1–4 nats, `doc0-4096`
   diverges at step 1. Short tails have nothing to do with it: they're long. The noise
   between variants of `eval` itself (state packing, `qwc-len-*`) is 0.05–0.5
   nats, occasionally 1.6. To separate "an extra chunk boundary" from "restoring a
   bf16 snapshot" we need an `eval` run with the prompt cut at the same position without
   the cache; there isn't one.

Artifacts: `bench/results/serve-vs-eval-2026-09-24/`.

## 3. Other changes of the day

- **Prefix cache eviction.** It used to be pure LRU. Under KV pressure it
  evicted the entry of an ongoing request: all its pages are shared with a live
  sequence, zero pages were freed, and the next dialog turn
  missed. Now nested entries go first (a prefix of a longer
  entry), then entries with no live page owners, and within that by
  age. Entries whose pages are all held by live sequences are not touched for the sake of pages.
  4 new tests in `cache.rs`; under the old LRU all
  four fail.
- **MTP head KV on dialog change.** `bind_prompt` reset KV only when
  the common prefix was empty. Two dialogs with a shared system prompt saw each other's
  KV after it. Now KV is zeroed from the page holding position
  `common − 1`: the pair (h_t, token t+1) of the last common position is foreign too.
  Not measured end to end.
- `Executor::logits_to_host` copies one row, not the whole logits buffer
  (32 × 248 320 float).

## What's next

1. ~~**Context splitting in the prefill MMA kernel for a small number of rows.**~~
   Done differently — with packed GQA attention: speculation at 30k 33.7 → 165
   tok/s, see `packed-attention-2026-09-24.md`.
2. An `eval` run with the prompt cut at the snapshot boundary — to separate snapshot restore
   from an extra chunk boundary.
3. Re-measure end to end the interleaving of dialogs with a shared system prompt
   (head acceptance before and after the KV trim).

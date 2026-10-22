# Experiment protocol: ablations, roofline and comparison with vLLM

The purpose of this protocol is to separate an engineering result from a lucky
number. What gets published is not the best run but all independent trials, the exact command,
GPU state before and after the run, and a 95% confidence interval.

## What exactly we claim

The project makes three different claims, and they must not be merged into one multiplier:

1. **Batch-1 decode:** how close a regular step gets to the bandwidth
   roofline, and whether it beats vLLM without speculation.
2. **Batch-1 MTP:** how many accepted tokens one pass over the weights yields;
   the best MTP configurations of both engines are compared.
3. **Serving under a budget:** aggregate throughput, TTFT and ITL under the same
   total VRAM budget. Here a smaller resident footprint is part of the
   result, but is not called an advantage of the CUDA kernels.

Prefill is published separately: it is compute-bound and doesn't follow the decode
roofline.

## Roofline

The decode denominator is the bytes actually read by a step, not all occupied VRAM:

```text
resident weights                           17.08 GB
ordinary decode streamed weights           14.96 GB
batch=1 state read+write + KV(ctx=2048)    ~0.16 GB
measured sustainable DRAM read bandwidth    1.605 TB/s
ideal ITL                                  ~9.42 ms
ideal throughput                            ~106 tok/s
```

MTP has a different denominator: one main pass can confirm more than
one token. For it, accepted tokens/step, draft time and verification time must be published
next to tok/s; dividing MTP tok/s by the regular
single-token roofline is not allowed.

Numbers are generated from code:

```bash
cargo run -q -p qwc-core --bin plan
```

## Repeated runs and intervals

`bench/experiments.py` runs each trial in a new process. The order of
variants rotates between trials so that card warm-up and thermal drift
don't always land on the same engine. Raw JSONL is appended after each
run with `fsync`, so a long experiment can be continued with `--resume`.

The report shows a percentile-bootstrap 95% CI of the mean over independent
process runs, 20 000 resamples with a fixed seed. For relative
speedup a paired bootstrap over matching trial numbers is used. The minimum is
3 repeats; a normal publishable run is 7 or more.

By default the runner refuses to start if more than 8 GiB of VRAM is in use before the experiment.
This protects against accidentally comparing next to someone else's inference job.
The threshold is changed via `--max-background-memory-gb`; the `--allow-busy-gpu` flag
is kept only for diagnostic, knowingly contaminated runs and is recorded
in the artifact.

```bash
# First check the exact commands.
python3 bench/experiments.py run --suite comparison --dry-run

# Fair comparison. Needs Python from the vLLM venv.
~/.venvs/baseline/bin/python bench/experiments.py run \
  --suite comparison --repeats 7 \
  --output bench/results/comparison-trials.jsonl

python3 bench/experiments.py report \
  --input bench/results/comparison-trials.jsonl \
  --output bench/results/comparison-with-ci.md
```

The comparison suite pins the same checkpoint, corpus, KV dtype, context,
number of requests, lengths and total memory budget. CUDA graphs are on, prefix
caching is explicitly off, there's a warmup before recording. For c=32 the same
token budgets of 2048 are used. For pure prefill the pre-chosen
best values from the sweep are published: chunk 2048 for QwenCore and
`max_num_batched_tokens=4096` for vLLM.

## Ablations

```bash
~/.venvs/baseline/bin/python bench/experiments.py run \
  --suite ablations --repeats 7 \
  --output bench/results/ablation-trials.jsonl

python3 bench/experiments.py report \
  --input bench/results/ablation-trials.jsonl \
  --output bench/results/ablations-with-ci.md
```

The matrix is arranged so that adjacent rows change one factor:

| scenario | transition | what is isolated |
|---|---|---|
| `ablate-speculation` | base -> MTP k=3 | speculative decoding |
| | MTP k=3 -> + shortlist 32K | incomplete draft `lm_head` |
| `ablate-prefill-scan` | recurrent BF16 -> recurrent FP32 -> WY | rounding and then the DeltaNet scan algorithm |
| `ablate-prefill-chunk` | 512 -> 2048 | step token budget |
| `ablate-kv` | BF16 -> FP8 | KV format under an unchanged workload |

Int8 DeltaNet state is not yet in the automatic matrix: the current tree
has no runtime switch back to BF16. Its existing A/B was done with two
builds. It should be published as a separate row with two commit/diff hashes,
not pretend it's the same executable.

## Publication rules

* Don't remove failed trials from JSONL. The report counts only successful ones, but explicitly
  shows the number of excluded errors.
* Don't run a publishable series with unrelated compute load on the
  GPU. A desktop footprint is acceptable if it's the same and recorded in the artifact.
* Don't pick the best of the repeats. The mean, CI and all raw trials are published.
* Any post-hoc choice of batch/token budget first goes into a tuning sweep, then
  is confirmed by a new series of trials.
* A speed ablation comes with a quality A/B on the same token IDs. For
  approximate formats one greedy match is not enough: top-k overlap and
  logprob error are saved.

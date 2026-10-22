# Experiment harness — report 2026-09-22

## Summary

The project got a reproducible harness for ablations and fair comparison with vLLM:

- each trial runs in a separate process;
- variant order rotates between trials;
- raw JSONL, the full command, commit, dirty state, source hash and
  GPU state before/after the run are saved;
- the report computes a bootstrap 95% CI of the mean and a paired 95% CI of the relative
  speedup;
- a publishable run is blocked if more than 8 GiB of VRAM is in use before start;
- vLLM runs with CUDA graphs, prefix caching explicitly off, the same
  checkpoint, corpus, KV dtype, context, lengths and total VRAM budget.

Implementation: `bench/experiments.py`. Methodology: `docs/11-experiments.md`.

## Corrected roofline

The old model mixed up the resident footprint and decode-step traffic.

| quantity | value |
|---|---:|
| resident weights | 17.08 GB |
| weights actually read by ordinary decode | **14.96 GB** |
| measured sustained DRAM bandwidth | **1.605 TB/s** |
| ideal ITL, batch=1, ctx=2048 | **9.42 ms** |
| physical ceiling | **106 tok/s** |

The MTP head isn't read by a regular step, and the embedding does a one-row lookup,
so both are excluded from the full weight pass. The calculation now
lives in `WeightPlan::decode_weight_bytes` and is used by `qwc-core plan`.

The last previously measured ordinary decode of 79.2 tok/s corresponds to roughly
75% of the corrected roofline. Speculative decode can't be divided by this ceiling:
one main-model pass confirms several tokens.

## Comparison matrix

The comparison suite pins four modes in advance:

| scenario | QwenCore | vLLM |
|---|---|---|
| batch 1 without MTP | ordinary decode | ordinary decode |
| batch 1 with MTP | k=3 + shortlist 32K | Qwen3.5 MTP k=3 |
| serving c=32 | chunk 2048 | max batched tokens 2048 |
| prefill 4096/8 | chunk 2048 | max batched tokens 4096 |

All rows use real non-repeating prompt token IDs.

## Ablation matrix

| transition | isolated factor |
|---|---|
| base -> MTP k=3 | speculative decoding |
| MTP k=3 -> shortlist 32K | reduced draft `lm_head` |
| recurrent BF16 -> recurrent FP32 | state rounding inside prefill |
| recurrent FP32 -> WY | DeltaNet scan algorithm |
| prefill chunk 512 -> 2048 | step token budget |
| BF16 KV -> FP8 KV | KV cache format |

Int8 DeltaNet state remains a cross-build ablation for now: there's no runtime switch
back to BF16. The existing measured result is 25.72 -> 24.67 ms at
batch=32 and −1.13 GB of VRAM; it isn't mixed with single-binary ablations.

## Checks

- Python unit tests: **7/7 passed**.
- `cargo test -p qwc-core`: **16/16 passed**.
- `cargo check -p qwc-engine --bin servebench`: **passed** in a separate target.
- `git diff --check`: **passed**.
- Both experiment suites passed dry-run; all commands and corpora resolve.
- The overall `cargo fmt --check` stays red on existing uncommitted
  files outside this change; automatic reformatting of someone else's tree
  was not performed.

## Why there's no new CI table

During the work the RTX 5090 had 28.9 of 32.6 GiB in use. Main consumers:

- `VLLM::EngineCore`: 24.2 GiB;
- a separate Python process: 3.1 GiB.

Running the 26 GB comparison in that state would have been either OOM or a measurement of
contention with someone else's workload. Old single points can't be turned into
confidence intervals after the fact, so no new numbers were made up.

## Commands for the final run

After the GPU is freed:

```bash
~/.venvs/baseline/bin/python bench/experiments.py run \
  --suite comparison --repeats 7 \
  --output bench/results/comparison-trials.jsonl

python3 bench/experiments.py report \
  --input bench/results/comparison-trials.jsonl \
  --output bench/results/comparison-with-ci.md

~/.venvs/baseline/bin/python bench/experiments.py run \
  --suite ablations --repeats 7 \
  --output bench/results/ablation-trials.jsonl

python3 bench/experiments.py report \
  --input bench/results/ablation-trials.jsonl \
  --output bench/results/ablations-with-ci.md
```

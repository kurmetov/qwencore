# Prefill attention on tensor cores — 2026-09-14

> **Single points from one session, without confidence intervals.** The comparison with
> vLLM here was taken before the protocol
> [`docs/11-experiments.md`](../../docs/11-experiments.md) existed: one or two runs per
> point, no variant order rotation and no bootstrap CI, on a partially busy
> card. These multipliers are superseded by a comparison suite run and must not
> be quoted as a project result. The conditions of each point are given below.

Kernel `paged_attention_prefill_mma_kernel`: `QK^T` and `PV` via `mma.m16n8k16.bf16`,
FA2 online softmax, KV tile in shared, a CTA is 4 warps per 16 query rows of one
group head. Wired into the engine instead of `decode_gated` for prefill rows.

## Kernel, 511 chunk rows, 16 layers

| chunk position | decode (before) | MMA | |
|---|---:|---:|---|
| 0 | 6.14 ms | 2.21 ms | 2.78x |
| 1024 | 33.77 ms | 7.97 ms | 4.24x |
| 3584 | 100.22 ms | 23.39 ms | 4.29x |
| 7680 | 211.77 ms | 48.01 ms | 4.41x |

On a bf16 cache — 3.8x / 4.9x / 5.5x / 5.7x.

## Engine step, 511-token chunk at position 3584

| | before | after |
|---|---:|---:|
| attention | 143.85 ms (73.7%) | 26.34 ms (35.3%) |
| whole step | 194.62 ms | 72.31 ms |

## End-to-end prefill (4096/8, c=16, bf16 KV)

3763 -> **7488 tok/s** (2.0x). TTFT p50: 34.8 -> 17.4 s.

## Comparison with vLLM 0.28 (methodology bench/results/qwc-vs-vllm-2026-09-12.md)

| point | qwencore before | qwencore after | vLLM | |
|---|---:|---:|---:|---|
| 256/128, c=80 | 1498 | **1610** | 1340 | qwc 1.20x |
| 256/512, c=16 | 668 | 679 | 748 | vLLM 1.10x |
| 4096/512, c=16 | 265 | **380** | 439 | vLLM 1.16x |
| 4096/8 (prefill tok/s) | 3763 | **7488** | 14656 | vLLM 1.96x |

## A fixed bug

The old path computed prefill **incorrectly on prompts longer than one chunk**.
Diff-eval, the arbiter is a saved vLLM run, case `long-601` (601 prompt tokens,
i.e. two chunks):

| | top-1 match with vLLM | median \|dlogprob\| |
|---|---|---|
| old path | 1/8 | 3.094 |
| MMA | **8/8** | 0.451 |

On the 13 cases with a prompt shorter than a chunk both paths are the same. The bug didn't surface
because no test checked the shape of the second chunk — the rows are the same, but
the causal prefix doesn't start at one. It's now covered by the test
`prefill_mma_matches_rowwise_decode_on_a_later_chunk`.

# Batched lm_head — 2026-09-12

> **Single points from one session, without confidence intervals.** The comparison with
> vLLM here was taken before the protocol
> [`docs/11-experiments.md`](../../docs/11-experiments.md) existed: one or two runs per
> point, no variant order rotation and no bootstrap CI, on a partially busy
> card. These multipliers are superseded by a comparison suite run and must not
> be quoted as a project result. The conditions of each point are given below.

RTX 5090, `Qwen3.8-27B-QUASAR-NVFP4`, KV fp8, context 2048, 200 requests of
256 prompt tokens and 128 new.

## Where it came from

A per-phase measurement (`--bin stepprofile`) showed that at concurrency 64 the
`lm_head` phase takes 15.8 ms of a 42.8 ms pure decode step — 36%. The cause isn't
the kernel as such: `MAX_LOGITS_BATCH = 8`, the hidden tile sat in static
shared memory, and the batch was cut into groups of eight, each of which re-read
all 1.27 GB of the table. The cost was strictly linear in batch, 0.247 ms
per row starting at batch 16, and at batch 1 the defect didn't show at all.

The batched kernel computes a `[128 vocab rows × BATCH]` tile: the table is read
once for any batch up to 96. Staging hidden in shared is amortized by
the number of rows per block — at 128 rows it's half the cost of the weights themselves, whereas
at eight it would be an order of magnitude more. The narrow kernel is kept for batch ≤ 8:
there it already runs at 1547 GB/s, and there's no reason to touch single-sequence
latency.

## Phase

| concurrency | before, ms | after, ms | step share before | after |
|---|---:|---:|---:|---:|
| 64 | 15.82 | 4.67 | 35.5% | 13.7% |

Pure decode step at c=64: 42.8 → 32.0 ms.

## End-to-end throughput

| concurrency | before, tok/s | after, tok/s | speedup | vLLM 0.28.0 |
|---|---:|---:|---:|---:|
| 32 | 749.97 | 839.15 | 1.12× | 982.64 |
| 64 | 952.60 | 1138.70 | 1.20× | 1001.80 |
| 80 | 1026.24 | 1254.10 | 1.22× | 984.70 |

Batch 1 latency didn't change: 13.81 → 13.47 ms per token (narrow kernel).

The phase stopped being bandwidth-bound: 1.27 GB in 4.67 ms is 272 GB/s, i.e.
about 35 TFLOP/s of scalar FP32. The remaining headroom is rewriting accumulation onto
tensor cores; that's roughly another 3 ms of 32.

## Caveat

At `--concurrency 80` the default `--memory-limit-gb 28` no longer fits:
16.43 GB of weights plus 11.88 GB of cache and state. Measured with a 30 GB limit.
That's the per-slot KV reservation that a shared page pool removes.

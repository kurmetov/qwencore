# Prefill DeltaNet — 2026-09-12

> **Single points from one session, without confidence intervals.** The comparison with
> vLLM here was taken before the protocol
> [`docs/11-experiments.md`](../../docs/11-experiments.md) existed: one or two runs per
> point, no variant order rotation and no bootstrap CI, on a partially busy
> card. These multipliers are superseded by a comparison suite run and must not
> be quoted as a project result. The conditions of each point are given below.

RTX 5090, `Qwen3.8-27B-QUASAR-NVFP4`, KV fp8, context 2048, 200 requests of
256 prompt tokens and 128 new.

## Where it came from

After the batched `lm_head`, the biggest block of the mixed step was DeltaNet
prefill: 37.1 ms of 90.6 on a 448-token segment. Measuring by segment length
(32 / 112 / 224 / 448 tokens) showed strict linearity — 62 µs per token — i.e.
it's not launch overhead, but that the per-token work runs almost
without parallelism.

## Scan: a warp instead of a block

The recurrence is sequential over tokens, but state rows are independent —
the only thing linking them is the scalar `k.q`. The old kernel kept `k` and `q` in shared,
put three `__syncthreads()` on every token and computed `k.q` with one warp out of
eight while the other seven waited at the barrier.

Now a warp owns a state row and walks the tokens by itself: `k` and `q`
are read as vectors straight from global, the warp computes `k.q` for itself in the same
butterfly as `S k` and `S q` — three reductions in five steps instead of two in a row with
a broadcast through shared. Not a single barrier is left. The grid puts one
state row per warp: 48 heads x 16 blocks = 768 blocks on 170 SMs.

## Convolution: parallel over tokens

`causal_conv_prefill_kernel` gave a thread a whole channel and walked tokens in
program order — 40 blocks on 170 SMs, less than a quarter of the chip. But a four-tap
convolution is parallel over tokens; only carrying the history
between chunks is sequential. A thread now computes one output `(channel, token)`, and
the history tail is written by a separate launch — writing it in the same
kernel would mean a race with blocks that read the same history on the first three
tokens.

## Mixed step phases (c=64, segment 448)

| phase | before, ms | after, ms |
|---|---:|---:|
| `delta.scan_prefill` | 27.71 | 15.93 |
| `delta.prepare_prefill` | 9.38 | 1.64 |
| whole step | 90.56 | 69.86 |

The pure decode step didn't change: 32.0 ms — both changes only touch prefill.

## End-to-end throughput

| concurrency | before today | after lm_head | after prefill | total | vLLM 0.28.0 |
|---|---:|---:|---:|---:|---:|
| 32 | 749.97 | 839.15 | 902.64 | 1.20× | 982.64 |
| 64 | 952.60 | 1138.70 | 1274.74 | 1.34× | 1001.80 |
| 80 | 1026.24 | 1254.10 | 1447.44 | 1.41× | 984.70 |

TTFT p50 at c=64: 9320 -> 6692 ms, at c=80: 9737 -> 6725 ms.

## What's left in the scan

15.9 ms is 36 µs per token. A diagnostic run with a shortened butterfly
(three steps instead of five) gives 13.2 ms, i.e. shuffles account for about 6 ms
of 15.9. Next is a layout with eight lanes per state row instead of
thirty-two: the butterfly becomes three steps, and a warp covers four rows,
i.e. six times less shuffle work per row. But the block count
stays at 192 on 170 SMs, and the gain will hit occupancy — that's for measurement, not
obvious.

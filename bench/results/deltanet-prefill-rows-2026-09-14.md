# DeltaNet prefill: four state rows per warp — 2026-09-14

Continuation of `deltanet-prefill.md`. There a warp got a whole state row and
got rid of barriers; 28.4 µs per token remained, a noticeable part of it in
a five-step butterfly over two values.

## What changed

The old layout gave a warp one state row. The address of `k` and `q` depends
only on the lane, so all warps of a block and all blocks along z read the same
`k` and `q` rows: at kBlock=256 and 16 blocks along z that's 128 reads of the same thing.
The second problem nearby: `warp_sum2` is five shfl steps over two independent sums,
i.e. the chain is bound by shuffle latency, not its throughput.

Both are cured by one thing: a warp handles `kPrefillRows = 4` rows at once. `k` and `q` are read
once per four rows, and the reduction is one butterfly over `2 * kRows = 8`
values (`warp_sum_n`) — the same five steps, but eight independent chains
overlap each other's latency.

Prefill also got its own block size: `kPrefillBlock = 64`. With four rows per
warp the grid is a choice between "few large blocks" and "many small ones"; small ones
spread more evenly over 170 SMs. Grid: 48 heads x 16 blocks = 768 blocks.

A note on the plan from the previous write-up: there the next step was a layout with
eight lanes per row (three butterfly steps instead of five). It wasn't needed —
the gain came from the `k`/`q` reads and shuffle overlap, and the butterfly length stayed
the same.

## Measurement

`cargo run --release -p qwc-cuda --bin deltaprefillbench`, three runs per
side back to back, RTX 5090.

| tokens per chunk | before, µs/token | after, µs/token | |
|---|---:|---:|---|
| 512 | 27.77 / 27.89 / 28.85 | 23.10 / 23.11 / 23.13 | 1.21x |
| 1024 | 28.35 / 28.41 / 28.42 | 23.01 / 23.08 / 23.11 | 1.23x |

Scan of 48 layers on a 1024-token chunk: 29.10 -> **23.56 ms** (-19%).

## Layout sweep

One session, two runs per point, 1024-token chunk:

| rows per warp | block | µs/token |
|---:|---:|---:|
| 2 | 64 | 24.61 / 24.91 |
| **4** | **64** | **23.99 / 24.06** |
| 8 | 64 | 43.52 / 43.99 |
| 2 | 128 | 24.56 / 24.64 |
| 4 | 128 | 25.29 / 25.42 |
| 4 | 256 | 32.15 / 32.42 |

Eight rows per warp is 32 registers for the state alone, and the point
collapses twofold: registers spill. The chosen point 4/64 is 2-3% better than
its neighbors, i.e. the choice is not critical, but not random either.

## On measurement cleanliness

The first sweep run had to be thrown out entirely: in its middle vLLM came up on the GPU
(27 GB, a startup profiling run), and the same 4/64 point showed
36.9 µs/token instead of 24.0. The numbers above were taken when vLLM was already idle.
Spread between runs on a quiet machine is about 1%, between sessions up to 6%.

## What's not here

The end-to-end effect on the engine step wasn't measured: `stepprofile` asks for about 30 GB, and
only 2.8 GB was left on the card at that moment. Expectation from the step breakdown: the prefill
scan gave 15.9 ms of a 69.9 ms step, minus 19% of it is about 3 ms per step,
and that's before MMA attention removed its 117 ms from the step.

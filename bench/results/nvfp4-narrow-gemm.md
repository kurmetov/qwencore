# Narrow W4A4 shapes — 2026-09-12

## What was wrong

A microbenchmark over batch up to 512 showed: `down_proj` [5120, 17408] is stuck
at ~780 GB/s at any batch from 3 to 256, while `gate_proj` with the same weight
mass gets 1335 GB/s. The only difference is the shape.

The 128x128 GEMM tile cuts the output into `ceil(M/128) * ceil(N/128)` blocks. For
`gate/up` at M<=128 that's 136 blocks on 170 SMs, and for `down` 40: three quarters
of the chip sit idle, and bandwidth is cut in exactly that proportion.

## What was done

A second kernel instance: cooperative + a stream-K scheduler, which splits
the problem along K and makes up occupancy. Ping-pong doesn't support stream-K
(a static assert in CUTLASS), hence a second instance rather than changing the scheduler
of the existing one. Choice by shape: a problem counts as narrow when there are half as many
tiles as SMs. The threshold has margin — right at the boundary stream-K already loses
(down at M=512: 78.3 vs 76.4 µs), because the reduction costs more than
the added occupancy.

As a side effect a sizing bug surfaced: the workspace requirement is not monotonic in M —
the smaller M, the more blocks take part in the reduction. The arena sized itself with one
value at the max row count and crashed on a mixed step with M<512.

## Measurement

`down`, weight 44.6 MB per matrix:

| batch | before, GB/s | after, GB/s |
|---|---:|---:|
| 3 | 784 | 1017 |
| 16 | 782 | 1016 |
| 64 | 785 | 1016 |
| 128 | 771 | 1018 |
| 256 | 774 | 1017 |
| 512 | 656 | 668 |

`gate/up` and `q+gate` didn't change — they're on ping-pong.

The `gemm.mlp_down` phase of a pure decode step at c=64: 4.77 -> 3.72 ms (mean
of two runs per configuration), step 33.38 -> 32.87 ms.

## End-to-end effect and a caveat about drift

End-to-end throughput at c=64, three measurements back to back in one session:

| build | tok/s |
|---|---:|
| HEAD (after deltanet) | 1206.25 |
| + ping-pong (same base) | 1211.24 |
| + stream-K | 1223.64 |

I.e. +1.4%. Less than on the decode step, because the mixed step runs at
M=511, where `down` gives 160 tiles and stays on ping-pong: only pure
decode steps win.

Caveat: the same HEAD gave 1274.74 tok/s in the morning, and 1206.25 now. The machine
sagged about 5% during the day — only measurements from one session can be compared.
The numbers in `deltanet-prefill.md` and `lm-head-batched.md` were taken in the morning and are
comparable with each other, but can't be compared with this table.

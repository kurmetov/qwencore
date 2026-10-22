# Gated DeltaNet: the recurrent decode step

48 of Qwen3.8's 64 layers are linear attention with a recurrent state
`[48 heads, 128, 128]`. For every token the state is read and rewritten
in full: at batch=32 that's 21% of all decode-step traffic, twice the KV cache.

There is no ready-made SM120 kernel for this — it's the only part of the model
that had to be written from scratch.

## A recurrence form suitable for a single pass

```
S_t = a·S(I − b·k kᵀ) + b·v kᵀ  =  a·S + b(v − a·(S k))·kᵀ
o_t = S_t q                     =  a·(S q) + b(v − a·(S k))·(k·q)
```

The key point is that `S k` and `S q` are taken **in one pass**. After that the state
row already sits in registers and is rewritten in place:

```
one read + one write   instead of   three reads + one write
```

A naive implementation would make three passes: `S k`, then `S q`, then the update.
With 1.57 MB of state per layer per sequence that's three times the traffic
on the most expensive part of the step after the weights.

## Layout

* a warp owns a state row: 32 threads x 4 elements = exactly 128;
* 8-byte vector access — fully coalesced 256-byte transactions per warp;
* block of 256 threads = 8 warps;
* `k·q` is a scalar shared by all rows, computed once per block;
* rows are additionally split across blocks via `gridDim.z`.

## Why row splitting was needed

The first version put one block per v head: `grid = (48, batch)`.
At batch=1 that's **48 blocks on 170 SMs** — three quarters of the chip sat idle.

| | before | after |
|---|---:|---:|
| batch=1, per layer | 9.1 us | **2.7 us** |
| batch=1, 48 layers | 0.44 ms | **0.13 ms** |
| batch=1, % of ceiling | 24% | **84%** |

The split is correct because state rows are updated independently:
the only thing linking them is the scalar `k·q`, which is cheaper to recompute in every block
than to pass between them.

## Measurements

The read-write ceiling was measured separately: **1408 GB/s**.
The pass goes over 48 separate buffers in a row, like in a real decode step —
looping over one buffer is not allowed, it would settle in the 96 MiB L2 and inflate the result.

| batch | working set | per layer | 48 layers | GB/s | % of ceiling |
|---:|---:|---:|---:|---:|---:|
| 1 | 75 MB | 2.7 us | 0.13 ms | 1182 | 84% |
| 4 | 302 MB | 9.9 us | 0.47 ms | 1272 | 90% |
| 8 | 604 MB | 17.1 us | 0.82 ms | 1472 | 105%* |
| 16 | 1208 MB | 37.6 us | 1.80 ms | 1340 | 95% |
| 32 | 2416 MB | 77.5 us | 3.72 ms | 1299 | 92% |

\* above 100% because the ceiling was measured on separate source and
destination buffers, while here the row is already in L2 after the read and the write doesn't need
to reload the line.

**The kernel saturates memory: 84-95% of the achievable ceiling.** There is nothing
left to optimize here — the remainder at batch=1 (2.7 us vs the theoretical 2.2 us) is
launch overhead, which CUDA graphs remove.

The batch=32 result — 3.72 ms for 48 layers — matches the prediction of the
roofline model (state = 21% of step traffic).

## Correctness

`cargo test -p qwc-cuda --test delta_net`

* checked against a CPU reference working with the same bf16 state representation,
  so a mismatch can only come from summation order;
* tolerance: output 2e-3 (computed in fp32), state 1 bf16 ULP (~0.8%) on no more
  than 2% of elements;
* a separate check: with `beta = 0` the write is disabled and the state must
  decay by exactly `alpha` — checked bitwise.

## What's next

Prefill requires a **chunked parallel scan**, not a recurrence:
the sequential dependency across tokens kills parallelism. It's noticeably
harder and affects only TTFT. Starting point — the CUTLASS example
`112_blackwell_ssd`.

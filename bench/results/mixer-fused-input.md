# Fused mixer input projection — 2026-09-12

## Where it came from

Splitting `gemm.la_qkvab` into separate phases showed: the two projections
`in_proj_a` and `in_proj_b` of shape [48, 5120] cost 2.37 ms — more than qkv
(1.56, N=10240) and z (1.37, N=6144) combined. With a 128x128 tile a projection with N=48
gives one block on 170 SMs.

## What was done

qkv, z, a and b read the same input and are concatenated into one matrix
[16480, 5120] at load. The concatenation is bit-exact: in all 48 layers they
share both `weight_global_scale` and `input_global_scale` (checked across all
five shards), and the GEMM alpha is one scalar per launch, so there would be no way to
merge different scales. A scale mismatch is now a load error, not a supported
case.

Consumers get not a separate buffer but a `RowView` — a view onto the arena's columns
with its stride. Convolution, gated normalization and gated RMSNorm were taught to read
inputs with an arbitrary stride; their outputs stay dense as before.

As a side effect: four launches per layer became one, and the input is quantized once
instead of four times.

## Measurement

| | before | after |
|---|---:|---:|
| qkv + z + ab phases, decode c=64 | 5.30 ms | 2.14 ms |
| pure decode step, c=64 | 32.87 ms | 30.12 ms |
| mixed step (segment 448) | 73.9 ms | 70.3 ms |

End-to-end throughput, all measurements from one session:

| concurrency | HEAD | + stream-K | + fused projection |
|---|---:|---:|---:|
| 32 | — | — | 959.36 |
| 64 | 1206.25 | 1223.64 | 1317.97 |
| 80 | — | — | 1478.09 |

At c=64 that's +9.3% over the session's HEAD.

## Verification

`bin/load` checks the first 26.2 MB of the fused matrix against the checkpoint's `in_proj_qkv`
byte by byte — which also checks that the concatenation didn't swap parts.
Generation at batch 1 (W4A16 path, no stride) and at batch 8 (W4A4 path, with
offset reads) gives the same token sequence.

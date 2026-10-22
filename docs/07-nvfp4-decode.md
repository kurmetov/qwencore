# NVFP4 decode: W4A16, W4A4 and fused MLP

## What is implemented

`qwc-cuda` got its first real projection dataplane:

* `nvfp4::Linear` loads `weight_packed` and `weight_scale` directly from
  compressed-tensors checkpoint;
* `forward_w4a16` computes NVFP4 weights × BF16 activations for batch 1..4;
* `swiglu_w4a16` merges `gate_proj + up_proj + SiLU × mul` into one kernel;
* our own BF16→NVFP4 quantizer writes packed E2M1 and the SM120 128×4
  scale layout directly, without an intermediate row-major copy;
* `forward_w4a4_quantized` launches block-scaled tensor-core MMA for batch
  3..32 with BF16 output;
* `select_decode_kernel` picks W4A16/W4A4 by batch and projection shape;
* there is a CPU oracle for the E2M1/E4M3 formats, a synthetic GPU test and a check on
  real QUASAR tensors.

So far this is a small-batch decode path, not a full token loop and not prefill.

## Format and scales

The checkpoint stores:

```text
weight_packed        u8      [out, in/2]   two E2M1 per byte
weight_scale         e4m3    [out, in/16]  local scale
weight_global_scale  f32     [1]           global divisor
```

Compressed-tensors stores the global scale as a divisor. So for W4A16:

```text
W[row, k] = E2M1(weight_packed[row, k])
          * E4M3(weight_scale[row, k/16])
          / weight_global_scale

Y = X_bf16 * W^T
```

`input_global_scale` is needed only for W4A4 activation quantization and is intentionally
unused in W4A16. For each group of 16 activations:

```text
SFA = E4M3(input_global_scale * max(abs(X_group)) / 6)
A4  = E2M1_RNE(X_group * input_global_scale / SFA)

Y = alpha * MMA(A4 * SFA, W4 * SFW)
alpha = 1 / (input_global_scale * weight_global_scale)
```

The formula and treating serialized globals as divisors match the handling of
NVFP4 in [vLLM](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py).

The 4-bit weights are not repacked on any path: the `[N,K]` row-major checkpoint
physically matches the required GEMM `B[K,N]` column-major. The scale matrix is
permuted once at load in 128×4 blocks and stays the only copy:
both W4A4 and the hand-written W4A16 read this layout. If raw and swizzled scales
were stored at the same time, across all the model's quantized parameters it would cost another
1.52 GB VRAM.

## Kernel layout

The main tile computes 16 output rows with a block of 128 threads:

```text
threadIdx.x = 8 K threads per row
threadIdx.y = 16 output rows
K-tile      = 256
X tile      = BF16 [batch, 256] in shared memory
W fragment  = 128 bit per thread
```

The activation tile is loaded into shared once per 16 rows. The weight fragment
is read once and used right away for all sequences in the batch.
Batch 1, 2, 3 and 4 are separate compile-time kernels: B=1 doesn't pay in registers
and branches for B=4.

For narrow projections (`out < 8192`) a different layout is used: 32 K threads
per row and a 512 K tile. This doubles the number of independent warps and halves
the number of barriers on the long `down_proj [5120, 17408]`.

The fused MLP reads two matrices but loads X only once and writes directly:

```text
intermediate = silu(gate_proj(X)) * up_proj(X)
```

There are no separate gate/up buffers and no separate elementwise kernel.

W4A4 uses SM120 block-scaled MMA:

```text
tile             128 × 128 × 256
schedule         warp-specialized ping-pong
A                packed E2M1 [batch,K], row-major
B                checkpoint packed E2M1 [N,K], as column-major [K,N]
SFA/SFB          E4M3, CUTLASS 128×4 swizzle
D                BF16 [batch,N]
workspace        0 bytes
```

The stock CUTLASS example uses K=128 cooperative. A direct sweep on the RTX
5090 showed K=256 ping-pong is better; N=64 and K=256 cooperative were rejected
after negative A/B measurements.

## Correctness

The synthetic test checks batch 1..4, both hardware layouts and an incomplete
last tile. A separate example loaded the real tensors
`layers.0.mlp.gate_proj/up_proj` from `Qwen3.8-27B-QUASAR-NVFP4`.

```text
gate shape                         17408 × 5120
checkpoint weight_global_scale     6371.5557
projection max scaled error        0.003859
fused SwiGLU max scaled error       0.003731
W4A4 batch=4 max scaled error       0.003887
CUTLASS workspace                   0 B
```

The error includes BF16 rounding of the output. Bit layout, nibble order and both scale
levels are confirmed on the real checkpoint, not just on synthetic data.

## RTX 5090: streaming benchmark

Command:

```bash
cargo run --release -p qwc-cuda --bin nvfp4bench
```

The benchmark cycles through four different matrices, so the working set
exceeds the 96 MiB L2. The number is the median of seven warmed-up series. On a busy desktop
GPU, clocks and competing graphics processes cause noticeable spread, so
direct A/B measurements in one series are what matters for decisions.

Fused MLP vs two separate projection launches (without a separate SiLU in the
baseline, so the comparison is conservative):

| batch | two projections | fused SwiGLU | speedup |
|---:|---:|---:|---:|
| 1 | 103.16 us | 99.19 us | 1.04× |
| 2 | 105.27 us | 99.61 us | 1.06× |
| 3 | 152.16 us | 100.38 us | 1.52× |
| 4 | 192.31 us | 116.86 us | 1.65× |

The main result is that one weight serves several sequences.
In early isolated series, separate gate/q+gate projections for B=1–2
reached 84–88% of the measured DRAM ceiling, and two sequences were often
computed in practically the time of one. The median benchmark intentionally doesn't hide
desktop contention and prints the current share of the ceiling on every run.

For comparison, the reference CUTLASS `91_fp4_gemv` on four different gate/up
matrices showed 0.135 ms for the whole streaming call and 1384 GiB/s. It's a strong
batch=1 oracle, but its stock batch duplicates matrix A, the scale layout is swizzled,
and the output is FP4 again. It doesn't replace the shared-weight server path.

### W4A4 crossover

The quantizer takes about 2.3 us for batch 3..32. Below is a direct A/B measurement of
K=256 ping-pong; the W4A4 numbers already include the quantizer:

| shape | batch | W4A16 | W4A4 total | W4A4 / A16 |
|---|---:|---:|---:|---:|
| gate/up 17408×5120 | 4 | 91.66 us | 75.60 us | 1.21× |
| q+gate 12288×5120 | 4 | 43.88 us | 41.10 us | 1.07× |
| down 5120×17408 | 4 | 50.67 us | 66.60 us | 0.76× |

At batch=32 one W4A4 projection took 69.34 us for gate/up, 63.62 us for
down and 36.05 us for q+gate. Desktop contention noticeably changes the absolute numbers,
so the dispatcher relies on the relative A/B result of one series.

## What is not yet proven

* W4A16 removes the activation quantization error, but its effect on the quality of the whole QAT
  model must be confirmed by perplexity/task-eval; "more bits" doesn't count
  as automatic proof of quality.
* The fused MLP doesn't yet include `down_proj`, residual and RMSNorm.
* The W4A4 MLP still launches gate/up as separate GEMMs; a shared persistent
  mainloop or fused layout is needed so that one CTA wave streams both matrices.
* The effect of W4A4 prefill/decode on perplexity relative to the reference runtime
  still has to be measured separately.

## Target dispatch

```text
batch 1..3   W4A16 custom GEMV / fused MLP  — quality, low latency
batch 4      N>K: W4A4; N<=K: W4A16         — measured per-shape boundary
batch 5..32  W4A4 SM120 MMA                 — one pass over the weights
prefill      W4A4 SM120 GEMM, M=128          — chunked causal scan
```

The boundary isn't hardwired forever: the final autotune picks it separately for
each shape by goodput and ITL, and a quality profile can force
W4A16 for longer.

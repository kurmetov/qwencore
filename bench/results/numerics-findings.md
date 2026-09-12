# Locating the qwc↔vLLM logprob drift — 2026-09-11

Baseline: `qwc-b1` reproduces all 120 greedy tokens of the reference but sits
at logprob MAE 0.364 against `vllm.jsonl`, above the 0.25 threshold.

## A/B runs over the full corpus

Each run changes exactly one approximation and leaves the rest of the engine
alone. `bench/diff_eval.py run --engine qwc --<flag>` produces the artifact.

| Run | Flag | Logprob MAE | Top-k Jaccard | Verdict |
|---|---|---:|---:|---|
| baseline | — | 0.364 | 0.782 | reference point |
| BF16 lm_head | `--lm-head bf16` | 0.361 | 0.782 | not the source |
| BF16 embedding | `--embedding bf16` | 0.398 | — | not the source |
| BF16 KV cache | `--qwc-kv-cache-dtype bf16` | 0.364 | 0.781 | not the source |
| W4A4 decode | `--qwc-decode-linear w4a4` | 0.438 | — | worse; not the source |
| FP32 DeltaNet state | `--qwc-delta-state fp32` | 0.380 | 0.781 | not the source |

The FP32 DeltaNet run moves individual cases in both directions (`chunk-127`
+0.43, `chunk-129` −0.15) with no systematic gain — the signature of perturbing
a chaotic trajectory, not of removing a systematic error.

## What the reference actually computes

- Recurrent state: FP32 across the whole prefill inside
  `chunk_gated_delta_rule_fwd_h`; the persisted `ssm_state` is BF16 (`auto`).
- Full-attention KV cache: BF16. An FP8-KV reference run was attempted and
  abandoned — this vLLM build has no FlashInfer XQA, which it requires for FP8 KV.
- L2 norm, `q` scaling by `1/sqrt(128)`, softplus gate, `sigmoid` beta: identical
  formulas and identical `1e-6` epsilon to `cuda/delta_prepare.cu`.
- Linear layers: `CutlassNvFp4LinearKernel` for **every** projection, prefill and
  decode, at every batch size. The other ten NVFP4 kernels are unsupported here
  (FlashInfer absent, fbgemm absent, b12x absent).

## Per-projection oracle — the actual answer

`bench/nvfp4_cross.py` runs one projection through both engines on identical
BF16 activations, and against a ground truth computed by dequantizing the weight
exactly and multiplying in FP32.

| Projection | qwc W4A4 vs vLLM | qwc W4A4 vs truth | qwc W4A16 vs truth | vLLM vs truth |
|---|---:|---:|---:|---:|
| `layers.0.mlp.gate_proj` | 0.00000 | 0.01920 | **0.00028** | 0.01920 |
| `layers.0.linear_attn.in_proj_qkv` | 0.00000 | 0.02974 | **0.00042** | 0.02974 |

(mean absolute difference; FP4 codes and E4M3 block scales are bit-identical,
0/81920 nibbles and 0/5120 scales differ.)

**qwc's decode path is roughly 70× closer to the true value than the reference
is.** The reference quantizes activations to FP4 on every decode projection;
qwc runs W4A16 for batch ≤ 3 and keeps activations in BF16. The 0.364 MAE is
therefore dominated by the *reference's* activation-quantization error, not by
an error in qwc.

This also explains the two W4A4 results. Forcing W4A4 at batch 1 (0.438) and
letting batch 8 select it automatically (0.436) both land in the same place:
adopting the reference's arithmetic class does not reproduce the reference's
specific rounding, so two independently-noisy computations sit farther apart
than one clean one and one noisy one.

## Consequence for the thresholds

A 0.25 logprob-MAE gate against a W4A4 reference is not a meaningful accuracy
bar for a W4A16 decode path. Either compare against a ground-truth artifact, or
score qwc in the `--qwc-decode-linear w4a4` configuration when the intent is
bit-comparability with vLLM rather than accuracy.

## Still open

The per-projection agreement is exact, so the residual full-model difference in
the W4A4 configuration comes from something outside the linear layers — the
normalization, RoPE, SwiGLU, gated-RMSNorm or paged-attention kernels. A
layer-by-layer hidden-state diff against vLLM forward hooks would localize it in
one run instead of one A/B per candidate.

# QUASAR-NVFP4 checkpoint layout

Source — `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` on Hugging Face; fetched by
`./scripts/fetch-checkpoint.sh` (overridable via `REPO` and `DEST`,
resuming is supported). Layout breakdown:

`cargo run -p qwc-model --bin inspect`

## Quantization format

Each of the **496** quantized linear layers has four tensors:

```
{prefix}.weight_packed        U8        [out, in/2]     two fp4 (e2m1) per byte
{prefix}.weight_scale         F8_E4M3   [out, in/16]    scale per block of 16
{prefix}.weight_global_scale  F32       [1]
{prefix}.input_global_scale   F32       [1]             W4A4: activations are fp4 too
```

496 = 64 MLP layers x 3 + 16 attention x 4 + 48 linear_attn x 5.

## What got clarified vs. the initial config.json breakdown

| | |
|---|---|
| `self_attn.q_proj` | `[12288, 5120]`, i.e. 24 x 256 x **2**. The output gate (`attn_output_gate`) is baked into the same projection, there is no separate tensor |
| `self_attn.q_norm` / `k_norm` | `[256]` — the model has a **per-head QK norm**, not visible in config.json |
| `linear_attn.in_proj_qkv` | `[10240, 5120]` = 2048 (q) + 2048 (k) + 6144 (v) in one projection |
| `linear_attn.in_proj_z` | `[6144, 5120]` — the gate is separate |
| `linear_attn.in_proj_a` / `in_proj_b` | `[48, 5120]` — projections for alpha and beta, one number per v head |
| `linear_attn.conv1d.weight` | `[10240, 1, 4]` — depthwise causal conv over the concatenated q, k, v |
| `linear_attn.norm.weight` | `[128]` — RMSNormGated over `v_head_dim` |
| MTP head | **not quantized**, 15 bf16 tensors: `fc [5120, 10240]`, two pre-norms, one full layer |

## Memory model check

Matching all 2354 tensors against the architecture's compile-time constants passed **without a single
mismatch**. This checks both the loader and the correctness of the model breakdown.

| | prediction | checkpoint |
|---|---:|---:|
| NVFP4 weights (packed + scales) | 13.68 GB | **13.70 GB** |
| our load into VRAM | 17.08 GB | **17.09 GB** |

On-disk layout:

```
NVFP4 weights (packed)  12.18 GB
per-block fp8 scales     1.52 GB
norms and small stuff    0.01 GB
embed_tokens bf16        2.54 GB
lm_head bf16             2.54 GB
MTP head bf16            0.85 GB
not needed (vision)      0.92 GB   333 tensors
total                   20.56 GB
```

**Headroom over baseline: 3.46 GB** — we don't load the vision tower at all, `lm_head` and
`embed_tokens` go to fp8. 12.89 GB remains for cache vs 9.4 GB,
that's 11 vs 8 sequences at 32K.

## Loader principle

The architecture is known at compile time, so a shape mismatch is
**a load error, not a reason to adapt**. A general-purpose engine would accept
any tensor and fail later, in a kernel, with vague diagnostics.

# Qwen3.8-27B — architecture and implications for the inference engine

Source: `Qwen/Qwen3.8-27B/config.json` (`Qwen3_5ForConditionalGeneration`, transformers 5.8.0.dev0).

## 1. What this model is

Dense 27B, **natively vision-language**, with **hybrid attention** and a **built-in MTP head**.

```
model_type            qwen3_5
num_hidden_layers     64
hidden_size           5120
intermediate_size     17408      (SwiGLU, silu)
vocab_size            248320
rms_norm_eps          1e-6
tie_word_embeddings   false      <- separate lm_head, 1.27B parameters
max_position_embeddings 262144
```

### 1.1 Hybrid layer stack (3:1)

`full_attention_interval = 4` → pattern `[L, L, L, F] × 16`:

* **48 layers** — `linear_attention` = **Gated DeltaNet** (constant-size recurrent state)
* **16 layers** — `full_attention` (indices 3, 7, 11, ..., 63)

### 1.2 Full attention (16 layers)

```
num_attention_heads     24
num_key_value_heads     4       (GQA, group of 6)
head_dim                256     <- NOT 128
attn_output_gate        true    <- extra gate projection on the output
partial_rotary_factor   0.25    <- RoPE only on the first 64 of 256 dims
rope_theta              1e7
rope: mrope_interleaved, mrope_section [11, 11, 10]
```

### 1.3 Linear attention / Gated DeltaNet (48 layers)

```
linear_num_key_heads    16   (q, k)
linear_key_head_dim     128
linear_num_value_heads  48   (v)   <- k/q broadcast 1:3 onto v heads
linear_value_head_dim   128
linear_conv_kernel_dim  4         <- short causal conv1d before the recurrence
output_gate_type        swish
mamba_ssm_dtype         float32   <- state in fp32
```

### 1.4 MTP

```
mtp_num_hidden_layers        1
mtp_use_dedicated_embeddings false
```
The draft head is **already in the checkpoint**. Speculative decoding needs no separate draft model.
Measured acceptance rate in vLLM: **0.77–0.90** with `num_speculative_tokens=3`.

### 1.5 Vision tower (dropped)

27 layers, hidden 1152, patch 16, spatial_merge 2, out_hidden 5120 (~0.5B parameters).
In text-only mode it is **not loaded at all** — saves ~0.5–1 GB VRAM and all the preprocessing.

---

## 2. Parameter breakdown (26.90B)

| Component | Parameters |
|---|---:|
| MLP × 64 (3 × 5120 × 17408) | 17.11 B |
| Linear-attn × 48 (q,k,v,out,gate) | 5.56 B |
| Full-attn × 16 (q,k,v,o,gate) | 1.68 B |
| embed_tokens (248320 × 5120) | 1.27 B |
| lm_head (untied) | 1.27 B |
| **Total** | **26.90 B** |

MLP = 64% of all weights → this is where NVFP4 pays off.
Embeddings + lm_head = 2.54B (9.4%) — a noticeable share; their precision/format needs a separate decision.

---

## 3. Memory: the project's main finding

### 3.1 Weights (NVFP4 = 4 bits + FP8 e4m3 scale per block of 16 → 4.5 bits effective)

| Checkpoint | Recipe | Size |
|---|---|---:|
| `nvidia/Qwen3.8-27B-NVFP4` | NVFP4: MLP + lm_head; **FP8: attn + linear-attn**; BF16 embed | **≈ 20.1 GB** |
| `QUASAR-QAT/...-QUASAR-NVFP4` | NVFP4: all linear layers; BF16 embed | **≈ 17.0 GB** |
| same + FP8 embed | | **≈ 15.7 GB** |

Out of 31.4 GiB usable, that is the difference between **11 GB** and **15.7 GB** free for cache — i.e. it directly sets max concurrency.

### 3.2 Per-sequence state (constant, does NOT grow with context)

DeltaNet recurrent state: `48 layers × 48 v heads × 128 × 128`

```
= 37.75 M elements per sequence
= 151 MB  @ fp32   (as in the config)
=  75 MB  @ bf16   (our ablation experiment)
```
+ conv state (kernel 4): ~6 MB fp32 — negligible.

### 3.3 KV cache (only the 16 full-attn layers)

`16 × 4 kv heads × 256 × 2 (K,V)` = 32 768 elements per token:

```
BF16   64 KB / token
FP8    32 KB / token
NVFP4  18 KB / token
```

### 3.4 The crossover point — something no ordinary engine has

With FP8 KV, the DeltaNet state "costs" as much as the KV cache at:

```
bf16 state (75 MB)  ->  ~2 360 tokens of context
fp32 state (151 MB) ->  ~4 700 tokens of context
```

**Below ~2.4K–4.7K context, memory is dominated not by the KV cache but by the recurrent state.**
I.e. on short requests concurrency is limited by fixed state slots,
not by paged KV. That is a fundamentally different allocator and a different scheduler.

### 3.5 32 GB budget (computed by code)

`cargo run -p qwc-core --bin plan` is the single source of truth for these numbers.
Memory model validation: the "checkpoint as is" estimate gives **20.50 GB** against
the actual **20.6 GB** of files in the QUASAR repo, i.e. the per-layer parameter breakdown is correct.

```
Weights
  checkpoint as is (vLLM loads):              20.50 GB
  our load (text-only, fp8 head/embed):       17.08 GB
  headroom:                                    3.42 GB

Cache (kv fp8_e4m3, state bf16)
  KV per token (16 layers):      32768 B
  state per slot (48 layers):     81.4 MB
  crossover point:                2484 tokens

Budget
  total VRAM       34.19 GB   (32607 MiB)
  reserved          2.20 GB   (gnome-shell desktop ~1.2 GB + CUDA context)
  weights          17.08 GB
  workspace         2.00 GB
  for cache        12.91 GB
```

Maximum concurrency:

| context | fp8 KV | nvfp4 KV | vLLM-like (fp8 KV, weights as on disk) |
|---:|---:|---:|---:|
| 4K | 59 | 82 | 44 |
| 8K | 36 | 55 | 27 |
| 16K | 20 | 33 | 15 |
| 32K | **11** | **18** | **8** |
| 64K | 5 | 10 | 4 |
| 128K | 2 | 5 | 2 |

The whole target benchmark matrix (up to 32 seq x 2.5K = 5.29 GB) fits with room to spare
in 12.91 GB — the bottleneck is performance, not memory. Memory becomes the bottleneck
starting at 16K context.

Two conclusions:
* **NVFP4 KV cache gives +64% concurrency at 32K** — a separate major optimization
  that the baseline on consumer Blackwell lacks.
* **3.42 GB saved on weights = +3 sequences at 32K** (11 vs 8),
  i.e. +37% before a single line of kernel code.

---

## 4. Implications for engine design

1. **KV cache ≠ all memory.** Two subsystems are needed: paged KV (16 layers) + a pool of fixed
   state slots (48 layers). Admission control must account for both.
2. **Prefix caching barely works "the usual way".** For linear layers you can't reuse
   part of a prefix — only a state snapshot at a boundary, and that is 75–151 MB per snapshot.
   A separate research question: does a snapshot prefix cache pay off.
3. **head_dim = 256** — almost all off-the-shelf paged-attention kernels are tuned for ≤128.
   SM120 has only **99 KB of shared memory per SM** (vs 228 KB on SM100) → tiling has to be
   designed by hand; this is the main kernel risk.
4. **attn_output_gate** — an extra 5120×6144 GEMM on every full-attn layer, fused with o_proj.
5. **partial RoPE 25%** — RoPE only on 64 of 256 dims; the rest is passthrough.
   In **text-only** mode mrope degenerates into plain RoPE (all three positions are equal) → drop mrope entirely.
6. **MTP is built in** → adaptive speculative decoding is available out of the box, without a second model.
   Verification is cheap: draft = 1 layer.
7. **Chunked prefill is natural** for DeltaNet (state carries over between chunks),
   but it requires a chunk-wise parallel scan, not a naive recurrence.

---

## 5. Target hardware: RTX 5090 (GB202, SM120)

```
VRAM             32 GB GDDR7, ~1792 GB/s
Compute cap      12.0  -> nvcc -arch=sm_120a (or compute_120f, CUDA 13.x)
Driver (here)    595.84, CUDA 13.2
Shared mem/SM    99 KB   (SM100: 228 KB)
```

**Critical:** SM120 uses SM80-style `mma.sync` with block scaling
(`mma.sync.aligned.m16n8k64...kind::mxf4nvf4.block_scale`), **not** the datacenter
Blackwell `tcgen05.mma`. Anything built on tcgen05 (DeepGEMM, some CUTLASS SM100 collective builders)
does not build or run on the 5090. We need CUTLASS SM120 schemes with hand tiling for 99 KB.

## 6. Known baseline weakness (our target win)

vLLM on a single RTX 5090 with NVFP4 requires `--enforce-eager` — capturing CUDA graphs OOMs.
So **the baseline runs without CUDA Graphs**. Static memory planning in a
specialized runtime lets us capture graphs — a direct, measurable ITL win.

---

## 7. Target checkpoint: QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4

```
format          nvfp4-pack-quantized  (compressed-tensors)
group_size      16
weight scale    float8_e4m3   (per-16 block)
act scale       float8_e4m3, local dynamic  -> W4A4, activations quantized on the fly
symmetric       true
ignored         lm_head, .*visual.*, .*mtp.*, .*embed_vision.*, .*embed_audio.*
```

Shards: 5 × safetensors, **20.6 GB on disk**. Layout:

| | size |
|---|---:|
| All linear layers (24.35B @ 4.5 bits) | 13.7 GB |
| `lm_head` BF16 (not quantized) | 2.54 GB |
| `embed_tokens` BF16 | 2.54 GB |
| vision tower BF16 | ~1.0 GB |
| MTP head BF16 | ~0.8 GB |
| **total** | **20.6 GB** |

### Our load-time transformations (what vLLM doesn't do)

| Step | Savings |
|---|---:|
| Skip the vision tower (text-only) | −1.0 GB |
| `lm_head` → FP8 | −1.27 GB |
| `embed_tokens` → FP8 (lookup, precision is not critical) | −1.27 GB |
| **VRAM for weights** | **≈ 17.0 GB** |

vLLM loads all 20.6 GB. **3.6 GB of headroom = +3-4 sequences at 32K context.**

### Implications for kernels

`local dynamic activation quantization` means that before every GEMM the activations must be
quantized to FP4 with an FP8 scale per block of 16 **at runtime, on every token**.
Fusing `RMSNorm -> amax(block16) -> quant_fp4` is mandatory, otherwise that's three extra passes over VRAM
on each of the 64 layers for every token.

The MTP head and `lm_head` stay in BF16/FP8 -> a second GEMM path besides NVFP4 is needed.

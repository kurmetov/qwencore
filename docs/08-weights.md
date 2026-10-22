# Loading weights into VRAM

`cargo run --release -p qwc-engine --bin load`

## Why a separate crate

`qwc-engine` is the first place where three boundaries meet: kernels
(`qwc-cuda`), the checkpoint (`qwc-model`) and scheduling (`qwc-runtime`). Below it they
intentionally know nothing about each other: the runtime stays CPU-testable and doesn't pull in
CUDA, and the checkpoint loader doesn't pull in device memory.

## Principle

Shapes and names come from `names::quant_linears` and `names::plain_tensors`, that
is, from the compile-time architecture, not from the checkpoint. The loader parses a layer
whole and requires every tensor described by the architecture to be used:
an extra tensor in a layer is also an error, not an "unknown field".

Shape and type checks happen before the first allocation, so a checkpoint with a
mismatch fails immediately and doesn't get to take VRAM.

## Host-RAM ceiling

After each layer `Checkpoint::evict_file_pages` is called: the pages are already
copied to VRAM, there's no reason to keep them in the working set. 13.7 GB passes through
the process, while peak resident memory is **0.37 GB**. Without eviction, mmap would leave
the whole read checkpoint in RSS, and the engine would eat RAM from neighboring processes.

## VRAM ceiling

`qwc_cuda::set_memory_limit` is set before the first allocation; every `DeviceBuffer`
reserves space under it and returns `cudaErrorMemoryAllocation` **before**
`cudaMalloc` if the request doesn't fit. The loader knows nothing about this:
it gets an ordinary CUDA error and aborts the load.

## Vocab matrices in FP8

`embed_tokens` and `lm_head` are stored in the checkpoint as `[248320, 5120]` BF16 and cost
2.54 GB each. We store them in E4M3 with an FP32 scale **per row**: exactly half
the size, and the error stays local — a badly scaled row doesn't spoil
the whole vocab. In a dot product the scale factors out of the sum, so
lm_head applies it once per row, not per element.

Quantization runs on the GPU in chunks of 8192 rows through a staging buffer: there's no reason
to hold 2.54 GB of BF16 in VRAM for the conversion. After each chunk the mmap pages
are evicted, otherwise peak host RAM would grow by the whole tensor (verified: 2.69 GB
vs 0.37 GB).

Three kernels in `cuda/vocab_fp8.cu`: per-row quantization, embedding lookup
straight to BF16, and the logits GEMV. Logits are computed one warp per vocab row, the hidden
staging lives in shared — without it, at batch 8 the hidden-state traffic from L2
already exceeds the weights themselves. The ceiling of one launch is batch 8; above that it's split into
groups at the cost of re-reading 1.27 GB.

### What the 2.54 GB cost us

Measured on real weights, not synthetic:

| | |
|---|---:|
| embed: max deviation relative to row range | 3.51% |
| embed: mean deviation | 0.324% |
| lm_head: max logit deviation (2048 rows vs exact BF16 arithmetic) | 2.76% of range |
| argmax on the checked subset | matches |

This is **not** yet a quality check: perplexity on real text is needed.
If it drops, the fallback is to keep `lm_head` in BF16: that's +1.27 GB,
i.e. the headroom over vLLM drops from 3.42 to 2.15 GB, but doesn't vanish.

## Measurements on the RTX 5090

| | |
|---|---:|
| layers | 64 (48 linear + 16 full) |
| quantized projections | 496 |
| NVFP4 weights + scales | 13.70 GB |
| norms, conv1d, A_log, dt_bias | 5.29 MB |
| embed + lm_head in FP8 | 2.54 GB (5.09 GB in the checkpoint) |
| **total in VRAM** | **16.25 GB** |
| checkpoint prediction for quantized | 13.70 GB (+2 MB scale-layout padding) |
| CUDA context beyond our buffers | 0.72 GB |
| peak host RAM | 0.37 GB |
| load, warm page cache | 2.3 s (6.93 GB/s) |
| load from cold disk | 9.2 s |

Copying is from pageable mmap memory. A pinned staging buffer and overlapping
H2D with the scale swizzle would speed up the warm case, but on a cold start the bottleneck is
the disk anyway.

The plan gives 17.08 GB for the whole load; this loader actually takes
16.25 GB: the MTP head (0.85 GB) doesn't go through it. 16.25 + 0.85 = 17.10 GB,
a 20 MB discrepancy with the plan.

## Outside the loader's scope

* MTP head: loaded by `qwc-engine::mtp` for the speculative path;
* the executor itself: pointer tables, CUDA graphs, token loop;
* perplexity and any comparisons with vLLM.

## Verification

```bash
cargo test -p qwc-engine
cargo test -p qwc-cuda --test vocab
cargo run --release -p qwc-engine --bin load
```

The loader tests need neither a GPU nor a checkpoint: they check that a missing
tensor is named, and a wrong shape is rejected before loading. The
`vocab` tests need a GPU: the per-row scale is checked against `absmax/448`, the reconstruction
error against the E4M3 bound, lookup and logits against a CPU reference on top of already
quantized data (this checks kernel arithmetic, not scale choice), and
splitting batch 12 into groups is checked separately.

The `load` binary additionally checks one loaded projection against the checkpoint
byte by byte and prints the FP8 error on real weights — without that the size report
would only confirm the number of bytes taken.

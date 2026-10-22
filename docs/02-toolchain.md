# Toolchain: RTX 5090 / SM120 / CUDA

## What is installed

```
GPU        NVIDIA GeForce RTX 5090, sm_120, 170 SM
driver     595.84 (CUDA runtime 13.2)
toolkit    CUDA 13.1.115 from the Ubuntu 26.04 repositories (/usr/local/cuda-13.1)
rustc      1.98.1
CUTLASS    4.8.0 (third_party/cutlass, not in git — see below)
```

## What you need to fetch yourself

Two directories in `third_party/` are not in git, and `qwc-cuda` won't build without them
(`crates/qwc-cuda/build.rs` requires both):

```bash
git clone --depth 1 --branch v4.8.0 https://github.com/NVIDIA/cutlass third_party/cutlass
```

```bash
./scripts/gen-cuda-shim.sh third_party/cuda-shim
```

The CUTLASS version matters: the NVFP4 atom `SM120_16x8x64_TN_VS` and examples `79b`,
`91`, `93`, `112` appeared in the 4.x branch. You can check that exactly
4.8.0 was fetched via `third_party/cutlass/include/cutlass/version.h`.

`nvcc` is not on `PATH` by default — you need to add it, otherwise both our build
and vLLM's FlashInfer silently fail:

```bash
export PATH=/usr/local/cuda/bin:$PATH
```

The checkpoint is fetched by a separate script (resuming is supported, it can be interrupted):

```bash
./scripts/fetch-checkpoint.sh
```

## Why the toolkit comes from the Ubuntu repositories

The toolkit is taken from the Ubuntu repositories rather than the NVIDIA site: version 13.1 is guaranteed
to be supported by driver 595.84 and doesn't drag in a driver replacement.
A minimal set is installed, without nsight and its dependencies:

```bash
sudo apt install -y cuda-nvcc-13-1 cuda-crt-13-1 cuda-cudart-dev-13-1 \
    cuda-cccl-13-1 cuda-nvrtc-dev-13-1 cuda-profiler-api-13-1 \
    cuda-nvtx-13-1 libcublas-dev-13-1
```

## Measured hardware limits

```
SM count                   170
SM clock                   2.47 GHz
memory                     512 bit @ 28.00 Gbps -> 1792 GB/s
L2 cache                   96 MiB          <- a lot, affects kernel design
shared mem / SM            100 KB
shared mem / block optin   99 KB           <- attention tile ceiling
registers / SM             65536
max threads / SM           1536
copy engines               2
```

### Consequence: head_dim = 256 and 99 KB of shared memory

Attention tile budget (Q in bf16, K and V together):

| Q tile | KV 32 | KV 64 | KV 128 |
|---|---|---|---|
| 16 | 40 KB ok | 72 KB ok | 136 KB **no** |
| 32 | 48 KB ok | 80 KB ok | 144 KB **no** |
| 64 | 64 KB ok | 96 KB ok | 160 KB **no** |
| 128 | 96 KB ok | 128 KB **no** | 192 KB **no** |

The maximum in bf16 is `64x64` or `128x32`.

**Non-obvious consequence:** quantizing the KV cache increases not only the available VRAM
but also the tile size. In fp8 a `128x64` tile takes 96 KB and fits, i.e. gives twice
the arithmetic intensity. That is an argument for fp8/nvfp4 KV in its own right,
beyond the memory savings.

## CUDA 13.1 + glibc 2.43 bug and workaround

Any `.cu` that includes `<cstdio>` or `cuda_runtime.h` fails to compile:

```
bits/mathcalls.h(206): error: exception specification is incompatible with
that of previous function "rsqrt" (declared at line 629 of crt/math_functions.h)
```

glibc >= 2.41 declares `rsqrt`/`rsqrtf` as `noexcept` (C23 IEC-60559, under `__USE_GNU`),
CUDA 13.1 declares them without `noexcept`. In C++17 the exception specification is part of the
function type, so this is a declaration conflict.

NVIDIA fixed this in CUDA 13.3 (`_NV_RSQRT_SPECIFIER`), but 13.3 is not yet in the Ubuntu
repositories.

What does **not** work:
* `-U_GNU_SOURCE` — fixes the conflict but breaks libstdc++ (`<mutex>` needs
  `clockid_t` and `pthread_mutex_clocklock`);
* `--diag-suppress` — the error can't be suppressed;
* `--pre-include` — doesn't reach the CUDA frontend, `cuda_runtime.h` is included earlier;
* disabling the glibc feature via `-D` — `libc-header-start.h` does `#undef` before `#define`.

What works: `scripts/gen-cuda-shim.sh`. It builds a shadow include tree of symlinks
to the real toolkit plus one patched copy of `crt/math_functions.h`. The trick
relies on CUDA including this file as `"crt/math_functions.h"` — in
quotes, i.e. relative to the including file's directory.

```bash
./scripts/gen-cuda-shim.sh third_party/cuda-shim
nvcc -arch=sm_120a -I third_party/cuda-shim ...
```

Remove as soon as CUDA >= 13.3 is available.

## NVFP4 on SM120: verified

SM120 uses SM80-style `mma.sync` with block scales, **not** the datacenter
Blackwell `tcgen05.mma`. Anything built on tcgen05 (DeepGEMM, some CUTLASS SM100 schemes)
won't build on the 5090.

CUTLASS 4.8.0 has the instruction we need:

```
mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4
atom: SM120_16x8x64_TN_VS   (include/cute/arch/mma_sm120.hpp)
```

Example `79b` (NVFP4 x NVFP4, i.e. our W4A4) builds in 31 seconds and passes
verification on a real 5090.

```bash
nvcc -arch=sm_120a -std=c++17 -O3 \
  -I third_party/cuda-shim -I third_party/cutlass/include \
  -I third_party/cutlass/tools/util/include -I third_party/cutlass/examples/common \
  --expt-relaxed-constexpr -diag-suppress 20012,20014 \
  -o gemm79b third_party/cutlass/examples/79_blackwell_geforce_gemm/79b_*.cu
```

### Useful CUTLASS examples for the project

| Example | Why |
|---|---|
| `79b_blackwell_geforce_nvfp4_nvfp4_gemm` | basic W4A4 GEMM, our main path |
| `91_fp4_gemv` | small-batch decode — GEMV, not GEMM |
| `93_blackwell_low_latency_gqa` | decode attention for GQA |
| `112_blackwell_ssd` | chunked state-space scan, close to Gated DeltaNet |
| `87_blackwell_geforce_gemm_blockwise` | block scales |

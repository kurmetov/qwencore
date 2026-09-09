# Тулчейн: RTX 5090 / SM120 / CUDA

## Что установлено

```
GPU        NVIDIA GeForce RTX 5090, sm_120, 170 SM
драйвер    595.84 (рантайм CUDA 13.2)
toolkit    CUDA 13.1.115 из репозиториев Ubuntu 26.04 (/usr/local/cuda-13.1)
rustc      1.98.1
CUTLASS    3.19 (third_party/cutlass, не в гите)
```

Toolkit взят из репозиториев Ubuntu, а не с сайта NVIDIA: версия 13.1 гарантированно
попадает в поддержку драйвера 595.84 и не тянет за собой замену драйвера.
Ставится минимальный набор, без nsight и его зависимостей:

```bash
sudo apt install -y cuda-nvcc-13-1 cuda-crt-13-1 cuda-cudart-dev-13-1 \
    cuda-cccl-13-1 cuda-nvrtc-dev-13-1 cuda-profiler-api-13-1 \
    cuda-nvtx-13-1 libcublas-dev-13-1
```

## Измеренные лимиты железа

```
SM count                   170
SM clock                   2.47 GHz
память                     512 bit @ 28.00 Gbps -> 1792 GB/s
L2 cache                   96 MiB          <- очень много, влияет на дизайн кернелов
shared mem / SM            100 KB
shared mem / block optin   99 KB           <- потолок тайла attention
registers / SM             65536
max threads / SM           1536
copy engines               2
```

### Следствие: head_dim = 256 и 99 KB shared memory

Бюджет тайла attention (Q в bf16, K и V вместе):

| Q-тайл | KV 32 | KV 64 | KV 128 |
|---|---|---|---|
| 16 | 40 KB ok | 72 KB ok | 136 KB **нет** |
| 32 | 48 KB ok | 80 KB ok | 144 KB **нет** |
| 64 | 64 KB ok | 96 KB ok | 160 KB **нет** |
| 128 | 96 KB ok | 128 KB **нет** | 192 KB **нет** |

Максимум в bf16 — `64x64` или `128x32`.

**Неочевидное следствие:** квантизация KV-кэша увеличивает не только доступную VRAM,
но и размер тайла. В fp8 тайл `128x64` занимает 96 KB и влезает, то есть даёт вдвое
большую арифметическую интенсивность. Это самостоятельный аргумент за fp8/nvfp4 KV,
помимо экономии памяти.

## Баг CUDA 13.1 + glibc 2.43 и обход

Любой `.cu`, включающий `<cstdio>` или `cuda_runtime.h`, не компилируется:

```
bits/mathcalls.h(206): error: exception specification is incompatible with
that of previous function "rsqrt" (declared at line 629 of crt/math_functions.h)
```

glibc >= 2.41 объявляет `rsqrt`/`rsqrtf` как `noexcept` (C23 IEC-60559, под `__USE_GNU`),
CUDA 13.1 объявляет их без `noexcept`. В C++17 спецификация исключений входит в тип
функции, поэтому это конфликт объявлений.

NVIDIA исправила это в CUDA 13.3 (`_NV_RSQRT_SPECIFIER`), но 13.3 в репозиториях
Ubuntu пока нет.

Что **не** работает:
* `-U_GNU_SOURCE` — чинит конфликт, но ломает libstdc++ (`<mutex>` требует
  `clockid_t` и `pthread_mutex_clocklock`);
* `--diag-suppress` — ошибка не подавляется;
* `--pre-include` — не доходит до CUDA-фронтенда, `cuda_runtime.h` включается раньше;
* отключение фичи glibc через `-D` — `libc-header-start.h` делает `#undef` перед `#define`.

Что работает: `scripts/gen-cuda-shim.sh`. Строит теневое дерево include из симлинков
на настоящий toolkit плюс одна пропатченная копия `crt/math_functions.h`. Приём
опирается на то, что CUDA подключает этот файл как `"crt/math_functions.h"` — в
кавычках, то есть относительно каталога включающего файла.

```bash
./scripts/gen-cuda-shim.sh third_party/cuda-shim
nvcc -arch=sm_120a -I third_party/cuda-shim ...
```

Убрать, как только появится CUDA >= 13.3.

## NVFP4 на SM120: проверено

SM120 использует SM80-style `mma.sync` с блочными шкалами, а **не** `tcgen05.mma`
датацентрового Blackwell. Всё, что построено на tcgen05 (DeepGEMM, часть SM100-схем
CUTLASS), на 5090 не соберётся.

Нужная инструкция в CUTLASS 3.19 есть:

```
mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4
атом: SM120_16x8x64_TN_VS   (include/cute/arch/mma_sm120.hpp)
```

Пример `79b` (NVFP4 x NVFP4, то есть наш W4A4) собирается за 31 секунду и проходит
верификацию на реальной 5090.

```bash
nvcc -arch=sm_120a -std=c++17 -O3 \
  -I third_party/cuda-shim -I third_party/cutlass/include \
  -I third_party/cutlass/tools/util/include -I third_party/cutlass/examples/common \
  --expt-relaxed-constexpr -diag-suppress 20012,20014 \
  -o gemm79b third_party/cutlass/examples/79_blackwell_geforce_gemm/79b_*.cu
```

### Полезные примеры CUTLASS для проекта

| Пример | Зачем |
|---|---|
| `79b_blackwell_geforce_nvfp4_nvfp4_gemm` | базовый W4A4 GEMM, наш основной путь |
| `91_fp4_gemv` | decode при малом batch — GEMV, а не GEMM |
| `93_blackwell_low_latency_gqa` | decode-attention для GQA |
| `112_blackwell_ssd` | чанковый скан state-space, близко к Gated DeltaNet |
| `87_blackwell_geforce_gemm_blockwise` | блочные шкалы |

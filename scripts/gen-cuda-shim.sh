#!/usr/bin/env bash
# Обход бага CUDA 13.1 + glibc >= 2.41.
#
# glibc объявляет rsqrt/rsqrtf как noexcept (C23 IEC-60559 под __USE_GNU),
# CUDA 13.1 объявляет их без noexcept. В C++17 спецификация исключений входит
# в тип функции, поэтому любой .cu, тянущий <cstdio> или cuda_runtime.h,
# не компилируется. NVIDIA исправила это в CUDA 13.3 (_NV_RSQRT_SPECIFIER).
#
# Здесь строится теневое дерево include: симлинки на настоящий toolkit плюс
# одна пропатченная копия crt/math_functions.h. Работает потому, что CUDA
# подключает его как "crt/math_functions.h" — в кавычках, то есть
# относительно каталога включающего файла, а он у нас свой.
#
# Убрать, как только на машине будет CUDA >= 13.3.
set -euo pipefail

CUDA_INC="${CUDA_INC:-$(dirname "$(readlink -f "$(command -v nvcc)")")/../targets/x86_64-linux/include}"
CUDA_INC="$(readlink -f "$CUDA_INC")"
SHIM="${1:?использование: gen-cuda-shim.sh <каталог>}"

rm -rf "$SHIM"
mkdir -p "$SHIM/crt"

for f in "$CUDA_INC"/*; do
    [ "$(basename "$f")" = crt ] && continue
    ln -sfn "$f" "$SHIM/$(basename "$f")"
done

for f in "$CUDA_INC"/crt/*; do
    ln -sfn "$f" "$SHIM/crt/$(basename "$f")"
done

rm -f "$SHIM/crt/math_functions.h"
sed -E 's/(double +rsqrt\(double x\));/\1 noexcept;/; s/(float +rsqrtf\(float x\));/\1 noexcept;/' \
    "$CUDA_INC/crt/math_functions.h" > "$SHIM/crt/math_functions.h"

if ! grep -q 'rsqrt(double x) noexcept;' "$SHIM/crt/math_functions.h"; then
    echo "gen-cuda-shim: патч не применился — заголовок изменился, проверьте sed" >&2
    exit 1
fi

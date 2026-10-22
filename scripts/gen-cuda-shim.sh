#!/usr/bin/env bash
# Workaround for the CUDA 13.1 + glibc >= 2.41 bug.
#
# glibc declares rsqrt/rsqrtf as noexcept (C23 IEC-60559 under __USE_GNU),
# CUDA 13.1 declares them without noexcept. In C++17 the exception specification is part
# of the function type, so any .cu pulling in <cstdio> or cuda_runtime.h
# fails to compile. NVIDIA fixed this in CUDA 13.3 (_NV_RSQRT_SPECIFIER).
#
# This builds a shadow include tree: symlinks to the real toolkit plus
# one patched copy of crt/math_functions.h. Works because CUDA
# includes it as "crt/math_functions.h" — in quotes, i.e.
# relative to the including file's directory, which here is ours.
#
# Remove as soon as the machine has CUDA >= 13.3.
set -euo pipefail

CUDA_INC="${CUDA_INC:-$(dirname "$(readlink -f "$(command -v nvcc)")")/../targets/x86_64-linux/include}"
CUDA_INC="$(readlink -f "$CUDA_INC")"
SHIM="${1:?usage: gen-cuda-shim.sh <dir>}"

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
    echo "gen-cuda-shim: patch didn't apply — the header changed, check the sed" >&2
    exit 1
fi

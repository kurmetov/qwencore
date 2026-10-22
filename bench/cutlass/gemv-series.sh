#!/usr/bin/env bash
# One series: our W4A16 (nvfp4bench, batch 1) vs CUTLASS 91_fp4_gemv
# on the same shapes. Rounds alternate ABBA so the order doesn't introduce a bias.
#
#   bench/cutlass/gemv-series.sh [rounds=7] > series.log
#
# The CUTLASS example is patched by bench/cutlass/91_fp4_gemv_bench.patch: measurement as in
# nvfp4bench (20 warmups, 7 series of 40 passes, median) and a
# --per_matrix mode, where 4 matrices go as four launches, as in the engine.
set -euo pipefail

root=$(cd "$(dirname "$0")/../.." && pwd)
rounds=${1:-7}
out=$root/target/cutlass
mkdir -p "$out"

src=$out/91_fp4_gemv_bench.cu
cp "$root/third_party/cutlass/examples/91_fp4_gemv/91_fp4_gemv.cu" "$src"
patch -s "$src" "$root/bench/cutlass/91_fp4_gemv_bench.patch"
nice -n 19 "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" -arch=sm_120a -std=c++17 -O3 \
  -I "$root/third_party/cuda-shim" -I "$root/third_party/cutlass/include" \
  -I "$root/third_party/cutlass/tools/util/include" \
  -I "$root/third_party/cutlass/examples/common" \
  --expt-relaxed-constexpr -diag-suppress 20012,20014 \
  -o "$out/gemv91_bench" "$src"
cargo build --release -q -p qwc-cuda --bin nvfp4bench --manifest-path "$root/Cargo.toml"

# Once with verification against the host reference, then without it (a minute per shape).
"$out/gemv91_bench" --m=17408 --k=5120 --batch=4 --profiling=false | grep "test pass" >/dev/null

run_qwc() {
  "$root/target/release/nvfp4bench" \
    | grep -E '^\s+(gate/up|down|q\+gate) \|\s+1 \|' | sed "s/^/QWC r=$1 /"
}
run_cutlass() {
  for shape in "17408 5120" "5120 17408" "12288 5120"; do
    set -- $shape
    for per_matrix in 1 0; do
      "$out/gemv91_bench" --m="$1" --k="$2" --batch=4 --iterations=40 \
        --verify=false --per_matrix=$per_matrix | grep RESULT | sed "s/^/CUT r=$round /"
    done
  done
}

for round in $(seq 1 "$rounds"); do
  echo "LOAD r=$round $(cut -d' ' -f1 /proc/loadavg)" \
    "$(nvidia-smi --query-gpu=utilization.gpu,clocks.sm,temperature.gpu --format=csv,noheader)"
  if (( round % 2 )); then run_qwc "$round"; run_cutlass; else run_cutlass; run_qwc "$round"; fi
done

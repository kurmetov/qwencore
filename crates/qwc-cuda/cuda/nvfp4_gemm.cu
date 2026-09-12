// SM120 block-scaled NVFP4 GEMM для decode batch 3..32.
//
// A: [M,K] row-major packed E2M1, B: [K,N] column-major packed E2M1.
// Физический B совпадает с checkpoint [N,K] row-major, поэтому 4-битные
// веса не репакуются. SFA/SFB обязаны быть в CUTLASS 128x4 layout.

#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace qwc::nvfp4_gemm {

using namespace cute;

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using LayoutATag = cutlass::layout::RowMajor;
using LayoutBTag = cutlass::layout::ColumnMajor;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;
constexpr int AlignmentB = 32;
constexpr int AlignmentC = 8;
constexpr int AlignmentD = 8;

using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using ElementAccumulator = float;
// K=256 amortizes scheduler/TMA setup. Ping-pong won the direct target-GPU
// comparison against both the stock K=128 cooperative tile and K=256 cooperative.
using ThreadBlockShape = Shape<_128, _128, _256>;
using ClusterShape = Shape<_1, _1, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag,
    OperatorClass,
    ThreadBlockShape,
    ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator,
    ElementAccumulator,
    ElementC,
    LayoutCTag,
    AlignmentC,
    ElementD,
    LayoutDTag,
    AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,
    OperatorClass,
    ElementA,
    LayoutATag,
    AlignmentA,
    ElementB,
    LayoutBTag,
    AlignmentB,
    ElementAccumulator,
    ThreadBlockShape,
    ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideA = typename GemmKernel::StrideA;
using StrideB = typename GemmKernel::StrideB;
using StrideC = typename GemmKernel::StrideC;
using StrideD = typename GemmKernel::StrideD;
using ScaleConfig = typename CollectiveMainloop::Sm1xxBlkScaledConfig;

static typename Gemm::Arguments make_arguments(
    const void* packed_a,
    const void* packed_b,
    const void* scales_a,
    const void* scales_b,
    void* output,
    int m,
    int n,
    int k,
    float alpha) {
  auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  auto stride_b = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  auto stride_c = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
  auto stride_d = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});
  auto shape = cute::make_shape(m, n, k, 1);
  auto layout_sfa = ScaleConfig::tile_atom_to_shape_SFA(shape);
  auto layout_sfb = ScaleConfig::tile_atom_to_shape_SFB(shape);

  using DataA = typename ElementA::DataType;
  using DataB = typename ElementB::DataType;
  using ScaleA = typename ElementA::ScaleFactorType;
  using ScaleB = typename ElementB::ScaleFactorType;

  return typename Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      shape,
      {reinterpret_cast<DataA const*>(packed_a),
       stride_a,
       reinterpret_cast<DataB const*>(packed_b),
       stride_b,
       reinterpret_cast<ScaleA const*>(scales_a),
       layout_sfa,
       reinterpret_cast<ScaleB const*>(scales_b),
       layout_sfb},
      {{alpha, 0.0f},
       reinterpret_cast<ElementC const*>(output),
       stride_c,
       reinterpret_cast<ElementD*>(output),
       stride_d}};
}

static cudaError_t cutlass_error(cutlass::Status status) {
  return status == cutlass::Status::kSuccess ? cudaSuccess : cudaErrorInvalidValue;
}

}  // namespace qwc::nvfp4_gemm

extern "C" cudaError_t qwc_nvfp4_w4a4_workspace_size(
    int m, int n, int k, size_t* bytes) {
  if (bytes == nullptr || m <= 0 || n <= 0 || k <= 0) {
    return cudaErrorInvalidValue;
  }
  auto args = qwc::nvfp4_gemm::make_arguments(
      nullptr, nullptr, nullptr, nullptr, nullptr, m, n, k, 1.0f);
  *bytes = qwc::nvfp4_gemm::Gemm::get_workspace_size(args);
  return cudaSuccess;
}

extern "C" cudaError_t qwc_nvfp4_w4a4(
    const void* packed_a,
    const void* packed_b,
    const void* scales_a,
    const void* scales_b,
    void* output,
    void* workspace,
    size_t workspace_bytes,
    int m,
    int n,
    int k,
    float alpha,
    cudaStream_t stream) {
  if (packed_a == nullptr || packed_b == nullptr || scales_a == nullptr ||
      scales_b == nullptr || output == nullptr || m <= 0 || n <= 0 || k <= 0) {
    return cudaErrorInvalidValue;
  }

  auto args = qwc::nvfp4_gemm::make_arguments(
      packed_a, packed_b, scales_a, scales_b, output, m, n, k, alpha);
  const size_t needed = qwc::nvfp4_gemm::Gemm::get_workspace_size(args);
  if (workspace_bytes < needed || (needed != 0 && workspace == nullptr)) {
    return cudaErrorInvalidValue;
  }

  qwc::nvfp4_gemm::Gemm gemm;
  auto status = gemm.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return qwc::nvfp4_gemm::cutlass_error(status);
  }
  status = gemm.initialize(args, workspace, stream);
  if (status != cutlass::Status::kSuccess) {
    return qwc::nvfp4_gemm::cutlass_error(status);
  }
  status = gemm.run(stream);
  if (status != cutlass::Status::kSuccess) {
    return qwc::nvfp4_gemm::cutlass_error(status);
  }
  return cudaGetLastError();
}

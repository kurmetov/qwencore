// SM120 block-scaled NVFP4 GEMM.
//
// A: [M,K] row-major packed E2M1, B: [K,N] column-major packed E2M1.
// Физический B совпадает с checkpoint [N,K] row-major, поэтому 4-битные
// веса не репакуются. SFA/SFB обязаны быть в CUTLASS 128x4 layout.
//
// Две инстанции, выбор по форме задачи. Тайл 128x128 нарезает выход на
// ceil(M/128) * ceil(N/128) блоков, и на узких по N проекциях их меньше, чем
// SM: down_proj [5120, 17408] при M<=128 даёт 40 блоков на 170 SM и стоит
// намертво на 780 GB/s при любом batch, тогда как gate_proj той же массы
// весов берёт 1335. Когда блоков меньше, чем SM, задача режется по K
// stream-K-планировщиком; ping-pong его не поддерживает, поэтому вторая
// инстанция идёт на cooperative.

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
constexpr int kTileM = 128;
constexpr int kTileN = 128;

template <class ScheduleTag, class TileSchedulerTag>
struct Config {
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
      ScheduleTag>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
      TileSchedulerTag>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;
  using StrideD = typename GemmKernel::StrideD;
  using ScaleConfig = typename CollectiveMainloop::Sm1xxBlkScaledConfig;
};

using Wide = Config<cutlass::gemm::KernelTmaWarpSpecializedPingpong, void>;
using Narrow = Config<cutlass::gemm::KernelTmaWarpSpecializedCooperative,
                      cutlass::gemm::StreamKScheduler>;

template <class C>
static typename C::Gemm::Arguments make_arguments(
    const void* packed_a,
    const void* packed_b,
    const void* scales_a,
    const void* scales_b,
    void* output,
    int m,
    int n,
    int k,
    float alpha) {
  auto stride_a = cutlass::make_cute_packed_stride(typename C::StrideA{}, {m, k, 1});
  auto stride_b = cutlass::make_cute_packed_stride(typename C::StrideB{}, {n, k, 1});
  auto stride_c = cutlass::make_cute_packed_stride(typename C::StrideC{}, {m, n, 1});
  auto stride_d = cutlass::make_cute_packed_stride(typename C::StrideD{}, {m, n, 1});
  auto shape = cute::make_shape(m, n, k, 1);
  auto layout_sfa = C::ScaleConfig::tile_atom_to_shape_SFA(shape);
  auto layout_sfb = C::ScaleConfig::tile_atom_to_shape_SFB(shape);

  using DataA = typename ElementA::DataType;
  using DataB = typename ElementB::DataType;
  using ScaleA = typename ElementA::ScaleFactorType;
  using ScaleB = typename ElementB::ScaleFactorType;

  return typename C::Gemm::Arguments{
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

static int sm_count() {
  static int cached = 0;
  if (cached == 0) {
    cudaDeviceGetAttribute(&cached, cudaDevAttrMultiProcessorCount, 0);
  }
  return cached;
}

// Узкой задача считается тогда, когда тайлов выхода вдвое меньше, чем SM.
// Порог именно с запасом: ровно на границе stream-K уже проигрывает ping-pong
// (замер down при M=512 — 78.3 против 76.4 us), потому что редукция стоит
// дороже, чем добавленная занятость.
static bool is_narrow(int m, int n) {
  const int tiles = ((m + kTileM - 1) / kTileM) * ((n + kTileN - 1) / kTileN);
  return tiles * 2 <= sm_count();
}

template <class C>
static cudaError_t run(
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
  auto args = make_arguments<C>(
      packed_a, packed_b, scales_a, scales_b, output, m, n, k, alpha);
  const size_t needed = C::Gemm::get_workspace_size(args);
  if (workspace_bytes < needed || (needed != 0 && workspace == nullptr)) {
    return cudaErrorInvalidValue;
  }
  typename C::Gemm gemm;
  auto status = gemm.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return cutlass_error(status);
  }
  status = gemm.initialize(args, workspace, stream);
  if (status != cutlass::Status::kSuccess) {
    return cutlass_error(status);
  }
  status = gemm.run(stream);
  if (status != cutlass::Status::kSuccess) {
    return cutlass_error(status);
  }
  return cudaGetLastError();
}

}  // namespace qwc::nvfp4_gemm

extern "C" cudaError_t qwc_nvfp4_w4a4_workspace_size(
    int m, int n, int k, size_t* bytes) {
  if (bytes == nullptr || m <= 0 || n <= 0 || k <= 0) {
    return cudaErrorInvalidValue;
  }
  // Максимум по обеим инстанциям: буфер выделяется один раз на форму, а
  // выбор между ними делается на каждом запуске.
  using Wide = qwc::nvfp4_gemm::Wide;
  using Narrow = qwc::nvfp4_gemm::Narrow;
  const size_t wide = Wide::Gemm::get_workspace_size(
      qwc::nvfp4_gemm::make_arguments<Wide>(
          nullptr, nullptr, nullptr, nullptr, nullptr, m, n, k, 1.0f));
  const size_t narrow = Narrow::Gemm::get_workspace_size(
      qwc::nvfp4_gemm::make_arguments<Narrow>(
          nullptr, nullptr, nullptr, nullptr, nullptr, m, n, k, 1.0f));
  *bytes = wide > narrow ? wide : narrow;
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

  if (qwc::nvfp4_gemm::is_narrow(m, n)) {
    return qwc::nvfp4_gemm::run<qwc::nvfp4_gemm::Narrow>(
        packed_a, packed_b, scales_a, scales_b, output, workspace,
        workspace_bytes, m, n, k, alpha, stream);
  }
  return qwc::nvfp4_gemm::run<qwc::nvfp4_gemm::Wide>(
      packed_a, packed_b, scales_a, scales_b, output, workspace,
      workspace_bytes, m, n, k, alpha, stream);
}

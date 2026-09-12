// Decode preprocessing for Qwen3.5 Gated DeltaNet.
//
// The first kernel updates the scheduler-selected depthwise-convolution state,
// applies SiLU, and splits q/k/v. The second performs FP32 L2 normalization
// and produces the recurrent decay/write gates.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int kKHeads = 16;
constexpr int kVHeads = 48;
constexpr int kHeadDim = 128;
constexpr int kQkElements = kKHeads * kHeadDim;
constexpr int kVElements = kVHeads * kHeadDim;
constexpr int kChannels = 2 * kQkElements + kVElements;
constexpr int kConvHistory = 3;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ float sigmoid(float value) {
  return 1.0f / (1.0f + __expf(-value));
}

__device__ __forceinline__ float softplus(float value) {
  if (value > 20.0f) {
    return value;
  }
  if (value < -20.0f) {
    return __expf(value);
  }
  return log1pf(__expf(value));
}

__global__ void causal_conv_split_kernel(
    const __nv_bfloat16* __restrict__ mixed_qkv,
    const __nv_bfloat16* __restrict__ conv_weight,
    float* __restrict__ conv_state,
    const uint32_t* __restrict__ state_slots,
    float* __restrict__ query,
    float* __restrict__ key,
    float* __restrict__ value,
    int batch,
    int row_stride) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch * kChannels;
  if (index >= total) {
    return;
  }
  const int sequence = index / kChannels;
  const int channel = index % kChannels;
  const uint32_t slot = state_slots[sequence];
  float* history =
      conv_state + (static_cast<size_t>(slot) * kChannels + channel) * kConvHistory;
  const float current =
      __bfloat162float(mixed_qkv[static_cast<size_t>(sequence) * row_stride + channel]);
  const __nv_bfloat16* weight = conv_weight + channel * 4;
  float convolved = __bfloat162float(weight[0]) * history[0];
  convolved = fmaf(__bfloat162float(weight[1]), history[1], convolved);
  convolved = fmaf(__bfloat162float(weight[2]), history[2], convolved);
  convolved = fmaf(__bfloat162float(weight[3]), current, convolved);
  history[0] = history[1];
  history[1] = history[2];
  history[2] = current;

  // causal_conv1d_update returns the activation in the projection dtype.
  const float activated = __bfloat162float(
      __float2bfloat16(convolved * sigmoid(convolved)));
  const int row_offset = sequence * kQkElements;
  if (channel < kQkElements) {
    query[row_offset + channel] = activated;
  } else if (channel < 2 * kQkElements) {
    key[row_offset + channel - kQkElements] = activated;
  } else {
    value[sequence * kVElements + channel - 2 * kQkElements] = activated;
  }
}

// Свёртка на четыре отвода по токенам параллельна: последовательна только
// переноска истории между чанками. Прежняя раскладка отдавала потоку канал
// целиком и шла по токенам в программном порядке — это 40 блоков на 170 SM,
// то есть меньше четверти чипа. Здесь поток считает один выход (канал, токен),
// а хвост истории дописывает отдельный запуск: писать её здесь значило бы
// гонку с блоками, которые ту же историю читают на первых трёх токенах.
__device__ __forceinline__ float conv_input(
    const __nv_bfloat16* __restrict__ mixed_qkv,
    const float* __restrict__ history,
    int token,
    int channel,
    int row_stride) {
  if (token >= 0) {
    return __bfloat162float(mixed_qkv[static_cast<size_t>(token) * row_stride + channel]);
  }
  return history[kConvHistory + token];
}

__global__ void causal_conv_prefill_kernel(
    const __nv_bfloat16* __restrict__ mixed_qkv,
    const __nv_bfloat16* __restrict__ conv_weight,
    const float* __restrict__ conv_state,
    float* __restrict__ query,
    float* __restrict__ key,
    float* __restrict__ value,
    int state_slot,
    int tokens,
    int row_stride) {
  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  if (channel >= kChannels) {
    return;
  }
  const int token = blockIdx.y;
  const float* history =
      conv_state + (static_cast<size_t>(state_slot) * kChannels + channel) * kConvHistory;
  const __nv_bfloat16* weight = conv_weight + channel * 4;

  float convolved =
      __bfloat162float(weight[0]) * conv_input(mixed_qkv, history, token - 3, channel, row_stride);
  convolved = fmaf(__bfloat162float(weight[1]),
                   conv_input(mixed_qkv, history, token - 2, channel, row_stride), convolved);
  convolved = fmaf(__bfloat162float(weight[2]),
                   conv_input(mixed_qkv, history, token - 1, channel, row_stride), convolved);
  convolved = fmaf(__bfloat162float(weight[3]),
                   conv_input(mixed_qkv, history, token, channel, row_stride), convolved);

  const float activated =
      __bfloat162float(__float2bfloat16(convolved * sigmoid(convolved)));
  if (channel < kQkElements) {
    query[static_cast<size_t>(token) * kQkElements + channel] = activated;
  } else if (channel < 2 * kQkElements) {
    key[static_cast<size_t>(token) * kQkElements + channel - kQkElements] = activated;
  } else {
    value[static_cast<size_t>(token) * kVElements + channel - 2 * kQkElements] = activated;
  }
}

// Хвост чанка становится историей следующего. Поток владеет тремя значениями
// своего канала и читает их до записи, поэтому чанк короче трёх токенов
// доносит остаток прежней истории без гонки.
__global__ void conv_state_prefill_kernel(
    const __nv_bfloat16* __restrict__ mixed_qkv,
    float* __restrict__ conv_state,
    int state_slot,
    int tokens,
    int row_stride) {
  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  if (channel >= kChannels) {
    return;
  }
  float* history =
      conv_state + (static_cast<size_t>(state_slot) * kChannels + channel) * kConvHistory;
  const float previous[kConvHistory] = {history[0], history[1], history[2]};
  #pragma unroll
  for (int i = 0; i < kConvHistory; ++i) {
    history[i] =
        conv_input(mixed_qkv, previous, tokens - kConvHistory + i, channel, row_stride);
  }
}

__global__ __launch_bounds__(kHeadDim) void normalize_and_gate_kernel(
    float* __restrict__ query,
    float* __restrict__ key,
    const __nv_bfloat16* __restrict__ a_projection,
    const __nv_bfloat16* __restrict__ b_projection,
    const __nv_bfloat16* __restrict__ a_log,
    const __nv_bfloat16* __restrict__ dt_bias,
    float* __restrict__ alpha,
    float* __restrict__ beta,
    int gate_stride) {
  const int head = blockIdx.x;
  const int batch = blockIdx.y;
  const int dimension = threadIdx.x;
  const size_t base =
      (static_cast<size_t>(batch) * kKHeads + head) * kHeadDim;
  const float q = query[base + dimension];
  const float k = key[base + dimension];
  float q_sum = warp_sum(q * q);
  float k_sum = warp_sum(k * k);

  __shared__ float warp_q[4];
  __shared__ float warp_k[4];
  __shared__ float q_inverse;
  __shared__ float k_inverse;
  const int lane = dimension & 31;
  const int warp = dimension >> 5;
  if (lane == 0) {
    warp_q[warp] = q_sum;
    warp_k[warp] = k_sum;
  }
  __syncthreads();
  if (warp == 0) {
    q_sum = lane < 4 ? warp_q[lane] : 0.0f;
    k_sum = lane < 4 ? warp_k[lane] : 0.0f;
    q_sum = warp_sum(q_sum);
    k_sum = warp_sum(k_sum);
    if (lane == 0) {
      q_inverse = rsqrtf(q_sum + 1e-6f) * (1.0f / sqrtf(128.0f));
      k_inverse = rsqrtf(k_sum + 1e-6f);
    }
  }
  __syncthreads();
  query[base + dimension] = q * q_inverse;
  key[base + dimension] = k * k_inverse;

  if (dimension < 3) {
    const int value_head = head * 3 + dimension;
    // Гейты читаются из арены с её шагом, а alpha/beta пишутся плотно.
    const size_t gate_read = static_cast<size_t>(batch) * gate_stride + value_head;
    const size_t gate_offset = static_cast<size_t>(batch) * kVHeads + value_head;
    const float projected_a = __bfloat162float(a_projection[gate_read]);
    const float projected_b = __bfloat162float(b_projection[gate_read]);
    const float log_a = __bfloat162float(a_log[value_head]);
    const float bias = __bfloat162float(dt_bias[value_head]);
    const float log_decay = -__expf(log_a) * softplus(projected_a + bias);
    alpha[gate_offset] = __expf(log_decay);
    beta[gate_offset] = sigmoid(projected_b);
  }
}

}  // namespace

extern "C" cudaError_t qwc_delta_prepare_decode(
    const void* mixed_qkv,
    const void* a_projection,
    const void* b_projection,
    const void* conv_weight,
    const void* a_log,
    const void* dt_bias,
    void* conv_state,
    const void* state_slots,
    void* query,
    void* key,
    void* value,
    void* alpha,
    void* beta,
    int state_capacity,
    int batch,
    int mixed_stride,
    int gate_stride,
    cudaStream_t stream) {
  if (mixed_qkv == nullptr || a_projection == nullptr || b_projection == nullptr ||
      conv_weight == nullptr || a_log == nullptr || dt_bias == nullptr ||
      conv_state == nullptr || state_slots == nullptr || query == nullptr ||
      key == nullptr || value == nullptr || alpha == nullptr || beta == nullptr ||
      state_capacity < batch || batch <= 0 || batch > 128 ||
      mixed_stride < kChannels || gate_stride < kVHeads) {
    return cudaErrorInvalidValue;
  }
  constexpr int threads = 256;
  const int blocks = (batch * kChannels + threads - 1) / threads;
  causal_conv_split_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(mixed_qkv),
      static_cast<const __nv_bfloat16*>(conv_weight),
      static_cast<float*>(conv_state),
      static_cast<const uint32_t*>(state_slots),
      static_cast<float*>(query),
      static_cast<float*>(key),
      static_cast<float*>(value),
      batch,
      mixed_stride);
  cudaError_t launched = cudaGetLastError();
  if (launched != cudaSuccess) {
    return launched;
  }
  dim3 grid(kKHeads, batch);
  normalize_and_gate_kernel<<<grid, kHeadDim, 0, stream>>>(
      static_cast<float*>(query),
      static_cast<float*>(key),
      static_cast<const __nv_bfloat16*>(a_projection),
      static_cast<const __nv_bfloat16*>(b_projection),
      static_cast<const __nv_bfloat16*>(a_log),
      static_cast<const __nv_bfloat16*>(dt_bias),
      static_cast<float*>(alpha),
      static_cast<float*>(beta),
      gate_stride);
  return cudaGetLastError();
}

extern "C" cudaError_t qwc_delta_prepare_prefill(
    const void* mixed_qkv,
    const void* a_projection,
    const void* b_projection,
    const void* conv_weight,
    const void* a_log,
    const void* dt_bias,
    void* conv_state,
    void* query,
    void* key,
    void* value,
    void* alpha,
    void* beta,
    int state_capacity,
    int state_slot,
    int tokens,
    int row_offset,
    int mixed_stride,
    int gate_stride,
    cudaStream_t stream) {
  if (mixed_qkv == nullptr || a_projection == nullptr || b_projection == nullptr ||
      conv_weight == nullptr || a_log == nullptr || dt_bias == nullptr ||
      conv_state == nullptr || query == nullptr || key == nullptr || value == nullptr ||
      alpha == nullptr || beta == nullptr || state_capacity <= 0 || state_slot < 0 ||
      state_slot >= state_capacity || tokens <= 0 || tokens > 1024 ||
      row_offset < 0 || mixed_stride < kChannels || gate_stride < kVHeads) {
    return cudaErrorInvalidValue;
  }
  // One slice of a fused multi-sequence token arena; see qwc_delta_prefill.
  const __nv_bfloat16* mixed_qkv_rows =
      static_cast<const __nv_bfloat16*>(mixed_qkv) +
      (size_t)row_offset * mixed_stride;
  const __nv_bfloat16* a_rows =
      static_cast<const __nv_bfloat16*>(a_projection) + (size_t)row_offset * gate_stride;
  const __nv_bfloat16* b_rows =
      static_cast<const __nv_bfloat16*>(b_projection) + (size_t)row_offset * gate_stride;
  float* query_rows = static_cast<float*>(query) + (size_t)row_offset * kQkElements;
  float* key_rows = static_cast<float*>(key) + (size_t)row_offset * kQkElements;
  float* value_rows = static_cast<float*>(value) + (size_t)row_offset * kVElements;
  float* alpha_rows = static_cast<float*>(alpha) + (size_t)row_offset * kVHeads;
  float* beta_rows = static_cast<float*>(beta) + (size_t)row_offset * kVHeads;
  constexpr int threads = 256;
  const int blocks = (kChannels + threads - 1) / threads;
  dim3 conv_grid(blocks, tokens);
  causal_conv_prefill_kernel<<<conv_grid, threads, 0, stream>>>(
      mixed_qkv_rows,
      static_cast<const __nv_bfloat16*>(conv_weight),
      static_cast<const float*>(conv_state),
      query_rows,
      key_rows,
      value_rows,
      state_slot,
      tokens,
      mixed_stride);
  cudaError_t launched = cudaGetLastError();
  if (launched != cudaSuccess) {
    return launched;
  }
  conv_state_prefill_kernel<<<blocks, threads, 0, stream>>>(
      mixed_qkv_rows, static_cast<float*>(conv_state), state_slot, tokens,
      mixed_stride);
  launched = cudaGetLastError();
  if (launched != cudaSuccess) {
    return launched;
  }
  dim3 grid(kKHeads, tokens);
  normalize_and_gate_kernel<<<grid, kHeadDim, 0, stream>>>(
      query_rows,
      key_rows,
      a_rows,
      b_rows,
      static_cast<const __nv_bfloat16*>(a_log),
      static_cast<const __nv_bfloat16*>(dt_bias),
      alpha_rows,
      beta_rows,
      gate_stride);
  return cudaGetLastError();
}

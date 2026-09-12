//! Gated DeltaNet: рекуррентный шаг decode.
//!
//! Реализует 48 из 64 слоёв Qwen3.8. Подробности математики и раскладки —
//! в `cuda/delta_net.cu`.

use crate::error::{Result, check};
use crate::{DeviceBuffer, Stream, ffi};
use qwc_core::arch::{LA_K_HEAD_DIM, LA_NUM_K_HEADS, LA_NUM_V_HEADS, LA_V_HEAD_DIM};

/// Элементов состояния на одну последовательность в одном слое.
pub const STATE_ELEMS: usize = LA_NUM_V_HEADS * LA_V_HEAD_DIM * LA_K_HEAD_DIM;
/// Элементов q или k на последовательность.
pub const QK_ELEMS: usize = LA_NUM_K_HEADS * LA_K_HEAD_DIM;
/// Элементов v или выхода на последовательность.
pub const V_ELEMS: usize = LA_NUM_V_HEADS * LA_V_HEAD_DIM;
/// Гейтов (alpha, beta) на последовательность.
pub const GATE_ELEMS: usize = LA_NUM_V_HEADS;
/// FP32 causal-convolution history per persistent sequence slot and layer.
pub const CONV_STATE_ELEMS: usize = qwc_core::arch::LA_CONV_CHANNELS * 3;

/// Точность рекуррентного состояния внутри одного chunk prefill.
///
/// `Bf16` округляет состояние до BF16 после каждого токена, поэтому chunk
/// воспроизводит повторный decode бит в бит. `Fp32` держит строку состояния в
/// FP32-регистрах до конца chunk и округляет только запись на границе — так
/// делает эталонная реализация (flash-linear-attention копит `b_h` в FP32
/// через все чанки последовательности). Диагностический A/B: изолирует
/// потокенное округление как источник расхождения.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeltaStateMode {
    Bf16,
    Fp32,
}

impl DeltaStateMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Fp32 => "fp32",
        }
    }

    fn round_per_token(self) -> i32 {
        match self {
            Self::Bf16 => 1,
            Self::Fp32 => 0,
        }
    }
}

pub struct PreparedDelta {
    pub q: DeviceBuffer<f32>,
    pub k: DeviceBuffer<f32>,
    pub v: DeviceBuffer<f32>,
    pub alpha: DeviceBuffer<f32>,
    pub beta: DeviceBuffer<f32>,
    batch: usize,
}

impl PreparedDelta {
    pub fn zeroed(batch: usize) -> Result<Self> {
        assert!((1..=1024).contains(&batch));
        Ok(Self {
            q: DeviceBuffer::zeroed(batch * QK_ELEMS)?,
            k: DeviceBuffer::zeroed(batch * QK_ELEMS)?,
            v: DeviceBuffer::zeroed(batch * V_ELEMS)?,
            alpha: DeviceBuffer::zeroed(batch * GATE_ELEMS)?,
            beta: DeviceBuffer::zeroed(batch * GATE_ELEMS)?,
            batch,
        })
    }

    pub fn inputs(&self) -> DeltaInputs<'_> {
        DeltaInputs {
            q: &self.q,
            k: &self.k,
            v: &self.v,
            alpha: &self.alpha,
            beta: &self.beta,
        }
    }
}

pub struct DeltaPreprocessor {
    conv_weight: DeviceBuffer<u16>,
    a_log: DeviceBuffer<u16>,
    dt_bias: DeviceBuffer<u16>,
}

impl DeltaPreprocessor {
    pub fn from_host(conv_weight: &[u16], a_log: &[u16], dt_bias: &[u16]) -> Result<Self> {
        assert_eq!(conv_weight.len(), qwc_core::arch::LA_CONV_CHANNELS * 4);
        assert_eq!(a_log.len(), GATE_ELEMS);
        assert_eq!(dt_bias.len(), GATE_ELEMS);
        Ok(Self {
            conv_weight: DeviceBuffer::from_slice(conv_weight)?,
            a_log: DeviceBuffer::from_slice(a_log)?,
            dt_bias: DeviceBuffer::from_slice(dt_bias)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prepare_decode(
        &self,
        mixed_qkv: RowView<'_>,
        a_projection: RowView<'_>,
        b_projection: RowView<'_>,
        conv_state_pool: &mut DeviceBuffer<f32>,
        state_slots: &DeviceBuffer<u32>,
        output: &mut PreparedDelta,
        state_capacity: usize,
        batch: usize,
        stream: &Stream,
    ) -> Result<()> {
        assert!(batch <= output.batch);
        assert!(mixed_qkv.fits(batch, qwc_core::arch::LA_CONV_CHANNELS));
        assert!(a_projection.fits(batch, GATE_ELEMS));
        assert!(b_projection.fits(batch, GATE_ELEMS));
        assert_eq!(a_projection.stride(), b_projection.stride());
        assert_eq!(conv_state_pool.len(), state_capacity * CONV_STATE_ELEMS);
        assert!(state_slots.len() >= batch);
        check(unsafe {
            ffi::qwc_delta_prepare_decode(
                mixed_qkv.as_ptr(),
                a_projection.as_ptr(),
                b_projection.as_ptr(),
                self.conv_weight.as_ptr(),
                self.a_log.as_ptr(),
                self.dt_bias.as_ptr(),
                conv_state_pool.as_mut_ptr(),
                state_slots.as_ptr(),
                output.q.as_mut_ptr(),
                output.k.as_mut_ptr(),
                output.v.as_mut_ptr(),
                output.alpha.as_mut_ptr(),
                output.beta.as_mut_ptr(),
                state_capacity as i32,
                batch as i32,
                mixed_qkv.stride() as i32,
                a_projection.stride() as i32,
                stream.raw(),
            )
        })
    }

    /// Prepares a causal chunk belonging to one persistent sequence slot.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_prefill(
        &self,
        mixed_qkv: RowView<'_>,
        a_projection: RowView<'_>,
        b_projection: RowView<'_>,
        conv_state_pool: &mut DeviceBuffer<f32>,
        output: &mut PreparedDelta,
        state_capacity: usize,
        state_slot: usize,
        tokens: usize,
        row_offset: usize,
        stream: &Stream,
    ) -> Result<()> {
        assert!(tokens > 0 && row_offset + tokens <= output.batch);
        assert!(state_slot < state_capacity);
        assert!(mixed_qkv.fits(output.batch, qwc_core::arch::LA_CONV_CHANNELS));
        assert!(a_projection.fits(output.batch, GATE_ELEMS));
        assert!(b_projection.fits(output.batch, GATE_ELEMS));
        assert_eq!(a_projection.stride(), b_projection.stride());
        assert_eq!(conv_state_pool.len(), state_capacity * CONV_STATE_ELEMS);
        check(unsafe {
            ffi::qwc_delta_prepare_prefill(
                mixed_qkv.as_ptr(),
                a_projection.as_ptr(),
                b_projection.as_ptr(),
                self.conv_weight.as_ptr(),
                self.a_log.as_ptr(),
                self.dt_bias.as_ptr(),
                conv_state_pool.as_mut_ptr(),
                output.q.as_mut_ptr(),
                output.k.as_mut_ptr(),
                output.v.as_mut_ptr(),
                output.alpha.as_mut_ptr(),
                output.beta.as_mut_ptr(),
                state_capacity as i32,
                state_slot as i32,
                tokens as i32,
                row_offset as i32,
                mixed_qkv.stride() as i32,
                a_projection.stride() as i32,
                stream.raw(),
            )
        })
    }
}

/// Срез арены проекций: строки идут с шагом `stride`, начиная с `offset`.
///
/// Слитая проекция миксера кладёт qkv, z, a и b в одну матрицу, поэтому
/// потребители получают не отдельный буфер, а вид на её столбцы. `packed`
/// описывает прежний случай — отдельный плотный буфер.
#[derive(Clone, Copy)]
pub struct RowView<'a> {
    buffer: &'a DeviceBuffer<u16>,
    offset: usize,
    stride: usize,
}

impl<'a> RowView<'a> {
    pub fn packed(buffer: &'a DeviceBuffer<u16>, width: usize) -> Self {
        Self {
            buffer,
            offset: 0,
            stride: width,
        }
    }

    pub fn strided(buffer: &'a DeviceBuffer<u16>, offset: usize, stride: usize) -> Self {
        assert!(offset < stride, "срез начинается за пределами строки");
        Self {
            buffer,
            offset,
            stride,
        }
    }

    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Помещаются ли `rows` строк шириной `width`.
    pub fn fits(&self, rows: usize, width: usize) -> bool {
        assert!(self.offset + width <= self.stride, "срез шире строки");
        rows == 0 || self.offset + (rows - 1) * self.stride + width <= self.buffer.len()
    }

    fn as_ptr(&self) -> *const std::ffi::c_void {
        // SAFETY: смещение внутри буфера проверяется `fits` у вызывающего.
        unsafe { (self.buffer.as_ptr() as *const u16).add(self.offset) as *const _ }
    }
}

pub struct DeltaOutputNorm {
    weight: DeviceBuffer<u16>,
    epsilon: f32,
}

impl DeltaOutputNorm {
    /// Unlike Qwen3.5's other norms, RMSNormGated has a direct (one-centered)
    /// weight and must not add one.
    pub fn from_host(weight: &[u16], epsilon: f32) -> Result<Self> {
        assert_eq!(weight.len(), LA_V_HEAD_DIM);
        assert!(epsilon.is_finite() && epsilon > 0.0);
        Ok(Self {
            weight: DeviceBuffer::from_slice(weight)?,
            epsilon,
        })
    }

    pub fn forward(
        &self,
        input: &DeviceBuffer<f32>,
        gate: RowView<'_>,
        output: &mut DeviceBuffer<u16>,
        batch: usize,
        stream: &Stream,
    ) -> Result<()> {
        assert!(input.len() >= batch * V_ELEMS);
        assert!(gate.fits(batch, V_ELEMS));
        assert!(output.len() >= batch * V_ELEMS);
        check(unsafe {
            ffi::qwc_delta_gated_rmsnorm(
                input.as_ptr().cast(),
                gate.as_ptr(),
                self.weight.as_ptr(),
                output.as_mut_ptr(),
                batch as i32,
                self.epsilon,
                gate.stride() as i32,
                stream.raw(),
            )
        })
    }
}

pub struct DeltaInputs<'a> {
    /// L2-нормированные, [batch, 16, 128].
    pub q: &'a DeviceBuffer<f32>,
    pub k: &'a DeviceBuffer<f32>,
    /// [batch, 48, 128].
    pub v: &'a DeviceBuffer<f32>,
    /// Затухание exp(g) в (0, 1), [batch, 48].
    pub alpha: &'a DeviceBuffer<f32>,
    /// Сила записи sigmoid(b), [batch, 48].
    pub beta: &'a DeviceBuffer<f32>,
}

/// Один шаг decode. Состояние обновляется на месте.
pub fn decode(
    state: &mut DeviceBuffer<u16>,
    inputs: &DeltaInputs<'_>,
    out: &mut DeviceBuffer<f32>,
    batch: usize,
    stream: &Stream,
) -> Result<()> {
    debug_assert_eq!(state.len(), batch * STATE_ELEMS);
    debug_assert_eq!(inputs.q.len(), batch * QK_ELEMS);
    debug_assert_eq!(out.len(), batch * V_ELEMS);
    check(unsafe {
        ffi::qwc_delta_decode(
            state.as_mut_ptr(),
            std::ptr::null(),
            inputs.q.as_ptr().cast(),
            inputs.k.as_ptr().cast(),
            inputs.v.as_ptr().cast(),
            inputs.alpha.as_ptr().cast(),
            inputs.beta.as_ptr().cast(),
            out.as_mut_ptr().cast(),
            batch as i32,
            batch as i32,
            stream.raw(),
        )
    })
}

/// Decode against the persistent state pool selected by scheduler slot IDs.
/// No state is gathered into a compact batch buffer.
pub fn decode_slots(
    state_pool: &mut DeviceBuffer<u16>,
    state_slots: &DeviceBuffer<u32>,
    inputs: &DeltaInputs<'_>,
    out: &mut DeviceBuffer<f32>,
    state_capacity: usize,
    batch: usize,
    stream: &Stream,
) -> Result<()> {
    assert_eq!(state_pool.len(), state_capacity * STATE_ELEMS);
    assert!(state_slots.len() >= batch);
    // The caller may pass the leading rows of a larger fused arena.
    debug_assert!(inputs.q.len() >= batch * QK_ELEMS);
    debug_assert!(inputs.k.len() >= batch * QK_ELEMS);
    debug_assert!(inputs.v.len() >= batch * V_ELEMS);
    debug_assert!(inputs.alpha.len() >= batch * GATE_ELEMS);
    debug_assert!(inputs.beta.len() >= batch * GATE_ELEMS);
    debug_assert!(out.len() >= batch * V_ELEMS);
    check(unsafe {
        ffi::qwc_delta_decode(
            state_pool.as_mut_ptr(),
            state_slots.as_ptr(),
            inputs.q.as_ptr().cast(),
            inputs.k.as_ptr().cast(),
            inputs.v.as_ptr().cast(),
            inputs.alpha.as_ptr().cast(),
            inputs.beta.as_ptr().cast(),
            out.as_mut_ptr().cast(),
            state_capacity as i32,
            batch as i32,
            stream.raw(),
        )
    })
}

/// Causal recurrent scan over one chunk of one persistent sequence.
#[allow(clippy::too_many_arguments)]
pub fn prefill_slot(
    state_pool: &mut DeviceBuffer<u16>,
    inputs: &DeltaInputs<'_>,
    out: &mut DeviceBuffer<f32>,
    state_capacity: usize,
    state_slot: usize,
    tokens: usize,
    row_offset: usize,
    state_mode: DeltaStateMode,
    stream: &Stream,
) -> Result<()> {
    assert_eq!(state_pool.len(), state_capacity * STATE_ELEMS);
    assert!(state_slot < state_capacity);
    assert!(tokens > 0);
    let end = row_offset + tokens;
    assert!(inputs.q.len() >= end * QK_ELEMS);
    assert!(inputs.k.len() >= end * QK_ELEMS);
    assert!(inputs.v.len() >= end * V_ELEMS);
    assert!(inputs.alpha.len() >= end * GATE_ELEMS);
    assert!(inputs.beta.len() >= end * GATE_ELEMS);
    assert!(out.len() >= end * V_ELEMS);
    check(unsafe {
        ffi::qwc_delta_prefill(
            state_pool.as_mut_ptr(),
            inputs.q.as_ptr().cast(),
            inputs.k.as_ptr().cast(),
            inputs.v.as_ptr().cast(),
            inputs.alpha.as_ptr().cast(),
            inputs.beta.as_ptr().cast(),
            out.as_mut_ptr().cast(),
            state_capacity as i32,
            state_slot as i32,
            tokens as i32,
            state_mode.round_per_token(),
            row_offset as i32,
            stream.raw(),
        )
    })
}

/// Трафик памяти одного вызова: состояние читается и переписывается целиком.
pub fn traffic_bytes(batch: usize) -> u64 {
    2 * (batch * STATE_ELEMS * std::mem::size_of::<u16>()) as u64
}

/// Эталон на CPU. Работает с тем же bf16-представлением состояния, что и GPU,
/// поэтому расхождение может давать только порядок суммирования.
pub mod reference {
    use super::*;
    use crate::bf16;

    // Эталон повторяет сигнатуру кернела один в один: группировать аргументы
    // в структуру значит спрятать расхождение с GPU-путём.
    #[allow(clippy::too_many_arguments)]
    pub fn decode(
        state: &mut [u16],
        q: &[f32],
        k: &[f32],
        v: &[f32],
        alpha: &[f32],
        beta: &[f32],
        out: &mut [f32],
        batch: usize,
    ) {
        const DK: usize = LA_K_HEAD_DIM;
        const DV: usize = LA_V_HEAD_DIM;
        const HV: usize = LA_NUM_V_HEADS;
        const RATIO: usize = LA_NUM_V_HEADS / LA_NUM_K_HEADS;

        for b in 0..batch {
            for h in 0..HV {
                let hk = h / RATIO;
                let kv = &k[(b * LA_NUM_K_HEADS + hk) * DK..][..DK];
                let qv = &q[(b * LA_NUM_K_HEADS + hk) * DK..][..DK];
                let kq: f32 = kv.iter().zip(qv).map(|(a, b)| a * b).sum();

                let a = alpha[b * HV + h];
                let bt = beta[b * HV + h];
                let s = &mut state[(b * HV + h) * DV * DK..][..DV * DK];
                let vv = &v[(b * HV + h) * DV..][..DV];
                let o = &mut out[(b * HV + h) * DV..][..DV];

                for row in 0..DV {
                    let sr = &mut s[row * DK..][..DK];
                    let mut u = 0.0f32;
                    let mut w = 0.0f32;
                    for j in 0..DK {
                        let x = bf16::to_f32(sr[j]);
                        u += x * kv[j];
                        w += x * qv[j];
                    }
                    let c = bt * (vv[row] - a * u);
                    o[row] = a * w + c * kq;
                    for j in 0..DK {
                        let x = bf16::to_f32(sr[j]);
                        sr[j] = bf16::from_f32(a * x + c * kv[j]);
                    }
                }
            }
        }
    }

    /// Эталон chunk-скана с FP32-состоянием: округление до BF16 происходит
    /// только на границе chunk, как в `DeltaStateMode::Fp32`.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_chunk_fp32(
        state: &mut [u16],
        q: &[f32],
        k: &[f32],
        v: &[f32],
        alpha: &[f32],
        beta: &[f32],
        out: &mut [f32],
        tokens: usize,
    ) {
        const DK: usize = LA_K_HEAD_DIM;
        const DV: usize = LA_V_HEAD_DIM;
        const HV: usize = LA_NUM_V_HEADS;
        const RATIO: usize = LA_NUM_V_HEADS / LA_NUM_K_HEADS;

        let mut wide: Vec<f32> = state.iter().map(|&x| bf16::to_f32(x)).collect();
        for h in 0..HV {
            let hk = h / RATIO;
            for row in 0..DV {
                let sr = &mut wide[(h * DV + row) * DK..][..DK];
                for token in 0..tokens {
                    let kv = &k[(token * LA_NUM_K_HEADS + hk) * DK..][..DK];
                    let qv = &q[(token * LA_NUM_K_HEADS + hk) * DK..][..DK];
                    let kq: f32 = kv.iter().zip(qv).map(|(a, b)| a * b).sum();
                    let a = alpha[token * HV + h];
                    let bt = beta[token * HV + h];
                    let mut u = 0.0f32;
                    let mut w = 0.0f32;
                    for j in 0..DK {
                        u += sr[j] * kv[j];
                        w += sr[j] * qv[j];
                    }
                    let c = bt * (v[(token * HV + h) * DV + row] - a * u);
                    out[(token * HV + h) * DV + row] = a * w + c * kq;
                    for j in 0..DK {
                        sr[j] = a * sr[j] + c * kv[j];
                    }
                }
            }
        }
        for (slot, value) in state.iter_mut().zip(&wide) {
            *slot = bf16::from_f32(*value);
        }
    }
}

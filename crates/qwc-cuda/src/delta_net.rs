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
/// Скаляров k.q на последовательность: по одному на k-голову.
pub const KQ_ELEMS: usize = LA_NUM_K_HEADS;
/// FP32 causal-convolution history per persistent sequence slot and layer.
pub const CONV_STATE_ELEMS: usize = qwc_core::arch::LA_CONV_CHANNELS * 3;

/// Matrix tile used by the parallel WY prefill scan.
pub const WY_CHUNK_SIZE: usize = 64;

/// Как считается chunk на prefill.
///
/// `Bf16` и `Fp32` — один и тот же рекуррентный проход по токенам, разной
/// точности состояния. `Bf16` округляет состояние после каждого токена,
/// поэтому chunk воспроизводит повторный decode бит в бит. `Fp32` держит
/// строку состояния в регистрах до конца chunk и округляет только запись на
/// границе — так делает эталонная реализация (flash-linear-attention копит
/// `b_h` в FP32 через все чанки). Их A/B изолирует потокенное округление как
/// источник расхождения.
///
/// `Wy` — другой алгоритм, а не другая точность: chunk считается матрицами
/// на тензорных ядрах (`bench/results/deltanet-wy-2026-09-16.md`). Состояние
/// внутри chunk копится в FP32, поэтому по округлениям `Wy` ближе всего к
/// `Fp32`. Это быстрейший путь на промптах длиннее пары чанков и он требует
/// `DeltaPrefillWorkspace`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeltaStateMode {
    Bf16,
    Fp32,
    Wy,
}

impl DeltaStateMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Fp32 => "fp32",
            Self::Wy => "wy",
        }
    }

    /// Только для рекуррентного пути: WY-скан токены поштучно не округляет,
    /// и вызов на нём — ошибка вызывающего, а не режим по умолчанию.
    fn round_per_token(self) -> i32 {
        match self {
            Self::Bf16 => 1,
            Self::Fp32 => 0,
            Self::Wy => unreachable!("WY-скан не округляет состояние по токенам"),
        }
    }
}

/// Чанков в самом длинном шаге: фазы, не зависящие от состояния, считаются
/// сразу по всем, поэтому их буферы рассчитаны на полную арену.
const WY_MAX_CHUNKS: usize = crate::MAX_STEP_ROWS / WY_CHUNK_SIZE;

const WY_QK_TILE_ELEMS: usize =
    WY_MAX_CHUNKS * LA_NUM_K_HEADS * WY_CHUNK_SIZE * LA_K_HEAD_DIM;
const WY_GRAM_ELEMS: usize =
    WY_MAX_CHUNKS * LA_NUM_K_HEADS * WY_CHUNK_SIZE * WY_CHUNK_SIZE;
const WY_HEAD_MATRIX_ELEMS: usize =
    WY_MAX_CHUNKS * GATE_ELEMS * WY_CHUNK_SIZE * WY_CHUNK_SIZE;
const WY_DECAY_ELEMS: usize = WY_MAX_CHUNKS * GATE_ELEMS * WY_CHUNK_SIZE;
const WY_VALUE_TILE_ELEMS: usize =
    WY_MAX_CHUNKS * GATE_ELEMS * WY_CHUNK_SIZE * LA_V_HEAD_DIM;

/// Resident scratch for the matrix (WY) prefill scan.
///
/// It is deliberately owned by the executor rather than allocated by the
/// launch.  A layer step invokes the scan up to 48 times and runtime CUDA
/// allocations would serialize that path.
///
/// Тайл фиксирован на 64 токена, но фазы, не зависящие от состояния,
/// считаются сразу по всем чанкам сегмента — поэтому буферы рассчитаны на
/// полную арену (`MAX_STEP_ROWS`) и стоят около 84 МБ.
pub struct DeltaPrefillWorkspace {
    query: DeviceBuffer<u16>,
    key: DeviceBuffer<u16>,
    gram_kk: DeviceBuffer<f32>,
    gram_qk: DeviceBuffer<f32>,
    inverse: DeviceBuffer<u16>,
    output_factor: DeviceBuffer<u16>,
    value_tile: DeviceBuffer<u16>,
    log_decay: DeviceBuffer<f32>,
}

impl DeltaPrefillWorkspace {
    pub fn new() -> Result<Self> {
        Ok(Self {
            query: DeviceBuffer::zeroed(WY_QK_TILE_ELEMS)?,
            key: DeviceBuffer::zeroed(WY_QK_TILE_ELEMS)?,
            gram_kk: DeviceBuffer::zeroed(WY_GRAM_ELEMS)?,
            gram_qk: DeviceBuffer::zeroed(WY_GRAM_ELEMS)?,
            inverse: DeviceBuffer::zeroed(WY_HEAD_MATRIX_ELEMS)?,
            output_factor: DeviceBuffer::zeroed(WY_HEAD_MATRIX_ELEMS)?,
            value_tile: DeviceBuffer::zeroed(WY_VALUE_TILE_ELEMS)?,
            log_decay: DeviceBuffer::zeroed(WY_DECAY_ELEMS)?,
        })
    }

    /// Байты, которые режим `Wy` добавляет к бюджету памяти движка.
    pub fn bytes(&self) -> usize {
        self.query.bytes()
            + self.key.bytes()
            + self.gram_kk.bytes()
            + self.gram_qk.bytes()
            + self.inverse.bytes()
            + self.output_factor.bytes()
            + self.value_tile.bytes()
            + self.log_decay.bytes()
    }
}

pub struct PreparedDelta {
    pub q: DeviceBuffer<f32>,
    pub k: DeviceBuffer<f32>,
    pub v: DeviceBuffer<f32>,
    pub alpha: DeviceBuffer<f32>,
    pub beta: DeviceBuffer<f32>,
    /// k.q на токен и k-голову: общий для всех строк состояния, поэтому
    /// считается один раз в prepare, а не в каждом варпе скана.
    pub kq: DeviceBuffer<f32>,
    batch: usize,
}

impl PreparedDelta {
    pub fn zeroed(batch: usize) -> Result<Self> {
        assert!((1..=crate::MAX_STEP_ROWS).contains(&batch));
        Ok(Self {
            q: DeviceBuffer::zeroed(batch * QK_ELEMS)?,
            k: DeviceBuffer::zeroed(batch * QK_ELEMS)?,
            v: DeviceBuffer::zeroed(batch * V_ELEMS)?,
            alpha: DeviceBuffer::zeroed(batch * GATE_ELEMS)?,
            beta: DeviceBuffer::zeroed(batch * GATE_ELEMS)?,
            kq: DeviceBuffer::zeroed(batch * KQ_ELEMS)?,
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
            kq: &self.kq,
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
                output.kq.as_mut_ptr(),
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
        // Проверяются строки, которых кернел действительно касается, а не вся
        // ёмкость арены: доигрывание принятых черновиков подаёт вход из
        // маленького сохранённого буфера на несколько строк.
        let used = row_offset + tokens;
        assert!(mixed_qkv.fits(used, qwc_core::arch::LA_CONV_CHANNELS));
        assert!(a_projection.fits(used, GATE_ELEMS));
        assert!(b_projection.fits(used, GATE_ELEMS));
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
                output.kq.as_mut_ptr(),
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
    /// Скаляр k.q, [batch, 16].
    pub kq: &'a DeviceBuffer<f32>,
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
            inputs.kq.as_ptr().cast(),
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
            inputs.kq.as_ptr().cast(),
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
    assert!(inputs.kq.len() >= end * KQ_ELEMS);
    assert!(out.len() >= end * V_ELEMS);
    check(unsafe {
        ffi::qwc_delta_prefill(
            state_pool.as_mut_ptr(),
            inputs.q.as_ptr().cast(),
            inputs.k.as_ptr().cast(),
            inputs.v.as_ptr().cast(),
            inputs.alpha.as_ptr().cast(),
            inputs.beta.as_ptr().cast(),
            inputs.kq.as_ptr().cast(),
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

/// Чанковый (WY) скан: `DeltaStateMode::Wy`.
///
/// Рекуррентный кернел остаётся реализацией `Bf16` и `Fp32`: округление после
/// каждого токена — само по себе рекуррентность, матричной формой она не
/// выражается. Поэтому это отдельная точка входа, а не флаг внутри
/// `prefill_slot`.
#[allow(clippy::too_many_arguments)]
pub fn prefill_slot_wy(
    state_pool: &mut DeviceBuffer<u16>,
    inputs: &DeltaInputs<'_>,
    out: &mut DeviceBuffer<f32>,
    workspace: &mut DeltaPrefillWorkspace,
    state_capacity: usize,
    state_slot: usize,
    tokens: usize,
    row_offset: usize,
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
        ffi::qwc_delta_prefill_wy(
            state_pool.as_mut_ptr(),
            inputs.q.as_ptr().cast(),
            inputs.k.as_ptr().cast(),
            inputs.v.as_ptr().cast(),
            inputs.alpha.as_ptr().cast(),
            inputs.beta.as_ptr().cast(),
            out.as_mut_ptr().cast(),
            workspace.query.as_mut_ptr(),
            workspace.key.as_mut_ptr(),
            workspace.gram_kk.as_mut_ptr(),
            workspace.gram_qk.as_mut_ptr(),
            workspace.inverse.as_mut_ptr(),
            workspace.output_factor.as_mut_ptr(),
            workspace.value_tile.as_mut_ptr(),
            workspace.log_decay.as_mut_ptr(),
            state_capacity as i32,
            state_slot as i32,
            tokens as i32,
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

    /// Скан в чанковой (WY) форме: рекуррентность по токенам заменена
    /// матрицами внутри чанка длины `chunk`.
    ///
    /// Вывод. Обозначим за `S_t` состояние после токена t, за gamma_t —
    /// произведение alpha по токенам чанка до t включительно. Из
    /// `S_t = a_t S_{t-1} + c_t k_t^T` по индукции следует
    /// `S_t = gamma_t (S_0 + sum_{r<=t} (c_r / gamma_r) k_r^T)`, и подстановка
    /// этого в `c_t = b_t (v_t - a_t S_{t-1} k_t)` даёт треугольную систему
    ///
    ///   c_t + sum_{r<t} A[t][r] c_r = b_t v_t - b_t gamma_t (S_0 k_t),
    ///   A[t][r] = b_t (gamma_t / gamma_r) (k_r . k_t).
    ///
    /// Выход и финальное состояние — тоже матрицы:
    ///
    ///   out_t = gamma_t (S_0 q_t) + sum_{r<=t} (gamma_t / gamma_r)(k_r . q_t) c_r,
    ///   S_C   = gamma_C S_0 + sum_t (gamma_C / gamma_t) c_t k_t^T.
    ///
    /// Все множители — отношения gamma по парам (r <= t), а alpha в (0, 1],
    /// поэтому каждое из них не больше единицы. Считать их делением всё равно
    /// нельзя: gamma само по себе — произведение до 64 множителей и уходит
    /// под fp32 уже при alpha около 0.2, после чего отношение превращается в
    /// 0/0. Поэтому затухание копится в логарифмах, а отношение берётся как
    /// exp от разности.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_chunked_fp32(
        state: &mut [u16],
        q: &[f32],
        k: &[f32],
        v: &[f32],
        alpha: &[f32],
        beta: &[f32],
        out: &mut [f32],
        tokens: usize,
        chunk: usize,
    ) {
        const DK: usize = LA_K_HEAD_DIM;
        const DV: usize = LA_V_HEAD_DIM;
        const HV: usize = LA_NUM_V_HEADS;
        const RATIO: usize = LA_NUM_V_HEADS / LA_NUM_K_HEADS;
        assert!(chunk > 0);

        let mut wide: Vec<f32> = state.iter().map(|&x| bf16::to_f32(x)).collect();
        for h in 0..HV {
            let hk = h / RATIO;
            let s = &mut wide[h * DV * DK..][..DV * DK];
            let mut start = 0usize;
            while start < tokens {
                let len = chunk.min(tokens - start);
                let key = |t: usize| &k[((start + t) * LA_NUM_K_HEADS + hk) * DK..][..DK];
                let query = |t: usize| &q[((start + t) * LA_NUM_K_HEADS + hk) * DK..][..DK];

                // Затухание копится в логарифмах: произведение alpha по чанку
                // уходит под fp32 уже при alpha около 0.2, и отношения
                // gamma_t / gamma_r превращаются в 0/0. Разность логарифмов
                // конечна всегда.
                let mut cumulative = vec![0.0f32; len];
                let mut running = 0.0f32;
                for (t, slot) in cumulative.iter_mut().enumerate() {
                    running += alpha[(start + t) * HV + h].max(1.0e-38).ln();
                    *slot = running;
                }
                let gamma: Vec<f32> = cumulative.iter().map(|x| x.exp()).collect();
                let ratio = |t: usize, r: usize| (cumulative[t] - cumulative[r]).exp();

                // Правая часть: b_t v_t - b_t gamma_t (S_0 k_t).
                let mut c = vec![0.0f32; len * DV];
                for t in 0..len {
                    let bt = beta[(start + t) * HV + h];
                    let kt = key(t);
                    for row in 0..DV {
                        let sk: f32 = s[row * DK..][..DK]
                            .iter()
                            .zip(kt)
                            .map(|(x, y)| x * y)
                            .sum();
                        c[t * DV + row] = bt
                            * (v[((start + t) * HV + h) * DV + row] - gamma[t] * sk);
                    }
                }
                // Прямая подстановка по треугольной системе: строка t видит
                // только уже посчитанные r < t.
                for t in 0..len {
                    let bt = beta[(start + t) * HV + h];
                    let kt = key(t);
                    for r in 0..t {
                        let kk: f32 = key(r).iter().zip(kt).map(|(x, y)| x * y).sum();
                        let factor = bt * ratio(t, r) * kk;
                        if factor == 0.0 {
                            continue;
                        }
                        for row in 0..DV {
                            c[t * DV + row] -= factor * c[r * DV + row];
                        }
                    }
                }

                // Выход: вклад входного состояния плюс нижнетреугольная часть.
                for t in 0..len {
                    let qt = query(t);
                    for row in 0..DV {
                        let sq: f32 = s[row * DK..][..DK]
                            .iter()
                            .zip(qt)
                            .map(|(x, y)| x * y)
                            .sum();
                        out[((start + t) * HV + h) * DV + row] = gamma[t] * sq;
                    }
                    for r in 0..=t {
                        let kq: f32 = key(r).iter().zip(qt).map(|(x, y)| x * y).sum();
                        let factor = ratio(t, r) * kq;
                        for row in 0..DV {
                            out[((start + t) * HV + h) * DV + row] += factor * c[r * DV + row];
                        }
                    }
                }

                // Состояние на конец чанка.
                let last = gamma[len - 1];
                for row in 0..DV {
                    for j in 0..DK {
                        s[row * DK + j] *= last;
                    }
                }
                for t in 0..len {
                    let weight = ratio(len - 1, t);
                    let kt = key(t);
                    for row in 0..DV {
                        let scaled = weight * c[t * DV + row];
                        for j in 0..DK {
                            s[row * DK + j] += scaled * kt[j];
                        }
                    }
                }
                start += len;
            }
        }
        for (slot, value) in state.iter_mut().zip(&wide) {
            *slot = bf16::from_f32(*value);
        }
    }

    /// Эталон рекуррентного chunk-скана с FP32-состоянием: округление до
    /// BF16 происходит только на границе chunk, как в `DeltaStateMode::Fp32`.
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

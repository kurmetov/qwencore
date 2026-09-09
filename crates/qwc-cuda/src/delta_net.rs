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
            inputs.q.as_ptr().cast(),
            inputs.k.as_ptr().cast(),
            inputs.v.as_ptr().cast(),
            inputs.alpha.as_ptr().cast(),
            inputs.beta.as_ptr().cast(),
            out.as_mut_ptr().cast(),
            batch as i32,
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
}

//! Сверка GPU-кернела Gated DeltaNet с эталоном на CPU.
//! Требует RTX 5090.

use qwc_cuda::delta_net::{self, DeltaInputs, GATE_ELEMS, QK_ELEMS, STATE_ELEMS, V_ELEMS};
use qwc_cuda::{DeviceBuffer, Stream, bf16};
use qwc_core::arch::{LA_K_HEAD_DIM, LA_NUM_K_HEADS};

/// Детерминированный генератор, чтобы падения воспроизводились.
struct Rng(u64);

impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
}

fn l2_normalize_heads(v: &mut [f32], heads: usize, dim: usize) {
    for h in 0..v.len() / dim {
        let _ = heads;
        let s = &mut v[h * dim..][..dim];
        let norm = s.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
        for x in s.iter_mut() {
            *x /= norm;
        }
    }
}

#[test]
fn matches_cpu_reference() {
    let batch = 3;
    let mut rng = Rng(0x51ED_2701);

    let mut state: Vec<u16> = (0..batch * STATE_ELEMS)
        .map(|_| bf16::from_f32(rng.next_f32() * 0.1))
        .collect();
    let mut q: Vec<f32> = (0..batch * QK_ELEMS).map(|_| rng.next_f32()).collect();
    let mut k: Vec<f32> = (0..batch * QK_ELEMS).map(|_| rng.next_f32()).collect();
    l2_normalize_heads(&mut q, LA_NUM_K_HEADS, LA_K_HEAD_DIM);
    l2_normalize_heads(&mut k, LA_NUM_K_HEADS, LA_K_HEAD_DIM);
    let v: Vec<f32> = (0..batch * V_ELEMS).map(|_| rng.next_f32()).collect();
    // alpha — затухание в (0,1), beta — сила записи в (0,1).
    let alpha: Vec<f32> = (0..batch * GATE_ELEMS).map(|_| 0.9 + rng.next_f32() * 0.09).collect();
    let beta: Vec<f32> = (0..batch * GATE_ELEMS).map(|_| 0.5 + rng.next_f32() * 0.4).collect();

    // Эталон.
    let mut ref_state = state.clone();
    let mut ref_out = vec![0.0f32; batch * V_ELEMS];
    delta_net::reference::decode(
        &mut ref_state, &q, &k, &v, &alpha, &beta, &mut ref_out, batch,
    );

    // GPU.
    let stream = Stream::new().unwrap();
    let mut d_state = DeviceBuffer::from_slice(&state).unwrap();
    let d_q = DeviceBuffer::from_slice(&q).unwrap();
    let d_k = DeviceBuffer::from_slice(&k).unwrap();
    let d_v = DeviceBuffer::from_slice(&v).unwrap();
    let d_alpha = DeviceBuffer::from_slice(&alpha).unwrap();
    let d_beta = DeviceBuffer::from_slice(&beta).unwrap();
    let mut d_out = DeviceBuffer::<f32>::zeroed(batch * V_ELEMS).unwrap();

    let inputs = DeltaInputs { q: &d_q, k: &d_k, v: &d_v, alpha: &d_alpha, beta: &d_beta };
    delta_net::decode(&mut d_state, &inputs, &mut d_out, batch, &stream).unwrap();
    stream.synchronize().unwrap();

    let gpu_out = d_out.to_vec().unwrap();
    state = d_state.to_vec().unwrap();

    // Выход считается в fp32, расхождение возможно только от порядка суммирования.
    let mut worst_out = 0.0f32;
    for (g, r) in gpu_out.iter().zip(&ref_out) {
        let d = (g - r).abs() / r.abs().max(1e-3);
        worst_out = worst_out.max(d);
    }
    assert!(worst_out < 2e-3, "выход расходится на {worst_out:.2e}");

    // Состояние хранится в bf16: допускаем расхождение в один ULP (~0.8%),
    // возникающее когда fp32-результат попадает на границу округления.
    let mut worst_state = 0.0f32;
    let mut ulp_diffs = 0usize;
    for (g, r) in state.iter().zip(&ref_state) {
        if g != r {
            ulp_diffs += 1;
            let (gf, rf) = (bf16::to_f32(*g), bf16::to_f32(*r));
            worst_state = worst_state.max((gf - rf).abs() / rf.abs().max(1e-3));
        }
    }
    let frac = ulp_diffs as f64 / state.len() as f64;
    assert!(worst_state < 1e-2, "состояние расходится на {worst_state:.2e}");
    assert!(frac < 0.02, "слишком много расхождений в состоянии: {:.2}%", frac * 100.0);
}

#[test]
fn decay_shrinks_state_when_write_disabled() {
    // При beta = 0 запись отключена и состояние должно просто затухать в alpha раз.
    let batch = 1;
    let mut rng = Rng(7);
    let state: Vec<u16> = (0..STATE_ELEMS).map(|_| bf16::from_f32(rng.next_f32())).collect();
    let mut q = vec![0.0f32; QK_ELEMS];
    let mut k = vec![0.0f32; QK_ELEMS];
    for (i, (a, b)) in q.iter_mut().zip(k.iter_mut()).enumerate() {
        *a = if i % LA_K_HEAD_DIM == 0 { 1.0 } else { 0.0 };
        *b = if i % LA_K_HEAD_DIM == 1 { 1.0 } else { 0.0 };
    }
    let v = vec![1.0f32; V_ELEMS];
    let alpha = vec![0.5f32; GATE_ELEMS];
    let beta = vec![0.0f32; GATE_ELEMS];

    let stream = Stream::new().unwrap();
    let mut d_state = DeviceBuffer::from_slice(&state).unwrap();
    let d_q = DeviceBuffer::from_slice(&q).unwrap();
    let d_k = DeviceBuffer::from_slice(&k).unwrap();
    let d_v = DeviceBuffer::from_slice(&v).unwrap();
    let d_alpha = DeviceBuffer::from_slice(&alpha).unwrap();
    let d_beta = DeviceBuffer::from_slice(&beta).unwrap();
    let mut d_out = DeviceBuffer::<f32>::zeroed(V_ELEMS).unwrap();

    let inputs = DeltaInputs { q: &d_q, k: &d_k, v: &d_v, alpha: &d_alpha, beta: &d_beta };
    delta_net::decode(&mut d_state, &inputs, &mut d_out, batch, &stream).unwrap();
    stream.synchronize().unwrap();

    let after = d_state.to_vec().unwrap();
    for (before, now) in state.iter().zip(&after) {
        let expect = bf16::from_f32(0.5 * bf16::to_f32(*before));
        assert_eq!(*now, expect, "затухание применено неверно");
    }
}

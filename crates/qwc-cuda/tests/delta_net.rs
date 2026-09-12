//! Сверка GPU-кернела Gated DeltaNet с эталоном на CPU.
//! Требует RTX 5090.

use qwc_core::arch::{LA_CONV_CHANNELS, LA_K_HEAD_DIM, LA_NUM_K_HEADS, LA_V_HEAD_DIM};
use qwc_cuda::delta_net::{
    self, CONV_STATE_ELEMS, DeltaInputs, DeltaOutputNorm, DeltaPreprocessor, DeltaStateMode,
    GATE_ELEMS, PreparedDelta, QK_ELEMS, RowView, STATE_ELEMS, V_ELEMS,
};
use qwc_cuda::{DeviceBuffer, Stream, bf16};

/// Детерминированный генератор, чтобы падения воспроизводились.
struct Rng(u64);

impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
}

/// k.q на каждую строку и k-голову — ровно то, что считает prepare на GPU.
/// Скан читает этот скаляр вместо того, чтобы пересчитывать его в каждом варпе.
fn kq_rows(q: &[f32], k: &[f32]) -> Vec<f32> {
    assert_eq!(q.len(), k.len());
    (0..q.len() / LA_K_HEAD_DIM)
        .map(|head| {
            let base = head * LA_K_HEAD_DIM;
            (0..LA_K_HEAD_DIM).map(|i| q[base + i] * k[base + i]).sum()
        })
        .collect()
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
    let alpha: Vec<f32> = (0..batch * GATE_ELEMS)
        .map(|_| 0.9 + rng.next_f32() * 0.09)
        .collect();
    let beta: Vec<f32> = (0..batch * GATE_ELEMS)
        .map(|_| 0.5 + rng.next_f32() * 0.4)
        .collect();

    // Эталон.
    let mut ref_state = state.clone();
    let mut ref_out = vec![0.0f32; batch * V_ELEMS];
    delta_net::reference::decode(
        &mut ref_state,
        &q,
        &k,
        &v,
        &alpha,
        &beta,
        &mut ref_out,
        batch,
    );

    // GPU.
    let stream = Stream::new().unwrap();
    let mut d_state = DeviceBuffer::from_slice(&state).unwrap();
    let d_q = DeviceBuffer::from_slice(&q).unwrap();
    let d_k = DeviceBuffer::from_slice(&k).unwrap();
    let d_kq = DeviceBuffer::from_slice(&kq_rows(&q, &k)).unwrap();
    let d_v = DeviceBuffer::from_slice(&v).unwrap();
    let d_alpha = DeviceBuffer::from_slice(&alpha).unwrap();
    let d_beta = DeviceBuffer::from_slice(&beta).unwrap();
    let mut d_out = DeviceBuffer::<f32>::zeroed(batch * V_ELEMS).unwrap();

    let inputs = DeltaInputs {
        q: &d_q,
        k: &d_k,
        v: &d_v,
        alpha: &d_alpha,
        beta: &d_beta,
        kq: &d_kq,
    };
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
    assert!(
        worst_state < 1e-2,
        "состояние расходится на {worst_state:.2e}"
    );
    assert!(
        frac < 0.02,
        "слишком много расхождений в состоянии: {:.2}%",
        frac * 100.0
    );
}

#[test]
fn decay_shrinks_state_when_write_disabled() {
    // При beta = 0 запись отключена и состояние должно просто затухать в alpha раз.
    let batch = 1;
    let mut rng = Rng(7);
    let state: Vec<u16> = (0..STATE_ELEMS)
        .map(|_| bf16::from_f32(rng.next_f32()))
        .collect();
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
    let d_kq = DeviceBuffer::from_slice(&kq_rows(&q, &k)).unwrap();
    let d_v = DeviceBuffer::from_slice(&v).unwrap();
    let d_alpha = DeviceBuffer::from_slice(&alpha).unwrap();
    let d_beta = DeviceBuffer::from_slice(&beta).unwrap();
    let mut d_out = DeviceBuffer::<f32>::zeroed(V_ELEMS).unwrap();

    let inputs = DeltaInputs {
        q: &d_q,
        k: &d_k,
        v: &d_v,
        alpha: &d_alpha,
        beta: &d_beta,
        kq: &d_kq,
    };
    delta_net::decode(&mut d_state, &inputs, &mut d_out, batch, &stream).unwrap();
    stream.synchronize().unwrap();

    let after = d_state.to_vec().unwrap();
    for (before, now) in state.iter().zip(&after) {
        let expect = bf16::from_f32(0.5 * bf16::to_f32(*before));
        assert_eq!(*now, expect, "затухание применено неверно");
    }
}

#[test]
fn scheduler_slots_address_persistent_state_without_gather() {
    let batch = 2;
    let capacity = 3;
    let slots = vec![2u32, 0];
    let state: Vec<u16> = (0..capacity * STATE_ELEMS)
        .map(|index| bf16::from_f32((index % 31) as f32 * 0.01 - 0.15))
        .collect();
    let mut q = vec![0.0f32; batch * QK_ELEMS];
    let mut k = vec![0.0f32; batch * QK_ELEMS];
    for head in 0..batch * LA_NUM_K_HEADS {
        q[head * LA_K_HEAD_DIM] = 1.0;
        k[head * LA_K_HEAD_DIM + 1] = 1.0;
    }
    let v = vec![1.0f32; batch * V_ELEMS];
    let alpha = vec![0.5f32; batch * GATE_ELEMS];
    let beta = vec![0.0f32; batch * GATE_ELEMS];
    let inputs_host = (&q, &k, &v, &alpha, &beta);

    let stream = Stream::new().unwrap();
    let mut device_state = DeviceBuffer::from_slice(&state).unwrap();
    let device_slots = DeviceBuffer::from_slice(&slots).unwrap();
    let device_q = DeviceBuffer::from_slice(inputs_host.0).unwrap();
    let device_k = DeviceBuffer::from_slice(inputs_host.1).unwrap();
    let device_kq = DeviceBuffer::from_slice(&kq_rows(inputs_host.0, inputs_host.1)).unwrap();
    let device_v = DeviceBuffer::from_slice(inputs_host.2).unwrap();
    let device_alpha = DeviceBuffer::from_slice(inputs_host.3).unwrap();
    let device_beta = DeviceBuffer::from_slice(inputs_host.4).unwrap();
    let inputs = DeltaInputs {
        q: &device_q,
        k: &device_k,
        v: &device_v,
        alpha: &device_alpha,
        beta: &device_beta,
        kq: &device_kq,
    };
    let mut output = DeviceBuffer::<f32>::zeroed(batch * V_ELEMS).unwrap();
    delta_net::decode_slots(
        &mut device_state,
        &device_slots,
        &inputs,
        &mut output,
        capacity,
        batch,
        &stream,
    )
    .unwrap();
    stream.synchronize().unwrap();

    let actual = device_state.to_vec().unwrap();
    for slot in 0..capacity {
        for element in 0..STATE_ELEMS {
            let index = slot * STATE_ELEMS + element;
            let expected = if slots.contains(&(slot as u32)) {
                bf16::from_f32(0.5 * bf16::to_f32(state[index]))
            } else {
                state[index]
            };
            assert_eq!(actual[index], expected, "slot={slot}, element={element}");
        }
    }
}

#[test]
fn chunk_prefill_matches_repeated_recurrent_decode() {
    let tokens = 4;
    let capacity = 2;
    let slot = 1;
    let mut rng = Rng(0xC4A5_5090);
    let initial: Vec<u16> = (0..capacity * STATE_ELEMS)
        .map(|_| bf16::from_f32(rng.next_f32() * 0.05))
        .collect();
    let mut q: Vec<f32> = (0..tokens * QK_ELEMS).map(|_| rng.next_f32()).collect();
    let mut k: Vec<f32> = (0..tokens * QK_ELEMS).map(|_| rng.next_f32()).collect();
    l2_normalize_heads(&mut q, LA_NUM_K_HEADS, LA_K_HEAD_DIM);
    l2_normalize_heads(&mut k, LA_NUM_K_HEADS, LA_K_HEAD_DIM);
    let v: Vec<f32> = (0..tokens * V_ELEMS)
        .map(|_| rng.next_f32() * 0.2)
        .collect();
    let alpha: Vec<f32> = (0..tokens * GATE_ELEMS)
        .map(|_| 0.92 + rng.next_f32() * 0.04)
        .collect();
    let beta: Vec<f32> = (0..tokens * GATE_ELEMS)
        .map(|_| 0.45 + rng.next_f32() * 0.2)
        .collect();

    let mut expected_state = initial[slot * STATE_ELEMS..][..STATE_ELEMS].to_vec();
    let mut expected_out = vec![0.0f32; tokens * V_ELEMS];
    for token in 0..tokens {
        delta_net::reference::decode(
            &mut expected_state,
            &q[token * QK_ELEMS..][..QK_ELEMS],
            &k[token * QK_ELEMS..][..QK_ELEMS],
            &v[token * V_ELEMS..][..V_ELEMS],
            &alpha[token * GATE_ELEMS..][..GATE_ELEMS],
            &beta[token * GATE_ELEMS..][..GATE_ELEMS],
            &mut expected_out[token * V_ELEMS..][..V_ELEMS],
            1,
        );
    }

    let stream = Stream::new().unwrap();
    let mut device_state = DeviceBuffer::from_slice(&initial).unwrap();
    let device_q = DeviceBuffer::from_slice(&q).unwrap();
    let device_k = DeviceBuffer::from_slice(&k).unwrap();
    let device_kq = DeviceBuffer::from_slice(&kq_rows(&q, &k)).unwrap();
    let device_v = DeviceBuffer::from_slice(&v).unwrap();
    let device_alpha = DeviceBuffer::from_slice(&alpha).unwrap();
    let device_beta = DeviceBuffer::from_slice(&beta).unwrap();
    let inputs = DeltaInputs {
        q: &device_q,
        k: &device_k,
        v: &device_v,
        alpha: &device_alpha,
        beta: &device_beta,
        kq: &device_kq,
    };
    let mut output = DeviceBuffer::<f32>::zeroed(tokens * V_ELEMS).unwrap();
    delta_net::prefill_slot(
        &mut device_state,
        &inputs,
        &mut output,
        capacity,
        slot,
        tokens,
        0,
        DeltaStateMode::Bf16,
        &stream,
    )
    .unwrap();
    stream.synchronize().unwrap();

    let actual_out = output.to_vec().unwrap();
    let actual_pool = device_state.to_vec().unwrap();
    let actual_state = &actual_pool[slot * STATE_ELEMS..][..STATE_ELEMS];
    let mut worst_out = 0.0f32;
    for (&actual, &expected) in actual_out.iter().zip(&expected_out) {
        worst_out = worst_out.max((actual - expected).abs() / expected.abs().max(1e-3));
    }
    assert!(
        worst_out < 2e-3,
        "prefill output differs by {worst_out:.2e}"
    );
    let differing = actual_state
        .iter()
        .zip(&expected_state)
        .filter(|(actual, expected)| actual != expected)
        .count();
    assert!(
        differing as f64 / (STATE_ELEMS as f64) < 0.02,
        "too many BF16 state differences: {differing}/{STATE_ELEMS}"
    );
    assert_eq!(&actual_pool[..STATE_ELEMS], &initial[..STATE_ELEMS]);
}

/// FP32-режим chunk-скана: состояние остаётся в регистрах до конца chunk.
/// Сверяется с отдельным CPU-эталоном и обязан отличаться от BF16-режима,
/// иначе переключатель ничего не изолирует.
#[test]
fn chunk_prefill_fp32_state_matches_fp32_reference_and_differs_from_bf16() {
    let tokens = 16;
    let capacity = 2;
    let slot = 1;
    let mut rng = Rng(0x5090_C4A5);
    let initial: Vec<u16> = (0..capacity * STATE_ELEMS)
        .map(|_| bf16::from_f32(rng.next_f32() * 0.05))
        .collect();
    let mut q: Vec<f32> = (0..tokens * QK_ELEMS).map(|_| rng.next_f32()).collect();
    let mut k: Vec<f32> = (0..tokens * QK_ELEMS).map(|_| rng.next_f32()).collect();
    l2_normalize_heads(&mut q, LA_NUM_K_HEADS, LA_K_HEAD_DIM);
    l2_normalize_heads(&mut k, LA_NUM_K_HEADS, LA_K_HEAD_DIM);
    let v: Vec<f32> = (0..tokens * V_ELEMS)
        .map(|_| rng.next_f32() * 0.2)
        .collect();
    let alpha: Vec<f32> = (0..tokens * GATE_ELEMS)
        .map(|_| 0.92 + rng.next_f32() * 0.04)
        .collect();
    let beta: Vec<f32> = (0..tokens * GATE_ELEMS)
        .map(|_| 0.45 + rng.next_f32() * 0.2)
        .collect();

    let mut expected_state = initial[slot * STATE_ELEMS..][..STATE_ELEMS].to_vec();
    let mut expected_out = vec![0.0f32; tokens * V_ELEMS];
    delta_net::reference::prefill_chunk_fp32(
        &mut expected_state,
        &q,
        &k,
        &v,
        &alpha,
        &beta,
        &mut expected_out,
        tokens,
    );

    let stream = Stream::new().unwrap();
    let device_q = DeviceBuffer::from_slice(&q).unwrap();
    let device_k = DeviceBuffer::from_slice(&k).unwrap();
    let device_kq = DeviceBuffer::from_slice(&kq_rows(&q, &k)).unwrap();
    let device_v = DeviceBuffer::from_slice(&v).unwrap();
    let device_alpha = DeviceBuffer::from_slice(&alpha).unwrap();
    let device_beta = DeviceBuffer::from_slice(&beta).unwrap();
    let inputs = DeltaInputs {
        q: &device_q,
        k: &device_k,
        v: &device_v,
        alpha: &device_alpha,
        beta: &device_beta,
        kq: &device_kq,
    };

    let run = |mode| {
        let mut device_state = DeviceBuffer::from_slice(&initial).unwrap();
        let mut output = DeviceBuffer::<f32>::zeroed(tokens * V_ELEMS).unwrap();
        delta_net::prefill_slot(
            &mut device_state,
            &inputs,
            &mut output,
            capacity,
            slot,
            tokens,
            0,
            mode,
            &stream,
        )
        .unwrap();
        stream.synchronize().unwrap();
        (output.to_vec().unwrap(), device_state.to_vec().unwrap())
    };
    let (fp32_out, fp32_pool) = run(DeltaStateMode::Fp32);
    let (bf16_out, _) = run(DeltaStateMode::Bf16);

    let mut worst = 0.0f32;
    for (&actual, &expected) in fp32_out.iter().zip(&expected_out) {
        worst = worst.max((actual - expected).abs() / expected.abs().max(1e-3));
    }
    assert!(worst < 2e-3, "fp32 prefill output differs by {worst:.2e}");

    let actual_state = &fp32_pool[slot * STATE_ELEMS..][..STATE_ELEMS];
    let differing = actual_state
        .iter()
        .zip(&expected_state)
        .filter(|(actual, expected)| actual != expected)
        .count();
    assert!(
        differing as f64 / (STATE_ELEMS as f64) < 0.02,
        "too many BF16 state differences: {differing}/{STATE_ELEMS}"
    );
    assert_eq!(&fp32_pool[..STATE_ELEMS], &initial[..STATE_ELEMS]);

    let drift = fp32_out
        .iter()
        .zip(&bf16_out)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        drift > 0.0,
        "fp32 and bf16 chunk scans produced identical output; the switch is inert"
    );
}

/// A fused step places several sequences in one token arena, so the scan has to
/// read and write only its own slice. Running the same tokens at offset zero
/// and at an offset must give the same answer and must leave the other rows
/// untouched.
#[test]
fn chunk_prefill_row_offset_reads_and_writes_only_its_slice() {
    let tokens = 6;
    let offset = 5;
    let arena = tokens + offset + 3;
    let capacity = 2;
    let slot = 1;
    let mut rng = Rng(0x0FF5_E701);
    let initial: Vec<u16> = (0..capacity * STATE_ELEMS)
        .map(|_| bf16::from_f32(rng.next_f32() * 0.05))
        .collect();
    let mut q: Vec<f32> = (0..tokens * QK_ELEMS).map(|_| rng.next_f32()).collect();
    let mut k: Vec<f32> = (0..tokens * QK_ELEMS).map(|_| rng.next_f32()).collect();
    l2_normalize_heads(&mut q, LA_NUM_K_HEADS, LA_K_HEAD_DIM);
    l2_normalize_heads(&mut k, LA_NUM_K_HEADS, LA_K_HEAD_DIM);
    let v: Vec<f32> = (0..tokens * V_ELEMS)
        .map(|_| rng.next_f32() * 0.2)
        .collect();
    let alpha: Vec<f32> = (0..tokens * GATE_ELEMS)
        .map(|_| 0.92 + rng.next_f32() * 0.04)
        .collect();
    let beta: Vec<f32> = (0..tokens * GATE_ELEMS)
        .map(|_| 0.45 + rng.next_f32() * 0.2)
        .collect();

    let stream = Stream::new().unwrap();
    let run = |row_offset: usize| {
        // The same tokens are placed at `row_offset` inside a wider arena; the
        // remaining rows are filled with a sentinel that must survive.
        let mut wide_q = vec![7.0f32; arena * QK_ELEMS];
        let mut wide_k = vec![7.0f32; arena * QK_ELEMS];
        let mut wide_v = vec![7.0f32; arena * V_ELEMS];
        let mut wide_alpha = vec![7.0f32; arena * GATE_ELEMS];
        let mut wide_beta = vec![7.0f32; arena * GATE_ELEMS];
        wide_q[row_offset * QK_ELEMS..][..tokens * QK_ELEMS].copy_from_slice(&q);
        wide_k[row_offset * QK_ELEMS..][..tokens * QK_ELEMS].copy_from_slice(&k);
        wide_v[row_offset * V_ELEMS..][..tokens * V_ELEMS].copy_from_slice(&v);
        wide_alpha[row_offset * GATE_ELEMS..][..tokens * GATE_ELEMS].copy_from_slice(&alpha);
        wide_beta[row_offset * GATE_ELEMS..][..tokens * GATE_ELEMS].copy_from_slice(&beta);

        let mut device_state = DeviceBuffer::from_slice(&initial).unwrap();
        let device_q = DeviceBuffer::from_slice(&wide_q).unwrap();
        let device_k = DeviceBuffer::from_slice(&wide_k).unwrap();
        let device_kq = DeviceBuffer::from_slice(&kq_rows(&wide_q, &wide_k)).unwrap();
        let device_v = DeviceBuffer::from_slice(&wide_v).unwrap();
        let device_alpha = DeviceBuffer::from_slice(&wide_alpha).unwrap();
        let device_beta = DeviceBuffer::from_slice(&wide_beta).unwrap();
        let inputs = DeltaInputs {
            q: &device_q,
            k: &device_k,
            v: &device_v,
            alpha: &device_alpha,
            beta: &device_beta,
            kq: &device_kq,
        };
        let sentinel = -3.5f32;
        let mut output = DeviceBuffer::from_slice(&vec![sentinel; arena * V_ELEMS]).unwrap();
        delta_net::prefill_slot(
            &mut device_state,
            &inputs,
            &mut output,
            capacity,
            slot,
            tokens,
            row_offset,
            DeltaStateMode::Bf16,
            &stream,
        )
        .unwrap();
        stream.synchronize().unwrap();
        (output.to_vec().unwrap(), device_state.to_vec().unwrap())
    };

    let (base_out, base_state) = run(0);
    let (shifted_out, shifted_state) = run(offset);

    assert_eq!(
        base_state, shifted_state,
        "row offset must not change the recurrent state"
    );
    assert_eq!(
        &base_out[..tokens * V_ELEMS],
        &shifted_out[offset * V_ELEMS..][..tokens * V_ELEMS],
        "row offset must not change the output"
    );
    assert!(
        shifted_out[..offset * V_ELEMS]
            .iter()
            .all(|&value| value == -3.5),
        "rows before the slice were overwritten"
    );
    assert!(
        shifted_out[(offset + tokens) * V_ELEMS..]
            .iter()
            .all(|&value| value == -3.5),
        "rows after the slice were overwritten"
    );
}

#[test]
fn delta_preprocessor_matches_cpu_and_updates_only_selected_conv_slots() {
    let batch = 2;
    let capacity = 4;
    let slots = vec![3u32, 1];
    let mixed: Vec<u16> = (0..batch * LA_CONV_CHANNELS)
        .map(|index| bf16::from_f32((index % 43) as f32 * 0.02 - 0.42))
        .collect();
    let conv_weight: Vec<u16> = (0..LA_CONV_CHANNELS * 4)
        .map(|index| bf16::from_f32((index % 9) as f32 * 0.025 - 0.1))
        .collect();
    let a_projection: Vec<u16> = (0..batch * GATE_ELEMS)
        .map(|index| bf16::from_f32((index % 17) as f32 * 0.05 - 0.4))
        .collect();
    let b_projection: Vec<u16> = (0..batch * GATE_ELEMS)
        .map(|index| bf16::from_f32((index % 13) as f32 * 0.07 - 0.35))
        .collect();
    let a_log: Vec<u16> = (0..GATE_ELEMS)
        .map(|index| bf16::from_f32((0.2 + (index % 7) as f32 * 0.03).ln()))
        .collect();
    let dt_bias: Vec<u16> = (0..GATE_ELEMS)
        .map(|index| bf16::from_f32((index % 11) as f32 * 0.04 - 0.2))
        .collect();
    let initial_state: Vec<f32> = (0..capacity * CONV_STATE_ELEMS)
        .map(|index| (index % 19) as f32 * 0.01 - 0.09)
        .collect();

    let mut expected_state = initial_state.clone();
    let mut q = vec![0.0f32; batch * QK_ELEMS];
    let mut k = vec![0.0f32; batch * QK_ELEMS];
    let mut v = vec![0.0f32; batch * V_ELEMS];
    for sequence in 0..batch {
        for channel in 0..LA_CONV_CHANNELS {
            let state_base = (slots[sequence] as usize * LA_CONV_CHANNELS + channel) * 3;
            let weight_base = channel * 4;
            let current = bf16::to_f32(mixed[sequence * LA_CONV_CHANNELS + channel]);
            let mut convolved = bf16::to_f32(conv_weight[weight_base]) * expected_state[state_base];
            for tap in 1..3 {
                convolved +=
                    bf16::to_f32(conv_weight[weight_base + tap]) * expected_state[state_base + tap];
            }
            convolved += bf16::to_f32(conv_weight[weight_base + 3]) * current;
            expected_state[state_base] = expected_state[state_base + 1];
            expected_state[state_base + 1] = expected_state[state_base + 2];
            expected_state[state_base + 2] = current;
            let activated = bf16::to_f32(bf16::from_f32(convolved / (1.0 + (-convolved).exp())));
            if channel < QK_ELEMS {
                q[sequence * QK_ELEMS + channel] = activated;
            } else if channel < 2 * QK_ELEMS {
                k[sequence * QK_ELEMS + channel - QK_ELEMS] = activated;
            } else {
                v[sequence * V_ELEMS + channel - 2 * QK_ELEMS] = activated;
            }
        }
    }
    for sequence in 0..batch {
        for head in 0..LA_NUM_K_HEADS {
            let base = (sequence * LA_NUM_K_HEADS + head) * LA_K_HEAD_DIM;
            let q_inverse = 1.0
                / (q[base..base + LA_K_HEAD_DIM]
                    .iter()
                    .map(|value| value * value)
                    .sum::<f32>()
                    + 1e-6)
                    .sqrt()
                / (LA_K_HEAD_DIM as f32).sqrt();
            let k_inverse = 1.0
                / (k[base..base + LA_K_HEAD_DIM]
                    .iter()
                    .map(|value| value * value)
                    .sum::<f32>()
                    + 1e-6)
                    .sqrt();
            for dimension in 0..LA_K_HEAD_DIM {
                q[base + dimension] *= q_inverse;
                k[base + dimension] *= k_inverse;
            }
        }
    }
    let mut alpha = vec![0.0f32; batch * GATE_ELEMS];
    let mut beta = vec![0.0f32; batch * GATE_ELEMS];
    for index in 0..batch * GATE_ELEMS {
        let head = index % GATE_ELEMS;
        let a = bf16::to_f32(a_projection[index]) + bf16::to_f32(dt_bias[head]);
        let softplus = (1.0 + a.exp()).ln();
        alpha[index] = (-bf16::to_f32(a_log[head]).exp() * softplus).exp();
        let b = bf16::to_f32(b_projection[index]);
        beta[index] = 1.0 / (1.0 + (-b).exp());
    }

    let stream = Stream::new().unwrap();
    let preprocessor = DeltaPreprocessor::from_host(&conv_weight, &a_log, &dt_bias).unwrap();
    let device_mixed = DeviceBuffer::from_slice(&mixed).unwrap();
    let device_a = DeviceBuffer::from_slice(&a_projection).unwrap();
    let device_b = DeviceBuffer::from_slice(&b_projection).unwrap();
    let mut device_state = DeviceBuffer::from_slice(&initial_state).unwrap();
    let device_slots = DeviceBuffer::from_slice(&slots).unwrap();
    let mut prepared = PreparedDelta::zeroed(batch).unwrap();
    preprocessor
        .prepare_decode(
            RowView::packed(&device_mixed, LA_CONV_CHANNELS),
            RowView::packed(&device_a, GATE_ELEMS),
            RowView::packed(&device_b, GATE_ELEMS),
            &mut device_state,
            &device_slots,
            &mut prepared,
            capacity,
            batch,
            &stream,
        )
        .unwrap();
    stream.synchronize().unwrap();

    let expected_kq = kq_rows(&q, &k);
    for (name, actual, expected, tolerance) in [
        ("q", prepared.q.to_vec().unwrap(), q, 2e-5f32),
        ("k", prepared.k.to_vec().unwrap(), k, 2e-4),
        ("v", prepared.v.to_vec().unwrap(), v, 2e-5),
        ("alpha", prepared.alpha.to_vec().unwrap(), alpha, 2e-4),
        ("beta", prepared.beta.to_vec().unwrap(), beta, 2e-4),
        // Скан больше не считает k.q сам, а читает его отсюда: если prepare
        // посчитает не то, разойдётся весь линейный слой, а не одна фаза.
        ("kq", prepared.kq.to_vec().unwrap(), expected_kq, 2e-5),
    ] {
        for (index, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "{name}[{index}]: GPU={actual}, CPU={expected}"
            );
        }
    }
    let state = device_state.to_vec().unwrap();
    for (index, (actual, expected)) in state.iter().zip(&expected_state).enumerate() {
        assert!(
            (actual - expected).abs() <= 2e-7,
            "conv_state[{index}]: GPU={actual}, CPU={expected}"
        );
    }
}

#[test]
fn gated_rmsnorm_matches_qwen_dtype_boundaries() {
    let batch = 2;
    let input: Vec<f32> = (0..batch * V_ELEMS)
        .map(|index| (index % 47) as f32 * 0.03 - 0.7)
        .collect();
    let gate: Vec<u16> = (0..batch * V_ELEMS)
        .map(|index| bf16::from_f32((index % 19) as f32 * 0.08 - 0.7))
        .collect();
    let weight: Vec<u16> = (0..LA_V_HEAD_DIM)
        .map(|index| bf16::from_f32(0.9 + (index % 13) as f32 * 0.015))
        .collect();
    let mut expected = vec![0u16; input.len()];
    for head in 0..batch * GATE_ELEMS {
        let base = head * LA_V_HEAD_DIM;
        let rounded: Vec<f32> = input[base..base + LA_V_HEAD_DIM]
            .iter()
            .map(|&value| bf16::to_f32(bf16::from_f32(value)))
            .collect();
        let inverse = 1.0
            / (rounded.iter().map(|value| value * value).sum::<f32>() / LA_V_HEAD_DIM as f32
                + 1e-6)
                .sqrt();
        for dimension in 0..LA_V_HEAD_DIM {
            let normalized = bf16::from_f32(rounded[dimension] * inverse);
            let weighted =
                bf16::from_f32(bf16::to_f32(weight[dimension]) * bf16::to_f32(normalized));
            let z = bf16::to_f32(gate[base + dimension]);
            expected[base + dimension] =
                bf16::from_f32(bf16::to_f32(weighted) * z / (1.0 + (-z).exp()));
        }
    }

    let stream = Stream::new().unwrap();
    let norm = DeltaOutputNorm::from_host(&weight, 1e-6).unwrap();
    let device_input = DeviceBuffer::from_slice(&input).unwrap();
    let device_gate = DeviceBuffer::from_slice(&gate).unwrap();
    let mut output = DeviceBuffer::<u16>::zeroed(input.len()).unwrap();
    norm.forward(
        &device_input,
        RowView::packed(&device_gate, V_ELEMS),
        &mut output,
        batch,
        &stream,
    )
        .unwrap();
    stream.synchronize().unwrap();
    let actual = output.to_vec().unwrap();
    let mut worst = 0.0f32;
    for (&actual, &expected) in actual.iter().zip(&expected) {
        worst = worst.max((bf16::to_f32(actual) - bf16::to_f32(expected)).abs());
    }
    assert!(worst <= 0.008, "gated RMSNorm max error {worst}");
}

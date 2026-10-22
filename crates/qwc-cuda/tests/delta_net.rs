//! Checks the Gated DeltaNet GPU kernel against a CPU reference.
//! Requires an RTX 5090.

use qwc_core::arch::{LA_CONV_CHANNELS, LA_K_HEAD_DIM, LA_NUM_K_HEADS, LA_V_HEAD_DIM};
use qwc_cuda::delta_net::{
    self, CONV_STATE_ELEMS, DeltaInputs, DeltaOutputNorm, DeltaPreprocessor, DeltaStateMode,
    GATE_ELEMS, PreparedDelta, QK_ELEMS, RowView, STATE_ELEMS, V_ELEMS,
};
use qwc_cuda::{DeviceBuffer, Stream, bf16};

/// A deterministic generator so failures are reproducible.
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

/// k.q for every row and k head — exactly what prepare computes on the GPU.
/// The scan reads this scalar instead of recomputing it in every warp.
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
    // alpha is decay in (0,1), beta is write strength in (0,1).
    let alpha: Vec<f32> = (0..batch * GATE_ELEMS)
        .map(|_| 0.9 + rng.next_f32() * 0.09)
        .collect();
    let beta: Vec<f32> = (0..batch * GATE_ELEMS)
        .map(|_| 0.5 + rng.next_f32() * 0.4)
        .collect();

    // Reference.
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

    // The output is computed in fp32, a mismatch is only possible from summation order.
    let mut worst_out = 0.0f32;
    for (g, r) in gpu_out.iter().zip(&ref_out) {
        let d = (g - r).abs() / r.abs().max(1e-3);
        worst_out = worst_out.max(d);
    }
    assert!(worst_out < 2e-3, "output diverges by {worst_out:.2e}");

    // The state is stored in bf16: we allow a one-ULP mismatch (~0.8%),
    // which occurs when the fp32 result lands on a rounding boundary.
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
        "state diverges by {worst_state:.2e}"
    );
    assert!(
        frac < 0.02,
        "too many mismatches in the state: {:.2}%",
        frac * 100.0
    );
}

#[test]
fn decay_shrinks_state_when_write_disabled() {
    // With beta = 0 the write is disabled and the state must simply decay by alpha.
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
        assert_eq!(*now, expect, "decay applied incorrectly");
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
    let worst_state = actual_state
        .iter()
        .zip(&expected_state)
        .map(|(&actual, &expected)| (bf16::to_f32(actual) - bf16::to_f32(expected)).abs())
        .fold(0.0f32, f32::max);
    assert!(
        differing as f64 / (STATE_ELEMS as f64) < 0.02,
        "too many BF16 state differences: {differing}/{STATE_ELEMS}, worst abs {worst_state:.2e}"
    );
    assert_eq!(&actual_pool[..STATE_ELEMS], &initial[..STATE_ELEMS]);
}

/// FP32 mode of the chunk scan: the state stays in registers until the end of the chunk.
/// Checked against a separate CPU reference and must differ from the BF16 mode,
/// otherwise the switch isolates nothing.
#[test]
fn chunk_prefill_fp32_state_matches_fp32_reference_and_differs_from_bf16() {
    // 65 crosses the usual 64-token boundary and catches state carry bugs.
    let tokens = 65;
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
    let mut worst_abs = 0.0f32;
    let mut worst_index = 0usize;
    for (index, (&actual, &expected)) in fp32_out.iter().zip(&expected_out).enumerate() {
        let relative = (actual - expected).abs() / expected.abs().max(1e-3);
        if relative > worst {
            worst = relative;
            worst_abs = (actual - expected).abs();
            worst_index = index;
        }
    }
    assert!(
        fp32_out
            .iter()
            .zip(&expected_out)
            .all(|(&actual, &expected)| (actual - expected).abs() <= 5e-4 + 2e-3 * expected.abs()),
        "fp32 prefill output differs by {worst:.2e} (abs {worst_abs:.2e}) at {worst_index}: gpu={}, cpu={}",
        fp32_out[worst_index],
        expected_out[worst_index],
    );

    let actual_state = &fp32_pool[slot * STATE_ELEMS..][..STATE_ELEMS];
    let differing = actual_state
        .iter()
        .zip(&expected_state)
        .filter(|(actual, expected)| actual != expected)
        .count();
    let mut state_worst_abs = 0.0f32;
    let mut state_squared_error = 0.0f64;
    for (&actual, &expected) in actual_state.iter().zip(&expected_state) {
        let error = (bf16::to_f32(actual) - bf16::to_f32(expected)).abs();
        state_worst_abs = state_worst_abs.max(error);
        state_squared_error += f64::from(error) * f64::from(error);
    }
    let state_rmse = (state_squared_error / STATE_ELEMS as f64).sqrt();
    assert!(
        differing as f64 / (STATE_ELEMS as f64) < 0.02,
        "too many BF16 state differences: {differing}/{STATE_ELEMS}; max abs {state_worst_abs:.3e}, RMSE {state_rmse:.3e}"
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
        // The scan no longer computes k.q itself but reads it from here: if prepare
        // computes the wrong thing, the whole linear layer diverges, not one phase.
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

/// The chunked (WY) form of the scan vs the recurrent one — both on CPU, both in fp32.
///
/// This checks the derivation, not the kernel: if the forms diverge here, there's nothing
/// to argue about with the GPU. The length is not a multiple of the chunk so the tail is computed
/// by the same code, and alpha is far from one — where gamma ratios between
/// distant tokens go to zero.
#[test]
fn chunked_wy_form_matches_the_recurrent_scan() {
    use qwc_core::arch::{LA_NUM_V_HEADS, LA_V_HEAD_DIM as DV};
    const DK: usize = LA_K_HEAD_DIM;
    const HV: usize = LA_NUM_V_HEADS;

    for (tokens, chunk, decay) in [(130usize, 64usize, 0.99f32), (64, 64, 0.9), (33, 16, 0.7)] {
        let mut rng = Rng(0x5ca_4_4a_7e5_7);
        let q: Vec<f32> = (0..tokens * LA_NUM_K_HEADS * DK)
            .map(|_| rng.next_f32() * 0.5)
            .collect();
        let k: Vec<f32> = (0..tokens * LA_NUM_K_HEADS * DK)
            .map(|_| rng.next_f32() * 0.5)
            .collect();
        let v: Vec<f32> = (0..tokens * HV * DV).map(|_| rng.next_f32()).collect();
        // alpha in (0, 1], beta in (0, 2) — as after sigmoid and softplus.
        let alpha: Vec<f32> = (0..tokens * HV)
            .map(|_| decay * (0.5 + 0.5 * (rng.next_f32() * 0.5 + 0.5)))
            .collect();
        let beta: Vec<f32> = (0..tokens * HV)
            .map(|_| rng.next_f32() * 0.4 + 0.6)
            .collect();
        let initial: Vec<u16> = (0..STATE_ELEMS)
            .map(|_| bf16::from_f32(rng.next_f32() * 0.1))
            .collect();

        let mut recurrent_state = initial.clone();
        let mut recurrent_out = vec![0.0f32; tokens * HV * DV];
        delta_net::reference::prefill_chunk_fp32(
            &mut recurrent_state,
            &q,
            &k,
            &v,
            &alpha,
            &beta,
            &mut recurrent_out,
            tokens,
        );

        let mut chunked_state = initial.clone();
        let mut chunked_out = vec![0.0f32; tokens * HV * DV];
        delta_net::reference::prefill_chunked_fp32(
            &mut chunked_state,
            &q,
            &k,
            &v,
            &alpha,
            &beta,
            &mut chunked_out,
            tokens,
            chunk,
        );

        let mut worst = 0.0f32;
        for (index, (&got, &want)) in chunked_out.iter().zip(&recurrent_out).enumerate() {
            let tolerance = 1e-3 + want.abs() * 2e-3;
            worst = worst.max((got - want).abs());
            assert!(
                (got - want).abs() <= tolerance,
                "tokens {tokens}, chunk {chunk}: output {index}: WY={got}, recurrent={want}"
            );
        }
        for (index, (&got, &want)) in chunked_state.iter().zip(&recurrent_state).enumerate() {
            let (got, want) = (bf16::to_f32(got), bf16::to_f32(want));
            let tolerance = 1e-2 + want.abs() * 1e-2;
            assert!(
                (got - want).abs() <= tolerance,
                "tokens {tokens}, chunk {chunk}: state {index}: WY={got}, recurrent={want}"
            );
        }
        println!("tokens {tokens}, chunk {chunk}, decay {decay}: worst output {worst:.3e}");
    }
}

/// The WY scan on GPU vs the chunked reference on CPU.
///
/// The reference here is specifically chunked: `prefill_chunked_fp32` with the same chunk
/// `WY_CHUNK_SIZE`, i.e. with the same order of operations. The divergence from the
/// recurrent form is checked separately on CPU
/// (`chunked_wy_form_matches_the_recurrent_scan`) — mixing the two checks
/// means not knowing what exactly broke.
///
/// Lengths are taken around the chunk boundary, and decay down to 0.05 per token.
/// Weak decay checks nothing: with alpha near one any form of
/// gamma accumulation converges. Everything breaks on strong decay, where the product of
/// alpha over a chunk underflows fp32.
#[test]
fn wy_prefill_matches_the_chunked_reference() {
    for (tokens, decay) in [
        (130usize, 0.94f32),
        (64, 0.5),
        (33, 0.15),
        (512, 0.05),
        (1, 0.9),
    ] {
        check_wy_prefill(tokens, decay);
    }
}

fn check_wy_prefill(tokens: usize, decay: f32) {
    use qwc_cuda::delta_net::{DeltaPrefillWorkspace, WY_CHUNK_SIZE};

    let capacity = 2;
    let slot = 1;
    let mut rng = Rng(0x000D_E17A_5CA4);
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
        .map(|_| decay + rng.next_f32() * 0.02 * decay)
        .collect();
    let beta: Vec<f32> = (0..tokens * GATE_ELEMS)
        .map(|_| 0.45 + rng.next_f32() * 0.2)
        .collect();

    let mut expected_state = initial[slot * STATE_ELEMS..][..STATE_ELEMS].to_vec();
    let mut expected_out = vec![0.0f32; tokens * V_ELEMS];
    delta_net::reference::prefill_chunked_fp32(
        &mut expected_state,
        &q,
        &k,
        &v,
        &alpha,
        &beta,
        &mut expected_out,
        tokens,
        WY_CHUNK_SIZE,
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
    let mut workspace = DeltaPrefillWorkspace::new().unwrap();
    let mut device_state = DeviceBuffer::from_slice(&initial).unwrap();
    let mut output = DeviceBuffer::<f32>::zeroed(tokens * V_ELEMS).unwrap();
    delta_net::prefill_slot_wy(
        &mut device_state,
        &inputs,
        &mut output,
        &mut workspace,
        capacity,
        slot,
        tokens,
        0,
        &stream,
    )
    .unwrap();
    stream.synchronize().unwrap();

    let actual_out = output.to_vec().unwrap();
    let pool = device_state.to_vec().unwrap();
    let actual_state = &pool[slot * STATE_ELEMS..][..STATE_ELEMS];

    // The reference computes in FP32, the kernel puts q, k and the state in BF16 for MMA,
    // so element-wise relative error is meaningless: the output has
    // values near zero where it's arbitrarily large. The threshold is absolute:
    // inputs are normalized (|q| = |k| = 1, v of order 0.2), and on all checked
    // shapes the error stays around 4e-4 — that's the BF16 floor, not a divergence
    // of algorithms. Bugs caught here before were 200 times
    // larger: a wrong form gave 9.9e-2, NaN from a zero in gamma — the same.
    let scale = expected_out.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    let mut worst = 0.0f32;
    let mut worst_index = 0usize;
    let mut squared = 0.0f64;
    for (index, (&actual, &expected)) in actual_out.iter().zip(&expected_out).enumerate() {
        let error = (actual - expected).abs();
        squared += f64::from(error) * f64::from(error);
        if error > worst {
            worst = error;
            worst_index = index;
        }
    }
    let rmse = (squared / actual_out.len() as f64).sqrt();
    assert!(
        worst <= 8e-4 && rmse <= 1.5e-4,
        "WY output diverges from the reference by {worst:.2e} (RMSE {rmse:.2e}) \
         at scale {scale:.2e} \
         at position {worst_index}: token {}, head {}, row {}: gpu={}, cpu={}",
        worst_index / V_ELEMS,
        (worst_index % V_ELEMS) / LA_V_HEAD_DIM,
        worst_index % LA_V_HEAD_DIM,
        actual_out[worst_index],
        expected_out[worst_index],
    );

    let state_scale = expected_state
        .iter()
        .fold(0.0f32, |a, &b| a.max(bf16::to_f32(b).abs()));
    let mut state_worst = 0.0f32;
    for (&actual, &expected) in actual_state.iter().zip(&expected_state) {
        let (actual, expected) = (bf16::to_f32(actual), bf16::to_f32(expected));
        state_worst = state_worst.max((actual - expected).abs());
    }
    assert!(
        state_worst <= 1e-3,
        "WY state diverges from the reference by {state_worst:.2e} at scale {state_scale:.2e}"
    );
    assert_eq!(&pool[..STATE_ELEMS], &initial[..STATE_ELEMS]);
    println!(
        "tokens {tokens}, decay {decay}: output max {worst:.3e}, \
         RMSE {rmse:.3e} at scale {scale:.3e}; \
         state max {state_worst:.3e} at scale {state_scale:.3e}"
    );
}

/// The fused step puts several sequences into one token arena, and
/// the WY scan must read and write only its own span. The same shape that
/// once hid a wrong attention prefill: nothing was checked with an offset,
/// because nothing was tested with an offset.
#[test]
fn wy_prefill_row_offset_reads_and_writes_only_its_slice() {
    use qwc_cuda::delta_net::{DeltaPrefillWorkspace, KQ_ELEMS};

    let tokens = 70;
    let offset = 37;
    let arena = tokens + offset + 5;
    let mut rng = Rng(0x0FF5_E700);
    let initial: Vec<u16> = (0..STATE_ELEMS)
        .map(|_| bf16::from_f32(rng.next_f32() * 0.05))
        .collect();
    let mut q: Vec<f32> = (0..arena * QK_ELEMS).map(|_| rng.next_f32()).collect();
    let mut k: Vec<f32> = (0..arena * QK_ELEMS).map(|_| rng.next_f32()).collect();
    l2_normalize_heads(&mut q, LA_NUM_K_HEADS, LA_K_HEAD_DIM);
    l2_normalize_heads(&mut k, LA_NUM_K_HEADS, LA_K_HEAD_DIM);
    let v: Vec<f32> = (0..arena * V_ELEMS).map(|_| rng.next_f32() * 0.2).collect();
    let alpha: Vec<f32> = (0..arena * GATE_ELEMS)
        .map(|_| 0.6 + rng.next_f32() * 0.2)
        .collect();
    let beta: Vec<f32> = (0..arena * GATE_ELEMS)
        .map(|_| 0.45 + rng.next_f32() * 0.2)
        .collect();

    let stream = Stream::new().unwrap();
    let kq = kq_rows(&q, &k);
    let mut workspace = DeltaPrefillWorkspace::new().unwrap();

    // The same span, fed from zero offset, is the reference for the shifted one.
    let mut run = |row_offset: usize, rows: usize| {
        let slice =
            |data: &[f32], width: usize| data[row_offset * width..][..rows * width].to_vec();
        let device_q = DeviceBuffer::from_slice(&slice(&q, QK_ELEMS)).unwrap();
        let device_k = DeviceBuffer::from_slice(&slice(&k, QK_ELEMS)).unwrap();
        let device_kq = DeviceBuffer::from_slice(&slice(&kq, KQ_ELEMS)).unwrap();
        let device_v = DeviceBuffer::from_slice(&slice(&v, V_ELEMS)).unwrap();
        let device_alpha = DeviceBuffer::from_slice(&slice(&alpha, GATE_ELEMS)).unwrap();
        let device_beta = DeviceBuffer::from_slice(&slice(&beta, GATE_ELEMS)).unwrap();
        let inputs = DeltaInputs {
            q: &device_q,
            k: &device_k,
            v: &device_v,
            alpha: &device_alpha,
            beta: &device_beta,
            kq: &device_kq,
        };
        let mut state = DeviceBuffer::from_slice(&initial).unwrap();
        let mut output = DeviceBuffer::<f32>::zeroed(rows * V_ELEMS).unwrap();
        delta_net::prefill_slot_wy(
            &mut state,
            &inputs,
            &mut output,
            &mut workspace,
            1,
            0,
            rows,
            0,
            &stream,
        )
        .unwrap();
        stream.synchronize().unwrap();
        (output.to_vec().unwrap(), state.to_vec().unwrap())
    };
    let (expected_out, expected_state) = run(offset, tokens);

    let device_q = DeviceBuffer::from_slice(&q).unwrap();
    let device_k = DeviceBuffer::from_slice(&k).unwrap();
    let device_kq = DeviceBuffer::from_slice(&kq).unwrap();
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
    let sentinel = -7.5f32;
    let mut state = DeviceBuffer::from_slice(&initial).unwrap();
    let mut output = DeviceBuffer::from_slice(&vec![sentinel; arena * V_ELEMS]).unwrap();
    delta_net::prefill_slot_wy(
        &mut state,
        &inputs,
        &mut output,
        &mut workspace,
        1,
        0,
        tokens,
        offset,
        &stream,
    )
    .unwrap();
    stream.synchronize().unwrap();

    let actual = output.to_vec().unwrap();
    for (index, (&got, &want)) in actual[offset * V_ELEMS..][..tokens * V_ELEMS]
        .iter()
        .zip(&expected_out)
        .enumerate()
    {
        assert_eq!(
            got, want,
            "the shifted run diverged from the unshifted one at position {index}"
        );
    }
    assert!(
        actual[..offset * V_ELEMS].iter().all(|&x| x == sentinel),
        "the scan wrote rows before its span"
    );
    assert!(
        actual[(offset + tokens) * V_ELEMS..]
            .iter()
            .all(|&x| x == sentinel),
        "the scan wrote rows after its span"
    );
    assert_eq!(state.to_vec().unwrap(), expected_state);
}

/// Per-row int8 must match the CPU reference element-wise and be
/// noticeably more accurate than e4m3 at the same byte: state rows are flat, and they don't need four
/// exponent bits. This property is the only reason to choose
/// int8, so it's checked rather than taken on faith.
#[test]
fn state_requantize_matches_host_and_beats_e4m3() {
    const SLOTS: usize = 2;
    let mut rng = Rng(0x51ed_2701);
    // The per-row scale differs: otherwise per-head and per-row scales are
    // indistinguishable and the test doesn't check that the scale is really per row.
    let mut original = vec![0.0f32; SLOTS * STATE_ELEMS];
    for (row, chunk) in original.chunks_mut(LA_K_HEAD_DIM).enumerate() {
        let scale = 2.0f32.powi((row % 7) as i32 - 3);
        for x in chunk.iter_mut() {
            *x = bf16::to_f32(bf16::from_f32(rng.next_f32() * scale));
        }
    }

    let stream = Stream::new().unwrap();
    let host: Vec<u16> = original.iter().map(|&x| bf16::from_f32(x)).collect();

    let mut errors = Vec::new();
    for quant in [delta_net::StateQuant::Int8, delta_net::StateQuant::E4m3] {
        let mut state = DeviceBuffer::from_slice(&host).unwrap();
        delta_net::requantize_state_slot(&mut state, quant, SLOTS, 1, &stream).unwrap();
        stream.synchronize().unwrap();
        let got: Vec<f32> = state
            .to_vec()
            .unwrap()
            .iter()
            .map(|&x| bf16::to_f32(x))
            .collect();

        // The kernel doesn't touch slot 0.
        assert_eq!(
            &got[..STATE_ELEMS],
            &original[..STATE_ELEMS],
            "{} touched another slot",
            quant.as_str()
        );

        let slot = &original[STATE_ELEMS..];
        let after = &got[STATE_ELEMS..];
        let mut worst = 0.0f32;
        let mut numerator = 0.0f64;
        let mut denominator = 0.0f64;
        for (row, (before, now)) in slot
            .chunks(LA_K_HEAD_DIM)
            .zip(after.chunks(LA_K_HEAD_DIM))
            .enumerate()
        {
            let peak = before.iter().fold(0.0f32, |m, x| m.max(x.abs()));
            for (index, (&x, &y)) in before.iter().zip(now).enumerate() {
                let expected = if quant == delta_net::StateQuant::Int8 {
                    // The arithmetic repeats the kernel operation by operation: it multiplies
                    // by the inverse step and rounds to even, and at the half step
                    // division with round-half-away-from-zero gives a different step.
                    let step = peak * (1.0 / 127.0);
                    let inverse = 1.0 / step;
                    let level = (x * inverse).round_ties_even().clamp(-127.0, 127.0);
                    bf16::to_f32(bf16::from_f32(level * step))
                } else {
                    y // e4m3 is checked by error magnitude, not element-wise
                };
                assert!(
                    (y - expected).abs() <= 1.0e-6 * peak.max(1.0e-6),
                    "{} row {row} element {index}: {y} vs {expected}",
                    quant.as_str()
                );
                worst = worst.max((y - x).abs() / peak.max(1.0e-9));
                numerator += ((y - x) as f64).powi(2);
                denominator += (x as f64).powi(2);
            }
        }
        let relative = (numerator / denominator).sqrt() as f32;
        // The grid step is peak/127 for int8 and up to peak/8 in the top binade for e4m3.
        // On top of that comes the rounding of writing back to bf16 (2^-9 of
        // magnitude): real storage dequantizes into fp32 registers and doesn't have
        // this extra, so a measurement through the kernel is an upper bound.
        const BF16_WRITEBACK: f32 = 1.0 / 512.0;
        let ceiling = BF16_WRITEBACK
            + if quant == delta_net::StateQuant::Int8 {
                0.5 / 127.0
            } else {
                0.5 / 8.0
            };
        assert!(
            worst <= ceiling * 1.01,
            "{}: element drifted by {worst} fractions of peak with a ceiling of {ceiling}",
            quant.as_str()
        );
        errors.push(relative);
    }

    let (int8, e4m3) = (errors[0], errors[1]);
    assert!(
        e4m3 > int8 * 2.5,
        "int8 must be noticeably more accurate than e4m3: {int8:.3e} vs {e4m3:.3e}"
    );
    eprintln!("state: int8 {int8:.3e}, e4m3 {e4m3:.3e} relative error");
}

/// Packing and unpacking a slot must agree: the unpacked
/// state is exactly what the 8-bit decode will read. A mismatch
/// between these two paths would mean the prompt and its continuation see
/// different states, and checking the output of one step doesn't catch that.
#[test]
fn packed_state_round_trip_matches_the_grid() {
    const CAPACITY: usize = 2;
    let mut rng = Rng(0x7f01_2ab4);
    let mut original = vec![0.0f32; STATE_ELEMS];
    for (row, chunk) in original.chunks_mut(LA_K_HEAD_DIM).enumerate() {
        let scale = 2.0f32.powi((row % 5) as i32 - 2);
        for x in chunk.iter_mut() {
            *x = bf16::to_f32(bf16::from_f32(rng.next_f32() * scale));
        }
    }

    let stream = Stream::new().unwrap();
    let host: Vec<u16> = original.iter().map(|&x| bf16::from_f32(x)).collect();
    let source = DeviceBuffer::from_slice(&host).unwrap();
    let mut pool = delta_net::PackedStatePool::zeroed(CAPACITY).unwrap();
    let mut restored = DeviceBuffer::<u16>::zeroed(STATE_ELEMS).unwrap();

    delta_net::pack_state_slot(&source, 1, 0, &mut pool, 1, &stream).unwrap();
    delta_net::unpack_state_slot(&pool, 1, &mut restored, 1, 0, &stream).unwrap();
    stream.synchronize().unwrap();

    let got: Vec<f32> = restored
        .to_vec()
        .unwrap()
        .iter()
        .map(|&x| bf16::to_f32(x))
        .collect();

    let mut worst = 0.0f32;
    for (before, now) in original
        .chunks(LA_K_HEAD_DIM)
        .zip(got.chunks(LA_K_HEAD_DIM))
    {
        let peak = before.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        let step = peak * (1.0 / 127.0);
        let inverse = 1.0 / step;
        for (&x, &y) in before.iter().zip(now) {
            let level = (x * inverse).round_ties_even().clamp(-127.0, 127.0);
            let expected = bf16::to_f32(bf16::from_f32(level * step));
            assert!(
                (y - expected).abs() <= 1.0e-6 * peak.max(1.0e-6),
                "unpacked {y} vs expected {expected}"
            );
            worst = worst.max((y - x).abs() / peak.max(1.0e-9));
        }
    }
    // Grid step plus the rounding of unpacking to bf16.
    assert!(
        worst <= 0.5 / 127.0 + 1.0 / 512.0,
        "element drifted by {worst}"
    );

    // The pool's zero slot is untouched, and zeros are representable without a scale.
    let mut zeroed = DeviceBuffer::<u16>::zeroed(STATE_ELEMS).unwrap();
    delta_net::unpack_state_slot(&pool, 0, &mut zeroed, 1, 0, &stream).unwrap();
    stream.synchronize().unwrap();
    assert!(zeroed.to_vec().unwrap().iter().all(|&x| x == 0));
}

/// The 8-bit decode must match the bf16 path whose state is
/// snapped onto the same grid: they can diverge only in the step math, and
/// it's the same function for both.
#[test]
fn packed_decode_matches_bf16_decode_on_the_same_grid() {
    const CAPACITY: usize = 2;
    const SLOT: usize = 1;
    const STEPS: usize = 6;
    let mut rng = Rng(0x2c8e_9917);

    let mut initial = vec![0.0f32; STATE_ELEMS];
    for (row, chunk) in initial.chunks_mut(LA_K_HEAD_DIM).enumerate() {
        let scale = 2.0f32.powi((row % 5) as i32 - 2);
        for x in chunk.iter_mut() {
            *x = rng.next_f32() * scale;
        }
    }
    let host: Vec<u16> = initial.iter().map(|&x| bf16::from_f32(x)).collect();

    let stream = Stream::new().unwrap();
    let slots = DeviceBuffer::from_slice(&[SLOT as u32]).unwrap();

    let source = DeviceBuffer::from_slice(&host).unwrap();
    let mut pool = delta_net::PackedStatePool::zeroed(CAPACITY).unwrap();
    delta_net::pack_state_slot(&source, 1, 0, &mut pool, SLOT, &stream).unwrap();

    // The bf16 path starts from the unpacked state, otherwise the very first step
    // would diverge on the packing itself, not on the math.
    let mut wide = DeviceBuffer::<u16>::zeroed(CAPACITY * STATE_ELEMS).unwrap();
    delta_net::unpack_state_slot(&pool, SLOT, &mut wide, CAPACITY, SLOT, &stream).unwrap();

    let mut prepared = PreparedDelta::zeroed(1).unwrap();
    let mut packed_out = DeviceBuffer::<f32>::zeroed(V_ELEMS).unwrap();
    let mut wide_out = DeviceBuffer::<f32>::zeroed(V_ELEMS).unwrap();

    let mut worst = 0.0f32;
    for step in 0..STEPS {
        let mut q: Vec<f32> = (0..QK_ELEMS).map(|_| rng.next_f32()).collect();
        let mut k: Vec<f32> = (0..QK_ELEMS).map(|_| rng.next_f32()).collect();
        l2_normalize_heads(&mut q, LA_NUM_K_HEADS, LA_K_HEAD_DIM);
        l2_normalize_heads(&mut k, LA_NUM_K_HEADS, LA_K_HEAD_DIM);
        let v: Vec<f32> = (0..V_ELEMS).map(|_| rng.next_f32()).collect();
        let alpha: Vec<f32> = (0..GATE_ELEMS)
            .map(|_| 0.5 + 0.49 * rng.next_f32())
            .collect();
        let beta: Vec<f32> = (0..GATE_ELEMS)
            .map(|_| 0.5 + 0.49 * rng.next_f32())
            .collect();
        let kq = kq_rows(&q, &k);

        prepared.q.copy_from_slice(&q).unwrap();
        prepared.k.copy_from_slice(&k).unwrap();
        prepared.v.copy_from_slice(&v).unwrap();
        prepared.alpha.copy_from_slice(&alpha).unwrap();
        prepared.beta.copy_from_slice(&beta).unwrap();
        prepared.kq.copy_from_slice(&kq).unwrap();

        delta_net::decode_slots_packed(
            &mut pool,
            &slots,
            &prepared.inputs(),
            &mut packed_out,
            1,
            &stream,
        )
        .unwrap();
        delta_net::decode_slots(
            &mut wide,
            &slots,
            &prepared.inputs(),
            &mut wide_out,
            CAPACITY,
            1,
            &stream,
        )
        .unwrap();
        // The bf16 path keeps the state in bf16, so after every step it
        // must be put back onto the same 8-bit grid — otherwise different
        // storages are compared, not different kernels.
        delta_net::requantize_state_slots(
            &mut wide,
            &slots,
            delta_net::StateQuant::Int8,
            CAPACITY,
            1,
            &stream,
        )
        .unwrap();
        stream.synchronize().unwrap();

        let a = packed_out.to_vec().unwrap();
        let b = wide_out.to_vec().unwrap();
        let magnitude = b.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1.0e-6);
        for (x, y) in a.iter().zip(&b) {
            worst = worst.max((x - y).abs() / magnitude);
        }
        assert!(
            worst < 5.0e-2,
            "step {step}: 8-bit decode diverged from bf16 by {worst}"
        );
    }
    eprintln!("8-bit decode vs bf16 on the same grid: {worst:.2e}");
}

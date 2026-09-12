//! Correctness for standalone and fused RMSNorm decode paths.

use qwc_cuda::nvfp4::{self, QuantizedActivation};
use qwc_cuda::rmsnorm::{self, RmsNorm};
use qwc_cuda::{DeviceBuffer, Stream, bf16};

struct Rng(u64);

impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as u32 as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
}

#[test]
fn bf16_rmsnorm_matches_cpu_with_and_without_residual() {
    const BATCH: usize = 3;
    const HIDDEN: usize = 512;
    const EPSILON: f32 = 1e-6;
    let mut rng = Rng(0x0509_0512);
    let input: Vec<u16> = (0..BATCH * HIDDEN)
        .map(|_| bf16::from_f32(rng.next_f32() * 2.0))
        .collect();
    let original_residual: Vec<u16> = (0..BATCH * HIDDEN)
        .map(|_| bf16::from_f32(rng.next_f32()))
        .collect();
    let weight: Vec<u16> = (0..HIDDEN)
        .map(|_| bf16::from_f32(rng.next_f32() * 0.25))
        .collect();
    let norm = RmsNorm::from_host(&weight, EPSILON).unwrap();
    let stream = Stream::new().unwrap();
    let device_input = DeviceBuffer::from_slice(&input).unwrap();

    for with_residual in [false, true] {
        let mut expected_residual = original_residual.clone();
        let mut expected = vec![0.0f32; BATCH * HIDDEN];
        rmsnorm::reference::rms_norm(
            &input,
            with_residual.then_some(expected_residual.as_mut_slice()),
            &weight,
            &mut expected,
            BATCH,
            HIDDEN,
            EPSILON,
        );

        let mut device_residual = DeviceBuffer::from_slice(&original_residual).unwrap();
        let mut output = DeviceBuffer::<u16>::zeroed(BATCH * HIDDEN).unwrap();
        norm.forward_bf16(
            &device_input,
            with_residual.then_some(&mut device_residual),
            &mut output,
            BATCH,
            &stream,
        )
        .unwrap();
        stream.synchronize().unwrap();
        let actual = output.to_vec().unwrap();
        for (index, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
            let got = bf16::to_f32(got);
            assert!(
                (got - want).abs() <= 0.012 + want.abs() * 0.003,
                "residual={with_residual}, index={index}: GPU={got}, CPU={want}"
            );
        }
        if with_residual {
            assert_eq!(device_residual.to_vec().unwrap(), expected_residual);
        }
    }
}

#[test]
fn fused_rmsnorm_nvfp4_reconstructs_cpu_result() {
    const BATCH: usize = 4;
    const HIDDEN: usize = 512;
    const EPSILON: f32 = 1e-6;
    const GLOBAL_SCALE: f32 = 128.0;
    let mut rng = Rng(0x4a4_512);
    let input: Vec<u16> = (0..BATCH * HIDDEN)
        .map(|_| bf16::from_f32(rng.next_f32() * 2.5))
        .collect();
    let residual: Vec<u16> = (0..BATCH * HIDDEN)
        .map(|_| bf16::from_f32(rng.next_f32()))
        .collect();
    let weight: Vec<u16> = (0..HIDDEN)
        .map(|_| bf16::from_f32(rng.next_f32() * 0.2))
        .collect();
    let mut expected_residual = residual.clone();
    let mut expected = vec![0.0f32; BATCH * HIDDEN];
    rmsnorm::reference::rms_norm(
        &input,
        Some(&mut expected_residual),
        &weight,
        &mut expected,
        BATCH,
        HIDDEN,
        EPSILON,
    );

    let stream = Stream::new().unwrap();
    let norm = RmsNorm::from_host(&weight, EPSILON).unwrap();
    let device_input = DeviceBuffer::from_slice(&input).unwrap();
    let mut device_residual = DeviceBuffer::from_slice(&residual).unwrap();
    let mut quantized = QuantizedActivation::zeroed(GLOBAL_SCALE, BATCH, HIDDEN).unwrap();
    norm.forward_nvfp4(
        &device_input,
        Some(&mut device_residual),
        &mut quantized,
        &stream,
    )
    .unwrap();
    stream.synchronize().unwrap();
    assert_eq!(device_residual.to_vec().unwrap(), expected_residual);
    let (packed, scales) = quantized.to_host_logical().unwrap();

    for row in 0..BATCH {
        for group in 0..HIDDEN / 16 {
            let scale = nvfp4::reference::e4m3(scales[row * (HIDDEN / 16) + group]) / GLOBAL_SCALE;
            let mut maximum = 0.0f32;
            let mut max_error = 0.0f32;
            for i in 0..16 {
                let column = group * 16 + i;
                let byte = packed[row * (HIDDEN / 2) + column / 2];
                let nibble = if column & 1 == 0 {
                    byte & 0x0f
                } else {
                    byte >> 4
                };
                let got = nvfp4::reference::e2m1(nibble) * scale;
                let want = expected[row * HIDDEN + column];
                maximum = maximum.max(want.abs());
                max_error = max_error.max((got - want).abs());
            }
            assert!(
                max_error <= maximum * 0.19 + 0.003,
                "row={row}, group={group}: max={maximum}, error={max_error}"
            );
        }
    }
}

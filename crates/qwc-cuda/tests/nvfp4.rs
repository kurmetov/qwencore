//! Сверка shared-weight NVFP4 x BF16 decode-проекции с CPU-oracle.
//! Требует RTX 5090.

use qwc_cuda::nvfp4::{self, Linear, QuantizedActivation, W4A4Workspace};
use qwc_cuda::{DeviceBuffer, Stream, bf16};

struct Rng(u64);

impl Rng {
    fn next_u8(&mut self) -> u8 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 56) as u8
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u8() as f32 / 255.0) * 2.0 - 1.0
    }
}

#[test]
fn w4a16_matches_cpu_for_all_supported_batches() {
    const OUT: usize = 37; // проверяет неполный последний tile из 16 строк
    const IN: usize = 512;
    let mut rng = Rng(0x4a16_5090);
    let packed: Vec<u8> = (0..OUT * IN / 2).map(|_| rng.next_u8()).collect();
    let up_packed: Vec<u8> = (0..OUT * IN / 2).map(|_| rng.next_u8()).collect();
    // Положительные конечные E4M3-шкалы: 0.5, 1.0, 1.5, 2.0.
    let scale_values = [0x30u8, 0x38, 0x3c, 0x40];
    let scales: Vec<u8> = (0..OUT * IN / 16)
        .map(|i| scale_values[i % scale_values.len()])
        .collect();
    let weight_global_scale = 8.0;
    let up_global_scale = 4.0;

    let linear = Linear::from_host(&packed, &scales, weight_global_scale, OUT, IN).unwrap();
    let up_linear = Linear::from_host(&up_packed, &scales, up_global_scale, OUT, IN).unwrap();
    let stream = Stream::new().unwrap();

    for batch in 1..=4 {
        let input: Vec<u16> = (0..batch * IN)
            .map(|_| bf16::from_f32(rng.next_f32()))
            .collect();
        let mut expected = vec![0.0f32; batch * OUT];
        nvfp4::reference::gemv_w4a16(
            &packed,
            &scales,
            weight_global_scale,
            &input,
            &mut expected,
            OUT,
            IN,
            batch,
        );

        let d_input = DeviceBuffer::from_slice(&input).unwrap();
        let mut d_output = DeviceBuffer::<u16>::zeroed(batch * OUT).unwrap();
        linear
            .forward_w4a16(&d_input, &mut d_output, batch, &stream)
            .unwrap();
        stream.synchronize().unwrap();
        let actual = d_output.to_vec().unwrap();

        for (index, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
            let got = bf16::to_f32(got);
            let tolerance = 0.035 + want.abs() * 0.012;
            assert!(
                (got - want).abs() <= tolerance,
                "batch={batch}, index={index}: GPU={got}, CPU={want}, допуск={tolerance}"
            );
        }

        let mut expected_up = vec![0.0f32; batch * OUT];
        nvfp4::reference::gemv_w4a16(
            &up_packed,
            &scales,
            up_global_scale,
            &input,
            &mut expected_up,
            OUT,
            IN,
            batch,
        );
        for (gate, up) in expected.iter_mut().zip(expected_up) {
            *gate = *gate / (1.0 + (-*gate).exp()) * up;
        }

        let mut fused_output = DeviceBuffer::<u16>::zeroed(batch * OUT).unwrap();
        nvfp4::swiglu_w4a16(
            &linear,
            &up_linear,
            &d_input,
            &mut fused_output,
            batch,
            &stream,
        )
        .unwrap();
        stream.synchronize().unwrap();
        let fused_actual = fused_output.to_vec().unwrap();
        for (index, (&got, &want)) in fused_actual.iter().zip(&expected).enumerate() {
            let got = bf16::to_f32(got);
            let tolerance = 0.075 + want.abs() * 0.025;
            assert!(
                (got - want).abs() <= tolerance,
                "fused batch={batch}, index={index}: GPU={got}, CPU={want}, допуск={tolerance}"
            );
        }
    }
}

#[test]
fn w4a4_tensor_core_matches_cpu_layout_oracle() {
    const BATCH: usize = 3;
    const OUT: usize = 128;
    const IN: usize = 256;
    let mut rng = Rng(0x4a4_5090);
    let packed_input: Vec<u8> = (0..BATCH * IN / 2).map(|_| rng.next_u8()).collect();
    let packed_weight: Vec<u8> = (0..OUT * IN / 2).map(|_| rng.next_u8()).collect();
    let scale_values = [0x30u8, 0x34, 0x38, 0x3c, 0x40];
    let input_scales: Vec<u8> = (0..BATCH * IN / 16)
        .map(|i| scale_values[(i * 3 + i / 7) % scale_values.len()])
        .collect();
    let weight_scales: Vec<u8> = (0..OUT * IN / 16)
        .map(|i| scale_values[(i * 5 + i / 11) % scale_values.len()])
        .collect();
    let input_global_scale = 16.0;
    let weight_global_scale = 32.0;

    let input =
        QuantizedActivation::from_host(&packed_input, &input_scales, input_global_scale, BATCH, IN)
            .unwrap();
    let linear =
        Linear::from_host(&packed_weight, &weight_scales, weight_global_scale, OUT, IN).unwrap();
    let mut expected = vec![0.0f32; BATCH * OUT];
    nvfp4::reference::gemm_w4a4(
        &packed_input,
        &input_scales,
        input_global_scale,
        &packed_weight,
        &weight_scales,
        weight_global_scale,
        &mut expected,
        BATCH,
        OUT,
        IN,
    );

    let stream = Stream::new().unwrap();
    let mut workspace = W4A4Workspace::new(BATCH, OUT, IN).unwrap();
    let mut output = DeviceBuffer::<u16>::zeroed(BATCH * OUT).unwrap();
    linear
        .forward_w4a4_quantized(&input, &mut output, &mut workspace, &stream)
        .unwrap();
    stream.synchronize().unwrap();

    for (index, (&got, &want)) in output.to_vec().unwrap().iter().zip(&expected).enumerate() {
        let got = bf16::to_f32(got);
        let tolerance = 0.02 + want.abs() * 0.012;
        assert!(
            (got - want).abs() <= tolerance,
            "index={index}: GPU={got}, CPU={want}, допуск={tolerance}, workspace={} B",
            workspace.bytes()
        );
    }
}

#[test]
fn bf16_quantizer_and_w4a4_gemm_match_dequantized_oracle() {
    const BATCH: usize = 3;
    const OUT: usize = 128;
    const IN: usize = 256;
    let mut rng = Rng(0x4a4_bf16_5090);
    let input: Vec<u16> = (0..BATCH * IN)
        .map(|i| {
            let shaped = rng.next_f32() * (0.25 + (i % 29) as f32 / 12.0);
            bf16::from_f32(shaped)
        })
        .collect();
    let input_global_scale = 128.0;
    let device_input = DeviceBuffer::from_slice(&input).unwrap();
    let stream = Stream::new().unwrap();
    let mut quantized = QuantizedActivation::zeroed(input_global_scale, BATCH, IN).unwrap();
    quantized.quantize_bf16(&device_input, &stream).unwrap();
    stream.synchronize().unwrap();
    let (packed_input, input_scales) = quantized.to_host_logical().unwrap();

    for batch in 0..BATCH {
        for group in 0..IN / 16 {
            let scale = nvfp4::reference::e4m3(input_scales[batch * (IN / 16) + group])
                / input_global_scale;
            let mut maximum = 0.0f32;
            let mut max_error = 0.0f32;
            for j in 0..16 {
                let k = group * 16 + j;
                let value = bf16::to_f32(input[batch * IN + k]);
                let byte = packed_input[batch * (IN / 2) + k / 2];
                let nibble = if k & 1 == 0 { byte & 0x0f } else { byte >> 4 };
                let reconstructed = nvfp4::reference::e2m1(nibble) * scale;
                maximum = maximum.max(value.abs());
                max_error = max_error.max((value - reconstructed).abs());
            }
            assert!(
                max_error <= maximum * 0.19 + 0.002,
                "batch={batch}, group={group}: max={maximum}, error={max_error}, scale={scale}"
            );
        }
    }

    let packed_weight: Vec<u8> = (0..OUT * IN / 2).map(|_| rng.next_u8()).collect();
    let scale_values = [0x30u8, 0x34, 0x38, 0x3c];
    let weight_scales: Vec<u8> = (0..OUT * IN / 16)
        .map(|i| scale_values[(i * 7 + i / 13) % scale_values.len()])
        .collect();
    let weight_global_scale = 32.0;
    let linear =
        Linear::from_host(&packed_weight, &weight_scales, weight_global_scale, OUT, IN).unwrap();
    let mut expected = vec![0.0f32; BATCH * OUT];
    nvfp4::reference::gemm_w4a4(
        &packed_input,
        &input_scales,
        input_global_scale,
        &packed_weight,
        &weight_scales,
        weight_global_scale,
        &mut expected,
        BATCH,
        OUT,
        IN,
    );
    let mut output = DeviceBuffer::<u16>::zeroed(BATCH * OUT).unwrap();
    let mut workspace = W4A4Workspace::new(BATCH, OUT, IN).unwrap();
    linear
        .forward_w4a4_quantized(&quantized, &mut output, &mut workspace, &stream)
        .unwrap();
    stream.synchronize().unwrap();
    for (index, (&got, &want)) in output.to_vec().unwrap().iter().zip(&expected).enumerate() {
        let got = bf16::to_f32(got);
        let tolerance = 0.02 + want.abs() * 0.012;
        assert!(
            (got - want).abs() <= tolerance,
            "index={index}: GPU={got}, CPU={want}, допуск={tolerance}"
        );
    }
}

/// Строки за старым потолком в 1024. Разметка block-scale укладывает строки
/// тайлами по 128, и до сегодняшнего чанка в 2048 ни один тест не предъявлял
/// кернелам M больше 1024 — проверяется и квантователь, и сам GEMM.
#[test]
fn quantizer_and_w4a4_match_oracle_past_the_old_row_cap() {
    const OUT: usize = 128;
    const IN: usize = 256;
    let mut rng = Rng(0x2048_5090);
    let packed_weight: Vec<u8> = (0..OUT * IN / 2).map(|_| rng.next_u8()).collect();
    let scale_values = [0x30u8, 0x34, 0x38, 0x3c];
    let weight_scales: Vec<u8> = (0..OUT * IN / 16)
        .map(|i| scale_values[(i * 7 + i / 13) % scale_values.len()])
        .collect();
    let weight_global_scale = 32.0;
    let linear =
        Linear::from_host(&packed_weight, &weight_scales, weight_global_scale, OUT, IN).unwrap();
    let stream = Stream::new().unwrap();

    // 1025 — первая строка следующего тайла, 2048 — потолок арены префилла.
    for rows in [1025usize, 2048] {
        let input: Vec<u16> = (0..rows * IN)
            .map(|i| bf16::from_f32(rng.next_f32() * (0.25 + (i % 29) as f32 / 12.0)))
            .collect();
        let input_global_scale = 128.0;
        let device_input = DeviceBuffer::from_slice(&input).unwrap();
        let mut quantized = QuantizedActivation::zeroed(input_global_scale, rows, IN).unwrap();
        quantized
            .quantize_bf16_rows(&device_input, input_global_scale, rows, &stream)
            .unwrap();
        stream.synchronize().unwrap();
        let (packed_input, input_scales) = quantized.to_host_logical().unwrap();

        // Квантование построчно независимо, поэтому достаточно граничных строк:
        // первой, последней и первой строки последнего тайла.
        for row in [0usize, rows - 128, rows - 1] {
            for group in 0..IN / 16 {
                let scale =
                    nvfp4::reference::e4m3(input_scales[row * (IN / 16) + group]) / input_global_scale;
                let mut maximum = 0.0f32;
                let mut max_error = 0.0f32;
                for j in 0..16 {
                    let k = group * 16 + j;
                    let value = bf16::to_f32(input[row * IN + k]);
                    let byte = packed_input[row * (IN / 2) + k / 2];
                    let nibble = if k & 1 == 0 { byte & 0x0f } else { byte >> 4 };
                    let reconstructed = nvfp4::reference::e2m1(nibble) * scale;
                    maximum = maximum.max(value.abs());
                    max_error = max_error.max((value - reconstructed).abs());
                }
                assert!(
                    max_error <= maximum * 0.19 + 0.002,
                    "rows={rows}, row={row}, group={group}: max={maximum}, error={max_error}"
                );
            }
        }

        let mut expected = vec![0.0f32; rows * OUT];
        nvfp4::reference::gemm_w4a4(
            &packed_input,
            &input_scales,
            input_global_scale,
            &packed_weight,
            &weight_scales,
            weight_global_scale,
            &mut expected,
            rows,
            OUT,
            IN,
        );
        let mut output = DeviceBuffer::<u16>::zeroed(rows * OUT).unwrap();
        let mut workspace = W4A4Workspace::new(rows, OUT, IN).unwrap();
        linear
            .forward_w4a4_quantized(&quantized, &mut output, &mut workspace, &stream)
            .unwrap();
        stream.synchronize().unwrap();
        for (index, (&got, &want)) in output.to_vec().unwrap().iter().zip(&expected).enumerate() {
            let got = bf16::to_f32(got);
            let tolerance = 0.02 + want.abs() * 0.012;
            assert!(
                (got - want).abs() <= tolerance,
                "rows={rows}, index={index}: GPU={got}, CPU={want}, допуск={tolerance}"
            );
        }
    }
}

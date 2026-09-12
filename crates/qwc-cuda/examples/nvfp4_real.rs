//! End-to-end проверка настоящего NVFP4-тензора из QUASAR checkpoint.
//!
//! `cargo run --release -p qwc-cuda --example nvfp4_real -- /path/to/model`

use qwc_core::arch::{HIDDEN_SIZE, INTERMEDIATE_SIZE};
use qwc_cuda::nvfp4::{self, Linear, QuantizedActivation, W4A4Workspace};
use qwc_cuda::{DeviceBuffer, Stream, bf16};
use qwc_model::Checkpoint;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("передайте путь к checkpoint первым аргументом")?;
    let checkpoint = Checkpoint::open(&model)?;
    let gate_prefix = "model.language_model.layers.0.mlp.gate_proj";
    let up_prefix = "model.language_model.layers.0.mlp.up_proj";
    let gate_weight = checkpoint
        .quant_linear(gate_prefix, INTERMEDIATE_SIZE, HIDDEN_SIZE)
        .map_err(|error| format!("{gate_prefix}: {error}"))?;
    let up_weight = checkpoint
        .quant_linear(up_prefix, INTERMEDIATE_SIZE, HIDDEN_SIZE)
        .map_err(|error| format!("{up_prefix}: {error}"))?;

    let input: Vec<u16> = (0..HIDDEN_SIZE)
        .map(|i| {
            let value = ((i as f32 * 0.013).sin() + (i as f32 * 0.0031).cos()) * 0.25;
            bf16::from_f32(value)
        })
        .collect();
    let mut expected = vec![0.0f32; INTERMEDIATE_SIZE];
    nvfp4::reference::gemv_w4a16(
        gate_weight.packed,
        gate_weight.block_scales,
        gate_weight.weight_global_scale,
        &input,
        &mut expected,
        INTERMEDIATE_SIZE,
        HIDDEN_SIZE,
        1,
    );
    let mut expected_up = vec![0.0f32; INTERMEDIATE_SIZE];
    nvfp4::reference::gemv_w4a16(
        up_weight.packed,
        up_weight.block_scales,
        up_weight.weight_global_scale,
        &input,
        &mut expected_up,
        INTERMEDIATE_SIZE,
        HIDDEN_SIZE,
        1,
    );

    let gate = Linear::from_host(
        gate_weight.packed,
        gate_weight.block_scales,
        gate_weight.weight_global_scale,
        INTERMEDIATE_SIZE,
        HIDDEN_SIZE,
    )?;
    let up = Linear::from_host(
        up_weight.packed,
        up_weight.block_scales,
        up_weight.weight_global_scale,
        INTERMEDIATE_SIZE,
        HIDDEN_SIZE,
    )?;
    let stream = Stream::new()?;
    let device_input = DeviceBuffer::from_slice(&input)?;
    let mut device_output = DeviceBuffer::<u16>::zeroed(INTERMEDIATE_SIZE)?;
    gate.forward_w4a16(&device_input, &mut device_output, 1, &stream)?;
    stream.synchronize()?;
    let actual = device_output.to_vec()?;

    let mut worst_absolute = 0.0f32;
    let mut worst_scaled = 0.0f32;
    for (&got, &want) in actual.iter().zip(&expected) {
        let absolute = (bf16::to_f32(got) - want).abs();
        worst_absolute = worst_absolute.max(absolute);
        worst_scaled = worst_scaled.max(absolute / want.abs().max(0.05));
    }

    println!("тензор: {gate_prefix}");
    println!("форма: {} x {}", INTERMEDIATE_SIZE, HIDDEN_SIZE);
    println!(
        "checkpoint weight_global_scale (делитель): {}",
        gate_weight.weight_global_scale
    );
    println!("max absolute error: {worst_absolute:.6}");
    println!("max scaled error:   {worst_scaled:.6}");
    assert!(worst_scaled < 0.025, "GPU расходится с CPU-oracle");

    let mut expected_swiglu = expected;
    for (gate_value, up_value) in expected_swiglu.iter_mut().zip(expected_up) {
        *gate_value = *gate_value / (1.0 + (-*gate_value).exp()) * up_value;
    }
    let mut fused_output = DeviceBuffer::<u16>::zeroed(INTERMEDIATE_SIZE)?;
    nvfp4::swiglu_w4a16(&gate, &up, &device_input, &mut fused_output, 1, &stream)?;
    stream.synchronize()?;
    let fused_actual = fused_output.to_vec()?;
    let mut fused_worst_scaled = 0.0f32;
    for (&got, &want) in fused_actual.iter().zip(&expected_swiglu) {
        let absolute = (bf16::to_f32(got) - want).abs();
        fused_worst_scaled = fused_worst_scaled.max(absolute / want.abs().max(0.05));
    }
    println!("fused gate+up+SwiGLU max scaled error: {fused_worst_scaled:.6}");
    assert!(
        fused_worst_scaled < 0.035,
        "fused GPU расходится с CPU-oracle"
    );

    const W4A4_BATCH: usize = 4;
    let batched_input: Vec<u16> = (0..W4A4_BATCH)
        .flat_map(|batch| {
            input.iter().enumerate().map(move |(i, &value)| {
                let perturbation = ((batch * 17 + i) as f32 * 0.007).sin() * 0.03;
                bf16::from_f32(bf16::to_f32(value) + perturbation)
            })
        })
        .collect();
    let device_batched_input = DeviceBuffer::from_slice(&batched_input)?;
    let mut quantized =
        QuantizedActivation::zeroed(gate_weight.input_global_scale, W4A4_BATCH, HIDDEN_SIZE)?;
    quantized.quantize_bf16(&device_batched_input, &stream)?;
    stream.synchronize()?;
    let (packed_input, input_scales) = quantized.to_host_logical()?;
    let mut expected_w4a4 = vec![0.0f32; W4A4_BATCH * INTERMEDIATE_SIZE];
    nvfp4::reference::gemm_w4a4(
        &packed_input,
        &input_scales,
        gate_weight.input_global_scale,
        gate_weight.packed,
        gate_weight.block_scales,
        gate_weight.weight_global_scale,
        &mut expected_w4a4,
        W4A4_BATCH,
        INTERMEDIATE_SIZE,
        HIDDEN_SIZE,
    );
    let mut w4a4_output = DeviceBuffer::<u16>::zeroed(W4A4_BATCH * INTERMEDIATE_SIZE)?;
    let mut workspace = W4A4Workspace::new(W4A4_BATCH, INTERMEDIATE_SIZE, HIDDEN_SIZE)?;
    gate.forward_w4a4_quantized(&quantized, &mut w4a4_output, &mut workspace, &stream)?;
    stream.synchronize()?;
    let actual_w4a4 = w4a4_output.to_vec()?;
    let mut w4a4_worst_scaled = 0.0f32;
    for (&got, &want) in actual_w4a4.iter().zip(&expected_w4a4) {
        let absolute = (bf16::to_f32(got) - want).abs();
        w4a4_worst_scaled = w4a4_worst_scaled.max(absolute / want.abs().max(0.05));
    }
    println!(
        "W4A4 batch {W4A4_BATCH} max scaled error: {w4a4_worst_scaled:.6}; workspace: {} B",
        workspace.bytes()
    );
    assert!(
        w4a4_worst_scaled < 0.035,
        "W4A4 GPU расходится с CPU-oracle"
    );
    Ok(())
}

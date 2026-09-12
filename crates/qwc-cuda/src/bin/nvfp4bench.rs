//! Streaming-бенчмарк NVFP4 x BF16 decode-проекции на реальных формах Qwen.
//! `cargo run --release -p qwc-cuda --bin nvfp4bench`
//!
//! Четыре разные матрицы гарантируют working set больше 96 MiB L2.

use qwc_core::arch::{HIDDEN_SIZE, INTERMEDIATE_SIZE, Q_PROJ_DIM};
use qwc_core::roofline::ACHIEVABLE_BANDWIDTH;
use qwc_cuda::nvfp4::{Linear, QuantizedActivation, W4A4Workspace, swiglu_w4a16};
use qwc_cuda::{Device, DeviceBuffer, Event, Stream, bf16};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::init(0)?;
    let stream = Stream::new()?;
    let shapes = [
        ("gate/up", INTERMEDIATE_SIZE, HIDDEN_SIZE),
        ("down", HIDDEN_SIZE, INTERMEDIATE_SIZE),
        ("q+gate", 2 * Q_PROJ_DIM, HIDDEN_SIZE),
    ];

    println!("NVFP4 x BF16 shared-weight decode, RTX 5090");
    println!(
        "достижимое DRAM-чтение: {:.0} GB/s; SM: {}\n",
        ACHIEVABLE_BANDWIDTH / 1e9,
        device.sm_count
    );
    println!(
        "  {:>7} | {:>5} | {:>13} | {:>9} | {:>9}",
        "операция", "batch", "время/слой", "weight BW", "% потолка"
    );
    println!(
        "  {:->7}-+-{:->5}-+-{:->13}-+-{:->9}-+-{:->9}",
        "", "", "", "", ""
    );

    for (name, out_features, in_features) in shapes {
        // Даже q+gate: 4 * 35.4 MB > 96 MiB L2.
        let linears: Vec<Linear> = (0..4)
            .map(|_| Linear::zeroed(out_features, in_features))
            .collect::<Result<_, _>>()?;

        let mut w4a16_us_by_batch = [0.0f64; 5];
        for (batch, w4a16_us) in w4a16_us_by_batch.iter_mut().enumerate().skip(1) {
            let input = DeviceBuffer::from_slice(&vec![bf16::from_f32(0.5); batch * in_features])?;
            let mut output = DeviceBuffer::<u16>::zeroed(batch * out_features)?;

            // Короткий kernel иначе попадает в период разгона частот GPU.
            for _ in 0..20 {
                for linear in &linears {
                    linear.forward_w4a16(&input, &mut output, batch, &stream)?;
                }
            }
            stream.synchronize()?;

            let iterations = 40usize;
            let mut samples = [0.0f32; 7];
            for elapsed in &mut samples {
                let (start, end) = (Event::new()?, Event::new()?);
                start.record(&stream)?;
                for _ in 0..iterations {
                    for linear in &linears {
                        linear.forward_w4a16(&input, &mut output, batch, &stream)?;
                    }
                }
                end.record(&stream)?;
                end.synchronize()?;
                *elapsed = Event::elapsed_ms(&start, &end)?;
            }
            samples.sort_by(f32::total_cmp);

            let calls = iterations * linears.len();
            let ms = samples[samples.len() / 2] as f64;
            let us = ms * 1e3 / calls as f64;
            *w4a16_us = us;
            let weight_bytes = linears[0].weight_bytes() as f64 * calls as f64;
            let gbps = weight_bytes / (ms * 1e-3) / 1e9;
            println!(
                "  {:>7} | {:>5} | {:>9.2} us | {:>6.0} GB/s | {:>8.1}%",
                name,
                batch,
                us,
                gbps,
                gbps * 1e9 / ACHIEVABLE_BANDWIDTH * 100.0,
            );
        }

        println!("\n  {name}: BF16 -> W4A4 quant + SM120 tensor-core GEMM");
        println!(
            "  {:>5} | {:>10} | {:>10} | {:>10} | {:>9} | {:>9}",
            "batch", "quant", "GEMM", "итого", "vs A16", "weight BW"
        );
        println!(
            "  {:->5}-+-{:->10}-+-{:->10}-+-{:->10}-+-{:->9}-+-{:->9}",
            "", "", "", "", "", ""
        );
        for batch in [3usize, 4, 8, 16, 32, 64, 128, 256, 512] {
            let input = DeviceBuffer::from_slice(&vec![bf16::from_f32(0.5); batch * in_features])?;
            let mut quantized = QuantizedActivation::zeroed(128.0, batch, in_features)?;
            let mut output = DeviceBuffer::<u16>::zeroed(batch * out_features)?;
            let mut workspace = W4A4Workspace::new(batch, out_features, in_features)?;

            for _ in 0..20 {
                quantized.quantize_bf16(&input, &stream)?;
                for linear in &linears {
                    linear.forward_w4a4_quantized(
                        &quantized,
                        &mut output,
                        &mut workspace,
                        &stream,
                    )?;
                }
            }
            stream.synchronize()?;

            let quant_iterations = 200usize;
            let mut quant_samples = [0.0f32; 7];
            let mut gemm_samples = [0.0f32; 7];
            for sample in 0..7 {
                let (start, end) = (Event::new()?, Event::new()?);
                start.record(&stream)?;
                for _ in 0..quant_iterations {
                    quantized.quantize_bf16(&input, &stream)?;
                }
                end.record(&stream)?;
                end.synchronize()?;
                quant_samples[sample] = Event::elapsed_ms(&start, &end)?;

                let (start, end) = (Event::new()?, Event::new()?);
                start.record(&stream)?;
                for _ in 0..40 {
                    for linear in &linears {
                        linear.forward_w4a4_quantized(
                            &quantized,
                            &mut output,
                            &mut workspace,
                            &stream,
                        )?;
                    }
                }
                end.record(&stream)?;
                end.synchronize()?;
                gemm_samples[sample] = Event::elapsed_ms(&start, &end)?;
            }
            quant_samples.sort_by(f32::total_cmp);
            gemm_samples.sort_by(f32::total_cmp);

            let quant_us = quant_samples[3] as f64 * 1e3 / quant_iterations as f64;
            let gemm_calls = 40 * linears.len();
            let gemm_ms = gemm_samples[3] as f64;
            let gemm_us = gemm_ms * 1e3 / gemm_calls as f64;
            let total_us = quant_us + gemm_us;
            let weight_bytes = linears[0].weight_bytes() as f64 * gemm_calls as f64;
            let gbps = weight_bytes / (gemm_ms * 1e-3) / 1e9;
            let versus = if batch <= 4 {
                format!("{:.2}x", w4a16_us_by_batch[batch] / total_us)
            } else {
                "-".to_string()
            };
            println!(
                "  {:>5} | {:>7.2} us | {:>7.2} us | {:>7.2} us | {:>9} | {:>6.0} GB/s",
                batch, quant_us, gemm_us, total_us, versus, gbps
            );
        }
        println!();
    }

    println!("\nFused MLP gate + up + SwiGLU (время на пару матриц)");
    println!(
        "  {:>5} | {:>13} | {:>13} | {:>9} | {:>9}",
        "batch", "2 projection", "fused", "ускорение", "weight BW"
    );
    println!(
        "  {:->5}-+-{:->13}-+-{:->13}-+-{:->9}-+-{:->9}",
        "", "", "", "", ""
    );

    let gates: Vec<Linear> = (0..4)
        .map(|_| Linear::zeroed(INTERMEDIATE_SIZE, HIDDEN_SIZE))
        .collect::<Result<_, _>>()?;
    let ups: Vec<Linear> = (0..4)
        .map(|_| Linear::zeroed(INTERMEDIATE_SIZE, HIDDEN_SIZE))
        .collect::<Result<_, _>>()?;

    for batch in 1..=4 {
        let input = DeviceBuffer::from_slice(&vec![bf16::from_f32(0.5); batch * HIDDEN_SIZE])?;
        let mut gate_output = DeviceBuffer::<u16>::zeroed(batch * INTERMEDIATE_SIZE)?;
        let mut up_output = DeviceBuffer::<u16>::zeroed(batch * INTERMEDIATE_SIZE)?;
        let mut fused_output = DeviceBuffer::<u16>::zeroed(batch * INTERMEDIATE_SIZE)?;

        for _ in 0..20 {
            for (gate, up) in gates.iter().zip(&ups) {
                swiglu_w4a16(gate, up, &input, &mut fused_output, batch, &stream)?;
            }
        }
        stream.synchronize()?;

        let iterations = 30usize;
        let mut separate_samples = [0.0f32; 7];
        let mut fused_samples = [0.0f32; 7];
        for sample in 0..7 {
            let (start, end) = (Event::new()?, Event::new()?);
            start.record(&stream)?;
            for _ in 0..iterations {
                for (gate, up) in gates.iter().zip(&ups) {
                    gate.forward_w4a16(&input, &mut gate_output, batch, &stream)?;
                    up.forward_w4a16(&input, &mut up_output, batch, &stream)?;
                }
            }
            end.record(&stream)?;
            end.synchronize()?;
            separate_samples[sample] = Event::elapsed_ms(&start, &end)?;

            let (start, end) = (Event::new()?, Event::new()?);
            start.record(&stream)?;
            for _ in 0..iterations {
                for (gate, up) in gates.iter().zip(&ups) {
                    swiglu_w4a16(gate, up, &input, &mut fused_output, batch, &stream)?;
                }
            }
            end.record(&stream)?;
            end.synchronize()?;
            fused_samples[sample] = Event::elapsed_ms(&start, &end)?;
        }
        separate_samples.sort_by(f32::total_cmp);
        fused_samples.sort_by(f32::total_cmp);

        let calls = iterations * gates.len();
        let separate_ms = separate_samples[3] as f64;
        let fused_ms = fused_samples[3] as f64;
        let separate_us = separate_ms * 1e3 / calls as f64;
        let fused_us = fused_ms * 1e3 / calls as f64;
        let bytes = 2.0 * gates[0].weight_bytes() as f64 * calls as f64;
        let fused_gbps = bytes / (fused_ms * 1e-3) / 1e9;
        println!(
            "  {:>5} | {:>9.2} us | {:>9.2} us | {:>8.2}x | {:>6.0} GB/s",
            batch,
            separate_us,
            fused_us,
            separate_us / fused_us,
            fused_gbps,
        );
    }
    Ok(())
}

//! Проекции W4A4 на формах префилла: сколько стоит чанк и что даёт рост M.
//! `cargo run --release -p qwc-cuda --bin prefillgemmbench`
//!
//! Каждая форма меряется на четырёх копиях весов: одна копия оседает в L2 и
//! показывает не ту цену, что шаг движка, где веса приходят из DRAM.

use qwc_core::arch::{
    HIDDEN_SIZE, INTERMEDIATE_SIZE, LA_CONV_CHANNELS, LA_NUM_V_HEADS, LA_V_PROJ_DIM,
    NUM_FULL_LAYERS, NUM_LAYERS, NUM_LINEAR_LAYERS, Q_PROJ_DIM,
};
use qwc_cuda::nvfp4::{Linear, QuantizedActivation, W4A4Workspace};
use qwc_cuda::{Device, DeviceBuffer, Event, Stream, bf16};

/// Сколько раз форма встречается за один проход по всем 64 слоям.
struct Shape {
    name: &'static str,
    out_features: usize,
    in_features: usize,
    per_step: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::init(0)?;
    let stream = Stream::new()?;
    let mixer_fused = LA_CONV_CHANNELS + LA_V_PROJ_DIM + 2 * LA_NUM_V_HEADS;

    let shapes = [
        Shape { name: "mlp.gate", out_features: INTERMEDIATE_SIZE, in_features: HIDDEN_SIZE, per_step: NUM_LAYERS },
        Shape { name: "mlp.up", out_features: INTERMEDIATE_SIZE, in_features: HIDDEN_SIZE, per_step: NUM_LAYERS },
        Shape { name: "mlp.gate+up", out_features: 2 * INTERMEDIATE_SIZE, in_features: HIDDEN_SIZE, per_step: 0 },
        Shape { name: "mlp.down", out_features: HIDDEN_SIZE, in_features: INTERMEDIATE_SIZE, per_step: NUM_LAYERS },
        Shape { name: "attn.q+gate", out_features: 2 * Q_PROJ_DIM, in_features: HIDDEN_SIZE, per_step: NUM_FULL_LAYERS },
        Shape { name: "attn.o", out_features: HIDDEN_SIZE, in_features: Q_PROJ_DIM, per_step: NUM_FULL_LAYERS },
        Shape { name: "mixer.in", out_features: mixer_fused, in_features: HIDDEN_SIZE, per_step: NUM_LINEAR_LAYERS },
        Shape { name: "mixer.out", out_features: HIDDEN_SIZE, in_features: LA_V_PROJ_DIM, per_step: NUM_LINEAR_LAYERS },
    ];

    let rows_list: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().expect("rows"))
        .collect();
    let rows_list = if rows_list.is_empty() {
        vec![256usize, 511, 1024]
    } else {
        rows_list
    };

    println!(
        "W4A4 на формах префилла, RTX 5090 ({} SM)\n",
        device.sm_count
    );

    for &rows in &rows_list {
        println!("M = {rows}");
        println!(
            "  {:>12} | {:>8} | {:>8} | {:>10} | {:>10} | {:>9}",
            "проекция", "N", "K", "quant", "GEMM", "TFLOP/s"
        );
        println!(
            "  {:->12}-+-{:->8}-+-{:->8}-+-{:->10}-+-{:->10}-+-{:->9}",
            "", "", "", "", "", ""
        );

        // Цена шага складывается из GEMM всех слоёв плюс квантование входа
        // каждой проекции: на префилле оно идёт по live-строкам, не по арене.
        let mut step_us = 0.0f64;
        let mut step_flops = 0.0f64;
        for shape in &shapes {
            let linears: Vec<Linear> = (0..4)
                .map(|_| Linear::zeroed(shape.out_features, shape.in_features))
                .collect::<Result<_, _>>()?;
            let input =
                DeviceBuffer::from_slice(&vec![bf16::from_f32(0.5); rows * shape.in_features])?;
            let mut quantized = QuantizedActivation::zeroed(128.0, rows, shape.in_features)?;
            let mut output = DeviceBuffer::<u16>::zeroed(rows * shape.out_features)?;
            let mut workspace =
                W4A4Workspace::new(rows, shape.out_features, shape.in_features)?;

            quantized.quantize_bf16_rows(&input, 128.0, rows, &stream)?;
            for _ in 0..10 {
                for linear in &linears {
                    linear.forward_w4a4_quantized(&quantized, &mut output, &mut workspace, &stream)?;
                }
            }
            stream.synchronize()?;

            let mut quant_samples = [0.0f32; 5];
            let mut gemm_samples = [0.0f32; 5];
            for sample in 0..5 {
                let (start, end) = (Event::new()?, Event::new()?);
                start.record(&stream)?;
                for _ in 0..50 {
                    quantized.quantize_bf16_rows(&input, 128.0, rows, &stream)?;
                }
                end.record(&stream)?;
                end.synchronize()?;
                quant_samples[sample] = Event::elapsed_ms(&start, &end)? / 50.0;

                let (start, end) = (Event::new()?, Event::new()?);
                start.record(&stream)?;
                for _ in 0..20 {
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
                gemm_samples[sample] = Event::elapsed_ms(&start, &end)? / (20.0 * 4.0);
            }
            quant_samples.sort_by(f32::total_cmp);
            gemm_samples.sort_by(f32::total_cmp);
            let quant_us = quant_samples[2] as f64 * 1e3;
            let gemm_us = gemm_samples[2] as f64 * 1e3;
            let flops = 2.0 * rows as f64 * shape.out_features as f64 * shape.in_features as f64;

            println!(
                "  {:>12} | {:>8} | {:>8} | {:>7.1} us | {:>7.1} us | {:>9.0}",
                shape.name,
                shape.out_features,
                shape.in_features,
                quant_us,
                gemm_us,
                flops / (gemm_us * 1e-6) / 1e12,
            );

            step_us += (quant_us + gemm_us) * shape.per_step as f64;
            step_flops += flops * shape.per_step as f64;
        }

        println!(
            "  проекции всего шага: {:.2} мс, {:.1} мкс/токен, {:.0} TFLOP/s средних\n",
            step_us / 1e3,
            step_us / rows as f64,
            step_flops / (step_us * 1e-6) / 1e12,
        );
    }

    Ok(())
}

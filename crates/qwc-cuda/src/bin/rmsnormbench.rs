//! Decode-бенчмарк RMSNorm + dynamic NVFP4 quantization.
//! `cargo run --release -p qwc-cuda --bin rmsnormbench`

use qwc_core::arch::HIDDEN_SIZE;
use qwc_cuda::nvfp4::QuantizedActivation;
use qwc_cuda::rmsnorm::RmsNorm;
use qwc_cuda::{Device, DeviceBuffer, Event, Stream, bf16};

const ITERATIONS: usize = 400;
const SAMPLES: usize = 9;

fn median_us<F>(stream: &Stream, mut enqueue: F) -> Result<f64, Box<dyn std::error::Error>>
where
    F: FnMut() -> qwc_cuda::Result<()>,
{
    for _ in 0..40 {
        enqueue()?;
    }
    stream.synchronize()?;

    let mut samples = [0.0f32; SAMPLES];
    for elapsed in &mut samples {
        let (start, end) = (Event::new()?, Event::new()?);
        start.record(stream)?;
        for _ in 0..ITERATIONS {
            enqueue()?;
        }
        end.record(stream)?;
        end.synchronize()?;
        *elapsed = Event::elapsed_ms(&start, &end)?;
    }
    samples.sort_by(f32::total_cmp);
    Ok(samples[SAMPLES / 2] as f64 * 1e3 / ITERATIONS as f64)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::init(0)?;
    let stream = Stream::new()?;
    let weight = vec![bf16::from_f32(0.0); HIDDEN_SIZE];
    let norm = RmsNorm::from_host(&weight, 1e-6)?;

    println!(
        "RMSNorm -> NVFP4 decode producer, RTX 5090 ({} SM)",
        device.sm_count
    );
    println!("hidden={HIDDEN_SIZE}; median из {SAMPLES} серий по {ITERATIONS} итераций\n");
    println!(
        "  {:>5} | {:>8} | {:>11} | {:>8} | {:>13} | {:>8}",
        "batch", "separate", "fused", "speedup", "+res separate", "+res fused"
    );
    println!(
        "  {:->5}-+-{:->8}-+-{:->11}-+-{:->8}-+-{:->13}-+-{:->8}",
        "", "", "", "", "", ""
    );

    for batch in [1usize, 4, 8, 16, 32] {
        let elements = batch * HIDDEN_SIZE;
        let input = DeviceBuffer::from_slice(&vec![bf16::from_f32(0.5); elements])?;
        let mut normalized = DeviceBuffer::<u16>::zeroed(elements)?;
        let mut quantized = QuantizedActivation::zeroed(128.0, batch, HIDDEN_SIZE)?;
        let separate_us = median_us(&stream, || {
            norm.forward_bf16(&input, None, &mut normalized, batch, &stream)?;
            quantized.quantize_bf16(&normalized, &stream)
        })?;
        let fused_us = median_us(&stream, || {
            norm.forward_nvfp4(&input, None, &mut quantized, &stream)
        })?;

        let mut separate_residual =
            DeviceBuffer::from_slice(&vec![bf16::from_f32(0.25); elements])?;
        let separate_residual_us = median_us(&stream, || {
            norm.forward_bf16(
                &input,
                Some(&mut separate_residual),
                &mut normalized,
                batch,
                &stream,
            )?;
            quantized.quantize_bf16(&normalized, &stream)
        })?;
        let mut fused_residual = DeviceBuffer::from_slice(&vec![bf16::from_f32(0.25); elements])?;
        let fused_residual_us = median_us(&stream, || {
            norm.forward_nvfp4(&input, Some(&mut fused_residual), &mut quantized, &stream)
        })?;

        println!(
            "  {batch:>5} | {separate_us:>5.2} us | {fused_us:>8.2} us | {:>7.2}x | \
             {separate_residual_us:>10.2} us | {fused_residual_us:>5.2} us",
            separate_us / fused_us,
        );
    }
    Ok(())
}

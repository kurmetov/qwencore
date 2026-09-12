//! Замер chunk-скана Gated DeltaNet на prefill.
//! `cargo run --release -p qwc-cuda --bin deltaprefillbench`
//!
//! Скан на prefill — самая дорогая фаза шага (28.8% на разметке), и он
//! последователен по токенам: варп держит строку состояния и идёт по чанку
//! сам. Здесь он меряется изолированно, слой за слоем, как в настоящем шаге —
//! один буфер в цикле осел бы в L2.

use qwc_core::arch::NUM_LINEAR_LAYERS;
use qwc_cuda::delta_net::{
    self, DeltaInputs, DeltaStateMode, GATE_ELEMS, KQ_ELEMS, QK_ELEMS, STATE_ELEMS, V_ELEMS,
};
use qwc_cuda::{Device, DeviceBuffer, Event, Stream, bf16};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    Device::init(0)?;
    let stream = Stream::new()?;

    println!("Gated DeltaNet, chunk-скан prefill (одна последовательность, слот 0)\n");
    println!(
        "  {:>6} | {:>10} | {:>11} | {:>10} | {:>9}",
        "токенов", "на слой", "48 слоёв", "на токен", "токен/с"
    );
    println!(
        "  {:->6}-+-{:->10}-+-{:->11}-+-{:->10}-+-{:->9}",
        "", "", "", "", ""
    );

    for tokens in [128usize, 256, 512, 1024] {
        let mut states: Vec<DeviceBuffer<u16>> = (0..NUM_LINEAR_LAYERS)
            .map(|_| {
                let init: Vec<u16> = (0..STATE_ELEMS)
                    .map(|i| bf16::from_f32((i % 17) as f32 * 0.01))
                    .collect();
                DeviceBuffer::from_slice(&init).unwrap()
            })
            .collect();

        let q = DeviceBuffer::from_slice(&vec![0.088f32; tokens * QK_ELEMS])?;
        let k = DeviceBuffer::from_slice(&vec![0.088f32; tokens * QK_ELEMS])?;
        let v = DeviceBuffer::from_slice(&vec![0.5f32; tokens * V_ELEMS])?;
        let alpha = DeviceBuffer::from_slice(&vec![0.95f32; tokens * GATE_ELEMS])?;
        let beta = DeviceBuffer::from_slice(&vec![0.7f32; tokens * GATE_ELEMS])?;
        let mut out = DeviceBuffer::<f32>::zeroed(tokens * V_ELEMS)?;
        let kq = DeviceBuffer::from_slice(&vec![128.0f32 * 0.088 * 0.088; tokens * KQ_ELEMS])?;
        let inputs = DeltaInputs {
            q: &q,
            k: &k,
            v: &v,
            alpha: &alpha,
            beta: &beta,
            kq: &kq,
        };

        let mut scan = |states: &mut Vec<DeviceBuffer<u16>>| -> Result<(), Box<dyn std::error::Error>> {
            for s in states.iter_mut() {
                delta_net::prefill_slot(
                    s, &inputs, &mut out, 1, 0, tokens, 0,
                    DeltaStateMode::Bf16, &stream,
                )?;
            }
            Ok(())
        };

        scan(&mut states)?;
        stream.synchronize()?;

        let steps = 10;
        let (start, end) = (Event::new()?, Event::new()?);
        start.record(&stream)?;
        for _ in 0..steps {
            scan(&mut states)?;
        }
        end.record(&stream)?;
        end.synchronize()?;

        let ms = Event::elapsed_ms(&start, &end)? as f64;
        let calls = (steps * NUM_LINEAR_LAYERS) as f64;
        let per_layer_us = ms * 1e3 / calls;
        let per_step_ms = ms / steps as f64;
        let per_token_us = per_step_ms * 1e3 / tokens as f64;

        println!(
            "  {:>6} | {:>7.1} us | {:>8.3} ms | {:>7.3} us | {:>9.0}",
            tokens,
            per_layer_us,
            per_step_ms,
            per_token_us,
            tokens as f64 / (per_step_ms * 1e-3),
        );
    }
    Ok(())
}

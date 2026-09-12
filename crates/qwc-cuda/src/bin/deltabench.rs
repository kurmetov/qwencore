//! Замер рекуррентного шага Gated DeltaNet.
//! `cargo run --release -p qwc-cuda --bin deltabench`
//!
//! Проход идёт по 48 отдельным буферам состояния подряд — так же, как в
//! настоящем decode-шаге. Гонять один буфер в цикле нельзя: он осядет в
//! 96 MiB L2 и даст завышенную цифру.

use qwc_core::arch::NUM_LINEAR_LAYERS;
use qwc_core::roofline::ACHIEVABLE_BANDWIDTH;
use qwc_cuda::delta_net::{self, DeltaInputs, GATE_ELEMS, KQ_ELEMS, QK_ELEMS, STATE_ELEMS, V_ELEMS};
use qwc_cuda::{Device, DeviceBuffer, Event, Stream, bf16};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::init(0)?;
    let stream = Stream::new()?;

    // Потолок для операции «чтение + запись» измерен отдельно: 1408 GB/s.
    const COPY_CEILING: f64 = 1408e9;

    println!("Gated DeltaNet, рекуррентный шаг decode");
    println!(
        "состояние на слой на последовательность: {:.2} MB",
        (STATE_ELEMS * 2) as f64 / 1e6
    );
    println!(
        "паспортный пик {:.0} GB/s, достижимое чтение {:.0} GB/s, чтение+запись {:.0} GB/s\n",
        dev.peak_bandwidth() / 1e9,
        ACHIEVABLE_BANDWIDTH / 1e9,
        COPY_CEILING / 1e9
    );

    println!(
        "  {:>5} | {:>10} | {:>9} | {:>9} | {:>8} | {:>7}",
        "batch", "раб. мн-во", "на слой", "48 слоёв", "GB/s", "% пика"
    );
    println!(
        "  {:->5}-+-{:->10}-+-{:->9}-+-{:->9}-+-{:->8}-+-{:->7}",
        "", "", "", "", "", ""
    );

    for batch in [1usize, 4, 8, 16, 32] {
        let mut states: Vec<DeviceBuffer<u16>> = (0..NUM_LINEAR_LAYERS)
            .map(|_| {
                let init: Vec<u16> = (0..batch * STATE_ELEMS)
                    .map(|i| bf16::from_f32((i % 17) as f32 * 0.01))
                    .collect();
                DeviceBuffer::from_slice(&init).unwrap()
            })
            .collect();

        let q = DeviceBuffer::from_slice(&vec![0.088f32; batch * QK_ELEMS])?;
        let k = DeviceBuffer::from_slice(&vec![0.088f32; batch * QK_ELEMS])?;
        let v = DeviceBuffer::from_slice(&vec![0.5f32; batch * V_ELEMS])?;
        let alpha = DeviceBuffer::from_slice(&vec![0.95f32; batch * GATE_ELEMS])?;
        let beta = DeviceBuffer::from_slice(&vec![0.7f32; batch * GATE_ELEMS])?;
        let mut out = DeviceBuffer::<f32>::zeroed(batch * V_ELEMS)?;
        // k.q для постоянных q = k = 0.088 на 128 измерениях.
        let kq = DeviceBuffer::from_slice(&vec![128.0f32 * 0.088 * 0.088; batch * KQ_ELEMS])?;
        let inputs = DeltaInputs {
            q: &q,
            k: &k,
            v: &v,
            alpha: &alpha,
            beta: &beta,
            kq: &kq,
        };

        // Прогрев.
        for s in states.iter_mut() {
            delta_net::decode(s, &inputs, &mut out, batch, &stream)?;
        }
        stream.synchronize()?;

        let steps = 20;
        let (start, end) = (Event::new()?, Event::new()?);
        start.record(&stream)?;
        for _ in 0..steps {
            for s in states.iter_mut() {
                delta_net::decode(s, &inputs, &mut out, batch, &stream)?;
            }
        }
        end.record(&stream)?;
        end.synchronize()?;

        let ms = Event::elapsed_ms(&start, &end)? as f64;
        let calls = steps * NUM_LINEAR_LAYERS;
        let per_layer_us = ms * 1e3 / calls as f64;
        let per_step_ms = ms / steps as f64;
        let traffic = delta_net::traffic_bytes(batch) as f64 * calls as f64;
        let gbps = traffic / (ms * 1e-3) / 1e9;
        let working_set = (NUM_LINEAR_LAYERS * batch * STATE_ELEMS * 2) as f64 / 1e6;

        println!(
            "  {:>5} | {:>7.0} MB | {:>6.1} us | {:>6.2} ms | {:>8.0} | {:>6.0}%",
            batch,
            working_set,
            per_layer_us,
            per_step_ms,
            gbps,
            100.0 * gbps * 1e9 / COPY_CEILING
        );
    }
    Ok(())
}

//! Isolated FP8 lm_head benchmark at the real Qwen dimensions.

use qwc_core::arch::{HIDDEN_SIZE, VOCAB_SIZE};
use qwc_cuda::vocab::Fp8Vocab;
use qwc_cuda::{Device, DeviceBuffer, Event, Stream};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    Device::init(0)?;
    let stream = Stream::new()?;
    let measured_vocab = std::env::var("QWC_BENCH_VOCAB")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(VOCAB_SIZE);
    let head = Fp8Vocab::zeroed(measured_vocab, HIDDEN_SIZE)?;

    println!("FP8 lm_head [{measured_vocab}, {HIDDEN_SIZE}]");
    println!(
        "  {:>5} | {:>9} | {:>13} | {:>10}",
        "batch", "на запуск", "полный vocab", "токен/с"
    );
    println!("  {:->5}-+-{:->9}-+-{:->13}-+-{:->10}", "", "", "", "");

    for batch in [16usize, 32, 64, 80] {
        let hidden = DeviceBuffer::<u16>::zeroed(batch * HIDDEN_SIZE)?;
        let mut logits = DeviceBuffer::<f32>::zeroed(batch * measured_vocab)?;
        for _ in 0..3 {
            head.logits(&hidden, &mut logits, batch, &stream)?;
        }
        stream.synchronize()?;

        let iterations = 20;
        let (start, end) = (Event::new()?, Event::new()?);
        start.record(&stream)?;
        for _ in 0..iterations {
            head.logits(&hidden, &mut logits, batch, &stream)?;
        }
        end.record(&stream)?;
        end.synchronize()?;
        let milliseconds = Event::elapsed_ms(&start, &end)? as f64 / iterations as f64;
        let full_vocab_ms = milliseconds * VOCAB_SIZE as f64 / measured_vocab as f64;
        println!(
            "  {:>5} | {:>6.3} ms | {:>10.3} ms | {:>10.0}",
            batch,
            milliseconds,
            full_vocab_ms,
            batch as f64 / (full_vocab_ms * 1e-3),
        );
    }
    Ok(())
}

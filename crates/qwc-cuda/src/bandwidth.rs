//! Измерение реально достижимой пропускной способности.
//!
//! Развёртка по размеру буфера важна: у RTX 5090 96 MiB L2, а состояние
//! DeltaNet одной последовательности — 81 MB. Нужно знать, где проходит
//! граница между L2 и DRAM, чтобы понимать, какой знаменатель применять
//! к какому кернелу.

use crate::error::{Result, check};
use crate::{DeviceBuffer, Event, Stream, ffi};

pub struct Measurement {
    pub bytes: usize,
    pub ms: f32,
    pub gb_per_sec: f64,
}

/// Чтение буфера целиком. Трафик — `bytes` (только чтение).
pub fn read(bytes: usize, iterations: usize) -> Result<Measurement> {
    let n = bytes / std::mem::size_of::<f32>();
    let src = DeviceBuffer::<f32>::zeroed(n)?;
    let mut out = DeviceBuffer::<f32>::zeroed(1)?;
    let stream = Stream::new()?;

    let mut run = |s: &Stream| -> Result<()> {
        check(unsafe { ffi::qwc_bw_read(src.as_ptr(), bytes, out.as_mut_ptr().cast(), s.raw()) })
    };

    run(&stream)?;
    stream.synchronize()?;

    let (start, end) = (Event::new()?, Event::new()?);
    start.record(&stream)?;
    for _ in 0..iterations {
        run(&stream)?;
    }
    end.record(&stream)?;
    end.synchronize()?;

    let ms = Event::elapsed_ms(&start, &end)?;
    let traffic = bytes as f64 * iterations as f64;
    Ok(Measurement {
        bytes,
        ms,
        gb_per_sec: traffic / (ms as f64 * 1e-3) / 1e9,
    })
}

/// Копирование. Трафик — `2 * bytes` (чтение и запись).
pub fn copy(bytes: usize, iterations: usize) -> Result<Measurement> {
    let n = bytes / std::mem::size_of::<f32>();
    let src = DeviceBuffer::<f32>::zeroed(n)?;
    let mut dst = DeviceBuffer::<f32>::zeroed(n)?;
    let stream = Stream::new()?;

    let run = |s: &Stream, dst: &mut DeviceBuffer<f32>| -> Result<()> {
        check(unsafe { ffi::qwc_bw_copy(src.as_ptr(), dst.as_mut_ptr(), bytes, s.raw()) })
    };

    run(&stream, &mut dst)?;
    stream.synchronize()?;

    let (start, end) = (Event::new()?, Event::new()?);
    start.record(&stream)?;
    for _ in 0..iterations {
        run(&stream, &mut dst)?;
    }
    end.record(&stream)?;
    end.synchronize()?;

    let ms = Event::elapsed_ms(&start, &end)?;
    let traffic = 2.0 * bytes as f64 * iterations as f64;
    Ok(Measurement {
        bytes,
        ms,
        gb_per_sec: traffic / (ms as f64 * 1e-3) / 1e9,
    })
}

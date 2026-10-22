//! Measuring actually achievable bandwidth.
//!
//! Sweeping the buffer size matters: the RTX 5090 has 96 MiB of L2, and the DeltaNet state
//! of one sequence is 81 MB. We need to know where the
//! L2/DRAM boundary lies to know which denominator to apply
//! to which kernel.

use crate::error::{Result, check};
use crate::{DeviceBuffer, Event, Stream, ffi};

pub struct Measurement {
    pub bytes: usize,
    pub ms: f32,
    pub gb_per_sec: f64,
}

/// Reads the whole buffer. Traffic is `bytes` (read only).
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

/// Copy. Traffic is `2 * bytes` (read and write).
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

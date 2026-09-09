//! Потоки и события CUDA. События нужны прежде всего для замеров:
//! точность порядка микросекунд, чего хватает для профилирования кернелов.

use crate::error::{Result, check};
use crate::ffi;

pub struct Stream(ffi::Stream);

unsafe impl Send for Stream {}

impl Stream {
    pub fn new() -> Result<Self> {
        let mut s: ffi::Stream = std::ptr::null_mut();
        check(unsafe { ffi::cudaStreamCreate(&mut s) })?;
        Ok(Self(s))
    }

    pub fn raw(&self) -> ffi::Stream {
        self.0
    }

    pub fn synchronize(&self) -> Result<()> {
        check(unsafe { ffi::cudaStreamSynchronize(self.0) })
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe { ffi::cudaStreamDestroy(self.0) };
    }
}

pub struct Event(ffi::Event);

unsafe impl Send for Event {}

impl Event {
    pub fn new() -> Result<Self> {
        let mut e: ffi::Event = std::ptr::null_mut();
        check(unsafe { ffi::cudaEventCreate(&mut e) })?;
        Ok(Self(e))
    }

    pub fn record(&self, stream: &Stream) -> Result<()> {
        check(unsafe { ffi::cudaEventRecord(self.0, stream.raw()) })
    }

    pub fn synchronize(&self) -> Result<()> {
        check(unsafe { ffi::cudaEventSynchronize(self.0) })
    }

    /// Миллисекунды между двумя событиями.
    pub fn elapsed_ms(start: &Event, end: &Event) -> Result<f32> {
        let mut ms = 0.0f32;
        check(unsafe { ffi::cudaEventElapsedTime(&mut ms, start.0, end.0) })?;
        Ok(ms)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        unsafe { ffi::cudaEventDestroy(self.0) };
    }
}

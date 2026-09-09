//! Прямые объявления CUDA Runtime API. Только то, что используется.

use std::ffi::{c_char, c_int, c_void};

pub type Stream = *mut c_void;
pub type Event = *mut c_void;

pub const MEMCPY_HOST_TO_DEVICE: c_int = 1;
pub const MEMCPY_DEVICE_TO_HOST: c_int = 2;

unsafe extern "C" {
    pub fn cudaGetErrorString(error: c_int) -> *const c_char;
    pub fn cudaSetDevice(device: c_int) -> c_int;
    pub fn cudaDeviceSynchronize() -> c_int;
    pub fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> c_int;
    pub fn cudaDeviceGetAttribute(value: *mut c_int, attr: c_int, device: c_int) -> c_int;

    pub fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> c_int;
    pub fn cudaFree(ptr: *mut c_void) -> c_int;
    pub fn cudaMemset(ptr: *mut c_void, value: c_int, count: usize) -> c_int;
    pub fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: c_int) -> c_int;

    pub fn cudaStreamCreate(stream: *mut Stream) -> c_int;
    pub fn cudaStreamDestroy(stream: Stream) -> c_int;
    pub fn cudaStreamSynchronize(stream: Stream) -> c_int;

    pub fn cudaEventCreate(event: *mut Event) -> c_int;
    pub fn cudaEventDestroy(event: Event) -> c_int;
    pub fn cudaEventRecord(event: Event, stream: Stream) -> c_int;
    pub fn cudaEventSynchronize(event: Event) -> c_int;
    pub fn cudaEventElapsedTime(ms: *mut f32, start: Event, end: Event) -> c_int;

    // Кернелы из cuda/bandwidth.cu
    pub fn qwc_bw_read(src: *const c_void, bytes: usize, out: *mut f32, stream: Stream) -> c_int;
    pub fn qwc_bw_copy(src: *const c_void, dst: *mut c_void, bytes: usize, stream: Stream) -> c_int;
}

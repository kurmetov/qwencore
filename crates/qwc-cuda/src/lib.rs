//! The engine's CUDA layer: FFI, memory, streams, kernels.
//!
//! There is exactly one target architecture — sm_120a (RTX 5090). It's not a build
//! setting but part of the specialization: kernels are written for 99 KB of shared memory,
//! 170 SMs and block-scaled MMA for NVFP4, and make no sense on other hardware.

/// Row cap of one engine step: the prefill arena capacity.
///
/// Step kernels scale linearly in rows, so this isn't their limit but
/// an agreement on arena size — the same as `kMaxStepRows` in `limits.cuh`
/// and `PREFILL_CHUNK_SIZE` in the engine. Row-wise decode counts its rows
/// separately: see `paged_attention::MAX_DECODE_ROWS`.
pub const MAX_STEP_ROWS: usize = 2048;

pub mod attention_prepare;
pub mod bandwidth;
pub mod bf16;
pub mod delta_net;
pub mod device;
pub mod error;
pub(crate) mod ffi;
pub mod graph;
pub mod memory;
pub mod mtp;
pub mod nvfp4;
pub mod paged_attention;
pub mod rmsnorm;
pub mod sampling;
pub mod stream;
pub mod timeline;
pub mod vocab;

pub use device::Device;
pub use error::{CudaError, Result};
pub use memory::{DeviceBuffer, MemoryUsage, device_free_bytes, memory_usage, set_memory_limit};
pub use stream::{Event, Stream};
pub use timeline::Timeline;

/// Compacts selected rows of a BF16 `[rows, cols]` arena into a dense buffer.
///
/// A fused multi-sequence step needs the last row of every sequence. Gathering
/// first lets the vocabulary projection run once instead of re-reading the
/// whole 1.27 GB lm_head per sequence.
pub fn gather_rows_bf16(
    source: &DeviceBuffer<u16>,
    row_indices: &DeviceBuffer<u32>,
    destination: &mut DeviceBuffer<u16>,
    rows: usize,
    cols: usize,
    stream: &Stream,
) -> Result<()> {
    assert!(rows > 0 && cols > 0);
    assert!(row_indices.len() >= rows);
    assert!(destination.len() >= rows * cols);
    error::check(unsafe {
        ffi::qwc_gather_rows_bf16(
            source.as_ptr(),
            row_indices.as_ptr(),
            destination.as_mut_ptr(),
            rows as i32,
            cols as i32,
            stream.raw(),
        )
    })
}

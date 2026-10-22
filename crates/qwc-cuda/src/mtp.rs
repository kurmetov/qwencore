//! MTP draft head linear layers: BF16 as-is from the checkpoint.
//!
//! The head isn't quantized (`ignored: .*mtp.*` in the quantization config), so
//! its matrices don't go through the NVFP4 path. Nor do they need to: a draft step reads
//! 849 MB per pass and is bound by bandwidth, not tensor cores.

use crate::error::{Result, check};
use crate::{DeviceBuffer, Stream, ffi};

/// Max rows per call. A draft step works with one row, draft
/// verification with k+1.
pub const MAX_ROWS: usize = 8;

/// `output[rows, n] = input[rows, k] * weights[n, k]^T`.
pub fn bf16_linear(
    weights: &DeviceBuffer<u16>,
    input: &DeviceBuffer<u16>,
    output: &mut DeviceBuffer<u16>,
    rows: usize,
    k: usize,
    n: usize,
    stream: &Stream,
) -> Result<()> {
    assert!((1..=MAX_ROWS).contains(&rows));
    assert_eq!(weights.len(), k * n, "weights [n, k]");
    assert!(input.len() >= rows * k);
    assert!(output.len() >= rows * n);
    check(unsafe {
        ffi::qwc_bf16_linear(
            weights.as_ptr().cast(),
            input.as_ptr().cast(),
            output.as_mut_ptr().cast(),
            rows as i32,
            k as i32,
            n as i32,
            stream.raw(),
        )
    })
}

/// `out = silu(gate) * up` element-wise.
pub fn swiglu(
    gate: &DeviceBuffer<u16>,
    up: &DeviceBuffer<u16>,
    out: &mut DeviceBuffer<u16>,
    elements: usize,
    stream: &Stream,
) -> Result<()> {
    assert!(gate.len() >= elements && up.len() >= elements && out.len() >= elements);
    check(unsafe {
        ffi::qwc_bf16_swiglu(
            gate.as_ptr().cast(),
            up.as_ptr().cast(),
            out.as_mut_ptr().cast(),
            elements as i32,
            stream.raw(),
        )
    })
}

/// Concatenates two normalized halves into the `fc` input: [embedding | hidden].
pub fn concat(
    left: &DeviceBuffer<u16>,
    right: &DeviceBuffer<u16>,
    out: &mut DeviceBuffer<u16>,
    rows: usize,
    width: usize,
    stream: &Stream,
) -> Result<()> {
    assert!(left.len() >= rows * width && right.len() >= rows * width);
    assert!(out.len() >= rows * 2 * width);
    check(unsafe {
        ffi::qwc_bf16_concat(
            left.as_ptr().cast(),
            right.as_ptr().cast(),
            out.as_mut_ptr().cast(),
            rows as i32,
            width as i32,
            stream.raw(),
        )
    })
}

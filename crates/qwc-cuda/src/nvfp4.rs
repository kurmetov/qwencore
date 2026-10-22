//! NVFP4 decode projections.
//!
//! The first production candidate is W4A16 for batch 1..4: checkpoint weights
//! are used directly, the activation stays BF16. This preserves quality
//! relative to W4A4 and removes the activation-quantization launch.

use crate::error::{Result, check};
use crate::{DeviceBuffer, Stream, bf16, ffi};

pub const GROUP_SIZE: usize = 16;
pub const MAX_W4A16_BATCH: usize = 6;
/// How many rows the kernel can handle at all: `kMaxBatch` in `cuda/nvfp4.cu`. The policy
/// (`MAX_W4A16_BATCH`) is set lower — where the step measurement stopped winning.
/// The sweep needs exactly this cap, to see what's beyond the boundary.
pub const MAX_W4A16_KERNEL_BATCH: usize = 8;
pub const MAX_W4A4_BATCH: usize = 128;
/// Row cap of one W4A4 call: the prefill arena capacity
/// (`PREFILL_CHUNK_SIZE`). A limit of buffer sizing, not of the kernel.
pub const MAX_W4A4_ROWS: usize = crate::MAX_STEP_ROWS;
const K_ALIGNMENT: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeKernel {
    W4A16,
    W4A4,
}

/// The boundary measured on the RTX 5090 for Qwen3.8 shapes.
///
/// Up to and including batch 6, W4A16 is used: it doesn't quantize activations, and after
/// amortizing shared loads over two weight rows it no longer loses to
/// tensor cores even on wide projections (`nvfp4bench`, batch 4:
/// q+gate 958 vs 953 GB/s for W4A4, gate/up 1156 vs 980, down 1108
/// vs 1100).
///
/// The boundary sat at 4 until the kernel was instantiated above that. A step measurement
/// (`stepprofile`, pure decode, ms per step) showed that switching to W4A4
/// costs more than the kernel itself:
///
/// | concurrency | W4A16 | W4A4 |
/// |---:|---:|---:|
/// | 5 | **18.63** | 21.67 |
/// | 6 | **20.48** | 21.80 |
/// | 7 | 24.94 | **22.39** |
/// | 8 | 30.52 | **22.27** |
///
/// Before the fix the fifth sequence **reduced** throughput: 4 rows in
/// 16.3 ms is 245 tok/s, while 5 in 21.7 is 231. Above six W4A16 is bound by
/// registers and shared, and tensor cores win.
///
/// The boundary matters not only for speed: W4A4 quantizes activations, and row 0
/// of speculative verification diverges from regular decode by 0.64 in logits
/// vs 0.11 for W4A16. On close logits this flips the argmax.
/// Shape affects the boundary: `down` has a long K and W4A4 wins already from the fifth
/// row, while for `k`/`v` the output is too narrow to fill a tensor tile, and
/// W4A4 loses at any batch. The table is in the function body.
pub fn select_decode_kernel(batch: usize, out_features: usize, in_features: usize) -> DecodeKernel {
    assert!((1..=MAX_W4A4_BATCH).contains(&batch));
    // Narrow output — k and v [1024, 5120]. Tensor cores have nothing to fill
    // a 128x128 tile with there, and W4A4 holds 157 GB/s at any batch vs 206..578 for
    // W4A16. The boundary here isn't in batch, but in the output being too narrow.
    if out_features <= 2048 {
        return match batch <= MAX_W4A16_KERNEL_BATCH {
            true => DecodeKernel::W4A16,
            false => DecodeKernel::W4A4,
        };
    }
    // Long K — down [5120, 17408]. On it W4A16 wins only up to
    // four rows (b=5: 976 vs 1083 GB/s, b=6: 901 vs 1098),
    // because the k tile has to be traversed three times as often as for other shapes.
    if in_features >= 3 * out_features {
        return match batch <= 4 {
            true => DecodeKernel::W4A16,
            false => DecodeKernel::W4A4,
        };
    }
    match batch <= MAX_W4A16_BATCH {
        true => DecodeKernel::W4A16,
        false => DecodeKernel::W4A4,
    }
}

/// An NVFP4 matrix in the original compressed-tensors layout.
pub struct Linear {
    packed: DeviceBuffer<u8>,
    cutlass_scales: DeviceBuffer<u8>,
    weight_global_scale: f32,
    out_features: usize,
    in_features: usize,
}

impl Linear {
    /// Loads the matrix without repacking the 4-bit weights and the scale matrix.
    /// `weight_global_scale` is the value from the checkpoint; there it's a divisor.
    pub fn from_host(
        packed: &[u8],
        scales: &[u8],
        weight_global_scale: f32,
        out_features: usize,
        in_features: usize,
    ) -> Result<Self> {
        assert_shape(
            packed.len(),
            scales.len(),
            weight_global_scale,
            out_features,
            in_features,
        );
        let cutlass_scales = swizzle_block_scales(scales, out_features, in_features);
        Ok(Self {
            packed: DeviceBuffer::from_slice(packed)?,
            cutlass_scales: DeviceBuffer::from_slice(&cutlass_scales)?,
            weight_global_scale,
            out_features,
            in_features,
        })
    }

    /// A matrix with zero data for a bandwidth benchmark of real shapes.
    pub fn zeroed(out_features: usize, in_features: usize) -> Result<Self> {
        assert_shape(
            out_features * in_features / 2,
            out_features * in_features / GROUP_SIZE,
            1.0,
            out_features,
            in_features,
        );
        Ok(Self {
            packed: DeviceBuffer::zeroed(out_features * in_features / 2)?,
            cutlass_scales: DeviceBuffer::zeroed(block_scale_storage_len(
                out_features,
                in_features,
            ))?,
            weight_global_scale: 1.0,
            out_features,
            in_features,
        })
    }

    /// Shared-weight W4A16 GEMV for decode batch 1..4.
    pub fn forward_w4a16(
        &self,
        input: &DeviceBuffer<u16>,
        output: &mut DeviceBuffer<u16>,
        batch: usize,
        stream: &Stream,
    ) -> Result<()> {
        // The cap here is the kernel's, not the policy's: on narrow shapes
        // `select_decode_kernel` picks W4A16 even above `MAX_W4A16_BATCH`.
        assert!((1..=MAX_W4A16_KERNEL_BATCH).contains(&batch));
        assert!(input.len() >= batch * self.in_features);
        assert!(output.len() >= batch * self.out_features);
        check(unsafe {
            ffi::qwc_nvfp4_w4a16(
                self.packed.as_ptr(),
                self.cutlass_scales.as_ptr(),
                input.as_ptr(),
                output.as_mut_ptr(),
                self.out_features as i32,
                self.in_features as i32,
                batch as i32,
                self.weight_global_scale,
                stream.raw(),
            )
        })
    }

    /// The same W4A16, but with geometry and layout from outside — for a shape sweep.
    /// The engine goes through `forward_w4a16`, where everything is chosen by shape.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_w4a16_tuned(
        &self,
        input: &DeviceBuffer<u16>,
        output: &mut DeviceBuffer<u16>,
        batch: usize,
        k_rows: usize,
        k_threads: usize,
        k_tile: usize,
        rows_per_thread: usize,
        stream: &Stream,
    ) -> Result<()> {
        assert!((1..=MAX_W4A16_KERNEL_BATCH).contains(&batch));
        assert!(input.len() >= batch * self.in_features);
        assert!(output.len() >= batch * self.out_features);
        check(unsafe {
            ffi::qwc_nvfp4_w4a16_tuned(
                self.packed.as_ptr(),
                self.cutlass_scales.as_ptr(),
                input.as_ptr(),
                output.as_mut_ptr(),
                self.out_features as i32,
                self.in_features as i32,
                batch as i32,
                k_rows as i32,
                k_threads as i32,
                k_tile as i32,
                rows_per_thread as i32,
                self.weight_global_scale,
                stream.raw(),
            )
        })
    }

    pub fn out_features(&self) -> usize {
        self.out_features
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Full DRAM traffic of the weights, without activations and output.
    pub fn weight_bytes(&self) -> usize {
        self.packed.bytes() + self.out_features * self.in_features / GROUP_SIZE
    }

    /// Actual persistent VRAM including scale-layout padding.
    pub fn resident_bytes(&self) -> usize {
        self.packed.bytes() + self.cutlass_scales.bytes()
    }

    /// A copy of the packed weights back to host — to check the load against the checkpoint
    /// without access to the internal buffer.
    pub fn packed_to_host(&self) -> Result<Vec<u8>> {
        self.packed.to_vec()
    }

    /// SM120 W4A4 tensor-core GEMM. Weights stay in the checkpoint layout;
    /// a permuted copy is needed only for the 8-bit scale factors.
    pub fn forward_w4a4_quantized(
        &self,
        input: &QuantizedActivation,
        output: &mut DeviceBuffer<u16>,
        workspace: &mut W4A4Workspace,
        stream: &Stream,
    ) -> Result<()> {
        assert!((3..=MAX_W4A4_ROWS).contains(&input.batch));
        assert_eq!(input.in_features, self.in_features);
        assert!(output.len() >= input.batch * self.out_features);
        assert!(self.out_features.is_multiple_of(8));
        workspace.assert_shape(input.batch, self.out_features, self.in_features)?;

        let alpha = 1.0 / (input.global_scale * self.weight_global_scale);
        check(unsafe {
            ffi::qwc_nvfp4_w4a4(
                input.packed.as_ptr(),
                self.packed.as_ptr(),
                input.cutlass_scales.as_ptr(),
                self.cutlass_scales.as_ptr(),
                output.as_mut_ptr(),
                workspace.storage.as_mut_ptr(),
                workspace.bytes,
                input.batch as i32,
                self.out_features as i32,
                self.in_features as i32,
                alpha,
                stream.raw(),
            )
        })
    }
}

/// An already quantized decode activation. The constructor is also needed for testing
/// the GEMM independently of the future fused BF16->NVFP4 quantizer.
pub struct QuantizedActivation {
    pub(crate) packed: DeviceBuffer<u8>,
    pub(crate) cutlass_scales: DeviceBuffer<u8>,
    pub(crate) global_scale: f32,
    pub(crate) batch: usize,
    capacity: usize,
    pub(crate) in_features: usize,
}

impl QuantizedActivation {
    /// `scales` are passed in logical row-major `[batch, K/16]` order.
    pub fn from_host(
        packed: &[u8],
        scales: &[u8],
        global_scale: f32,
        batch: usize,
        in_features: usize,
    ) -> Result<Self> {
        assert!((1..=MAX_W4A4_ROWS).contains(&batch));
        assert!(in_features > 0 && in_features.is_multiple_of(K_ALIGNMENT));
        assert_eq!(packed.len(), batch * in_features / 2);
        assert_eq!(scales.len(), batch * in_features / GROUP_SIZE);
        assert!(global_scale.is_finite() && global_scale > 0.0);
        let cutlass_scales = swizzle_block_scales(scales, batch, in_features);
        Ok(Self {
            packed: DeviceBuffer::from_slice(packed)?,
            cutlass_scales: DeviceBuffer::from_slice(&cutlass_scales)?,
            global_scale,
            batch,
            capacity: batch,
            in_features,
        })
    }

    /// Creates reusable quantizer output buffers. Padded scale slots
    /// are zeroed here once and stay zero between decode steps.
    pub fn zeroed(global_scale: f32, batch: usize, in_features: usize) -> Result<Self> {
        assert!((1..=MAX_W4A4_ROWS).contains(&batch));
        assert!(in_features > 0 && in_features.is_multiple_of(K_ALIGNMENT));
        assert!(global_scale.is_finite() && global_scale > 0.0);
        Ok(Self {
            packed: DeviceBuffer::zeroed(batch * in_features / 2)?,
            cutlass_scales: DeviceBuffer::zeroed(block_scale_storage_len(batch, in_features))?,
            global_scale,
            batch,
            capacity: batch,
            in_features,
        })
    }

    /// Quantizes BF16 `[batch,K]` directly into the SM120 GEMM format.
    pub fn quantize_bf16(&mut self, input: &DeviceBuffer<u16>, stream: &Stream) -> Result<()> {
        self.quantize_bf16_with_scale(input, self.global_scale, stream)
    }

    /// Reuses the same resident buffers with another checkpoint activation
    /// scale. Projections sharing an input shape generally have distinct
    /// `input_global_scale` values, so prefill can quantize into one scratch
    /// allocation instead of keeping a copy per projection.
    pub fn quantize_bf16_with_scale(
        &mut self,
        input: &DeviceBuffer<u16>,
        global_scale: f32,
        stream: &Stream,
    ) -> Result<()> {
        assert_eq!(input.len(), self.batch * self.in_features);
        assert!(global_scale.is_finite() && global_scale > 0.0);
        self.global_scale = global_scale;
        check(unsafe {
            ffi::qwc_nvfp4_quantize_bf16(
                input.as_ptr(),
                self.packed.as_mut_ptr(),
                self.cutlass_scales.as_mut_ptr(),
                self.batch as i32,
                self.in_features as i32,
                global_scale,
                stream.raw(),
            )
        })
    }

    /// Quantizes the live prefix of a capacity-sized activation buffer.
    pub fn quantize_bf16_rows(
        &mut self,
        input: &DeviceBuffer<u16>,
        global_scale: f32,
        rows: usize,
        stream: &Stream,
    ) -> Result<()> {
        assert!(rows > 0 && rows <= self.capacity);
        assert!(input.len() >= rows * self.in_features);
        assert!(global_scale.is_finite() && global_scale > 0.0);
        self.batch = rows;
        self.global_scale = global_scale;
        check(unsafe {
            ffi::qwc_nvfp4_quantize_bf16(
                input.as_ptr(),
                self.packed.as_mut_ptr(),
                self.cutlass_scales.as_mut_ptr(),
                rows as i32,
                self.in_features as i32,
                global_scale,
                stream.raw(),
            )
        })
    }

    /// Diagnostic readback in logical `[batch,K/16]` order.
    pub fn to_host_logical(&self) -> Result<(Vec<u8>, Vec<u8>)> {
        let packed = self.packed.to_vec()?;
        let swizzled = self.cutlass_scales.to_vec()?;
        let scales = unswizzle_block_scales(&swizzled, self.batch, self.in_features);
        Ok((packed, scales))
    }
}

/// Preallocated CUTLASS scratch. No allocations on the decode hot path.
pub struct W4A4Workspace {
    storage: DeviceBuffer<u8>,
    bytes: usize,
    max_batch: usize,
}

impl W4A4Workspace {
    pub fn new(batch: usize, out_features: usize, in_features: usize) -> Result<Self> {
        Self::for_shapes(batch, &[(out_features, in_features)])
    }

    /// One allocation large enough for every projection shape in a fixed-M
    /// prefill chunk. CUTLASS currently needs zero bytes for the selected
    /// schedule, but keeping the capacity check here makes that an
    /// implementation detail rather than an executor assumption.
    pub fn for_shapes(batch: usize, shapes: &[(usize, usize)]) -> Result<Self> {
        assert!(batch > 0);
        assert!(!shapes.is_empty());
        // The requirement isn't monotonic in the number of rows: a narrow problem is split along K,
        // and the smaller M, the more blocks take part in the reduction. The arena
        // presents any M up to `batch`, so the max over all of them is taken.
        let mut bytes = 0usize;
        for &(out_features, in_features) in shapes {
            for rows in 1..=batch {
                let mut needed = 0usize;
                check(unsafe {
                    ffi::qwc_nvfp4_w4a4_workspace_size(
                        rows as i32,
                        out_features as i32,
                        in_features as i32,
                        &mut needed,
                    )
                })?;
                bytes = bytes.max(needed);
            }
        }
        Ok(Self {
            // cudaMalloc(0) has no portable semantics; with a zero workspace
            // CUTLASS doesn't dereference the pointer anyway.
            storage: DeviceBuffer::zeroed(bytes.max(1))?,
            bytes,
            max_batch: batch,
        })
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    fn assert_shape(&self, batch: usize, out_features: usize, in_features: usize) -> Result<()> {
        assert!(batch > 0 && batch <= self.max_batch);
        let mut needed = 0usize;
        check(unsafe {
            ffi::qwc_nvfp4_w4a4_workspace_size(
                batch as i32,
                out_features as i32,
                in_features as i32,
                &mut needed,
            )
        })?;
        assert!(needed <= self.bytes, "W4A4 workspace capacity is too small");
        Ok(())
    }
}

/// Elementwise `silu(gate) * up` for the two W4A4 MLP projections.
pub fn swiglu_bf16(
    gate: &DeviceBuffer<u16>,
    up: &DeviceBuffer<u16>,
    output: &mut DeviceBuffer<u16>,
    batch: usize,
    features: usize,
    stream: &Stream,
) -> Result<()> {
    assert!(batch > 0 && features > 0);
    assert!(gate.len() >= batch * features);
    assert!(up.len() >= batch * features);
    assert!(output.len() >= batch * features);
    check(unsafe {
        ffi::qwc_swiglu_bf16(
            gate.as_ptr(),
            up.as_ptr(),
            output.as_mut_ptr(),
            (batch * features) as i32,
            stream.raw(),
        )
    })
}

fn round_up(value: usize, multiple: usize) -> usize {
    value.div_ceil(multiple) * multiple
}

fn block_scale_storage_len(rows: usize, in_features: usize) -> usize {
    round_up(rows, 128) * round_up(in_features / GROUP_SIZE, 4)
}

/// compressed-tensors `[row, K/16]` -> SM120 block-scale 128x4 swizzle.
pub fn swizzle_block_scales(scales: &[u8], rows: usize, in_features: usize) -> Vec<u8> {
    assert_eq!(scales.len(), rows * in_features / GROUP_SIZE);
    let groups = in_features / GROUP_SIZE;
    let group_blocks = round_up(groups, 4) / 4;
    let mut output = vec![0u8; block_scale_storage_len(rows, in_features)];

    for row in 0..rows {
        let row_block = row / 128;
        let row_in_block = row % 128;
        let row_quad = row_in_block / 32;
        let row_in_quad = row_in_block % 32;
        for group in 0..groups {
            let group_block = group / 4;
            let group_in_block = group % 4;
            let dst = ((((row_block * group_blocks + group_block) * 32 + row_in_quad) * 4
                + row_quad)
                * 4)
                + group_in_block;
            output[dst] = scales[row * groups + group];
        }
    }
    output
}

fn unswizzle_block_scales(scales: &[u8], rows: usize, in_features: usize) -> Vec<u8> {
    assert_eq!(scales.len(), block_scale_storage_len(rows, in_features));
    let groups = in_features / GROUP_SIZE;
    let group_blocks = round_up(groups, 4) / 4;
    let mut output = vec![0u8; rows * groups];
    for row in 0..rows {
        let row_block = row / 128;
        let row_in_block = row % 128;
        let row_quad = row_in_block / 32;
        let row_in_quad = row_in_block % 32;
        for group in 0..groups {
            let dst =
                ((((row_block * group_blocks + group / 4) * 32 + row_in_quad) * 4 + row_quad) * 4)
                    + group % 4;
            output[row * groups + group] = scales[dst];
        }
    }
    output
}

/// Fused `silu(gate(x)) * up(x)` for Qwen MLP decode.
pub fn swiglu_w4a16(
    gate: &Linear,
    up: &Linear,
    input: &DeviceBuffer<u16>,
    output: &mut DeviceBuffer<u16>,
    batch: usize,
    stream: &Stream,
) -> Result<()> {
    assert_eq!(gate.out_features, up.out_features);
    assert_eq!(gate.in_features, up.in_features);
    assert!((1..=MAX_W4A16_BATCH).contains(&batch));
    assert!(input.len() >= batch * gate.in_features);
    assert!(output.len() >= batch * gate.out_features);
    check(unsafe {
        ffi::qwc_nvfp4_swiglu_w4a16(
            gate.packed.as_ptr(),
            gate.cutlass_scales.as_ptr(),
            up.packed.as_ptr(),
            up.cutlass_scales.as_ptr(),
            input.as_ptr(),
            output.as_mut_ptr(),
            gate.out_features as i32,
            gate.in_features as i32,
            batch as i32,
            gate.weight_global_scale,
            up.weight_global_scale,
            stream.raw(),
        )
    })
}

fn assert_shape(
    packed_len: usize,
    scale_len: usize,
    weight_global_scale: f32,
    out_features: usize,
    in_features: usize,
) {
    assert!(out_features > 0);
    assert!(in_features > 0 && in_features.is_multiple_of(K_ALIGNMENT));
    assert_eq!(packed_len, out_features * in_features / 2);
    assert_eq!(scale_len, out_features * in_features / GROUP_SIZE);
    assert!(weight_global_scale.is_finite() && weight_global_scale > 0.0);
}

/// CPU oracle for bit-exact layout checks and numeric GPU checks.
pub mod reference {
    use super::*;

    const E2M1: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];

    pub fn e2m1(raw: u8) -> f32 {
        E2M1[(raw & 0x0f) as usize]
    }

    /// NVIDIA E4M3 finite (no infinity; 0x7f/0xff are NaN).
    pub fn e4m3(raw: u8) -> f32 {
        let sign = if raw & 0x80 == 0 { 1.0 } else { -1.0 };
        let exponent = (raw >> 3) & 0x0f;
        let mantissa = raw & 0x07;
        if exponent == 0 {
            return sign * (mantissa as f32) * 2.0f32.powi(-9);
        }
        if exponent == 0x0f && mantissa == 0x07 {
            return f32::NAN;
        }
        sign * (1.0 + mantissa as f32 / 8.0) * 2.0f32.powi(exponent as i32 - 7)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gemv_w4a16(
        packed: &[u8],
        scales: &[u8],
        weight_global_scale: f32,
        input: &[u16],
        output: &mut [f32],
        out_features: usize,
        in_features: usize,
        batch: usize,
    ) {
        assert_shape(
            packed.len(),
            scales.len(),
            weight_global_scale,
            out_features,
            in_features,
        );
        assert_eq!(input.len(), batch * in_features);
        assert_eq!(output.len(), batch * out_features);

        for b in 0..batch {
            for row in 0..out_features {
                let mut acc = 0.0f32;
                for group in 0..in_features / GROUP_SIZE {
                    let scale = e4m3(scales[row * (in_features / GROUP_SIZE) + group])
                        / weight_global_scale;
                    let mut partial = 0.0f32;
                    for j in 0..GROUP_SIZE {
                        let k = group * GROUP_SIZE + j;
                        let byte = packed[row * (in_features / 2) + k / 2];
                        let nibble = if k & 1 == 0 { byte & 0x0f } else { byte >> 4 };
                        partial += e2m1(nibble) * bf16::to_f32(input[b * in_features + k]);
                    }
                    acc += partial * scale;
                }
                output[b * out_features + row] = acc;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gemm_w4a4(
        packed_input: &[u8],
        input_scales: &[u8],
        input_global_scale: f32,
        packed_weight: &[u8],
        weight_scales: &[u8],
        weight_global_scale: f32,
        output: &mut [f32],
        batch: usize,
        out_features: usize,
        in_features: usize,
    ) {
        assert_eq!(packed_input.len(), batch * in_features / 2);
        assert_eq!(input_scales.len(), batch * in_features / GROUP_SIZE);
        assert_shape(
            packed_weight.len(),
            weight_scales.len(),
            weight_global_scale,
            out_features,
            in_features,
        );
        assert_eq!(output.len(), batch * out_features);
        let groups = in_features / GROUP_SIZE;
        let alpha = 1.0 / (input_global_scale * weight_global_scale);

        for b in 0..batch {
            for row in 0..out_features {
                let mut acc = 0.0f32;
                for group in 0..groups {
                    let scale = e4m3(input_scales[b * groups + group])
                        * e4m3(weight_scales[row * groups + group]);
                    let mut partial = 0.0f32;
                    for j in 0..GROUP_SIZE {
                        let k = group * GROUP_SIZE + j;
                        let a_byte = packed_input[b * (in_features / 2) + k / 2];
                        let w_byte = packed_weight[row * (in_features / 2) + k / 2];
                        let a = if k & 1 == 0 {
                            a_byte & 0x0f
                        } else {
                            a_byte >> 4
                        };
                        let w = if k & 1 == 0 {
                            w_byte & 0x0f
                        } else {
                            w_byte >> 4
                        };
                        partial += e2m1(a) * e2m1(w);
                    }
                    acc += partial * scale;
                }
                output[b * out_features + row] = acc * alpha;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::reference::{e2m1, e4m3};

    #[test]
    fn decodes_e2m1_table() {
        assert_eq!(e2m1(0x0), 0.0);
        assert_eq!(e2m1(0x1), 0.5);
        assert_eq!(e2m1(0x7), 6.0);
        assert_eq!(e2m1(0xf), -6.0);
    }

    #[test]
    fn decodes_e4m3_boundaries() {
        assert_eq!(e4m3(0x00), 0.0);
        assert_eq!(e4m3(0x01), 2.0f32.powi(-9));
        assert_eq!(e4m3(0x38), 1.0);
        assert_eq!(e4m3(0x7e), 448.0);
        assert_eq!(e4m3(0xb8), -1.0);
        assert!(e4m3(0x7f).is_nan());
    }

    #[test]
    fn swizzles_128_by_4_scale_tiles() {
        let rows = 129;
        let k = 256;
        let groups = k / super::GROUP_SIZE;
        let input: Vec<u8> = (0..rows * groups).map(|i| i as u8).collect();
        let got = super::swizzle_block_scales(&input, rows, k);
        assert_eq!(got.len(), 256 * groups);

        let index = |row: usize, group: usize| {
            let rb = row / 128;
            let rq = row % 128 / 32;
            let ri = row % 32;
            let gb = group / 4;
            let gi = group % 4;
            ((((rb * (groups / 4) + gb) * 32 + ri) * 4 + rq) * 4) + gi
        };
        for &(row, group) in &[(0, 0), (31, 3), (32, 4), (127, 15), (128, 7)] {
            assert_eq!(got[index(row, group)], input[row * groups + group]);
        }
    }

    #[test]
    fn decode_dispatch_uses_measured_shape_boundary() {
        assert_eq!(
            super::select_decode_kernel(3, 17_408, 5_120),
            super::DecodeKernel::W4A16
        );
        assert_eq!(
            super::select_decode_kernel(4, 17_408, 5_120),
            super::DecodeKernel::W4A16
        );
        assert_eq!(
            super::select_decode_kernel(4, 5_120, 17_408),
            super::DecodeKernel::W4A16
        );
        // down [5120, 17408]: long K, W4A4 wins already from the fifth row.
        assert_eq!(
            super::select_decode_kernel(5, 5_120, 17_408),
            super::DecodeKernel::W4A4
        );
        // la_out [5120, 6144]: the common boundary is six rows.
        assert_eq!(
            super::select_decode_kernel(6, 5_120, 6_144),
            super::DecodeKernel::W4A16
        );
        assert_eq!(
            super::select_decode_kernel(7, 5_120, 6_144),
            super::DecodeKernel::W4A4
        );
        // k and v [1024, 5120]: W4A4 loses at any batch.
        assert_eq!(
            super::select_decode_kernel(8, 1_024, 5_120),
            super::DecodeKernel::W4A16
        );
    }
}

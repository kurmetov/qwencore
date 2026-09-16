//! Decode RMSNorm and fused residual/RMSNorm/NVFP4 producer.

use crate::error::{Result, check};
use crate::nvfp4::QuantizedActivation;
use crate::{DeviceBuffer, Stream, ffi};
use std::ffi::c_void;

/// Qwen3.5 zero-centered RMSNorm: `norm(x) * (1 + weight)`.
pub struct RmsNorm {
    weight: DeviceBuffer<u16>,
    epsilon: f32,
    hidden: usize,
}

impl RmsNorm {
    pub fn from_host(weight: &[u16], epsilon: f32) -> Result<Self> {
        assert!(!weight.is_empty() && weight.len().is_multiple_of(256));
        assert!(epsilon.is_finite() && epsilon > 0.0);
        Ok(Self {
            weight: DeviceBuffer::from_slice(weight)?,
            epsilon,
            hidden: weight.len(),
        })
    }

    /// RMSNorm в BF16. Если передан residual, сначала выполняется
    /// residual = BF16(residual + input), и нормируется именно округлённая сумма.
    pub fn forward_bf16(
        &self,
        input: &DeviceBuffer<u16>,
        mut residual: Option<&mut DeviceBuffer<u16>>,
        output: &mut DeviceBuffer<u16>,
        batch: usize,
        stream: &Stream,
    ) -> Result<()> {
        self.assert_io(input, output.len(), batch);
        let residual_ptr = match residual.as_mut() {
            Some(buffer) => {
                assert!(buffer.len() >= batch * self.hidden);
                buffer.as_mut_ptr()
            }
            None => std::ptr::null_mut::<c_void>(),
        };
        check(unsafe {
            ffi::qwc_rmsnorm_bf16(
                input.as_ptr(),
                residual_ptr,
                self.weight.as_ptr(),
                output.as_mut_ptr(),
                batch as i32,
                self.hidden as i32,
                self.epsilon,
                stream.raw(),
            )
        })
    }

    /// Fused residual-add + RMSNorm + dynamic NVFP4 quantization.
    /// Нормализованный BF16-тензор не материализуется в VRAM.
    pub fn forward_nvfp4(
        &self,
        input: &DeviceBuffer<u16>,
        mut residual: Option<&mut DeviceBuffer<u16>>,
        output: &mut QuantizedActivation,
        stream: &Stream,
    ) -> Result<()> {
        assert!(input.len() >= output.batch * self.hidden);
        assert_eq!(output.in_features, self.hidden);
        let residual_ptr = match residual.as_mut() {
            Some(buffer) => {
                assert!(buffer.len() >= output.batch * self.hidden);
                buffer.as_mut_ptr()
            }
            None => std::ptr::null_mut::<c_void>(),
        };
        check(unsafe {
            ffi::qwc_rmsnorm_nvfp4(
                input.as_ptr(),
                residual_ptr,
                self.weight.as_ptr(),
                output.packed.as_mut_ptr(),
                output.cutlass_scales.as_mut_ptr(),
                output.batch as i32,
                self.hidden as i32,
                self.epsilon,
                output.global_scale,
                stream.raw(),
            )
        })
    }

    fn assert_io(&self, input: &DeviceBuffer<u16>, output_len: usize, batch: usize) {
        assert!((1..=crate::MAX_STEP_ROWS).contains(&batch));
        assert!(input.len() >= batch * self.hidden);
        assert!(output_len >= batch * self.hidden);
    }
}

pub mod reference {
    use crate::bf16;

    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm(
        input: &[u16],
        mut residual: Option<&mut [u16]>,
        weight: &[u16],
        output: &mut [f32],
        batch: usize,
        hidden: usize,
        epsilon: f32,
    ) {
        assert_eq!(input.len(), batch * hidden);
        assert_eq!(weight.len(), hidden);
        assert_eq!(output.len(), batch * hidden);
        if let Some(buffer) = residual.as_ref() {
            assert_eq!(buffer.len(), batch * hidden);
        }

        let mut staged = vec![0.0f32; hidden];
        for row in 0..batch {
            let base = row * hidden;
            let mut sum = 0.0f32;
            for column in 0..hidden {
                let mut value = bf16::to_f32(input[base + column]);
                if let Some(buffer) = residual.as_deref_mut() {
                    value += bf16::to_f32(buffer[base + column]);
                    buffer[base + column] = bf16::from_f32(value);
                    value = bf16::to_f32(buffer[base + column]);
                }
                staged[column] = value;
                sum += value * value;
            }
            let inverse = 1.0 / (sum / hidden as f32 + epsilon).sqrt();
            for column in 0..hidden {
                output[base + column] =
                    staged[column] * inverse * (1.0 + bf16::to_f32(weight[column]));
            }
        }
    }
}

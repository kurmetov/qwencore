//! Storage formats for weights and cache.
//!
//! NVFP4 isn't just "4 bits": every block of 16 elements gets
//! its own float8_e4m3 scale, which gives 4.5 bits per element.
//! Memory planning can't ignore this — at 24B parameters
//! forgotten scales cause a 1.5 GB miss.

pub const NVFP4_BLOCK: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dtype {
    /// Weights and activations in bfloat16.
    Bf16,
    /// float8_e4m3 — block scales, KV cache, optionally embeddings.
    Fp8E4m3,
    /// NVFP4: 4 bits per element + an fp8 scale per block of 16.
    Nvfp4,
    /// int8 with an f32 scale per state row (128 elements). The storage
    /// format of the DeltaNet recurrent state: rows are flat, and on them
    /// it's four times more precise than fp8 for the same byte
    /// (`bench/results/delta-state-8bit.md`).
    Int8Row128,
    Fp32,
}

impl Dtype {
    /// Bytes for `n` elements. Integer arithmetic: for NVFP4 this is
    /// n/2 data bytes plus n/16 scale bytes.
    pub const fn bytes(self, n: usize) -> u64 {
        match self {
            Dtype::Fp32 => (n * 4) as u64,
            Dtype::Bf16 => (n * 2) as u64,
            Dtype::Fp8E4m3 => n as u64,
            Dtype::Nvfp4 => (n / 2 + n.div_ceil(NVFP4_BLOCK)) as u64,
            Dtype::Int8Row128 => (n + n.div_ceil(128) * 4) as u64,
        }
    }

    /// Effective bit width, including scale overhead.
    pub const fn effective_bits(self) -> f32 {
        match self {
            Dtype::Fp32 => 32.0,
            Dtype::Bf16 => 16.0,
            Dtype::Fp8E4m3 => 8.0,
            Dtype::Nvfp4 => 4.5,
            Dtype::Int8Row128 => 8.25,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Dtype::Fp32 => "fp32",
            Dtype::Bf16 => "bf16",
            Dtype::Fp8E4m3 => "fp8_e4m3",
            Dtype::Nvfp4 => "nvfp4",
            Dtype::Int8Row128 => "int8_row128",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvfp4_includes_block_scales() {
        // Exactly one block: 8 data bytes + 1 scale byte.
        assert_eq!(Dtype::Nvfp4.bytes(16), 9);
        // 4.5 bits per element on a large tensor.
        let n = 1_000_000;
        let bits = Dtype::Nvfp4.bytes(n) as f64 * 8.0 / n as f64;
        assert!((bits - 4.5).abs() < 0.01, "{bits} bits/element");
    }

    #[test]
    fn quantized_body_size() {
        use crate::arch::QUANTIZED_PARAMS;
        let gb = Dtype::Nvfp4.bytes(QUANTIZED_PARAMS) as f64 / 1e9;
        assert!((13.0..14.5).contains(&gb), "model body in NVFP4: {gb} GB");
    }
}

//! An analytical performance model of the decode step.
//!
//! Decode is bound by memory bandwidth, not compute:
//! every step reads ALL the model weights. So the tokens/sec ceiling
//! is computed in advance, and it shows how much the kernels actually fall short.
//!
//! CUTLASS NVFP4 GEMM measurements on the RTX 5090 (see docs/03-roofline.md):
//! at M=2048 it reaches 1400 TFLOPS (84% of FP4 peak), but at M=1..32 —
//! only 44-57% of memory bandwidth. That's exactly the gap
//! a specialized GEMV kernel is meant to close.

use crate::dtype::Dtype;
use crate::memory::CacheConfig;

/// The RTX 5090 spec peak: 512 bits @ 28 Gbps.
pub const PEAK_BANDWIDTH: f64 = 1792e9;

/// Actually achievable DRAM read, measured on this card
/// (`cargo run -p qwc-cuda --bin gpuinfo`): 90% of the spec peak.
/// All estimates are based on it — the spec number is unreachable in principle.
pub const ACHIEVABLE_BANDWIDTH: f64 = 1605e9;

/// L2 read (96 MiB): 2.4x faster than DRAM. The engine's working sets
/// are orders of magnitude larger, so it's only applicable within one layer.
pub const L2_BANDWIDTH: f64 = 3886e9;
/// Dense FP4 peak on the RTX 5090, FLOP/s.
pub const PEAK_FP4_FLOPS: f64 = 1.676e15;

#[derive(Debug, Clone, Copy)]
pub struct DecodeStep {
    pub batch: usize,
    pub context_len: usize,
    /// The share of peak bandwidth the kernels actually deliver.
    pub bandwidth_efficiency: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct StepCost {
    /// Weights are read entirely, regardless of batch.
    pub weight_bytes: u64,
    /// Recurrent state: read + write for each sequence.
    pub state_bytes: u64,
    /// KV cache: reading the whole context for each sequence.
    pub kv_bytes: u64,
}

impl StepCost {
    pub fn total(&self) -> u64 {
        self.weight_bytes + self.state_bytes + self.kv_bytes
    }

    /// `efficiency` is the share of achievable bandwidth, not of the spec.
    pub fn seconds(&self, efficiency: f64) -> f64 {
        self.total() as f64 / (ACHIEVABLE_BANDWIDTH * efficiency)
    }
}

impl DecodeStep {
    pub fn cost(&self, streamed_weights: u64, cache: &CacheConfig) -> StepCost {
        // The DeltaNet state is rewritten entirely: we read and write.
        let state = 2 * cache.state_bytes_per_slot() * self.batch as u64;
        let kv = cache.kv_bytes_per_token() * self.context_len as u64 * self.batch as u64;
        StepCost {
            weight_bytes: streamed_weights,
            state_bytes: state,
            kv_bytes: kv,
        }
    }

    /// Aggregate generation speed, tokens per second.
    pub fn tokens_per_sec(&self, streamed_weights: u64, cache: &CacheConfig) -> f64 {
        let t = self
            .cost(streamed_weights, cache)
            .seconds(self.bandwidth_efficiency);
        self.batch as f64 / t
    }

    /// Inter-token latency, milliseconds.
    pub fn itl_ms(&self, streamed_weights: u64, cache: &CacheConfig) -> f64 {
        self.cost(streamed_weights, cache)
            .seconds(self.bandwidth_efficiency)
            * 1e3
    }
}

/// GEMM arithmetic intensity at a given batch: FLOP per weight byte.
/// The transition point from memory-bound to compute-bound is where it equals
/// the ratio of peak compute to peak bandwidth.
pub fn gemm_intensity(m: usize, weight_dtype: Dtype, n: usize, k: usize) -> f64 {
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    flops / weight_dtype.bytes(n * k) as f64
}

/// The batch at which a GEMM stops being memory-bound.
pub fn compute_bound_batch(weight_dtype: Dtype) -> f64 {
    // intensity(M) = 2*M / bytes_per_elem; threshold = PEAK_FLOPS / PEAK_BW
    let bytes_per_elem = weight_dtype.effective_bits() as f64 / 8.0;
    (PEAK_FP4_FLOPS / ACHIEVABLE_BANDWIDTH) * bytes_per_elem / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::{HIDDEN_SIZE, INTERMEDIATE_SIZE};
    use crate::memory::WeightPlan;

    #[test]
    fn single_stream_decode_ceiling() {
        let w = WeightPlan::default().decode_weight_bytes(1);
        let cache = CacheConfig::default();

        // An ideal kernel: the whole memory bandwidth.
        let ideal = DecodeStep {
            batch: 1,
            context_len: 2048,
            bandwidth_efficiency: 1.0,
        };
        let tps = ideal.tokens_per_sec(w, &cache);
        assert!(
            (100.0..112.0).contains(&tps),
            "batch=1 ceiling: {tps:.0} tok/s"
        );

        // What CUTLASS GEMM gives at M=1 (measured: 47%).
        let real = DecodeStep {
            bandwidth_efficiency: 0.47,
            ..ideal
        };
        let tps_real = real.tokens_per_sec(w, &cache);
        assert!(tps_real < tps * 0.5);
    }

    #[test]
    fn state_traffic_is_a_tenth_of_the_step_at_high_batch() {
        let w = WeightPlan::default().decode_weight_bytes(32);
        let cache = CacheConfig::default();
        let step = DecodeStep {
            batch: 32,
            context_len: 2048,
            bandwidth_efficiency: 0.8,
        };
        let c = step.cost(w, &cache);

        // At batch=32 the DeltaNet state is read and written 32 times, and that's
        // still more than reading the KV cache.
        assert!(
            c.state_bytes > c.kv_bytes,
            "state {} vs KV {}",
            c.state_bytes,
            c.kv_bytes
        );
        // This used to be a fifth of the step. 8-bit storage cut the share
        // in half: the measured step at c=32 confirmed this separately —
        // `delta.scan` 1.963 -> 1.786 ms with half the traffic
        // (`bench/results/delta-state-8bit.md`).
        let share = c.state_bytes as f64 / c.total() as f64;
        assert!((0.08..0.20).contains(&share), "state share {share:.2}");

        let wide = CacheConfig {
            state_dtype: Dtype::Bf16,
            ..cache
        };
        let wide_share =
            step.cost(w, &wide).state_bytes as f64 / step.cost(w, &wide).total() as f64;
        assert!(
            wide_share > share * 1.6,
            "bf16 {wide_share:.2} vs {share:.2}"
        );
    }

    #[test]
    fn bf16_state_saves_real_time() {
        let w = WeightPlan::default().decode_weight_bytes(32);
        let step = DecodeStep {
            batch: 32,
            context_len: 2048,
            bandwidth_efficiency: 0.8,
        };

        let bf16 = CacheConfig::default();
        let fp32 = CacheConfig {
            state_dtype: Dtype::Fp32,
            ..bf16
        };

        let saved = step.itl_ms(w, &fp32) - step.itl_ms(w, &bf16);
        // Converting the state to bf16 saves milliseconds on every step.
        assert!(saved > 2.0, "only {saved:.2} ms saved");
    }

    #[test]
    fn decode_is_memory_bound_until_large_batch() {
        let b = compute_bound_batch(Dtype::Nvfp4);
        // With NVFP4 the transition to compute-bound happens around batch ~260,
        // i.e. our whole 1..32 range is purely memory-bound.
        assert!((200.0..350.0).contains(&b), "transition at batch {b:.0}");

        let i32 = gemm_intensity(32, Dtype::Nvfp4, INTERMEDIATE_SIZE, HIDDEN_SIZE);
        let threshold = PEAK_FP4_FLOPS / ACHIEVABLE_BANDWIDTH;
        assert!(i32 < threshold);
    }
}

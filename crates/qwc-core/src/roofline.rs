//! Аналитическая модель производительности decode-шага.
//!
//! Decode упирается в пропускную способность памяти, а не в вычисления:
//! на каждый шаг читаются ВСЕ веса модели. Поэтому потолок tokens/sec
//! считается заранее, и по нему видно, сколько реально недобирают кернелы.
//!
//! Замеры CUTLASS NVFP4 GEMM на RTX 5090 (см. docs/03-roofline.md):
//! при M=2048 достигается 1400 TFLOPS (84% пика FP4), но при M=1..32 —
//! только 44-57% пропускной способности памяти. Именно этот разрыв
//! специализированный GEMV-кернел и должен закрыть.

use crate::dtype::Dtype;
use crate::memory::CacheConfig;

/// Пиковая пропускная способность RTX 5090: 512 бит @ 28 Gbps.
pub const PEAK_BANDWIDTH: f64 = 1792e9;
/// Пик плотного FP4 на RTX 5090, FLOP/s.
pub const PEAK_FP4_FLOPS: f64 = 1.676e15;

#[derive(Debug, Clone, Copy)]
pub struct DecodeStep {
    pub batch: usize,
    pub context_len: usize,
    /// Доля пиковой пропускной способности, которую реально дают кернелы.
    pub bandwidth_efficiency: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct StepCost {
    /// Веса читаются целиком, независимо от batch.
    pub weight_bytes: u64,
    /// Рекуррентное состояние: чтение + запись на каждую последовательность.
    pub state_bytes: u64,
    /// KV-кэш: чтение всего контекста на каждую последовательность.
    pub kv_bytes: u64,
}

impl StepCost {
    pub fn total(&self) -> u64 {
        self.weight_bytes + self.state_bytes + self.kv_bytes
    }

    pub fn seconds(&self, efficiency: f64) -> f64 {
        self.total() as f64 / (PEAK_BANDWIDTH * efficiency)
    }
}

impl DecodeStep {
    pub fn cost(&self, weights: u64, cache: &CacheConfig) -> StepCost {
        // Состояние DeltaNet перезаписывается целиком: читаем и пишем.
        let state = 2 * cache.state_bytes_per_slot() * self.batch as u64;
        let kv = cache.kv_bytes_per_token() * self.context_len as u64 * self.batch as u64;
        StepCost {
            weight_bytes: weights,
            state_bytes: state,
            kv_bytes: kv,
        }
    }

    /// Совокупная скорость генерации, токенов в секунду.
    pub fn tokens_per_sec(&self, weights: u64, cache: &CacheConfig) -> f64 {
        let t = self.cost(weights, cache).seconds(self.bandwidth_efficiency);
        self.batch as f64 / t
    }

    /// Межтокенная задержка, миллисекунды.
    pub fn itl_ms(&self, weights: u64, cache: &CacheConfig) -> f64 {
        self.cost(weights, cache).seconds(self.bandwidth_efficiency) * 1e3
    }
}

/// Арифметическая интенсивность GEMM при данном batch: FLOP на байт весов.
/// Точка перехода из memory-bound в compute-bound — там, где она сравнивается
/// с отношением пика вычислений к пику пропускной способности.
pub fn gemm_intensity(m: usize, weight_dtype: Dtype, n: usize, k: usize) -> f64 {
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    flops / weight_dtype.bytes(n * k) as f64
}

/// Batch, при котором GEMM перестаёт быть memory-bound.
pub fn compute_bound_batch(weight_dtype: Dtype) -> f64 {
    // intensity(M) = 2*M / bytes_per_elem; порог = PEAK_FLOPS / PEAK_BW
    let bytes_per_elem = weight_dtype.effective_bits() as f64 / 8.0;
    (PEAK_FP4_FLOPS / PEAK_BANDWIDTH) * bytes_per_elem / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::{HIDDEN_SIZE, INTERMEDIATE_SIZE};
    use crate::memory::WeightPlan;

    #[test]
    fn single_stream_decode_ceiling() {
        let w = WeightPlan::default().bytes();
        let cache = CacheConfig::default();

        // Идеальный кернел: вся полоса памяти.
        let ideal = DecodeStep { batch: 1, context_len: 2048, bandwidth_efficiency: 1.0 };
        let tps = ideal.tokens_per_sec(w, &cache);
        assert!((95.0..115.0).contains(&tps), "потолок batch=1: {tps:.0} tok/s");

        // То, что даёт CUTLASS GEMM на M=1 (замерено: 47%).
        let real = DecodeStep { bandwidth_efficiency: 0.47, ..ideal };
        let tps_real = real.tokens_per_sec(w, &cache);
        assert!(tps_real < tps * 0.5);
    }

    #[test]
    fn state_traffic_dominates_at_high_batch() {
        let w = WeightPlan::default().bytes();
        let cache = CacheConfig::default();
        let step = DecodeStep { batch: 32, context_len: 2048, bandwidth_efficiency: 0.8 };
        let c = step.cost(w, &cache);

        // При batch=32 состояние DeltaNet читается и пишется 32 раза:
        // это сопоставимо с чтением KV-кэша и заметная доля всего шага.
        assert!(c.state_bytes > c.kv_bytes, "состояние {} vs KV {}", c.state_bytes, c.kv_bytes);
        let share = c.state_bytes as f64 / c.total() as f64;
        assert!((0.15..0.35).contains(&share), "доля состояния {share:.2}");
    }

    #[test]
    fn bf16_state_saves_real_time() {
        let w = WeightPlan::default().bytes();
        let step = DecodeStep { batch: 32, context_len: 2048, bandwidth_efficiency: 0.8 };

        let bf16 = CacheConfig::default();
        let fp32 = CacheConfig { state_dtype: Dtype::Fp32, ..bf16 };

        let saved = step.itl_ms(w, &fp32) - step.itl_ms(w, &bf16);
        // Перевод состояния в bf16 экономит миллисекунды на каждом шаге.
        assert!(saved > 2.0, "экономия всего {saved:.2} мс");
    }

    #[test]
    fn decode_is_memory_bound_until_large_batch() {
        let b = compute_bound_batch(Dtype::Nvfp4);
        // При NVFP4 переход в compute-bound происходит в районе batch ~260,
        // то есть весь наш диапазон 1..32 — чисто memory-bound.
        assert!((200.0..350.0).contains(&b), "переход при batch {b:.0}");

        let i32 = gemm_intensity(32, Dtype::Nvfp4, INTERMEDIATE_SIZE, HIDDEN_SIZE);
        let threshold = PEAK_FP4_FLOPS / PEAK_BANDWIDTH;
        assert!(i32 < threshold);
    }
}

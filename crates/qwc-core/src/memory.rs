//! VRAM planning for a single RTX 5090 (32 GB).
//!
//! The key difference from an ordinary transformer: per-sequence memory
//! consists of TWO independent parts.
//!
//!   1. The DeltaNet recurrent state (48 layers) — a constant per slot,
//!      independent of context length: ~39 MB @ int8, ~81 MB @ bf16,
//!      ~157 MB @ fp32. Beyond the slots there's one dequantized copy per
//!      engine, ~76 MB, on which prefill runs.
//!   2. The paged KV cache (16 layers) — grows linearly: 32 KB per token @ fp8.
//!
//! The break-even point is around 2.5K tokens. Below it concurrency is limited
//! by the number of state slots, above it by the KV cache. The scheduler must account for
//! both parts, otherwise admission control is off by multiples.

use crate::arch::*;
use crate::dtype::Dtype;

pub const GB: u64 = 1_000_000_000;
pub const MIB: u64 = 1024 * 1024;

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

/// Weight layout by format. The defaults correspond to
/// our text-only load of the QUASAR checkpoint.
#[derive(Debug, Clone, Copy)]
pub struct WeightPlan {
    /// All linear layers of the model body.
    pub body: Dtype,
    /// `lm_head` isn't quantized in the checkpoint (BF16); we convert it to FP8.
    pub lm_head: Dtype,
    /// `embed_tokens` is a lookup, precision is not critical.
    pub embed: Dtype,
    /// The MTP draft head for speculative decoding.
    pub mtp: Option<Dtype>,
    /// Vision tower. Always `false` in a text-only engine.
    pub vision: bool,
}

impl Default for WeightPlan {
    fn default() -> Self {
        Self {
            body: Dtype::Nvfp4,
            lm_head: Dtype::Fp8E4m3,
            embed: Dtype::Fp8E4m3,
            mtp: Some(Dtype::Bf16),
            vision: false,
        }
    }
}

impl WeightPlan {
    /// How the checkpoint lies on disk: lm_head and embeddings in BF16,
    /// vision tower and MTP loaded. This is what vLLM loads.
    pub fn as_shipped() -> Self {
        Self {
            body: Dtype::Nvfp4,
            lm_head: Dtype::Bf16,
            embed: Dtype::Bf16,
            mtp: Some(Dtype::Bf16),
            vision: true,
        }
    }

    /// The full resident weight footprint. This quantity is needed for the VRAM budget,
    /// but is not the traffic of one decode step: a regular step doesn't read MTP,
    /// the vision tower or the whole embedding table.
    pub fn bytes(&self) -> u64 {
        let mut total = self.body.bytes(QUANTIZED_PARAMS);
        total += self.lm_head.bytes(LM_HEAD_PARAMS);
        total += self.embed.bytes(EMBED_PARAMS);
        if let Some(d) = self.mtp {
            total += d.bytes(MTP_PARAMS_EST);
        }
        if self.vision {
            total += Dtype::Bf16.bytes(VISION_PARAMS_EST);
        }
        total
    }

    /// Weight bytes a regular decode step actually reads from
    /// memory. The model body and `lm_head` are scanned entirely, while the embedding does
    /// a lookup of one row per sequence. MTP and vision don't run in regular
    /// decode.
    pub fn decode_weight_bytes(&self, batch: usize) -> u64 {
        self.body.bytes(QUANTIZED_PARAMS)
            + self.lm_head.bytes(LM_HEAD_PARAMS)
            + self.embed.bytes(HIDDEN_SIZE * batch)
    }
}

// ---------------------------------------------------------------------------
// Cache
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct CacheConfig {
    /// KV cache format for the 16 full-attention layers.
    pub kv_dtype: Dtype,
    /// Recurrent state format of resting slots. The model config
    /// requires fp32; the engine stores int8 with a per-row scale, and prefill
    /// runs on a separate bf16 copy (see `state_work_bytes`).
    pub state_dtype: Dtype,
    /// Paged KV cache block size in tokens.
    pub block_size: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            kv_dtype: Dtype::Fp8E4m3,
            state_dtype: Dtype::Int8Row128,
            block_size: 64,
        }
    }
}

impl CacheConfig {
    /// KV cache bytes per token across all full-attention layers.
    pub fn kv_bytes_per_token(&self) -> u64 {
        self.kv_dtype.bytes(KV_ELEMS_PER_TOKEN)
    }

    pub fn kv_bytes_per_block(&self) -> u64 {
        self.kv_bytes_per_token() * self.block_size as u64
    }

    /// Bytes per state slot: DeltaNet recurrent state + conv window.
    /// The conv state is always fp32 — it's tiny, no point saving on it.
    pub fn state_bytes_per_slot(&self) -> u64 {
        self.state_dtype.bytes(STATE_ELEMS_PER_SEQ) + Dtype::Fp32.bytes(CONV_ELEMS_PER_SEQ)
    }

    /// Dequantized state copy of one sequence. There's one per
    /// engine, not per slot, so it's not part of a slot's cost — but it is part
    /// of the card's budget, and with two slots it eats all the savings.
    pub fn state_work_bytes(&self) -> u64 {
        Dtype::Bf16.bytes(STATE_ELEMS_PER_SEQ)
    }

    /// Full cost of a sequence of the given length.
    pub fn bytes_per_seq(&self, context_len: usize) -> u64 {
        let blocks = context_len.div_ceil(self.block_size) as u64;
        self.state_bytes_per_slot() + blocks * self.kv_bytes_per_block()
    }

    /// The context length at which the KV cache equals the recurrent state
    /// in size. Below it the "expensive" part is the states.
    pub fn crossover_tokens(&self) -> usize {
        (self.state_bytes_per_slot() / self.kv_bytes_per_token()) as usize
    }
}

// ---------------------------------------------------------------------------
// Budget
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct Budget {
    pub total_vram: u64,
    /// Taken before the engine starts: CUDA context, driver, other processes
    /// (on a desktop gnome-shell easily eats more than a gigabyte).
    pub reserved: u64,
    pub weights: u64,
    /// Activations, CUTLASS workspace, sampling buffers, CUDA graphs.
    pub workspace: u64,
}

impl Budget {
    /// Memory available for the KV cache and state slots.
    pub fn cache_available(&self) -> u64 {
        self.total_vram
            .saturating_sub(self.reserved)
            .saturating_sub(self.weights)
            .saturating_sub(self.workspace)
    }

    /// How many sequences of the given length fit.
    pub fn max_concurrency(&self, cfg: &CacheConfig, context_len: usize) -> usize {
        let per_seq = cfg.bytes_per_seq(context_len);
        if per_seq == 0 {
            return 0;
        }
        (self.cache_available() / per_seq) as usize
    }

    /// Maximum context for a given number of sequences.
    pub fn max_context(&self, cfg: &CacheConfig, concurrency: usize) -> usize {
        if concurrency == 0 {
            return 0;
        }
        let per_seq = self.cache_available() / concurrency as u64;
        let for_kv = per_seq.saturating_sub(cfg.state_bytes_per_slot());
        (for_kv / cfg.kv_bytes_per_token()) as usize
    }

    pub fn fits(&self, cfg: &CacheConfig, concurrency: usize, context_len: usize) -> bool {
        cfg.bytes_per_seq(context_len) * concurrency as u64 <= self.cache_available()
    }
}

/// RTX 5090 budget with the desktop running.
/// nvidia-smi shows 32607 MiB total and ~1238 MiB taken by gnome-shell.
pub fn rtx5090(weights: u64, workspace: u64) -> Budget {
    Budget {
        total_vram: 32_607 * MIB,
        reserved: 2_100 * MIB, // desktop + CUDA context
        weights,
        workspace,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_budget() -> Budget {
        rtx5090(WeightPlan::default().bytes(), 2 * GB)
    }

    #[test]
    fn weight_plan_matches_checkpoint_size() {
        // On disk the QUASAR checkpoint takes 20.6 GB.
        let shipped = WeightPlan::as_shipped().bytes() as f64 / 1e9;
        assert!((20.0..21.2).contains(&shipped), "as_shipped: {shipped} GB");

        // After our transformations — noticeably less.
        let ours = WeightPlan::default().bytes() as f64 / 1e9;
        assert!((16.5..17.5).contains(&ours), "default: {ours} GB");

        let saved = shipped - ours;
        assert!(saved > 3.0, "only {saved} GB saved, expected >3");

        // The resident footprint includes MTP and the full embedding table, but
        // regular batch-1 decode doesn't stream them in full.
        let decode = WeightPlan::default().decode_weight_bytes(1) as f64 / 1e9;
        assert!(
            (14.5..15.5).contains(&decode),
            "decode traffic: {decode} GB"
        );
        assert!(
            decode < ours - 1.5,
            "resident {ours} GB vs decode {decode} GB"
        );
    }

    #[test]
    fn state_dominates_below_crossover() {
        let cfg = CacheConfig::default();
        let x = cfg.crossover_tokens();
        // 8-bit state equals the KV cache at about 1.3K tokens: a slot half as
        // expensive brings the break-even point twice as close, and the range where
        // concurrency is limited by slots rather than KV narrowed accordingly.
        assert!((1000..1700).contains(&x), "crossover point: {x}");

        let wide = CacheConfig {
            state_dtype: Dtype::Bf16,
            ..cfg
        };
        assert!(
            wide.crossover_tokens() > x * 3 / 2,
            "bf16 {} vs int8 {x}",
            wide.crossover_tokens()
        );

        // Below it the state is more expensive than KV.
        let short: usize = 512;
        let blocks = short.div_ceil(cfg.block_size) as u64;
        assert!(cfg.state_bytes_per_slot() > blocks * cfg.kv_bytes_per_block());
    }

    #[test]
    fn fp32_state_shifts_crossover_much_higher() {
        let cfg = CacheConfig {
            state_dtype: Dtype::Fp32,
            ..CacheConfig::default()
        };
        let x = cfg.crossover_tokens();
        assert!((4200..5600).contains(&x), "fp32 crossover point: {x}");
    }

    #[test]
    fn benchmark_matrix_feasibility() {
        let b = default_budget();
        let cfg = CacheConfig::default();

        // Target matrix: everything up to 32x2K must fit.
        assert!(b.fits(&cfg, 1, 512));
        assert!(b.fits(&cfg, 1, 16 * 1024));
        assert!(b.fits(&cfg, 4, 2048));
        assert!(b.fits(&cfg, 8, 2048));
        assert!(b.fits(&cfg, 16, 2048));
        assert!(b.fits(&cfg, 32, 2048));
    }

    #[test]
    fn long_context_concurrency_limits() {
        let b = default_budget();

        let fp8 = CacheConfig::default();
        let n_fp8 = b.max_concurrency(&fp8, 32 * 1024);
        assert!((8..=14).contains(&n_fp8), "fp8 KV @32K: {n_fp8} seq");

        // An NVFP4 KV cache nearly doubles concurrency on long context.
        let fp4 = CacheConfig {
            kv_dtype: Dtype::Nvfp4,
            ..CacheConfig::default()
        };
        let n_fp4 = b.max_concurrency(&fp4, 32 * 1024);
        assert!(
            n_fp4 as f64 / n_fp8 as f64 > 1.5,
            "fp4 {n_fp4} vs fp8 {n_fp8}"
        );
    }

    #[test]
    fn our_weight_savings_buy_concurrency() {
        let cfg = CacheConfig::default();
        let ours = rtx5090(WeightPlan::default().bytes(), 2 * GB);
        let vllm_like = rtx5090(WeightPlan::as_shipped().bytes(), 2 * GB);

        let ctx = 32 * 1024;
        let gain = ours.max_concurrency(&cfg, ctx) - vllm_like.max_concurrency(&cfg, ctx);
        assert!(gain >= 3, "gain of only {gain} sequences");
    }
}

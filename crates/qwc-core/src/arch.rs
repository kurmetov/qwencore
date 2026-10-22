//! Qwen3.8-27B architecture (`Qwen3_5ForConditionalGeneration`, text-only).
//!
//! Source: `Qwen/Qwen3.8-27B/config.json`, the `text_config` section.
//! The vision tower is intentionally absent — the engine is text-only.

// ---------------------------------------------------------------------------
// Base stack
// ---------------------------------------------------------------------------

pub const NUM_LAYERS: usize = 64;
pub const HIDDEN_SIZE: usize = 5120;
pub const INTERMEDIATE_SIZE: usize = 17408;
pub const VOCAB_SIZE: usize = 248_320;
pub const RMS_NORM_EPS: f32 = 1e-6;
pub const MAX_POSITION_EMBEDDINGS: usize = 262_144;

/// lm_head weights aren't tied to embed_tokens: that's a separate 1.27B parameters.
pub const TIE_WORD_EMBEDDINGS: bool = false;

// ---------------------------------------------------------------------------
// Hybrid layout: [linear, linear, linear, full] x 16
// ---------------------------------------------------------------------------

pub const FULL_ATTENTION_INTERVAL: usize = 4;
pub const NUM_FULL_LAYERS: usize = NUM_LAYERS / FULL_ATTENTION_INTERVAL;
pub const NUM_LINEAR_LAYERS: usize = NUM_LAYERS - NUM_FULL_LAYERS;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// Gated DeltaNet: a constant-size recurrent state.
    LinearAttention,
    /// Classic GQA attention with a paged KV cache.
    FullAttention,
}

/// Layer type by index. Full-attention is every 4th: 3, 7, 11, ..., 63.
pub const fn layer_kind(idx: usize) -> LayerKind {
    debug_assert!(idx < NUM_LAYERS);
    if (idx + 1).is_multiple_of(FULL_ATTENTION_INTERVAL) {
        LayerKind::FullAttention
    } else {
        LayerKind::LinearAttention
    }
}

/// The layer's ordinal among layers of its type — the index in the KV cache
/// (for full) or in the state pool (for linear).
pub const fn layer_slot(idx: usize) -> usize {
    match layer_kind(idx) {
        LayerKind::FullAttention => idx / FULL_ATTENTION_INTERVAL,
        LayerKind::LinearAttention => idx - (idx / FULL_ATTENTION_INTERVAL),
    }
}

// ---------------------------------------------------------------------------
// Full attention (16 layers)
// ---------------------------------------------------------------------------

pub const NUM_ATTN_HEADS: usize = 24;
pub const NUM_KV_HEADS: usize = 4;
/// Atypically large: almost all off-the-shelf attention kernels are designed for <= 128.
pub const ATTN_HEAD_DIM: usize = 256;
pub const GQA_GROUP: usize = NUM_ATTN_HEADS / NUM_KV_HEADS;

pub const Q_PROJ_DIM: usize = NUM_ATTN_HEADS * ATTN_HEAD_DIM;
pub const KV_PROJ_DIM: usize = NUM_KV_HEADS * ATTN_HEAD_DIM;

/// `attn_output_gate: true` — an extra 5120 -> 6144 projection,
/// a swish gate on the attention output before o_proj.
pub const ATTN_OUTPUT_GATE: bool = true;

/// `partial_rotary_factor: 0.25` — RoPE is applied to the first 64 of 256 dims,
/// the other 192 pass through.
pub const ROPE_DIM: usize = ATTN_HEAD_DIM / 4;
pub const ROPE_THETA: f64 = 10_000_000.0;

/// mrope with sections [11, 11, 10] degenerates into plain RoPE when all three
/// position indices are equal — and in text-only they're always equal.
/// So mrope isn't implemented in the engine at all.
pub const MROPE_SECTIONS: [usize; 3] = [11, 11, 10];

// ---------------------------------------------------------------------------
// Linear attention / Gated DeltaNet (48 layers)
// ---------------------------------------------------------------------------

pub const LA_NUM_K_HEADS: usize = 16;
pub const LA_NUM_V_HEADS: usize = 48;
pub const LA_K_HEAD_DIM: usize = 128;
pub const LA_V_HEAD_DIM: usize = 128;
/// k/q heads are broadcast onto v heads 1:3.
pub const LA_HEAD_RATIO: usize = LA_NUM_V_HEADS / LA_NUM_K_HEADS;

pub const LA_QK_PROJ_DIM: usize = LA_NUM_K_HEADS * LA_K_HEAD_DIM;
pub const LA_V_PROJ_DIM: usize = LA_NUM_V_HEADS * LA_V_HEAD_DIM;

/// A short causal conv1d before the recurrence, over the q|k|v channels.
pub const LA_CONV_KERNEL: usize = 4;
pub const LA_CONV_CHANNELS: usize = LA_QK_PROJ_DIM * 2 + LA_V_PROJ_DIM;

// ---------------------------------------------------------------------------
// MTP (speculative decoding)
// ---------------------------------------------------------------------------

/// The draft head is built into the checkpoint: speculative decoding needs no
/// separate draft model.
pub const MTP_NUM_LAYERS: usize = 1;
pub const MTP_DEDICATED_EMBEDDINGS: bool = false;

// ---------------------------------------------------------------------------
// Derived cache quantities
// ---------------------------------------------------------------------------

/// KV cache elements per token across all full-attention layers.
/// 16 layers x 4 kv heads x 256 dim x 2 (K and V) = 32768.
pub const KV_ELEMS_PER_TOKEN: usize = NUM_FULL_LAYERS * NUM_KV_HEADS * ATTN_HEAD_DIM * 2;

/// DeltaNet recurrent state elements per sequence.
/// 48 layers x 48 v heads x 128 x 128 = 37 748 736. Independent of context length.
pub const STATE_ELEMS_PER_SEQ: usize =
    NUM_LINEAR_LAYERS * LA_NUM_V_HEADS * LA_K_HEAD_DIM * LA_V_HEAD_DIM;

/// Conv state elements per sequence (a window of kernel-1).
pub const CONV_ELEMS_PER_SEQ: usize = NUM_LINEAR_LAYERS * LA_CONV_CHANNELS * (LA_CONV_KERNEL - 1);

// ---------------------------------------------------------------------------
// Parameter count
// ---------------------------------------------------------------------------

pub const MLP_PARAMS_PER_LAYER: usize = 3 * HIDDEN_SIZE * INTERMEDIATE_SIZE;

pub const FULL_ATTN_PARAMS_PER_LAYER: usize = HIDDEN_SIZE * Q_PROJ_DIM        // q_proj
    + HIDDEN_SIZE * KV_PROJ_DIM                                               // k_proj
    + HIDDEN_SIZE * KV_PROJ_DIM                                               // v_proj
    + Q_PROJ_DIM * HIDDEN_SIZE                                                // o_proj
    + HIDDEN_SIZE * Q_PROJ_DIM; // output gate

pub const LINEAR_ATTN_PARAMS_PER_LAYER: usize = HIDDEN_SIZE * LA_QK_PROJ_DIM  // q_proj
    + HIDDEN_SIZE * LA_QK_PROJ_DIM                                            // k_proj
    + HIDDEN_SIZE * LA_V_PROJ_DIM                                             // v_proj
    + LA_V_PROJ_DIM * HIDDEN_SIZE                                             // out_proj
    + HIDDEN_SIZE * LA_V_PROJ_DIM; // output gate

pub const EMBED_PARAMS: usize = VOCAB_SIZE * HIDDEN_SIZE;
pub const LM_HEAD_PARAMS: usize = VOCAB_SIZE * HIDDEN_SIZE;

/// Parameters quantized to NVFP4 (all linear layers of the model body).
pub const QUANTIZED_PARAMS: usize = NUM_LAYERS * MLP_PARAMS_PER_LAYER
    + NUM_FULL_LAYERS * FULL_ATTN_PARAMS_PER_LAYER
    + NUM_LINEAR_LAYERS * LINEAR_ATTN_PARAMS_PER_LAYER;

/// Total parameter count of the text model (without the vision tower and the MTP head).
pub const TEXT_PARAMS: usize = QUANTIZED_PARAMS + EMBED_PARAMS + LM_HEAD_PARAMS;

/// Estimate of the MTP draft head size: one block (attention + MLP) plus
/// the projection of the concatenation [hidden_state; embedding] -> hidden.
/// `mtp_use_dedicated_embeddings: false`, so it has no embeddings of its own.
/// Refined from `model.safetensors.index.json` when the checkpoint is loaded.
pub const MTP_PARAMS_EST: usize =
    FULL_ATTN_PARAMS_PER_LAYER + MLP_PARAMS_PER_LAYER + 2 * HIDDEN_SIZE * HIDDEN_SIZE;

/// Estimate of the vision tower size (depth 27, hidden 1152, intermediate 4304,
/// patch 16, spatial_merge 2, out_hidden 5120). Not loaded in text-only,
/// needed only to show the savings relative to the baseline.
pub const VISION_PARAMS_EST: usize = 27 * (4 * 1152 * 1152 + 2 * 1152 * 4304)  // blocks
    + 3 * 16 * 16 * 2 * 1152                                                   // patch embed
    + 2304 * 1152                                                              // pos embed
    + 1152 * 4 * HIDDEN_SIZE; // merger

// ---------------------------------------------------------------------------
// Compile-time checks
// ---------------------------------------------------------------------------

const _: () = {
    assert!(NUM_FULL_LAYERS == 16);
    assert!(NUM_LINEAR_LAYERS == 48);
    assert!(GQA_GROUP == 6);
    assert!(LA_HEAD_RATIO == 3);
    assert!(ROPE_DIM == 64);
    assert!(KV_ELEMS_PER_TOKEN == 32_768);
    assert!(STATE_ELEMS_PER_SEQ == 37_748_736);
    // The mrope sections cover exactly half of the RoPE dimension (dim/2 pairs).
    assert!(MROPE_SECTIONS[0] + MROPE_SECTIONS[1] + MROPE_SECTIONS[2] == ROPE_DIM / 2);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_layout_matches_config() {
        // layer_types from config.json: full_attention at indices 3, 7, ..., 63.
        let full: Vec<usize> = (0..NUM_LAYERS)
            .filter(|&i| layer_kind(i) == LayerKind::FullAttention)
            .collect();
        assert_eq!(full.len(), 16);
        assert_eq!(full[0], 3);
        assert_eq!(full[15], 63);
    }

    #[test]
    fn layer_slots_are_dense_and_unique() {
        let mut full = vec![];
        let mut linear = vec![];
        for i in 0..NUM_LAYERS {
            match layer_kind(i) {
                LayerKind::FullAttention => full.push(layer_slot(i)),
                LayerKind::LinearAttention => linear.push(layer_slot(i)),
            }
        }
        assert_eq!(full, (0..NUM_FULL_LAYERS).collect::<Vec<_>>());
        assert_eq!(linear, (0..NUM_LINEAR_LAYERS).collect::<Vec<_>>());
    }

    #[test]
    fn parameter_count_is_27b() {
        // The stated 27B: we agree within 0.5%.
        let b = TEXT_PARAMS as f64 / 1e9;
        assert!((26.8..27.1).contains(&b), "got {b} B");
    }

    #[test]
    fn mlp_dominates_weights() {
        let mlp = (NUM_LAYERS * MLP_PARAMS_PER_LAYER) as f64;
        let share = mlp / TEXT_PARAMS as f64;
        // MLP is ~64% of the weights, which is why they're where NVFP4 pays off.
        assert!((0.62..0.66).contains(&share), "MLP share {share}");
    }
}

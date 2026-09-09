//! Архитектура Qwen3.8-27B (`Qwen3_5ForConditionalGeneration`, text-only).
//!
//! Источник: `Qwen/Qwen3.8-27B/config.json`, секция `text_config`.
//! Vision tower намеренно отсутствует — движок text-only.

// ---------------------------------------------------------------------------
// Базовый стек
// ---------------------------------------------------------------------------

pub const NUM_LAYERS: usize = 64;
pub const HIDDEN_SIZE: usize = 5120;
pub const INTERMEDIATE_SIZE: usize = 17408;
pub const VOCAB_SIZE: usize = 248_320;
pub const RMS_NORM_EPS: f32 = 1e-6;
pub const MAX_POSITION_EMBEDDINGS: usize = 262_144;

/// Веса lm_head не связаны с embed_tokens: это отдельные 1.27B параметров.
pub const TIE_WORD_EMBEDDINGS: bool = false;

// ---------------------------------------------------------------------------
// Гибридный layout: [linear, linear, linear, full] x 16
// ---------------------------------------------------------------------------

pub const FULL_ATTENTION_INTERVAL: usize = 4;
pub const NUM_FULL_LAYERS: usize = NUM_LAYERS / FULL_ATTENTION_INTERVAL;
pub const NUM_LINEAR_LAYERS: usize = NUM_LAYERS - NUM_FULL_LAYERS;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// Gated DeltaNet: рекуррентное состояние константного размера.
    LinearAttention,
    /// Классический GQA-attention с paged KV-кэшем.
    FullAttention,
}

/// Тип слоя по индексу. Full-attention — каждый 4-й: 3, 7, 11, ..., 63.
pub const fn layer_kind(idx: usize) -> LayerKind {
    debug_assert!(idx < NUM_LAYERS);
    if (idx + 1) % FULL_ATTENTION_INTERVAL == 0 {
        LayerKind::FullAttention
    } else {
        LayerKind::LinearAttention
    }
}

/// Порядковый номер слоя среди слоёв своего типа — индекс в KV-кэше
/// (для full) или в пуле состояний (для linear).
pub const fn layer_slot(idx: usize) -> usize {
    match layer_kind(idx) {
        LayerKind::FullAttention => idx / FULL_ATTENTION_INTERVAL,
        LayerKind::LinearAttention => idx - (idx / FULL_ATTENTION_INTERVAL),
    }
}

// ---------------------------------------------------------------------------
// Full attention (16 слоёв)
// ---------------------------------------------------------------------------

pub const NUM_ATTN_HEADS: usize = 24;
pub const NUM_KV_HEADS: usize = 4;
/// Нетипично большой: почти все готовые attention-кернелы рассчитаны на <= 128.
pub const ATTN_HEAD_DIM: usize = 256;
pub const GQA_GROUP: usize = NUM_ATTN_HEADS / NUM_KV_HEADS;

pub const Q_PROJ_DIM: usize = NUM_ATTN_HEADS * ATTN_HEAD_DIM;
pub const KV_PROJ_DIM: usize = NUM_KV_HEADS * ATTN_HEAD_DIM;

/// `attn_output_gate: true` — дополнительный проекшн 5120 -> 6144,
/// swish-гейт на выходе attention перед o_proj.
pub const ATTN_OUTPUT_GATE: bool = true;

/// `partial_rotary_factor: 0.25` — RoPE применяется к первым 64 из 256 dims,
/// остальные 192 проходят насквозь.
pub const ROPE_DIM: usize = ATTN_HEAD_DIM / 4;
pub const ROPE_THETA: f64 = 10_000_000.0;

/// mrope с секциями [11, 11, 10] вырождается в обычный RoPE, когда все три
/// позиционных индекса равны — а в text-only они всегда равны.
/// Поэтому mrope в движке не реализуется вообще.
pub const MROPE_SECTIONS: [usize; 3] = [11, 11, 10];

// ---------------------------------------------------------------------------
// Linear attention / Gated DeltaNet (48 слоёв)
// ---------------------------------------------------------------------------

pub const LA_NUM_K_HEADS: usize = 16;
pub const LA_NUM_V_HEADS: usize = 48;
pub const LA_K_HEAD_DIM: usize = 128;
pub const LA_V_HEAD_DIM: usize = 128;
/// k/q-головы броадкастятся на v-головы 1:3.
pub const LA_HEAD_RATIO: usize = LA_NUM_V_HEADS / LA_NUM_K_HEADS;

pub const LA_QK_PROJ_DIM: usize = LA_NUM_K_HEADS * LA_K_HEAD_DIM;
pub const LA_V_PROJ_DIM: usize = LA_NUM_V_HEADS * LA_V_HEAD_DIM;

/// Короткая causal conv1d перед рекуррентностью, по каналам q|k|v.
pub const LA_CONV_KERNEL: usize = 4;
pub const LA_CONV_CHANNELS: usize = LA_QK_PROJ_DIM * 2 + LA_V_PROJ_DIM;

// ---------------------------------------------------------------------------
// MTP (speculative decoding)
// ---------------------------------------------------------------------------

/// Draft-голова встроена в чекпоинт: спекулятивный декодинг не требует
/// отдельной draft-модели.
pub const MTP_NUM_LAYERS: usize = 1;
pub const MTP_DEDICATED_EMBEDDINGS: bool = false;

// ---------------------------------------------------------------------------
// Производные величины кэша
// ---------------------------------------------------------------------------

/// Элементов KV-кэша на один токен по всем full-attention слоям.
/// 16 слоёв x 4 kv-головы x 256 dim x 2 (K и V) = 32768.
pub const KV_ELEMS_PER_TOKEN: usize = NUM_FULL_LAYERS * NUM_KV_HEADS * ATTN_HEAD_DIM * 2;

/// Элементов рекуррентного состояния DeltaNet на одну последовательность.
/// 48 слоёв x 48 v-голов x 128 x 128 = 37 748 736. Не зависит от длины контекста.
pub const STATE_ELEMS_PER_SEQ: usize =
    NUM_LINEAR_LAYERS * LA_NUM_V_HEADS * LA_K_HEAD_DIM * LA_V_HEAD_DIM;

/// Элементов conv-состояния на последовательность (окно kernel-1).
pub const CONV_ELEMS_PER_SEQ: usize = NUM_LINEAR_LAYERS * LA_CONV_CHANNELS * (LA_CONV_KERNEL - 1);

// ---------------------------------------------------------------------------
// Подсчёт параметров
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

/// Параметры, которые квантованы в NVFP4 (все linear-слои тела модели).
pub const QUANTIZED_PARAMS: usize = NUM_LAYERS * MLP_PARAMS_PER_LAYER
    + NUM_FULL_LAYERS * FULL_ATTN_PARAMS_PER_LAYER
    + NUM_LINEAR_LAYERS * LINEAR_ATTN_PARAMS_PER_LAYER;

/// Полное число параметров текстовой модели (без vision tower и MTP-головы).
pub const TEXT_PARAMS: usize = QUANTIZED_PARAMS + EMBED_PARAMS + LM_HEAD_PARAMS;


/// Оценка размера MTP draft-головы: один блок (attention + MLP) плюс
/// проекция конкатенации [hidden_state; embedding] -> hidden.
/// `mtp_use_dedicated_embeddings: false`, поэтому своих эмбеддингов нет.
/// Уточняется по `model.safetensors.index.json` при загрузке чекпоинта.
pub const MTP_PARAMS_EST: usize =
    FULL_ATTN_PARAMS_PER_LAYER + MLP_PARAMS_PER_LAYER + 2 * HIDDEN_SIZE * HIDDEN_SIZE;

/// Оценка размера vision tower (depth 27, hidden 1152, intermediate 4304,
/// patch 16, spatial_merge 2, out_hidden 5120). В text-only не грузится,
/// нужна только чтобы показать экономию относительно baseline.
pub const VISION_PARAMS_EST: usize = 27 * (4 * 1152 * 1152 + 2 * 1152 * 4304)  // блоки
    + 3 * 16 * 16 * 2 * 1152                                                   // patch embed
    + 2304 * 1152                                                              // pos embed
    + 1152 * 4 * HIDDEN_SIZE;                                                  // merger

// ---------------------------------------------------------------------------
// Проверки на этапе компиляции
// ---------------------------------------------------------------------------

const _: () = {
    assert!(NUM_FULL_LAYERS == 16);
    assert!(NUM_LINEAR_LAYERS == 48);
    assert!(GQA_GROUP == 6);
    assert!(LA_HEAD_RATIO == 3);
    assert!(ROPE_DIM == 64);
    assert!(KV_ELEMS_PER_TOKEN == 32_768);
    assert!(STATE_ELEMS_PER_SEQ == 37_748_736);
    // Секции mrope покрывают ровно половину RoPE-размерности (пары dim/2).
    assert!(MROPE_SECTIONS[0] + MROPE_SECTIONS[1] + MROPE_SECTIONS[2] == ROPE_DIM / 2);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_layout_matches_config() {
        // layer_types из config.json: full_attention на индексах 3, 7, ..., 63.
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
        // Заявленные 27B: сходимся в пределах 0.5%.
        let b = TEXT_PARAMS as f64 / 1e9;
        assert!((26.8..27.1).contains(&b), "получилось {b} B");
    }

    #[test]
    fn mlp_dominates_weights() {
        let mlp = (NUM_LAYERS * MLP_PARAMS_PER_LAYER) as f64;
        let share = mlp / TEXT_PARAMS as f64;
        // MLP — ~64% весов, поэтому именно они дают выигрыш от NVFP4.
        assert!((0.62..0.66).contains(&share), "доля MLP {share}");
    }
}

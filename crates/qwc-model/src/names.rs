//! Имена тензоров в чекпоинте. Собраны в одном месте, чтобы раскладка
//! конкретного чекпоинта не расползалась по коду загрузчика.


pub const EMBED: &str = "model.language_model.embed_tokens.weight";
pub const FINAL_NORM: &str = "model.language_model.norm.weight";
pub const LM_HEAD: &str = "lm_head.weight";

pub fn layer(i: usize) -> String {
    format!("model.language_model.layers.{i}")
}

/// Все квантованные linear-слои слоя `i` как (префикс, out_features, in_features).
pub fn quant_linears(i: usize) -> Vec<(String, usize, usize)> {
    use qwc_core::arch::*;
    let l = layer(i);
    let mut v = vec![
        (format!("{l}.mlp.gate_proj"), INTERMEDIATE_SIZE, HIDDEN_SIZE),
        (format!("{l}.mlp.up_proj"), INTERMEDIATE_SIZE, HIDDEN_SIZE),
        (format!("{l}.mlp.down_proj"), HIDDEN_SIZE, INTERMEDIATE_SIZE),
    ];
    match layer_kind(i) {
        LayerKind::FullAttention => {
            // q_proj выдаёт [q; выходной гейт] одной проекцией: attn_output_gate = true.
            v.push((format!("{l}.self_attn.q_proj"), 2 * Q_PROJ_DIM, HIDDEN_SIZE));
            v.push((format!("{l}.self_attn.k_proj"), KV_PROJ_DIM, HIDDEN_SIZE));
            v.push((format!("{l}.self_attn.v_proj"), KV_PROJ_DIM, HIDDEN_SIZE));
            v.push((format!("{l}.self_attn.o_proj"), HIDDEN_SIZE, Q_PROJ_DIM));
        }
        LayerKind::LinearAttention => {
            v.push((format!("{l}.linear_attn.in_proj_qkv"), LA_CONV_CHANNELS, HIDDEN_SIZE));
            v.push((format!("{l}.linear_attn.in_proj_z"), LA_V_PROJ_DIM, HIDDEN_SIZE));
            v.push((format!("{l}.linear_attn.in_proj_a"), LA_NUM_V_HEADS, HIDDEN_SIZE));
            v.push((format!("{l}.linear_attn.in_proj_b"), LA_NUM_V_HEADS, HIDDEN_SIZE));
            v.push((format!("{l}.linear_attn.out_proj"), HIDDEN_SIZE, LA_V_PROJ_DIM));
        }
    }
    v
}

/// Неквантованные тензоры слоя `i` как (имя, ожидаемая форма).
pub fn plain_tensors(i: usize) -> Vec<(String, Vec<usize>)> {
    use qwc_core::arch::*;
    let l = layer(i);
    let mut v = vec![
        (format!("{l}.input_layernorm.weight"), vec![HIDDEN_SIZE]),
        (format!("{l}.post_attention_layernorm.weight"), vec![HIDDEN_SIZE]),
    ];
    match layer_kind(i) {
        LayerKind::FullAttention => {
            // QK-норма на голову: нормируются q и k перед attention.
            v.push((format!("{l}.self_attn.q_norm.weight"), vec![ATTN_HEAD_DIM]));
            v.push((format!("{l}.self_attn.k_norm.weight"), vec![ATTN_HEAD_DIM]));
        }
        LayerKind::LinearAttention => {
            v.push((
                format!("{l}.linear_attn.conv1d.weight"),
                vec![LA_CONV_CHANNELS, 1, LA_CONV_KERNEL],
            ));
            v.push((format!("{l}.linear_attn.A_log"), vec![LA_NUM_V_HEADS]));
            v.push((format!("{l}.linear_attn.dt_bias"), vec![LA_NUM_V_HEADS]));
            v.push((format!("{l}.linear_attn.norm.weight"), vec![LA_V_HEAD_DIM]));
        }
    }
    v
}

/// Тензоры MTP-головы. В чекпоинте она не квантована.
pub fn mtp_tensors() -> Vec<(String, Vec<usize>)> {
    use qwc_core::arch::*;
    vec![
        ("mtp.fc.weight".into(), vec![HIDDEN_SIZE, 2 * HIDDEN_SIZE]),
        ("mtp.norm.weight".into(), vec![HIDDEN_SIZE]),
        ("mtp.pre_fc_norm_embedding.weight".into(), vec![HIDDEN_SIZE]),
        ("mtp.pre_fc_norm_hidden.weight".into(), vec![HIDDEN_SIZE]),
        ("mtp.layers.0.input_layernorm.weight".into(), vec![HIDDEN_SIZE]),
        ("mtp.layers.0.post_attention_layernorm.weight".into(), vec![HIDDEN_SIZE]),
        ("mtp.layers.0.self_attn.q_proj.weight".into(), vec![2 * Q_PROJ_DIM, HIDDEN_SIZE]),
        ("mtp.layers.0.self_attn.k_proj.weight".into(), vec![KV_PROJ_DIM, HIDDEN_SIZE]),
        ("mtp.layers.0.self_attn.v_proj.weight".into(), vec![KV_PROJ_DIM, HIDDEN_SIZE]),
        ("mtp.layers.0.self_attn.o_proj.weight".into(), vec![HIDDEN_SIZE, Q_PROJ_DIM]),
        ("mtp.layers.0.self_attn.q_norm.weight".into(), vec![ATTN_HEAD_DIM]),
        ("mtp.layers.0.self_attn.k_norm.weight".into(), vec![ATTN_HEAD_DIM]),
        ("mtp.layers.0.mlp.gate_proj.weight".into(), vec![INTERMEDIATE_SIZE, HIDDEN_SIZE]),
        ("mtp.layers.0.mlp.up_proj.weight".into(), vec![INTERMEDIATE_SIZE, HIDDEN_SIZE]),
        ("mtp.layers.0.mlp.down_proj.weight".into(), vec![HIDDEN_SIZE, INTERMEDIATE_SIZE]),
    ]
}

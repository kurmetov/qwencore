//! MTP draft-голова: один слой внимания поверх скрытого состояния основной
//! модели.
//!
//! Голова лежит в чекпоинте (`mtp.*`, 849 МБ BF16) и до сих пор не грузилась.
//! Её вход — пара (h_t, эмбеддинг токена t+1), выход — предсказание токена
//! t+2. Замеренный acceptance: 0.93 на первом черновике и 0.90 без
//! синтетических повторов корпуса (`bench/results/mtp-acceptance-2026-09-16.md`).
//!
//! Черновой шаг читает 849 МБ, то есть 5% от веса основного шага: за эти
//! проценты и покупается второй-третий токен за проход.

use qwc_core::arch::*;
use qwc_cuda::attention_prepare::AttentionPreprocessor;
use qwc_cuda::mtp as kernels;
use qwc_cuda::paged_attention::{self, KvCacheDtype, PAGE_SIZE, PagedAttentionWorkspace};
use qwc_cuda::rmsnorm::RmsNorm;
use qwc_cuda::{DeviceBuffer, Stream, bf16};

use crate::weights::ModelWeights;

/// Ширина выхода q_proj головы: на голову лежит [q | выходной гейт].
const Q_GATE_DIM: usize = 2 * Q_PROJ_DIM;

pub struct MtpHead {
    fc: DeviceBuffer<u16>,
    query: DeviceBuffer<u16>,
    key: DeviceBuffer<u16>,
    value: DeviceBuffer<u16>,
    output: DeviceBuffer<u16>,
    gate: DeviceBuffer<u16>,
    up: DeviceBuffer<u16>,
    down: DeviceBuffer<u16>,
    pre_fc_norm_embedding: RmsNorm,
    pre_fc_norm_hidden: RmsNorm,
    input_norm: RmsNorm,
    post_attention_norm: RmsNorm,
    final_norm: RmsNorm,
    prepare: AttentionPreprocessor,
    bytes: usize,
}

impl MtpHead {
    pub fn resident_bytes(&self) -> usize {
        self.bytes
    }
}

/// Скретч и собственный KV-кэш головы. Слой у неё один, поэтому кэш стоит
/// 1/16 от кэша основной модели на тот же контекст.
pub struct MtpScratch {
    embedding: DeviceBuffer<u16>,
    normed_embedding: DeviceBuffer<u16>,
    normed_hidden: DeviceBuffer<u16>,
    joined: DeviceBuffer<u16>,
    residual: DeviceBuffer<u16>,
    normed: DeviceBuffer<u16>,
    query_gate: DeviceBuffer<u16>,
    key_projection: DeviceBuffer<u16>,
    value_projection: DeviceBuffer<u16>,
    query: DeviceBuffer<u16>,
    attention_out: DeviceBuffer<u16>,
    attention_workspace: PagedAttentionWorkspace,
    mlp_gate: DeviceBuffer<u16>,
    mlp_up: DeviceBuffer<u16>,
    mlp_hidden: DeviceBuffer<u16>,
    key_cache: DeviceBuffer<u8>,
    value_cache: DeviceBuffer<u8>,

    tokens: DeviceBuffer<u32>,
    cosine: DeviceBuffer<u16>,
    sine: DeviceBuffer<u16>,
    physical_blocks: DeviceBuffer<u32>,
    block_offsets: DeviceBuffer<u32>,
    block_tables: DeviceBuffer<u32>,
    context_lengths: DeviceBuffer<u32>,
    host_cosine: Vec<u16>,
    host_sine: Vec<u16>,

    max_blocks: usize,
    rows: usize,
    cache_dtype: KvCacheDtype,
}

impl MtpScratch {
    /// `rows` — сколько позиций голова считает за вызов (черновик идёт по
    /// одной), `max_context` — потолок длины последовательности.
    pub fn new(rows: usize, max_context: usize, cache_dtype: KvCacheDtype) -> qwc_cuda::Result<Self> {
        assert!((1..=kernels::MAX_ROWS).contains(&rows));
        let max_blocks = max_context.div_ceil(PAGE_SIZE);
        let cache_bytes = max_blocks * PAGE_SIZE * NUM_KV_HEADS * ATTN_HEAD_DIM
            * cache_dtype.bytes_per_element();
        Ok(Self {
            embedding: DeviceBuffer::zeroed(rows * HIDDEN_SIZE)?,
            normed_embedding: DeviceBuffer::zeroed(rows * HIDDEN_SIZE)?,
            normed_hidden: DeviceBuffer::zeroed(rows * HIDDEN_SIZE)?,
            joined: DeviceBuffer::zeroed(rows * 2 * HIDDEN_SIZE)?,
            residual: DeviceBuffer::zeroed(rows * HIDDEN_SIZE)?,
            normed: DeviceBuffer::zeroed(rows * HIDDEN_SIZE)?,
            query_gate: DeviceBuffer::zeroed(rows * Q_GATE_DIM)?,
            key_projection: DeviceBuffer::zeroed(rows * KV_PROJ_DIM)?,
            value_projection: DeviceBuffer::zeroed(rows * KV_PROJ_DIM)?,
            query: DeviceBuffer::zeroed(rows * Q_PROJ_DIM)?,
            attention_out: DeviceBuffer::zeroed(rows * Q_PROJ_DIM)?,
            attention_workspace: PagedAttentionWorkspace::new(rows, max_context)?,
            mlp_gate: DeviceBuffer::zeroed(rows * INTERMEDIATE_SIZE)?,
            mlp_up: DeviceBuffer::zeroed(rows * INTERMEDIATE_SIZE)?,
            mlp_hidden: DeviceBuffer::zeroed(rows * INTERMEDIATE_SIZE)?,
            key_cache: DeviceBuffer::zeroed(cache_bytes)?,
            value_cache: DeviceBuffer::zeroed(cache_bytes)?,
            tokens: DeviceBuffer::zeroed(rows)?,
            cosine: DeviceBuffer::zeroed(rows * ROPE_DIM)?,
            sine: DeviceBuffer::zeroed(rows * ROPE_DIM)?,
            physical_blocks: DeviceBuffer::zeroed(rows)?,
            block_offsets: DeviceBuffer::zeroed(rows)?,
            block_tables: DeviceBuffer::zeroed(rows * max_blocks)?,
            context_lengths: DeviceBuffer::zeroed(rows)?,
            host_cosine: vec![0; rows * ROPE_DIM],
            host_sine: vec![0; rows * ROPE_DIM],
            max_blocks,
            rows,
            cache_dtype,
        })
    }

    pub fn resident_bytes(&self) -> usize {
        self.key_cache.bytes() + self.value_cache.bytes()
    }

    /// Позиционные таблицы и адрес страницы для строк [0, rows).
    fn upload(&mut self, tokens: &[u32], positions: &[u32]) -> qwc_cuda::Result<()> {
        let rows = tokens.len();
        assert_eq!(positions.len(), rows);
        let mut physical = vec![0u32; rows];
        let mut offsets = vec![0u32; rows];
        let mut lengths = vec![0u32; rows];
        let mut tables = vec![0u32; rows * self.max_blocks];
        for (row, &position) in positions.iter().enumerate() {
            for dimension in 0..ROPE_DIM {
                let pair = dimension % (ROPE_DIM / 2);
                let inverse = (ROPE_THETA as f32).powf(-(2.0 * pair as f32) / ROPE_DIM as f32);
                let angle = position as f32 * inverse;
                self.host_cosine[row * ROPE_DIM + dimension] = bf16::from_f32(angle.cos());
                self.host_sine[row * ROPE_DIM + dimension] = bf16::from_f32(angle.sin());
            }
            physical[row] = (position as usize / PAGE_SIZE) as u32;
            offsets[row] = (position as usize % PAGE_SIZE) as u32;
            lengths[row] = position + 1;
            for block in 0..self.max_blocks {
                tables[row * self.max_blocks + block] = block as u32;
            }
        }
        self.tokens.copy_from_slice_at(0, tokens)?;
        self.cosine.copy_from_slice_at(0, &self.host_cosine[..rows * ROPE_DIM])?;
        self.sine.copy_from_slice_at(0, &self.host_sine[..rows * ROPE_DIM])?;
        self.physical_blocks.copy_from_slice_at(0, &physical)?;
        self.block_offsets.copy_from_slice_at(0, &offsets)?;
        self.context_lengths.copy_from_slice_at(0, &lengths)?;
        self.block_tables.copy_from_slice_at(0, &tables)
    }
}

/// Один проход головы: из (hidden, следующий токен) получить скрытое
/// состояние черновика. Логиты из него считает `lm_head` основной модели —
/// голова своей не имеет.
///
/// `hidden` — финальные скрытые состояния основной модели для тех же строк,
/// `tokens[i]` — токен на позиции `positions[i] + 1`, то есть тот, который
/// модель только что выдала.
#[allow(clippy::too_many_arguments)]
pub fn draft(
    head: &MtpHead,
    weights: &ModelWeights,
    scratch: &mut MtpScratch,
    hidden: &DeviceBuffer<u16>,
    tokens: &[u32],
    positions: &[u32],
    out: &mut DeviceBuffer<u16>,
    stream: &Stream,
) -> qwc_cuda::Result<()> {
    let rows = tokens.len();
    assert!(rows > 0 && rows <= scratch.rows);
    assert!(hidden.len() >= rows * HIDDEN_SIZE);
    assert!(out.len() >= rows * HIDDEN_SIZE);
    scratch.upload(tokens, positions)?;

    weights
        .embed
        .gather_rows(&scratch.tokens, &mut scratch.embedding, rows, stream)?;
    head.pre_fc_norm_embedding.forward_bf16(
        &scratch.embedding,
        None,
        &mut scratch.normed_embedding,
        rows,
        stream,
    )?;
    head.pre_fc_norm_hidden
        .forward_bf16(hidden, None, &mut scratch.normed_hidden, rows, stream)?;
    kernels::concat(
        &scratch.normed_embedding,
        &scratch.normed_hidden,
        &mut scratch.joined,
        rows,
        HIDDEN_SIZE,
        stream,
    )?;
    kernels::bf16_linear(
        &head.fc,
        &scratch.joined,
        &mut scratch.residual,
        rows,
        2 * HIDDEN_SIZE,
        HIDDEN_SIZE,
        stream,
    )?;

    head.input_norm
        .forward_bf16(&scratch.residual, None, &mut scratch.normed, rows, stream)?;
    kernels::bf16_linear(&head.query, &scratch.normed, &mut scratch.query_gate,
        rows, HIDDEN_SIZE, Q_GATE_DIM, stream)?;
    kernels::bf16_linear(&head.key, &scratch.normed, &mut scratch.key_projection,
        rows, HIDDEN_SIZE, KV_PROJ_DIM, stream)?;
    kernels::bf16_linear(&head.value, &scratch.normed, &mut scratch.value_projection,
        rows, HIDDEN_SIZE, KV_PROJ_DIM, stream)?;
    head.prepare.prepare_decode(
        &scratch.query_gate,
        &scratch.key_projection,
        &scratch.value_projection,
        &scratch.cosine,
        &scratch.sine,
        &scratch.physical_blocks,
        &scratch.block_offsets,
        &mut scratch.query,
        &mut scratch.key_cache,
        &mut scratch.value_cache,
        scratch.max_blocks,
        rows,
        scratch.cache_dtype,
        stream,
    )?;
    let max_context = positions.iter().max().copied().unwrap_or(0) as usize + 1;
    paged_attention::decode_gated(
        &scratch.query,
        &scratch.query_gate,
        &scratch.key_cache,
        &scratch.value_cache,
        scratch.max_blocks,
        &scratch.block_tables,
        &scratch.context_lengths,
        scratch.max_blocks,
        &mut scratch.attention_out,
        &mut scratch.attention_workspace,
        rows,
        max_context,
        scratch.cache_dtype,
        stream,
    )?;
    kernels::bf16_linear(&head.output, &scratch.attention_out, &mut scratch.normed,
        rows, Q_PROJ_DIM, HIDDEN_SIZE, stream)?;

    // residual = residual + attn, и та же сумма нормируется дальше.
    head.post_attention_norm.forward_bf16(
        &scratch.normed,
        Some(&mut scratch.residual),
        &mut scratch.normed_hidden,
        rows,
        stream,
    )?;
    kernels::bf16_linear(&head.gate, &scratch.normed_hidden, &mut scratch.mlp_gate,
        rows, HIDDEN_SIZE, INTERMEDIATE_SIZE, stream)?;
    kernels::bf16_linear(&head.up, &scratch.normed_hidden, &mut scratch.mlp_up,
        rows, HIDDEN_SIZE, INTERMEDIATE_SIZE, stream)?;
    kernels::swiglu(&scratch.mlp_gate, &scratch.mlp_up, &mut scratch.mlp_hidden,
        rows * INTERMEDIATE_SIZE, stream)?;
    kernels::bf16_linear(&head.down, &scratch.mlp_hidden, &mut scratch.normed,
        rows, INTERMEDIATE_SIZE, HIDDEN_SIZE, stream)?;
    head.final_norm
        .forward_bf16(&scratch.normed, Some(&mut scratch.residual), out, rows, stream)
}

/// Подъём головы из чекпоинта. Формы сверяются с архитектурой, как и у
/// основных весов: расхождение — ошибка загрузки, а не повод подстроиться.
pub fn load(checkpoint: &qwc_model::Checkpoint) -> Result<MtpHead, crate::weights::LoadError> {
    use crate::weights::{bf16_device_tensor, load_norm_named};

    let fc = bf16_device_tensor(checkpoint, "mtp.fc.weight", HIDDEN_SIZE * 2 * HIDDEN_SIZE)?;
    let query = bf16_device_tensor(
        checkpoint,
        "mtp.layers.0.self_attn.q_proj.weight",
        Q_GATE_DIM * HIDDEN_SIZE,
    )?;
    let key = bf16_device_tensor(
        checkpoint,
        "mtp.layers.0.self_attn.k_proj.weight",
        KV_PROJ_DIM * HIDDEN_SIZE,
    )?;
    let value = bf16_device_tensor(
        checkpoint,
        "mtp.layers.0.self_attn.v_proj.weight",
        KV_PROJ_DIM * HIDDEN_SIZE,
    )?;
    let output = bf16_device_tensor(
        checkpoint,
        "mtp.layers.0.self_attn.o_proj.weight",
        HIDDEN_SIZE * Q_PROJ_DIM,
    )?;
    let gate = bf16_device_tensor(
        checkpoint,
        "mtp.layers.0.mlp.gate_proj.weight",
        INTERMEDIATE_SIZE * HIDDEN_SIZE,
    )?;
    let up = bf16_device_tensor(
        checkpoint,
        "mtp.layers.0.mlp.up_proj.weight",
        INTERMEDIATE_SIZE * HIDDEN_SIZE,
    )?;
    let down = bf16_device_tensor(
        checkpoint,
        "mtp.layers.0.mlp.down_proj.weight",
        HIDDEN_SIZE * INTERMEDIATE_SIZE,
    )?;
    let bytes = [&fc, &query, &key, &value, &output, &gate, &up, &down]
        .iter()
        .map(|buffer| buffer.bytes())
        .sum();

    Ok(MtpHead {
        fc,
        query,
        key,
        value,
        output,
        gate,
        up,
        down,
        pre_fc_norm_embedding: load_norm_named(checkpoint, "mtp.pre_fc_norm_embedding.weight", HIDDEN_SIZE)?,
        pre_fc_norm_hidden: load_norm_named(checkpoint, "mtp.pre_fc_norm_hidden.weight", HIDDEN_SIZE)?,
        input_norm: load_norm_named(checkpoint, "mtp.layers.0.input_layernorm.weight", HIDDEN_SIZE)?,
        post_attention_norm: load_norm_named(
            checkpoint,
            "mtp.layers.0.post_attention_layernorm.weight",
            HIDDEN_SIZE,
        )?,
        final_norm: load_norm_named(checkpoint, "mtp.norm.weight", HIDDEN_SIZE)?,
        prepare: AttentionPreprocessor::from_host(
            &crate::weights::bf16_host_tensor(
                checkpoint,
                "mtp.layers.0.self_attn.q_norm.weight",
                ATTN_HEAD_DIM,
            )?,
            &crate::weights::bf16_host_tensor(
                checkpoint,
                "mtp.layers.0.self_attn.k_norm.weight",
                ATTN_HEAD_DIM,
            )?,
            RMS_NORM_EPS,
        )?,
        bytes,
    })
}

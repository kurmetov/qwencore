//! Шаг decode целиком: от идентификатора токена до логитов.
//!
//! Все буферы шага резидентны и выделяются один раз: адреса не меняются от
//! шага к шагу, иначе позже нечего будет захватывать в CUDA graph. Планировщик
//! из `qwc-runtime` сюда пока не подключён — блоки KV и слоты состояния
//! раздаются по последовательностям статически, чтобы сначала проверить
//! численный тракт, а не политику вытеснения.

use crate::weights::{
    DecodeLinearMode, FullAttention, LinearAttention, Mixer, ModelWeights, Projection,
};
use qwc_core::arch::*;
use qwc_cuda::delta_net::{
    self, CONV_STATE_ELEMS, DeltaStateMode, GATE_ELEMS, PreparedDelta, STATE_ELEMS, V_ELEMS,
};
use qwc_cuda::graph::CudaGraph;
use qwc_cuda::nvfp4::{self, QuantizedActivation, W4A4Workspace};
use qwc_cuda::paged_attention::{self, KvCacheDtype, PAGE_SIZE, PagedAttentionWorkspace};
use qwc_cuda::sampling::Argmax;
use qwc_cuda::{DeviceBuffer, Result, Stream, Timeline, bf16};
use qwc_runtime::BatchLayout;

/// Persistent sequence slots supported by the decode kernels.
///
/// Bounded by memory, not by the kernels: one sequence costs ~0.148 GB of
/// DeltaNet state plus KV cache at ctx 2048, so 64 slots need ~9.5 GB on top
/// of the 16.25 GB of weights. The decode kernels themselves accept 128.
pub const MAX_BATCH: usize = 96;
/// Fixed tensor-core M dimension used by chunked prefill.
///
/// Every engine step reads all 16.25 GB of weights regardless of how many
/// tokens it carries, so a small chunk makes prefill pay that read over and
/// over. 512 quarters the number of prefill steps against the previous 128.
pub const PREFILL_CHUNK_SIZE: usize = 512;

/// One sequence's slice of a fused step.
///
/// A decode sequence is simply a segment carrying a single token, which is why
/// decode and prefill can share one forward pass.
#[derive(Clone, Copy, Debug)]
pub struct Segment {
    pub state_slot: usize,
    pub position_start: usize,
    /// First row of this sequence inside the fused token arena.
    pub row_begin: usize,
    pub tokens: usize,
    pub logits_row: usize,
}

pub struct ExecutorConfig {
    pub max_batch: usize,
    pub max_context: usize,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            max_batch: 1,
            max_context: 2048,
        }
    }
}

/// Резидентные буферы одного шага decode.
pub struct Executor {
    stream: Stream,
    max_batch: usize,
    max_context: usize,
    max_blocks: usize,
    kv_cache_dtype: KvCacheDtype,
    decode_linear_mode: DecodeLinearMode,
    delta_state_mode: DeltaStateMode,

    // Магистраль слоя.
    residual: DeviceBuffer<u16>,
    normed: DeviceBuffer<u16>,
    mixer_out: DeviceBuffer<u16>,
    mlp_gate: DeviceBuffer<u16>,
    mlp_up: DeviceBuffer<u16>,
    mlp_hidden: DeviceBuffer<u16>,
    mlp_out: DeviceBuffer<u16>,
    quant_hidden: QuantizedActivation,
    quant_mixer: QuantizedActivation,
    quant_mlp: QuantizedActivation,
    projection_workspace: W4A4Workspace,

    // Gated DeltaNet.
    mixed_qkv: DeviceBuffer<u16>,
    z_gate: DeviceBuffer<u16>,
    a_projection: DeviceBuffer<u16>,
    b_projection: DeviceBuffer<u16>,
    prepared: PreparedDelta,
    delta_out: DeviceBuffer<f32>,
    delta_normed: DeviceBuffer<u16>,
    state_slots: DeviceBuffer<u32>,
    state_pools: Vec<DeviceBuffer<u16>>,
    conv_pools: Vec<DeviceBuffer<f32>>,

    // Full attention.
    query_gate: DeviceBuffer<u16>,
    key_projection: DeviceBuffer<u16>,
    value_projection: DeviceBuffer<u16>,
    query: DeviceBuffer<u16>,
    attention_out: DeviceBuffer<u16>,
    cosine: DeviceBuffer<u16>,
    sine: DeviceBuffer<u16>,
    physical_blocks: DeviceBuffer<u32>,
    block_offsets: DeviceBuffer<u32>,
    block_tables: DeviceBuffer<u32>,
    context_lengths: DeviceBuffer<u32>,
    key_caches: Vec<DeviceBuffer<u8>>,
    value_caches: Vec<DeviceBuffer<u8>>,
    workspace: PagedAttentionWorkspace,

    // Вход и выход шага.
    tokens: DeviceBuffer<u32>,
    logits: DeviceBuffer<f32>,
    sampler: Argmax,
    decode_graphs: Vec<Option<CudaGraph>>,
    captured_weights: Option<usize>,

    // Пофазный таймлайн. Пока он включён, шаг decode идёт мимо CUDA graph:
    // внутри захвата события не измеряют время. Разница между шагом с
    // таймлайном и шагом с графом — это и есть цена запусков.
    profile: Option<Timeline>,

    // Хостовые заготовки, чтобы не аллоцировать на каждом шаге.
    host_cosine: Vec<u16>,
    host_sine: Vec<u16>,
    host_physical: Vec<u32>,
    host_offsets: Vec<u32>,
    host_lengths: Vec<u32>,
    host_state_slots: Vec<u32>,
    host_block_tables: Vec<u32>,

    // Tensor-core causal prefill for sequence slot zero. It is deliberately a
    // separate resident arena: decode addresses stay stable for CUDA graphs.
    prefill: PrefillBuffers,
}

struct PrefillBuffers {
    residual: DeviceBuffer<u16>,
    normed: DeviceBuffer<u16>,
    mixer_out: DeviceBuffer<u16>,
    mlp_gate: DeviceBuffer<u16>,
    mlp_up: DeviceBuffer<u16>,
    mlp_hidden: DeviceBuffer<u16>,
    mlp_out: DeviceBuffer<u16>,

    mixed_qkv: DeviceBuffer<u16>,
    z_gate: DeviceBuffer<u16>,
    a_projection: DeviceBuffer<u16>,
    b_projection: DeviceBuffer<u16>,
    prepared: PreparedDelta,
    delta_out: DeviceBuffer<f32>,
    delta_normed: DeviceBuffer<u16>,

    query_gate: DeviceBuffer<u16>,
    key_projection: DeviceBuffer<u16>,
    value_projection: DeviceBuffer<u16>,
    query: DeviceBuffer<u16>,
    attention_out: DeviceBuffer<u16>,
    cosine: DeviceBuffer<u16>,
    sine: DeviceBuffer<u16>,
    physical_blocks: DeviceBuffer<u32>,
    block_offsets: DeviceBuffer<u32>,
    block_tables: DeviceBuffer<u32>,
    context_lengths: DeviceBuffer<u32>,
    attention_workspace: PagedAttentionWorkspace,

    tokens: DeviceBuffer<u32>,
    /// Last row of each segment, and the dense gather of those rows.
    last_rows: DeviceBuffer<u32>,
    last_hidden: DeviceBuffer<u16>,
    /// State slots of the leading one-token rows of a fused step.
    decode_slots: DeviceBuffer<u32>,
    quant_hidden: QuantizedActivation,
    quant_mixer: QuantizedActivation,
    quant_mlp: QuantizedActivation,
    projection_workspace: W4A4Workspace,

    host_tokens: Vec<u32>,
    host_last_rows: Vec<u32>,
    host_decode_slots: Vec<u32>,
    host_cosine: Vec<u16>,
    host_sine: Vec<u16>,
    host_physical: Vec<u32>,
    host_offsets: Vec<u32>,
    host_lengths: Vec<u32>,
    host_block_tables: Vec<u32>,
}

impl PrefillBuffers {
    fn new(max_context: usize, max_blocks: usize) -> Result<Self> {
        let rows = PREFILL_CHUNK_SIZE;
        // Each query row represents another time step of sequence zero, so all
        // rows share its physical page table while keeping their own context
        // length for the causal mask.
        let mut tables = vec![0u32; rows * max_blocks];
        for row in 0..rows {
            for block in 0..max_blocks {
                tables[row * max_blocks + block] = block as u32;
            }
        }
        let shapes = [
            (LA_CONV_CHANNELS, HIDDEN_SIZE),
            (LA_V_PROJ_DIM, HIDDEN_SIZE),
            (GATE_ELEMS, HIDDEN_SIZE),
            (HIDDEN_SIZE, LA_V_PROJ_DIM),
            (2 * Q_PROJ_DIM, HIDDEN_SIZE),
            (KV_PROJ_DIM, HIDDEN_SIZE),
            (HIDDEN_SIZE, Q_PROJ_DIM),
            (INTERMEDIATE_SIZE, HIDDEN_SIZE),
            (HIDDEN_SIZE, INTERMEDIATE_SIZE),
        ];
        Ok(Self {
            residual: DeviceBuffer::zeroed(rows * HIDDEN_SIZE)?,
            normed: DeviceBuffer::zeroed(rows * HIDDEN_SIZE)?,
            mixer_out: DeviceBuffer::zeroed(rows * HIDDEN_SIZE)?,
            mlp_gate: DeviceBuffer::zeroed(rows * INTERMEDIATE_SIZE)?,
            mlp_up: DeviceBuffer::zeroed(rows * INTERMEDIATE_SIZE)?,
            mlp_hidden: DeviceBuffer::zeroed(rows * INTERMEDIATE_SIZE)?,
            mlp_out: DeviceBuffer::zeroed(rows * HIDDEN_SIZE)?,
            mixed_qkv: DeviceBuffer::zeroed(rows * LA_CONV_CHANNELS)?,
            z_gate: DeviceBuffer::zeroed(rows * LA_V_PROJ_DIM)?,
            a_projection: DeviceBuffer::zeroed(rows * GATE_ELEMS)?,
            b_projection: DeviceBuffer::zeroed(rows * GATE_ELEMS)?,
            prepared: PreparedDelta::zeroed(rows)?,
            delta_out: DeviceBuffer::zeroed(rows * V_ELEMS)?,
            delta_normed: DeviceBuffer::zeroed(rows * V_ELEMS)?,
            query_gate: DeviceBuffer::zeroed(rows * 2 * Q_PROJ_DIM)?,
            key_projection: DeviceBuffer::zeroed(rows * KV_PROJ_DIM)?,
            value_projection: DeviceBuffer::zeroed(rows * KV_PROJ_DIM)?,
            query: DeviceBuffer::zeroed(rows * Q_PROJ_DIM)?,
            attention_out: DeviceBuffer::zeroed(rows * Q_PROJ_DIM)?,
            cosine: DeviceBuffer::zeroed(rows * ROPE_DIM)?,
            sine: DeviceBuffer::zeroed(rows * ROPE_DIM)?,
            physical_blocks: DeviceBuffer::zeroed(rows)?,
            block_offsets: DeviceBuffer::zeroed(rows)?,
            block_tables: DeviceBuffer::from_slice(&tables)?,
            context_lengths: DeviceBuffer::zeroed(rows)?,
            attention_workspace: PagedAttentionWorkspace::new(rows, max_context)?,
            tokens: DeviceBuffer::zeroed(rows)?,
            last_rows: DeviceBuffer::zeroed(MAX_BATCH)?,
            last_hidden: DeviceBuffer::zeroed(MAX_BATCH * HIDDEN_SIZE)?,
            decode_slots: DeviceBuffer::zeroed(MAX_BATCH)?,
            // The scale is replaced before every projection.
            quant_hidden: QuantizedActivation::zeroed(1.0, rows, HIDDEN_SIZE)?,
            quant_mixer: QuantizedActivation::zeroed(1.0, rows, LA_V_PROJ_DIM)?,
            quant_mlp: QuantizedActivation::zeroed(1.0, rows, INTERMEDIATE_SIZE)?,
            projection_workspace: W4A4Workspace::for_shapes(rows, &shapes)?,
            host_tokens: vec![0; rows],
            host_last_rows: vec![0; MAX_BATCH],
            host_decode_slots: vec![0; MAX_BATCH],
            host_cosine: vec![0; rows * ROPE_DIM],
            host_sine: vec![0; rows * ROPE_DIM],
            host_physical: vec![0; rows],
            host_offsets: vec![0; rows],
            host_lengths: vec![0; rows],
            host_block_tables: vec![0; rows * max_blocks],
        })
    }
}

impl Executor {
    pub fn new(config: ExecutorConfig) -> Result<Self> {
        Self::new_with_options(
            config,
            KvCacheDtype::Fp8,
            DecodeLinearMode::Auto,
            DeltaStateMode::Bf16,
        )
    }

    pub fn new_with_kv_cache(config: ExecutorConfig, kv_cache_dtype: KvCacheDtype) -> Result<Self> {
        Self::new_with_options(
            config,
            kv_cache_dtype,
            DecodeLinearMode::Auto,
            DeltaStateMode::Bf16,
        )
    }

    pub fn new_with_options(
        config: ExecutorConfig,
        kv_cache_dtype: KvCacheDtype,
        decode_linear_mode: DecodeLinearMode,
        delta_state_mode: DeltaStateMode,
    ) -> Result<Self> {
        let batch = config.max_batch;
        assert!(
            (1..=MAX_BATCH).contains(&batch),
            "decode держит batch до {MAX_BATCH}"
        );
        assert!(config.max_context > 0);
        // CUTLASS W4A4 requires M >= 3. Diagnostic batch-1/2 runs keep
        // ignored padding rows in the resident activation arena.
        let decode_rows = match decode_linear_mode {
            DecodeLinearMode::Auto => batch,
            DecodeLinearMode::W4A4 => batch.max(3),
        };
        let max_blocks = config.max_context.div_ceil(PAGE_SIZE);
        let cache_elements = batch * max_blocks * NUM_KV_HEADS * PAGE_SIZE * ATTN_HEAD_DIM;
        let cache_bytes = cache_elements * kv_cache_dtype.bytes_per_element();

        // Каждая последовательность владеет своим непрерывным диапазоном
        // страниц: block table фиксирована, меняется только длина контекста.
        let mut tables = vec![0u32; batch * max_blocks];
        for sequence in 0..batch {
            for block in 0..max_blocks {
                tables[sequence * max_blocks + block] = (sequence * max_blocks + block) as u32;
            }
        }
        let slots: Vec<u32> = (0..batch as u32).collect();

        let mut state_pools = Vec::with_capacity(NUM_LINEAR_LAYERS);
        let mut conv_pools = Vec::with_capacity(NUM_LINEAR_LAYERS);
        for _ in 0..NUM_LINEAR_LAYERS {
            state_pools.push(DeviceBuffer::zeroed(batch * STATE_ELEMS)?);
            conv_pools.push(DeviceBuffer::zeroed(batch * CONV_STATE_ELEMS)?);
        }
        let mut key_caches = Vec::with_capacity(NUM_FULL_LAYERS);
        let mut value_caches = Vec::with_capacity(NUM_FULL_LAYERS);
        for _ in 0..NUM_FULL_LAYERS {
            key_caches.push(DeviceBuffer::zeroed(cache_bytes)?);
            value_caches.push(DeviceBuffer::zeroed(cache_bytes)?);
        }

        let prefill = PrefillBuffers::new(config.max_context, max_blocks)?;
        Ok(Self {
            stream: Stream::new()?,
            max_batch: batch,
            max_context: config.max_context,
            max_blocks,
            kv_cache_dtype,
            decode_linear_mode,
            delta_state_mode,
            residual: DeviceBuffer::zeroed(decode_rows * HIDDEN_SIZE)?,
            normed: DeviceBuffer::zeroed(decode_rows * HIDDEN_SIZE)?,
            mixer_out: DeviceBuffer::zeroed(decode_rows * HIDDEN_SIZE)?,
            mlp_gate: DeviceBuffer::zeroed(decode_rows * INTERMEDIATE_SIZE)?,
            mlp_up: DeviceBuffer::zeroed(decode_rows * INTERMEDIATE_SIZE)?,
            mlp_hidden: DeviceBuffer::zeroed(decode_rows * INTERMEDIATE_SIZE)?,
            mlp_out: DeviceBuffer::zeroed(decode_rows * HIDDEN_SIZE)?,
            quant_hidden: QuantizedActivation::zeroed(1.0, decode_rows, HIDDEN_SIZE)?,
            quant_mixer: QuantizedActivation::zeroed(1.0, decode_rows, LA_V_PROJ_DIM)?,
            quant_mlp: QuantizedActivation::zeroed(1.0, decode_rows, INTERMEDIATE_SIZE)?,
            projection_workspace: W4A4Workspace::for_shapes(
                decode_rows,
                &[
                    (LA_CONV_CHANNELS, HIDDEN_SIZE),
                    (LA_V_PROJ_DIM, HIDDEN_SIZE),
                    (GATE_ELEMS, HIDDEN_SIZE),
                    (HIDDEN_SIZE, LA_V_PROJ_DIM),
                    (2 * Q_PROJ_DIM, HIDDEN_SIZE),
                    (KV_PROJ_DIM, HIDDEN_SIZE),
                    (HIDDEN_SIZE, Q_PROJ_DIM),
                    (INTERMEDIATE_SIZE, HIDDEN_SIZE),
                    (HIDDEN_SIZE, INTERMEDIATE_SIZE),
                ],
            )?,
            mixed_qkv: DeviceBuffer::zeroed(decode_rows * LA_CONV_CHANNELS)?,
            z_gate: DeviceBuffer::zeroed(decode_rows * LA_V_PROJ_DIM)?,
            a_projection: DeviceBuffer::zeroed(decode_rows * GATE_ELEMS)?,
            b_projection: DeviceBuffer::zeroed(decode_rows * GATE_ELEMS)?,
            prepared: PreparedDelta::zeroed(batch)?,
            delta_out: DeviceBuffer::zeroed(batch * V_ELEMS)?,
            delta_normed: DeviceBuffer::zeroed(decode_rows * V_ELEMS)?,
            state_slots: DeviceBuffer::from_slice(&slots)?,
            state_pools,
            conv_pools,
            query_gate: DeviceBuffer::zeroed(decode_rows * 2 * Q_PROJ_DIM)?,
            key_projection: DeviceBuffer::zeroed(decode_rows * KV_PROJ_DIM)?,
            value_projection: DeviceBuffer::zeroed(decode_rows * KV_PROJ_DIM)?,
            query: DeviceBuffer::zeroed(decode_rows * Q_PROJ_DIM)?,
            attention_out: DeviceBuffer::zeroed(decode_rows * Q_PROJ_DIM)?,
            cosine: DeviceBuffer::zeroed(batch * ROPE_DIM)?,
            sine: DeviceBuffer::zeroed(batch * ROPE_DIM)?,
            physical_blocks: DeviceBuffer::zeroed(batch)?,
            block_offsets: DeviceBuffer::zeroed(batch)?,
            block_tables: DeviceBuffer::from_slice(&tables)?,
            context_lengths: DeviceBuffer::zeroed(batch)?,
            key_caches,
            value_caches,
            workspace: PagedAttentionWorkspace::new(batch, config.max_context)?,
            tokens: DeviceBuffer::zeroed(batch)?,
            logits: DeviceBuffer::zeroed(batch * VOCAB_SIZE)?,
            sampler: Argmax::new(batch, VOCAB_SIZE)?,
            decode_graphs: (0..batch).map(|_| None).collect(),
            captured_weights: None,
            profile: None,
            host_cosine: vec![0; batch * ROPE_DIM],
            host_sine: vec![0; batch * ROPE_DIM],
            host_physical: vec![0; batch],
            host_offsets: vec![0; batch],
            host_lengths: vec![0; batch],
            host_state_slots: slots,
            host_block_tables: vec![0; batch * max_blocks],
            prefill,
        })
    }

    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    /// Включает пофазный замер. Шаг чистого decode перестаёт идти через CUDA
    /// graph — события внутри захвата времени не дают.
    pub fn profile_enable(&mut self) {
        self.profile = Some(Timeline::new());
    }

    pub fn profile_disable(&mut self) {
        self.profile = None;
    }

    /// Сбрасывает метки перед шагом. Итоги читаются после него.
    pub fn profile_reset(&mut self) {
        if let Some(timeline) = self.profile.as_mut() {
            timeline.reset();
        }
    }

    /// Суммы по фазам последнего шага в порядке первого появления.
    pub fn profile_totals(&self) -> Result<Vec<(&'static str, f32)>> {
        match self.profile.as_ref() {
            Some(timeline) => timeline.totals(),
            None => Ok(Vec::new()),
        }
    }

    /// Время от первой метки шага до последней и число меток.
    pub fn profile_span(&self) -> Result<(f32, usize)> {
        match self.profile.as_ref() {
            Some(timeline) => Ok((timeline.total_ms()?, timeline.marks())),
            None => Ok((0.0, 0)),
        }
    }

    /// Открывает фазу: интервал до следующей метки уйдёт в `label`.
    fn mark(&mut self, label: &'static str) -> Result<()> {
        if let Some(timeline) = self.profile.as_mut() {
            timeline.mark(label, &self.stream)?;
        }
        Ok(())
    }

    pub fn logits(&self) -> &DeviceBuffer<f32> {
        &self.logits
    }

    /// VRAM, занятая кэшем и состоянием: она соревнуется с весами за карту.
    pub fn cache_bytes(&self) -> usize {
        let state: usize = self.state_pools.iter().map(DeviceBuffer::bytes).sum();
        let conv: usize = self.conv_pools.iter().map(DeviceBuffer::bytes).sum();
        let keys: usize = self.key_caches.iter().map(DeviceBuffer::bytes).sum();
        let values: usize = self.value_caches.iter().map(DeviceBuffer::bytes).sum();
        state + conv + keys + values
    }

    /// Один шаг decode для `tokens[i]` на позиции `positions[i]`.
    /// Логиты остаются в `self.logits` — их читает sampling.
    pub fn decode(
        &mut self,
        weights: &ModelWeights,
        tokens: &[u32],
        positions: &[u32],
    ) -> Result<()> {
        let batch = tokens.len();
        assert_eq!(positions.len(), batch);
        assert!(batch > 0 && batch <= self.max_batch);
        let max_position = *positions.iter().max().expect("непустой batch") as usize;
        assert!(
            max_position < self.max_context,
            "позиция {max_position} за пределами контекста {}",
            self.max_context
        );

        self.profile_reset();
        self.mark("upload")?;
        self.upload_step(tokens, positions)?;
        self.launch_decode(weights, batch, max_position)?;
        self.mark("end")
    }

    /// Executes scheduler-produced metadata against the persistent state/KV
    /// pools. Input tokens follow `BatchLayout::token_offsets`.
    pub fn execute_layout(
        &mut self,
        weights: &ModelWeights,
        layout: &BatchLayout,
        input_tokens: &[u32],
    ) -> Result<Vec<(u32, u32)>> {
        let sequences = layout.num_seqs();
        let decode = layout.num_decode as usize;
        assert!(sequences > 0 && sequences <= self.max_batch);
        assert!(decode <= sequences);
        assert_eq!(input_tokens.len(), layout.num_tokens());
        assert_eq!(layout.token_offsets.len(), sequences + 1);
        assert_eq!(layout.block_table_offsets.len(), sequences + 1);

        self.profile_reset();
        self.mark("upload")?;
        if sequences == decode {
            // Pure decode: this shape is captured in a CUDA graph, which is
            // where the single-sequence latency comes from. Keep it.
            let mut decode_tokens = Vec::with_capacity(decode);
            for row in 0..decode {
                let begin = layout.token_offsets[row] as usize;
                let end = layout.token_offsets[row + 1] as usize;
                assert_eq!(end - begin, 1, "decode layout must contain one token");
                decode_tokens.push(input_tokens[begin]);
            }
            let max_position = self.upload_layout_decode(layout, &decode_tokens)?;
            self.launch_decode(weights, decode, max_position)?;
        } else {
            // Mixed step: one forward pass over every token in the batch. The
            // previous shape ran one pass for decode plus one more per prefill
            // sequence, so a step read the weights several times over.
            let mut segments = Vec::with_capacity(sequences);
            let mut block_bounds = Vec::with_capacity(sequences + 1);
            block_bounds.push(0usize);
            let mut row_begin = 0usize;
            for row in 0..sequences {
                let tokens = (layout.token_offsets[row + 1] - layout.token_offsets[row]) as usize;
                let position_start = layout.position_starts[row] as usize;
                let state_slot = layout.state_slots[row] as usize;
                if position_start == 0 {
                    self.mark("reset")?;
                    self.reset_slot(state_slot)?;
                    self.mark("upload")?;
                }
                segments.push(Segment {
                    state_slot,
                    position_start,
                    row_begin,
                    tokens,
                    logits_row: row,
                });
                row_begin += tokens;
                block_bounds.push(layout.block_table_offsets[row + 1] as usize);
            }
            assert!(
                row_begin <= PREFILL_CHUNK_SIZE,
                "a step carries {row_begin} tokens, arena holds {PREFILL_CHUNK_SIZE}"
            );
            self.forward_segments(
                weights,
                &segments,
                decode,
                input_tokens,
                &layout.block_ids,
                &block_bounds,
            )?;
            self.stream.synchronize()?;
        }

        self.mark("sample")?;
        let sampled = self.argmax_to_host(sequences)?;
        self.mark("end")?;
        Ok(layout.seq_ids.iter().copied().zip(sampled).collect())
    }

    fn launch_decode(
        &mut self,
        weights: &ModelWeights,
        batch: usize,
        max_position: usize,
    ) -> Result<()> {
        assert_eq!(
            weights.decode_linear_mode(),
            self.decode_linear_mode,
            "weights and executor use different decode linear modes"
        );
        if self.profile.is_some() {
            // Захват графа не измеряется событиями: под таймлайном шаг идёт
            // обычными запусками, и цена этих запусков видна в сумме фаз.
            self.decode_forward(weights, batch, max_position)?;
            return self.stream.synchronize();
        }
        let weights_key = weights as *const ModelWeights as usize;
        if let Some(captured) = self.captured_weights {
            assert_eq!(
                captured, weights_key,
                "CUDA graph belongs to another ModelWeights instance"
            );
        } else {
            self.captured_weights = Some(weights_key);
        }
        let graph_index = batch - 1;
        if self.decode_graphs[graph_index].is_none() {
            CudaGraph::begin(&self.stream)?;
            if let Err(error) = self.decode_forward(weights, batch, max_position) {
                // End capture so the stream is not left permanently poisoned.
                let _ = CudaGraph::end(&self.stream);
                return Err(error);
            }
            self.decode_graphs[graph_index] = Some(CudaGraph::end(&self.stream)?);
        }
        self.decode_graphs[graph_index]
            .as_ref()
            .expect("graph was captured")
            .launch(&self.stream)?;
        self.stream.synchronize()
    }

    fn decode_forward(
        &mut self,
        weights: &ModelWeights,
        batch: usize,
        max_position: usize,
    ) -> Result<()> {
        self.mark("embed")?;
        weights
            .embed
            .gather_rows(&self.tokens, &mut self.residual, batch, &self.stream)?;

        // Схема слоя: residual несёт магистраль, а норма следующего блока
        // сама добавляет в неё выход предыдущего. Отдельного сложения нет —
        // это и есть fused residual-add в RmsNorm.
        self.mark("norm")?;
        weights.layers[0].input_norm.forward_bf16(
            &self.residual,
            None,
            &mut self.normed,
            batch,
            &self.stream,
        )?;

        let mut linear_layer = 0;
        let mut full_layer = 0;
        for (index, layer) in weights.layers.iter().enumerate() {
            match &layer.mixer {
                Mixer::Linear(mixer) => {
                    self.linear_attention(mixer, linear_layer, batch)?;
                    linear_layer += 1;
                }
                Mixer::Full(mixer) => {
                    self.full_attention(mixer, full_layer, batch, max_position)?;
                    full_layer += 1;
                }
            }

            self.mark("norm")?;
            layer.post_attention_norm.forward_bf16(
                &self.mixer_out,
                Some(&mut self.residual),
                &mut self.normed,
                batch,
                &self.stream,
            )?;

            self.mark("gemm.mlp_gate_up")?;
            if self.decode_linear_mode == DecodeLinearMode::Auto && batch <= nvfp4::MAX_W4A16_BATCH
            {
                nvfp4::swiglu_w4a16(
                    &layer.mlp.gate.linear,
                    &layer.mlp.up.linear,
                    &self.normed,
                    &mut self.mlp_hidden,
                    batch,
                    &self.stream,
                )?;
            } else {
                project_decode(
                    &layer.mlp.gate,
                    &self.normed,
                    &mut self.quant_hidden,
                    &mut self.mlp_gate,
                    &mut self.projection_workspace,
                    batch,
                    &self.stream,
                )?;
                project_decode(
                    &layer.mlp.up,
                    &self.normed,
                    &mut self.quant_hidden,
                    &mut self.mlp_up,
                    &mut self.projection_workspace,
                    batch,
                    &self.stream,
                )?;
                self.mark("swiglu")?;
                nvfp4::swiglu_bf16(
                    &self.mlp_gate,
                    &self.mlp_up,
                    &mut self.mlp_hidden,
                    batch,
                    INTERMEDIATE_SIZE,
                    &self.stream,
                )?;
            }
            self.mark("gemm.mlp_down")?;
            project_decode(
                &layer.mlp.down,
                &self.mlp_hidden,
                &mut self.quant_mlp,
                &mut self.mlp_out,
                &mut self.projection_workspace,
                batch,
                &self.stream,
            )?;

            let next_norm = match weights.layers.get(index + 1) {
                Some(next) => &next.input_norm,
                None => &weights.final_norm,
            };
            self.mark("norm")?;
            next_norm.forward_bf16(
                &self.mlp_out,
                Some(&mut self.residual),
                &mut self.normed,
                batch,
                &self.stream,
            )?;
        }

        self.mark("lm_head")?;
        weights
            .lm_head
            .logits(&self.normed, &mut self.logits, batch, &self.stream)
    }

    /// Causal tensor-core prefill for sequence slot zero. The prompt is split
    /// into fixed-M chunks; projections process 128 rows together while
    /// DeltaNet and attention see only the live prefix of the final chunk.
    /// Logits for the last prompt token are left in row zero of `self.logits`.
    pub fn prefill(
        &mut self,
        weights: &ModelWeights,
        tokens: &[u32],
        start_position: usize,
    ) -> Result<()> {
        self.prefill_sequence(weights, tokens, start_position, 0)
    }

    /// Causal prefill into an explicit persistent sequence slot.
    pub fn prefill_sequence(
        &mut self,
        weights: &ModelWeights,
        tokens: &[u32],
        start_position: usize,
        sequence: usize,
    ) -> Result<()> {
        assert!(sequence < self.max_batch, "invalid sequence slot");
        let block_ids: Vec<u32> = (0..self.max_blocks)
            .map(|block| (sequence * self.max_blocks + block) as u32)
            .collect();
        self.prefill_mapped(
            weights,
            tokens,
            start_position,
            sequence,
            sequence,
            &block_ids,
        )
    }

    fn prefill_mapped(
        &mut self,
        weights: &ModelWeights,
        tokens: &[u32],
        start_position: usize,
        state_slot: usize,
        logits_row: usize,
        block_ids: &[u32],
    ) -> Result<()> {
        assert!(!tokens.is_empty(), "prefill chunk is empty");
        assert!(state_slot < self.max_batch, "invalid state slot");
        assert!(logits_row < self.max_batch, "invalid logits row");
        assert!(
            start_position + tokens.len() <= self.max_context,
            "prefill exceeds configured context"
        );
        let needed_blocks = (start_position + tokens.len()).div_ceil(PAGE_SIZE);
        assert!(
            block_ids.len() >= needed_blocks,
            "incomplete KV block table"
        );
        assert!(
            block_ids
                .iter()
                .all(|&block| (block as usize) < self.max_batch * self.max_blocks),
            "KV block ID exceeds executor pool"
        );
        if start_position == 0 {
            self.reset_slot(state_slot)?;
        }
        for (chunk_index, chunk) in tokens.chunks(PREFILL_CHUNK_SIZE).enumerate() {
            let position = start_position + chunk_index * PREFILL_CHUNK_SIZE;
            let segment = Segment {
                state_slot,
                position_start: position,
                row_begin: 0,
                tokens: chunk.len(),
                logits_row,
            };
            self.forward_segments(
                weights,
                &[segment],
                0,
                chunk,
                block_ids,
                &[0, block_ids.len()],
            )?;
        }
        self.stream.synchronize()
    }

    /// One forward pass covering every segment of a step.
    ///
    /// Weights are read once here no matter how many sequences the step
    /// carries; only the DeltaNet scan, which touches recurrent state rather
    /// than weights, still runs per sequence.
    fn forward_segments(
        &mut self,
        weights: &ModelWeights,
        segments: &[Segment],
        decode_rows: usize,
        input_tokens: &[u32],
        block_ids: &[u32],
        block_bounds: &[usize],
    ) -> Result<()> {
        // Таймлайн вынимается из `self`: внутри прохода арена шага занята
        // изменяемым заимствованием, и метод `mark` туда уже не пройдёт.
        let mut timeline = self.profile.take();
        let result = self.forward_segments_inner(
            weights,
            segments,
            decode_rows,
            input_tokens,
            block_ids,
            block_bounds,
            timeline.as_mut(),
        );
        self.profile = timeline;
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_segments_inner(
        &mut self,
        weights: &ModelWeights,
        segments: &[Segment],
        decode_rows: usize,
        input_tokens: &[u32],
        block_ids: &[u32],
        block_bounds: &[usize],
        mut timeline: Option<&mut Timeline>,
    ) -> Result<()> {
        macro_rules! mark {
            ($label:expr) => {
                if let Some(tl) = timeline.as_deref_mut() {
                    tl.mark($label, &self.stream)?;
                }
            };
        }
        debug_assert!(!segments.is_empty());
        // Leading one-token rows share one recurrent-step launch; only the
        // remaining sequences need a scan of their own. Without this a step at
        // concurrency 64 would issue thousands of kernel launches per layer.
        debug_assert!(segments[..decode_rows].iter().all(|s| s.tokens == 1));
        for (index, segment) in segments[..decode_rows].iter().enumerate() {
            self.prefill.host_decode_slots[index] = segment.state_slot as u32;
        }
        if decode_rows > 0 {
            let slots = self.prefill.host_decode_slots[..decode_rows].to_vec();
            self.prefill.decode_slots.copy_from_slice_at(0, &slots)?;
        }
        let live: usize = segments.iter().map(|segment| segment.tokens).sum();
        // Attention partitioning is sized by the longest context in the step.
        let max_context = segments
            .iter()
            .map(|segment| segment.position_start + segment.tokens)
            .max()
            .expect("a step has at least one segment");
        debug_assert!(live > 0 && live <= PREFILL_CHUNK_SIZE);
        debug_assert_eq!(block_bounds.len(), segments.len() + 1);
        mark!("upload");
        self.upload_segments(segments, input_tokens, block_ids, block_bounds)?;

        let prefill = &mut self.prefill;
        mark!("embed");
        weights
            .embed
            .gather(&prefill.tokens, &mut prefill.residual, &self.stream)?;
        mark!("norm");
        weights.layers[0].input_norm.forward_bf16(
            &prefill.residual,
            None,
            &mut prefill.normed,
            live,
            &self.stream,
        )?;

        let mut linear_layer = 0;
        let mut full_layer = 0;
        for (index, layer) in weights.layers.iter().enumerate() {
            match &layer.mixer {
                Mixer::Linear(mixer) => {
                    mark!("gemm.la_qkv");
                    project_w4a4(
                        &mixer.qkv,
                        &prefill.normed,
                        &mut prefill.quant_hidden,
                        &mut prefill.mixed_qkv,
                        &mut prefill.projection_workspace,
                        live,
                        &self.stream,
                    )?;
                    mark!("gemm.la_z");
                    project_w4a4(
                        &mixer.z,
                        &prefill.normed,
                        &mut prefill.quant_hidden,
                        &mut prefill.z_gate,
                        &mut prefill.projection_workspace,
                        live,
                        &self.stream,
                    )?;
                    mark!("gemm.la_ab");
                    project_w4a4(
                        &mixer.a,
                        &prefill.normed,
                        &mut prefill.quant_hidden,
                        &mut prefill.a_projection,
                        &mut prefill.projection_workspace,
                        live,
                        &self.stream,
                    )?;
                    project_w4a4(
                        &mixer.b,
                        &prefill.normed,
                        &mut prefill.quant_hidden,
                        &mut prefill.b_projection,
                        &mut prefill.projection_workspace,
                        live,
                        &self.stream,
                    )?;
                    // The recurrence is per sequence, but the one-token rows
                    // are contiguous from row zero and share a single launch.
                    if decode_rows > 0 {
                        mark!("delta.prepare");
                        mixer.prepare.prepare_decode(
                            &prefill.mixed_qkv,
                            &prefill.a_projection,
                            &prefill.b_projection,
                            &mut self.conv_pools[linear_layer],
                            &prefill.decode_slots,
                            &mut prefill.prepared,
                            self.max_batch,
                            decode_rows,
                            &self.stream,
                        )?;
                        mark!("delta.scan");
                        delta_net::decode_slots(
                            &mut self.state_pools[linear_layer],
                            &prefill.decode_slots,
                            &prefill.prepared.inputs(),
                            &mut prefill.delta_out,
                            self.max_batch,
                            decode_rows,
                            &self.stream,
                        )?;
                    }
                    for segment in &segments[decode_rows..] {
                        mark!("delta.prepare_prefill");
                        mixer.prepare.prepare_prefill(
                            &prefill.mixed_qkv,
                            &prefill.a_projection,
                            &prefill.b_projection,
                            &mut self.conv_pools[linear_layer],
                            &mut prefill.prepared,
                            self.max_batch,
                            segment.state_slot,
                            segment.tokens,
                            segment.row_begin,
                            &self.stream,
                        )?;
                        mark!("delta.scan_prefill");
                        delta_net::prefill_slot(
                            &mut self.state_pools[linear_layer],
                            &prefill.prepared.inputs(),
                            &mut prefill.delta_out,
                            self.max_batch,
                            segment.state_slot,
                            segment.tokens,
                            segment.row_begin,
                            self.delta_state_mode,
                            &self.stream,
                        )?;
                    }
                    mark!("delta.norm");
                    mixer.output_norm.forward(
                        &prefill.delta_out,
                        &prefill.z_gate,
                        &mut prefill.delta_normed,
                        live,
                        &self.stream,
                    )?;
                    mark!("gemm.la_out");
                    project_w4a4(
                        &mixer.out,
                        &prefill.delta_normed,
                        &mut prefill.quant_mixer,
                        &mut prefill.mixer_out,
                        &mut prefill.projection_workspace,
                        live,
                        &self.stream,
                    )?;
                    linear_layer += 1;
                }
                Mixer::Full(mixer) => {
                    mark!("gemm.attn_qkv");
                    project_w4a4(
                        &mixer.q,
                        &prefill.normed,
                        &mut prefill.quant_hidden,
                        &mut prefill.query_gate,
                        &mut prefill.projection_workspace,
                        live,
                        &self.stream,
                    )?;
                    project_w4a4(
                        &mixer.k,
                        &prefill.normed,
                        &mut prefill.quant_hidden,
                        &mut prefill.key_projection,
                        &mut prefill.projection_workspace,
                        live,
                        &self.stream,
                    )?;
                    project_w4a4(
                        &mixer.v,
                        &prefill.normed,
                        &mut prefill.quant_hidden,
                        &mut prefill.value_projection,
                        &mut prefill.projection_workspace,
                        live,
                        &self.stream,
                    )?;
                    let blocks = self.max_batch * self.max_blocks;
                    mark!("attn.prepare");
                    mixer.prepare.prepare_decode(
                        &prefill.query_gate,
                        &prefill.key_projection,
                        &prefill.value_projection,
                        &prefill.cosine,
                        &prefill.sine,
                        &prefill.physical_blocks,
                        &prefill.block_offsets,
                        &mut prefill.query,
                        &mut self.key_caches[full_layer],
                        &mut self.value_caches[full_layer],
                        blocks,
                        live,
                        self.kv_cache_dtype,
                        &self.stream,
                    )?;
                    mark!("attn.decode");
                    paged_attention::decode_gated(
                        &prefill.query,
                        &prefill.query_gate,
                        &self.key_caches[full_layer],
                        &self.value_caches[full_layer],
                        blocks,
                        &prefill.block_tables,
                        &prefill.context_lengths,
                        self.max_blocks,
                        &mut prefill.attention_out,
                        &mut prefill.attention_workspace,
                        live,
                        max_context,
                        self.kv_cache_dtype,
                        &self.stream,
                    )?;
                    mark!("gemm.attn_out");
                    project_w4a4(
                        &mixer.o,
                        &prefill.attention_out,
                        &mut prefill.quant_mixer,
                        &mut prefill.mixer_out,
                        &mut prefill.projection_workspace,
                        live,
                        &self.stream,
                    )?;
                    full_layer += 1;
                }
            }

            mark!("norm");
            layer.post_attention_norm.forward_bf16(
                &prefill.mixer_out,
                Some(&mut prefill.residual),
                &mut prefill.normed,
                live,
                &self.stream,
            )?;
            mark!("gemm.mlp_gate_up");
            project_w4a4(
                &layer.mlp.gate,
                &prefill.normed,
                &mut prefill.quant_hidden,
                &mut prefill.mlp_gate,
                &mut prefill.projection_workspace,
                live,
                &self.stream,
            )?;
            project_w4a4(
                &layer.mlp.up,
                &prefill.normed,
                &mut prefill.quant_hidden,
                &mut prefill.mlp_up,
                &mut prefill.projection_workspace,
                live,
                &self.stream,
            )?;
            mark!("swiglu");
            nvfp4::swiglu_bf16(
                &prefill.mlp_gate,
                &prefill.mlp_up,
                &mut prefill.mlp_hidden,
                live,
                INTERMEDIATE_SIZE,
                &self.stream,
            )?;
            mark!("gemm.mlp_down");
            project_w4a4(
                &layer.mlp.down,
                &prefill.mlp_hidden,
                &mut prefill.quant_mlp,
                &mut prefill.mlp_out,
                &mut prefill.projection_workspace,
                live,
                &self.stream,
            )?;

            let next_norm = match weights.layers.get(index + 1) {
                Some(next) => &next.input_norm,
                None => &weights.final_norm,
            };
            mark!("norm");
            next_norm.forward_bf16(
                &prefill.mlp_out,
                Some(&mut prefill.residual),
                &mut prefill.normed,
                live,
                &self.stream,
            )?;
        }

        // One vocabulary pass for the whole step. Running the projection per
        // sequence would re-read the whole 1.27 GB lm_head each time, so the
        // last row of every segment is gathered into a dense buffer first.
        if let [only] = segments {
            mark!("lm_head");
            return weights.lm_head.logits_row_to(
                &prefill.normed,
                only.row_begin + only.tokens - 1,
                &mut self.logits,
                only.logits_row,
                &self.stream,
            );
        }
        for (index, segment) in segments.iter().enumerate() {
            debug_assert_eq!(
                segment.logits_row, index,
                "a fused step writes logits in segment order"
            );
            prefill.host_last_rows[index] = (segment.row_begin + segment.tokens - 1) as u32;
        }
        prefill
            .last_rows
            .copy_from_slice_at(0, &prefill.host_last_rows[..segments.len()])?;
        mark!("gather_rows");
        qwc_cuda::gather_rows_bf16(
            &prefill.normed,
            &prefill.last_rows,
            &mut prefill.last_hidden,
            segments.len(),
            HIDDEN_SIZE,
            &self.stream,
        )?;
        mark!("lm_head");
        weights.lm_head.logits(
            &prefill.last_hidden,
            &mut self.logits,
            segments.len(),
            &self.stream,
        )
    }

    fn linear_attention(
        &mut self,
        mixer: &LinearAttention,
        layer: usize,
        batch: usize,
    ) -> Result<()> {
        self.mark("gemm.la_qkv")?;
        project_decode(
            &mixer.qkv,
            &self.normed,
            &mut self.quant_hidden,
            &mut self.mixed_qkv,
            &mut self.projection_workspace,
            batch,
            &self.stream,
        )?;
        self.mark("gemm.la_z")?;
        project_decode(
            &mixer.z,
            &self.normed,
            &mut self.quant_hidden,
            &mut self.z_gate,
            &mut self.projection_workspace,
            batch,
            &self.stream,
        )?;
        self.mark("gemm.la_ab")?;
        project_decode(
            &mixer.a,
            &self.normed,
            &mut self.quant_hidden,
            &mut self.a_projection,
            &mut self.projection_workspace,
            batch,
            &self.stream,
        )?;
        project_decode(
            &mixer.b,
            &self.normed,
            &mut self.quant_hidden,
            &mut self.b_projection,
            &mut self.projection_workspace,
            batch,
            &self.stream,
        )?;

        self.mark("delta.prepare")?;
        mixer.prepare.prepare_decode(
            &self.mixed_qkv,
            &self.a_projection,
            &self.b_projection,
            &mut self.conv_pools[layer],
            &self.state_slots,
            &mut self.prepared,
            self.max_batch,
            batch,
            &self.stream,
        )?;
        self.mark("delta.scan")?;
        delta_net::decode_slots(
            &mut self.state_pools[layer],
            &self.state_slots,
            &self.prepared.inputs(),
            &mut self.delta_out,
            self.max_batch,
            batch,
            &self.stream,
        )?;
        self.mark("delta.norm")?;
        mixer.output_norm.forward(
            &self.delta_out,
            &self.z_gate,
            &mut self.delta_normed,
            batch,
            &self.stream,
        )?;
        self.mark("gemm.la_out")?;
        project_decode(
            &mixer.out,
            &self.delta_normed,
            &mut self.quant_mixer,
            &mut self.mixer_out,
            &mut self.projection_workspace,
            batch,
            &self.stream,
        )
    }

    fn full_attention(
        &mut self,
        mixer: &FullAttention,
        layer: usize,
        batch: usize,
        max_position: usize,
    ) -> Result<()> {
        self.mark("gemm.attn_qkv")?;
        project_decode(
            &mixer.q,
            &self.normed,
            &mut self.quant_hidden,
            &mut self.query_gate,
            &mut self.projection_workspace,
            batch,
            &self.stream,
        )?;
        project_decode(
            &mixer.k,
            &self.normed,
            &mut self.quant_hidden,
            &mut self.key_projection,
            &mut self.projection_workspace,
            batch,
            &self.stream,
        )?;
        project_decode(
            &mixer.v,
            &self.normed,
            &mut self.quant_hidden,
            &mut self.value_projection,
            &mut self.projection_workspace,
            batch,
            &self.stream,
        )?;

        let blocks = self.max_batch * self.max_blocks;
        self.mark("attn.prepare")?;
        mixer.prepare.prepare_decode(
            &self.query_gate,
            &self.key_projection,
            &self.value_projection,
            &self.cosine,
            &self.sine,
            &self.physical_blocks,
            &self.block_offsets,
            &mut self.query,
            &mut self.key_caches[layer],
            &mut self.value_caches[layer],
            blocks,
            batch,
            self.kv_cache_dtype,
            &self.stream,
        )?;
        self.mark("attn.decode")?;
        paged_attention::decode_gated(
            &self.query,
            &self.query_gate,
            &self.key_caches[layer],
            &self.value_caches[layer],
            blocks,
            &self.block_tables,
            &self.context_lengths,
            self.max_blocks,
            &mut self.attention_out,
            &mut self.workspace,
            batch,
            max_position + 1,
            self.kv_cache_dtype,
            &self.stream,
        )?;
        self.mark("gemm.attn_out")?;
        project_decode(
            &mixer.o,
            &self.attention_out,
            &mut self.quant_mixer,
            &mut self.mixer_out,
            &mut self.projection_workspace,
            batch,
            &self.stream,
        )
    }

    fn upload_layout_decode(&mut self, layout: &BatchLayout, tokens: &[u32]) -> Result<usize> {
        let batch = tokens.len();
        assert!(layout.state_slots.len() >= batch);
        assert!(layout.position_starts.len() >= batch);
        assert!(layout.context_lens.len() >= batch);
        self.host_block_tables.fill(0);
        let total_blocks = self.max_batch * self.max_blocks;
        let mut max_position = 0usize;
        for row in 0..batch {
            let position = layout.position_starts[row] as usize;
            let context = layout.context_lens[row] as usize;
            assert!(position < self.max_context && context <= self.max_context);
            assert!((layout.state_slots[row] as usize) < self.max_batch);
            max_position = max_position.max(position);
            for dimension in 0..ROPE_DIM {
                let pair = dimension % (ROPE_DIM / 2);
                let inverse = (ROPE_THETA as f32).powf(-(2.0 * pair as f32) / ROPE_DIM as f32);
                let angle = position as f32 * inverse;
                self.host_cosine[row * ROPE_DIM + dimension] = bf16::from_f32(angle.cos());
                self.host_sine[row * ROPE_DIM + dimension] = bf16::from_f32(angle.sin());
            }
            let block_begin = layout.block_table_offsets[row] as usize;
            let block_end = layout.block_table_offsets[row + 1] as usize;
            let blocks = &layout.block_ids[block_begin..block_end];
            assert!(!blocks.is_empty() && blocks.len() <= self.max_blocks);
            assert!(blocks.iter().all(|&block| (block as usize) < total_blocks));
            let logical_block = position / PAGE_SIZE;
            assert!(logical_block < blocks.len());
            self.host_physical[row] = blocks[logical_block];
            self.host_offsets[row] = (position % PAGE_SIZE) as u32;
            self.host_lengths[row] = context as u32;
            let table =
                &mut self.host_block_tables[row * self.max_blocks..(row + 1) * self.max_blocks];
            table[..blocks.len()].copy_from_slice(blocks);
        }
        self.tokens.copy_from_slice(tokens)?;
        self.state_slots
            .copy_from_slice(&layout.state_slots[..batch])?;
        self.cosine
            .copy_from_slice(&self.host_cosine[..batch * ROPE_DIM])?;
        self.sine
            .copy_from_slice(&self.host_sine[..batch * ROPE_DIM])?;
        self.physical_blocks
            .copy_from_slice(&self.host_physical[..batch])?;
        self.block_offsets
            .copy_from_slice(&self.host_offsets[..batch])?;
        self.context_lengths
            .copy_from_slice(&self.host_lengths[..batch])?;
        self.block_tables
            .copy_from_slice(&self.host_block_tables[..batch * self.max_blocks])?;
        Ok(max_position)
    }

    /// Позиционные таблицы и адреса страницы нового токена.
    fn upload_step(&mut self, tokens: &[u32], positions: &[u32]) -> Result<()> {
        self.host_block_tables.fill(0);
        for (sequence, &position) in positions.iter().enumerate() {
            for dimension in 0..ROPE_DIM {
                let pair = dimension % (ROPE_DIM / 2);
                let inverse = (ROPE_THETA as f32).powf(-(2.0 * pair as f32) / ROPE_DIM as f32);
                let angle = position as f32 * inverse;
                self.host_cosine[sequence * ROPE_DIM + dimension] = bf16::from_f32(angle.cos());
                self.host_sine[sequence * ROPE_DIM + dimension] = bf16::from_f32(angle.sin());
            }
            let block = position as usize / PAGE_SIZE;
            self.host_physical[sequence] = (sequence * self.max_blocks + block) as u32;
            self.host_offsets[sequence] = (position as usize % PAGE_SIZE) as u32;
            self.host_lengths[sequence] = position + 1;
            for block in 0..self.max_blocks {
                self.host_block_tables[sequence * self.max_blocks + block] =
                    (sequence * self.max_blocks + block) as u32;
            }
        }
        self.tokens.copy_from_slice(tokens)?;
        self.state_slots
            .copy_from_slice(&self.host_state_slots[..tokens.len()])?;
        self.cosine
            .copy_from_slice(&self.host_cosine[..tokens.len() * ROPE_DIM])?;
        self.sine
            .copy_from_slice(&self.host_sine[..tokens.len() * ROPE_DIM])?;
        self.physical_blocks
            .copy_from_slice(&self.host_physical[..tokens.len()])?;
        self.block_offsets
            .copy_from_slice(&self.host_offsets[..tokens.len()])?;
        self.context_lengths
            .copy_from_slice(&self.host_lengths[..tokens.len()])?;
        self.block_tables
            .copy_from_slice(&self.host_block_tables[..tokens.len() * self.max_blocks])
    }

    /// Per-row metadata for a fused step. Every row carries its own position,
    /// physical page and context length, so sequences at different offsets mix
    /// freely inside one arena.
    fn upload_segments(
        &mut self,
        segments: &[Segment],
        input_tokens: &[u32],
        block_ids: &[u32],
        block_bounds: &[usize],
    ) -> Result<()> {
        let prefill = &mut self.prefill;
        let live: usize = segments.iter().map(|segment| segment.tokens).sum();
        assert_eq!(
            input_tokens.len(),
            live,
            "token count does not match segments"
        );
        // Padding rows go through GEMMs but never through a causal state update;
        // token zero merely provides a finite embedding for those dead rows.
        prefill.host_tokens.fill(0);
        prefill.host_tokens[..live].copy_from_slice(input_tokens);
        prefill.host_block_tables.fill(0);

        for (index, segment) in segments.iter().enumerate() {
            let blocks = &block_ids[block_bounds[index]..block_bounds[index + 1]];
            assert!(blocks.len() <= self.max_blocks, "KV block table overflows");
            for step in 0..segment.tokens {
                let row = segment.row_begin + step;
                let position = segment.position_start + step;
                for dimension in 0..ROPE_DIM {
                    let pair = dimension % (ROPE_DIM / 2);
                    let inverse = (ROPE_THETA as f32).powf(-(2.0 * pair as f32) / ROPE_DIM as f32);
                    let angle = position as f32 * inverse;
                    prefill.host_cosine[row * ROPE_DIM + dimension] = bf16::from_f32(angle.cos());
                    prefill.host_sine[row * ROPE_DIM + dimension] = bf16::from_f32(angle.sin());
                }
                prefill.host_physical[row] = blocks[position / PAGE_SIZE];
                prefill.host_offsets[row] = (position % PAGE_SIZE) as u32;
                prefill.host_lengths[row] = (position + 1) as u32;
                let table = &mut prefill.host_block_tables
                    [row * self.max_blocks..(row + 1) * self.max_blocks];
                table[..blocks.len()].copy_from_slice(blocks);
            }
        }

        prefill.tokens.copy_from_slice(&prefill.host_tokens)?;
        prefill
            .cosine
            .copy_from_slice(&prefill.host_cosine[..live * ROPE_DIM])?;
        prefill
            .sine
            .copy_from_slice(&prefill.host_sine[..live * ROPE_DIM])?;
        prefill
            .physical_blocks
            .copy_from_slice(&prefill.host_physical[..live])?;
        prefill
            .block_offsets
            .copy_from_slice(&prefill.host_offsets[..live])?;
        prefill
            .context_lengths
            .copy_from_slice(&prefill.host_lengths[..live])?;
        prefill
            .block_tables
            .copy_from_slice(&prefill.host_block_tables)
    }

    /// Забыть контекст: состояние DeltaNet и conv-история обнуляются, KV
    /// перестаёт быть виден через context_lengths.
    pub fn reset(&mut self) -> Result<()> {
        for pool in &mut self.state_pools {
            pool.zero_range(0, pool.len())?;
        }
        for pool in &mut self.conv_pools {
            pool.zero_range(0, pool.len())?;
        }
        Ok(())
    }

    pub fn reset_slot(&mut self, slot: usize) -> Result<()> {
        assert!(slot < self.max_batch);
        for pool in &mut self.state_pools {
            pool.zero_range(slot * STATE_ELEMS, STATE_ELEMS)?;
        }
        for pool in &mut self.conv_pools {
            pool.zero_range(slot * CONV_STATE_ELEMS, CONV_STATE_ELEMS)?;
        }
        Ok(())
    }

    /// Логиты одной последовательности на хост.
    pub fn logits_to_host(&self, sequence: usize) -> Result<Vec<f32>> {
        let all = self.logits.to_vec()?;
        let start = sequence * VOCAB_SIZE;
        Ok(all[start..start + VOCAB_SIZE].to_vec())
    }

    /// Greedy sampling on the GPU; only the resulting token IDs cross PCIe.
    pub fn argmax_to_host(&mut self, batch: usize) -> Result<Vec<u32>> {
        self.sampler.sample(&self.logits, batch, &self.stream)?;
        self.sampler.to_host(batch)
    }
}

fn project_decode(
    projection: &Projection,
    input: &DeviceBuffer<u16>,
    quantized: &mut QuantizedActivation,
    output: &mut DeviceBuffer<u16>,
    workspace: &mut W4A4Workspace,
    batch: usize,
    stream: &Stream,
) -> Result<()> {
    let kernel = match projection.decode_linear_mode {
        DecodeLinearMode::Auto => nvfp4::select_decode_kernel(
            batch,
            projection.linear.out_features(),
            projection.linear.in_features(),
        ),
        DecodeLinearMode::W4A4 => nvfp4::DecodeKernel::W4A4,
    };
    match kernel {
        nvfp4::DecodeKernel::W4A16 => projection
            .linear
            .forward_w4a16(input, output, batch, stream),
        nvfp4::DecodeKernel::W4A4 => {
            let rows = match projection.decode_linear_mode {
                DecodeLinearMode::Auto => batch,
                DecodeLinearMode::W4A4 => batch.max(3),
            };
            quantized.quantize_bf16_rows(input, projection.input_global_scale, rows, stream)?;
            projection
                .linear
                .forward_w4a4_quantized(quantized, output, workspace, stream)
        }
    }
}

/// `rows` is the live prefix of the arena. Quantizing and multiplying the dead
/// padding rows as well costs real tensor-core time once the arena is large, so
/// a fused step pays only for the tokens it carries. CUTLASS needs M >= 3.
fn project_w4a4(
    projection: &Projection,
    input: &DeviceBuffer<u16>,
    quantized: &mut QuantizedActivation,
    output: &mut DeviceBuffer<u16>,
    workspace: &mut W4A4Workspace,
    rows: usize,
    stream: &Stream,
) -> Result<()> {
    quantized.quantize_bf16_rows(input, projection.input_global_scale, rows.max(3), stream)?;
    projection
        .linear
        .forward_w4a4_quantized(quantized, output, workspace, stream)
}

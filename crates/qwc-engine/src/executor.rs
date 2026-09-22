//! Шаг decode целиком: от идентификатора токена до логитов.
//!
//! Все буферы шага резидентны и выделяются один раз: адреса не меняются от
//! шага к шагу, иначе позже нечего будет захватывать в CUDA graph. Планировщик
//! из `qwc-runtime` сюда пока не подключён — блоки KV и слоты состояния
//! раздаются по последовательностям статически, чтобы сначала проверить
//! численный тракт, а не политику вытеснения.

use crate::weights::{
    DecodeLinearMode, FullAttention, LinearAttention, MIXER_A_OFFSET, MIXER_B_OFFSET,
    MIXER_FUSED_WIDTH, MIXER_Z_OFFSET, Mixer, ModelWeights, Projection,
};
use qwc_core::arch::*;
use qwc_cuda::delta_net::{
    self, CONV_STATE_ELEMS, DeltaPrefillWorkspace, DeltaStateMode, PackedStatePool, PreparedDelta,
    RowView, STATE_ELEMS, V_ELEMS,
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

/// Максимум строк в одном шаге проверки черновиков: принятый токен плюс до
/// семи спекулятивных. Потолок задан bf16-линейкой MTP-головы.
pub const MAX_SPECULATION_ROWS: usize = 8;

/// С какой длины отрезка prefill-скан идёт чанковой (WY) формой.
/// Ниже этого рекуррентный проход быстрее: см. `deltanet-wy-2026-09-16.md`,
/// на 128 токенах формы сравниваются, на 256 WY уже вдвое лучше.
const MIN_WY_TOKENS: usize = 192;
/// Fixed tensor-core M dimension used by chunked prefill. Это ёмкость арены:
/// планировщику можно выдать бюджет меньше, но не больше.
///
/// Every engine step reads all 16.25 GB of weights regardless of how many
/// tokens it carries, so a small chunk makes prefill pay that read over and
/// over. Но упирается всё не в вес чтения, а в форму: W4A4 на M=511 идёт на
/// 1097 TFLOP/s, на M=2048 — на 1350, а внимание на большом тайле реже
/// перечитывает KV. Замер на формах модели (`prefillgemmbench`,
/// `prefillattnbench`, промпт 4096):
///
/// | чанк | проекции | внимание |
/// |---|---:|---:|
/// | 512 | 44.1 мкс/токен | 23.3 мкс/токен |
/// | 1024 | 38.8 | 19.7 |
/// | 2048 | 35.8 | 17.2 |
///
/// Плата — гранулярность шага: decode-строка ждёт весь чанк, поэтому 2048 это
/// потолок, а не обязательный бюджет.
pub const PREFILL_CHUNK_SIZE: usize = qwc_cuda::MAX_STEP_ROWS;

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
    /// Total number of physical pages shared by every sequence. This is
    /// deliberately independent of `max_batch * max_blocks`: 32K is a per-
    /// request address-space limit, not a promise to reserve 32K for every
    /// concurrent request.
    kv_pool_blocks: usize,
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
    /// Слитый вход миксера: qkv, z, a и b одной строкой.
    mixer_in: DeviceBuffer<u16>,
    prepared: PreparedDelta,
    delta_out: DeviceBuffer<f32>,
    delta_normed: DeviceBuffer<u16>,
    state_slots: DeviceBuffer<u32>,
    /// Состояние покоящихся слотов: int8 плюс построчный масштаб. Decode
    /// читает и пишет прямо сюда — в этом половина экономии трафика шага.
    state_pools: Vec<PackedStatePool>,
    /// Расквантованная копия состояния одной последовательности, по слоям.
    /// Префилл идёт по ней, а не по 8-битному слоту: сегмент, стартующий с
    /// квантованного состояния, портит весь остаток промпта — MAE
    /// подскакивает в 11 раз на границе `PREFILL_CHUNK_SIZE`
    /// (`bench/results/delta-state-8bit.md`).
    ///
    /// Копия одна на движок, потому что недопрефилленной в каждый момент
    /// может быть ровно одна последовательность: чанк выходит частичным
    /// только когда исчерпал token budget шага, а это обрывает набор
    /// (`qwc-runtime/src/scheduler.rs`).
    state_work: Vec<DeviceBuffer<u16>>,
    /// Чей слот лежит в копии — по слою, потому что сегменты проходят все
    /// слои по очереди и распаковка каждого слоя своя.
    work_owner: Vec<Option<usize>>,
    conv_pools: Vec<DeviceBuffer<f32>>,
    /// Скретч WY-скана. Общий на все слои и все последовательности: скан
    /// вызывается до 48 раз за шаг, и аллокация внутри вызова сериализовала
    /// бы этот путь. Заводится только в режиме `Wy` — он стоит десятки
    /// мегабайт, которые остальным режимам не нужны.
    delta_prefill: Option<DeltaPrefillWorkspace>,
    /// Входы миксера всех линейных слоёв за шаг проверки черновиков.
    ///
    /// Проверка идёт по черновому слоту состояния, а принятые токены потом
    /// доигрываются на настоящем — для этого и нужны сохранённые входы: они
    /// позволяют повторить conv и скан, не перечитывая веса второй раз.
    speculation_mixer: Vec<DeviceBuffer<u16>>,
    speculation_rows: usize,

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

    /// Слитый вход миксера: qkv, z, a и b одной строкой.
    mixer_in: DeviceBuffer<u16>,
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
            (MIXER_FUSED_WIDTH, HIDDEN_SIZE),
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
            mixer_in: DeviceBuffer::zeroed(rows * MIXER_FUSED_WIDTH)?,
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
            // Workspace обслуживает только строки decode, а их в шаге не
            // больше, чем последовательностей: разметка партиций рассчитана
            // на MAX_DECODE_ROWS.
            attention_workspace: PagedAttentionWorkspace::new(
                rows.min(paged_attention::MAX_DECODE_ROWS),
                max_context,
            )?,
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
            DeltaStateMode::Wy,
        )
    }

    pub fn new_with_kv_cache(config: ExecutorConfig, kv_cache_dtype: KvCacheDtype) -> Result<Self> {
        Self::new_with_options(
            config,
            kv_cache_dtype,
            DecodeLinearMode::Auto,
            DeltaStateMode::Wy,
        )
    }

    /// Builds an executor backed by one shared physical KV page pool.
    ///
    /// `kv_pool_blocks` is the global capacity handed to `CacheManager`; the
    /// logical block table of a sequence may still contain up to
    /// `ceil(max_context / PAGE_SIZE)` entries.
    pub fn new_with_kv_pool(
        config: ExecutorConfig,
        kv_cache_dtype: KvCacheDtype,
        kv_pool_blocks: usize,
    ) -> Result<Self> {
        Self::new_with_pool_options(
            config,
            kv_cache_dtype,
            DecodeLinearMode::Auto,
            DeltaStateMode::Wy,
            Some(kv_pool_blocks),
        )
    }

    pub fn new_with_options(
        config: ExecutorConfig,
        kv_cache_dtype: KvCacheDtype,
        decode_linear_mode: DecodeLinearMode,
        delta_state_mode: DeltaStateMode,
    ) -> Result<Self> {
        Self::new_with_pool_options(
            config,
            kv_cache_dtype,
            decode_linear_mode,
            delta_state_mode,
            None,
        )
    }

    /// Полный набор переключателей: KV, путь проекций decode и форма
    /// prefill-скана. Стенды берут именно его, потому что свипают все три.
    pub fn new_with_pool_options(
        config: ExecutorConfig,
        kv_cache_dtype: KvCacheDtype,
        decode_linear_mode: DecodeLinearMode,
        delta_state_mode: DeltaStateMode,
        kv_pool_blocks: Option<usize>,
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
        let kv_pool_blocks = kv_pool_blocks.unwrap_or(batch * max_blocks);
        assert!(
            kv_pool_blocks >= max_blocks,
            "KV pool must hold at least one max-context sequence ({max_blocks} blocks)"
        );
        assert!(
            u32::try_from(kv_pool_blocks).is_ok(),
            "KV pool block IDs must fit in u32"
        );
        let cache_elements = kv_pool_blocks * NUM_KV_HEADS * PAGE_SIZE * ATTN_HEAD_DIM;
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

        let speculation_mixer = Vec::new();
        let speculation_rows = 0;
        let delta_prefill = match delta_state_mode {
            DeltaStateMode::Wy => Some(DeltaPrefillWorkspace::new()?),
            _ => None,
        };
        let mut state_pools = Vec::with_capacity(NUM_LINEAR_LAYERS);
        let mut state_work = Vec::with_capacity(NUM_LINEAR_LAYERS);
        let mut conv_pools = Vec::with_capacity(NUM_LINEAR_LAYERS);
        for _ in 0..NUM_LINEAR_LAYERS {
            // Слотов на один больше, чем последовательностей: последний —
            // черновой. Проверка спекуляции гоняет состояние по нему, а
            // настоящее остаётся нетронутым, пока не станет ясно, сколько
            // токенов принято.
            state_pools.push(PackedStatePool::zeroed(batch + 1)?);
            state_work.push(DeviceBuffer::zeroed(STATE_ELEMS)?);
            conv_pools.push(DeviceBuffer::zeroed((batch + 1) * CONV_STATE_ELEMS)?);
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
            kv_pool_blocks,
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
                    (MIXER_FUSED_WIDTH, HIDDEN_SIZE),
                    (HIDDEN_SIZE, LA_V_PROJ_DIM),
                    (2 * Q_PROJ_DIM, HIDDEN_SIZE),
                    (KV_PROJ_DIM, HIDDEN_SIZE),
                    (HIDDEN_SIZE, Q_PROJ_DIM),
                    (INTERMEDIATE_SIZE, HIDDEN_SIZE),
                    (HIDDEN_SIZE, INTERMEDIATE_SIZE),
                ],
            )?,
            mixer_in: DeviceBuffer::zeroed(decode_rows * MIXER_FUSED_WIDTH)?,
            prepared: PreparedDelta::zeroed(batch)?,
            delta_out: DeviceBuffer::zeroed(batch * V_ELEMS)?,
            delta_normed: DeviceBuffer::zeroed(decode_rows * V_ELEMS)?,
            state_slots: DeviceBuffer::from_slice(&slots)?,
            state_pools,
            state_work,
            work_owner: vec![None; NUM_LINEAR_LAYERS],
            conv_pools,
            delta_prefill,
            speculation_mixer,
            speculation_rows,
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
            // Проверка черновиков просит логиты сразу на k+1 строк, даже
            // когда последовательность одна.
            logits: DeviceBuffer::zeroed(batch.max(MAX_SPECULATION_ROWS) * VOCAB_SIZE)?,
            sampler: Argmax::new(batch.max(MAX_SPECULATION_ROWS), VOCAB_SIZE)?,
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
        let state: usize = self.state_pools.iter().map(PackedStatePool::bytes).sum();
        let work: usize = self.state_work.iter().map(DeviceBuffer::bytes).sum();
        let conv: usize = self.conv_pools.iter().map(DeviceBuffer::bytes).sum();
        let keys: usize = self.key_caches.iter().map(DeviceBuffer::bytes).sum();
        let values: usize = self.value_caches.iter().map(DeviceBuffer::bytes).sum();
        // Скретч WY-скана живёт столько же, сколько состояние, и в бюджете
        // карты весит так же — молчать о нём значит занижать отчёт.
        let scan: usize = self
            .delta_prefill
            .as_ref()
            .map_or(0, DeltaPrefillWorkspace::bytes);
        state + work + conv + keys + values + scan
    }

    /// Прогон k+1 токенов по черновому слоту состояния: возвращает argmax
    /// модели для каждой строки.
    ///
    /// Настоящее состояние последовательности не трогается — пока неизвестно,
    /// сколько токенов принято, трогать его нельзя. KV-страницы пишутся сразу
    /// всем строкам: отвергнутые позиции перезапишет следующий шаг.
    ///
    /// `block_ids` — страницы KV той же последовательности. Раздаёт их
    /// `CacheManager`, и номера у него произвольные, поэтому выводить таблицу
    /// из индекса слота нельзя: в сервере это чужие страницы.
    pub fn verify_speculation(
        &mut self,
        weights: &ModelWeights,
        tokens: &[u32],
        start_position: usize,
        sequence: usize,
        block_ids: &[u32],
    ) -> Result<Vec<u32>> {
        let rows = tokens.len();
        assert!((1..=MAX_SPECULATION_ROWS).contains(&rows));
        assert!(sequence < self.max_batch);
        assert!(start_position + rows <= self.max_context);

        if self.speculation_mixer.is_empty() {
            for _ in 0..NUM_LINEAR_LAYERS {
                self.speculation_mixer.push(DeviceBuffer::zeroed(
                    MAX_SPECULATION_ROWS * MIXER_FUSED_WIDTH,
                )?);
            }
        }
        let scratch = self.speculation_slot();
        for layer in 0..NUM_LINEAR_LAYERS {
            let (state, conv) = (&mut self.state_pools[layer], &mut self.conv_pools[layer]);
            state.copy_slot(sequence, scratch, &self.stream)?;
            conv.copy_within(
                scratch * CONV_STATE_ELEMS,
                sequence * CONV_STATE_ELEMS,
                CONV_STATE_ELEMS,
                &self.stream,
            )?;
        }

        self.speculation_rows = rows;
        assert!(
            block_ids.len() >= (start_position + rows).div_ceil(PAGE_SIZE),
            "таблица блоков короче проверяемых позиций"
        );
        let segment = Segment {
            state_slot: scratch,
            position_start: start_position,
            row_begin: 0,
            tokens: rows,
            logits_row: 0,
        };
        let result = self.forward_segments(
            weights,
            &[segment],
            0,
            tokens,
            block_ids,
            &[0, block_ids.len()],
        );
        self.speculation_rows = 0;
        result?;
        self.stream.synchronize()?;
        self.argmax_to_host(rows)
    }

    /// Принять первые `accepted` строк последнего шага проверки.
    ///
    /// Если приняты все, достаточно перенести черновое состояние в слот
    /// последовательности. Если нет — conv и скан доигрываются по сохранённым
    /// входам миксера, и второго прохода по весам это не стоит.
    pub fn commit_speculation(
        &mut self,
        weights: &ModelWeights,
        rows: usize,
        accepted: usize,
        sequence: usize,
    ) -> Result<()> {
        assert!(accepted > 0 && accepted <= rows && rows <= MAX_SPECULATION_ROWS);
        let scratch = self.speculation_slot();
        if accepted == rows {
            for layer in 0..NUM_LINEAR_LAYERS {
                self.state_pools[layer].copy_slot(scratch, sequence, &self.stream)?;
                self.conv_pools[layer].copy_within(
                    sequence * CONV_STATE_ELEMS,
                    scratch * CONV_STATE_ELEMS,
                    CONV_STATE_ELEMS,
                    &self.stream,
                )?;
            }
            return self.stream.synchronize();
        }

        assert!(
            !self.speculation_mixer.is_empty(),
            "фиксация без предшествующей проверки"
        );
        let capacity = self.max_batch + 1;
        let replay_mode = self.delta_state_mode_for_replay();
        let mut linear_layer = 0;
        for layer in weights.layers.iter() {
            let Mixer::Linear(mixer) = &layer.mixer else {
                continue;
            };
            let saved = &self.speculation_mixer[linear_layer];
            mixer.prepare.prepare_prefill(
                RowView::packed(saved, MIXER_FUSED_WIDTH),
                RowView::strided(saved, MIXER_A_OFFSET, MIXER_FUSED_WIDTH),
                RowView::strided(saved, MIXER_B_OFFSET, MIXER_FUSED_WIDTH),
                &mut self.conv_pools[linear_layer],
                &mut self.prefill.prepared,
                capacity,
                sequence,
                accepted,
                0,
                &self.stream,
            )?;
            // Доигрывание — тот же скан, поэтому и оно идёт по копии.
            delta_net::unpack_state_slot(
                &self.state_pools[linear_layer],
                sequence,
                &mut self.state_work[linear_layer],
                1,
                0,
                &self.stream,
            )?;
            self.work_owner[linear_layer] = Some(sequence);
            delta_net::prefill_slot(
                &mut self.state_work[linear_layer],
                &self.prefill.prepared.inputs(),
                &mut self.prefill.delta_out,
                1,
                0,
                accepted,
                0,
                replay_mode,
                &self.stream,
            )?;
            delta_net::pack_state_slot(
                &self.state_work[linear_layer],
                1,
                0,
                &mut self.state_pools[linear_layer],
                sequence,
                &self.stream,
            )?;
            linear_layer += 1;
        }
        self.stream.synchronize()
    }

    /// Доигрывание идёт рекуррентным сканом: строк там единицы, а чанковая
    /// форма на такой длине только теряет на подготовке.
    fn delta_state_mode_for_replay(&self) -> DeltaStateMode {
        match self.delta_state_mode {
            DeltaStateMode::Wy => DeltaStateMode::Bf16,
            other => other,
        }
    }

    /// Слотов состояния: по одному на последовательность плюс черновой,
    /// на котором проверяются спекулятивные токены.
    pub fn state_capacity(&self) -> usize {
        self.max_batch + 1
    }

    /// Индекс чернового слота.
    pub fn speculation_slot(&self) -> usize {
        self.max_batch
    }

    /// Сколько страниц KV покрывает контекст одной последовательности.
    pub fn max_blocks(&self) -> usize {
        self.max_blocks
    }

    pub fn kv_pool_blocks(&self) -> usize {
        self.kv_pool_blocks
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
                .all(|&block| (block as usize) < self.kv_pool_blocks),
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

        let state_capacity = self.max_batch + 1;
        let speculation_rows = self.speculation_rows;
        // Рекуррентная форма — это режим точности состояния; `Wy` в ней не
        // выражается, поэтому короткий отрезок считается bf16-проходом.
        let recurrent_mode = match self.delta_state_mode {
            DeltaStateMode::Wy => DeltaStateMode::Bf16,
            other => other,
        };
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
                    mark!("gemm.la_in");
                    project_w4a4(
                        &mixer.in_proj,
                        &prefill.normed,
                        &mut prefill.quant_hidden,
                        &mut prefill.mixer_in,
                        &mut prefill.projection_workspace,
                        live,
                        &self.stream,
                    )?;
                    if speculation_rows > 0 {
                        self.speculation_mixer[linear_layer].copy_from_device_at(
                            0,
                            &prefill.mixer_in,
                            0,
                            live * MIXER_FUSED_WIDTH,
                            &self.stream,
                        )?;
                    }
                    // The recurrence is per sequence, but the one-token rows
                    // are contiguous from row zero and share a single launch.
                    if decode_rows > 0 {
                        mark!("delta.prepare");
                        mixer.prepare.prepare_decode(
                            RowView::packed(&prefill.mixer_in, MIXER_FUSED_WIDTH),
                            RowView::strided(
                                &prefill.mixer_in,
                                MIXER_A_OFFSET,
                                MIXER_FUSED_WIDTH,
                            ),
                            RowView::strided(
                                &prefill.mixer_in,
                                MIXER_B_OFFSET,
                                MIXER_FUSED_WIDTH,
                            ),
                            &mut self.conv_pools[linear_layer],
                            &prefill.decode_slots,
                            &mut prefill.prepared,
                            state_capacity,
                            decode_rows,
                            &self.stream,
                        )?;
                        mark!("delta.scan");
                        delta_net::decode_slots_packed(
                            &mut self.state_pools[linear_layer],
                            &prefill.decode_slots,
                            &prefill.prepared.inputs(),
                            &mut prefill.delta_out,
                            decode_rows,
                            &self.stream,
                        )?;
                    }
                    for segment in &segments[decode_rows..] {
                        mark!("delta.prepare_prefill");
                        mixer.prepare.prepare_prefill(
                            RowView::packed(&prefill.mixer_in, MIXER_FUSED_WIDTH),
                            RowView::strided(
                                &prefill.mixer_in,
                                MIXER_A_OFFSET,
                                MIXER_FUSED_WIDTH,
                            ),
                            RowView::strided(
                                &prefill.mixer_in,
                                MIXER_B_OFFSET,
                                MIXER_FUSED_WIDTH,
                            ),
                            &mut self.conv_pools[linear_layer],
                            &mut prefill.prepared,
                            state_capacity,
                            segment.state_slot,
                            segment.tokens,
                            segment.row_begin,
                            &self.stream,
                        )?;
                        mark!("delta.scan_prefill");
                        // Чанковая форма окупается от пары сотен токенов: на
                        // коротком отрезке её шесть подготовительных запусков
                        // на слой дороже самого скана. Проверка спекуляции
                        // несёт k+1 строку — это как раз тот случай.
                        let wy = self.delta_prefill.is_some()
                            && segment.tokens >= MIN_WY_TOKENS;
                        // Скан идёт по расквантованной копии. Если копия уже
                        // держит этот слот — распаковки нет, и продолжение
                        // многосегментного промпта стартует с того же
                        // состояния, каким его оставил прошлый сегмент.
                        if self.work_owner[linear_layer] != Some(segment.state_slot) {
                            delta_net::unpack_state_slot(
                                &self.state_pools[linear_layer],
                                segment.state_slot,
                                &mut self.state_work[linear_layer],
                                1,
                                0,
                                &self.stream,
                            )?;
                            self.work_owner[linear_layer] = Some(segment.state_slot);
                        }
                        match &mut self.delta_prefill.as_mut().filter(|_| wy) {
                            Some(workspace) => delta_net::prefill_slot_wy(
                                &mut self.state_work[linear_layer],
                                &prefill.prepared.inputs(),
                                &mut prefill.delta_out,
                                workspace,
                                1,
                                0,
                                segment.tokens,
                                segment.row_begin,
                                &self.stream,
                            )?,
                            None => delta_net::prefill_slot(
                                &mut self.state_work[linear_layer],
                                &prefill.prepared.inputs(),
                                &mut prefill.delta_out,
                                1,
                                0,
                                segment.tokens,
                                segment.row_begin,
                                recurrent_mode,
                                &self.stream,
                            )?,
                        }
                        // Упаковка после каждого сегмента, а не в конце
                        // промпта: так 8-битный слот всегда актуален, и знать,
                        // какой сегмент последний, не нужно.
                        delta_net::pack_state_slot(
                            &self.state_work[linear_layer],
                            1,
                            0,
                            &mut self.state_pools[linear_layer],
                            segment.state_slot,
                            &self.stream,
                        )?;
                    }
                    mark!("delta.norm");
                    mixer.output_norm.forward(
                        &prefill.delta_out,
                        RowView::strided(&prefill.mixer_in, MIXER_Z_OFFSET, MIXER_FUSED_WIDTH),
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
                    let blocks = self.kv_pool_blocks;
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
                    // Строки decode идут построчным ядром, чанки префилла —
                    // ядром на тензорных ядрах. Чанк вызывается отдельно:
                    // тайл запросов не имеет права пересекать границу
                    // последовательности, у него одна таблица страниц на тайл.
                    if decode_rows > 0 {
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
                            decode_rows,
                            max_context,
                            self.kv_cache_dtype,
                            &self.stream,
                        )?;
                    }
                    mark!("attn.prefill");
                    for segment in &segments[decode_rows..] {
                        paged_attention::prefill_gated_mma(
                            &prefill.query,
                            &prefill.query_gate,
                            &self.key_caches[full_layer],
                            &self.value_caches[full_layer],
                            blocks,
                            &prefill.block_tables,
                            &prefill.context_lengths,
                            self.max_blocks,
                            &mut prefill.attention_out,
                            segment.tokens,
                            segment.row_begin,
                            self.kv_cache_dtype,
                            &self.stream,
                        )?;
                    }
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

        // Проверке черновиков нужен argmax каждой строки, а не только
        // последней: именно по ним и решается, сколько токенов принято.
        if speculation_rows > 0 {
            for row in 0..speculation_rows {
                prefill.host_last_rows[row] = row as u32;
            }
            prefill
                .last_rows
                .copy_from_slice_at(0, &prefill.host_last_rows[..speculation_rows])?;
            mark!("gather_rows");
            qwc_cuda::gather_rows_bf16(
                &prefill.normed,
                &prefill.last_rows,
                &mut prefill.last_hidden,
                speculation_rows,
                HIDDEN_SIZE,
                &self.stream,
            )?;
            mark!("lm_head");
            return weights.lm_head.logits(
                &prefill.last_hidden,
                &mut self.logits,
                speculation_rows,
                &self.stream,
            );
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
        self.mark("gemm.la_in")?;
        project_decode(
            &mixer.in_proj,
            &self.normed,
            &mut self.quant_hidden,
            &mut self.mixer_in,
            &mut self.projection_workspace,
            batch,
            &self.stream,
        )?;

        let state_capacity = self.state_capacity();
        self.mark("delta.prepare")?;
        mixer.prepare.prepare_decode(
            RowView::packed(&self.mixer_in, MIXER_FUSED_WIDTH),
            RowView::strided(&self.mixer_in, MIXER_A_OFFSET, MIXER_FUSED_WIDTH),
            RowView::strided(&self.mixer_in, MIXER_B_OFFSET, MIXER_FUSED_WIDTH),
            &mut self.conv_pools[layer],
            &self.state_slots,
            &mut self.prepared,
            state_capacity,
            batch,
            &self.stream,
        )?;
        self.mark("delta.scan")?;
        delta_net::decode_slots_packed(
            &mut self.state_pools[layer],
            &self.state_slots,
            &self.prepared.inputs(),
            &mut self.delta_out,
            batch,
            &self.stream,
        )?;
        self.mark("delta.norm")?;
        mixer.output_norm.forward(
            &self.delta_out,
            RowView::strided(&self.mixer_in, MIXER_Z_OFFSET, MIXER_FUSED_WIDTH),
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

        let blocks = self.kv_pool_blocks;
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
        let total_blocks = self.kv_pool_blocks;
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
            assert!(
                blocks
                    .iter()
                    .all(|&block| (block as usize) < self.kv_pool_blocks),
                "KV block ID exceeds executor pool"
            );
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
            pool.reset()?;
        }
        self.work_owner.fill(None);
        for pool in &mut self.conv_pools {
            pool.zero_range(0, pool.len())?;
        }
        Ok(())
    }

    pub fn reset_slot(&mut self, slot: usize) -> Result<()> {
        assert!(slot < self.max_batch);
        for (layer, pool) in self.state_pools.iter_mut().enumerate() {
            pool.reset_slot(slot)?;
            // Копия этого слота больше ничего не значит: слот получил новую
            // последовательность.
            if self.work_owner[layer] == Some(slot) {
                self.work_owner[layer] = None;
            }
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
    /// Одна строка скрытых состояний в отдельный буфер, не покидая карту.
    /// Черновой шаг MTP принимает её как вход; копия ряда в 10 КБ на фоне
    /// шага в 12 мс ничего не стоит, а вид на чужой буфер стоил бы времени
    /// жизни.
    pub fn copy_decode_hidden_row(
        &self,
        row: usize,
        destination: &mut DeviceBuffer<u16>,
    ) -> Result<()> {
        copy_hidden_row(&self.normed, row, destination, &self.stream)
    }

    pub fn copy_prefill_hidden_row(
        &self,
        row: usize,
        destination: &mut DeviceBuffer<u16>,
    ) -> Result<()> {
        copy_hidden_row(&self.prefill.normed, row, destination, &self.stream)
    }

    /// Скрытые состояния после финальной нормы — ровно тот тензор, который
    /// уходит в `lm_head`, и ровно тот, который MTP-голова ждёт на входе.
    ///
    /// Нужны замеру acceptance draft-головы: её вход — это (h_t, эмбеддинг
    /// токена t+1). Копия на хост, поэтому путь диагностический, не горячий.
    pub fn decode_hidden_to_host(&self, batch: usize) -> Result<Vec<u16>> {
        assert!(batch > 0 && batch <= self.max_batch);
        let all = self.normed.to_vec()?;
        Ok(all[..batch * HIDDEN_SIZE].to_vec())
    }

    /// То же для строк последнего prefill-шага: `rows` строк с начала арены.
    pub fn prefill_hidden_to_host(&self, rows: usize) -> Result<Vec<u16>> {
        assert!(rows * HIDDEN_SIZE <= self.prefill.normed.len());
        let all = self.prefill.normed.to_vec()?;
        Ok(all[..rows * HIDDEN_SIZE].to_vec())
    }

    pub fn argmax_to_host(&mut self, batch: usize) -> Result<Vec<u32>> {
        self.sampler.sample(&self.logits, batch, &self.stream)?;
        self.sampler.to_host(batch)
    }
}

fn copy_hidden_row(
    source: &DeviceBuffer<u16>,
    row: usize,
    destination: &mut DeviceBuffer<u16>,
    stream: &Stream,
) -> Result<()> {
    assert!((row + 1) * HIDDEN_SIZE <= source.len());
    assert!(destination.len() >= HIDDEN_SIZE);
    destination.copy_from_device_at(0, source, row * HIDDEN_SIZE, HIDDEN_SIZE, stream)
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
/// Проекция строк шага. На широком шаге это W4A4 на тензорных ядрах, на
/// узком — тот же выбор, что в decode.
///
/// Узкий шаг бывает не только в decode: проверка спекулятивных токенов несёт
/// k+1 строку одной последовательности, и на такой ширине W4A4 читает веса
/// вдвое медленнее W4A16 (530 против 1379 ГБ/с по `nvfp4bench`). Пока выбор
/// был прибит к W4A4, шаг проверки стоил дороже целого шага decode.
fn project_w4a4(
    projection: &Projection,
    input: &DeviceBuffer<u16>,
    quantized: &mut QuantizedActivation,
    output: &mut DeviceBuffer<u16>,
    workspace: &mut W4A4Workspace,
    rows: usize,
    stream: &Stream,
) -> Result<()> {
    if rows <= nvfp4::MAX_W4A4_BATCH {
        return project_decode(projection, input, quantized, output, workspace, rows, stream);
    }
    quantized.quantize_bf16_rows(input, projection.input_global_scale, rows.max(3), stream)?;
    projection
        .linear
        .forward_w4a4_quantized(quantized, output, workspace, stream)
}

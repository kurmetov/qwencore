//! Подъём весов чекпоинта в VRAM.
//!
//! Загрузчик не «подстраивается» под чекпоинт: формы и типы берутся из
//! `qwc_model::names` и `qwc_core::arch`, а расхождение — ошибка загрузки.
//! Страницы mmap вытесняются после каждого слоя, поэтому host-RAM держится
//! на уровне одного слоя, а не всего чекпоинта.

use qwc_core::arch::*;
use qwc_cuda::attention_prepare::AttentionPreprocessor;
use qwc_cuda::delta_net::{DeltaOutputNorm, DeltaPreprocessor};
use qwc_cuda::nvfp4::Linear;
use qwc_cuda::rmsnorm::RmsNorm;
use qwc_cuda::vocab::{Bf16Vocab, Fp8Vocab};
use qwc_cuda::{DeviceBuffer, Stream};
use qwc_model::Checkpoint;
use qwc_model::names;
use qwc_model::safetensors::StDtype;
use std::collections::HashMap;

/// Квантованная проекция вместе с масштабом квантизации активаций: он нужен
/// только W4A4-пути, но принадлежит той же паре весов.
pub struct Projection {
    pub linear: Linear,
    pub input_global_scale: f32,
    pub decode_linear_mode: DecodeLinearMode,
}

impl Projection {
    pub fn resident_bytes(&self) -> usize {
        self.linear.resident_bytes()
    }
}

pub struct Mlp {
    pub gate: Projection,
    pub up: Projection,
    pub down: Projection,
}

/// Full-attention слой. `q_proj` выдаёт [q; выходной гейт] одной проекцией.
pub struct FullAttention {
    pub q: Projection,
    pub k: Projection,
    pub v: Projection,
    pub o: Projection,
    /// QK-норма на голову, частичный RoPE и запись FP8-страниц KV.
    pub prepare: AttentionPreprocessor,
}

/// Ширина слитой входной проекции миксера и смещения её частей.
///
/// qkv, z, a и b читают один и тот же вход, а их global scales совпадают во
/// всех слоях чекпоинта — склейка точна, не приближённа. Порознь они дают
/// четыре запуска на слой, из них два по N=48: при тайле 128 это один блок на
/// 170 SM, и две крошечные проекции стоили дороже, чем qkv и z вместе.
pub const MIXER_GATE_WIDTH: usize = LA_NUM_V_HEADS;
pub const MIXER_Z_OFFSET: usize = LA_CONV_CHANNELS;
pub const MIXER_A_OFFSET: usize = MIXER_Z_OFFSET + LA_V_PROJ_DIM;
pub const MIXER_B_OFFSET: usize = MIXER_A_OFFSET + MIXER_GATE_WIDTH;
pub const MIXER_FUSED_WIDTH: usize = MIXER_B_OFFSET + MIXER_GATE_WIDTH;

/// Gated DeltaNet слой.
pub struct LinearAttention {
    /// Слитая [MIXER_FUSED_WIDTH, HIDDEN_SIZE]: qkv, z, a, b подряд.
    pub in_proj: Projection,
    pub out: Projection,
    /// conv1d + SiLU, L2-норма q/k, alpha/beta.
    pub prepare: DeltaPreprocessor,
    /// RMSNormGated по v_head_dim: вес прямой, единица не добавляется.
    pub output_norm: DeltaOutputNorm,
}

pub enum Mixer {
    Full(FullAttention),
    Linear(LinearAttention),
}

pub struct LayerWeights {
    pub input_norm: RmsNorm,
    pub post_attention_norm: RmsNorm,
    pub mixer: Mixer,
    pub mlp: Mlp,
}

impl LayerWeights {
    pub fn kind(&self) -> LayerKind {
        match self.mixer {
            Mixer::Full(_) => LayerKind::FullAttention,
            Mixer::Linear(_) => LayerKind::LinearAttention,
        }
    }
}

/// Веса всех 64 слоёв, финальная норма и словарные матрицы в FP8.
///
/// MTP-голова сюда не входит: она нужна только спекулятивному декодированию.
pub struct ModelWeights {
    pub layers: Vec<LayerWeights>,
    pub final_norm: RmsNorm,
    /// Таблица эмбеддингов: BF16 в чекпоинте, E4M3 со шкалой на строку у нас.
    pub embed: Embedding,
    pub lm_head: LmHead,
    decode_linear_mode: DecodeLinearMode,
    stats: LoadStats,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum LmHeadDtype {
    #[default]
    Fp8,
    Bf16,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DecodeLinearMode {
    #[default]
    Auto,
    W4A4,
}

impl DecodeLinearMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::W4A4 => "w4a4",
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum EmbeddingDtype {
    #[default]
    Fp8,
    Bf16,
}

impl EmbeddingDtype {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fp8 => "fp8",
            Self::Bf16 => "bf16",
        }
    }
}

impl LmHeadDtype {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fp8 => "fp8",
            Self::Bf16 => "bf16",
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LoadConfig {
    pub embedding: EmbeddingDtype,
    pub lm_head: LmHeadDtype,
    pub decode_linear: DecodeLinearMode,
}

pub enum Embedding {
    Fp8(Fp8Vocab),
    Bf16(Bf16Vocab),
}

impl Embedding {
    pub fn gather(
        &self,
        token_ids: &DeviceBuffer<u32>,
        output: &mut DeviceBuffer<u16>,
        stream: &Stream,
    ) -> qwc_cuda::Result<()> {
        match self {
            Self::Fp8(table) => table.gather(token_ids, output, stream),
            Self::Bf16(table) => table.gather(token_ids, output, stream),
        }
    }

    pub fn gather_rows(
        &self,
        token_ids: &DeviceBuffer<u32>,
        output: &mut DeviceBuffer<u16>,
        batch: usize,
        stream: &Stream,
    ) -> qwc_cuda::Result<()> {
        match self {
            Self::Fp8(table) => table.gather_rows(token_ids, output, batch, stream),
            Self::Bf16(table) => table.gather_rows(token_ids, output, batch, stream),
        }
    }

    pub fn dtype(&self) -> EmbeddingDtype {
        match self {
            Self::Fp8(_) => EmbeddingDtype::Fp8,
            Self::Bf16(_) => EmbeddingDtype::Bf16,
        }
    }

    pub fn resident_bytes(&self) -> usize {
        match self {
            Self::Fp8(table) => table.resident_bytes(),
            Self::Bf16(table) => table.resident_bytes(),
        }
    }
}

pub enum LmHead {
    Fp8(Fp8Vocab),
    Bf16(Bf16Vocab),
}

impl LmHead {
    pub fn logits(
        &self,
        hidden: &DeviceBuffer<u16>,
        logits: &mut DeviceBuffer<f32>,
        batch: usize,
        stream: &Stream,
    ) -> qwc_cuda::Result<()> {
        match self {
            Self::Fp8(head) => head.logits(hidden, logits, batch, stream),
            Self::Bf16(head) => head.logits(hidden, logits, batch, stream),
        }
    }

    pub fn logits_row_to(
        &self,
        hidden: &DeviceBuffer<u16>,
        hidden_row: usize,
        logits: &mut DeviceBuffer<f32>,
        logits_row: usize,
        stream: &Stream,
    ) -> qwc_cuda::Result<()> {
        match self {
            Self::Fp8(head) => head.logits_row_to(hidden, hidden_row, logits, logits_row, stream),
            Self::Bf16(head) => head.logits_row_to(hidden, hidden_row, logits, logits_row, stream),
        }
    }

    pub fn dtype(&self) -> LmHeadDtype {
        match self {
            Self::Fp8(_) => LmHeadDtype::Fp8,
            Self::Bf16(_) => LmHeadDtype::Bf16,
        }
    }

    pub fn resident_bytes(&self) -> usize {
        match self {
            Self::Fp8(head) => head.resident_bytes(),
            Self::Bf16(head) => head.resident_bytes(),
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct LoadStats {
    pub layers: usize,
    pub projections: usize,
    /// Упакованные веса вместе с переставленной scale-матрицей.
    pub quantized_bytes: usize,
    /// Нормы, conv1d, A_log, dt_bias.
    pub plain_bytes: usize,
    /// `embed_tokens` и `lm_head` после перевода в FP8.
    pub vocab_bytes: usize,
}

impl LoadStats {
    pub fn resident_bytes(&self) -> usize {
        self.quantized_bytes + self.plain_bytes + self.vocab_bytes
    }
}

impl ModelWeights {
    pub fn load(checkpoint: &Checkpoint) -> Result<Self, LoadError> {
        Self::load_with_config(checkpoint, LoadConfig::default())
    }

    pub fn load_with_config(
        checkpoint: &Checkpoint,
        config: LoadConfig,
    ) -> Result<Self, LoadError> {
        Self::load_with_progress_config(checkpoint, config, |_, _| {})
    }

    /// `on_layer` вызывается после подъёма каждого слоя: загрузка идёт
    /// десятки секунд, и молчащий процесс на 14 GB неотличим от зависшего.
    pub fn load_with_progress(
        checkpoint: &Checkpoint,
        on_layer: impl FnMut(usize, LoadStats),
    ) -> Result<Self, LoadError> {
        Self::load_with_progress_config(checkpoint, LoadConfig::default(), on_layer)
    }

    pub fn load_with_progress_config(
        checkpoint: &Checkpoint,
        config: LoadConfig,
        mut on_layer: impl FnMut(usize, LoadStats),
    ) -> Result<Self, LoadError> {
        let mut stats = LoadStats::default();
        let mut layers = Vec::with_capacity(NUM_LAYERS);
        for index in 0..NUM_LAYERS {
            layers.push(load_layer(
                checkpoint,
                index,
                config.decode_linear,
                &mut stats,
            )?);
            stats.layers += 1;
            // Страницы уже скопированы в VRAM: держать их в working set незачем.
            checkpoint.evict_file_pages()?;
            on_layer(index, stats);
        }

        let final_norm = load_norm(checkpoint, names::FINAL_NORM, HIDDEN_SIZE, &mut stats)?;

        // Словарные матрицы идут последними: они самые большие на диске и
        // после них вытеснять уже нечего.
        let stream = Stream::new()?;
        let embed = match config.embedding {
            EmbeddingDtype::Fp8 => {
                Embedding::Fp8(load_vocab(checkpoint, names::EMBED, &stream, &mut stats)?)
            }
            EmbeddingDtype::Bf16 => {
                Embedding::Bf16(load_bf16_vocab(checkpoint, names::EMBED, &mut stats)?)
            }
        };
        let lm_head = match config.lm_head {
            LmHeadDtype::Fp8 => {
                LmHead::Fp8(load_vocab(checkpoint, names::LM_HEAD, &stream, &mut stats)?)
            }
            LmHeadDtype::Bf16 => {
                LmHead::Bf16(load_bf16_vocab(checkpoint, names::LM_HEAD, &mut stats)?)
            }
        };

        Ok(Self {
            layers,
            final_norm,
            embed,
            lm_head,
            decode_linear_mode: config.decode_linear,
            stats,
        })
    }

    pub fn stats(&self) -> LoadStats {
        self.stats
    }

    pub fn decode_linear_mode(&self) -> DecodeLinearMode {
        self.decode_linear_mode
    }
}

fn load_layer(
    checkpoint: &Checkpoint,
    index: usize,
    decode_linear_mode: DecodeLinearMode,
    stats: &mut LoadStats,
) -> Result<LayerWeights, LoadError> {
    let prefix = names::layer(index);
    let mut projections = load_projections(checkpoint, index, decode_linear_mode, stats)?;
    let mut plain = load_plain(checkpoint, index, stats)?;
    let mut projection = |name: &str| -> Result<Projection, LoadError> {
        projections
            .remove(name)
            .ok_or_else(|| LoadError::Checkpoint(format!("нет проекции {prefix}.{name}")))
    };

    let mlp = Mlp {
        gate: projection("gate_proj")?,
        up: projection("up_proj")?,
        down: projection("down_proj")?,
    };

    let mixer = match layer_kind(index) {
        LayerKind::FullAttention => Mixer::Full(FullAttention {
            q: projection("q_proj")?,
            k: projection("k_proj")?,
            v: projection("v_proj")?,
            o: projection("o_proj")?,
            prepare: AttentionPreprocessor::from_host(
                &take(&mut plain, index, "self_attn.q_norm.weight")?,
                &take(&mut plain, index, "self_attn.k_norm.weight")?,
                RMS_NORM_EPS,
            )?,
        }),
        LayerKind::LinearAttention => Mixer::Linear(LinearAttention {
            in_proj: projection("in_proj")?,
            out: projection("out_proj")?,
            prepare: DeltaPreprocessor::from_host(
                &take(&mut plain, index, "linear_attn.conv1d.weight")?,
                &take(&mut plain, index, "linear_attn.A_log")?,
                &take(&mut plain, index, "linear_attn.dt_bias")?,
            )?,
            output_norm: DeltaOutputNorm::from_host(
                &take(&mut plain, index, "linear_attn.norm.weight")?,
                RMS_NORM_EPS,
            )?,
        }),
    };

    let input_norm = RmsNorm::from_host(
        &take(&mut plain, index, "input_layernorm.weight")?,
        RMS_NORM_EPS,
    )?;
    let post_attention_norm = RmsNorm::from_host(
        &take(&mut plain, index, "post_attention_layernorm.weight")?,
        RMS_NORM_EPS,
    )?;
    // Ни один описанный архитектурой тензор слоя не должен остаться неиспользованным.
    if let Some(unused) = plain.keys().next() {
        return Err(LoadError::Checkpoint(format!(
            "тензор {unused} слоя {index} не разобран загрузчиком"
        )));
    }

    Ok(LayerWeights {
        input_norm,
        post_attention_norm,
        mixer,
        mlp,
    })
}

/// Все квантованные проекции слоя по короткому имени: `gate_proj`, `q_proj`,
/// `in_proj_qkv` и так далее. Формы приходят из `names`, а не из чекпоинта.
fn load_projections(
    checkpoint: &Checkpoint,
    index: usize,
    decode_linear_mode: DecodeLinearMode,
    stats: &mut LoadStats,
) -> Result<HashMap<String, Projection>, LoadError> {
    let mut out = HashMap::new();
    let mut fused: Vec<(String, qwc_model::checkpoint::QuantLinear<'_>)> = Vec::new();
    for (prefix, out_features, in_features) in names::quant_linears(index) {
        let quant = checkpoint
            .quant_linear(&prefix, out_features, in_features)
            .map_err(LoadError::Checkpoint)?;
        let short = prefix
            .rsplit('.')
            .next()
            .expect("имя проекции не пустое")
            .to_string();
        if MIXER_PARTS.contains(&short.as_str()) {
            fused.push((short, quant));
            continue;
        }
        let linear = Linear::from_host(
            quant.packed,
            quant.block_scales,
            quant.weight_global_scale,
            out_features,
            in_features,
        )?;
        stats.quantized_bytes += linear.resident_bytes();
        stats.projections += 1;
        out.insert(
            short,
            Projection {
                linear,
                input_global_scale: quant.input_global_scale,
                decode_linear_mode,
            },
        );
    }
    if !fused.is_empty() {
        let projection = fuse_mixer_input(&fused, decode_linear_mode, stats)?;
        out.insert("in_proj".to_string(), projection);
    }
    Ok(out)
}

/// Части слитой входной проекции — в том порядке, в каком они лежат в строке.
const MIXER_PARTS: [&str; 4] = ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"];

/// Склеивает qkv, z, a и b в одну матрицу.
///
/// Обе шкалы обязаны совпадать: alpha GEMM'а — один скаляр на запуск, и
/// разные global scales склеить было бы нечем. В чекпоинте они совпадают во
/// всех 48 слоях, поэтому расхождение здесь — это ошибка загрузки, а не
/// случай, который нужно поддерживать.
fn fuse_mixer_input(
    parts: &[(String, qwc_model::checkpoint::QuantLinear<'_>)],
    decode_linear_mode: DecodeLinearMode,
    stats: &mut LoadStats,
) -> Result<Projection, LoadError> {
    if parts.len() != MIXER_PARTS.len() {
        return Err(LoadError::Checkpoint(format!(
            "слитая проекция миксера: частей {}, ожидалось {}",
            parts.len(),
            MIXER_PARTS.len()
        )));
    }
    let mut packed = Vec::new();
    let mut scales = Vec::new();
    let mut out_features = 0usize;
    let in_features = parts[0].1.in_features;
    let weight_global_scale = parts[0].1.weight_global_scale;
    let input_global_scale = parts[0].1.input_global_scale;
    for name in MIXER_PARTS {
        let quant = &parts
            .iter()
            .find(|(short, _)| short == name)
            .ok_or_else(|| LoadError::Checkpoint(format!("нет {name}")))?
            .1;
        if quant.weight_global_scale != weight_global_scale
            || quant.input_global_scale != input_global_scale
        {
            return Err(LoadError::Checkpoint(format!(
                "{name}: global scale расходится с остальными частями миксера"
            )));
        }
        if quant.in_features != in_features {
            return Err(LoadError::Checkpoint(format!(
                "{name}: K={} против {in_features}",
                quant.in_features
            )));
        }
        packed.extend_from_slice(quant.packed);
        scales.extend_from_slice(quant.block_scales);
        out_features += quant.out_features;
    }
    if out_features != MIXER_FUSED_WIDTH {
        return Err(LoadError::Checkpoint(format!(
            "слитая проекция миксера: N={out_features}, ожидалось {MIXER_FUSED_WIDTH}"
        )));
    }
    let linear = Linear::from_host(
        &packed,
        &scales,
        weight_global_scale,
        out_features,
        in_features,
    )?;
    stats.quantized_bytes += linear.resident_bytes();
    stats.projections += 1;
    Ok(Projection {
        linear,
        input_global_scale,
        decode_linear_mode,
    })
}

/// Неквантованные тензоры слоя по суффиксу имени. Список и формы приходят
/// из `names`, поэтому лишний или отсутствующий тензор виден сразу здесь.
fn load_plain(
    checkpoint: &Checkpoint,
    index: usize,
    stats: &mut LoadStats,
) -> Result<HashMap<String, Vec<u16>>, LoadError> {
    let layer = names::layer(index);
    let mut out = HashMap::new();
    for (name, shape) in names::plain_tensors(index) {
        let suffix = name
            .strip_prefix(&format!("{layer}."))
            .ok_or_else(|| LoadError::Checkpoint(format!("{name} не принадлежит слою {index}")))?
            .to_string();
        let data = bf16_tensor(checkpoint, &name, shape.iter().product())?;
        stats.plain_bytes += data.len() * 2;
        out.insert(suffix, data);
    }
    Ok(out)
}

fn take(
    plain: &mut HashMap<String, Vec<u16>>,
    index: usize,
    suffix: &str,
) -> Result<Vec<u16>, LoadError> {
    plain
        .remove(suffix)
        .ok_or_else(|| LoadError::Checkpoint(format!("нет {suffix} в слое {index}")))
}

fn load_norm(
    checkpoint: &Checkpoint,
    name: &str,
    elems: usize,
    stats: &mut LoadStats,
) -> Result<RmsNorm, LoadError> {
    let weight = bf16_tensor(checkpoint, name, elems)?;
    stats.plain_bytes += weight.len() * 2;
    Ok(RmsNorm::from_host(&weight, RMS_NORM_EPS)?)
}

/// `[vocab, hidden]` BF16 из чекпоинта в FP8 со шкалой на строку.
/// Данные идут кусками через staging-буфер: держать 2.54 GB BF16 в VRAM
/// ради конверсии незачем, а срез mmap отдаётся кернелу как есть.
fn load_vocab(
    checkpoint: &Checkpoint,
    name: &str,
    stream: &Stream,
    stats: &mut LoadStats,
) -> Result<Fp8Vocab, LoadError> {
    /// 84 MB staging: достаточно, чтобы запуск кернела не тонул в накладных
    /// расходах, и достаточно мало, чтобы не мешать весам.
    const CHUNK_ROWS: usize = 8192;

    let info = checkpoint
        .info(name)
        .ok_or_else(|| LoadError::Checkpoint(format!("нет {name}")))?;
    let want = vec![VOCAB_SIZE, HIDDEN_SIZE];
    if info.shape != want {
        return Err(LoadError::Checkpoint(format!(
            "{name}: форма {:?}, ожидалась {want:?}",
            info.shape
        )));
    }
    if info.dtype != StDtype::Bf16 {
        return Err(LoadError::Checkpoint(format!(
            "{name}: тип {}, ожидался bf16",
            info.dtype.name()
        )));
    }

    let raw = checkpoint
        .bytes(name)
        .ok_or_else(|| LoadError::Checkpoint(format!("нет данных {name}")))?;
    let row_bytes = HIDDEN_SIZE * 2;
    let mut vocab = Fp8Vocab::zeroed(VOCAB_SIZE, HIDDEN_SIZE)?;
    let mut staging = DeviceBuffer::<u8>::zeroed(CHUNK_ROWS * row_bytes)?;
    let mut first_row = 0;
    while first_row < VOCAB_SIZE {
        let rows = CHUNK_ROWS.min(VOCAB_SIZE - first_row);
        let start = first_row * row_bytes;
        vocab.quantize_rows(
            first_row,
            &raw[start..start + rows * row_bytes],
            &mut staging,
            stream,
        )?;
        // Вытеснение по куску, а не по тензору: иначе пик host-RAM вырастает
        // на все 2.54 GB словарной матрицы.
        checkpoint.evict_file_pages()?;
        first_row += rows;
    }
    stats.vocab_bytes += vocab.resident_bytes();
    Ok(vocab)
}

/// Загружает `lm_head` без квантизации для чистого A/B-замера.
fn load_bf16_vocab(
    checkpoint: &Checkpoint,
    name: &str,
    stats: &mut LoadStats,
) -> Result<Bf16Vocab, LoadError> {
    const CHUNK_ROWS: usize = 8192;

    let info = checkpoint
        .info(name)
        .ok_or_else(|| LoadError::Checkpoint(format!("нет {name}")))?;
    let want = vec![VOCAB_SIZE, HIDDEN_SIZE];
    if info.shape != want {
        return Err(LoadError::Checkpoint(format!(
            "{name}: форма {:?}, ожидалась {want:?}",
            info.shape
        )));
    }
    if info.dtype != StDtype::Bf16 {
        return Err(LoadError::Checkpoint(format!(
            "{name}: тип {}, ожидался bf16",
            info.dtype.name()
        )));
    }

    let raw = checkpoint
        .bytes(name)
        .ok_or_else(|| LoadError::Checkpoint(format!("нет данных {name}")))?;
    let row_bytes = HIDDEN_SIZE * 2;
    let mut table = Bf16Vocab::zeroed(VOCAB_SIZE, HIDDEN_SIZE)?;
    let mut first_row = 0;
    while first_row < VOCAB_SIZE {
        let rows = CHUNK_ROWS.min(VOCAB_SIZE - first_row);
        let start = first_row * row_bytes;
        table.copy_rows(first_row, &raw[start..start + rows * row_bytes])?;
        checkpoint.evict_file_pages()?;
        first_row += rows;
    }
    stats.vocab_bytes += table.resident_bytes();
    Ok(table)
}

/// Помощники для MTP-головы: она грузится отдельно от основного стека, но из
/// того же чекпоинта и теми же проверками форм.
pub(crate) fn bf16_host_tensor(
    checkpoint: &Checkpoint,
    name: &str,
    elems: usize,
) -> Result<Vec<u16>, LoadError> {
    bf16_tensor(checkpoint, name, elems)
}

pub(crate) fn bf16_device_tensor(
    checkpoint: &Checkpoint,
    name: &str,
    elems: usize,
) -> Result<DeviceBuffer<u16>, LoadError> {
    let host = bf16_tensor(checkpoint, name, elems)?;
    let buffer = DeviceBuffer::from_slice(&host)?;
    checkpoint.evict_file_pages()?;
    Ok(buffer)
}

pub(crate) fn load_norm_named(
    checkpoint: &Checkpoint,
    name: &str,
    elems: usize,
) -> Result<RmsNorm, LoadError> {
    let weight = bf16_tensor(checkpoint, name, elems)?;
    Ok(RmsNorm::from_host(&weight, RMS_NORM_EPS)?)
}

/// Копия BF16-тензора на хост. Срез mmap не выровнен под `u16`, поэтому
/// байты собираются явно, а не приводятся указателем.
fn bf16_tensor(checkpoint: &Checkpoint, name: &str, elems: usize) -> Result<Vec<u16>, LoadError> {
    let info = checkpoint
        .info(name)
        .ok_or_else(|| LoadError::Checkpoint(format!("нет {name}")))?;
    if info.dtype != StDtype::Bf16 {
        return Err(LoadError::Checkpoint(format!(
            "{name}: тип {}, ожидался bf16",
            info.dtype.name()
        )));
    }
    if info.elems() != elems {
        return Err(LoadError::Checkpoint(format!(
            "{name}: {} элементов, ожидалось {elems}",
            info.elems()
        )));
    }
    let raw = checkpoint
        .bytes(name)
        .ok_or_else(|| LoadError::Checkpoint(format!("нет данных {name}")))?;
    Ok(raw
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u16::from_le_bytes(*pair))
        .collect())
}

#[derive(Debug)]
pub enum LoadError {
    /// Чекпоинт разошёлся с архитектурой.
    Checkpoint(String),
    /// Отказ CUDA, в том числе упор в потолок VRAM процесса.
    Cuda(qwc_cuda::CudaError),
    Io(std::io::Error),
}

impl From<qwc_cuda::CudaError> for LoadError {
    fn from(e: qwc_cuda::CudaError) -> Self {
        Self::Cuda(e)
    }
}

impl From<std::io::Error> for LoadError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Checkpoint(m) => write!(f, "чекпоинт: {m}"),
            Self::Cuda(e) => write!(f, "{e}"),
            Self::Io(e) => write!(f, "ввод-вывод: {e}"),
        }
    }
}

impl std::error::Error for LoadError {}

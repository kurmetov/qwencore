//! Loading checkpoint weights into VRAM.
//!
//! The loader doesn't "adapt" to the checkpoint: shapes and types come from
//! `qwc_model::names` and `qwc_core::arch`, and a mismatch is a load error.
//! mmap pages are evicted after every layer, so host RAM stays
//! at the level of one layer rather than the whole checkpoint.

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

/// A quantized projection together with the activation quantization scale: it's needed
/// only by the W4A4 path, but belongs to the same weight pair.
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

/// Full-attention layer. `q_proj` produces [q; output gate] in one projection.
pub struct FullAttention {
    pub q: Projection,
    pub k: Projection,
    pub v: Projection,
    pub o: Projection,
    /// Per-head QK norm, partial RoPE and writing FP8 KV pages.
    pub prepare: AttentionPreprocessor,
}

/// Width of the fused mixer input projection and offsets of its parts.
///
/// qkv, z, a and b read the same input, and their global scales match in
/// all layers of the checkpoint — the concatenation is exact, not approximate. Separately they give
/// four launches per layer, two of them with N=48: with a 128 tile that's one block on
/// 170 SMs, and the two tiny projections cost more than qkv and z together.
pub const MIXER_GATE_WIDTH: usize = LA_NUM_V_HEADS;
pub const MIXER_Z_OFFSET: usize = LA_CONV_CHANNELS;
pub const MIXER_A_OFFSET: usize = MIXER_Z_OFFSET + LA_V_PROJ_DIM;
pub const MIXER_B_OFFSET: usize = MIXER_A_OFFSET + MIXER_GATE_WIDTH;
pub const MIXER_FUSED_WIDTH: usize = MIXER_B_OFFSET + MIXER_GATE_WIDTH;

/// Gated DeltaNet layer.
pub struct LinearAttention {
    /// Fused [MIXER_FUSED_WIDTH, HIDDEN_SIZE]: qkv, z, a, b in a row.
    pub in_proj: Projection,
    pub out: Projection,
    /// conv1d + SiLU, L2 norm of q/k, alpha/beta.
    pub prepare: DeltaPreprocessor,
    /// RMSNormGated over v_head_dim: the weight is direct, no one is added.
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

/// Weights of all 64 layers, the final norm and the vocab matrices in FP8.
///
/// The MTP head isn't included: it's only needed for speculative decoding.
pub struct ModelWeights {
    pub layers: Vec<LayerWeights>,
    pub final_norm: RmsNorm,
    /// Embedding table: BF16 in the checkpoint, E4M3 with a per-row scale here.
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

    /// Logits only for the rows in `row_ids`. Only the FP8 head has this:
    /// the BF16 path is diagnostic, and a shortlist has no point there.
    pub fn logits_subset(
        &self,
        hidden: &DeviceBuffer<u16>,
        row_ids: &DeviceBuffer<u32>,
        logits: &mut DeviceBuffer<f32>,
        count: usize,
        stream: &Stream,
    ) -> qwc_cuda::Result<bool> {
        match self {
            Self::Fp8(head) => head
                .logits_subset(hidden, row_ids, logits, count, stream)
                .map(|()| true),
            Self::Bf16(_) => Ok(false),
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
    /// Packed weights together with the permuted scale matrix.
    pub quantized_bytes: usize,
    /// Norms, conv1d, A_log, dt_bias.
    pub plain_bytes: usize,
    /// `embed_tokens` and `lm_head` after conversion to FP8.
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

    /// `on_layer` is called after each layer is loaded: loading takes
    /// tens of seconds, and a silent 14 GB process is indistinguishable from a hung one.
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
            // The pages are already copied to VRAM: no reason to keep them in the working set.
            checkpoint.evict_file_pages()?;
            on_layer(index, stats);
        }

        let final_norm = load_norm(checkpoint, names::FINAL_NORM, HIDDEN_SIZE, &mut stats)?;

        // The vocab matrices go last: they're the largest on disk and
        // after them there's nothing left to evict.
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
            .ok_or_else(|| LoadError::Checkpoint(format!("no projection {prefix}.{name}")))
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
    // No layer tensor described by the architecture may remain unused.
    if let Some(unused) = plain.keys().next() {
        return Err(LoadError::Checkpoint(format!(
            "tensor {unused} of layer {index} not consumed by the loader"
        )));
    }

    Ok(LayerWeights {
        input_norm,
        post_attention_norm,
        mixer,
        mlp,
    })
}

/// All quantized projections of a layer by short name: `gate_proj`, `q_proj`,
/// `in_proj_qkv` and so on. Shapes come from `names`, not from the checkpoint.
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
            .expect("projection name is not empty")
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

/// Parts of the fused input projection — in the order they lie in a row.
const MIXER_PARTS: [&str; 4] = ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"];

/// Concatenates qkv, z, a and b into one matrix.
///
/// Both scales must match: the GEMM alpha is one scalar per launch, and
/// there would be no way to merge different global scales. In the checkpoint they match in
/// all 48 layers, so a mismatch here is a load error, not
/// a case that needs supporting.
fn fuse_mixer_input(
    parts: &[(String, qwc_model::checkpoint::QuantLinear<'_>)],
    decode_linear_mode: DecodeLinearMode,
    stats: &mut LoadStats,
) -> Result<Projection, LoadError> {
    if parts.len() != MIXER_PARTS.len() {
        return Err(LoadError::Checkpoint(format!(
            "fused mixer projection: {} parts, expected {}",
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
            .ok_or_else(|| LoadError::Checkpoint(format!("missing {name}")))?
            .1;
        if quant.weight_global_scale != weight_global_scale
            || quant.input_global_scale != input_global_scale
        {
            return Err(LoadError::Checkpoint(format!(
                "{name}: global scale differs from the other mixer parts"
            )));
        }
        if quant.in_features != in_features {
            return Err(LoadError::Checkpoint(format!(
                "{name}: K={} vs {in_features}",
                quant.in_features
            )));
        }
        packed.extend_from_slice(quant.packed);
        scales.extend_from_slice(quant.block_scales);
        out_features += quant.out_features;
    }
    if out_features != MIXER_FUSED_WIDTH {
        return Err(LoadError::Checkpoint(format!(
            "fused mixer projection: N={out_features}, expected {MIXER_FUSED_WIDTH}"
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

/// Non-quantized layer tensors by name suffix. The list and shapes come
/// from `names`, so an extra or missing tensor is visible right here.
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
            .ok_or_else(|| LoadError::Checkpoint(format!("{name} doesn't belong to layer {index}")))?
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
        .ok_or_else(|| LoadError::Checkpoint(format!("missing {suffix} in layer {index}")))
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

/// `[vocab, hidden]` BF16 from the checkpoint to FP8 with a per-row scale.
/// Data goes in chunks through a staging buffer: there's no reason to hold 2.54 GB of BF16 in VRAM
/// for the conversion, and the mmap slice is handed to the kernel as is.
fn load_vocab(
    checkpoint: &Checkpoint,
    name: &str,
    stream: &Stream,
    stats: &mut LoadStats,
) -> Result<Fp8Vocab, LoadError> {
    /// 84 MB staging: enough for the kernel launch not to drown in overhead,
    /// and small enough not to get in the weights' way.
    const CHUNK_ROWS: usize = 8192;

    let info = checkpoint
        .info(name)
        .ok_or_else(|| LoadError::Checkpoint(format!("missing {name}")))?;
    let want = vec![VOCAB_SIZE, HIDDEN_SIZE];
    if info.shape != want {
        return Err(LoadError::Checkpoint(format!(
            "{name}: shape {:?}, expected {want:?}",
            info.shape
        )));
    }
    if info.dtype != StDtype::Bf16 {
        return Err(LoadError::Checkpoint(format!(
            "{name}: type {}, expected bf16",
            info.dtype.name()
        )));
    }

    let raw = checkpoint
        .bytes(name)
        .ok_or_else(|| LoadError::Checkpoint(format!("no data for {name}")))?;
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
        // Evict per chunk, not per tensor: otherwise peak host RAM grows
        // by the whole 2.54 GB of the vocab matrix.
        checkpoint.evict_file_pages()?;
        first_row += rows;
    }
    stats.vocab_bytes += vocab.resident_bytes();
    Ok(vocab)
}

/// Loads `lm_head` without quantization for a clean A/B measurement.
fn load_bf16_vocab(
    checkpoint: &Checkpoint,
    name: &str,
    stats: &mut LoadStats,
) -> Result<Bf16Vocab, LoadError> {
    const CHUNK_ROWS: usize = 8192;

    let info = checkpoint
        .info(name)
        .ok_or_else(|| LoadError::Checkpoint(format!("missing {name}")))?;
    let want = vec![VOCAB_SIZE, HIDDEN_SIZE];
    if info.shape != want {
        return Err(LoadError::Checkpoint(format!(
            "{name}: shape {:?}, expected {want:?}",
            info.shape
        )));
    }
    if info.dtype != StDtype::Bf16 {
        return Err(LoadError::Checkpoint(format!(
            "{name}: type {}, expected bf16",
            info.dtype.name()
        )));
    }

    let raw = checkpoint
        .bytes(name)
        .ok_or_else(|| LoadError::Checkpoint(format!("no data for {name}")))?;
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

/// Helpers for the MTP head: it's loaded separately from the main stack, but from
/// the same checkpoint and with the same shape checks.
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

/// A copy of a BF16 tensor to host. The mmap slice isn't aligned for `u16`, so
/// the bytes are assembled explicitly rather than cast through a pointer.
fn bf16_tensor(checkpoint: &Checkpoint, name: &str, elems: usize) -> Result<Vec<u16>, LoadError> {
    let info = checkpoint
        .info(name)
        .ok_or_else(|| LoadError::Checkpoint(format!("missing {name}")))?;
    if info.dtype != StDtype::Bf16 {
        return Err(LoadError::Checkpoint(format!(
            "{name}: type {}, expected bf16",
            info.dtype.name()
        )));
    }
    if info.elems() != elems {
        return Err(LoadError::Checkpoint(format!(
            "{name}: {} elements, expected {elems}",
            info.elems()
        )));
    }
    let raw = checkpoint
        .bytes(name)
        .ok_or_else(|| LoadError::Checkpoint(format!("no data for {name}")))?;
    Ok(raw
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u16::from_le_bytes(*pair))
        .collect())
}

#[derive(Debug)]
pub enum LoadError {
    /// The checkpoint diverged from the architecture.
    Checkpoint(String),
    /// A CUDA failure, including hitting the process's VRAM cap.
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
            Self::Checkpoint(m) => write!(f, "checkpoint: {m}"),
            Self::Cuda(e) => write!(f, "{e}"),
            Self::Io(e) => write!(f, "I/O: {e}"),
        }
    }
}

impl std::error::Error for LoadError {}

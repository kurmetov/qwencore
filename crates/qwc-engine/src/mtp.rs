//! MTP draft head: one attention layer on top of the main model's hidden
//! state.
//!
//! The head is in the checkpoint (`mtp.*`, 849 MB BF16) and wasn't loaded before.
//! Its input is the pair (h_t, embedding of token t+1), its output is a prediction of token
//! t+2. Measured acceptance: 0.93 on the first draft and 0.90 without
//! the corpus's synthetic repeats (`bench/results/mtp-acceptance-2026-09-16.md`).
//!
//! A draft step reads 849 MB, i.e. 5% of the main step's weights: these
//! percent buy the second or third token per pass.

use qwc_core::arch::*;
use qwc_cuda::attention_prepare::AttentionPreprocessor;
use qwc_cuda::mtp as kernels;
use qwc_cuda::paged_attention::{
    self, KvCacheDtype, PAGE_SIZE, PackedAttentionWorkspace, PackedShape, PagedAttentionWorkspace,
};
use qwc_cuda::rmsnorm::RmsNorm;
use qwc_cuda::{DeviceBuffer, Stream, bf16};

use crate::weights::ModelWeights;

/// Output width of the head's q_proj: per head it's [q | output gate].
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

/// The head's scratch and its own KV cache. It has one layer, so the cache costs
/// 1/16 of the main model's cache for the same context.
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
    packed_workspace: PackedAttentionWorkspace,
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
    /// `rows` — how many positions the head computes per call (a draft goes
    /// one at a time), `max_context` — the sequence length cap.
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
            packed_workspace: PackedAttentionWorkspace::new(&[PackedShape::Segment], rows, max_context)?,
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

    /// Zeroes the head's KV. Needed when switching to another sequence:
    /// the positions are the same, but the keys are left over from the previous one.
    pub fn reset_cache(&mut self) -> qwc_cuda::Result<()> {
        self.truncate_cache(0)
    }

    /// Zeroes the head's KV starting from the page holding position `keep`.
    /// Positions before it remain; the page's tail before `keep` is zeroed too,
    /// that's cheaper than knowing the layout within a page.
    pub fn truncate_cache(&mut self, keep: usize) -> qwc_cuda::Result<()> {
        let page_bytes =
            PAGE_SIZE * NUM_KV_HEADS * ATTN_HEAD_DIM * self.cache_dtype.bytes_per_element();
        let start = (keep / PAGE_SIZE * page_bytes).min(self.key_cache.len());
        let bytes = self.key_cache.len() - start;
        self.key_cache.zero_range(start, bytes)?;
        self.value_cache.zero_range(start, bytes)
    }

    /// Position tables and page address for rows [0, rows).
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

/// One head pass: from (hidden, next token) get the draft's hidden
/// state. Logits from it are computed by the main model's `lm_head` —
/// the head doesn't have its own.
///
/// `hidden` — the main model's final hidden states for the same rows,
/// `tokens[i]` — the token at position `positions[i] + 1`, i.e. the one the
/// model has just emitted.
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
    if scratch.cache_dtype == KvCacheDtype::Fp8 {
        // A call's rows are always one sequence: the scratch has one KV,
        // and `upload` gives all rows one page table. So the shape is
        // a segment: six heads of eight priming rows go as one tile and
        // read KV once, not eight times. Each row has its own mask, by its
        // position.
        paged_attention::gated_packed(
            &scratch.query,
            &scratch.query_gate,
            &scratch.key_cache,
            &scratch.value_cache,
            scratch.max_blocks,
            &scratch.block_tables,
            &scratch.context_lengths,
            scratch.max_blocks,
            &mut scratch.attention_out,
            PackedShape::Segment,
            rows,
            0,
            max_context,
            &mut scratch.packed_workspace,
            stream,
        )?;
    } else {
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
    }
    kernels::bf16_linear(&head.output, &scratch.attention_out, &mut scratch.normed,
        rows, Q_PROJ_DIM, HIDDEN_SIZE, stream)?;

    // residual = residual + attn, and the same sum is normalized next.
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

/// Loading the head from the checkpoint. Shapes are checked against the architecture, as for
/// the main weights: a mismatch is a load error, not a reason to adapt.
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

/// The head, its scratch and buffers for the draft chain.
///
/// The scratch carries its own single-layer KV for the full context length, so
/// one `Speculator` serves one sequence. When switching to
/// another, the KV must be zeroed: the new sequence's positions are the same, but
/// the contents are foreign. Drafts don't affect correctness — the main model
/// verifies them — but foreign KV drops acceptance.
pub struct Speculator {
    head: MtpHead,
    scratch: MtpScratch,
    hidden: DeviceBuffer<u16>,
    draft_hidden: DeviceBuffer<u16>,
    logits: DeviceBuffer<f32>,
    argmax: qwc_cuda::sampling::Argmax,
    stream: Stream,
    owner: Option<u32>,
    /// The owner's prompt: `bind_prompt` uses it to decide which part of the head's KV
    /// is usable for the next turn of the same dialog.
    owner_prompt: Vec<u32>,
    shortlist: Option<Shortlist>,
}

/// The list of vocab rows over which the draft head looks for its argmax.
///
/// The full vocab projection reads 1.27 GB and costs 0.82 of the draft pass's 1.49 ms.
/// The draft doesn't need an exact argmax over all 248 320 rows: its
/// proposal is verified by the main model, and a list miss costs a rejected
/// draft, not a wrong output.
///
/// The list is a frequency prefix of the vocab plus context tokens. Coverage measured
/// on 25 600 actually generated tokens: context alone gives 0.814,
/// `id < 32768` — 0.915, together — 0.958. The context part is the main one here:
/// in code and agent prompts the model repeats a lot of what's already been said.
struct Shortlist {
    ids: DeviceBuffer<u32>,
    host: Vec<u32>,
    present: std::collections::HashSet<u32>,
    /// How many first vocab rows are always in the list.
    frequent: usize,
    dirty: bool,
}

impl Shortlist {
    fn new(frequent: usize, context_capacity: usize) -> qwc_cuda::Result<Self> {
        let capacity = frequent + context_capacity;
        Ok(Self {
            ids: DeviceBuffer::zeroed(capacity)?,
            host: Vec::with_capacity(capacity),
            present: std::collections::HashSet::with_capacity(capacity),
            frequent,
            dirty: true,
        })
    }

    /// Start the list over for another sequence.
    fn reset(&mut self, context: &[u32]) {
        self.host.clear();
        self.present.clear();
        for &token in context {
            self.push(token);
        }
        for row in 0..self.frequent as u32 {
            self.push(row);
        }
        self.dirty = true;
    }

    fn push(&mut self, token: u32) {
        if self.host.len() < self.host.capacity() && self.present.insert(token) {
            self.host.push(token);
            self.dirty = true;
        }
    }

    fn upload(&mut self) -> qwc_cuda::Result<usize> {
        if self.dirty {
            self.ids.copy_from_slice(&self.host)?;
            self.dirty = false;
        }
        Ok(self.host.len())
    }
}

impl Speculator {
    pub fn new(
        checkpoint: &qwc_model::Checkpoint,
        max_context: usize,
        cache_dtype: KvCacheDtype,
    ) -> Result<Self, crate::weights::LoadError> {
        Ok(Self {
            head: load(checkpoint)?,
            scratch: MtpScratch::new(kernels::MAX_ROWS, max_context, cache_dtype)?,
            hidden: DeviceBuffer::zeroed(kernels::MAX_ROWS * HIDDEN_SIZE)?,
            draft_hidden: DeviceBuffer::zeroed(kernels::MAX_ROWS * HIDDEN_SIZE)?,
            logits: DeviceBuffer::zeroed(VOCAB_SIZE)?,
            argmax: qwc_cuda::sampling::Argmax::new(1, VOCAB_SIZE)?,
            stream: Stream::new()?,
            owner: None,
            owner_prompt: Vec::new(),
            shortlist: None,
        })
    }

    /// Enables the vocab shortlist for draft logits. `frequent` — how many
    /// first vocab rows to always keep, `context_capacity` — how many
    /// distinct sequence tokens the list can hold beyond them.
    pub fn enable_shortlist(
        &mut self,
        frequent: usize,
        context_capacity: usize,
    ) -> Result<(), crate::weights::LoadError> {
        assert!(frequent > 0 && frequent <= VOCAB_SIZE);
        self.shortlist = Some(Shortlist::new(frequent, context_capacity)?);
        Ok(())
    }

    /// Context tokens for the shortlist: the sequence's prompt and everything
    /// it has already emitted. Without them coverage drops from 0.958 to 0.915.
    pub fn set_context(&mut self, context: &[u32]) {
        if let Some(shortlist) = self.shortlist.as_mut() {
            shortlist.reset(context);
        }
    }

    /// An accepted token continues the context: the next draft must be able
    /// to repeat it.
    pub fn observe(&mut self, token: u32) {
        if let Some(shortlist) = self.shortlist.as_mut() {
            shortlist.push(token);
        }
    }

    pub fn resident_bytes(&self) -> usize {
        self.head.resident_bytes() + self.scratch.resident_bytes()
    }

    /// Buffer for the main model's hidden states, up to `MAX_ROWS` rows:
    /// the executor copies them here before `draft_chain`.
    pub fn hidden_mut(&mut self) -> &mut DeviceBuffer<u16> {
        &mut self.hidden
    }

    /// Switches the speculator to another sequence if needed.
    /// Returns `true` if the head's KV was reset.
    pub fn bind(&mut self, sequence: u32) -> qwc_cuda::Result<bool> {
        if self.owner == Some(sequence) {
            return Ok(false);
        }
        self.scratch.reset_cache()?;
        self.owner = Some(sequence);
        self.owner_prompt.clear();
        Ok(true)
    }

    /// Bind the speculator to a sequence with prompt `prompt`.
    ///
    /// If the new prompt continues the previous owner's prompt — the next turn
    /// of the same dialog — the head's KV on the common part stays: the (h, token) pairs
    /// are the same there. This matters together with the prefix cache: the engine doesn't
    /// run the cached history, and the head won't see it again.
    ///
    /// Everything beyond the common part is zeroed. Otherwise two dialogs with the same system
    /// prompt would see each other's KV after it — that's worse than zeros. Position t
    /// carries the pair (h_t, token t+1), so the last common position is
    /// foreign too: its pair has the token where the prompts diverged. Returns
    /// the length of the inherited part; 0 — the KV was reset entirely.
    pub fn bind_prompt(&mut self, sequence: u32, prompt: &[u32]) -> qwc_cuda::Result<usize> {
        if self.owner == Some(sequence) {
            return Ok(prompt.len());
        }
        let common = self
            .owner_prompt
            .iter()
            .zip(prompt)
            .take_while(|(old, new)| old == new)
            .count();
        let keep = common.saturating_sub(1);
        self.scratch.truncate_cache(keep)?;
        self.owner = Some(sequence);
        self.owner_prompt = prompt.to_vec();
        self.set_context(prompt);
        Ok(keep)
    }

    /// Fills the head's KV from the rows of a prefill step: row `first_row + i`
    /// carries h_t, `tokens[i]` is token t+1, and `first_position` is the position of
    /// `tokens[0]`. The head's output is discarded, only its KV is needed.
    ///
    /// Without this the head sees zeros on the prompt: `bind` zeroes the cache, and
    /// drafts write only their own positions. An offline reference with full KV gives
    /// 0.913 on the first draft, a head with empty KV — 0.850. vLLM runs
    /// the MTP layer over the whole prompt during prefill.
    pub fn prime(
        &mut self,
        weights: &ModelWeights,
        executor: &crate::Executor,
        first_row: usize,
        tokens: &[u32],
        first_position: usize,
    ) -> qwc_cuda::Result<()> {
        for (chunk, part) in tokens.chunks(kernels::MAX_ROWS).enumerate() {
            let offset = chunk * kernels::MAX_ROWS;
            executor
                .copy_prefill_hidden_rows(first_row + offset, part.len(), &mut self.hidden)?;
            let positions: Vec<u32> = (0..part.len())
                .map(|i| (first_position + offset + i) as u32)
                .collect();
            draft(
                &self.head,
                weights,
                &mut self.scratch,
                &self.hidden,
                part,
                &positions,
                &mut self.draft_hidden,
                &self.stream,
            )?;
            // The next batch will overwrite `hidden` from another stream.
            self.stream.synchronize()?;
        }
        Ok(())
    }

    /// A chain of `depth` drafts.
    ///
    /// `tokens` — tokens whose (h, token) pairs the head hasn't yet seen with
    /// real h: `hidden` rows for them are contiguous, `position` is the
    /// position of `tokens[0]`. After a regular step it's one pending token,
    /// after verification — all accepted plus the corrected one: their KV was written by drafts
    /// from the head's own output, and the first pass rewrites it with
    /// the main model's real states. The rows go in one pass over the
    /// head's weights, so the rewrite is almost free. After that the head
    /// continues from its own output.
    pub fn draft_chain(
        &mut self,
        weights: &ModelWeights,
        tokens: &[u32],
        position: usize,
        depth: usize,
    ) -> qwc_cuda::Result<Vec<u32>> {
        assert!((1..=kernels::MAX_ROWS).contains(&tokens.len()));
        let last = tokens.len() - 1;
        let mut drafts = Vec::with_capacity(depth);
        let mut input = tokens[last];
        for step in 0..depth {
            let source: *const DeviceBuffer<u16> = if step == 0 {
                &self.hidden
            } else {
                &self.draft_hidden
            };
            // SAFETY: both buffers live in self and don't overlap with the output.
            let source = unsafe { &*source };
            if step == 0 {
                let positions: Vec<u32> =
                    (0..tokens.len()).map(|i| (position + i) as u32).collect();
                draft(
                    &self.head,
                    weights,
                    &mut self.scratch,
                    source,
                    tokens,
                    &positions,
                    &mut self.draft_hidden,
                    &self.stream,
                )?;
                // The vocab and the next step read row 0.
                self.draft_hidden
                    .copy_within(0, last * HIDDEN_SIZE, HIDDEN_SIZE, &self.stream)?;
            } else {
                draft(
                    &self.head,
                    weights,
                    &mut self.scratch,
                    source,
                    &[input],
                    &[(position + last + step) as u32],
                    &mut self.draft_hidden,
                    &self.stream,
                )?;
            }
            let shortlisted = match self.shortlist.as_mut() {
                Some(shortlist) => {
                    let count = shortlist.upload()?;
                    let used = weights.lm_head.logits_subset(
                        &self.draft_hidden,
                        &shortlist.ids,
                        &mut self.logits,
                        count,
                        &self.stream,
                    )?;
                    // argmax runs over the dense output, so it returns a position
                    // in the list, not a token.
                    match used {
                        true => {
                            let slot =
                                self.argmax.sample_prefix(&self.logits, count, &self.stream)?;
                            Some(shortlist.host[slot])
                        }
                        false => None,
                    }
                }
                None => None,
            };
            input = match shortlisted {
                Some(token) => token,
                None => {
                    weights
                        .lm_head
                        .logits(&self.draft_hidden, &mut self.logits, 1, &self.stream)?;
                    self.argmax.sample(&self.logits, 1, &self.stream)?;
                    self.argmax.to_host(1)?[0]
                }
            };
            drafts.push(input);
        }
        Ok(drafts)
    }
}

//! Specialized paged FP8 attention for Qwen3.8 decode.

use crate::error::{Result, check};
use crate::{DeviceBuffer, Stream, ffi};
use qwc_core::arch::{ATTN_HEAD_DIM, GQA_GROUP, NUM_ATTN_HEADS, NUM_KV_HEADS};

pub const PAGE_SIZE: usize = 64;
const TARGET_CTAS: usize = 170;
const QUERY_HEAD_TARGET_CTAS: usize = 340;
const MIN_TOKENS_PER_PARTITION: usize = 32;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum KvCacheDtype {
    #[default]
    Fp8,
    Bf16,
}

impl KvCacheDtype {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fp8 => "fp8",
            Self::Bf16 => "bf16",
        }
    }

    pub fn bytes_per_element(self) -> usize {
        match self {
            Self::Fp8 => 1,
            Self::Bf16 => 2,
        }
    }
}

/// The row-wise path serves one row per sequence, so its
/// partition sizing is computed for the number of slots, not the prefill chunk
/// size: a chunk goes through a separate kernel and doesn't touch the workspace.
pub const MAX_DECODE_ROWS: usize = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeKernel {
    QueryHead,
    SharedKv,
}

/// Measured crossover on RTX 5090 with 16 distinct layer caches. Query-head
/// CTAs usually win because L2 retains shared K/V between the six GQA heads;
/// explicit sharing pays off only where its lower traffic outweighs the
/// 52 KiB/CTA footprint and reduced occupancy.
pub fn select_decode_kernel(batch: usize, max_context: usize) -> DecodeKernel {
    assert!((1..=MAX_DECODE_ROWS).contains(&batch));
    assert!(max_context > 0);
    if (batch == 1 && max_context >= 16_384) || ((8..=16).contains(&batch) && max_context >= 4_096)
    {
        DecodeKernel::SharedKv
    } else {
        DecodeKernel::QueryHead
    }
}

pub fn partition_count(kernel: DecodeKernel, batch: usize, max_context: usize) -> usize {
    assert!((1..=MAX_DECODE_ROWS).contains(&batch));
    assert!(max_context > 0);
    let (target, heads) = match kernel {
        DecodeKernel::QueryHead => (QUERY_HEAD_TARGET_CTAS, NUM_ATTN_HEADS),
        DecodeKernel::SharedKv => (TARGET_CTAS, NUM_KV_HEADS),
    };
    let needed = target.div_ceil(batch * heads);
    let useful = max_context.div_ceil(MIN_TOKENS_PER_PARTITION);
    needed.min(useful).max(1)
}

pub struct PagedAttentionWorkspace {
    storage: DeviceBuffer<f32>,
    bytes: usize,
    batch: usize,
    max_context: usize,
    partitions: usize,
    kernel: DecodeKernel,
    /// The kernel and partitions are chosen on every call by the actual number
    /// of rows, not by capacity. Otherwise a 32-slot executor would run
    /// a single decode at 32K with one partition: 24 CTAs on 170 SMs and 15x the
    /// attention time vs a split for batch 1.
    adaptive: bool,
}

/// Floats of split-K scratch that `batch` rows need across `partitions`.
fn workspace_floats(batch: usize, partitions: usize) -> usize {
    if partitions == 1 {
        0
    } else {
        batch * NUM_ATTN_HEADS * partitions * (ATTN_HEAD_DIM + 2)
    }
}

impl PagedAttentionWorkspace {
    /// Workspace for up to `batch` rows; kernel and partitions follow the
    /// row count of each call.
    pub fn new(batch: usize, max_context: usize) -> Result<Self> {
        assert!((1..=MAX_DECODE_ROWS).contains(&batch));
        assert!(max_context > 0);
        let floats = (1..=batch)
            .map(|rows| {
                let kernel = select_decode_kernel(rows, max_context);
                workspace_floats(rows, partition_count(kernel, rows, max_context))
            })
            .max()
            .unwrap_or(0);
        let kernel = select_decode_kernel(batch, max_context);
        Ok(Self {
            storage: DeviceBuffer::zeroed(floats.max(1))?,
            bytes: floats * size_of::<f32>(),
            batch,
            max_context,
            partitions: partition_count(kernel, batch, max_context),
            kernel,
            adaptive: true,
        })
    }

    pub fn with_kernel(kernel: DecodeKernel, batch: usize, max_context: usize) -> Result<Self> {
        let partitions = partition_count(kernel, batch, max_context);
        Self::with_partitions(kernel, batch, max_context, partitions)
    }

    pub fn with_partitions(
        kernel: DecodeKernel,
        batch: usize,
        max_context: usize,
        partitions: usize,
    ) -> Result<Self> {
        assert!((1..=MAX_DECODE_ROWS).contains(&batch));
        assert!(max_context > 0);
        assert!((1..=max_context).contains(&partitions));
        let floats = workspace_floats(batch, partitions);
        Ok(Self {
            storage: DeviceBuffer::zeroed(floats.max(1))?,
            bytes: floats * size_of::<f32>(),
            batch,
            max_context,
            partitions,
            kernel,
            adaptive: false,
        })
    }

    /// Kernel and partitions a call with `rows` rows will use.
    pub fn plan_for(&self, rows: usize) -> (DecodeKernel, usize) {
        if self.adaptive {
            let kernel = select_decode_kernel(rows, self.max_context);
            (kernel, partition_count(kernel, rows, self.max_context))
        } else {
            (self.kernel, self.partitions)
        }
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn partitions(&self) -> usize {
        self.partitions
    }

    pub fn kernel(&self) -> DecodeKernel {
        self.kernel
    }
}

/// Packed rows (query row x group head) in a packed-kernel tile:
/// at most four m16 tiles.
const PACKED_MAX_TILE_SIZE: usize = 64;
/// SMs on the RTX 5090: the packed kernel holds one CTA per SM, its shared is
/// 72-98 KiB.
const PACKED_SMS: usize = 170;
/// Four pages: a shorter span doesn't pay for writing partial sums.
const PACKED_MIN_TOKENS_PER_PARTITION: usize = 4 * PAGE_SIZE;
/// Fixed CTA overhead in span keys: query staging, pipeline
/// ramp-up, writing partial sums.
const PACKED_CTA_OVERHEAD_KEYS: usize = 64;
/// Cap on (row, partition) pairs: keeps partial sums within 51 MB. The partition
/// output is in f16, and twice as many pairs fit as in fp32: 384 and 512
/// rows take P=7 — 6 and 8 nearly full waves — instead of P=1 (144 CTAs on
/// 170 SMs) and P=4 (4.5 waves, the fifth half empty).
const PACKED_MAX_PARTIAL_ROWS: usize = 4096;
/// Partial-sum bytes per (query row x head, partition) pair: max and
/// sum in fp32, the normalized partition output in f16.
const PACKED_PARTIAL_BYTES: usize = 2 * size_of::<f32>() + ATTN_HEAD_DIM * size_of::<u16>();

/// Partial-sum bytes of the packed kernel for `rows` rows and `partitions`.
fn packed_workspace_bytes(rows: usize, partitions: usize) -> usize {
    if partitions == 1 { 0 } else { rows * NUM_ATTN_HEADS * partitions * PACKED_PARTIAL_BYTES }
}

/// Whose rows go into the packed kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackedShape {
    /// Decode rows of different sequences: each has its own page
    /// table, a tile is one row.
    Rows,
    /// A segment of one sequence: draft verification, prompt tail,
    /// chunk. A tile is several rows with a shared table.
    Segment,
}

/// Packed rows in a packed-kernel tile.
pub fn packed_tile_size(shape: PackedShape, rows: usize) -> usize {
    assert!(rows > 0);
    match shape {
        // A tile is one row: the neighbor has its own page table.
        PackedShape::Rows => GQA_GROUP,
        // Segment rows share the table, and a tile is cut by packed rows
        // regardless of query row boundaries. Up to five rows a segment fits
        // in one tile of 16 or 32 rows.
        PackedShape::Segment => (rows * GQA_GROUP).min(PACKED_MAX_TILE_SIZE),
    }
}

/// m16 tiles in a packed-kernel tile for `tile_size` packed rows.
fn packed_m_tiles(tile_size: usize) -> usize {
    match tile_size {
        0..=16 => 1,
        17..=32 => 2,
        _ => 4,
    }
}

/// Context partitions of the packed kernel for `rows` rows with context up to
/// `context`. Numbers are from `packedbench --sweep`, 16 layers, fp8.
pub fn packed_partition_count(shape: PackedShape, rows: usize, context: usize) -> usize {
    assert!(rows > 0 && context > 0);
    let tile_size = packed_tile_size(shape, rows);
    let ctas = NUM_KV_HEADS * (rows * GQA_GROUP).div_ceil(tile_size);
    let most = packed_most_partitions(rows, context);
    if packed_m_tiles(tile_size) < 4 {
        // One or two m-tiles are memory-bandwidth bound, and 64 CTAs per
        // m-tile saturate it; extra partitions only write partial sums. Decode at 60k:
        // P=16 — 84 µs, P=42 — 91; verifying 4 rows at 30k: P=32 — 50, P=16 — 70.
        return (64 * packed_m_tiles(tile_size)).div_ceil(ctas).min(most);
    }
    // A 64-row tile is tensor-core bound, one CTA per SM, and every
    // started wave costs a full one. Time is waves x (span + CTA overhead);
    // the overhead is fitted on a dense sweep (`packedbench --dense`): 64 keys.
    // The model guesses the best P on 28 of 36 shapes, missing by 1.8% on average;
    // the old "3.5-6 waves" rule missed by 14%, up to 53%. 64 rows
    // at 8k: P=7 (168 CTAs, one wave) — 81 µs, P=28 — 109.
    (1..=most)
        .min_by_key(|&partitions| {
            let (used, span) = packed_split(context, partitions);
            (ctas * used).div_ceil(PACKED_SMS) * (span + PACKED_CTA_OVERHEAD_KEYS)
        })
        .unwrap()
}

/// `packed_partition_count` gives no more partitions than this: a span no shorter than
/// four pages, (row, partition) pairs no more than the cap.
fn packed_most_partitions(rows: usize, context: usize) -> usize {
    context
        .div_ceil(PACKED_MIN_TOKENS_PER_PARTITION)
        .min(PACKED_MAX_PARTIAL_ROWS / rows)
        .max(1)
}

/// Partitions that will suffice for `rows` rows at any context up to
/// `max_context`. The wave model isn't monotonic in context: eight MTP priming
/// rows take P=40 at 32768 but P=42 at 32256, and a workspace sized by
/// the context cap turned out too small. For it the bound is the largest allowed
/// P; for one or two m-tiles P grows with context, and the value at the cap
/// suffices.
fn packed_partition_bound(shape: PackedShape, rows: usize, max_context: usize) -> usize {
    if packed_m_tiles(packed_tile_size(shape, rows)) < 4 {
        packed_partition_count(shape, rows, max_context)
    } else {
        packed_most_partitions(rows, max_context)
    }
}

/// Partial-sum bytes that will suffice for any call of shapes `shapes` up to
/// `max_rows` rows with context up to `max_context`.
fn packed_workspace_capacity(shapes: &[PackedShape], max_rows: usize, max_context: usize) -> usize {
    (1..=max_rows)
        .flat_map(|rows| {
            shapes.iter().map(move |&shape| {
                packed_workspace_bytes(rows, packed_partition_bound(shape, rows, max_context))
            })
        })
        .max()
        .unwrap_or(0)
}

/// How many partitions get keys and what span length, if context
/// `context` is split into `partitions`: the kernel rounds the span to a page.
fn packed_split(context: usize, partitions: usize) -> (usize, usize) {
    let span = context.div_ceil(partitions * PAGE_SIZE) * PAGE_SIZE;
    (context.div_ceil(span), span)
}

/// Partial sums of packed attention.
pub struct PackedAttentionWorkspace {
    storage: DeviceBuffer<u8>,
    bytes: usize,
    max_rows: usize,
    max_context: usize,
    /// Number of partitions for benches and tests; `None` — by call shape.
    forced: Option<usize>,
}

/// Call plan for the packed kernel. The kernel computes the partition span itself, from
/// the tile's actual context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedPlan {
    /// Packed rows in a tile.
    pub tile_size: usize,
    pub partitions: usize,
}

impl PackedAttentionWorkspace {
    /// For any call of shapes `shapes` up to `max_rows` rows with context up to
    /// `max_context`. Decode rows need kilobytes, segments need up to
    /// 51 MB.
    pub fn new(shapes: &[PackedShape], max_rows: usize, max_context: usize) -> Result<Self> {
        assert!(max_rows > 0 && max_context > 0 && !shapes.is_empty());
        let bytes = packed_workspace_capacity(shapes, max_rows, max_context);
        Self::allocate(bytes, max_rows, max_context, None)
    }

    pub fn with_partitions(max_rows: usize, max_context: usize, partitions: usize) -> Result<Self> {
        assert!(max_rows > 0 && max_context > 0);
        assert!((1..=max_context.div_ceil(PAGE_SIZE)).contains(&partitions));
        let bytes = packed_workspace_bytes(max_rows, partitions);
        Self::allocate(bytes, max_rows, max_context, Some(partitions))
    }

    fn allocate(
        bytes: usize,
        max_rows: usize,
        max_context: usize,
        forced: Option<usize>,
    ) -> Result<Self> {
        Ok(Self {
            storage: DeviceBuffer::zeroed(bytes.max(1))?,
            bytes,
            max_rows,
            max_context,
            forced,
        })
    }

    /// Tile and partitions for a call.
    pub fn plan_for(&self, shape: PackedShape, rows: usize, context: usize) -> PackedPlan {
        assert!(rows > 0 && rows <= self.max_rows);
        assert!(context > 0 && context <= self.max_context);
        let partitions = self
            .forced
            .unwrap_or_else(|| packed_partition_count(shape, rows, context));
        PackedPlan { tile_size: packed_tile_size(shape, rows), partitions }
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

/// Attention with a packed GQA group over the fp8 cache: a CTA per KV head, the six
/// heads of a group go as rows of one MMA tile, and KV is read once.
///
/// `context` picks the number of partitions and must be at least the largest
/// context of the rows; the kernel cuts the spans itself by the actual context. Under
/// a CUDA graph the context cap should be passed here, not the current one. Rows are
/// `row_base..row_base + rows`; for [`PackedShape::Segment`] they share a
/// page table.
#[allow(clippy::too_many_arguments)]
pub fn gated_packed(
    query: &DeviceBuffer<u16>,
    query_gate_projection: &DeviceBuffer<u16>,
    key_cache: &DeviceBuffer<u8>,
    value_cache: &DeviceBuffer<u8>,
    num_blocks: usize,
    block_tables: &DeviceBuffer<u32>,
    context_lengths: &DeviceBuffer<u32>,
    max_blocks_per_sequence: usize,
    output: &mut DeviceBuffer<u16>,
    shape: PackedShape,
    rows: usize,
    row_base: usize,
    context: usize,
    workspace: &mut PackedAttentionWorkspace,
    stream: &Stream,
) -> Result<()> {
    let plan = workspace.plan_for(shape, rows, context);
    assert!(workspace.bytes >= packed_workspace_bytes(rows, plan.partitions));
    let end = row_base + rows;
    assert!(query.len() >= end * NUM_ATTN_HEADS * ATTN_HEAD_DIM);
    assert!(query_gate_projection.len() >= end * NUM_ATTN_HEADS * ATTN_HEAD_DIM * 2);
    assert!(output.len() >= end * NUM_ATTN_HEADS * ATTN_HEAD_DIM);
    assert!(context_lengths.len() >= end);
    assert!(block_tables.len() >= end * max_blocks_per_sequence);
    let cache_bytes = num_blocks * NUM_KV_HEADS * PAGE_SIZE * ATTN_HEAD_DIM;
    assert_eq!(key_cache.len(), cache_bytes, "the packed kernel is fp8-cache only");
    assert_eq!(value_cache.len(), cache_bytes);
    check(unsafe {
        ffi::qwc_paged_attention_packed_fp8(
            query.as_ptr(),
            query_gate_projection.as_ptr(),
            key_cache.as_ptr(),
            value_cache.as_ptr(),
            block_tables.as_ptr(),
            context_lengths.as_ptr(),
            output.as_mut_ptr(),
            workspace.storage.as_mut_ptr().cast(),
            workspace.bytes,
            rows as i32,
            row_base as i32,
            max_blocks_per_sequence as i32,
            plan.tile_size as i32,
            plan.partitions as i32,
            1.0 / (ATTN_HEAD_DIM as f32).sqrt(),
            stream.raw(),
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub fn decode_fp8(
    query: &DeviceBuffer<u16>,
    key_cache: &DeviceBuffer<u8>,
    value_cache: &DeviceBuffer<u8>,
    num_blocks: usize,
    block_tables: &DeviceBuffer<u32>,
    context_lengths: &DeviceBuffer<u32>,
    max_blocks_per_sequence: usize,
    output: &mut DeviceBuffer<u16>,
    workspace: &mut PagedAttentionWorkspace,
    batch: usize,
    max_context: usize,
    stream: &Stream,
) -> Result<()> {
    decode_impl(
        query,
        None,
        key_cache,
        value_cache,
        num_blocks,
        block_tables,
        context_lengths,
        max_blocks_per_sequence,
        output,
        workspace,
        batch,
        max_context,
        KvCacheDtype::Fp8,
        stream,
    )
}

/// Attention with Qwen's per-head output gate fused into the final write.
#[allow(clippy::too_many_arguments)]
pub fn decode_fp8_gated(
    query: &DeviceBuffer<u16>,
    query_gate_projection: &DeviceBuffer<u16>,
    key_cache: &DeviceBuffer<u8>,
    value_cache: &DeviceBuffer<u8>,
    num_blocks: usize,
    block_tables: &DeviceBuffer<u32>,
    context_lengths: &DeviceBuffer<u32>,
    max_blocks_per_sequence: usize,
    output: &mut DeviceBuffer<u16>,
    workspace: &mut PagedAttentionWorkspace,
    batch: usize,
    max_context: usize,
    stream: &Stream,
) -> Result<()> {
    decode_impl(
        query,
        Some(query_gate_projection),
        key_cache,
        value_cache,
        num_blocks,
        block_tables,
        context_lengths,
        max_blocks_per_sequence,
        output,
        workspace,
        batch,
        max_context,
        KvCacheDtype::Fp8,
        stream,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn decode_gated(
    query: &DeviceBuffer<u16>,
    query_gate_projection: &DeviceBuffer<u16>,
    key_cache: &DeviceBuffer<u8>,
    value_cache: &DeviceBuffer<u8>,
    num_blocks: usize,
    block_tables: &DeviceBuffer<u32>,
    context_lengths: &DeviceBuffer<u32>,
    max_blocks_per_sequence: usize,
    output: &mut DeviceBuffer<u16>,
    workspace: &mut PagedAttentionWorkspace,
    batch: usize,
    max_context: usize,
    cache_dtype: KvCacheDtype,
    stream: &Stream,
) -> Result<()> {
    decode_impl(
        query,
        Some(query_gate_projection),
        key_cache,
        value_cache,
        num_blocks,
        block_tables,
        context_lengths,
        max_blocks_per_sequence,
        output,
        workspace,
        batch,
        max_context,
        cache_dtype,
        stream,
    )
}

#[allow(clippy::too_many_arguments)]
fn decode_impl(
    query: &DeviceBuffer<u16>,
    query_gate_projection: Option<&DeviceBuffer<u16>>,
    key_cache: &DeviceBuffer<u8>,
    value_cache: &DeviceBuffer<u8>,
    num_blocks: usize,
    block_tables: &DeviceBuffer<u32>,
    context_lengths: &DeviceBuffer<u32>,
    max_blocks_per_sequence: usize,
    output: &mut DeviceBuffer<u16>,
    workspace: &mut PagedAttentionWorkspace,
    batch: usize,
    max_context: usize,
    cache_dtype: KvCacheDtype,
    stream: &Stream,
) -> Result<()> {
    assert!(batch > 0 && batch <= workspace.batch);
    assert!(max_context > 0 && max_context <= workspace.max_context);
    assert!((1..=max_blocks_per_sequence * PAGE_SIZE).contains(&max_context));
    assert!(query.len() >= batch * NUM_ATTN_HEADS * ATTN_HEAD_DIM);
    if let Some(gate) = query_gate_projection {
        assert!(gate.len() >= batch * NUM_ATTN_HEADS * ATTN_HEAD_DIM * 2);
    }
    assert!(output.len() >= batch * NUM_ATTN_HEADS * ATTN_HEAD_DIM);
    assert!(context_lengths.len() >= batch);
    assert!(block_tables.len() >= batch * max_blocks_per_sequence);
    let cache_elements = num_blocks * NUM_KV_HEADS * PAGE_SIZE * ATTN_HEAD_DIM;
    let cache_bytes = cache_elements * cache_dtype.bytes_per_element();
    assert_eq!(key_cache.len(), cache_bytes);
    assert_eq!(value_cache.len(), cache_bytes);
    assert_eq!(GQA_GROUP, 6);

    let (kernel, partitions) = workspace.plan_for(batch);
    let launch = match cache_dtype {
        KvCacheDtype::Fp8 => ffi::qwc_paged_attention_fp8,
        KvCacheDtype::Bf16 => ffi::qwc_paged_attention_bf16,
    };
    check(unsafe {
        launch(
            query.as_ptr(),
            query_gate_projection.map_or(std::ptr::null(), DeviceBuffer::as_ptr),
            key_cache.as_ptr(),
            value_cache.as_ptr(),
            block_tables.as_ptr(),
            context_lengths.as_ptr(),
            output.as_mut_ptr(),
            workspace.storage.as_mut_ptr(),
            workspace.bytes,
            batch as i32,
            max_blocks_per_sequence as i32,
            partitions as i32,
            i32::from(kernel == DecodeKernel::SharedKv),
            1.0 / (ATTN_HEAD_DIM as f32).sqrt(),
            stream.raw(),
        )
    })
}

/// Causal attention over a contiguous chunk of prefill rows of one sequence.
///
/// The decode kernel serves one query row per pass and so re-reads
/// KV pages as many times as there are tokens in the chunk. Here one KV pass
/// serves a tile of rows, so traffic drops by a multiple of the tile size.
/// Rows must be consecutive and belong to one sequence: the tile
/// shares a block table, and the mask stays causal because
/// each row has its own `context_lengths`.
///
/// Reductions here are warp-level. The kernel is kept as a reference: tests check
/// the tensor-core kernel against it.
#[allow(clippy::too_many_arguments)]
pub fn prefill_gated(
    query: &DeviceBuffer<u16>,
    query_gate_projection: &DeviceBuffer<u16>,
    key_cache: &DeviceBuffer<u8>,
    value_cache: &DeviceBuffer<u8>,
    num_blocks: usize,
    block_tables: &DeviceBuffer<u32>,
    context_lengths: &DeviceBuffer<u32>,
    max_blocks_per_sequence: usize,
    output: &mut DeviceBuffer<u16>,
    rows: usize,
    row_base: usize,
    cache_dtype: KvCacheDtype,
    stream: &Stream,
) -> Result<()> {
    assert!(rows > 0);
    let end = row_base + rows;
    assert!(query.len() >= end * NUM_ATTN_HEADS * ATTN_HEAD_DIM);
    assert!(query_gate_projection.len() >= end * NUM_ATTN_HEADS * ATTN_HEAD_DIM * 2);
    assert!(output.len() >= end * NUM_ATTN_HEADS * ATTN_HEAD_DIM);
    assert!(context_lengths.len() >= end);
    assert!(block_tables.len() >= end * max_blocks_per_sequence);
    let cache_elements = num_blocks * NUM_KV_HEADS * PAGE_SIZE * ATTN_HEAD_DIM;
    let cache_bytes = cache_elements * cache_dtype.bytes_per_element();
    assert_eq!(key_cache.len(), cache_bytes);
    assert_eq!(value_cache.len(), cache_bytes);

    let launch = match cache_dtype {
        KvCacheDtype::Fp8 => ffi::qwc_paged_attention_prefill_fp8,
        KvCacheDtype::Bf16 => ffi::qwc_paged_attention_prefill_bf16,
    };
    check(unsafe {
        launch(
            query.as_ptr(),
            query_gate_projection.as_ptr(),
            key_cache.as_ptr(),
            value_cache.as_ptr(),
            block_tables.as_ptr(),
            context_lengths.as_ptr(),
            output.as_mut_ptr(),
            rows as i32,
            row_base as i32,
            max_blocks_per_sequence as i32,
            1.0 / (ATTN_HEAD_DIM as f32).sqrt(),
            stream.raw(),
        )
    })
}

/// The same on tensor cores: QK^T and PV go through mma.m16n8k16 rather than
/// warp reductions.
///
/// A tile of rows must lie in one sequence — the page table
/// is taken from the tile's first row. A prefill chunk satisfies this condition.
#[allow(clippy::too_many_arguments)]
pub fn prefill_gated_mma(
    query: &DeviceBuffer<u16>,
    query_gate_projection: &DeviceBuffer<u16>,
    key_cache: &DeviceBuffer<u8>,
    value_cache: &DeviceBuffer<u8>,
    num_blocks: usize,
    block_tables: &DeviceBuffer<u32>,
    context_lengths: &DeviceBuffer<u32>,
    max_blocks_per_sequence: usize,
    output: &mut DeviceBuffer<u16>,
    rows: usize,
    row_base: usize,
    cache_dtype: KvCacheDtype,
    stream: &Stream,
) -> Result<()> {
    assert!(rows > 0);
    let end = row_base + rows;
    assert!(query.len() >= end * NUM_ATTN_HEADS * ATTN_HEAD_DIM);
    assert!(query_gate_projection.len() >= end * NUM_ATTN_HEADS * ATTN_HEAD_DIM * 2);
    assert!(output.len() >= end * NUM_ATTN_HEADS * ATTN_HEAD_DIM);
    assert!(context_lengths.len() >= end);
    assert!(block_tables.len() >= end * max_blocks_per_sequence);
    let cache_elements = num_blocks * NUM_KV_HEADS * PAGE_SIZE * ATTN_HEAD_DIM;
    let cache_bytes = cache_elements * cache_dtype.bytes_per_element();
    assert_eq!(key_cache.len(), cache_bytes);
    assert_eq!(value_cache.len(), cache_bytes);

    let launch = match cache_dtype {
        KvCacheDtype::Fp8 => ffi::qwc_paged_attention_prefill_mma_fp8,
        KvCacheDtype::Bf16 => ffi::qwc_paged_attention_prefill_mma_bf16,
    };
    check(unsafe {
        launch(
            query.as_ptr(),
            query_gate_projection.as_ptr(),
            key_cache.as_ptr(),
            value_cache.as_ptr(),
            block_tables.as_ptr(),
            context_lengths.as_ptr(),
            output.as_mut_ptr(),
            rows as i32,
            row_base as i32,
            max_blocks_per_sequence as i32,
            1.0 / (ATTN_HEAD_DIM as f32).sqrt(),
            stream.raw(),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measured_dispatch_table_keeps_both_paths_live() {
        assert_eq!(select_decode_kernel(1, 8_192), DecodeKernel::QueryHead);
        assert_eq!(select_decode_kernel(1, 32_768), DecodeKernel::SharedKv);
        assert_eq!(select_decode_kernel(4, 8_192), DecodeKernel::QueryHead);
        assert_eq!(select_decode_kernel(8, 8_192), DecodeKernel::SharedKv);
        assert_eq!(select_decode_kernel(16, 2_048), DecodeKernel::QueryHead);
        assert_eq!(select_decode_kernel(32, 8_192), DecodeKernel::QueryHead);
    }

    /// The best points of `packedbench --sweep` at 60k: the partition choice must
    /// hit them.
    #[test]
    fn packed_partitions_follow_the_measured_sweep() {
        let rows_plan = |rows| packed_partition_count(PackedShape::Rows, rows, 60_000);
        assert_eq!([1, 2, 4, 8, 32].map(rows_plan), [16, 8, 4, 2, 1]);
        let segment_plan = |rows| packed_partition_count(PackedShape::Segment, rows, 60_000);
        assert_eq!(segment_plan(1), 16);
        assert_eq!(segment_plan(4), 32);
        // One tile: a full wave of 164 CTAs.
        assert_eq!(segment_plan(8), 41);
        // 64-row tiles are 24 CTAs per partition, and 7 partitions give 168 CTAs:
        // one nearly full wave or a whole number of them.
        assert_eq!([64, 128, 192, 256].map(segment_plan), [7, 7, 7, 7]);
        // 384 and 512 rows are 144 and 192 CTAs per partition: P=7 gives 6 and 8 nearly
        // full waves. In fp32 the partial sums allowed them only up to P=5 and P=4.
        assert_eq!([384, 512].map(segment_plan), [7, 7]);
        // Beyond that P hits the partial-sum cap.
        assert_eq!(segment_plan(1_024), 3);
        assert_eq!(segment_plan(2_048), 1);
        // A short context isn't cut finer than four pages.
        assert_eq!(packed_partition_count(PackedShape::Segment, 4, 1_000), 4);
        // Partial sums don't exceed the (row, partition) pair cap.
        for rows in 1..=2_048 {
            let partitions = segment_plan(rows);
            assert!(partitions == 1 || rows * partitions <= PACKED_MAX_PARTIAL_ROWS, "{rows}");
        }
    }

    /// A workspace sized by the context cap must fit a call with
    /// any smaller context: MTP priming with eight rows at 32256
    /// asked for P=42 with P=40 at the 32768 cap and failed the size check.
    #[test]
    fn packed_workspace_covers_every_context_below_the_ceiling() {
        for (shapes, max_rows) in [
            (&[PackedShape::Segment][..], 8),
            (&[PackedShape::Rows][..], 8),
            (&[PackedShape::Rows, PackedShape::Segment][..], 2_048),
        ] {
            let max_context = 32_768;
            let capacity = packed_workspace_capacity(shapes, max_rows, max_context);
            for &shape in shapes {
                let sizes = (1..=12).chain([64, 96, 384, 512, 585, 1_024, 2_048]);
                for rows in sizes.filter(|&rows| rows <= max_rows) {
                    for context in (PAGE_SIZE..=max_context).step_by(PAGE_SIZE) {
                        let partitions = packed_partition_count(shape, rows, context);
                        assert!(
                            packed_workspace_bytes(rows, partitions) <= capacity,
                            "{shape:?}: {rows} rows at {context}, P={partitions}"
                        );
                    }
                }
            }
        }
        // Eight segment rows: 12.8 MB, not 51 as for the prefill arena.
        assert_eq!(packed_workspace_capacity(&[PackedShape::Segment], 8, 32_768), 8 * 24 * 128 * 520);
        assert_eq!(
            packed_workspace_capacity(&[PackedShape::Rows, PackedShape::Segment], 2_048, 32_768),
            PACKED_MAX_PARTIAL_ROWS * NUM_ATTN_HEADS * PACKED_PARTIAL_BYTES
        );
    }
}

pub mod reference {
    use super::*;
    use crate::{bf16, nvfp4};

    #[allow(clippy::too_many_arguments)]
    pub fn decode_fp8(
        query: &[u16],
        key_cache: &[u8],
        value_cache: &[u8],
        block_tables: &[u32],
        context_lengths: &[u32],
        max_blocks_per_sequence: usize,
        output: &mut [f32],
        batch: usize,
    ) {
        assert_eq!(query.len(), batch * NUM_ATTN_HEADS * ATTN_HEAD_DIM);
        assert_eq!(output.len(), query.len());
        let scale = 1.0 / (ATTN_HEAD_DIM as f32).sqrt();
        for b in 0..batch {
            let context = context_lengths[b] as usize;
            for query_head in 0..NUM_ATTN_HEADS {
                let kv_head = query_head / GQA_GROUP;
                let query_base = (b * NUM_ATTN_HEADS + query_head) * ATTN_HEAD_DIM;
                let mut maximum = f32::NEG_INFINITY;
                let mut denominator = 0.0f32;
                let mut accumulator = [0.0f32; ATTN_HEAD_DIM];
                for token in 0..context {
                    let physical = block_tables[b * max_blocks_per_sequence + token / PAGE_SIZE];
                    let cache_base = ((physical as usize * NUM_KV_HEADS + kv_head) * PAGE_SIZE
                        + token % PAGE_SIZE)
                        * ATTN_HEAD_DIM;
                    let mut score = 0.0f32;
                    for dimension in 0..ATTN_HEAD_DIM {
                        score += bf16::to_f32(query[query_base + dimension])
                            * nvfp4::reference::e4m3(key_cache[cache_base + dimension]);
                    }
                    score *= scale;
                    let next_maximum = maximum.max(score);
                    let old_weight = (maximum - next_maximum).exp();
                    let token_weight = (score - next_maximum).exp();
                    denominator = denominator * old_weight + token_weight;
                    for dimension in 0..ATTN_HEAD_DIM {
                        accumulator[dimension] = accumulator[dimension] * old_weight
                            + token_weight
                                * nvfp4::reference::e4m3(value_cache[cache_base + dimension]);
                    }
                    maximum = next_maximum;
                }
                for dimension in 0..ATTN_HEAD_DIM {
                    output[query_base + dimension] = accumulator[dimension] / denominator;
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn decode_bf16(
        query: &[u16],
        key_cache: &[u16],
        value_cache: &[u16],
        block_tables: &[u32],
        context_lengths: &[u32],
        max_blocks_per_sequence: usize,
        output: &mut [f32],
        batch: usize,
    ) {
        assert_eq!(query.len(), batch * NUM_ATTN_HEADS * ATTN_HEAD_DIM);
        assert_eq!(output.len(), query.len());
        let scale = 1.0 / (ATTN_HEAD_DIM as f32).sqrt();
        for b in 0..batch {
            let context = context_lengths[b] as usize;
            for query_head in 0..NUM_ATTN_HEADS {
                let kv_head = query_head / GQA_GROUP;
                let query_base = (b * NUM_ATTN_HEADS + query_head) * ATTN_HEAD_DIM;
                let mut maximum = f32::NEG_INFINITY;
                let mut denominator = 0.0f32;
                let mut accumulator = [0.0f32; ATTN_HEAD_DIM];
                for token in 0..context {
                    let physical = block_tables[b * max_blocks_per_sequence + token / PAGE_SIZE];
                    let cache_base = ((physical as usize * NUM_KV_HEADS + kv_head) * PAGE_SIZE
                        + token % PAGE_SIZE)
                        * ATTN_HEAD_DIM;
                    let mut score = 0.0f32;
                    for dimension in 0..ATTN_HEAD_DIM {
                        score += bf16::to_f32(query[query_base + dimension])
                            * bf16::to_f32(key_cache[cache_base + dimension]);
                    }
                    score *= scale;
                    let next_maximum = maximum.max(score);
                    let old_weight = (maximum - next_maximum).exp();
                    let token_weight = (score - next_maximum).exp();
                    denominator = denominator * old_weight + token_weight;
                    for dimension in 0..ATTN_HEAD_DIM {
                        accumulator[dimension] = accumulator[dimension] * old_weight
                            + token_weight * bf16::to_f32(value_cache[cache_base + dimension]);
                    }
                    maximum = next_maximum;
                }
                for dimension in 0..ATTN_HEAD_DIM {
                    output[query_base + dimension] = accumulator[dimension] / denominator;
                }
            }
        }
    }
}

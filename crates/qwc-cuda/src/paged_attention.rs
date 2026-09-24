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

/// Построчный путь обслуживает по строке на последовательность, поэтому его
/// разметка партиций рассчитана на число слотов, а не на размер чанка
/// префилла: чанк идёт отдельным ядром и workspace не трогает.
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
    /// Ядро и партиции выбираются на каждом вызове по фактическому числу
    /// строк, а не по ёмкости. Иначе исполнитель на 32 слота гонял бы
    /// одиночный decode на 32K одной партицией: 24 CTA на 170 SM и 15x к
    /// времени внимания против разбиения под batch 1.
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
/// Decode-ядро обслуживает одну строку запроса за проход и потому перечитывает
/// страницы KV столько раз, сколько в чанке токенов. Здесь один проход по KV
/// обслуживает тайл строк, так что трафик падает кратно размеру тайла.
/// Строки должны идти подряд и принадлежать одной последовательности: тайл
/// делит между собой таблицу блоков, а маска остаётся причинной за счёт того,
/// что у каждой строки свой `context_lengths`.
///
/// Редукции здесь варповые. Ядро оставлено как эталон: тесты сверяют с ним
/// ядро на тензорных ядрах.
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

/// То же самое на тензорных ядрах: QK^T и PV идут через mma.m16n8k16, а не
/// через варповые редукции.
///
/// Тайл строк должен лежать в одной последовательности — таблица страниц
/// берётся у первой строки тайла. Чанк префилла это условие выполняет.
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

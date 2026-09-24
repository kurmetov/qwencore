//! Прямые объявления CUDA Runtime API. Только то, что используется.

use std::ffi::{c_char, c_int, c_void};

pub type Stream = *mut c_void;
pub type Event = *mut c_void;
pub type Graph = *mut c_void;
pub type GraphExec = *mut c_void;

pub const MEMCPY_HOST_TO_DEVICE: c_int = 1;
pub const MEMCPY_DEVICE_TO_HOST: c_int = 2;
pub const MEMCPY_DEVICE_TO_DEVICE: c_int = 3;

unsafe extern "C" {
    pub fn cudaGetErrorString(error: c_int) -> *const c_char;
    pub fn cudaSetDevice(device: c_int) -> c_int;
    pub fn cudaDeviceSynchronize() -> c_int;
    pub fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> c_int;
    pub fn cudaDeviceGetAttribute(value: *mut c_int, attr: c_int, device: c_int) -> c_int;

    pub fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> c_int;
    pub fn cudaFree(ptr: *mut c_void) -> c_int;
    pub fn cudaMemset(ptr: *mut c_void, value: c_int, count: usize) -> c_int;
    pub fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: c_int) -> c_int;
    pub fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: c_int,
        stream: Stream,
    ) -> c_int;

    pub fn cudaStreamCreate(stream: *mut Stream) -> c_int;
    pub fn cudaStreamDestroy(stream: Stream) -> c_int;
    pub fn cudaStreamSynchronize(stream: Stream) -> c_int;
    pub fn cudaStreamBeginCapture(stream: Stream, mode: c_int) -> c_int;
    pub fn cudaStreamEndCapture(stream: Stream, graph: *mut Graph) -> c_int;

    pub fn cudaGraphInstantiateWithFlags(
        graph_exec: *mut GraphExec,
        graph: Graph,
        flags: u64,
    ) -> c_int;
    pub fn cudaGraphLaunch(graph_exec: GraphExec, stream: Stream) -> c_int;
    pub fn cudaGraphExecDestroy(graph_exec: GraphExec) -> c_int;
    pub fn cudaGraphDestroy(graph: Graph) -> c_int;

    pub fn cudaEventCreate(event: *mut Event) -> c_int;
    pub fn cudaEventDestroy(event: Event) -> c_int;
    pub fn cudaEventRecord(event: Event, stream: Stream) -> c_int;
    pub fn cudaEventSynchronize(event: Event) -> c_int;
    pub fn cudaEventElapsedTime(ms: *mut f32, start: Event, end: Event) -> c_int;

    // Кернелы из cuda/bandwidth.cu
    pub fn qwc_bw_read(src: *const c_void, bytes: usize, out: *mut f32, stream: Stream) -> c_int;
    pub fn qwc_bw_copy(src: *const c_void, dst: *mut c_void, bytes: usize, stream: Stream)
    -> c_int;

    // Кернелы из cuda/vocab_fp8.cu
    pub fn qwc_fp8_quantize_rows(
        input: *const c_void,
        output: *mut c_void,
        row_scales: *mut c_void,
        rows: c_int,
        cols: c_int,
        first_row: c_int,
        stream: Stream,
    ) -> c_int;

    #[allow(clippy::too_many_arguments)]
    pub fn qwc_fp8_embedding_gather(
        table: *const c_void,
        row_scales: *const c_void,
        token_ids: *const c_void,
        output: *mut c_void,
        batch: c_int,
        cols: c_int,
        vocab: c_int,
        stream: Stream,
    ) -> c_int;

    #[allow(clippy::too_many_arguments)]
    pub fn qwc_bf16_embedding_gather(
        table: *const c_void,
        token_ids: *const c_void,
        output: *mut c_void,
        batch: c_int,
        cols: c_int,
        vocab: c_int,
        stream: Stream,
    ) -> c_int;

    #[allow(clippy::too_many_arguments)]
    pub fn qwc_fp8_lm_head(
        weights: *const c_void,
        row_scales: *const c_void,
        hidden: *const c_void,
        logits: *mut c_void,
        batch: c_int,
        hidden_size: c_int,
        vocab: c_int,
        stream: Stream,
    ) -> c_int;

    /// Логиты только по списку строк словаря — для черновой головы.
    #[allow(clippy::too_many_arguments)]
    pub fn qwc_fp8_lm_head_subset(
        weights: *const c_void,
        row_scales: *const c_void,
        hidden: *const c_void,
        row_ids: *const c_void,
        logits: *mut c_void,
        count: c_int,
        hidden_size: c_int,
        vocab: c_int,
        stream: Stream,
    ) -> c_int;

    #[allow(clippy::too_many_arguments)]
    pub fn qwc_bf16_lm_head(
        weights: *const c_void,
        hidden: *const c_void,
        logits: *mut c_void,
        batch: c_int,
        hidden_size: c_int,
        vocab: c_int,
        stream: Stream,
    ) -> c_int;

    #[allow(clippy::too_many_arguments)]
    pub fn qwc_fp8_lm_head_batched(
        weights: *const c_void,
        row_scales: *const c_void,
        hidden: *const c_void,
        logits: *mut c_void,
        batch: c_int,
        hidden_size: c_int,
        vocab: c_int,
        stream: Stream,
    ) -> c_int;

    pub fn qwc_fp8_max_logits_batch() -> c_int;

    pub fn qwc_fp8_max_batched_logits() -> c_int;

    // Кернел из cuda/delta_net.cu
    #[allow(clippy::too_many_arguments)]
    pub fn qwc_delta_decode(
        state: *mut c_void,
        state_slots: *const c_void,
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        alpha: *const c_void,
        beta: *const c_void,
        kq: *const c_void,
        out: *mut c_void,
        state_capacity: c_int,
        batch: c_int,
        stream: Stream,
    ) -> c_int;

    #[allow(clippy::too_many_arguments)]
    pub fn qwc_delta_decode_int8(
        state: *mut c_void,
        state_scales: *mut f32,
        state_slots: *const c_void,
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        alpha: *const c_void,
        beta: *const c_void,
        kq: *const c_void,
        out: *mut c_void,
        state_capacity: c_int,
        batch: c_int,
        stream: Stream,
    ) -> c_int;

    #[allow(clippy::too_many_arguments)]
    pub fn qwc_delta_state_pack(
        source: *const c_void,
        packed: *mut c_void,
        scales: *mut f32,
        source_capacity: c_int,
        source_slot: c_int,
        packed_capacity: c_int,
        packed_slot: c_int,
        stream: Stream,
    ) -> c_int;

    #[allow(clippy::too_many_arguments)]
    pub fn qwc_delta_state_unpack(
        packed: *const c_void,
        scales: *const f32,
        destination: *mut c_void,
        packed_capacity: c_int,
        packed_slot: c_int,
        destination_capacity: c_int,
        destination_slot: c_int,
        stream: Stream,
    ) -> c_int;

    pub fn qwc_delta_state_requantize(
        state: *mut c_void,
        state_slots: *const c_void,
        first_slot: c_int,
        state_capacity: c_int,
        count: c_int,
        mode: c_int,
        stream: Stream,
    ) -> c_int;

    #[allow(clippy::too_many_arguments)]
    pub fn qwc_delta_prefill(
        state: *mut c_void,
        q: *const f32,
        k: *const f32,
        v: *const f32,
        alpha: *const f32,
        beta: *const f32,
        kq: *const f32,
        out: *mut f32,
        state_capacity: c_int,
        state_slot: c_int,
        tokens: c_int,
        round_state_per_token: c_int,
        row_offset: c_int,
        stream: Stream,
    ) -> c_int;

    pub fn qwc_bf16_linear(
        weights: *const c_void,
        input: *const c_void,
        output: *mut c_void,
        rows: c_int,
        k: c_int,
        n: c_int,
        stream: Stream,
    ) -> c_int;

    pub fn qwc_bf16_swiglu(
        gate: *const c_void,
        up: *const c_void,
        out: *mut c_void,
        elements: c_int,
        stream: Stream,
    ) -> c_int;

    pub fn qwc_bf16_concat(
        left: *const c_void,
        right: *const c_void,
        out: *mut c_void,
        rows: c_int,
        width: c_int,
        stream: Stream,
    ) -> c_int;

    #[allow(clippy::too_many_arguments)]
    pub fn qwc_delta_prefill_wy(
        state: *mut c_void,
        q: *const f32,
        k: *const f32,
        v: *const f32,
        alpha: *const f32,
        beta: *const f32,
        out: *mut f32,
        query_tile: *mut c_void,
        key_tile: *mut c_void,
        gram_kk: *mut c_void,
        gram_qk: *mut c_void,
        inverse: *mut c_void,
        output_factor: *mut c_void,
        value_tile: *mut c_void,
        log_decay: *mut c_void,
        state_capacity: c_int,
        state_slot: c_int,
        tokens: c_int,
        row_offset: c_int,
        stream: Stream,
    ) -> c_int;

    #[allow(clippy::too_many_arguments)]
    pub fn qwc_delta_prepare_decode(
        mixed_qkv: *const c_void,
        a_projection: *const c_void,
        b_projection: *const c_void,
        conv_weight: *const c_void,
        a_log: *const c_void,
        dt_bias: *const c_void,
        conv_state: *mut c_void,
        state_slots: *const c_void,
        query: *mut c_void,
        key: *mut c_void,
        value: *mut c_void,
        alpha: *mut c_void,
        beta: *mut c_void,
        kq: *mut c_void,
        state_capacity: c_int,
        batch: c_int,
        mixed_stride: c_int,
        gate_stride: c_int,
        stream: Stream,
    ) -> c_int;

    #[allow(clippy::too_many_arguments)]
    pub fn qwc_delta_prepare_prefill(
        mixed_qkv: *const c_void,
        a_projection: *const c_void,
        b_projection: *const c_void,
        conv_weight: *const c_void,
        a_log: *const c_void,
        dt_bias: *const c_void,
        conv_state: *mut c_void,
        query: *mut c_void,
        key: *mut c_void,
        value: *mut c_void,
        alpha: *mut c_void,
        beta: *mut c_void,
        kq: *mut c_void,
        state_capacity: c_int,
        state_slot: c_int,
        tokens: c_int,
        row_offset: c_int,
        mixed_stride: c_int,
        gate_stride: c_int,
        stream: Stream,
    ) -> c_int;

    pub fn qwc_gather_rows_bf16(
        source: *const c_void,
        row_indices: *const c_void,
        destination: *mut c_void,
        rows: c_int,
        cols: c_int,
        stream: Stream,
    ) -> c_int;

    pub fn qwc_delta_gated_rmsnorm(
        input: *const f32,
        gate: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        batch: c_int,
        epsilon: f32,
        gate_stride: c_int,
        stream: Stream,
    ) -> c_int;

    // NVFP4 weight-only decode projection из cuda/nvfp4.cu.
    #[allow(clippy::too_many_arguments)]
    pub fn qwc_nvfp4_w4a16(
        packed: *const c_void,
        scales: *const c_void,
        input: *const c_void,
        output: *mut c_void,
        out_features: c_int,
        in_features: c_int,
        batch: c_int,
        weight_global_scale: f32,
        stream: Stream,
    ) -> c_int;

    /// Свиповая точка входа: геометрия задаётся снаружи. Движок её не зовёт.
    #[allow(clippy::too_many_arguments)]
    pub fn qwc_nvfp4_w4a16_tuned(
        packed: *const c_void,
        scales: *const c_void,
        input: *const c_void,
        output: *mut c_void,
        out_features: c_int,
        in_features: c_int,
        batch: c_int,
        k_rows: c_int,
        k_threads: c_int,
        k_tile: c_int,
        rows_per_thread: c_int,
        weight_global_scale: f32,
        stream: Stream,
    ) -> c_int;

    #[allow(clippy::too_many_arguments)]
    pub fn qwc_nvfp4_swiglu_w4a16(
        gate_packed: *const c_void,
        gate_scales: *const c_void,
        up_packed: *const c_void,
        up_scales: *const c_void,
        input: *const c_void,
        output: *mut c_void,
        out_features: c_int,
        in_features: c_int,
        batch: c_int,
        gate_global_scale: f32,
        up_global_scale: f32,
        stream: Stream,
    ) -> c_int;

    pub fn qwc_nvfp4_w4a4_workspace_size(
        batch: c_int,
        out_features: c_int,
        in_features: c_int,
        bytes: *mut usize,
    ) -> c_int;

    #[allow(clippy::too_many_arguments)]
    pub fn qwc_nvfp4_w4a4(
        packed_input: *const c_void,
        packed_weight: *const c_void,
        input_scales: *const c_void,
        weight_scales: *const c_void,
        output: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        batch: c_int,
        out_features: c_int,
        in_features: c_int,
        alpha: f32,
        stream: Stream,
    ) -> c_int;

    pub fn qwc_nvfp4_quantize_bf16(
        input: *const c_void,
        packed: *mut c_void,
        scales: *mut c_void,
        batch: c_int,
        in_features: c_int,
        global_scale: f32,
        stream: Stream,
    ) -> c_int;

    pub fn qwc_swiglu_bf16(
        gate: *const c_void,
        up: *const c_void,
        output: *mut c_void,
        elements: c_int,
        stream: Stream,
    ) -> c_int;

    #[allow(clippy::too_many_arguments)]
    pub fn qwc_rmsnorm_bf16(
        input: *const c_void,
        residual: *mut c_void,
        weight: *const c_void,
        output: *mut c_void,
        batch: c_int,
        hidden: c_int,
        epsilon: f32,
        stream: Stream,
    ) -> c_int;

    #[allow(clippy::too_many_arguments)]
    pub fn qwc_rmsnorm_nvfp4(
        input: *const c_void,
        residual: *mut c_void,
        weight: *const c_void,
        packed: *mut c_void,
        scales: *mut c_void,
        batch: c_int,
        hidden: c_int,
        epsilon: f32,
        global_scale: f32,
        stream: Stream,
    ) -> c_int;

    #[allow(clippy::too_many_arguments)]
    pub fn qwc_paged_attention_fp8(
        query: *const c_void,
        query_gate_projection: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        block_tables: *const c_void,
        context_lengths: *const c_void,
        output: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        batch: c_int,
        max_blocks: c_int,
        partitions: c_int,
        share_kv: c_int,
        softmax_scale: f32,
        stream: Stream,
    ) -> c_int;

    #[allow(clippy::too_many_arguments)]
    pub fn qwc_paged_attention_bf16(
        query: *const c_void,
        query_gate_projection: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        block_tables: *const c_void,
        context_lengths: *const c_void,
        output: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        batch: c_int,
        max_blocks: c_int,
        partitions: c_int,
        share_kv: c_int,
        softmax_scale: f32,
        stream: Stream,
    ) -> c_int;

    #[allow(clippy::too_many_arguments)]
    pub fn qwc_paged_attention_prefill_mma_bf16(
        query: *const c_void,
        query_gate_projection: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        block_tables: *const c_void,
        context_lengths: *const c_void,
        output: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        rows: c_int,
        row_base: c_int,
        max_blocks: c_int,
        partitions: c_int,
        partition_tokens: c_int,
        softmax_scale: f32,
        stream: Stream,
    ) -> c_int;

    #[allow(clippy::too_many_arguments)]
    pub fn qwc_paged_attention_packed_fp8(
        query: *const c_void,
        query_gate_projection: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        block_tables: *const c_void,
        context_lengths: *const c_void,
        output: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        rows: c_int,
        row_base: c_int,
        max_blocks: c_int,
        tile_size: c_int,
        partitions: c_int,
        softmax_scale: f32,
        stream: Stream,
    ) -> c_int;

    #[allow(clippy::too_many_arguments)]
    pub fn qwc_paged_attention_prefill_mma_fp8(
        query: *const c_void,
        query_gate_projection: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        block_tables: *const c_void,
        context_lengths: *const c_void,
        output: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        rows: c_int,
        row_base: c_int,
        max_blocks: c_int,
        partitions: c_int,
        partition_tokens: c_int,
        softmax_scale: f32,
        stream: Stream,
    ) -> c_int;

    #[allow(clippy::too_many_arguments)]
    pub fn qwc_paged_attention_prefill_bf16(
        query: *const c_void,
        query_gate_projection: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        block_tables: *const c_void,
        context_lengths: *const c_void,
        output: *mut c_void,
        rows: c_int,
        row_base: c_int,
        max_blocks: c_int,
        softmax_scale: f32,
        stream: Stream,
    ) -> c_int;

    #[allow(clippy::too_many_arguments)]
    pub fn qwc_paged_attention_prefill_fp8(
        query: *const c_void,
        query_gate_projection: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        block_tables: *const c_void,
        context_lengths: *const c_void,
        output: *mut c_void,
        rows: c_int,
        row_base: c_int,
        max_blocks: c_int,
        softmax_scale: f32,
        stream: Stream,
    ) -> c_int;

    #[allow(clippy::too_many_arguments)]
    pub fn qwc_prepare_attention_decode_fp8(
        query_gate_projection: *const c_void,
        key_projection: *const c_void,
        value_projection: *const c_void,
        query_norm_weight: *const c_void,
        key_norm_weight: *const c_void,
        cosine: *const c_void,
        sine: *const c_void,
        physical_blocks: *const c_void,
        block_offsets: *const c_void,
        query: *mut c_void,
        key_cache: *mut c_void,
        value_cache: *mut c_void,
        batch: c_int,
        epsilon: f32,
        stream: Stream,
    ) -> c_int;

    #[allow(clippy::too_many_arguments)]
    pub fn qwc_prepare_attention_decode_bf16(
        query_gate_projection: *const c_void,
        key_projection: *const c_void,
        value_projection: *const c_void,
        query_norm_weight: *const c_void,
        key_norm_weight: *const c_void,
        cosine: *const c_void,
        sine: *const c_void,
        physical_blocks: *const c_void,
        block_offsets: *const c_void,
        query: *mut c_void,
        key_cache: *mut c_void,
        value_cache: *mut c_void,
        batch: c_int,
        epsilon: f32,
        stream: Stream,
    ) -> c_int;

    pub fn qwc_argmax(
        logits: *const f32,
        partial_values: *mut f32,
        partial_indices: *mut u32,
        output: *mut u32,
        vocab: c_int,
        batch: c_int,
        parts: c_int,
        stream: Stream,
    ) -> c_int;
}

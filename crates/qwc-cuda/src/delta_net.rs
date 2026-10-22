//! Gated DeltaNet: the recurrent decode step.
//!
//! Implements 48 of Qwen3.8's 64 layers. Details of the math and layout are
//! in `cuda/delta_net.cu`.

use crate::error::{Result, check};
use crate::{DeviceBuffer, Stream, ffi};
use qwc_core::arch::{LA_K_HEAD_DIM, LA_NUM_K_HEADS, LA_NUM_V_HEADS, LA_V_HEAD_DIM};

/// State elements per sequence in one layer.
pub const STATE_ELEMS: usize = LA_NUM_V_HEADS * LA_V_HEAD_DIM * LA_K_HEAD_DIM;
/// q or k elements per sequence.
pub const QK_ELEMS: usize = LA_NUM_K_HEADS * LA_K_HEAD_DIM;
/// v or output elements per sequence.
pub const V_ELEMS: usize = LA_NUM_V_HEADS * LA_V_HEAD_DIM;
/// Gates (alpha, beta) per sequence.
pub const GATE_ELEMS: usize = LA_NUM_V_HEADS;
/// k.q scalars per sequence: one per k head.
pub const KQ_ELEMS: usize = LA_NUM_K_HEADS;
/// FP32 causal-convolution history per persistent sequence slot and layer.
pub const CONV_STATE_ELEMS: usize = qwc_core::arch::LA_CONV_CHANNELS * 3;

/// Matrix tile used by the parallel WY prefill scan.
pub const WY_CHUNK_SIZE: usize = 64;

/// How a prefill chunk is computed.
///
/// `Bf16` and `Fp32` are the same recurrent pass over tokens with different
/// state precision. `Bf16` rounds the state after every token,
/// so a chunk reproduces repeated decode bit for bit. `Fp32` keeps
/// the state row in registers until the end of the chunk and rounds only the write at
/// the boundary — that's what the reference implementation does (flash-linear-attention accumulates
/// `b_h` in FP32 across all chunks). Their A/B isolates per-token rounding as
/// a source of divergence.
///
/// `Wy` is a different algorithm, not a different precision: a chunk is computed with matrices
/// on tensor cores (`bench/results/deltanet-wy-2026-09-16.md`). The state
/// within a chunk accumulates in FP32, so in terms of rounding `Wy` is closest to
/// `Fp32`. It's the fastest path on prompts longer than a couple of chunks and requires
/// `DeltaPrefillWorkspace`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeltaStateMode {
    Bf16,
    Fp32,
    Wy,
}

impl DeltaStateMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Fp32 => "fp32",
            Self::Wy => "wy",
        }
    }

    /// Only for the recurrent path: the WY scan doesn't round tokens one by one,
    /// and calling this on it is a caller error, not a default mode.
    fn round_per_token(self) -> i32 {
        match self {
            Self::Bf16 => 1,
            Self::Fp32 => 0,
            Self::Wy => unreachable!("the WY scan doesn't round the state per token"),
        }
    }
}

/// Chunks in the longest step: state-independent phases are computed
/// for all of them at once, so their buffers are sized for the full arena.
const WY_MAX_CHUNKS: usize = crate::MAX_STEP_ROWS / WY_CHUNK_SIZE;

const WY_QK_TILE_ELEMS: usize = WY_MAX_CHUNKS * LA_NUM_K_HEADS * WY_CHUNK_SIZE * LA_K_HEAD_DIM;
const WY_GRAM_ELEMS: usize = WY_MAX_CHUNKS * LA_NUM_K_HEADS * WY_CHUNK_SIZE * WY_CHUNK_SIZE;
const WY_HEAD_MATRIX_ELEMS: usize = WY_MAX_CHUNKS * GATE_ELEMS * WY_CHUNK_SIZE * WY_CHUNK_SIZE;
const WY_DECAY_ELEMS: usize = WY_MAX_CHUNKS * GATE_ELEMS * WY_CHUNK_SIZE;
const WY_VALUE_TILE_ELEMS: usize = WY_MAX_CHUNKS * GATE_ELEMS * WY_CHUNK_SIZE * LA_V_HEAD_DIM;

/// Resident scratch for the matrix (WY) prefill scan.
///
/// It is deliberately owned by the executor rather than allocated by the
/// launch.  A layer step invokes the scan up to 48 times and runtime CUDA
/// allocations would serialize that path.
///
/// The tile is fixed at 64 tokens, but state-independent phases
/// are computed for all chunks of a segment at once — so the buffers are sized for
/// the full arena (`MAX_STEP_ROWS`) and cost about 84 MB.
pub struct DeltaPrefillWorkspace {
    query: DeviceBuffer<u16>,
    key: DeviceBuffer<u16>,
    gram_kk: DeviceBuffer<f32>,
    gram_qk: DeviceBuffer<f32>,
    inverse: DeviceBuffer<u16>,
    output_factor: DeviceBuffer<u16>,
    value_tile: DeviceBuffer<u16>,
    log_decay: DeviceBuffer<f32>,
}

impl DeltaPrefillWorkspace {
    pub fn new() -> Result<Self> {
        Ok(Self {
            query: DeviceBuffer::zeroed(WY_QK_TILE_ELEMS)?,
            key: DeviceBuffer::zeroed(WY_QK_TILE_ELEMS)?,
            gram_kk: DeviceBuffer::zeroed(WY_GRAM_ELEMS)?,
            gram_qk: DeviceBuffer::zeroed(WY_GRAM_ELEMS)?,
            inverse: DeviceBuffer::zeroed(WY_HEAD_MATRIX_ELEMS)?,
            output_factor: DeviceBuffer::zeroed(WY_HEAD_MATRIX_ELEMS)?,
            value_tile: DeviceBuffer::zeroed(WY_VALUE_TILE_ELEMS)?,
            log_decay: DeviceBuffer::zeroed(WY_DECAY_ELEMS)?,
        })
    }

    /// Bytes that `Wy` mode adds to the engine's memory budget.
    pub fn bytes(&self) -> usize {
        self.query.bytes()
            + self.key.bytes()
            + self.gram_kk.bytes()
            + self.gram_qk.bytes()
            + self.inverse.bytes()
            + self.output_factor.bytes()
            + self.value_tile.bytes()
            + self.log_decay.bytes()
    }
}

pub struct PreparedDelta {
    pub q: DeviceBuffer<f32>,
    pub k: DeviceBuffer<f32>,
    pub v: DeviceBuffer<f32>,
    pub alpha: DeviceBuffer<f32>,
    pub beta: DeviceBuffer<f32>,
    /// k.q per token and k head: shared by all state rows, so
    /// computed once in prepare, not in every scan warp.
    pub kq: DeviceBuffer<f32>,
    batch: usize,
}

impl PreparedDelta {
    pub fn zeroed(batch: usize) -> Result<Self> {
        assert!((1..=crate::MAX_STEP_ROWS).contains(&batch));
        Ok(Self {
            q: DeviceBuffer::zeroed(batch * QK_ELEMS)?,
            k: DeviceBuffer::zeroed(batch * QK_ELEMS)?,
            v: DeviceBuffer::zeroed(batch * V_ELEMS)?,
            alpha: DeviceBuffer::zeroed(batch * GATE_ELEMS)?,
            beta: DeviceBuffer::zeroed(batch * GATE_ELEMS)?,
            kq: DeviceBuffer::zeroed(batch * KQ_ELEMS)?,
            batch,
        })
    }

    pub fn inputs(&self) -> DeltaInputs<'_> {
        DeltaInputs {
            q: &self.q,
            k: &self.k,
            v: &self.v,
            alpha: &self.alpha,
            beta: &self.beta,
            kq: &self.kq,
        }
    }
}

pub struct DeltaPreprocessor {
    conv_weight: DeviceBuffer<u16>,
    a_log: DeviceBuffer<u16>,
    dt_bias: DeviceBuffer<u16>,
}

impl DeltaPreprocessor {
    pub fn from_host(conv_weight: &[u16], a_log: &[u16], dt_bias: &[u16]) -> Result<Self> {
        assert_eq!(conv_weight.len(), qwc_core::arch::LA_CONV_CHANNELS * 4);
        assert_eq!(a_log.len(), GATE_ELEMS);
        assert_eq!(dt_bias.len(), GATE_ELEMS);
        Ok(Self {
            conv_weight: DeviceBuffer::from_slice(conv_weight)?,
            a_log: DeviceBuffer::from_slice(a_log)?,
            dt_bias: DeviceBuffer::from_slice(dt_bias)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prepare_decode(
        &self,
        mixed_qkv: RowView<'_>,
        a_projection: RowView<'_>,
        b_projection: RowView<'_>,
        conv_state_pool: &mut DeviceBuffer<f32>,
        state_slots: &DeviceBuffer<u32>,
        output: &mut PreparedDelta,
        state_capacity: usize,
        batch: usize,
        stream: &Stream,
    ) -> Result<()> {
        assert!(batch <= output.batch);
        assert!(mixed_qkv.fits(batch, qwc_core::arch::LA_CONV_CHANNELS));
        assert!(a_projection.fits(batch, GATE_ELEMS));
        assert!(b_projection.fits(batch, GATE_ELEMS));
        assert_eq!(a_projection.stride(), b_projection.stride());
        assert_eq!(conv_state_pool.len(), state_capacity * CONV_STATE_ELEMS);
        assert!(state_slots.len() >= batch);
        check(unsafe {
            ffi::qwc_delta_prepare_decode(
                mixed_qkv.as_ptr(),
                a_projection.as_ptr(),
                b_projection.as_ptr(),
                self.conv_weight.as_ptr(),
                self.a_log.as_ptr(),
                self.dt_bias.as_ptr(),
                conv_state_pool.as_mut_ptr(),
                state_slots.as_ptr(),
                output.q.as_mut_ptr(),
                output.k.as_mut_ptr(),
                output.v.as_mut_ptr(),
                output.alpha.as_mut_ptr(),
                output.beta.as_mut_ptr(),
                output.kq.as_mut_ptr(),
                state_capacity as i32,
                batch as i32,
                mixed_qkv.stride() as i32,
                a_projection.stride() as i32,
                stream.raw(),
            )
        })
    }

    /// Prepares a causal chunk belonging to one persistent sequence slot.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_prefill(
        &self,
        mixed_qkv: RowView<'_>,
        a_projection: RowView<'_>,
        b_projection: RowView<'_>,
        conv_state_pool: &mut DeviceBuffer<f32>,
        output: &mut PreparedDelta,
        state_capacity: usize,
        state_slot: usize,
        tokens: usize,
        row_offset: usize,
        stream: &Stream,
    ) -> Result<()> {
        assert!(tokens > 0 && row_offset + tokens <= output.batch);
        assert!(state_slot < state_capacity);
        // Check the rows the kernel actually touches, not the whole
        // arena capacity: replaying accepted drafts feeds input from
        // a small saved buffer of a few rows.
        let used = row_offset + tokens;
        assert!(mixed_qkv.fits(used, qwc_core::arch::LA_CONV_CHANNELS));
        assert!(a_projection.fits(used, GATE_ELEMS));
        assert!(b_projection.fits(used, GATE_ELEMS));
        assert_eq!(a_projection.stride(), b_projection.stride());
        assert_eq!(conv_state_pool.len(), state_capacity * CONV_STATE_ELEMS);
        check(unsafe {
            ffi::qwc_delta_prepare_prefill(
                mixed_qkv.as_ptr(),
                a_projection.as_ptr(),
                b_projection.as_ptr(),
                self.conv_weight.as_ptr(),
                self.a_log.as_ptr(),
                self.dt_bias.as_ptr(),
                conv_state_pool.as_mut_ptr(),
                output.q.as_mut_ptr(),
                output.k.as_mut_ptr(),
                output.v.as_mut_ptr(),
                output.alpha.as_mut_ptr(),
                output.beta.as_mut_ptr(),
                output.kq.as_mut_ptr(),
                state_capacity as i32,
                state_slot as i32,
                tokens as i32,
                row_offset as i32,
                mixed_qkv.stride() as i32,
                a_projection.stride() as i32,
                stream.raw(),
            )
        })
    }
}

/// A slice of the projection arena: rows go with stride `stride`, starting at `offset`.
///
/// The fused mixer projection puts qkv, z, a and b into one matrix, so
/// consumers get not a separate buffer but a view onto its columns. `packed`
/// describes the old case — a separate dense buffer.
#[derive(Clone, Copy)]
pub struct RowView<'a> {
    buffer: &'a DeviceBuffer<u16>,
    offset: usize,
    stride: usize,
}

impl<'a> RowView<'a> {
    pub fn packed(buffer: &'a DeviceBuffer<u16>, width: usize) -> Self {
        Self {
            buffer,
            offset: 0,
            stride: width,
        }
    }

    pub fn strided(buffer: &'a DeviceBuffer<u16>, offset: usize, stride: usize) -> Self {
        assert!(offset < stride, "slice starts beyond the row");
        Self {
            buffer,
            offset,
            stride,
        }
    }

    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Whether `rows` rows of width `width` fit.
    pub fn fits(&self, rows: usize, width: usize) -> bool {
        assert!(self.offset + width <= self.stride, "slice is wider than the row");
        rows == 0 || self.offset + (rows - 1) * self.stride + width <= self.buffer.len()
    }

    fn as_ptr(&self) -> *const std::ffi::c_void {
        // SAFETY: the offset within the buffer is checked by the caller's `fits`.
        unsafe { (self.buffer.as_ptr() as *const u16).add(self.offset) as *const _ }
    }
}

pub struct DeltaOutputNorm {
    weight: DeviceBuffer<u16>,
    epsilon: f32,
}

impl DeltaOutputNorm {
    /// Unlike Qwen3.5's other norms, RMSNormGated has a direct (one-centered)
    /// weight and must not add one.
    pub fn from_host(weight: &[u16], epsilon: f32) -> Result<Self> {
        assert_eq!(weight.len(), LA_V_HEAD_DIM);
        assert!(epsilon.is_finite() && epsilon > 0.0);
        Ok(Self {
            weight: DeviceBuffer::from_slice(weight)?,
            epsilon,
        })
    }

    pub fn forward(
        &self,
        input: &DeviceBuffer<f32>,
        gate: RowView<'_>,
        output: &mut DeviceBuffer<u16>,
        batch: usize,
        stream: &Stream,
    ) -> Result<()> {
        assert!(input.len() >= batch * V_ELEMS);
        assert!(gate.fits(batch, V_ELEMS));
        assert!(output.len() >= batch * V_ELEMS);
        check(unsafe {
            ffi::qwc_delta_gated_rmsnorm(
                input.as_ptr().cast(),
                gate.as_ptr(),
                self.weight.as_ptr(),
                output.as_mut_ptr(),
                batch as i32,
                self.epsilon,
                gate.stride() as i32,
                stream.raw(),
            )
        })
    }
}

pub struct DeltaInputs<'a> {
    /// L2-normalized, [batch, 16, 128].
    pub q: &'a DeviceBuffer<f32>,
    pub k: &'a DeviceBuffer<f32>,
    /// [batch, 48, 128].
    pub v: &'a DeviceBuffer<f32>,
    /// Decay exp(g) in (0, 1), [batch, 48].
    pub alpha: &'a DeviceBuffer<f32>,
    /// Write strength sigmoid(b), [batch, 48].
    pub beta: &'a DeviceBuffer<f32>,
    /// Scalar k.q, [batch, 16].
    pub kq: &'a DeviceBuffer<f32>,
}

/// One decode step. The state is updated in place.
pub fn decode(
    state: &mut DeviceBuffer<u16>,
    inputs: &DeltaInputs<'_>,
    out: &mut DeviceBuffer<f32>,
    batch: usize,
    stream: &Stream,
) -> Result<()> {
    debug_assert_eq!(state.len(), batch * STATE_ELEMS);
    debug_assert_eq!(inputs.q.len(), batch * QK_ELEMS);
    debug_assert_eq!(out.len(), batch * V_ELEMS);
    check(unsafe {
        ffi::qwc_delta_decode(
            state.as_mut_ptr(),
            std::ptr::null(),
            inputs.q.as_ptr().cast(),
            inputs.k.as_ptr().cast(),
            inputs.v.as_ptr().cast(),
            inputs.alpha.as_ptr().cast(),
            inputs.beta.as_ptr().cast(),
            inputs.kq.as_ptr().cast(),
            out.as_mut_ptr().cast(),
            batch as i32,
            batch as i32,
            stream.raw(),
        )
    })
}

/// Decode against the persistent state pool selected by scheduler slot IDs.
/// No state is gathered into a compact batch buffer.
pub fn decode_slots(
    state_pool: &mut DeviceBuffer<u16>,
    state_slots: &DeviceBuffer<u32>,
    inputs: &DeltaInputs<'_>,
    out: &mut DeviceBuffer<f32>,
    state_capacity: usize,
    batch: usize,
    stream: &Stream,
) -> Result<()> {
    assert_eq!(state_pool.len(), state_capacity * STATE_ELEMS);
    assert!(state_slots.len() >= batch);
    // The caller may pass the leading rows of a larger fused arena.
    debug_assert!(inputs.q.len() >= batch * QK_ELEMS);
    debug_assert!(inputs.k.len() >= batch * QK_ELEMS);
    debug_assert!(inputs.v.len() >= batch * V_ELEMS);
    debug_assert!(inputs.alpha.len() >= batch * GATE_ELEMS);
    debug_assert!(inputs.beta.len() >= batch * GATE_ELEMS);
    debug_assert!(out.len() >= batch * V_ELEMS);
    check(unsafe {
        ffi::qwc_delta_decode(
            state_pool.as_mut_ptr(),
            state_slots.as_ptr(),
            inputs.q.as_ptr().cast(),
            inputs.k.as_ptr().cast(),
            inputs.v.as_ptr().cast(),
            inputs.alpha.as_ptr().cast(),
            inputs.beta.as_ptr().cast(),
            inputs.kq.as_ptr().cast(),
            out.as_mut_ptr().cast(),
            state_capacity as i32,
            batch as i32,
            stream.raw(),
        )
    })
}

/// Causal recurrent scan over one chunk of one persistent sequence.
#[allow(clippy::too_many_arguments)]
pub fn prefill_slot(
    state_pool: &mut DeviceBuffer<u16>,
    inputs: &DeltaInputs<'_>,
    out: &mut DeviceBuffer<f32>,
    state_capacity: usize,
    state_slot: usize,
    tokens: usize,
    row_offset: usize,
    state_mode: DeltaStateMode,
    stream: &Stream,
) -> Result<()> {
    assert_eq!(state_pool.len(), state_capacity * STATE_ELEMS);
    assert!(state_slot < state_capacity);
    assert!(tokens > 0);
    let end = row_offset + tokens;
    assert!(inputs.q.len() >= end * QK_ELEMS);
    assert!(inputs.k.len() >= end * QK_ELEMS);
    assert!(inputs.v.len() >= end * V_ELEMS);
    assert!(inputs.alpha.len() >= end * GATE_ELEMS);
    assert!(inputs.beta.len() >= end * GATE_ELEMS);
    assert!(inputs.kq.len() >= end * KQ_ELEMS);
    assert!(out.len() >= end * V_ELEMS);
    check(unsafe {
        ffi::qwc_delta_prefill(
            state_pool.as_mut_ptr(),
            inputs.q.as_ptr().cast(),
            inputs.k.as_ptr().cast(),
            inputs.v.as_ptr().cast(),
            inputs.alpha.as_ptr().cast(),
            inputs.beta.as_ptr().cast(),
            inputs.kq.as_ptr().cast(),
            out.as_mut_ptr().cast(),
            state_capacity as i32,
            state_slot as i32,
            tokens as i32,
            state_mode.round_per_token(),
            row_offset as i32,
            stream.raw(),
        )
    })
}

/// Chunked (WY) scan: `DeltaStateMode::Wy`.
///
/// The recurrent kernel remains the implementation of `Bf16` and `Fp32`: rounding after
/// every token is itself a recurrence, it can't be expressed in matrix form.
/// So this is a separate entry point, not a flag inside
/// `prefill_slot`.
#[allow(clippy::too_many_arguments)]
pub fn prefill_slot_wy(
    state_pool: &mut DeviceBuffer<u16>,
    inputs: &DeltaInputs<'_>,
    out: &mut DeviceBuffer<f32>,
    workspace: &mut DeltaPrefillWorkspace,
    state_capacity: usize,
    state_slot: usize,
    tokens: usize,
    row_offset: usize,
    stream: &Stream,
) -> Result<()> {
    assert_eq!(state_pool.len(), state_capacity * STATE_ELEMS);
    assert!(state_slot < state_capacity);
    assert!(tokens > 0);
    let end = row_offset + tokens;
    assert!(inputs.q.len() >= end * QK_ELEMS);
    assert!(inputs.k.len() >= end * QK_ELEMS);
    assert!(inputs.v.len() >= end * V_ELEMS);
    assert!(inputs.alpha.len() >= end * GATE_ELEMS);
    assert!(inputs.beta.len() >= end * GATE_ELEMS);
    assert!(out.len() >= end * V_ELEMS);
    check(unsafe {
        ffi::qwc_delta_prefill_wy(
            state_pool.as_mut_ptr(),
            inputs.q.as_ptr().cast(),
            inputs.k.as_ptr().cast(),
            inputs.v.as_ptr().cast(),
            inputs.alpha.as_ptr().cast(),
            inputs.beta.as_ptr().cast(),
            out.as_mut_ptr().cast(),
            workspace.query.as_mut_ptr(),
            workspace.key.as_mut_ptr(),
            workspace.gram_kk.as_mut_ptr(),
            workspace.gram_qk.as_mut_ptr(),
            workspace.inverse.as_mut_ptr(),
            workspace.output_factor.as_mut_ptr(),
            workspace.value_tile.as_mut_ptr(),
            workspace.log_decay.as_mut_ptr(),
            state_capacity as i32,
            state_slot as i32,
            tokens as i32,
            row_offset as i32,
            stream.raw(),
        )
    })
}

/// Scales per slot and layer: one per state row.
pub const STATE_SCALE_ELEMS: usize = LA_NUM_V_HEADS * LA_V_HEAD_DIM;

/// Eight-bit state storage for one layer: int8 plus a per-row scale.
///
/// Data and scales are kept together not for convenience: a row without its
/// scale is garbage, and two parallel `DeviceBuffer`s in the executor would sooner or
/// later drift apart in capacity.
///
/// Prefill doesn't write here. The prompt is computed on a dequantized bf16 copy and
/// packed once at the end: restarting a prefill segment from the 8-bit
/// state spoils the whole rest of the prompt
/// (`bench/results/delta-state-8bit.md`).
pub struct PackedStatePool {
    data: DeviceBuffer<u8>,
    scales: DeviceBuffer<f32>,
    capacity: usize,
}

impl PackedStatePool {
    pub fn zeroed(capacity: usize) -> Result<Self> {
        assert!(capacity > 0);
        Ok(Self {
            data: DeviceBuffer::zeroed(capacity * STATE_ELEMS)?,
            scales: DeviceBuffer::zeroed(capacity * STATE_SCALE_ELEMS)?,
            capacity,
        })
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn bytes(&self) -> usize {
        self.data.bytes() + self.scales.bytes()
    }

    /// Zeroes a slot. A zero row is representable at any scale, so
    /// scales are zeroed along with the data rather than set to one.
    pub fn reset_slot(&mut self, slot: usize) -> Result<()> {
        assert!(slot < self.capacity);
        self.data.zero_range(slot * STATE_ELEMS, STATE_ELEMS)?;
        self.scales
            .zero_range(slot * STATE_SCALE_ELEMS, STATE_SCALE_ELEMS)
    }

    /// Zeroes the whole pool.
    pub fn reset(&mut self) -> Result<()> {
        for slot in 0..self.capacity {
            self.reset_slot(slot)?;
        }
        Ok(())
    }

    /// Copies a whole slot — both data and scales. The argument order
    /// here is "from, to", while `copy_within` has it the other way around, destination first.
    pub fn copy_slot(&mut self, from: usize, to: usize, stream: &Stream) -> Result<()> {
        assert!(from < self.capacity && to < self.capacity);
        self.data
            .copy_within(to * STATE_ELEMS, from * STATE_ELEMS, STATE_ELEMS, stream)?;
        self.scales.copy_within(
            to * STATE_SCALE_ELEMS,
            from * STATE_SCALE_ELEMS,
            STATE_SCALE_ELEMS,
            stream,
        )
    }
}

/// One decode step over the 8-bit state.
pub fn decode_slots_packed(
    pool: &mut PackedStatePool,
    state_slots: &DeviceBuffer<u32>,
    inputs: &DeltaInputs<'_>,
    out: &mut DeviceBuffer<f32>,
    batch: usize,
    stream: &Stream,
) -> Result<()> {
    assert!(state_slots.len() >= batch);
    debug_assert!(inputs.q.len() >= batch * QK_ELEMS);
    debug_assert!(inputs.k.len() >= batch * QK_ELEMS);
    debug_assert!(inputs.v.len() >= batch * V_ELEMS);
    debug_assert!(inputs.alpha.len() >= batch * GATE_ELEMS);
    debug_assert!(inputs.beta.len() >= batch * GATE_ELEMS);
    debug_assert!(out.len() >= batch * V_ELEMS);
    let capacity = pool.capacity;
    check(unsafe {
        ffi::qwc_delta_decode_int8(
            pool.data.as_mut_ptr().cast(),
            pool.scales.as_mut_ptr().cast(),
            state_slots.as_ptr().cast(),
            inputs.q.as_ptr().cast(),
            inputs.k.as_ptr().cast(),
            inputs.v.as_ptr().cast(),
            inputs.alpha.as_ptr().cast(),
            inputs.beta.as_ptr().cast(),
            inputs.kq.as_ptr().cast(),
            out.as_mut_ptr().cast(),
            capacity as i32,
            batch as i32,
            stream.raw(),
        )
    })
}

/// bf16 copy -> 8-bit slot.
pub fn pack_state_slot(
    source: &DeviceBuffer<u16>,
    source_capacity: usize,
    source_slot: usize,
    pool: &mut PackedStatePool,
    packed_slot: usize,
    stream: &Stream,
) -> Result<()> {
    assert_eq!(source.len(), source_capacity * STATE_ELEMS);
    assert!(source_slot < source_capacity);
    assert!(packed_slot < pool.capacity);
    let capacity = pool.capacity;
    check(unsafe {
        ffi::qwc_delta_state_pack(
            source.as_ptr(),
            pool.data.as_mut_ptr().cast(),
            pool.scales.as_mut_ptr().cast(),
            source_capacity as i32,
            source_slot as i32,
            capacity as i32,
            packed_slot as i32,
            stream.raw(),
        )
    })
}

/// 8-bit slot -> bf16 copy.
pub fn unpack_state_slot(
    pool: &PackedStatePool,
    packed_slot: usize,
    destination: &mut DeviceBuffer<u16>,
    destination_capacity: usize,
    destination_slot: usize,
    stream: &Stream,
) -> Result<()> {
    assert_eq!(destination.len(), destination_capacity * STATE_ELEMS);
    assert!(packed_slot < pool.capacity);
    assert!(destination_slot < destination_capacity);
    check(unsafe {
        ffi::qwc_delta_state_unpack(
            pool.data.as_ptr(),
            pool.scales.as_ptr().cast(),
            destination.as_mut_ptr().cast(),
            pool.capacity as i32,
            packed_slot as i32,
            destination_capacity as i32,
            destination_slot as i32,
            stream.raw(),
        )
    })
}

/// Recurrent state storage width — for measurement, not for
/// savings: the pool stays bf16, and values are snapped onto the 8-bit grid after
/// every update. This way diff-eval answers the accuracy question before
/// three kernels, the pools and the budget are reworked.
///
/// `Int8` keeps a scale per state row and beats `E4m3` fourfold at the same
/// byte: rows are flat, and they don't need four exponent bits
/// (`bench/results/delta-state-8bit.md`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum StateQuant {
    #[default]
    Off,
    Int8,
    E4m3,
}

impl StateQuant {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "off" | "bf16" => Some(Self::Off),
            "int8" => Some(Self::Int8),
            "e4m3" | "fp8" => Some(Self::E4m3),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Int8 => "int8",
            Self::E4m3 => "e4m3",
        }
    }

    fn mode(self) -> i32 {
        match self {
            Self::Off => 0,
            Self::Int8 => 1,
            Self::E4m3 => 2,
        }
    }
}

/// Snaps the state of the listed slots onto the 8-bit grid.
pub fn requantize_state_slots(
    state_pool: &mut DeviceBuffer<u16>,
    state_slots: &DeviceBuffer<u32>,
    quant: StateQuant,
    state_capacity: usize,
    batch: usize,
    stream: &Stream,
) -> Result<()> {
    if quant == StateQuant::Off || batch == 0 {
        return Ok(());
    }
    assert_eq!(state_pool.len(), state_capacity * STATE_ELEMS);
    assert!(state_slots.len() >= batch);
    check(unsafe {
        ffi::qwc_delta_state_requantize(
            state_pool.as_mut_ptr(),
            state_slots.as_ptr().cast(),
            0,
            state_capacity as i32,
            batch as i32,
            quant.mode(),
            stream.raw(),
        )
    })
}

/// The same for one slot: prefill always has one segment.
pub fn requantize_state_slot(
    state_pool: &mut DeviceBuffer<u16>,
    quant: StateQuant,
    state_capacity: usize,
    state_slot: usize,
    stream: &Stream,
) -> Result<()> {
    if quant == StateQuant::Off {
        return Ok(());
    }
    assert_eq!(state_pool.len(), state_capacity * STATE_ELEMS);
    assert!(state_slot < state_capacity);
    check(unsafe {
        ffi::qwc_delta_state_requantize(
            state_pool.as_mut_ptr(),
            std::ptr::null(),
            state_slot as i32,
            state_capacity as i32,
            1,
            quant.mode(),
            stream.raw(),
        )
    })
}

/// Memory traffic of one call: the state is read and rewritten in full.
pub fn traffic_bytes(batch: usize) -> u64 {
    2 * (batch * STATE_ELEMS * std::mem::size_of::<u16>()) as u64
}

/// CPU reference. Works with the same bf16 state representation as the GPU,
/// so a mismatch can only come from summation order.
pub mod reference {
    use super::*;
    use crate::bf16;

    // The reference repeats the kernel signature one to one: grouping the arguments
    // into a struct would hide divergence from the GPU path.
    #[allow(clippy::too_many_arguments)]
    pub fn decode(
        state: &mut [u16],
        q: &[f32],
        k: &[f32],
        v: &[f32],
        alpha: &[f32],
        beta: &[f32],
        out: &mut [f32],
        batch: usize,
    ) {
        const DK: usize = LA_K_HEAD_DIM;
        const DV: usize = LA_V_HEAD_DIM;
        const HV: usize = LA_NUM_V_HEADS;
        const RATIO: usize = LA_NUM_V_HEADS / LA_NUM_K_HEADS;

        for b in 0..batch {
            for h in 0..HV {
                let hk = h / RATIO;
                let kv = &k[(b * LA_NUM_K_HEADS + hk) * DK..][..DK];
                let qv = &q[(b * LA_NUM_K_HEADS + hk) * DK..][..DK];
                let kq: f32 = kv.iter().zip(qv).map(|(a, b)| a * b).sum();

                let a = alpha[b * HV + h];
                let bt = beta[b * HV + h];
                let s = &mut state[(b * HV + h) * DV * DK..][..DV * DK];
                let vv = &v[(b * HV + h) * DV..][..DV];
                let o = &mut out[(b * HV + h) * DV..][..DV];

                for row in 0..DV {
                    let sr = &mut s[row * DK..][..DK];
                    let mut u = 0.0f32;
                    let mut w = 0.0f32;
                    for j in 0..DK {
                        let x = bf16::to_f32(sr[j]);
                        u += x * kv[j];
                        w += x * qv[j];
                    }
                    let c = bt * (vv[row] - a * u);
                    o[row] = a * w + c * kq;
                    for j in 0..DK {
                        let x = bf16::to_f32(sr[j]);
                        sr[j] = bf16::from_f32(a * x + c * kv[j]);
                    }
                }
            }
        }
    }

    /// Scan in the chunked (WY) form: the recurrence over tokens is replaced by
    /// matrices within a chunk of length `chunk`.
    ///
    /// Derivation. Let `S_t` be the state after token t, gamma_t the
    /// product of alpha over the chunk's tokens up to and including t. From
    /// `S_t = a_t S_{t-1} + c_t k_t^T` by induction it follows that
    /// `S_t = gamma_t (S_0 + sum_{r<=t} (c_r / gamma_r) k_r^T)`, and substituting
    /// this into `c_t = b_t (v_t - a_t S_{t-1} k_t)` gives a triangular system
    ///
    ///   c_t + sum_{r<t} A[t][r] c_r = b_t v_t - b_t gamma_t (S_0 k_t),
    ///   A[t][r] = b_t (gamma_t / gamma_r) (k_r . k_t).
    ///
    /// The output and the final state are matrices too:
    ///
    ///   out_t = gamma_t (S_0 q_t) + sum_{r<=t} (gamma_t / gamma_r)(k_r . q_t) c_r,
    ///   S_C   = gamma_C S_0 + sum_t (gamma_C / gamma_t) c_t k_t^T.
    ///
    /// All multipliers are ratios of gamma over pairs (r <= t), and alpha is in (0, 1],
    /// so each of them is at most one. They still can't be computed by division:
    /// gamma itself is a product of up to 64 factors and underflows
    /// fp32 already at alpha around 0.2, after which the ratio turns into
    /// 0/0. So decay is accumulated in logarithms, and the ratio is taken as
    /// exp of the difference.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_chunked_fp32(
        state: &mut [u16],
        q: &[f32],
        k: &[f32],
        v: &[f32],
        alpha: &[f32],
        beta: &[f32],
        out: &mut [f32],
        tokens: usize,
        chunk: usize,
    ) {
        const DK: usize = LA_K_HEAD_DIM;
        const DV: usize = LA_V_HEAD_DIM;
        const HV: usize = LA_NUM_V_HEADS;
        const RATIO: usize = LA_NUM_V_HEADS / LA_NUM_K_HEADS;
        assert!(chunk > 0);

        let mut wide: Vec<f32> = state.iter().map(|&x| bf16::to_f32(x)).collect();
        for h in 0..HV {
            let hk = h / RATIO;
            let s = &mut wide[h * DV * DK..][..DV * DK];
            let mut start = 0usize;
            while start < tokens {
                let len = chunk.min(tokens - start);
                let key = |t: usize| &k[((start + t) * LA_NUM_K_HEADS + hk) * DK..][..DK];
                let query = |t: usize| &q[((start + t) * LA_NUM_K_HEADS + hk) * DK..][..DK];

                // Decay is accumulated in logarithms: the product of alpha over a chunk
                // underflows fp32 already at alpha around 0.2, and the ratios
                // gamma_t / gamma_r turn into 0/0. A difference of logarithms
                // is always finite.
                let mut cumulative = vec![0.0f32; len];
                let mut running = 0.0f32;
                for (t, slot) in cumulative.iter_mut().enumerate() {
                    running += alpha[(start + t) * HV + h].max(1.0e-38).ln();
                    *slot = running;
                }
                let gamma: Vec<f32> = cumulative.iter().map(|x| x.exp()).collect();
                let ratio = |t: usize, r: usize| (cumulative[t] - cumulative[r]).exp();

                // Right-hand side: b_t v_t - b_t gamma_t (S_0 k_t).
                let mut c = vec![0.0f32; len * DV];
                for t in 0..len {
                    let bt = beta[(start + t) * HV + h];
                    let kt = key(t);
                    for row in 0..DV {
                        let sk: f32 = s[row * DK..][..DK].iter().zip(kt).map(|(x, y)| x * y).sum();
                        c[t * DV + row] =
                            bt * (v[((start + t) * HV + h) * DV + row] - gamma[t] * sk);
                    }
                }
                // Forward substitution over the triangular system: row t sees
                // only the already computed r < t.
                for t in 0..len {
                    let bt = beta[(start + t) * HV + h];
                    let kt = key(t);
                    for r in 0..t {
                        let kk: f32 = key(r).iter().zip(kt).map(|(x, y)| x * y).sum();
                        let factor = bt * ratio(t, r) * kk;
                        if factor == 0.0 {
                            continue;
                        }
                        for row in 0..DV {
                            c[t * DV + row] -= factor * c[r * DV + row];
                        }
                    }
                }

                // Output: the input state's contribution plus the lower-triangular part.
                for t in 0..len {
                    let qt = query(t);
                    for row in 0..DV {
                        let sq: f32 = s[row * DK..][..DK].iter().zip(qt).map(|(x, y)| x * y).sum();
                        out[((start + t) * HV + h) * DV + row] = gamma[t] * sq;
                    }
                    for r in 0..=t {
                        let kq: f32 = key(r).iter().zip(qt).map(|(x, y)| x * y).sum();
                        let factor = ratio(t, r) * kq;
                        for row in 0..DV {
                            out[((start + t) * HV + h) * DV + row] += factor * c[r * DV + row];
                        }
                    }
                }

                // State at the end of the chunk.
                let last = gamma[len - 1];
                for row in 0..DV {
                    for j in 0..DK {
                        s[row * DK + j] *= last;
                    }
                }
                for t in 0..len {
                    let weight = ratio(len - 1, t);
                    let kt = key(t);
                    for row in 0..DV {
                        let scaled = weight * c[t * DV + row];
                        for j in 0..DK {
                            s[row * DK + j] += scaled * kt[j];
                        }
                    }
                }
                start += len;
            }
        }
        for (slot, value) in state.iter_mut().zip(&wide) {
            *slot = bf16::from_f32(*value);
        }
    }

    /// Reference recurrent chunk scan with FP32 state: rounding to
    /// BF16 happens only at the chunk boundary, as in `DeltaStateMode::Fp32`.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_chunk_fp32(
        state: &mut [u16],
        q: &[f32],
        k: &[f32],
        v: &[f32],
        alpha: &[f32],
        beta: &[f32],
        out: &mut [f32],
        tokens: usize,
    ) {
        const DK: usize = LA_K_HEAD_DIM;
        const DV: usize = LA_V_HEAD_DIM;
        const HV: usize = LA_NUM_V_HEADS;
        const RATIO: usize = LA_NUM_V_HEADS / LA_NUM_K_HEADS;

        let mut wide: Vec<f32> = state.iter().map(|&x| bf16::to_f32(x)).collect();
        for h in 0..HV {
            let hk = h / RATIO;
            for row in 0..DV {
                let sr = &mut wide[(h * DV + row) * DK..][..DK];
                for token in 0..tokens {
                    let kv = &k[(token * LA_NUM_K_HEADS + hk) * DK..][..DK];
                    let qv = &q[(token * LA_NUM_K_HEADS + hk) * DK..][..DK];
                    let kq: f32 = kv.iter().zip(qv).map(|(a, b)| a * b).sum();
                    let a = alpha[token * HV + h];
                    let bt = beta[token * HV + h];
                    let mut u = 0.0f32;
                    let mut w = 0.0f32;
                    for j in 0..DK {
                        u += sr[j] * kv[j];
                        w += sr[j] * qv[j];
                    }
                    let c = bt * (v[(token * HV + h) * DV + row] - a * u);
                    out[(token * HV + h) * DV + row] = a * w + c * kq;
                    for j in 0..DK {
                        sr[j] = a * sr[j] + c * kv[j];
                    }
                }
            }
        }
        for (slot, value) in state.iter_mut().zip(&wide) {
            *slot = bf16::from_f32(*value);
        }
    }
}

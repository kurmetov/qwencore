//! Vocab matrices in FP8: the embedding table and lm_head.
//!
//! In the checkpoint both `[vocab, hidden]` matrices are BF16 and cost 2.54 GB each.
//! We store them in E4M3 with an FP32 scale per row: exactly half, and the error
//! stays local — one badly scaled row doesn't spoil the whole
//! vocab. The scale factors out of the dot product, so lm_head
//! applies it once per row, not per element.

use crate::error::{Result, check};
use crate::{DeviceBuffer, Stream, ffi};
use std::ffi::c_void;

/// Batch cap of the narrow logits kernel: the hidden tile lives in static
/// shared memory. At batch 1 it runs at the bandwidth ceiling, so it stays
/// the fast path for a single sequence.
pub const MAX_LOGITS_BATCH: usize = 8;

/// Batch cap of the batched kernel: the table is read once for any batch
/// up to this value. Above it — groups, as before.
pub const MAX_BATCHED_LOGITS: usize = 96;

/// The same caps, but read from the kernel: the constants must not drift apart.
pub fn kernel_max_logits_batch() -> usize {
    unsafe { ffi::qwc_fp8_max_logits_batch() as usize }
}

pub fn kernel_max_batched_logits() -> usize {
    unsafe { ffi::qwc_fp8_max_batched_logits() as usize }
}

pub struct Fp8Vocab {
    data: DeviceBuffer<u8>,
    row_scales: DeviceBuffer<f32>,
    rows: usize,
    cols: usize,
}

/// The original BF16 vocab matrix without extra quantization.
///
/// The diagnostic path is shared by the embedding and `lm_head`: each matrix
/// costs 1.27 GB more than the FP8 variant.
pub struct Bf16Vocab {
    data: DeviceBuffer<u8>,
    rows: usize,
    cols: usize,
}

impl Bf16Vocab {
    pub fn zeroed(rows: usize, cols: usize) -> Result<Self> {
        assert!(rows > 0 && cols > 0);
        Ok(Self {
            data: DeviceBuffer::zeroed(rows * cols * 2)?,
            rows,
            cols,
        })
    }

    /// Copies whole BF16 rows from mmap without an unaligned `u16` cast.
    pub fn copy_rows(&mut self, first_row: usize, bf16_le: &[u8]) -> Result<()> {
        let row_bytes = self.cols * 2;
        assert!(bf16_le.len().is_multiple_of(row_bytes));
        let rows = bf16_le.len() / row_bytes;
        assert!(first_row + rows <= self.rows);
        self.data.copy_from_slice_at(first_row * row_bytes, bf16_le)
    }

    pub fn gather(
        &self,
        token_ids: &DeviceBuffer<u32>,
        output: &mut DeviceBuffer<u16>,
        stream: &Stream,
    ) -> Result<()> {
        self.gather_rows(token_ids, output, token_ids.len(), stream)
    }

    pub fn gather_rows(
        &self,
        token_ids: &DeviceBuffer<u32>,
        output: &mut DeviceBuffer<u16>,
        batch: usize,
        stream: &Stream,
    ) -> Result<()> {
        assert!(batch > 0 && token_ids.len() >= batch);
        assert!(output.len() >= batch * self.cols);
        check(unsafe {
            ffi::qwc_bf16_embedding_gather(
                self.data.as_ptr(),
                token_ids.as_ptr(),
                output.as_mut_ptr(),
                batch as i32,
                self.cols as i32,
                self.rows as i32,
                stream.raw(),
            )
        })
    }

    pub fn logits(
        &self,
        hidden: &DeviceBuffer<u16>,
        logits: &mut DeviceBuffer<f32>,
        batch: usize,
        stream: &Stream,
    ) -> Result<()> {
        assert!(batch > 0);
        assert!(hidden.len() >= batch * self.cols);
        assert!(logits.len() >= batch * self.rows);
        let mut done = 0;
        while done < batch {
            let group = (batch - done).min(MAX_LOGITS_BATCH);
            // SAFETY: offsets within the buffers checked above.
            let hidden_ptr =
                unsafe { (hidden.as_ptr() as *const u16).add(done * self.cols) as *const c_void };
            let logits_ptr =
                unsafe { (logits.as_mut_ptr() as *mut f32).add(done * self.rows) as *mut c_void };
            check(unsafe {
                ffi::qwc_bf16_lm_head(
                    self.data.as_ptr(),
                    hidden_ptr,
                    logits_ptr,
                    group as i32,
                    self.cols as i32,
                    self.rows as i32,
                    stream.raw(),
                )
            })?;
            done += group;
        }
        Ok(())
    }

    pub fn logits_row_to(
        &self,
        hidden: &DeviceBuffer<u16>,
        hidden_row: usize,
        logits: &mut DeviceBuffer<f32>,
        logits_row: usize,
        stream: &Stream,
    ) -> Result<()> {
        assert!((hidden_row + 1) * self.cols <= hidden.len());
        assert!((logits_row + 1) * self.rows <= logits.len());
        // SAFETY: both offsets are covered by the assertion above.
        let hidden_ptr =
            unsafe { (hidden.as_ptr() as *const u16).add(hidden_row * self.cols) as *const c_void };
        let logits_ptr =
            unsafe { (logits.as_mut_ptr() as *mut f32).add(logits_row * self.rows) as *mut c_void };
        check(unsafe {
            ffi::qwc_bf16_lm_head(
                self.data.as_ptr(),
                hidden_ptr,
                logits_ptr,
                1,
                self.cols as i32,
                self.rows as i32,
                stream.raw(),
            )
        })
    }

    pub fn resident_bytes(&self) -> usize {
        self.data.bytes()
    }

    /// A copy for the GPU oracle test.
    pub fn to_host(&self) -> Result<Vec<u8>> {
        self.data.to_vec()
    }
}

impl Fp8Vocab {
    pub fn zeroed(rows: usize, cols: usize) -> Result<Self> {
        assert!(rows > 0 && cols > 0);
        Ok(Self {
            data: DeviceBuffer::zeroed(rows * cols)?,
            row_scales: DeviceBuffer::zeroed(rows)?,
            rows,
            cols,
        })
    }

    /// Quantizes rows `[first_row, first_row + rows)` from BF16. The input is
    /// the checkpoint's raw little-endian bytes: the mmap slice isn't aligned for `u16`, and
    /// an extra host copy here would cost 2.54 GB.
    ///
    /// `staging` is reused between calls, so the launch is waited on right
    /// here: otherwise the next copy would overwrite data of a still-running kernel.
    pub fn quantize_rows(
        &mut self,
        first_row: usize,
        bf16_le: &[u8],
        staging: &mut DeviceBuffer<u8>,
        stream: &Stream,
    ) -> Result<()> {
        let row_bytes = self.cols * 2;
        assert!(bf16_le.len().is_multiple_of(row_bytes));
        let rows = bf16_le.len() / row_bytes;
        assert!(first_row + rows <= self.rows);
        assert!(staging.len() >= bf16_le.len());
        staging.copy_from_slice(bf16_le)?;
        check(unsafe {
            ffi::qwc_fp8_quantize_rows(
                staging.as_ptr(),
                self.data.as_mut_ptr(),
                self.row_scales.as_mut_ptr(),
                rows as i32,
                self.cols as i32,
                first_row as i32,
                stream.raw(),
            )
        })?;
        stream.synchronize()
    }

    /// Table rows by token ids, straight into BF16.
    pub fn gather(
        &self,
        token_ids: &DeviceBuffer<u32>,
        output: &mut DeviceBuffer<u16>,
        stream: &Stream,
    ) -> Result<()> {
        let batch = token_ids.len();
        assert!(batch > 0);
        assert_eq!(output.len(), batch * self.cols);
        check(unsafe {
            ffi::qwc_fp8_embedding_gather(
                self.data.as_ptr(),
                self.row_scales.as_ptr(),
                token_ids.as_ptr(),
                output.as_mut_ptr(),
                batch as i32,
                self.cols as i32,
                self.rows as i32,
                stream.raw(),
            )
        })
    }

    /// Gather the live prefix of capacity-sized executor buffers.
    pub fn gather_rows(
        &self,
        token_ids: &DeviceBuffer<u32>,
        output: &mut DeviceBuffer<u16>,
        batch: usize,
        stream: &Stream,
    ) -> Result<()> {
        assert!(batch > 0 && token_ids.len() >= batch);
        assert!(output.len() >= batch * self.cols);
        check(unsafe {
            ffi::qwc_fp8_embedding_gather(
                self.data.as_ptr(),
                self.row_scales.as_ptr(),
                token_ids.as_ptr(),
                output.as_mut_ptr(),
                batch as i32,
                self.cols as i32,
                self.rows as i32,
                stream.raw(),
            )
        })
    }

    /// Logits `[batch, vocab]` in FP32.
    ///
    /// Up to `MAX_LOGITS_BATCH` the narrow kernel runs — at batch 1 it's already at
    /// the bandwidth ceiling. Beyond that the batched one: it reads the table once, whereas
    /// groups of eight re-read it at 1.27 GB per group.
    pub fn logits(
        &self,
        hidden: &DeviceBuffer<u16>,
        logits: &mut DeviceBuffer<f32>,
        batch: usize,
        stream: &Stream,
    ) -> Result<()> {
        assert!(batch > 0);
        assert!(hidden.len() >= batch * self.cols);
        assert!(logits.len() >= batch * self.rows);
        if batch > MAX_LOGITS_BATCH && self.cols.is_multiple_of(4) {
            return self.logits_batched(hidden, logits, batch, stream);
        }
        let mut done = 0;
        while done < batch {
            let group = (batch - done).min(MAX_LOGITS_BATCH);
            // SAFETY: offsets within buffers already checked by length.
            let hidden_ptr =
                unsafe { (hidden.as_ptr() as *const u16).add(done * self.cols) as *const c_void };
            let logits_ptr =
                unsafe { (logits.as_mut_ptr() as *mut f32).add(done * self.rows) as *mut c_void };
            check(unsafe {
                ffi::qwc_fp8_lm_head(
                    self.data.as_ptr(),
                    self.row_scales.as_ptr(),
                    hidden_ptr,
                    logits_ptr,
                    group as i32,
                    self.cols as i32,
                    self.rows as i32,
                    stream.raw(),
                )
            })?;
            done += group;
        }
        Ok(())
    }

    /// Logits of one hidden-state row, but only for the vocab rows
    /// in `row_ids`. The output is dense: `logits[i]` corresponds to `row_ids[i]`,
    /// so the argmax over it gives an index into the list, not a token.
    ///
    /// Needed by the draft head: its proposal is verified by the main model, and
    /// a list miss costs a rejected draft, not a wrong output.
    pub fn logits_subset(
        &self,
        hidden: &DeviceBuffer<u16>,
        row_ids: &DeviceBuffer<u32>,
        logits: &mut DeviceBuffer<f32>,
        count: usize,
        stream: &Stream,
    ) -> Result<()> {
        assert!(count > 0 && count <= row_ids.len());
        assert!(hidden.len() >= self.cols);
        assert!(logits.len() >= count);
        check(unsafe {
            ffi::qwc_fp8_lm_head_subset(
                self.data.as_ptr(),
                self.row_scales.as_ptr(),
                hidden.as_ptr(),
                row_ids.as_ptr(),
                logits.as_mut_ptr(),
                count as i32,
                self.cols as i32,
                self.rows as i32,
                stream.raw(),
            )
        })
    }

    /// One pass over the table for the whole batch. Batches larger than
    /// `MAX_BATCHED_LOGITS` are still split, but that's far beyond the executor's
    /// `MAX_BATCH`.
    fn logits_batched(
        &self,
        hidden: &DeviceBuffer<u16>,
        logits: &mut DeviceBuffer<f32>,
        batch: usize,
        stream: &Stream,
    ) -> Result<()> {
        let mut done = 0;
        while done < batch {
            let group = (batch - done).min(MAX_BATCHED_LOGITS);
            // SAFETY: offsets within buffers already checked by length.
            let hidden_ptr =
                unsafe { (hidden.as_ptr() as *const u16).add(done * self.cols) as *const c_void };
            let logits_ptr =
                unsafe { (logits.as_mut_ptr() as *mut f32).add(done * self.rows) as *mut c_void };
            check(unsafe {
                ffi::qwc_fp8_lm_head_batched(
                    self.data.as_ptr(),
                    self.row_scales.as_ptr(),
                    hidden_ptr,
                    logits_ptr,
                    group as i32,
                    self.cols as i32,
                    self.rows as i32,
                    stream.raw(),
                )
            })?;
            done += group;
        }
        Ok(())
    }

    /// Compute one row from a capacity-sized hidden-state buffer and place it
    /// at the beginning of `logits`. Prefill only needs the final prompt row.
    pub fn logits_row(
        &self,
        hidden: &DeviceBuffer<u16>,
        hidden_row: usize,
        logits: &mut DeviceBuffer<f32>,
        stream: &Stream,
    ) -> Result<()> {
        self.logits_row_to(hidden, hidden_row, logits, 0, stream)
    }

    pub fn logits_row_to(
        &self,
        hidden: &DeviceBuffer<u16>,
        hidden_row: usize,
        logits: &mut DeviceBuffer<f32>,
        logits_row: usize,
        stream: &Stream,
    ) -> Result<()> {
        assert!((hidden_row + 1) * self.cols <= hidden.len());
        assert!((logits_row + 1) * self.rows <= logits.len());
        // SAFETY: both offsets are covered by the assertions above.
        let hidden_ptr =
            unsafe { (hidden.as_ptr() as *const u16).add(hidden_row * self.cols) as *const c_void };
        let logits_ptr =
            unsafe { (logits.as_mut_ptr() as *mut f32).add(logits_row * self.rows) as *mut c_void };
        check(unsafe {
            ffi::qwc_fp8_lm_head(
                self.data.as_ptr(),
                self.row_scales.as_ptr(),
                hidden_ptr,
                logits_ptr,
                1,
                self.cols as i32,
                self.rows as i32,
                stream.raw(),
            )
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn resident_bytes(&self) -> usize {
        self.data.bytes() + self.row_scales.bytes()
    }

    /// Copies to host for checking against the reference.
    pub fn to_host(&self) -> Result<(Vec<u8>, Vec<f32>)> {
        Ok((self.data.to_vec()?, self.row_scales.to_vec()?))
    }
}

/// A CPU reference on top of already quantized data: it checks the kernels'
/// arithmetic, not the choice of scale.
pub mod reference {
    use crate::bf16;
    use crate::nvfp4::reference::e4m3;

    pub fn gather(
        table: &[u8],
        row_scales: &[f32],
        token_ids: &[u32],
        cols: usize,
        output: &mut [u16],
    ) {
        assert_eq!(output.len(), token_ids.len() * cols);
        for (index, &token) in token_ids.iter().enumerate() {
            let base = token as usize * cols;
            let scale = row_scales[token as usize];
            for column in 0..cols {
                output[index * cols + column] = bf16::from_f32(e4m3(table[base + column]) * scale);
            }
        }
    }

    pub fn logits(
        weights: &[u8],
        row_scales: &[f32],
        hidden: &[u16],
        batch: usize,
        cols: usize,
        vocab: usize,
        output: &mut [f32],
    ) {
        assert_eq!(output.len(), batch * vocab);
        for row in 0..vocab {
            let base = row * cols;
            for b in 0..batch {
                let mut sum = 0.0f32;
                for column in 0..cols {
                    sum += e4m3(weights[base + column]) * bf16::to_f32(hidden[b * cols + column]);
                }
                output[b * vocab + row] = sum * row_scales[row];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    /// The Rust constant and the kernel constant must not drift apart: the size
    /// of the shared tile depends on this cap.
    #[test]
    fn max_logits_batch_matches_kernel() {
        assert_eq!(super::MAX_LOGITS_BATCH, super::kernel_max_logits_batch());
        assert_eq!(super::MAX_BATCHED_LOGITS, super::kernel_max_batched_logits());
    }
}

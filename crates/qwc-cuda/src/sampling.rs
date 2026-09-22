//! GPU-side greedy sampling.

use crate::error::{Result, check};
use crate::{DeviceBuffer, Stream, ffi};

const PARTS: usize = 256;

pub struct Argmax {
    partial_values: DeviceBuffer<f32>,
    partial_indices: DeviceBuffer<u32>,
    tokens: DeviceBuffer<u32>,
    max_batch: usize,
    vocab: usize,
}

impl Argmax {
    pub fn new(max_batch: usize, vocab: usize) -> Result<Self> {
        assert!((1..=128).contains(&max_batch));
        assert!(vocab > 0);
        Ok(Self {
            partial_values: DeviceBuffer::zeroed(max_batch * PARTS)?,
            partial_indices: DeviceBuffer::zeroed(max_batch * PARTS)?,
            tokens: DeviceBuffer::zeroed(max_batch)?,
            max_batch,
            vocab,
        })
    }

    pub fn sample(
        &mut self,
        logits: &DeviceBuffer<f32>,
        batch: usize,
        stream: &Stream,
    ) -> Result<()> {
        assert!(batch > 0 && batch <= self.max_batch);
        assert!(logits.len() >= batch * self.vocab);
        check(unsafe {
            ffi::qwc_argmax(
                logits.as_ptr().cast(),
                self.partial_values.as_mut_ptr().cast(),
                self.partial_indices.as_mut_ptr().cast(),
                self.tokens.as_mut_ptr().cast(),
                self.vocab as i32,
                batch as i32,
                PARTS as i32,
                stream.raw(),
            )
        })
    }

    /// Argmax по первым `count` значениям одной строки, сразу на хост.
    ///
    /// Нужен шортлисту черновой головы: его выход плотный и короче словаря,
    /// поэтому возвращается место в списке, а не идентификатор токена.
    pub fn sample_prefix(
        &mut self,
        logits: &DeviceBuffer<f32>,
        count: usize,
        stream: &Stream,
    ) -> Result<usize> {
        assert!(count > 0 && count <= self.vocab);
        assert!(logits.len() >= count);
        check(unsafe {
            ffi::qwc_argmax(
                logits.as_ptr().cast(),
                self.partial_values.as_mut_ptr().cast(),
                self.partial_indices.as_mut_ptr().cast(),
                self.tokens.as_mut_ptr().cast(),
                count as i32,
                1,
                PARTS as i32,
                stream.raw(),
            )
        })?;
        Ok(self.to_host(1)?[0] as usize)
    }

    pub fn to_host(&self, batch: usize) -> Result<Vec<u32>> {
        assert!(batch > 0 && batch <= self.max_batch);
        let mut tokens = self.tokens.to_vec()?;
        tokens.truncate(batch);
        Ok(tokens)
    }
}

//! Плоская CPU-раскладка batch metadata для загрузки в GPU executor.
//!
//! Вектора имеют структуру-of-arrays и используют `u32`: CUDA-кернелам не
//! нужны Rust-структуры с padding, а один и тот же device buffer позднее можно
//! преаллоцировать под максимальный batch и обновлять только активный префикс.

use crate::cache::{CacheManager, SeqId};
use crate::scheduler::{Batch, PrefillChunk};
use std::collections::HashSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutError {
    MissingSequence(SeqId),
    DuplicateSequence(SeqId),
    InvalidPrefillRange(SeqId),
    MissingKvBlocks(SeqId),
    IntegerOverflow(&'static str),
}

impl std::fmt::Display for LayoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingSequence(id) => write!(f, "SeqId {id} отсутствует в кэше"),
            Self::DuplicateSequence(id) => write!(f, "SeqId {id} дважды присутствует в batch"),
            Self::InvalidPrefillRange(id) => write!(f, "prefill-диапазон SeqId {id} некорректен"),
            Self::MissingKvBlocks(id) => write!(f, "не хватает выделенных KV-блоков SeqId {id}"),
            Self::IntegerOverflow(field) => write!(f, "{field} не помещается в GPU metadata"),
        }
    }
}

impl std::error::Error for LayoutError {}

/// Компактные metadata одного смешанного batch. Decode-последовательности
/// всегда идут первыми (`0..num_decode`), затем prefill-чанки.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchLayout {
    pub num_decode: u32,
    pub seq_ids: Vec<SeqId>,
    pub state_slots: Vec<u32>,
    /// Начальная абсолютная позиция входных токенов каждой последовательности.
    pub position_starts: Vec<u32>,
    /// Длина видимого контекста после исполнения элемента batch.
    pub context_lens: Vec<u32>,
    /// CSR offsets токенов; последний элемент равен общему числу токенов.
    pub token_offsets: Vec<u32>,
    /// CSR offsets таблиц KV-блоков каждой последовательности.
    pub block_table_offsets: Vec<u32>,
    /// Плоские физические ID страниц KV-кэша.
    pub block_ids: Vec<u32>,
}

impl BatchLayout {
    pub fn build(batch: &Batch, cache: &CacheManager) -> Result<Self, LayoutError> {
        let mut layout = Self {
            num_decode: as_u32(batch.decode.len(), "num_decode")?,
            seq_ids: Vec::with_capacity(batch.num_seqs()),
            state_slots: Vec::with_capacity(batch.num_seqs()),
            position_starts: Vec::with_capacity(batch.num_seqs()),
            context_lens: Vec::with_capacity(batch.num_seqs()),
            token_offsets: vec![0],
            block_table_offsets: vec![0],
            block_ids: Vec::new(),
        };
        let mut seen = HashSet::with_capacity(batch.num_seqs());

        for &id in &batch.decode {
            if !seen.insert(id) {
                return Err(LayoutError::DuplicateSequence(id));
            }
            let seq = cache.sequence(id).ok_or(LayoutError::MissingSequence(id))?;
            let position = seq
                .tokens
                .checked_sub(1)
                .ok_or(LayoutError::IntegerOverflow("decode position"))?;
            layout.push_sequence(id, seq.state_slot, position, seq.tokens, 1, cache)?;
        }
        for &chunk in &batch.prefill {
            if !seen.insert(chunk.id) {
                return Err(LayoutError::DuplicateSequence(chunk.id));
            }
            layout.push_prefill(chunk, cache)?;
        }

        Ok(layout)
    }

    pub fn num_seqs(&self) -> usize {
        self.seq_ids.len()
    }

    pub fn num_tokens(&self) -> usize {
        self.token_offsets.last().copied().unwrap_or(0) as usize
    }

    fn push_prefill(
        &mut self,
        chunk: PrefillChunk,
        cache: &CacheManager,
    ) -> Result<(), LayoutError> {
        let seq = cache
            .sequence(chunk.id)
            .ok_or(LayoutError::MissingSequence(chunk.id))?;
        let context = chunk
            .offset
            .checked_add(chunk.tokens)
            .ok_or(LayoutError::IntegerOverflow("prefill context"))?;
        if chunk.tokens == 0 || context > seq.tokens {
            return Err(LayoutError::InvalidPrefillRange(chunk.id));
        }
        self.push_sequence(
            chunk.id,
            seq.state_slot,
            chunk.offset,
            context,
            chunk.tokens,
            cache,
        )
    }

    fn push_sequence(
        &mut self,
        id: SeqId,
        state_slot: u32,
        position: usize,
        context: usize,
        tokens: usize,
        cache: &CacheManager,
    ) -> Result<(), LayoutError> {
        let seq = cache.sequence(id).ok_or(LayoutError::MissingSequence(id))?;
        let blocks = cache.kv().blocks_for(context);
        if blocks > seq.blocks.len() {
            return Err(LayoutError::MissingKvBlocks(id));
        }

        self.seq_ids.push(id);
        self.state_slots.push(state_slot);
        self.position_starts.push(as_u32(position, "position")?);
        self.context_lens.push(as_u32(context, "context_len")?);

        let total_tokens = self
            .num_tokens()
            .checked_add(tokens)
            .ok_or(LayoutError::IntegerOverflow("суммарное число токенов"))?;
        self.token_offsets
            .push(as_u32(total_tokens, "token offset")?);

        self.block_ids.extend_from_slice(&seq.blocks[..blocks]);
        self.block_table_offsets
            .push(as_u32(self.block_ids.len(), "block table offset")?);
        Ok(())
    }
}

fn as_u32(value: usize, field: &'static str) -> Result<u32, LayoutError> {
    u32::try_from(value).map_err(|_| LayoutError::IntegerOverflow(field))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_batch_becomes_csr_metadata() {
        let mut cache = CacheManager::new(2, 10, 64);
        cache.admit(1, 64).unwrap();
        cache.append_token(1).unwrap();
        cache.admit(2, 100).unwrap();

        let batch = Batch {
            decode: vec![1],
            prefill: vec![PrefillChunk {
                id: 2,
                offset: 32,
                tokens: 16,
            }],
            ..Default::default()
        };
        let layout = BatchLayout::build(&batch, &cache).unwrap();

        assert_eq!(layout.num_decode, 1);
        assert_eq!(layout.seq_ids, vec![1, 2]);
        assert_eq!(layout.state_slots, vec![0, 1]);
        assert_eq!(layout.position_starts, vec![64, 32]);
        assert_eq!(layout.context_lens, vec![65, 48]);
        assert_eq!(layout.token_offsets, vec![0, 1, 17]);
        assert_eq!(layout.block_table_offsets, vec![0, 2, 3]);
        assert_eq!(layout.block_ids, vec![0, 1, 2]);
    }

    #[test]
    fn duplicate_or_missing_sequence_is_rejected() {
        let mut cache = CacheManager::new(1, 2, 64);
        cache.admit(1, 1).unwrap();
        cache.append_token(1).unwrap();

        let duplicate = Batch {
            decode: vec![1],
            prefill: vec![PrefillChunk {
                id: 1,
                offset: 0,
                tokens: 1,
            }],
            ..Default::default()
        };
        assert_eq!(
            BatchLayout::build(&duplicate, &cache),
            Err(LayoutError::DuplicateSequence(1))
        );

        let missing = Batch {
            decode: vec![7],
            prefill: vec![],
            ..Default::default()
        };
        assert_eq!(
            BatchLayout::build(&missing, &cache),
            Err(LayoutError::MissingSequence(7))
        );
    }

    #[test]
    fn invalid_prefill_range_is_rejected() {
        let mut cache = CacheManager::new(1, 2, 64);
        cache.admit(1, 10).unwrap();
        let batch = Batch {
            decode: vec![],
            prefill: vec![PrefillChunk {
                id: 1,
                offset: 8,
                tokens: 4,
            }],
            ..Default::default()
        };
        assert_eq!(
            BatchLayout::build(&batch, &cache),
            Err(LayoutError::InvalidPrefillRange(1))
        );
    }
}

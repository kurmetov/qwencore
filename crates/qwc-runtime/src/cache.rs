//! Менеджер кэша для гибридной архитектуры.
//!
//! Ключевое отличие от обычного трансформера: последовательность занимает
//! ДВА независимых ресурса.
//!
//!   1. Слот состояния DeltaNet (48 слоёв) — фиксированные 81 MB,
//!      не зависят от длины контекста;
//!   2. Блоки paged KV-кэша (16 слоёв) — растут по мере генерации.
//!
//! Обычный движок считает только второе. Из-за этого на коротких контекстах
//! (ниже ~2500 токенов) он ошибается в оценке ёмкости: там дефицитен именно
//! слот состояния, а KV-блоков ещё вдоволь.
//!
//! Отсюда два следствия для admission control:
//!   * принять запрос можно только при наличии ОБОИХ ресурсов;
//!   * узкое место зависит от длины контекста, поэтому планировщик должен
//!     смотреть на обе загрузки, а не на одну.

use qwc_core::memory::{Budget, CacheConfig};
use std::collections::HashMap;

pub type SeqId = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rejected {
    /// Свободных слотов состояния нет — упёрлись в concurrency.
    NoStateSlot,
    /// Не хватает KV-блоков под запрошенный контекст.
    NoKvBlocks { need: usize, free: usize },
    /// Последовательность уже принята.
    AlreadyAdmitted,
    /// Последовательность не была принята или уже завершена.
    UnknownSequence,
    /// Откатывать нечего: у последовательности нет ни одного токена.
    NothingToRollback,
}

impl std::fmt::Display for Rejected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoStateSlot => write!(f, "нет свободного слота состояния DeltaNet"),
            Self::NoKvBlocks { need, free } => {
                write!(f, "нужно {need} KV-блоков, свободно {free}")
            }
            Self::AlreadyAdmitted => write!(f, "последовательность уже принята"),
            Self::UnknownSequence => write!(f, "последовательность не принята"),
            Self::NothingToRollback => write!(f, "нет токена для отката"),
        }
    }
}

/// Пул слотов рекуррентного состояния. Слоты одинаковы и неделимы:
/// состояние DeltaNet имеет фиксированный размер независимо от контекста.
#[derive(Debug)]
pub struct StatePool {
    capacity: usize,
    free: Vec<u32>,
}

impl StatePool {
    pub fn new(capacity: usize) -> Self {
        assert!(
            u32::try_from(capacity).is_ok(),
            "число state-слотов не помещается в u32"
        );
        Self {
            capacity,
            free: (0..capacity as u32).rev().collect(),
        }
    }

    fn acquire(&mut self) -> Option<u32> {
        self.free.pop()
    }

    fn release(&mut self, slot: u32) {
        debug_assert!(
            !self.free.contains(&slot),
            "двойное освобождение слота {slot}"
        );
        self.free.push(slot);
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn in_use(&self) -> usize {
        self.capacity - self.free.len()
    }

    pub fn utilization(&self) -> f64 {
        self.in_use() as f64 / self.capacity.max(1) as f64
    }
}

/// Постраничный KV-кэш. Блоки одинакового размера, таблица блоков на
/// последовательность — при таком устройстве нет внешней фрагментации.
#[derive(Debug)]
pub struct KvPool {
    block_size: usize,
    capacity: usize,
    free: Vec<u32>,
}

impl KvPool {
    pub fn new(capacity: usize, block_size: usize) -> Self {
        assert!(block_size > 0, "размер KV-блока должен быть положительным");
        assert!(
            u32::try_from(capacity).is_ok(),
            "число KV-блоков не помещается в u32"
        );
        Self {
            block_size,
            capacity,
            free: (0..capacity as u32).rev().collect(),
        }
    }

    pub fn blocks_for(&self, tokens: usize) -> usize {
        tokens.div_ceil(self.block_size)
    }

    pub fn free_blocks(&self) -> usize {
        self.free.len()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn utilization(&self) -> f64 {
        (self.capacity - self.free.len()) as f64 / self.capacity.max(1) as f64
    }

    fn acquire(&mut self, n: usize) -> Option<Vec<u32>> {
        if self.free.len() < n {
            return None;
        }
        Some((0..n).map(|_| self.free.pop().unwrap()).collect())
    }

    fn release(&mut self, blocks: &[u32]) {
        self.free.extend_from_slice(blocks);
    }
}

/// Состояние принятой последовательности.
#[derive(Debug)]
pub struct Sequence {
    pub state_slot: u32,
    pub blocks: Vec<u32>,
    /// Сколько токенов уже лежит в KV-кэше.
    pub tokens: usize,
}

#[derive(Debug)]
pub struct CacheManager {
    states: StatePool,
    kv: KvPool,
    seqs: HashMap<SeqId, Sequence>,
}

impl CacheManager {
    pub fn new(state_slots: usize, kv_blocks: usize, block_size: usize) -> Self {
        Self {
            states: StatePool::new(state_slots),
            kv: KvPool::new(kv_blocks, block_size),
            seqs: HashMap::new(),
        }
    }

    /// Делит доступную VRAM между двумя подсистемами.
    ///
    /// `target_context` — длина контекста, под которую оптимизируем. Слотов
    /// состояния делаем ровно столько, сколько последовательностей такой длины
    /// поместится; остальное отдаём под KV-блоки. Смысл в том, что лишний слот
    /// состояния бесполезен, если под него нет KV-блоков, и наоборот.
    pub fn from_budget(budget: &Budget, cfg: &CacheConfig, target_context: usize) -> Self {
        let available = budget.cache_available();
        let per_seq = cfg.bytes_per_seq(target_context).max(1);
        let slots = (available / per_seq) as usize;

        let state_bytes = slots as u64 * cfg.state_bytes_per_slot();
        let kv_bytes = available.saturating_sub(state_bytes);
        let blocks = (kv_bytes / cfg.kv_bytes_per_block().max(1)) as usize;

        Self::new(slots, blocks, cfg.block_size)
    }

    pub fn admit(&mut self, id: SeqId, prompt_tokens: usize) -> Result<(), Rejected> {
        if self.seqs.contains_key(&id) {
            return Err(Rejected::AlreadyAdmitted);
        }
        let need = self.kv.blocks_for(prompt_tokens);
        if self.kv.free_blocks() < need {
            return Err(Rejected::NoKvBlocks {
                need,
                free: self.kv.free_blocks(),
            });
        }
        // Слот берём только после проверки KV, иначе пришлось бы откатывать.
        let Some(slot) = self.states.acquire() else {
            return Err(Rejected::NoStateSlot);
        };
        let blocks = self.kv.acquire(need).expect("наличие блоков уже проверено");
        self.seqs.insert(
            id,
            Sequence {
                state_slot: slot,
                blocks,
                tokens: prompt_tokens,
            },
        );
        Ok(())
    }

    /// Добавляет один сгенерированный токен, при необходимости выделяя блок.
    /// Планировщик вызывает этот метод до запуска decode: тем самым память
    /// резервируется заранее и GPU-шаг уже не может упасть из-за роста кэша.
    pub fn append_token(&mut self, id: SeqId) -> Result<(), Rejected> {
        let Some(seq) = self.seqs.get(&id) else {
            return Err(Rejected::UnknownSequence);
        };
        let need = self.kv.blocks_for(seq.tokens + 1);
        if need > seq.blocks.len() {
            let extra = need - seq.blocks.len();
            if self.kv.free_blocks() < extra {
                return Err(Rejected::NoKvBlocks {
                    need: extra,
                    free: self.kv.free_blocks(),
                });
            }
            let more = self.kv.acquire(extra).unwrap();
            self.seqs.get_mut(&id).unwrap().blocks.extend(more);
        }
        self.seqs.get_mut(&id).unwrap().tokens += 1;
        Ok(())
    }

    /// Откатывает последнюю резервацию decode-токена. Если при `append_token`
    /// был добавлен новый блок, он тоже возвращается в пул.
    pub fn rollback_token(&mut self, id: SeqId) -> Result<(), Rejected> {
        let Some(seq) = self.seqs.get(&id) else {
            return Err(Rejected::UnknownSequence);
        };
        if seq.tokens == 0 {
            return Err(Rejected::NothingToRollback);
        }

        let tokens = seq.tokens - 1;
        let keep = self.kv.blocks_for(tokens);
        let released = {
            let seq = self.seqs.get_mut(&id).unwrap();
            seq.tokens = tokens;
            seq.blocks.split_off(keep)
        };
        self.kv.release(&released);
        Ok(())
    }

    pub fn release(&mut self, id: SeqId) {
        if let Some(seq) = self.seqs.remove(&id) {
            self.states.release(seq.state_slot);
            self.kv.release(&seq.blocks);
        }
    }

    pub fn sequence(&self, id: SeqId) -> Option<&Sequence> {
        self.seqs.get(&id)
    }

    pub fn active(&self) -> usize {
        self.seqs.len()
    }

    pub fn states(&self) -> &StatePool {
        &self.states
    }

    pub fn kv(&self) -> &KvPool {
        &self.kv
    }

    /// Какая из подсистем ближе к исчерпанию. Планировщику важно знать
    /// именно это: давление на них устроено по-разному.
    pub fn pressure(&self) -> Pressure {
        Pressure {
            states: self.states.utilization(),
            kv: self.kv.utilization(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Pressure {
    pub states: f64,
    pub kv: f64,
}

impl Pressure {
    pub fn bottleneck(&self) -> &'static str {
        if self.states >= self.kv {
            "слоты состояния"
        } else {
            "KV-блоки"
        }
    }

    pub fn worst(&self) -> f64 {
        self.states.max(self.kv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qwc_core::memory::{GB, WeightPlan, rtx5090};

    fn manager(target_ctx: usize) -> CacheManager {
        let b = rtx5090(WeightPlan::default().bytes(), 2 * GB);
        CacheManager::from_budget(&b, &CacheConfig::default(), target_ctx)
    }

    #[test]
    fn admission_requires_both_resources() {
        let mut m = CacheManager::new(2, 1000, 64);
        assert!(m.admit(1, 128).is_ok());
        assert!(m.admit(2, 128).is_ok());
        // Слоты состояния кончились, хотя KV-блоков ещё вдоволь.
        assert_eq!(m.admit(3, 128), Err(Rejected::NoStateSlot));
        assert!(m.kv().free_blocks() > 900);

        m.release(1);
        assert!(m.admit(3, 128).is_ok());
    }

    #[test]
    fn kv_can_be_the_binding_constraint() {
        // Слотов много, блоков мало.
        let mut m = CacheManager::new(64, 10, 64);
        assert!(m.admit(1, 640).is_ok()); // ровно 10 блоков
        assert!(matches!(m.admit(2, 64), Err(Rejected::NoKvBlocks { .. })));
        // Слот состояния при этом не израсходован.
        assert_eq!(m.states().in_use(), 1);
    }

    #[test]
    fn bottleneck_flips_with_context_length() {
        // При одной и той же раскладке пула короткие запросы упираются в
        // слоты состояния, а длинные — в KV-блоки.
        let mut short = CacheManager::new(64, 100, 64);
        let mut id = 0;
        while short.admit(id, 64).is_ok() {
            id += 1;
        }
        assert_eq!(short.pressure().bottleneck(), "слоты состояния");

        let mut long = CacheManager::new(64, 100, 64);
        let mut id = 0;
        while long.admit(id, 640).is_ok() {
            id += 1;
        }
        assert_eq!(long.pressure().bottleneck(), "KV-блоки");
    }

    #[test]
    fn budget_split_matches_memory_model() {
        let b = rtx5090(WeightPlan::default().bytes(), 2 * GB);
        let cfg = CacheConfig::default();

        // Число слотов должно совпасть с независимой оценкой из qwc-core.
        for ctx in [4096usize, 8192, 32768] {
            let m = CacheManager::from_budget(&b, &cfg, ctx);
            assert_eq!(
                m.states().capacity(),
                b.max_concurrency(&cfg, ctx),
                "контекст {ctx}"
            );
        }
    }

    #[test]
    fn append_allocates_lazily() {
        let mut m = CacheManager::new(4, 100, 64);
        m.admit(1, 64).unwrap();
        assert_eq!(m.sequence(1).unwrap().blocks.len(), 1);

        // Блок заполнен ровно, следующий токен требует нового.
        m.append_token(1).unwrap();
        assert_eq!(m.sequence(1).unwrap().blocks.len(), 2);
        assert_eq!(m.sequence(1).unwrap().tokens, 65);

        // Внутри блока новых выделений нет.
        let before = m.kv().free_blocks();
        for _ in 0..60 {
            m.append_token(1).unwrap();
        }
        assert_eq!(m.kv().free_blocks(), before);
    }

    #[test]
    fn append_and_rollback_are_transactional() {
        let mut m = CacheManager::new(1, 2, 64);
        m.admit(1, 64).unwrap();

        m.append_token(1).unwrap();
        assert_eq!(m.sequence(1).unwrap().tokens, 65);
        assert_eq!(m.sequence(1).unwrap().blocks.len(), 2);

        m.rollback_token(1).unwrap();
        assert_eq!(m.sequence(1).unwrap().tokens, 64);
        assert_eq!(m.sequence(1).unwrap().blocks.len(), 1);
        assert_eq!(m.kv().free_blocks(), 1);
    }

    #[test]
    fn unknown_sequence_is_an_error_not_a_panic() {
        let mut m = CacheManager::new(1, 2, 64);
        assert_eq!(m.append_token(7), Err(Rejected::UnknownSequence));
        assert_eq!(m.rollback_token(7), Err(Rejected::UnknownSequence));
    }

    #[test]
    #[should_panic(expected = "размер KV-блока должен быть положительным")]
    fn zero_block_size_is_rejected_at_construction() {
        CacheManager::new(1, 1, 0);
    }

    #[test]
    fn release_returns_everything() {
        let mut m = CacheManager::new(4, 100, 64);
        let (slots, blocks) = (m.states().capacity(), m.kv().free_blocks());
        for id in 0..4 {
            m.admit(id, 200).unwrap();
        }
        for id in 0..4 {
            m.release(id);
        }
        assert_eq!(m.states().in_use(), 0);
        assert_eq!(m.states().capacity(), slots);
        assert_eq!(m.kv().free_blocks(), blocks);
        assert_eq!(m.active(), 0);
    }

    #[test]
    fn short_context_wastes_kv_capacity() {
        // Пул статически спланирован под максимальный контекст, но реальные
        // запросы короче. Они исчерпывают слоты, оставляя KV недогруженным.
        let mut m = manager(32 * 1024);
        let mut id = 0;
        while m.admit(id, 1024).is_ok() {
            id += 1;
        }
        let p = m.pressure();
        assert!(p.states > 0.99, "слоты {:.2}", p.states);
        assert!(
            p.kv < 0.5,
            "KV использован на {:.2}, ожидали недоиспользование",
            p.kv
        );
    }
}

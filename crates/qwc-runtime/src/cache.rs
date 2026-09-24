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
//!
//! Кэш префиксов устроен под ту же гибридность. Общие KV-страницы делятся
//! счётчиком ссылок, как у обычного трансформера, но их одних мало: чтобы
//! продолжить чужой префикс, нужно ещё состояние DeltaNet ровно на его
//! границе. Поэтому запись кэша — это снимок состояния (bf16, ~76 МБ, лежит у
//! исполнителя) плюс страницы, покрывающие префикс. Снимок берётся на границе
//! истории, перед последним `<|im_start|>`: промпт генерации в историю
//! следующего хода не попадает, и снимок в конце промпта никогда не стал бы
//! чужим префиксом.

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
///
/// Страница может принадлежать нескольким владельцам сразу: последовательности
/// и записям кэша префиксов. В пул она возвращается, когда уходит последний.
#[derive(Debug)]
pub struct KvPool {
    block_size: usize,
    capacity: usize,
    free: Vec<u32>,
    refs: Vec<u32>,
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
            refs: vec![0; capacity],
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
        let blocks: Vec<u32> = (0..n).map(|_| self.free.pop().unwrap()).collect();
        for &block in &blocks {
            self.refs[block as usize] = 1;
        }
        Some(blocks)
    }

    fn retain(&mut self, blocks: &[u32]) {
        for &block in blocks {
            debug_assert!(self.refs[block as usize] > 0, "ссылка на свободный блок {block}");
            self.refs[block as usize] += 1;
        }
    }

    fn release(&mut self, blocks: &[u32]) {
        for &block in blocks {
            let refs = &mut self.refs[block as usize];
            debug_assert!(*refs > 0, "двойное освобождение блока {block}");
            *refs -= 1;
            if *refs == 0 {
                self.free.push(block);
            }
        }
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

/// Как продолжить закэшированный префикс: снимок состояния, который надо
/// поднять в слот последовательности, и страница, которую надо скопировать,
/// если префикс кончается посреди неё.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefixRestore {
    pub snapshot: u32,
    /// `(откуда, куда)`: хвост префикса лежит в неполной странице, а писать
    /// в неё дальше будет новая последовательность — страница копируется.
    pub copy_block: Option<(u32, u32)>,
}

/// Итог приёма запроса: сколько токенов промпта уже посчитано.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Admission {
    pub reused: usize,
    pub restore: Option<PrefixRestore>,
}

#[derive(Debug)]
struct PrefixEntry {
    tokens: Vec<u32>,
    blocks: Vec<u32>,
    snapshot: u32,
    last_used: u64,
    /// Сколько принятых последовательностей ждут подъёма этого снимка.
    /// Закреплённую запись вытеснять нельзя: её слот снимка перезапишут.
    pins: u32,
}

#[derive(Debug, Default)]
struct PrefixCache {
    entries: Vec<PrefixEntry>,
    free_snapshots: Vec<u32>,
    capacity: usize,
    clock: u64,
    stats: PrefixStats,
}

/// Счётчики кэша префиксов для логов и замеров.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrefixStats {
    pub lookups: u64,
    pub hits: u64,
    pub reused_tokens: u64,
    pub evictions: u64,
}

impl PrefixCache {
    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    fn position(&self, snapshot: u32) -> Option<usize> {
        self.entries.iter().position(|entry| entry.snapshot == snapshot)
    }

    /// Самая длинная запись, которая целиком является префиксом промпта и
    /// короче него: хотя бы один токен промпта надо прогнать, иначе нечем
    /// получить логиты под первый ответный токен.
    fn best_match(&self, prompt: &[u32], min_tokens: usize) -> Option<u32> {
        self.entries
            .iter()
            .filter(|entry| {
                entry.tokens.len() >= min_tokens
                    && entry.tokens.len() < prompt.len()
                    && prompt[..entry.tokens.len()] == entry.tokens[..]
            })
            .max_by_key(|entry| entry.tokens.len())
            .map(|entry| entry.snapshot)
    }

    /// Выбросить наименее ценную незакреплённую запись.
    ///
    /// Первыми уходят вложенные записи: их префикс целиком лежит в более
    /// длинной записи того же диалога. Дальше — записи, чьих страниц не
    /// держит ни одна живая последовательность: запись идущего запроса нужна
    /// его следующему ходу, как бы давно её ни трогали. Внутри — по давности.
    ///
    /// `for_pages` — вытесняем ради страниц KV, а не слота снимка. Тогда
    /// запись, все страницы которой держат живые последовательности, не
    /// трогаем: она ничего не освободит, а кэш потеряет.
    fn evict_one(&mut self, kv: &mut KvPool, for_pages: bool) -> bool {
        let mut holders = vec![0u32; kv.capacity()];
        for entry in &self.entries {
            for &block in &entry.blocks {
                holders[block as usize] += 1;
            }
        }
        // Страницу держит кто-то кроме записей — значит, живая последовательность.
        let live = |block: u32| kv.refs[block as usize] > holders[block as usize];
        let Some(index) = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| entry.pins == 0)
            .filter(|(_, entry)| !for_pages || !entry.blocks.iter().all(|&block| live(block)))
            .min_by_key(|(_, entry)| {
                let superseded = self.entries.iter().any(|other| {
                    other.tokens.len() > entry.tokens.len()
                        && other.tokens.starts_with(&entry.tokens)
                });
                let in_use = entry.blocks.iter().any(|&block| live(block));
                (!superseded, in_use, entry.last_used)
            })
            .map(|(index, _)| index)
        else {
            return false;
        };
        let entry = self.entries.swap_remove(index);
        kv.release(&entry.blocks);
        self.free_snapshots.push(entry.snapshot);
        self.stats.evictions += 1;
        true
    }
}

#[derive(Debug)]
pub struct CacheManager {
    states: StatePool,
    kv: KvPool,
    seqs: HashMap<SeqId, Sequence>,
    prefix: PrefixCache,
}

impl CacheManager {
    pub fn new(state_slots: usize, kv_blocks: usize, block_size: usize) -> Self {
        Self {
            states: StatePool::new(state_slots),
            kv: KvPool::new(kv_blocks, block_size),
            seqs: HashMap::new(),
            prefix: PrefixCache::default(),
        }
    }

    /// Включает кэш префиксов на `snapshots` снимков состояния. Память под
    /// снимки держит исполнитель; здесь только их номера.
    pub fn enable_prefix_cache(&mut self, snapshots: usize) {
        assert!(self.prefix.entries.is_empty(), "кэш префиксов уже в работе");
        assert!(u32::try_from(snapshots).is_ok());
        self.prefix.capacity = snapshots;
        self.prefix.free_snapshots = (0..snapshots as u32).rev().collect();
    }

    pub fn prefix_capacity(&self) -> usize {
        self.prefix.capacity
    }

    pub fn prefix_entries(&self) -> usize {
        self.prefix.entries.len()
    }

    pub fn prefix_stats(&self) -> PrefixStats {
        self.prefix.stats
    }

    /// Блоки из пула, при нехватке — ценой вытеснения записей кэша
    /// префиксов. Живые последовательности важнее любого кэша.
    fn acquire_blocks(&mut self, n: usize) -> Option<Vec<u32>> {
        loop {
            if self.kv.free_blocks() >= n {
                return self.kv.acquire(n);
            }
            if !self.prefix.evict_one(&mut self.kv, true) {
                return None;
            }
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
        // Слот проверяем до блоков: выделенные блоки пришлось бы откатывать,
        // а вытесненный ради них кэш — уже нет.
        if self.states.free.is_empty() {
            return Err(Rejected::NoStateSlot);
        }
        let Some(blocks) = self.acquire_blocks(need) else {
            return Err(Rejected::NoKvBlocks {
                need,
                free: self.kv.free_blocks(),
            });
        };
        let slot = self.states.acquire().expect("свободный слот уже проверен");
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
            let Some(more) = self.acquire_blocks(extra) else {
                return Err(Rejected::NoKvBlocks {
                    need: extra,
                    free: self.kv.free_blocks(),
                });
            };
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

    /// Принять запрос, по возможности продолжив закэшированный префикс.
    ///
    /// Без попадания это обычный `admit`. С попаданием последовательность
    /// получает общие страницы префикса по ссылке, а свои — только под
    /// остаток промпта; запись закрепляется, пока исполнитель не поднимет
    /// снимок (`finish_restore`).
    pub fn admit_prompt(&mut self, id: SeqId, prompt: &[u32]) -> Result<Admission, Rejected> {
        if self.seqs.contains_key(&id) {
            return Err(Rejected::AlreadyAdmitted);
        }
        self.prefix.stats.lookups += 1;
        let Some(snapshot) = self.prefix.best_match(prompt, self.kv.block_size) else {
            self.admit(id, prompt.len())?;
            return Ok(Admission::default());
        };
        if self.states.free.is_empty() {
            return Err(Rejected::NoStateSlot);
        }

        let index = self.prefix.position(snapshot).expect("запись только что найдена");
        self.prefix.entries[index].pins += 1;
        let (reused, shared, tail) = {
            let entry = &self.prefix.entries[index];
            let reused = entry.tokens.len();
            let full = reused / self.kv.block_size;
            let tail = (!reused.is_multiple_of(self.kv.block_size)).then(|| entry.blocks[full]);
            (reused, entry.blocks[..full].to_vec(), tail)
        };
        // Промпт длиннее префикса, поэтому своя страница есть всегда; при
        // неполном хвосте первая из своих — копия чужой.
        let need = self.kv.blocks_for(prompt.len()) - shared.len();
        let Some(fresh) = self.acquire_blocks(need) else {
            self.finish_restore(snapshot);
            return Err(Rejected::NoKvBlocks {
                need,
                free: self.kv.free_blocks(),
            });
        };
        let slot = self.states.acquire().expect("свободный слот уже проверен");
        self.kv.retain(&shared);
        let copy_block = tail.map(|source| (source, fresh[0]));
        let mut blocks = shared;
        blocks.extend(fresh);
        self.seqs.insert(
            id,
            Sequence {
                state_slot: slot,
                blocks,
                tokens: prompt.len(),
            },
        );

        let now = self.prefix.tick();
        if let Some(index) = self.prefix.position(snapshot) {
            self.prefix.entries[index].last_used = now;
        }
        self.prefix.stats.hits += 1;
        self.prefix.stats.reused_tokens += reused as u64;
        Ok(Admission {
            reused,
            restore: Some(PrefixRestore {
                snapshot,
                copy_block,
            }),
        })
    }

    /// Снимок поднят (или подъём отменён): запись снова можно вытеснять.
    pub fn finish_restore(&mut self, snapshot: u32) {
        if let Some(index) = self.prefix.position(snapshot) {
            let entry = &mut self.prefix.entries[index];
            debug_assert!(entry.pins > 0, "снимок {snapshot} не был закреплён");
            entry.pins = entry.pins.saturating_sub(1);
        }
    }

    /// Номер слота под новый снимок; при нехватке вытесняется самая давняя
    /// запись. `None` — кэш выключен или все записи закреплены.
    pub fn reserve_snapshot(&mut self) -> Option<u32> {
        if self.prefix.capacity == 0 {
            return None;
        }
        if self.prefix.free_snapshots.is_empty() && !self.prefix.evict_one(&mut self.kv, false) {
            return None;
        }
        self.prefix.free_snapshots.pop()
    }

    /// Снимок не состоялся (шаг откатили) — слот возвращается.
    pub fn cancel_snapshot(&mut self, snapshot: u32) {
        debug_assert!(!self.prefix.free_snapshots.contains(&snapshot));
        self.prefix.free_snapshots.push(snapshot);
    }

    /// Исполнитель снял состояние последовательности `id` после `tokens`:
    /// запись получает ссылки на страницы, покрывающие эти токены.
    pub fn commit_snapshot(&mut self, snapshot: u32, id: SeqId, tokens: Vec<u32>) {
        let Some(seq) = self.seqs.get(&id) else {
            self.cancel_snapshot(snapshot);
            return;
        };
        let now = self.prefix.tick();
        if let Some(existing) = self.prefix.entries.iter_mut().find(|e| e.tokens == tokens) {
            existing.last_used = now;
            self.prefix.free_snapshots.push(snapshot);
            return;
        }
        let covered = self.kv.blocks_for(tokens.len());
        debug_assert!(covered <= seq.blocks.len());
        let blocks = seq.blocks[..covered].to_vec();
        self.kv.retain(&blocks);
        self.prefix.entries.push(PrefixEntry {
            tokens,
            blocks,
            snapshot,
            last_used: now,
            pins: 0,
        });
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

    fn prompt(len: usize) -> Vec<u32> {
        (0..len as u32).collect()
    }

    /// Снимок в конце промпта `tokens` у последовательности `id`.
    fn snapshot(m: &mut CacheManager, id: SeqId, tokens: &[u32]) -> u32 {
        let snapshot = m.reserve_snapshot().unwrap();
        m.commit_snapshot(snapshot, id, tokens.to_vec());
        snapshot
    }

    #[test]
    fn prefix_hit_shares_full_pages_and_copies_the_tail() {
        let mut m = CacheManager::new(4, 100, 64);
        m.enable_prefix_cache(2);
        let first = prompt(150);
        assert_eq!(m.admit_prompt(1, &first).unwrap(), Admission::default());
        let snap = snapshot(&mut m, 1, &first);
        let owner = m.sequence(1).unwrap().blocks.clone();
        m.release(1);

        let second = prompt(200);
        let admission = m.admit_prompt(2, &second).unwrap();
        assert_eq!(admission.reused, 150);
        let restore = admission.restore.unwrap();
        assert_eq!(restore.snapshot, snap);
        let seq = m.sequence(2).unwrap();
        // Две полные страницы общие, третья (хвост 150 - 128) — копия.
        assert_eq!(&seq.blocks[..2], &owner[..2]);
        assert_eq!(restore.copy_block, Some((owner[2], seq.blocks[2])));
        assert_ne!(seq.blocks[2], owner[2]);
        assert_eq!(seq.blocks.len(), 4);
        assert_eq!(m.prefix_stats().hits, 1);
    }

    #[test]
    fn aligned_prefix_needs_no_copy() {
        let mut m = CacheManager::new(4, 100, 64);
        m.enable_prefix_cache(1);
        m.admit_prompt(1, &prompt(128)).unwrap();
        snapshot(&mut m, 1, &prompt(128));
        m.release(1);
        let admission = m.admit_prompt(2, &prompt(130)).unwrap();
        assert_eq!(admission.reused, 128);
        assert_eq!(admission.restore.unwrap().copy_block, None);
    }

    #[test]
    fn identical_prompt_does_not_hit() {
        // Логиты под первый ответ нужно чем-то посчитать: полный повтор
        // промпта идёт мимо кэша, а не с пустым префиллом.
        let mut m = CacheManager::new(4, 100, 64);
        m.enable_prefix_cache(1);
        m.admit_prompt(1, &prompt(100)).unwrap();
        snapshot(&mut m, 1, &prompt(100));
        m.release(1);
        assert_eq!(m.admit_prompt(2, &prompt(100)).unwrap().reused, 0);
    }

    #[test]
    fn different_prefix_misses() {
        let mut m = CacheManager::new(4, 100, 64);
        m.enable_prefix_cache(1);
        m.admit_prompt(1, &prompt(100)).unwrap();
        snapshot(&mut m, 1, &prompt(100));
        m.release(1);
        let mut other = prompt(150);
        other[10] = 999_999;
        assert_eq!(m.admit_prompt(2, &other).unwrap().reused, 0);
    }

    #[test]
    fn shared_pages_return_to_the_pool_only_after_every_owner() {
        let mut m = CacheManager::new(4, 20, 64);
        m.enable_prefix_cache(1);
        let total = m.kv().free_blocks();
        m.admit_prompt(1, &prompt(150)).unwrap();
        snapshot(&mut m, 1, &prompt(150));
        m.release(1);
        // Запись держит три страницы после ухода владельца.
        assert_eq!(m.kv().free_blocks(), total - 3);
        m.admit_prompt(2, &prompt(200)).unwrap();
        m.finish_restore(0);
        m.release(2);
        assert_eq!(m.kv().free_blocks(), total - 3);
        // Давление на пул вытесняет запись и возвращает её страницы.
        m.admit(3, 20 * 64).unwrap();
        assert_eq!(m.prefix_entries(), 0);
        assert_eq!(m.prefix_stats().evictions, 1);
        m.release(3);
        assert_eq!(m.kv().free_blocks(), total);
    }

    fn offset_prompt(start: u32, len: usize) -> Vec<u32> {
        (start..start + len as u32).collect()
    }

    #[test]
    fn page_pressure_skips_entry_held_by_live_sequence() {
        let mut m = CacheManager::new(4, 10, 64);
        m.enable_prefix_cache(2);
        // Запись A старше, но все её страницы держит идущий запрос 1: её
        // вытеснение ничего не освободит, а следующий ход диалога промахнётся.
        m.admit_prompt(1, &prompt(150)).unwrap();
        snapshot(&mut m, 1, &prompt(150));
        let idle = offset_prompt(1000, 150);
        m.admit_prompt(2, &idle).unwrap();
        snapshot(&mut m, 2, &idle);
        m.release(2);
        assert_eq!(m.kv().free_blocks(), 4);

        m.admit(3, 6 * 64).unwrap();
        assert_eq!(m.prefix_stats().evictions, 1);
        assert_eq!(m.prefix_entries(), 1);
        m.release(3);
        assert_eq!(m.admit_prompt(4, &prompt(200)).unwrap().reused, 150);
    }

    #[test]
    fn page_pressure_fails_without_evicting_live_entries() {
        let mut m = CacheManager::new(4, 4, 64);
        m.enable_prefix_cache(1);
        m.admit_prompt(1, &prompt(150)).unwrap();
        snapshot(&mut m, 1, &prompt(150));
        assert!(m.admit(2, 2 * 64).is_err());
        assert_eq!(m.prefix_entries(), 1);
        assert_eq!(m.prefix_stats().evictions, 0);
    }

    #[test]
    fn snapshot_pressure_keeps_entry_of_running_sequence() {
        let mut m = CacheManager::new(4, 100, 64);
        m.enable_prefix_cache(2);
        m.admit_prompt(1, &prompt(150)).unwrap();
        snapshot(&mut m, 1, &prompt(150));
        let idle = offset_prompt(1000, 150);
        m.admit_prompt(2, &idle).unwrap();
        snapshot(&mut m, 2, &idle);
        m.release(2);
        // Слотов нет: уходит простаивающая запись, хотя она новее.
        let fresh = offset_prompt(5000, 100);
        m.admit_prompt(3, &fresh).unwrap();
        snapshot(&mut m, 3, &fresh);
        assert_eq!(m.admit_prompt(4, &offset_prompt(1000, 200)).unwrap().reused, 0);
        assert_eq!(m.admit_prompt(5, &prompt(200)).unwrap().reused, 150);
    }

    #[test]
    fn snapshot_pressure_evicts_superseded_entry_first() {
        let mut m = CacheManager::new(4, 100, 64);
        m.enable_prefix_cache(2);
        // Ход N и ход N+1 одного диалога: N целиком лежит внутри N+1.
        m.admit_prompt(1, &prompt(100)).unwrap();
        snapshot(&mut m, 1, &prompt(100));
        m.release(1);
        let admission = m.admit_prompt(2, &prompt(200)).unwrap();
        m.finish_restore(admission.restore.unwrap().snapshot);
        snapshot(&mut m, 2, &prompt(180));
        m.release(2);
        // N трогали последним — по давности ушла бы N+1.
        let mut branch = prompt(100);
        branch.push(999_999);
        let admission = m.admit_prompt(3, &branch).unwrap();
        assert_eq!(admission.reused, 100);
        m.finish_restore(admission.restore.unwrap().snapshot);
        m.release(3);

        let other = offset_prompt(5000, 100);
        m.admit_prompt(4, &other).unwrap();
        snapshot(&mut m, 4, &other);
        m.release(4);
        assert_eq!(m.admit_prompt(5, &prompt(250)).unwrap().reused, 180);
    }

    #[test]
    fn pinned_entry_survives_until_restored() {
        let mut m = CacheManager::new(4, 100, 64);
        m.enable_prefix_cache(1);
        m.admit_prompt(1, &prompt(100)).unwrap();
        let snap = snapshot(&mut m, 1, &prompt(100));
        m.release(1);
        m.admit_prompt(2, &prompt(120)).unwrap();
        // Единственный слот снимка закреплён: новый снимок брать неоткуда.
        assert_eq!(m.reserve_snapshot(), None);
        m.finish_restore(snap);
        assert_eq!(m.reserve_snapshot(), Some(snap));
        assert_eq!(m.prefix_entries(), 0);
    }

    #[test]
    fn disabled_cache_is_plain_admission() {
        let mut m = CacheManager::new(4, 100, 64);
        assert_eq!(m.reserve_snapshot(), None);
        assert_eq!(m.admit_prompt(1, &prompt(100)).unwrap(), Admission::default());
    }
}

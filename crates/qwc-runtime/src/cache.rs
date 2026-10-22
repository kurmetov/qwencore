//! Cache manager for the hybrid architecture.
//!
//! The key difference from an ordinary transformer: a sequence occupies
//! TWO independent resources.
//!
//!   1. A DeltaNet state slot (48 layers) — a fixed 81 MB,
//!      independent of context length;
//!   2. Paged KV cache blocks (16 layers) — they grow as generation proceeds.
//!
//! An ordinary engine counts only the second. Because of this, on short contexts
//! (below ~2500 tokens) it misjudges capacity: there it's the state slot
//! that's scarce, while KV blocks are still plentiful.
//!
//! Hence two consequences for admission control:
//!   * a request can be admitted only when BOTH resources are available;
//!   * the bottleneck depends on context length, so the scheduler must
//!     look at both utilizations, not one.
//!
//! The prefix cache is built for the same hybridity. Shared KV pages are shared
//! by reference counting, as in an ordinary transformer, but they alone aren't enough: to
//! continue someone else's prefix you also need the DeltaNet state exactly at its
//! boundary. So a cache entry is a state snapshot (bf16, ~76 MB, held by
//! the executor) plus the pages covering the prefix. The snapshot is taken at the
//! history boundary, before the last `<|im_start|>`: the generation prompt doesn't make it into
//! the next turn's history, and a snapshot at the end of the prompt would never become
//! someone else's prefix.

use qwc_core::memory::{Budget, CacheConfig};
use std::collections::HashMap;

pub type SeqId = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rejected {
    /// No free state slots — hit the concurrency limit.
    NoStateSlot,
    /// Not enough KV blocks for the requested context.
    NoKvBlocks { need: usize, free: usize },
    /// The sequence is already admitted.
    AlreadyAdmitted,
    /// The sequence wasn't admitted or has already finished.
    UnknownSequence,
    /// Nothing to roll back: the sequence has no tokens.
    NothingToRollback,
}

impl std::fmt::Display for Rejected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoStateSlot => write!(f, "no free DeltaNet state slot"),
            Self::NoKvBlocks { need, free } => {
                write!(f, "need {need} KV blocks, {free} free")
            }
            Self::AlreadyAdmitted => write!(f, "sequence already admitted"),
            Self::UnknownSequence => write!(f, "sequence not admitted"),
            Self::NothingToRollback => write!(f, "no token to roll back"),
        }
    }
}

/// Pool of recurrent state slots. Slots are identical and indivisible:
/// the DeltaNet state has a fixed size regardless of context.
#[derive(Debug)]
pub struct StatePool {
    capacity: usize,
    free: Vec<u32>,
}

impl StatePool {
    pub fn new(capacity: usize) -> Self {
        assert!(
            u32::try_from(capacity).is_ok(),
            "state slot count doesn't fit in u32"
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
            "double free of slot {slot}"
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

/// Paged KV cache. Blocks of equal size, a block table per
/// sequence — with this design there's no external fragmentation.
///
/// A page can belong to several owners at once: sequences
/// and prefix cache entries. It returns to the pool when the last one leaves.
#[derive(Debug)]
pub struct KvPool {
    block_size: usize,
    capacity: usize,
    free: Vec<u32>,
    refs: Vec<u32>,
}

impl KvPool {
    pub fn new(capacity: usize, block_size: usize) -> Self {
        assert!(block_size > 0, "KV block size must be positive");
        assert!(
            u32::try_from(capacity).is_ok(),
            "KV block count doesn't fit in u32"
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
            debug_assert!(self.refs[block as usize] > 0, "reference to a free block {block}");
            self.refs[block as usize] += 1;
        }
    }

    fn release(&mut self, blocks: &[u32]) {
        for &block in blocks {
            let refs = &mut self.refs[block as usize];
            debug_assert!(*refs > 0, "double free of block {block}");
            *refs -= 1;
            if *refs == 0 {
                self.free.push(block);
            }
        }
    }
}

/// State of an admitted sequence.
#[derive(Debug)]
pub struct Sequence {
    pub state_slot: u32,
    pub blocks: Vec<u32>,
    /// How many tokens are already in the KV cache.
    pub tokens: usize,
}

/// How to continue a cached prefix: the state snapshot to
/// restore into the sequence's slot, and the page to copy
/// if the prefix ends in the middle of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefixRestore {
    pub snapshot: u32,
    /// `(from, to)`: the prefix tail lies in an incomplete page, and the new sequence
    /// will keep writing into it — the page is copied.
    pub copy_block: Option<(u32, u32)>,
}

/// Result of admitting a request: how many prompt tokens are already computed.
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
    /// How many admitted sequences are waiting to restore this snapshot.
    /// A pinned entry can't be evicted: its snapshot slot would be overwritten.
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

/// Prefix cache counters for logs and measurements.
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

    /// The longest entry that is entirely a prefix of the prompt and
    /// shorter than it: at least one prompt token has to be run, otherwise there's
    /// nothing to get logits for the first response token from.
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

    /// Evict the least valuable unpinned entry.
    ///
    /// Nested entries go first: their prefix lies entirely in a longer
    /// entry of the same dialog. Next — entries whose pages aren't held by
    /// any live sequence: an ongoing request's entry is needed by
    /// its next turn, however long ago it was touched. Within that — by age.
    ///
    /// `for_pages` — we evict for KV pages, not for a snapshot slot. Then
    /// an entry all of whose pages are held by live sequences is not
    /// touched: it won't free anything, and the cache would lose it.
    fn evict_one(&mut self, kv: &mut KvPool, for_pages: bool) -> bool {
        let mut holders = vec![0u32; kv.capacity()];
        for entry in &self.entries {
            for &block in &entry.blocks {
                holders[block as usize] += 1;
            }
        }
        // The page is held by someone other than entries — so, a live sequence.
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

    /// Enables the prefix cache with `snapshots` state snapshots. Memory for
    /// snapshots is held by the executor; only their numbers are here.
    pub fn enable_prefix_cache(&mut self, snapshots: usize) {
        assert!(self.prefix.entries.is_empty(), "prefix cache already in use");
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

    /// Blocks from the pool, when short — at the cost of evicting prefix cache
    /// entries. Live sequences matter more than any cache.
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

    /// Splits available VRAM between the two subsystems.
    ///
    /// `target_context` is the context length we optimize for. We make exactly as many
    /// state slots as sequences of that length will fit;
    /// the rest goes to KV blocks. The point is that an extra state slot
    /// is useless if there are no KV blocks for it, and vice versa.
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
        // Check the slot before the blocks: allocated blocks would have to be rolled back,
        // and the cache evicted for them couldn't be.
        if self.states.free.is_empty() {
            return Err(Rejected::NoStateSlot);
        }
        let Some(blocks) = self.acquire_blocks(need) else {
            return Err(Rejected::NoKvBlocks {
                need,
                free: self.kv.free_blocks(),
            });
        };
        let slot = self.states.acquire().expect("free slot already checked");
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

    /// Adds one generated token, allocating a block if needed.
    /// The scheduler calls this before launching decode: this way memory
    /// is reserved in advance and the GPU step can no longer fail due to cache growth.
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

    /// Rolls back the last decode token reservation. If `append_token`
    /// added a new block, it's returned to the pool as well.
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

    /// Admit a request, continuing a cached prefix if possible.
    ///
    /// Without a hit this is an ordinary `admit`. With a hit the sequence
    /// gets the prefix's shared pages by reference, and its own only for
    /// the rest of the prompt; the entry is pinned until the executor restores
    /// the snapshot (`finish_restore`).
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

        let index = self.prefix.position(snapshot).expect("entry was just found");
        self.prefix.entries[index].pins += 1;
        let (reused, shared, tail) = {
            let entry = &self.prefix.entries[index];
            let reused = entry.tokens.len();
            let full = reused / self.kv.block_size;
            let tail = (!reused.is_multiple_of(self.kv.block_size)).then(|| entry.blocks[full]);
            (reused, entry.blocks[..full].to_vec(), tail)
        };
        // The prompt is longer than the prefix, so there's always an own page; with
        // an incomplete tail the first own page is a copy of the shared one.
        let need = self.kv.blocks_for(prompt.len()) - shared.len();
        let Some(fresh) = self.acquire_blocks(need) else {
            self.finish_restore(snapshot);
            return Err(Rejected::NoKvBlocks {
                need,
                free: self.kv.free_blocks(),
            });
        };
        let slot = self.states.acquire().expect("free slot already checked");
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

    /// The snapshot is restored (or the restore cancelled): the entry can be evicted again.
    pub fn finish_restore(&mut self, snapshot: u32) {
        if let Some(index) = self.prefix.position(snapshot) {
            let entry = &mut self.prefix.entries[index];
            debug_assert!(entry.pins > 0, "snapshot {snapshot} was not pinned");
            entry.pins = entry.pins.saturating_sub(1);
        }
    }

    /// Slot number for a new snapshot; when short, the oldest entry is evicted.
    /// `None` — the cache is off or all entries are pinned.
    pub fn reserve_snapshot(&mut self) -> Option<u32> {
        if self.prefix.capacity == 0 {
            return None;
        }
        if self.prefix.free_snapshots.is_empty() && !self.prefix.evict_one(&mut self.kv, false) {
            return None;
        }
        self.prefix.free_snapshots.pop()
    }

    /// The snapshot didn't happen (the step was rolled back) — the slot is returned.
    pub fn cancel_snapshot(&mut self, snapshot: u32) {
        debug_assert!(!self.prefix.free_snapshots.contains(&snapshot));
        self.prefix.free_snapshots.push(snapshot);
    }

    /// The executor captured the state of sequence `id` after `tokens`:
    /// the entry gets references to the pages covering these tokens.
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

    /// Which subsystem is closer to exhaustion. This is exactly what
    /// the scheduler needs to know: pressure on them works differently.
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
            "state slots"
        } else {
            "KV blocks"
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
        // State slots ran out while KV blocks are still plentiful.
        assert_eq!(m.admit(3, 128), Err(Rejected::NoStateSlot));
        assert!(m.kv().free_blocks() > 900);

        m.release(1);
        assert!(m.admit(3, 128).is_ok());
    }

    #[test]
    fn kv_can_be_the_binding_constraint() {
        // Many slots, few blocks.
        let mut m = CacheManager::new(64, 10, 64);
        assert!(m.admit(1, 640).is_ok()); // exactly 10 blocks
        assert!(matches!(m.admit(2, 64), Err(Rejected::NoKvBlocks { .. })));
        // The state slot isn't consumed in that case.
        assert_eq!(m.states().in_use(), 1);
    }

    #[test]
    fn bottleneck_flips_with_context_length() {
        // With the same pool layout short requests hit
        // state slots, and long ones hit KV blocks.
        let mut short = CacheManager::new(64, 100, 64);
        let mut id = 0;
        while short.admit(id, 64).is_ok() {
            id += 1;
        }
        assert_eq!(short.pressure().bottleneck(), "state slots");

        let mut long = CacheManager::new(64, 100, 64);
        let mut id = 0;
        while long.admit(id, 640).is_ok() {
            id += 1;
        }
        assert_eq!(long.pressure().bottleneck(), "KV blocks");
    }

    #[test]
    fn budget_split_matches_memory_model() {
        let b = rtx5090(WeightPlan::default().bytes(), 2 * GB);
        let cfg = CacheConfig::default();

        // The number of slots must match the independent estimate from qwc-core.
        for ctx in [4096usize, 8192, 32768] {
            let m = CacheManager::from_budget(&b, &cfg, ctx);
            assert_eq!(
                m.states().capacity(),
                b.max_concurrency(&cfg, ctx),
                "context {ctx}"
            );
        }
    }

    #[test]
    fn append_allocates_lazily() {
        let mut m = CacheManager::new(4, 100, 64);
        m.admit(1, 64).unwrap();
        assert_eq!(m.sequence(1).unwrap().blocks.len(), 1);

        // The block is filled exactly, the next token needs a new one.
        m.append_token(1).unwrap();
        assert_eq!(m.sequence(1).unwrap().blocks.len(), 2);
        assert_eq!(m.sequence(1).unwrap().tokens, 65);

        // Within a block there are no new allocations.
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
    #[should_panic(expected = "KV block size must be positive")]
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
        // The pool is statically planned for the max context, but real
        // requests are shorter. They exhaust the slots, leaving KV underused.
        let mut m = manager(32 * 1024);
        let mut id = 0;
        while m.admit(id, 1024).is_ok() {
            id += 1;
        }
        let p = m.pressure();
        assert!(p.states > 0.99, "slots {:.2}", p.states);
        assert!(
            p.kv < 0.5,
            "KV used at {:.2}, expected underuse",
            p.kv
        );
    }

    fn prompt(len: usize) -> Vec<u32> {
        (0..len as u32).collect()
    }

    /// A snapshot at the end of prompt `tokens` for sequence `id`.
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
        // Two full pages are shared, the third (tail 150 - 128) is a copy.
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
        // Logits for the first response have to be computed with something: a full repeat
        // of the prompt bypasses the cache rather than going with an empty prefill.
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
        // The entry holds three pages after the owner leaves.
        assert_eq!(m.kv().free_blocks(), total - 3);
        m.admit_prompt(2, &prompt(200)).unwrap();
        m.finish_restore(0);
        m.release(2);
        assert_eq!(m.kv().free_blocks(), total - 3);
        // Pool pressure evicts the entry and returns its pages.
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
        // Entry A is older, but all its pages are held by ongoing request 1: its
        // eviction frees nothing, and the next dialog turn would miss.
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
        // No slots: the idle entry goes, even though it's newer.
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
        // Turn N and turn N+1 of one dialog: N lies entirely inside N+1.
        m.admit_prompt(1, &prompt(100)).unwrap();
        snapshot(&mut m, 1, &prompt(100));
        m.release(1);
        let admission = m.admit_prompt(2, &prompt(200)).unwrap();
        m.finish_restore(admission.restore.unwrap().snapshot);
        snapshot(&mut m, 2, &prompt(180));
        m.release(2);
        // N was touched last — by age N+1 would have gone.
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
        // The only snapshot slot is pinned: there's nowhere to take a new snapshot from.
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

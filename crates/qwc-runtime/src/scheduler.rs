//! Continuous batching поверх двух независимых пулов кэша.
//!
//! Планировщик резервирует память до отправки работы на GPU. В каждый момент
//! разрешён только один in-flight batch: это делает переходы состояний
//! однозначными и позволяет полностью откатить план, если запуск кернела не
//! состоялся.

use crate::cache::{CacheManager, PrefixRestore, Rejected, SeqId};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchedulerConfig {
    /// Максимум последовательностей в одном смешанном batch.
    pub max_num_seqs: usize,
    /// Общий token budget: decode стоит один токен, prefill — размер чанка.
    pub max_num_batched_tokens: usize,
}

impl SchedulerConfig {
    pub fn new(max_num_seqs: usize, max_num_batched_tokens: usize) -> Self {
        assert!(
            max_num_seqs > 0,
            "batch должен вмещать хотя бы одну последовательность"
        );
        assert!(
            max_num_batched_tokens > 0,
            "token budget должен быть положительным"
        );
        Self {
            max_num_seqs,
            max_num_batched_tokens,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Request {
    pub id: SeqId,
    pub prompt_tokens: usize,
    pub max_new_tokens: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillChunk {
    pub id: SeqId,
    /// Смещение чанка в исходном prompt.
    pub offset: usize,
    pub tokens: usize,
}

/// Поднять снимок префикса в слот до того, как пойдёт первый чанк.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefixRestoreOp {
    pub id: SeqId,
    pub state_slot: u32,
    pub snapshot: u32,
    /// `(откуда, куда)`: неполную страницу префикса надо скопировать до шага.
    pub copy_block: Option<(u32, u32)>,
}

/// Снять состояние слота в снимок сразу после чанка, закончившего промпт.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefixSaveOp {
    pub id: SeqId,
    pub state_slot: u32,
    pub snapshot: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Batch {
    pub prefill: Vec<PrefillChunk>,
    pub decode: Vec<SeqId>,
    /// Операции кэша префиксов. Исполнитель обязан выполнить их в этом же
    /// шаге: подъём — до прохода, снятие — внутри, после скана чанка.
    pub restores: Vec<PrefixRestoreOp>,
    pub saves: Vec<PrefixSaveOp>,
}

impl Batch {
    pub fn num_seqs(&self) -> usize {
        self.prefill.len() + self.decode.len()
    }

    pub fn num_tokens(&self) -> usize {
        self.prefill.iter().map(|chunk| chunk.tokens).sum::<usize>() + self.decode.len()
    }

    pub fn is_empty(&self) -> bool {
        self.prefill.is_empty() && self.decode.is_empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Length,
    Stopped,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Completion {
    pub id: SeqId,
    pub generated_tokens: usize,
    pub reason: FinishReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmitError {
    Duplicate(SeqId),
    EmptyPrompt,
    NoOutputTokens,
    NoStateCapacity,
    ContextTooLong { need: usize, capacity: usize },
}

impl std::fmt::Display for SubmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Duplicate(id) => write!(f, "SeqId {id} уже находится в scheduler"),
            Self::EmptyPrompt => write!(f, "prompt не содержит токенов"),
            Self::NoOutputTokens => write!(f, "max_new_tokens должен быть положительным"),
            Self::NoStateCapacity => write!(f, "пул не содержит state-слотов"),
            Self::ContextTooLong { need, capacity } => {
                write!(
                    f,
                    "prompt и генерация требуют {need} KV-блоков, ёмкость пула {capacity}"
                )
            }
        }
    }
}

impl std::error::Error for SubmitError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerError {
    BatchInFlight,
    NoBatchInFlight,
    SequenceInFlight(SeqId),
    InvalidStopped(SeqId),
    DuplicateStopped(SeqId),
    Invariant(SeqId),
    Cache(Rejected),
}

impl std::fmt::Display for SchedulerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BatchInFlight => write!(f, "предыдущий batch ещё не завершён"),
            Self::NoBatchInFlight => write!(f, "нет batch для завершения или отката"),
            Self::SequenceInFlight(id) => {
                write!(f, "SeqId {id} нельзя отменить во время исполнения batch")
            }
            Self::InvalidStopped(id) => write!(f, "SeqId {id} не входил в decode batch"),
            Self::DuplicateStopped(id) => write!(f, "SeqId {id} дважды указан как завершённый"),
            Self::Invariant(id) => write!(f, "нарушено внутреннее состояние SeqId {id}"),
            Self::Cache(err) => err.fmt(f),
        }
    }
}

impl std::error::Error for SchedulerError {}

impl From<Rejected> for SchedulerError {
    fn from(value: Rejected) -> Self {
        Self::Cache(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    PrefillReady,
    PrefillInFlight,
    DecodeReady,
    DecodeInFlight,
}

#[derive(Debug)]
struct ActiveRequest {
    request: Request,
    reserved_blocks: usize,
    prefilled: usize,
    generated: usize,
    phase: Phase,
    /// Снимок префикса ещё не поднят: первый чанк его не отработал.
    restore: Option<PrefixRestore>,
}

/// FIFO admission + round-robin decode. Длинный prompt, временно не
/// помещающийся в KV, не блокирует меньший запрос позади него.
#[derive(Debug)]
pub struct Scheduler {
    config: SchedulerConfig,
    cache: CacheManager,
    waiting: VecDeque<Request>,
    prefill_ready: VecDeque<SeqId>,
    decode_ready: VecDeque<SeqId>,
    active: HashMap<SeqId, ActiveRequest>,
    /// Логически обещанные активным запросам KV-блоки на полный lifetime.
    /// Физически CacheManager выделяет их лениво.
    committed_kv_blocks: usize,
    in_flight: Option<Batch>,
    /// Токены промптов, по которым ищется и сохраняется кэш префиксов.
    /// Без них запрос принимается обычным путём.
    prompts: HashMap<SeqId, PromptTokens>,
}

#[derive(Debug)]
struct PromptTokens {
    tokens: Arc<[u32]>,
    /// Где снять снимок. Не конец промпта: хвост промпта генерации
    /// (`<|im_start|>assistant\n<think>\n`) в историю следующего хода не
    /// попадает, и снимок после него никогда не стал бы её префиксом.
    snapshot_at: usize,
}

impl Scheduler {
    pub fn new(config: SchedulerConfig, cache: CacheManager) -> Self {
        Self {
            config,
            cache,
            waiting: VecDeque::new(),
            prefill_ready: VecDeque::new(),
            decode_ready: VecDeque::new(),
            active: HashMap::new(),
            committed_kv_blocks: 0,
            in_flight: None,
            prompts: HashMap::new(),
        }
    }

    /// Включает кэш префиксов на `snapshots` снимков. Исполнитель должен
    /// держать столько же слотов под снимки.
    pub fn enable_prefix_cache(&mut self, snapshots: usize) {
        self.cache.enable_prefix_cache(snapshots);
    }

    /// То же, что `submit`, но с самими токенами промпта: только такой
    /// запрос может продолжить закэшированный префикс и оставить свой.
    ///
    /// `snapshot_at` — длина префикса, который стоит закэшировать: у чата
    /// это граница истории перед промптом генерации. Чанк префилла режется
    /// на ней, чтобы снимок пришёлся ровно туда.
    pub fn submit_prompt(
        &mut self,
        request: Request,
        prompt: Arc<[u32]>,
        snapshot_at: usize,
    ) -> Result<(), SubmitError> {
        assert_eq!(prompt.len(), request.prompt_tokens, "длина промпта не сходится");
        assert!(snapshot_at <= prompt.len(), "точка снимка за концом промпта");
        self.submit(request)?;
        self.prompts.insert(
            request.id,
            PromptTokens {
                tokens: prompt,
                snapshot_at,
            },
        );
        Ok(())
    }

    pub fn submit(&mut self, request: Request) -> Result<(), SubmitError> {
        if self.active.contains_key(&request.id)
            || self.waiting.iter().any(|queued| queued.id == request.id)
        {
            return Err(SubmitError::Duplicate(request.id));
        }
        if request.prompt_tokens == 0 {
            return Err(SubmitError::EmptyPrompt);
        }
        if request.max_new_tokens == 0 {
            return Err(SubmitError::NoOutputTokens);
        }
        if self.cache.states().capacity() == 0 {
            return Err(SubmitError::NoStateCapacity);
        }

        let context_tokens = request.prompt_tokens.saturating_add(request.max_new_tokens);
        let need = self.cache.kv().blocks_for(context_tokens);
        let capacity = self.cache.kv().capacity();
        if need > capacity {
            return Err(SubmitError::ContextTooLong { need, capacity });
        }

        self.waiting.push_back(request);
        Ok(())
    }

    /// Формирует смешанный continuous batch. Decode получает приоритет ради
    /// низкого ITL, остаток лимита заполняется chunked prefill.
    ///
    /// Для decode KV под следующий токен резервируется прямо здесь. До вызова
    /// `complete_batch` или `abort_batch` новый batch планировать нельзя.
    pub fn next_batch(&mut self) -> Result<Option<Batch>, SchedulerError> {
        if self.in_flight.is_some() {
            return Err(SchedulerError::BatchInFlight);
        }

        let mut batch = Batch::default();
        let decode_candidates = self.decode_ready.len();
        for _ in 0..decode_candidates {
            if batch.num_seqs() == self.config.max_num_seqs
                || batch.num_tokens() == self.config.max_num_batched_tokens
            {
                break;
            }

            let id = self.decode_ready.pop_front().unwrap();
            match self.cache.append_token(id) {
                Ok(()) => {
                    self.set_phase(id, Phase::DecodeReady, Phase::DecodeInFlight)?;
                    batch.decode.push(id);
                }
                Err(Rejected::NoKvBlocks { .. }) => self.decode_ready.push_back(id),
                Err(err) => return Err(err.into()),
            }
        }

        while batch.num_seqs() < self.config.max_num_seqs
            && batch.num_tokens() < self.config.max_num_batched_tokens
        {
            let id = match self.prefill_ready.pop_front() {
                Some(id) => id,
                None => match self.admit_one()? {
                    Some(id) => id,
                    None => break,
                },
            };

            let active = self.active.get(&id).ok_or(SchedulerError::Invariant(id))?;
            if active.phase != Phase::PrefillReady {
                return Err(SchedulerError::Invariant(id));
            }
            let remaining = active.request.prompt_tokens - active.prefilled;
            let budget = self.config.max_num_batched_tokens - batch.num_tokens();
            let mut tokens = remaining.min(budget);
            let offset = active.prefilled;
            let restore = active.restore;
            // Точка снимка внутри чанка — чанк кончается на ней, остаток
            // промпта уходит следующим шагом.
            let snapshot_at = self
                .prompts
                .get(&id)
                .map(|prompt| prompt.snapshot_at)
                .filter(|&at| self.cache.prefix_capacity() > 0 && at > offset && at <= offset + tokens);
            if let Some(at) = snapshot_at {
                tokens = at - offset;
            }

            let state_slot = self
                .cache
                .sequence(id)
                .ok_or(SchedulerError::Invariant(id))?
                .state_slot;
            if let Some(restore) = restore {
                batch.restores.push(PrefixRestoreOp {
                    id,
                    state_slot,
                    snapshot: restore.snapshot,
                    copy_block: restore.copy_block,
                });
            }
            // Следующий ход того же диалога начнётся с этой истории целиком.
            if snapshot_at.is_some_and(|at| at >= self.cache.kv().block_size())
                && let Some(snapshot) = self.cache.reserve_snapshot()
            {
                batch.saves.push(PrefixSaveOp {
                    id,
                    state_slot,
                    snapshot,
                });
            }

            self.set_phase(id, Phase::PrefillReady, Phase::PrefillInFlight)?;
            batch.prefill.push(PrefillChunk { id, offset, tokens });
        }

        if batch.is_empty() {
            return Ok(None);
        }
        self.in_flight = Some(batch.clone());
        Ok(Some(batch))
    }

    /// Подтверждает успешно исполненный batch. `stopped` содержит decode-
    /// последовательности, для которых модель вернула EOS/stop condition.
    pub fn complete_batch(&mut self, stopped: &[SeqId]) -> Result<Vec<Completion>, SchedulerError> {
        self.complete_batch_multi(stopped, &[])
    }

    /// То же, но decode-последовательность могла выдать больше одного токена.
    ///
    /// Спекуляция за шаг принимает от одного до k+1 токенов, и планировщик
    /// обязан знать сколько: иначе `generated` разъедется с реальностью и
    /// запрос закончится не на своей длине. Отсутствующие в `produced`
    /// последовательности считаются выдавшими один токен.
    pub fn complete_batch_multi(
        &mut self,
        stopped: &[SeqId],
        produced: &[(SeqId, usize)],
    ) -> Result<Vec<Completion>, SchedulerError> {
        let Some(in_flight) = self.in_flight.as_ref() else {
            return Err(SchedulerError::NoBatchInFlight);
        };
        for &(id, tokens) in produced {
            if tokens == 0 {
                return Err(SchedulerError::Invariant(id));
            }
            if !in_flight.decode.contains(&id) {
                return Err(SchedulerError::InvalidStopped(id));
            }
        }

        let mut stopped_set = HashSet::with_capacity(stopped.len());
        for &id in stopped {
            if !stopped_set.insert(id) {
                return Err(SchedulerError::DuplicateStopped(id));
            }
            if !in_flight.decode.contains(&id) {
                return Err(SchedulerError::InvalidStopped(id));
            }
        }

        let batch = self.in_flight.take().unwrap();
        for op in &batch.restores {
            if let Some(active) = self.active.get_mut(&op.id) {
                active.restore = None;
            }
            self.cache.finish_restore(op.snapshot);
        }
        for op in &batch.saves {
            match self.prompts.get(&op.id) {
                Some(prompt) => self.cache.commit_snapshot(
                    op.snapshot,
                    op.id,
                    prompt.tokens[..prompt.snapshot_at].to_vec(),
                ),
                None => self.cache.cancel_snapshot(op.snapshot),
            }
        }
        for chunk in batch.prefill {
            let active = self
                .active
                .get_mut(&chunk.id)
                .ok_or(SchedulerError::Invariant(chunk.id))?;
            if active.phase != Phase::PrefillInFlight {
                return Err(SchedulerError::Invariant(chunk.id));
            }
            active.prefilled += chunk.tokens;
            if active.prefilled == active.request.prompt_tokens {
                active.phase = Phase::DecodeReady;
                self.decode_ready.push_back(chunk.id);
            } else {
                active.phase = Phase::PrefillReady;
                self.prefill_ready.push_back(chunk.id);
            }
        }

        let mut completed = Vec::new();
        for id in batch.decode {
            let active = self
                .active
                .get_mut(&id)
                .ok_or(SchedulerError::Invariant(id))?;
            if active.phase != Phase::DecodeInFlight {
                return Err(SchedulerError::Invariant(id));
            }
            let tokens = produced
                .iter()
                .find(|&&(other, _)| other == id)
                .map_or(1, |&(_, tokens)| tokens);
            active.generated = (active.generated + tokens).min(active.request.max_new_tokens);

            let reason = if stopped_set.contains(&id) {
                Some(FinishReason::Stopped)
            } else if active.generated >= active.request.max_new_tokens {
                Some(FinishReason::Length)
            } else {
                None
            };

            if let Some(reason) = reason {
                let generated_tokens = active.generated;
                let active = self.active.remove(&id).unwrap();
                self.committed_kv_blocks -= active.reserved_blocks;
                self.cache.release(id);
                self.prompts.remove(&id);
                completed.push(Completion {
                    id,
                    generated_tokens,
                    reason,
                });
            } else {
                active.phase = Phase::DecodeReady;
                self.decode_ready.push_back(id);
            }
        }
        Ok(completed)
    }

    /// Резервирует в KV ещё `tokens` позиций для decode-последовательности
    /// сверх той одной, что уже взял `next_batch`.
    ///
    /// Спекулятивная проверка пишет k+1 строк, а не одну, и страницы под них
    /// нужны до прогона: узнать число принятых можно только после него.
    /// Возвращает, сколько удалось зарезервировать — при исчерпании пула
    /// глубина черновика просто уменьшается.
    pub fn reserve_extra(&mut self, id: SeqId, tokens: usize) -> usize {
        let mut reserved = 0;
        while reserved < tokens {
            match self.cache.append_token(id) {
                Ok(()) => reserved += 1,
                Err(_) => break,
            }
        }
        reserved
    }

    /// Возвращает в пул позиции, которые спекуляция зарезервировала, но
    /// проверка отвергла.
    pub fn release_extra(&mut self, id: SeqId, tokens: usize) -> Result<(), SchedulerError> {
        for _ in 0..tokens {
            self.cache.rollback_token(id)?;
        }
        Ok(())
    }

    /// Откатывает ещё не исполненный batch и возвращает работу в начало
    /// соответствующих очередей. Резервации decode-токенов освобождаются.
    pub fn abort_batch(&mut self) -> Result<(), SchedulerError> {
        let Some(batch) = self.in_flight.take() else {
            return Err(SchedulerError::NoBatchInFlight);
        };

        // Подъёмы остаются за последовательностями и уйдут со следующим
        // чанком; снимки не сняты — их слоты свободны.
        for op in &batch.saves {
            self.cache.cancel_snapshot(op.snapshot);
        }
        for chunk in batch.prefill.into_iter().rev() {
            self.set_phase(chunk.id, Phase::PrefillInFlight, Phase::PrefillReady)?;
            self.prefill_ready.push_front(chunk.id);
        }
        for id in batch.decode.into_iter().rev() {
            self.cache.rollback_token(id)?;
            self.set_phase(id, Phase::DecodeInFlight, Phase::DecodeReady)?;
            self.decode_ready.push_front(id);
        }
        Ok(())
    }

    /// Удаляет ожидающий или готовый запрос и сразу освобождает его кэш.
    /// In-flight запрос сначала нужно завершить либо откатить целиком вместе
    /// с batch, иначе GPU мог бы обратиться к уже освобождённым блокам.
    pub fn cancel(&mut self, id: SeqId) -> Result<bool, SchedulerError> {
        if self.in_flight.as_ref().is_some_and(|batch| {
            batch.decode.contains(&id) || batch.prefill.iter().any(|c| c.id == id)
        }) {
            return Err(SchedulerError::SequenceInFlight(id));
        }

        if let Some(index) = self.waiting.iter().position(|request| request.id == id) {
            self.waiting.remove(index);
            self.prompts.remove(&id);
            return Ok(true);
        }
        if let Some(active) = self.active.remove(&id) {
            self.prefill_ready.retain(|queued| *queued != id);
            self.decode_ready.retain(|queued| *queued != id);
            self.committed_kv_blocks -= active.reserved_blocks;
            if let Some(restore) = active.restore {
                self.cache.finish_restore(restore.snapshot);
            }
            self.cache.release(id);
            self.prompts.remove(&id);
            return Ok(true);
        }
        Ok(false)
    }

    /// Сколько токенов запросу ещё осталось выдать. Спекуляция обязана это
    /// знать: шаг выдаёт от одного до k+1 токенов, и перескочить через
    /// `max_new_tokens` значит посчитать лишнее и сделать лишнюю работу.
    pub fn remaining_tokens(&self, id: SeqId) -> Option<usize> {
        self.active
            .get(&id)
            .map(|active| active.request.max_new_tokens - active.generated)
    }

    pub fn waiting(&self) -> usize {
        self.waiting.len()
    }

    pub fn active(&self) -> usize {
        self.active.len()
    }

    pub fn cache(&self) -> &CacheManager {
        &self.cache
    }

    pub fn committed_kv_blocks(&self) -> usize {
        self.committed_kv_blocks
    }

    pub fn in_flight(&self) -> Option<&Batch> {
        self.in_flight.as_ref()
    }

    fn admit_one(&mut self) -> Result<Option<SeqId>, SchedulerError> {
        let candidates = self.waiting.len();
        for _ in 0..candidates {
            let request = self.waiting.pop_front().unwrap();
            let lifetime_tokens = request.prompt_tokens.saturating_add(request.max_new_tokens);
            let reserved_blocks = self.cache.kv().blocks_for(lifetime_tokens);
            if reserved_blocks > self.cache.kv().capacity() - self.committed_kv_blocks {
                self.waiting.push_back(request);
                continue;
            }
            let admitted = match self.prompts.get(&request.id) {
                Some(prompt) => self.cache.admit_prompt(request.id, &prompt.tokens),
                None => self
                    .cache
                    .admit(request.id, request.prompt_tokens)
                    .map(|()| Default::default()),
            };
            match admitted {
                Ok(admission) => {
                    let id = request.id;
                    self.committed_kv_blocks += reserved_blocks;
                    self.active.insert(
                        id,
                        ActiveRequest {
                            request,
                            reserved_blocks,
                            prefilled: admission.reused,
                            generated: 0,
                            phase: Phase::PrefillReady,
                            restore: admission.restore,
                        },
                    );
                    return Ok(Some(id));
                }
                Err(Rejected::NoStateSlot) => {
                    self.waiting.push_front(request);
                    return Ok(None);
                }
                Err(Rejected::NoKvBlocks { .. }) => self.waiting.push_back(request),
                Err(err) => return Err(err.into()),
            }
        }
        Ok(None)
    }

    fn set_phase(&mut self, id: SeqId, expected: Phase, next: Phase) -> Result<(), SchedulerError> {
        let active = self
            .active
            .get_mut(&id)
            .ok_or(SchedulerError::Invariant(id))?;
        if active.phase != expected {
            return Err(SchedulerError::Invariant(id));
        }
        active.phase = next;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scheduler(states: usize, blocks: usize, max_seqs: usize, tokens: usize) -> Scheduler {
        Scheduler::new(
            SchedulerConfig::new(max_seqs, tokens),
            CacheManager::new(states, blocks, 64),
        )
    }

    fn request(id: SeqId, prompt_tokens: usize, max_new_tokens: usize) -> Request {
        Request {
            id,
            prompt_tokens,
            max_new_tokens,
        }
    }

    #[test]
    fn long_prefill_is_split_into_chunks() {
        let mut s = scheduler(2, 100, 2, 4);
        s.submit(request(1, 10, 1)).unwrap();

        for (offset, tokens) in [(0, 4), (4, 4), (8, 2)] {
            let batch = s.next_batch().unwrap().unwrap();
            assert_eq!(
                batch.prefill,
                vec![PrefillChunk {
                    id: 1,
                    offset,
                    tokens
                }]
            );
            assert!(batch.decode.is_empty());
            s.complete_batch(&[]).unwrap();
        }

        let decode = s.next_batch().unwrap().unwrap();
        assert_eq!(decode.decode, vec![1]);
    }

    #[test]
    fn continuous_batch_mixes_decode_and_prefill() {
        let mut s = scheduler(2, 100, 2, 8);
        s.submit(request(1, 4, 2)).unwrap();
        s.next_batch().unwrap();
        s.complete_batch(&[]).unwrap();

        s.submit(request(2, 4, 1)).unwrap();
        let batch = s.next_batch().unwrap().unwrap();
        assert_eq!(batch.decode, vec![1]);
        assert_eq!(
            batch.prefill,
            vec![PrefillChunk {
                id: 2,
                offset: 0,
                tokens: 4
            }]
        );
        assert_eq!(batch.num_tokens(), 5);
    }

    #[test]
    fn length_completion_releases_both_resources() {
        let mut s = scheduler(1, 2, 1, 64);
        s.submit(request(1, 64, 2)).unwrap();
        s.next_batch().unwrap();
        s.complete_batch(&[]).unwrap();

        s.next_batch().unwrap();
        assert!(s.complete_batch(&[]).unwrap().is_empty());
        s.next_batch().unwrap();
        let done = s.complete_batch(&[]).unwrap();

        assert_eq!(
            done,
            vec![Completion {
                id: 1,
                generated_tokens: 2,
                reason: FinishReason::Length,
            }]
        );
        assert_eq!(s.active(), 0);
        assert_eq!(s.cache().states().in_use(), 0);
        assert_eq!(s.cache().kv().free_blocks(), 2);
    }

    #[test]
    fn kv_blocked_head_does_not_hide_smaller_request() {
        let mut s = scheduler(3, 4, 3, 256);
        s.submit(request(1, 128, 2)).unwrap();
        s.next_batch().unwrap();
        s.complete_batch(&[]).unwrap();

        // После резервации decode для seq 1 свободен один блок. Seq 2 требует
        // три, но seq 3 помещается и должен попасть в этот же batch.
        s.submit(request(2, 192, 1)).unwrap();
        s.submit(request(3, 63, 1)).unwrap();
        let batch = s.next_batch().unwrap().unwrap();
        assert_eq!(batch.decode, vec![1]);
        assert_eq!(batch.prefill[0].id, 3);
        assert_eq!(s.waiting(), 1);
    }

    #[test]
    fn abort_rolls_back_decode_reservation() {
        let mut s = scheduler(1, 2, 1, 64);
        s.submit(request(1, 64, 2)).unwrap();
        s.next_batch().unwrap();
        s.complete_batch(&[]).unwrap();

        let batch = s.next_batch().unwrap().unwrap();
        assert_eq!(batch.decode, vec![1]);
        assert_eq!(s.cache().sequence(1).unwrap().tokens, 65);
        assert_eq!(s.cache().kv().free_blocks(), 0);

        s.abort_batch().unwrap();
        assert_eq!(s.cache().sequence(1).unwrap().tokens, 64);
        assert_eq!(s.cache().kv().free_blocks(), 1);
        assert_eq!(s.next_batch().unwrap().unwrap().decode, vec![1]);
    }

    #[test]
    fn one_in_flight_batch_is_enforced() {
        let mut s = scheduler(1, 2, 1, 64);
        s.submit(request(1, 1, 1)).unwrap();
        s.next_batch().unwrap();
        assert_eq!(s.next_batch(), Err(SchedulerError::BatchInFlight));
    }

    #[test]
    fn invalid_requests_are_rejected_before_queueing() {
        let mut s = scheduler(1, 2, 1, 64);
        assert_eq!(s.submit(request(1, 0, 1)), Err(SubmitError::EmptyPrompt));
        assert_eq!(
            s.submit(request(2, 129, 1)),
            Err(SubmitError::ContextTooLong {
                need: 3,
                capacity: 2,
            })
        );
        s.submit(request(3, 1, 1)).unwrap();
        assert_eq!(s.submit(request(3, 1, 1)), Err(SubmitError::Duplicate(3)));
    }

    #[test]
    fn stopped_sequence_is_released() {
        let mut s = scheduler(1, 2, 1, 64);
        s.submit(request(1, 1, 10)).unwrap();
        s.next_batch().unwrap();
        s.complete_batch(&[]).unwrap();
        s.next_batch().unwrap();

        let done = s.complete_batch(&[1]).unwrap();
        assert_eq!(done[0].reason, FinishReason::Stopped);
        assert_eq!(done[0].generated_tokens, 1);
        assert_eq!(s.active(), 0);
    }

    #[test]
    fn lifetime_reservation_prevents_decode_deadlock() {
        let mut s = scheduler(2, 2, 2, 64);
        s.submit(request(1, 64, 1)).unwrap();
        s.submit(request(2, 64, 1)).unwrap();

        // Одна последовательность коммитит оба блока: prompt и страницу,
        // которая понадобится первому decode-токену.
        let prefill = s.next_batch().unwrap().unwrap();
        assert_eq!(prefill.prefill.len(), 1);
        assert_eq!(s.committed_kv_blocks(), 2);
        assert_eq!(s.waiting(), 1);
        s.complete_batch(&[]).unwrap();

        let decode = s.next_batch().unwrap().unwrap();
        assert_eq!(decode.decode, vec![1]);
        s.complete_batch(&[]).unwrap();
        assert_eq!(s.committed_kv_blocks(), 0);

        // После release второй запрос может быть принят и продолжить работу.
        assert_eq!(s.next_batch().unwrap().unwrap().prefill[0].id, 2);
    }

    #[test]
    fn cancellation_releases_admitted_request() {
        let mut s = scheduler(1, 2, 1, 64);
        s.submit(request(1, 64, 10)).unwrap();
        s.next_batch().unwrap();
        assert_eq!(s.cancel(1), Err(SchedulerError::SequenceInFlight(1)));
        s.complete_batch(&[]).unwrap();

        assert_eq!(s.cancel(1), Ok(true));
        assert_eq!(s.active(), 0);
        assert_eq!(s.cache().states().in_use(), 0);
        assert_eq!(s.cache().kv().free_blocks(), 2);
        assert_eq!(s.cancel(1), Ok(false));
    }

    #[test]
    fn queued_request_can_be_cancelled_without_admission() {
        let mut s = scheduler(1, 2, 1, 64);
        s.submit(request(1, 64, 10)).unwrap();
        assert_eq!(s.cancel(1), Ok(true));
        assert_eq!(s.waiting(), 0);
        assert_eq!(s.active(), 0);
        assert!(s.next_batch().unwrap().is_none());
    }

    fn tokens(len: usize) -> Arc<[u32]> {
        (0..len as u32).collect::<Vec<_>>().into()
    }

    /// Прогнать запрос до конца, возвращая батчи, которые он породил.
    fn drain(s: &mut Scheduler) -> Vec<Batch> {
        let mut batches = Vec::new();
        while let Some(batch) = s.next_batch().unwrap() {
            s.complete_batch(&[]).unwrap();
            batches.push(batch);
        }
        batches
    }

    #[test]
    fn next_turn_continues_from_the_previous_prompt() {
        let mut s = scheduler(2, 100, 2, 512);
        s.enable_prefix_cache(4);
        s.submit_prompt(request(1, 150, 2), tokens(150), 150).unwrap();
        let first = drain(&mut s);
        // Промпт закончился в первом же шаге — там и снимок.
        assert_eq!(first[0].saves.len(), 1);
        assert!(first[0].restores.is_empty());
        assert_eq!(s.cache().prefix_entries(), 1);

        s.submit_prompt(request(2, 200, 1), tokens(200), 200).unwrap();
        let batch = s.next_batch().unwrap().unwrap();
        assert_eq!(
            batch.prefill,
            vec![PrefillChunk {
                id: 2,
                offset: 150,
                tokens: 50,
            }]
        );
        assert_eq!(batch.restores.len(), 1);
        assert_eq!(batch.restores[0].snapshot, first[0].saves[0].snapshot);
        assert!(batch.restores[0].copy_block.is_some());
        s.complete_batch(&[]).unwrap();
        assert_eq!(s.cache().prefix_stats().reused_tokens, 150);
    }

    #[test]
    fn aborted_step_keeps_the_restore_and_frees_the_save() {
        let mut s = scheduler(2, 100, 2, 512);
        s.enable_prefix_cache(1);
        s.submit_prompt(request(1, 100, 1), tokens(100), 100).unwrap();
        drain(&mut s);

        s.submit_prompt(request(2, 130, 1), tokens(130), 130).unwrap();
        let batch = s.next_batch().unwrap().unwrap();
        assert_eq!(batch.restores.len(), 1);
        // Единственный слот держит закреплённая запись: снимать некуда.
        assert!(batch.saves.is_empty());
        s.abort_batch().unwrap();

        let retry = s.next_batch().unwrap().unwrap();
        assert_eq!(retry.restores, batch.restores);
        assert_eq!(retry.prefill, batch.prefill);
    }

    #[test]
    fn cancelling_before_restore_unpins_the_entry() {
        let mut s = scheduler(2, 100, 2, 512);
        s.enable_prefix_cache(1);
        s.submit_prompt(request(1, 100, 1), tokens(100), 100).unwrap();
        drain(&mut s);
        s.submit_prompt(request(2, 130, 1), tokens(130), 130).unwrap();
        s.next_batch().unwrap().unwrap();
        s.abort_batch().unwrap();
        assert!(s.cancel(2).unwrap());
        // Запись снова вытесняема: новый снимок забирает её слот.
        let other: Arc<[u32]> = tokens(64).iter().map(|t| t + 7).collect::<Vec<_>>().into();
        s.submit_prompt(request(3, 64, 1), other, 64).unwrap();
        let batch = s.next_batch().unwrap().unwrap();
        assert_eq!(batch.saves.len(), 1);
    }

    #[test]
    fn prefill_is_cut_at_the_history_boundary() {
        // Промпт 150 токенов, из них последние 6 — промпт генерации. Снимок
        // должен лечь на 144, а следующий ход — продолжить именно оттуда.
        let mut s = scheduler(2, 100, 2, 512);
        s.enable_prefix_cache(2);
        s.submit_prompt(request(1, 150, 1), tokens(150), 144).unwrap();
        let first = s.next_batch().unwrap().unwrap();
        assert_eq!(first.prefill[0].tokens, 144);
        assert_eq!(first.saves.len(), 1);
        s.complete_batch(&[]).unwrap();
        let tail = s.next_batch().unwrap().unwrap();
        assert_eq!(
            tail.prefill,
            vec![PrefillChunk {
                id: 1,
                offset: 144,
                tokens: 6,
            }]
        );
        assert!(tail.saves.is_empty());
        s.complete_batch(&[]).unwrap();
        drain(&mut s);

        // Следующий ход: те же 144 токена истории, дальше другое.
        let mut next: Vec<u32> = (0..144).collect();
        next.extend(1000..1100);
        s.submit_prompt(request(2, 244, 1), next.into(), 238).unwrap();
        let batch = s.next_batch().unwrap().unwrap();
        assert_eq!(batch.prefill[0].offset, 144);
        assert_eq!(batch.restores.len(), 1);
    }
}

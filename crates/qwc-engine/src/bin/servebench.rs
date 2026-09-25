//! Serving throughput under concurrency: N requests, at most C in flight.
//!
//! Every request is submitted at t=0, so the engine runs saturated and the
//! number this reports is steady-state throughput, not a latency figure. TTFT
//! therefore includes queueing for everything past the first C requests, which
//! is what a real backlog looks like.
//!
//! Emits JSON for `bench/servebench.py` to score against the reference engines.

use qwc_core::arch::{KV_ELEMS_PER_TOKEN, NUM_LINEAR_LAYERS, VOCAB_SIZE};
use qwc_cuda::delta_net::{CONV_STATE_ELEMS, DeltaStateMode, STATE_ELEMS, STATE_SCALE_ELEMS};
use qwc_cuda::paged_attention::KvCacheDtype;
use qwc_engine::DecodeLinearMode;
use qwc_engine::{Executor, ExecutorConfig, ModelWeights, PREFILL_CHUNK_SIZE, mtp};
use qwc_model::Checkpoint;
use qwc_runtime::{BatchLayout, CacheManager, Request, Scheduler, SchedulerConfig, SeqId};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

struct Args {
    model: PathBuf,
    requests: usize,
    concurrency: usize,
    prompt_tokens: usize,
    max_new: usize,
    context: usize,
    prefill_chunk: usize,
    memory_limit: usize,
    kv_cache_bytes: usize,
    kv_cache: KvCacheDtype,
    delta_state: DeltaStateMode,
    speculative: usize,
    shortlist: usize,
    /// Сколько последних позиций промпта MTP-голова прогоняет на префилле,
    /// чтобы её KV не был пустым: 0 — ни одной, `usize::MAX` — весь промпт.
    mtp_prime: usize,
    corpus: Option<PathBuf>,
}

/// Активации, скретч проекций, логиты, буферы сэмплинга и CUDA-графы.
/// Ровно та же величина, что закладывает планировщик в `qwc-core`.
const WORKSPACE_RESERVE: usize = 2_000_000_000;

/// Запас поверх свободной памяти карты: округление аллокаций драйвером плюс
/// то, что успевает занять чужой процесс между замером и захватом пула.
const FREE_MARGIN: usize = 1_000_000_000;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse()?;
    qwc_cuda::Device::init(0)?;
    // Фактический лимит может оказаться ниже запрошенного: карта делится с
    // чужими процессами. В артефакт пишется тот, под которым шёл замер.
    let memory_limit = qwc_cuda::set_memory_limit(args.memory_limit)?;
    if memory_limit < args.memory_limit {
        eprintln!(
            "бюджет урезан по свободной памяти: {:.2} ГБ вместо запрошенных {:.2}",
            memory_limit as f64 / 1e9,
            args.memory_limit as f64 / 1e9,
        );
    }
    let checkpoint = Checkpoint::open(&args.model)?;
    let load_started = Instant::now();
    let weights = ModelWeights::load(&checkpoint)?;
    let load_seconds = load_started.elapsed().as_secs_f64();

    let max_blocks = args.context.div_ceil(qwc_cuda::paged_attention::PAGE_SIZE);
    let bytes_per_block = qwc_cuda::paged_attention::PAGE_SIZE
        * KV_ELEMS_PER_TOKEN
        * args.kv_cache.bytes_per_element();

    // `--kv-cache-gb 0` — отдать под KV весь остаток бюджета, как это делает
    // vLLM со своим `gpu_memory_utilization`. Иначе сравнение по общему
    // бюджету врёт: соперник забирает остаток, а мы требуем свои гигабайты
    // сверх весов и падаем там, где на самом деле помещаемся.
    let kv_cache_bytes = if args.kv_cache_bytes == 0 {
        let used = qwc_cuda::memory_usage().used;
        let slots = args.concurrency + 1;
        // Состояние покоящихся слотов — int8 плюс масштаб f32 на строку;
        // сверх слотов лежит одна расквантованная bf16-копия на движок, по
        // которой идёт префилл.
        let state = NUM_LINEAR_LAYERS
            * (slots
                * (STATE_ELEMS
                    + STATE_SCALE_ELEMS * std::mem::size_of::<f32>()
                    + CONV_STATE_ELEMS * std::mem::size_of::<f32>())
                + STATE_ELEMS * std::mem::size_of::<u16>());
        // Остаток бюджета по нашему учёту — верхняя граница, а не правда:
        // драйвер округляет аллокации, а контекст и графы мы не считаем.
        // Поэтому пул режется ещё и по тому, что карта показывает свободным
        // прямо сейчас. Ровно это делает vLLM, профилируя память перед тем,
        // как взять остаток под KV.
        let by_budget = memory_limit
            .saturating_sub(used)
            .saturating_sub(state)
            .saturating_sub(WORKSPACE_RESERVE);
        match qwc_cuda::device_free_bytes() {
            Some(free) => by_budget.min(
                free.saturating_sub(state)
                    .saturating_sub(WORKSPACE_RESERVE)
                    .saturating_sub(FREE_MARGIN),
            ),
            None => by_budget,
        }
    } else {
        args.kv_cache_bytes
    };
    let kv_pool_blocks = kv_cache_bytes / bytes_per_block;
    if kv_pool_blocks < max_blocks {
        return Err(format!(
            "KV pool has {kv_pool_blocks} blocks ({:.2} GB), but --context {} needs at least {max_blocks}; raise --memory-limit-gb or --kv-cache-gb",
            kv_cache_bytes as f64 / 1e9,
            args.context
        )
        .into());
    }
    let mut executor = Executor::new_with_pool_options(
        ExecutorConfig {
            max_batch: args.concurrency,
            max_context: args.context,
        },
        args.kv_cache,
        DecodeLinearMode::Auto,
        args.delta_state,
        Some(kv_pool_blocks),
    )?;
    let cache_gb = executor.cache_bytes() as f64 / 1e9;
    let cache = CacheManager::new(
        args.concurrency,
        executor.kv_pool_blocks(),
        qwc_cuda::paged_attention::PAGE_SIZE,
    );
    let mut scheduler = Scheduler::new(
        SchedulerConfig::new(args.concurrency, args.prefill_chunk),
        cache,
    );

    // Same length for every request, but distinct content: identical prompts
    // would let a prefix-caching engine skip prefill entirely and report a
    // throughput no engine can actually sustain.
    let prompts: HashMap<SeqId, Vec<u32>> = match args.corpus.as_ref() {
        Some(path) => corpus_prompts(path, args.requests, args.prompt_tokens)?,
        None => (1..=args.requests as u32)
            .map(|id| (id, prompt_for(id, args.prompt_tokens)))
            .collect(),
    };

    for id in 1..=args.requests as u32 {
        scheduler.submit(Request {
            id,
            prompt_tokens: args.prompt_tokens,
            max_new_tokens: args.max_new,
        })?;
    }

    // Спекуляция окупается, пока шаг упирается в чтение весов. На широком
    // batch они уже размазаны по строкам, а проверка k+1 строк стоит полного
    // прохода, поэтому черновики берутся только для одиночной decode-строки.
    let mut speculator = match args.speculative {
        0 => None,
        _ => {
            let mut speculator =
                mtp::Speculator::new(&checkpoint, args.context, args.kv_cache)?;
            if args.shortlist > 0 {
                speculator.enable_shortlist(args.shortlist, args.context)?;
            }
            Some(speculator)
        }
    };
    // Для какой последовательности в спекуляторе лежит актуальное скрытое
    // состояние. Пока его нет, шаг идёт обычным путём и заодно его добывает.
    let mut hidden_for: Option<SeqId> = None;
    // Токены, чьи строки лежат в `spec.hidden`, в порядке позиций: после
    // обычного шага — один ожидающий, после проверки — все принятые плюс
    // исправленный. Первый черновой проход пишет их KV настоящими h.
    let mut head_tokens: Vec<u32> = Vec::new();
    let mut prime_ms = 0.0f64;
    let mut speculative_steps = 0usize;
    let mut speculative_tokens = 0usize;
    // Сколько черновиков принято на шаге, по длинам: без этого «acceptance»
    // остаётся средним, а среднее прячет обрыв на конкретной глубине.
    let mut accept_histogram = vec![0usize; qwc_engine::executor::MAX_SPECULATION_ROWS + 1];
    // Обе фазы кончаются копией на хост, поэтому их время меряется часами
    // напрямую — отдельная синхронизация не нужна.
    let mut draft_ms = 0.0f64;
    let mut verify_ms = 0.0f64;

    let mut pending = HashMap::<SeqId, u32>::new();
    let mut produced = HashMap::<SeqId, usize>::new();
    let mut first_token_ms = HashMap::<SeqId, f64>::new();
    let mut last_token_at = HashMap::<SeqId, Instant>::new();
    // Выданные токены по запросу: для TPOT, (последний − первый)/(n − 1).
    // `produced` для этого не годится — он считает входы decode-шага.
    let mut emitted = HashMap::<SeqId, usize>::new();
    // TTFT без очереди: от шага, взявшего запрос в работу, до первого токена.
    // Все запросы подаются в t=0, и при c=1 обычный TTFT — почти одна очередь.
    let mut admitted_at = HashMap::<SeqId, Instant>::new();
    let mut inter_token_ms: Vec<f64> = Vec::with_capacity(args.requests * args.max_new);

    let started = Instant::now();
    let mut completed = 0usize;
    let mut steps = 0usize;
    // One full pass over the weights per step.
    let mut passes = 0usize;
    while completed < args.requests {
        let batch = scheduler
            .next_batch()?
            .ok_or("scheduler made no progress")?;
        // Запрос взят в работу — шаг с первым чанком его промпта, как
        // `scheduled_ts` у vLLM.
        let scheduled_at = Instant::now();
        for chunk in &batch.prefill {
            if chunk.offset == 0 {
                admitted_at.entry(chunk.id).or_insert(scheduled_at);
            }
        }
        let layout = BatchLayout::build(&batch, scheduler.cache())?;

        // Глубина урезается остатком запроса: шаг выдаёт rows токенов разом,
        // и перескочить через max_new значит сделать лишнюю работу и
        // посчитать несуществующий токен.
        let depth = match (speculator.as_ref(), batch.decode.first()) {
            (Some(_), Some(&id)) if batch.prefill.is_empty() && batch.decode.len() == 1 => {
                let remaining = scheduler.remaining_tokens(id).unwrap_or(0);
                args.speculative.min(remaining.saturating_sub(1))
            }
            _ => 0,
        };

        if let Some(spec) = speculator.as_mut()
            && depth > 0
            && hidden_for == Some(batch.decode[0])
        {
            let id = batch.decode[0];
            let token = *pending.get(&id).ok_or("decode token is missing")?;
            let position = layout.position_starts[0] as usize;
            debug_assert_eq!(head_tokens.last(), Some(&token));
            let state_slot = layout.state_slots[0] as usize;
            if spec.bind(id)? {
                // Новая последовательность — шортлисту нужен её промпт.
                spec.set_context(&prompts[&id]);
            }

            let draft_started = Instant::now();
            let first = position + 1 - head_tokens.len();
            let drafts = spec.draft_chain(&weights, &head_tokens, first, depth)?;
            draft_ms += draft_started.elapsed().as_secs_f64() * 1e3;
            // Проверка пишет KV всем строкам разом, поэтому страницы под них
            // нужны заранее. Если пул не дал — черновик просто короче.
            let reserved = scheduler.reserve_extra(id, drafts.len());
            let drafts = &drafts[..reserved];

            // Таблицу берём после резервирования: `layout` построен до него и
            // новых страниц ещё не знает.
            let blocks = scheduler
                .cache()
                .sequence(id)
                .ok_or("sequence is gone")?
                .blocks
                .clone();

            let mut rows_in = Vec::with_capacity(reserved + 1);
            rows_in.push(token);
            rows_in.extend_from_slice(drafts);
            let verify_started = Instant::now();
            let truth =
                executor.verify_speculation(&weights, &rows_in, position, state_slot, &blocks)?;
            verify_ms += verify_started.elapsed().as_secs_f64() * 1e3;

            let mut accepted = 0usize;
            while accepted < drafts.len() && truth[accepted] == drafts[accepted] {
                accepted += 1;
            }
            let rows = accepted + 1;
            executor.commit_speculation(&weights, rows_in.len(), rows, state_slot)?;
            scheduler.release_extra(id, reserved - accepted)?;
            // Строки 0..rows проверки несут настоящие h для принятых позиций
            // и для исправленной: следующий черновик перепишет ими KV головы.
            executor.copy_prefill_hidden_rows(0, rows, spec.hidden_mut())?;
            head_tokens.clear();
            head_tokens.extend_from_slice(&truth[..rows]);

            let now = Instant::now();
            first_token_ms
                .entry(id)
                .or_insert_with(|| (now - started).as_secs_f64() * 1e3);
            if let Some(previous) = last_token_at.insert(id, now) {
                inter_token_ms.push((now - previous).as_secs_f64() * 1e3 / rows as f64);
            }
            *emitted.entry(id).or_insert(0) += rows;
            for token in &truth[..rows] {
                spec.observe(*token);
            }
            pending.insert(id, truth[accepted]);
            *produced.entry(id).or_insert(0) += rows;
            speculative_steps += 1;
            speculative_tokens += rows;
            accept_histogram[accepted] += 1;

            passes += 1;
            // Сбрасываем скрытое состояние только если закончился именно этот
            // запрос: `completed` накопительный и после первого же финиша
            // выключал бы спекуляцию на каждом втором шаге.
            let finished = scheduler.complete_batch_multi(&[], &[(id, rows)])?;
            completed += finished.len();
            steps += 1;
            if !finished.is_empty() {
                hidden_for = None;
            }
            continue;
        }

        let mut input = Vec::with_capacity(layout.num_tokens());
        for &id in &batch.decode {
            input.push(*pending.get(&id).ok_or("decode token is missing")?);
        }
        for chunk in &batch.prefill {
            input.extend_from_slice(&prompts[&chunk.id][chunk.offset..chunk.offset + chunk.tokens]);
        }
        for &id in &batch.decode {
            *produced.entry(id).or_insert(0) += 1;
        }

        let sampled = executor.execute_layout(&weights, &layout, &input)?;
        // Голова заполняет свой KV по промпту: без этого на промпте у неё
        // нули, и acceptance проседает с 0.91 до 0.85. Обслуживает она одну
        // последовательность, поэтому только в шаге с единственным чанком.
        if let Some(spec) = speculator.as_mut()
            && args.mtp_prime > 0
            && let [chunk] = batch.prefill.as_slice()
        {
            let prompt = &prompts[&chunk.id];
            let row = layout
                .seq_ids
                .iter()
                .position(|&id| id == chunk.id)
                .ok_or("prefill chunk is missing from the layout")?;
            let row_begin = layout.token_offsets[row] as usize;
            let window_start = prompt.len().saturating_sub(args.mtp_prime);
            let begin = chunk.offset.max(window_start);
            let end = chunk.offset + chunk.tokens;
            if begin < end {
                // Пара строки t — (h_t, токен t+1). Для последней позиции
                // промпта токен t+1 — тот, что шаг только что выдал.
                let mut next = Vec::with_capacity(end - begin);
                for t in begin..end {
                    next.push(match prompt.get(t + 1) {
                        Some(&token) => token,
                        None => sampled
                            .iter()
                            .find(|(id, _)| *id == chunk.id)
                            .map(|&(_, token)| token)
                            .ok_or("prefill produced no token")?,
                    });
                }
                if spec.bind(chunk.id)? {
                    spec.set_context(prompt);
                }
                let prime_started = Instant::now();
                spec.prime(&weights, &executor, row_begin + begin - chunk.offset, &next, begin + 1)?;
                prime_ms += prime_started.elapsed().as_secs_f64() * 1e3;
            }
        }
        for (id, token) in sampled {
            pending.insert(id, token);
            // Сэмпл есть у каждой строки шага, но токен запрос получает только
            // от decode и от последнего чанка промпта. Сэмпл недопрефилленного
            // чанка — не токен: иначе TTFT длинного промпта — время первого
            // чанка, а шаги остальных чанков попадают в ITL и TPOT.
            let partial = batch
                .prefill
                .iter()
                .any(|chunk| chunk.id == id && chunk.offset + chunk.tokens < prompts[&id].len());
            if partial {
                continue;
            }
            let now = Instant::now();
            first_token_ms
                .entry(id)
                .or_insert_with(|| (now - started).as_secs_f64() * 1e3);
            if let Some(previous) = last_token_at.insert(id, now) {
                inter_token_ms.push((now - previous).as_secs_f64() * 1e3);
            }
            *emitted.entry(id).or_insert(0) += 1;
        }
        // Скрытое состояние для следующего чернового шага берётся отсюда:
        // спекуляции нужен вход головы, а он есть только после прохода.
        hidden_for = None;
        if let Some(spec) = speculator.as_mut()
            && batch.prefill.is_empty()
            && batch.decode.len() == 1
        {
            executor.copy_decode_hidden_row(0, spec.hidden_mut())?;
            hidden_for = Some(batch.decode[0]);
            head_tokens.clear();
            head_tokens.push(*pending.get(&batch.decode[0]).ok_or("decode token is missing")?);
        }

        // A step is one forward pass now, mixed or not.
        passes += 1;
        completed += scheduler.complete_batch(&[])?.len();
        steps += 1;
    }
    let elapsed = started.elapsed().as_secs_f64();

    let output_tokens: usize = produced.values().sum();
    let mut ttft: Vec<f64> = first_token_ms.values().copied().collect();
    // TPOT — среднее время на токен после первого, по запросу; перцентили —
    // по запросам. ITL — по промежуткам между выдачами, при спекуляции
    // промежуток делится на число выданных за шаг токенов.
    let mut tpot: Vec<f64> = emitted
        .iter()
        .filter(|&(_, &count)| count > 1)
        .filter_map(|(id, &count)| {
            let last = (*last_token_at.get(id)? - started).as_secs_f64() * 1e3;
            Some((last - first_token_ms.get(id)?) / (count - 1) as f64)
        })
        .collect();
    let mut ttft_service: Vec<f64> = first_token_ms
        .iter()
        .filter_map(|(id, &first)| {
            Some(first - (*admitted_at.get(id)? - started).as_secs_f64() * 1e3)
        })
        .collect();
    println!(
        concat!(
            "{{\"engine\":\"qwc\",\"version\":\"{}\",\"requests\":{},\"concurrency\":{},",
            "\"prompt_tokens\":{},\"max_new_tokens\":{},\"context\":{},\"kv_cache\":\"{}\",",
            "\"load_seconds\":{:.3},\"cache_gb\":{:.2},\"kv_pool_blocks\":{},\"engine_steps\":{},\"weight_passes\":{},",
            "\"speculative\":{},\"speculative_steps\":{},\"speculative_tokens\":{},",
            "\"accept_histogram\":{:?},",
            "\"draft_ms_total\":{:.1},\"verify_ms_total\":{:.1},\"prime_ms_total\":{:.1},",
            "\"mtp_prime\":\"{}\",\"resident_weights_gb\":{:.2},",
            "\"prompts\":\"{}\",\"memory_limit_gb\":{:.2},\"shortlist\":{},",
            "\"prefill_chunk\":{},\"delta_state\":\"{}\",\"cuda_graphs\":true,",
            "\"wall_seconds\":{:.4},\"output_tokens\":{},",
            "\"output_tokens_per_second\":{:.2},\"requests_per_second\":{:.3},",
            "\"ttft_ms_p50\":{:.2},\"ttft_ms_p95\":{:.2},",
            "\"itl_ms_p50\":{:.3},\"itl_ms_p95\":{:.3},",
            "\"tpot_ms_p50\":{:.3},\"tpot_ms_p95\":{:.3},",
            "\"ttft_service_ms_p50\":{:.2},\"ttft_service_ms_p95\":{:.2}}}"
        ),
        env!("CARGO_PKG_VERSION"),
        args.requests,
        args.concurrency,
        args.prompt_tokens,
        args.max_new,
        args.context,
        args.kv_cache.as_str(),
        load_seconds,
        cache_gb,
        executor.kv_pool_blocks(),
        steps,
        passes,
        args.speculative,
        speculative_steps,
        speculative_tokens,
        accept_histogram,
        draft_ms,
        verify_ms,
        prime_ms,
        match args.mtp_prime {
            0 => "off".to_string(),
            usize::MAX => "all".to_string(),
            window => window.to_string(),
        },
        weights.stats().resident_bytes() as f64 / 1e9,
        args.corpus
            .as_deref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "synthetic".to_string()),
        memory_limit as f64 / 1e9,
        args.shortlist,
        args.prefill_chunk,
        args.delta_state.as_str(),
        elapsed,
        output_tokens,
        output_tokens as f64 / elapsed,
        args.requests as f64 / elapsed,
        percentile(&mut ttft, 0.50),
        percentile(&mut ttft, 0.95),
        percentile(&mut inter_token_ms, 0.50),
        percentile(&mut inter_token_ms, 0.95),
        percentile(&mut tpot, 0.50),
        percentile(&mut tpot, 0.95),
        percentile(&mut ttft_service, 0.50),
        percentile(&mut ttft_service, 0.95),
    );
    Ok(())
}

/// Промпты из корпуса: те же строки, что читает `bench/servebench.py` для
/// vLLM, поэтому оба движка видят один и тот же текст. Парсер намеренно
/// примитивный — формат пишет `bench/make_corpus.py`, одна запись в строке и
/// `prompt_token_ids` массивом целых.
fn corpus_prompts(
    path: &std::path::Path,
    requests: usize,
    prompt_tokens: usize,
) -> Result<HashMap<SeqId, Vec<u32>>, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(path)?;
    let mut prompts = HashMap::new();
    for (index, line) in text.lines().filter(|l| !l.trim().is_empty()).enumerate() {
        if index == requests {
            break;
        }
        let key = "\"prompt_token_ids\"";
        let start = line
            .find(key)
            .and_then(|at| line[at..].find('[').map(|b| at + b + 1))
            .ok_or_else(|| {
                format!(
                    "{}: строка {} без prompt_token_ids",
                    path.display(),
                    index + 1
                )
            })?;
        let end = line[start..].find(']').ok_or_else(|| {
            format!(
                "{}: строка {} с незакрытым массивом",
                path.display(),
                index + 1
            )
        })? + start;
        let ids: Vec<u32> = line[start..end]
            .split(',')
            .map(|token| token.trim().parse::<u32>())
            .collect::<Result<_, _>>()?;
        if ids.len() < prompt_tokens {
            return Err(format!(
                "{}: промпт {} даёт {} токенов, а нужно {prompt_tokens}",
                path.display(),
                index + 1,
                ids.len()
            )
            .into());
        }
        prompts.insert(index as u32 + 1, ids[..prompt_tokens].to_vec());
    }
    if prompts.len() < requests {
        return Err(format!(
            "{}: {} промптов, а запрошено {requests}",
            path.display(),
            prompts.len()
        )
        .into());
    }
    Ok(prompts)
}

/// Distinct token ids per request; mirrors `prompt_ids` in bench/servebench.py
/// so every engine sees byte-identical prompts.
fn prompt_for(id: u32, length: usize) -> Vec<u32> {
    (0..length)
        .map(|index| 1000 + ((id.wrapping_mul(7919) + index as u32) % 20000))
        .collect()
}

fn percentile(values: &mut [f64], fraction: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(f64::total_cmp);
    let index = ((values.len() - 1) as f64 * fraction).round() as usize;
    values[index]
}

fn parse() -> Result<Args, Box<dyn std::error::Error>> {
    let mut model = home().join("models/Qwen3.8-27B-QUASAR-NVFP4");
    let mut requests = 200usize;
    let mut concurrency = 32usize;
    let mut prompt_tokens = 256usize;
    let mut max_new = 128usize;
    let mut context = 2048usize;
    // Бюджет токенов на шаг; потолок — ёмкость арены префилла.
    let mut prefill_chunk = PREFILL_CHUNK_SIZE;
    let mut speculative = 0usize;
    let mut memory_limit = 28_000_000_000usize;
    // Physical pages are shared. Five GB is enough for four full 32K
    // sequences or many short requests without reserving 32K for each slot.
    let mut kv_cache_bytes = 5_000_000_000usize;
    let mut kv_cache = KvCacheDtype::Fp8;
    let mut corpus: Option<PathBuf> = None;
    // 0 — полная проекция в словарь; иначе столько первых строк словаря
    // держатся в шортлисте всегда, сверх токенов контекста.
    let mut shortlist = 0usize;
    // Как у `serve`: стенд должен мерить тот же прогрев, что работает в
    // сервере, иначе на длинных промптах он показывает не ту цену TTFT.
    let mut mtp_prime = 2048usize;
    let mut delta_state = DeltaStateMode::Wy;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model = PathBuf::from(args.next().ok_or("--model")?),
            "--corpus" => corpus = Some(PathBuf::from(args.next().ok_or("--corpus")?)),
            "--shortlist" => {
                shortlist = args.next().ok_or("--shortlist")?.parse()?;
                if shortlist > VOCAB_SIZE {
                    return Err("--shortlist не может быть больше словаря".into());
                }
            }
            "--mtp-prime" => {
                mtp_prime = match args.next().ok_or("--mtp-prime needs all, off or N")?.as_str() {
                    "all" => usize::MAX,
                    "off" => 0,
                    window => window.parse()?,
                }
            }
            "--requests" => requests = args.next().ok_or("--requests")?.parse()?,
            "--concurrency" => concurrency = args.next().ok_or("--concurrency")?.parse()?,
            "--prompt-tokens" => prompt_tokens = args.next().ok_or("--prompt-tokens")?.parse()?,
            "--max-new" => max_new = args.next().ok_or("--max-new")?.parse()?,
            "--context" => context = args.next().ok_or("--context")?.parse()?,
            "--prefill-chunk" => prefill_chunk = args.next().ok_or("--prefill-chunk")?.parse()?,
            "--delta-state" => {
                delta_state = match args
                    .next()
                    .ok_or("--delta-state needs bf16, fp32 or wy")?
                    .as_str()
                {
                    "bf16" => DeltaStateMode::Bf16,
                    "fp32" => DeltaStateMode::Fp32,
                    "wy" => DeltaStateMode::Wy,
                    other => {
                        return Err(format!("неизвестное значение --delta-state: {other}").into());
                    }
                };
            }
            "--kv-cache" => {
                kv_cache = match args.next().ok_or("--kv-cache needs fp8 or bf16")?.as_str() {
                    "fp8" => KvCacheDtype::Fp8,
                    "bf16" => KvCacheDtype::Bf16,
                    other => return Err(format!("unsupported --kv-cache value: {other}").into()),
                };
            }
            "--memory-limit-gb" => {
                let gb: f64 = args.next().ok_or("--memory-limit-gb")?.parse()?;
                memory_limit = (gb * 1e9) as usize;
            }
            "--speculative" => {
                speculative = args.next().ok_or("--speculative")?.parse()?;
                if speculative > 7 {
                    return Err("--speculative поддерживает глубину 0..7".into());
                }
            }
            "--kv-cache-gb" => {
                let gb: f64 = args.next().ok_or("--kv-cache-gb")?.parse()?;
                // 0 — отдать под KV весь остаток `--memory-limit-gb`.
                if !gb.is_finite() || gb < 0.0 {
                    return Err("--kv-cache-gb must not be negative".into());
                }
                kv_cache_bytes = (gb * 1e9) as usize;
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }
    if concurrency == 0 || concurrency > qwc_engine::executor::MAX_BATCH {
        return Err(format!(
            "--concurrency must be 1..={}",
            qwc_engine::executor::MAX_BATCH
        )
        .into());
    }
    if requests < concurrency {
        return Err("--requests must be at least --concurrency".into());
    }
    if !(1..=PREFILL_CHUNK_SIZE).contains(&prefill_chunk) {
        return Err(format!("--prefill-chunk must be 1..={PREFILL_CHUNK_SIZE}").into());
    }
    if prompt_tokens == 0 || prompt_tokens + max_new > context {
        return Err("prompt + generation must fit in --context".into());
    }
    if prompt_tokens >= VOCAB_SIZE {
        return Err("prompt too long".into());
    }
    Ok(Args {
        model,
        requests,
        concurrency,
        prompt_tokens,
        max_new,
        context,
        prefill_chunk,
        memory_limit,
        kv_cache_bytes,
        speculative,
        shortlist,
        mtp_prime,
        kv_cache,
        delta_state,
        corpus,
    })
}

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/root".into()))
}

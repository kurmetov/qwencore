//! Пофазная разбивка шага движка на событиях CUDA.
//!
//! Отвечает на один вопрос: куда уходит время шага. Трафик весов и состояния
//! объясняет только часть, остальное до сих пор приписывалось узким местам по
//! roofline-арифметике, а не по замеру. Здесь каждая фаза — интервал между
//! двумя событиями на потоке, и сумма фаз равна шагу целиком.
//!
//! Под таймлайном шаг чистого decode идёт мимо CUDA graph: внутри захвата
//! события времени не дают. Поэтому сначала меряется шаг с графом, потом тот
//! же шаг с разметкой — разница и есть цена запусков кернелов.
//!
//! ```
//! cargo run --release -p qwc-engine --bin stepprofile -- \
//!     --model PATH --concurrency 64 --context 2048 --kv-cache fp8
//! ```

use qwc_core::arch::VOCAB_SIZE;
use qwc_cuda::delta_net::{self, DeltaStateMode};
use qwc_cuda::paged_attention::{KvCacheDtype, PAGE_SIZE};
use qwc_engine::executor::PREFILL_CHUNK_SIZE;
use qwc_engine::weights::{DecodeLinearMode, Mixer};
use qwc_engine::{Executor, ExecutorConfig, ModelWeights};
use qwc_model::Checkpoint;
use qwc_runtime::BatchLayout;
use std::path::PathBuf;
use std::time::Instant;

struct Args {
    model: PathBuf,
    concurrency: usize,
    prompt_tokens: usize,
    prefill_tokens: usize,
    prefill_start: usize,
    steps: usize,
    warmup: usize,
    context: usize,
    memory_limit: usize,
    kv_cache: KvCacheDtype,
    delta_state: DeltaStateMode,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse()?;
    qwc_cuda::Device::init(0)?;
    qwc_cuda::set_memory_limit(args.memory_limit)?;
    let checkpoint = Checkpoint::open(&args.model)?;
    let load_started = Instant::now();
    let weights = ModelWeights::load(&checkpoint)?;
    let load_seconds = load_started.elapsed().as_secs_f64();

    let mut executor = Executor::new_with_options(
        ExecutorConfig {
            max_batch: args.concurrency,
            max_context: args.context,
        },
        args.kv_cache,
        DecodeLinearMode::Auto,
        args.delta_state,
    )?;
    let cache_gb = executor.cache_bytes() as f64 / 1e9;
    let usage = qwc_cuda::memory_usage();
    eprintln!(
        "загружено за {load_seconds:.1} с, кэш и состояние {cache_gb:.2} ГБ, \
         занято {:.2} ГБ из лимита {:.2} ГБ",
        usage.used as f64 / 1e9,
        usage.limit as f64 / 1e9,
    );

    // Каждый слот получает свой префикс: и KV-кэш, и рекуррентное состояние
    // должны быть заполнены, иначе шаг меряется на пустой карте.
    let prompt: Vec<u32> = (0..args.prompt_tokens)
        .map(|index| 1000 + (index as u32 % 20000))
        .collect();
    for slot in 0..args.concurrency {
        executor.prefill_sequence(&weights, &prompt, 0, slot)?;
    }

    let mixed = args.prefill_tokens > 0;
    let decode_rows = if mixed {
        args.concurrency - 1
    } else {
        args.concurrency
    };
    let mut position = args.prompt_tokens;

    // Шаг с графом: эталон, с которым сверяется сумма фаз.
    let mut graph_ms = Vec::with_capacity(args.steps);
    for step in 0..args.warmup + args.steps {
        let elapsed = run_step(&mut executor, &weights, &args, decode_rows, position, mixed)?;
        position += 1;
        if step >= args.warmup {
            graph_ms.push(elapsed);
        }
    }

    // Тот же шаг с разметкой.
    executor.profile_enable();
    let mut timeline_ms = Vec::with_capacity(args.steps);
    let mut totals: Vec<(&'static str, f64)> = Vec::new();
    let mut marks = 0usize;
    let mut span_sum = 0.0f64;
    for step in 0..args.warmup + args.steps {
        let elapsed = run_step(&mut executor, &weights, &args, decode_rows, position, mixed)?;
        position += 1;
        let phases = executor.profile_totals()?;
        let (span, step_marks) = executor.profile_span()?;
        if step < args.warmup {
            continue;
        }
        timeline_ms.push(elapsed);
        span_sum += span as f64;
        marks = step_marks;
        for (label, ms) in phases {
            match totals.iter_mut().find(|(name, _)| *name == label) {
                Some((_, sum)) => *sum += ms as f64,
                None => totals.push((label, ms as f64)),
            }
        }
    }
    executor.profile_disable();

    let steps = args.steps as f64;
    let graph = median(&mut graph_ms);
    let with_timeline = median(&mut timeline_ms);
    let span = span_sum / steps;
    let phase_sum: f64 = totals.iter().map(|(_, ms)| ms).sum::<f64>() / steps;

    let traffic = Traffic::measure(&weights, decode_rows, args.prefill_tokens);
    totals.sort_by(|a, b| b.1.total_cmp(&a.1));

    println!();
    println!(
        "шаг {} на concurrency {} (decode-строк {decode_rows}{}), контекст {}, KV {}, DeltaNet {}",
        if mixed { "смешанный" } else { "чистый decode" },
        args.concurrency,
        if mixed {
            format!(", prefill-токенов {}", args.prefill_tokens)
        } else {
            String::new()
        },
        position,
        args.kv_cache.as_str(),
        args.delta_state.as_str(),
    );
    println!(
        "{:<24} {:>9} {:>7} {:>10} {:>10}",
        "фаза", "мс", "доля", "трафик ГБ", "ГБ/с"
    );
    println!("{}", "-".repeat(64));
    for (label, ms) in &totals {
        let ms = ms / steps;
        let share = 100.0 * ms / phase_sum;
        match traffic.bytes(label) {
            Some(bytes) => println!(
                "{label:<24} {ms:>9.3} {share:>6.1}% {:>10.2} {:>10.0}",
                bytes as f64 / 1e9,
                bytes as f64 / (ms * 1e-3) / 1e9,
            ),
            None => println!("{label:<24} {ms:>9.3} {share:>6.1}% {:>10} {:>10}", "", ""),
        }
    }
    println!("{}", "-".repeat(64));
    println!(
        "{:<24} {:>9.3} {:>6.1}% {:>10.2} {:>10.0}",
        "сумма фаз",
        phase_sum,
        100.0,
        traffic.total as f64 / 1e9,
        traffic.total as f64 / (phase_sum * 1e-3) / 1e9,
    );
    println!();
    println!("шаг с графом (стенные часы):     {graph:>8.3} мс");
    println!("шаг с разметкой (стенные часы):  {with_timeline:>8.3} мс");
    println!("от первой метки до последней:    {span:>8.3} мс");
    println!(
        "накладные запусков (разметка − граф): {:>+7.3} мс при {marks} метках",
        with_timeline - graph
    );
    println!(
        "объяснено трафиком при 1.5 ТБ/с: {:.3} мс из {:.3} мс шага",
        traffic.total as f64 / 1.5e12 * 1e3,
        graph,
    );
    Ok(())
}

/// Один шаг: чистый decode идёт коротким путём, смешанный — через раскладку.
fn run_step(
    executor: &mut Executor,
    weights: &ModelWeights,
    args: &Args,
    decode_rows: usize,
    position: usize,
    mixed: bool,
) -> Result<f64, Box<dyn std::error::Error>> {
    let tokens: Vec<u32> = (0..decode_rows).map(|row| 1000 + row as u32).collect();
    let started = Instant::now();
    if mixed {
        let layout = mixed_layout(args, decode_rows, position)?;
        let mut input = tokens.clone();
        input.extend((0..args.prefill_tokens).map(|index| 1000 + index as u32 % 20000));
        executor.execute_layout(weights, &layout, &input)?;
    } else {
        let positions = vec![position as u32; decode_rows];
        executor.decode(weights, &tokens, &positions)?;
    }
    Ok(started.elapsed().as_secs_f64() * 1e3)
}

/// Раскладка шага: `decode_rows` однотокеновых строк плюс один prefill-чанк в
/// последнем слоте. Это форма, которую планировщик выдаёт под нагрузкой, пока
/// в очереди остаются непрогретые запросы.
fn mixed_layout(
    args: &Args,
    decode_rows: usize,
    position: usize,
) -> Result<BatchLayout, Box<dyn std::error::Error>> {
    let max_blocks = args.context.div_ceil(PAGE_SIZE);
    let mut layout = BatchLayout {
        num_decode: decode_rows as u32,
        seq_ids: Vec::new(),
        state_slots: Vec::new(),
        position_starts: Vec::new(),
        context_lens: Vec::new(),
        token_offsets: vec![0],
        block_table_offsets: vec![0],
        block_ids: Vec::new(),
    };
    let mut push = |slot: usize, start: usize, tokens: usize| {
        layout.seq_ids.push(slot as u32 + 1);
        layout.state_slots.push(slot as u32);
        layout.position_starts.push(start as u32);
        layout.context_lens.push((start + tokens) as u32);
        let offset = *layout.token_offsets.last().expect("непустой CSR");
        layout.token_offsets.push(offset + tokens as u32);
        let blocks = (start + tokens).div_ceil(PAGE_SIZE);
        layout
            .block_ids
            .extend((0..blocks).map(|block| (slot * max_blocks + block) as u32));
        let bound = *layout.block_table_offsets.last().expect("непустой CSR");
        layout
            .block_table_offsets
            .push(bound + blocks as u32);
    };
    for row in 0..decode_rows {
        push(row, position, 1);
    }
    push(decode_rows, args.prefill_start, args.prefill_tokens);
    Ok(layout)
}

/// Трафик памяти фазы: у GEMM это веса, у скана — состояние DeltaNet.
/// Делить замер на трафик имеет смысл только там, где фаза упирается в память.
struct Traffic {
    phases: Vec<(&'static str, u64)>,
    total: u64,
}

impl Traffic {
    fn measure(weights: &ModelWeights, decode_rows: usize, prefill_tokens: usize) -> Self {
        let mut la_in = 0u64;
        let mut la_out = 0u64;
        let mut attn_qkv = 0u64;
        let mut attn_out = 0u64;
        let mut mlp_gate_up = 0u64;
        let mut mlp_down = 0u64;
        let mut linear_layers = 0usize;
        for layer in &weights.layers {
            match &layer.mixer {
                Mixer::Linear(mixer) => {
                    la_in += mixer.in_proj.resident_bytes() as u64;
                    la_out += mixer.out.resident_bytes() as u64;
                    linear_layers += 1;
                }
                Mixer::Full(mixer) => {
                    attn_qkv += (mixer.q.resident_bytes()
                        + mixer.k.resident_bytes()
                        + mixer.v.resident_bytes()) as u64;
                    attn_out += mixer.o.resident_bytes() as u64;
                }
            }
            mlp_gate_up +=
                (layer.mlp.gate.resident_bytes() + layer.mlp.up.resident_bytes()) as u64;
            mlp_down += layer.mlp.down.resident_bytes() as u64;
        }
        // Скан рекуррентного состояния: полное чтение и полная перезапись на
        // каждый слой. Для prefill-сегмента состояние проходится один раз на
        // чанк, а не на токен.
        let scan = delta_net::traffic_bytes(decode_rows) * linear_layers as u64;
        let scan_prefill = if prefill_tokens > 0 {
            delta_net::traffic_bytes(1) * linear_layers as u64
        } else {
            0
        };
        let lm_head = weights.lm_head.resident_bytes() as u64;

        let phases = vec![
            ("gemm.la_in", la_in),
            ("gemm.la_out", la_out),
            ("gemm.attn_qkv", attn_qkv),
            ("gemm.attn_out", attn_out),
            ("gemm.mlp_gate_up", mlp_gate_up),
            ("gemm.mlp_down", mlp_down),
            ("lm_head", lm_head),
            ("delta.scan", scan),
            ("delta.scan_prefill", scan_prefill),
        ];
        let total = phases.iter().map(|(_, bytes)| bytes).sum();
        Self { phases, total }
    }

    fn bytes(&self, label: &str) -> Option<u64> {
        self.phases
            .iter()
            .find(|(name, bytes)| *name == label && *bytes > 0)
            .map(|(_, bytes)| *bytes)
    }
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn parse() -> Result<Args, Box<dyn std::error::Error>> {
    let mut model = home().join("models/Qwen3.8-27B-QUASAR-NVFP4");
    let mut concurrency = 64usize;
    let mut prompt_tokens = 256usize;
    let mut prefill_tokens = 0usize;
    // Чанк не в нуле, а на своей позиции в промпте: внимание на префилле
    // читает весь контекст слева, и в нуле его цена не видна.
    let mut prefill_start = 0usize;
    let mut steps = 21usize;
    let mut warmup = 3usize;
    let mut context = 2048usize;
    let mut memory_limit = 30_000_000_000usize;
    let mut kv_cache = KvCacheDtype::Fp8;
    let mut delta_state = DeltaStateMode::Wy;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model = PathBuf::from(args.next().ok_or("--model")?),
            "--concurrency" => concurrency = args.next().ok_or("--concurrency")?.parse()?,
            "--prompt-tokens" => prompt_tokens = args.next().ok_or("--prompt-tokens")?.parse()?,
            "--prefill-tokens" => prefill_tokens = args.next().ok_or("--prefill-tokens")?.parse()?,
            "--prefill-start" => prefill_start = args.next().ok_or("--prefill-start")?.parse()?,
            "--steps" => steps = args.next().ok_or("--steps")?.parse()?,
            "--warmup" => warmup = args.next().ok_or("--warmup")?.parse()?,
            "--context" => context = args.next().ok_or("--context")?.parse()?,
            "--kv-cache" => {
                kv_cache = match args.next().ok_or("--kv-cache needs fp8 or bf16")?.as_str() {
                    "fp8" => KvCacheDtype::Fp8,
                    "bf16" => KvCacheDtype::Bf16,
                    other => return Err(format!("неизвестное значение --kv-cache: {other}").into()),
                };
            }
            "--delta-state" => {
                delta_state = match args.next().ok_or("--delta-state needs bf16, fp32 or wy")?.as_str() {
                    "bf16" => DeltaStateMode::Bf16,
                    "fp32" => DeltaStateMode::Fp32,
                    "wy" => DeltaStateMode::Wy,
                    other => return Err(format!("неизвестное значение --delta-state: {other}").into()),
                };
            }
            "--memory-limit-gb" => {
                let gb: f64 = args.next().ok_or("--memory-limit-gb")?.parse()?;
                memory_limit = (gb * 1e9) as usize;
            }
            other => return Err(format!("неизвестный аргумент: {other}").into()),
        }
    }
    if concurrency == 0 || concurrency > qwc_engine::executor::MAX_BATCH {
        return Err(format!(
            "--concurrency должен быть 1..={}",
            qwc_engine::executor::MAX_BATCH
        )
        .into());
    }
    if steps == 0 {
        return Err("--steps должен быть положительным".into());
    }
    if prompt_tokens == 0 || prompt_tokens >= VOCAB_SIZE {
        return Err("--prompt-tokens вне диапазона".into());
    }
    if prefill_tokens > 0 {
        if concurrency < 2 {
            return Err("смешанный шаг требует --concurrency не меньше 2".into());
        }
        let live = concurrency - 1 + prefill_tokens;
        if live > PREFILL_CHUNK_SIZE {
            return Err(format!("шаг несёт {live} токенов, арена держит {PREFILL_CHUNK_SIZE}").into());
        }
        if prefill_start + prefill_tokens > context {
            return Err("--prefill-tokens не помещается в контекст".into());
        }
    }
    if prompt_tokens + warmup + steps > context {
        return Err("prompt плюс шаги не помещаются в --context".into());
    }
    Ok(Args {
        model,
        concurrency,
        prompt_tokens,
        prefill_tokens,
        prefill_start,
        steps,
        warmup,
        context,
        memory_limit,
        kv_cache,
        delta_state,
    })
}

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/root".into()))
}

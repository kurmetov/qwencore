//! Подъём весов модели в VRAM: время, реальная VRAM и потолок host-RAM.
//! `cargo run --release -p qwc-engine --bin load -- [каталог] [--vram-gb N]`

use qwc_core::arch::{HIDDEN_SIZE, NUM_FULL_LAYERS, NUM_LAYERS, NUM_LINEAR_LAYERS, VOCAB_SIZE};
use qwc_cuda::{DeviceBuffer, Stream, bf16};
use qwc_engine::ModelWeights;
use qwc_engine::weights::Mixer;
use qwc_model::Checkpoint;
use std::path::PathBuf;
use std::time::Instant;

/// Потолок по умолчанию: веса плюс запас под кэш, но заметно ниже 32 GB —
/// чтобы упор в лимит был виден в тесте, а не только на пустой карте.
const DEFAULT_VRAM_GB: f64 = 24.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut dir: Option<PathBuf> = None;
    let mut vram_gb = DEFAULT_VRAM_GB;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--vram-gb" => {
                vram_gb = args
                    .next()
                    .ok_or("--vram-gb требует значение")?
                    .parse::<f64>()?;
            }
            other => dir = Some(PathBuf::from(other)),
        }
    }
    let dir = dir.unwrap_or_else(|| home().join("models/Qwen3.8-27B-QUASAR-NVFP4"));

    let device = qwc_cuda::Device::init(0)?;
    let (free_before, total) = qwc_cuda::Device::mem_info()?;
    let limit = (vram_gb * 1e9) as usize;
    qwc_cuda::set_memory_limit(limit)?;

    println!("чекпоинт: {}", dir.display());
    println!(
        "GPU: {} SM, VRAM свободно {:.2} / {:.2} GB, потолок процесса {:.2} GB\n",
        device.sm_count,
        free_before as f64 / 1e9,
        total as f64 / 1e9,
        limit as f64 / 1e9
    );

    let checkpoint = Checkpoint::open(&dir)?;
    let (problems, checkpoint_stats) = checkpoint.validate();
    if !problems.is_empty() {
        for problem in problems.iter().take(10) {
            println!("  {problem}");
        }
        return Err(format!("чекпоинт разошёлся с архитектурой: {}", problems.len()).into());
    }

    let started = Instant::now();
    let weights = ModelWeights::load_with_progress(&checkpoint, |index, stats| {
        if index % 8 == 7 || index + 1 == NUM_LAYERS {
            println!(
                "  слой {:>2}/{NUM_LAYERS}  {:>6.2} GB  {:>6.1} s",
                index + 1,
                stats.resident_bytes() as f64 / 1e9,
                started.elapsed().as_secs_f64()
            );
        }
    })?;
    let elapsed = started.elapsed().as_secs_f64();

    let stats = weights.stats();
    let usage = qwc_cuda::memory_usage();
    let (free_after, _) = qwc_cuda::Device::mem_info()?;
    let expected = checkpoint_stats.packed_bytes + checkpoint_stats.scale_bytes;

    println!("\nЗагружено");
    println!(
        "  слоёв                 {} ({} linear + {} full)",
        stats.layers, NUM_LINEAR_LAYERS, NUM_FULL_LAYERS
    );
    println!("  квантованных проекций {}", stats.projections);
    println!(
        "  веса NVFP4 + шкалы    {:>7.2} GB",
        stats.quantized_bytes as f64 / 1e9
    );
    println!(
        "                        {:>7.2} GB в чекпоинте (разница {:+.0} MB — padding scale-layout)",
        expected as f64 / 1e9,
        (stats.quantized_bytes as f64 - expected as f64) / 1e6
    );
    println!(
        "  нормы, conv1d, гейты  {:>7.2} MB",
        stats.plain_bytes as f64 / 1e6
    );
    println!(
        "  embed + lm_head в fp8 {:>7.2} GB  (в чекпоинте bf16 {:.2} GB)",
        stats.vocab_bytes as f64 / 1e9,
        (checkpoint_stats.embed_bytes + checkpoint_stats.lm_head_bytes) as f64 / 1e9
    );
    println!(
        "  всего в VRAM          {:>7.2} GB",
        stats.resident_bytes() as f64 / 1e9
    );

    println!("\nПамять");
    println!(
        "  учтено потолком       {:>7.2} GB из {:.2} GB",
        usage.used as f64 / 1e9,
        usage.limit as f64 / 1e9
    );
    let occupied = (free_before - free_after) as f64;
    println!(
        "  занято на карте       {:>7.2} GB  (свободно {:.2} -> {:.2} GB)",
        occupied / 1e9,
        free_before as f64 / 1e9,
        free_after as f64 / 1e9
    );
    // Потолок считает только наши буферы; разница — контекст CUDA и
    // гранулярность аллокатора драйвера, их движок не контролирует.
    println!(
        "  контекст CUDA         {:>7.2} GB",
        (occupied - usage.used as f64) / 1e9
    );
    println!("  пик host-RAM          {:>7.2} GB", peak_rss_bytes() / 1e9);

    println!("\nВремя");
    println!(
        "  загрузка              {:>7.1} s  ({:.2} GB/s)",
        elapsed,
        stats.resident_bytes() as f64 / 1e9 / elapsed
    );

    verify_first_projection(&checkpoint, &weights)?;
    verify_vocab(&checkpoint, &weights)?;
    Ok(())
}

/// Сверка одной поднятой проекции с чекпоинтом байт в байт: без неё отчёт
/// подтверждает только объём, но не то, что в VRAM лежат те самые веса.
fn verify_first_projection(
    checkpoint: &Checkpoint,
    weights: &ModelWeights,
) -> Result<(), Box<dyn std::error::Error>> {
    let layer = &weights.layers[0];
    let Mixer::Linear(mixer) = &layer.mixer else {
        return Err("слой 0 должен быть linear-attention".into());
    };
    let name = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight_packed";
    let expected = checkpoint
        .bytes(name)
        .ok_or("нет in_proj_qkv.weight_packed")?;
    // qkv лежит в начале слитой входной проекции: сверка заодно проверяет,
    // что склейка не переставила части местами.
    let fused = mixer.in_proj.linear.packed_to_host()?;
    let actual = fused
        .get(..expected.len())
        .ok_or("слитая проекция короче in_proj_qkv")?;
    if actual != expected {
        return Err(format!(
            "in_proj_qkv слоя 0 разошёлся с чекпоинтом ({} байт)",
            actual.len()
        )
        .into());
    }
    println!(
        "\nСверка: in_proj_qkv слоя 0 совпадает с началом слитой проекции \
         побайтно ({:.1} MB из {:.1} MB)",
        actual.len() as f64 / 1e6,
        fused.len() as f64 / 1e6
    );
    Ok(())
}

/// Ошибка перевода словарных матриц в FP8 на настоящих весах: без этих
/// цифр экономия 2.54 GB — обещание, а не измеренный размен.
fn verify_vocab(
    checkpoint: &Checkpoint,
    weights: &ModelWeights,
) -> Result<(), Box<dyn std::error::Error>> {
    const TOKENS: [u32; 6] = [0, 1, 151_643, 1000, 100_000, 248_319];
    /// Столько строк lm_head сверяется с точной BF16-арифметикой на CPU.
    const CHECKED_ROWS: usize = 2048;

    let stream = Stream::new()?;
    let embed_source = checkpoint
        .bytes("model.language_model.embed_tokens.weight")
        .ok_or("нет embed_tokens")?;
    let head_source = checkpoint.bytes("lm_head.weight").ok_or("нет lm_head")?;

    // Выборка эмбеддингов против BF16 из чекпоинта.
    let tokens = DeviceBuffer::from_slice(&TOKENS)?;
    let mut gathered = DeviceBuffer::<u16>::zeroed(TOKENS.len() * HIDDEN_SIZE)?;
    weights.embed.gather(&tokens, &mut gathered, &stream)?;
    stream.synchronize()?;
    let gathered = gathered.to_vec()?;

    let mut worst_embed = 0.0f64;
    let mut sum_embed = 0.0f64;
    for (index, &token) in TOKENS.iter().enumerate() {
        let row = bf16_row(embed_source, token as usize, HIDDEN_SIZE);
        let magnitude = row.iter().fold(0.0f32, |acc, v| acc.max(v.abs())) as f64;
        for column in 0..HIDDEN_SIZE {
            let actual = bf16::to_f32(gathered[index * HIDDEN_SIZE + column]) as f64;
            let error = (actual - row[column] as f64).abs() / magnitude;
            worst_embed = worst_embed.max(error);
            sum_embed += error;
        }
    }
    println!("\nFP8 против BF16 (настоящие веса)");
    println!(
        "  embed: макс {:.2}% от размаха строки, средняя {:.3}%",
        worst_embed * 100.0,
        sum_embed / (TOKENS.len() * HIDDEN_SIZE) as f64 * 100.0
    );

    // Логиты: эталон считается в BF16-арифметике на подмножестве строк.
    let hidden = bf16_row(embed_source, 1000, HIDDEN_SIZE)
        .iter()
        .map(|&value| bf16::from_f32(value))
        .collect::<Vec<u16>>();
    let device_hidden = DeviceBuffer::from_slice(&hidden)?;
    let mut device_logits = DeviceBuffer::<f32>::zeroed(VOCAB_SIZE)?;
    weights
        .lm_head
        .logits(&device_hidden, &mut device_logits, 1, &stream)?;
    stream.synchronize()?;
    let actual = device_logits.to_vec()?;

    let mut expected = vec![0.0f32; CHECKED_ROWS];
    for (row, slot) in expected.iter_mut().enumerate() {
        let weights_row = bf16_row(head_source, row, HIDDEN_SIZE);
        *slot = weights_row
            .iter()
            .zip(hidden.iter())
            .map(|(&w, &h)| w * bf16::to_f32(h))
            .sum();
    }
    let magnitude = expected.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
    let worst = expected
        .iter()
        .zip(actual.iter())
        .map(|(&want, &got)| (want - got).abs())
        .fold(0.0f32, f32::max);
    println!(
        "  lm_head: макс отклонение логита {:.4} при размахе {:.2} ({:.2}%), строк сверено {CHECKED_ROWS}",
        worst,
        magnitude,
        worst / magnitude * 100.0
    );

    let top_exact = argmax(&expected);
    let top_fp8 = argmax(&actual[..CHECKED_ROWS]);
    println!(
        "  argmax на этом подмножестве: точный {top_exact}, fp8 {top_fp8}{}",
        if top_exact == top_fp8 {
            ""
        } else {
            "  <- разошлись"
        }
    );
    Ok(())
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

/// Строка BF16-матрицы из mmap как f32.
fn bf16_row(raw: &[u8], row: usize, cols: usize) -> Vec<f32> {
    let start = row * cols * 2;
    raw[start..start + cols * 2]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| bf16::to_f32(u16::from_le_bytes(*pair)))
        .collect()
}

/// VmHWM — пик резидентной памяти процесса с момента старта.
fn peak_rss_bytes() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status
                .lines()
                .find(|line| line.starts_with("VmHWM:"))
                .and_then(|line| line.split_whitespace().nth(1))
                .and_then(|kb| kb.parse::<f64>().ok())
        })
        .map(|kb| kb * 1024.0)
        .unwrap_or(f64::NAN)
}

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/root".into()))
}

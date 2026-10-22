//! Loading model weights into VRAM: time, actual VRAM and the host-RAM ceiling.
//! `cargo run --release -p qwc-engine --bin load -- [directory] [--vram-gb N] [--mtp]`
//!
//! `--mtp` additionally loads the draft head: regular decode doesn't read it,
//! but the speculative path keeps it in the same VRAM, and the budget has to be counted with it.

use qwc_core::arch::{HIDDEN_SIZE, NUM_FULL_LAYERS, NUM_LAYERS, NUM_LINEAR_LAYERS, VOCAB_SIZE};
use qwc_cuda::{DeviceBuffer, Stream, bf16};
use qwc_engine::ModelWeights;
use qwc_engine::weights::Mixer;
use qwc_model::Checkpoint;
use std::path::PathBuf;
use std::time::Instant;

/// Default cap: weights plus headroom for cache, but noticeably below 32 GB —
/// so that hitting the limit shows up in a test, not only on an empty card.
const DEFAULT_VRAM_GB: f64 = 24.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut dir: Option<PathBuf> = None;
    let mut vram_gb = DEFAULT_VRAM_GB;
    let mut with_mtp = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--mtp" => with_mtp = true,
            "--vram-gb" => {
                vram_gb = args
                    .next()
                    .ok_or("--vram-gb requires a value")?
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

    println!("checkpoint: {}", dir.display());
    println!(
        "GPU: {} SM, VRAM free {:.2} / {:.2} GB, process cap {:.2} GB\n",
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
        return Err(format!("checkpoint diverged from the architecture: {}", problems.len()).into());
    }

    let started = Instant::now();
    let weights = ModelWeights::load_with_progress(&checkpoint, |index, stats| {
        if index % 8 == 7 || index + 1 == NUM_LAYERS {
            println!(
                "  layer {:>2}/{NUM_LAYERS}  {:>6.2} GB  {:>6.1} s",
                index + 1,
                stats.resident_bytes() as f64 / 1e9,
                started.elapsed().as_secs_f64()
            );
        }
    })?;
    let elapsed = started.elapsed().as_secs_f64();

    let mtp = if with_mtp {
        let started = Instant::now();
        let head = qwc_engine::mtp::load(&checkpoint)?;
        println!(
            "  MTP head       {:>6.2} GB  {:>6.1} s",
            head.resident_bytes() as f64 / 1e9,
            started.elapsed().as_secs_f64()
        );
        Some(head)
    } else {
        None
    };

    let stats = weights.stats();
    let usage = qwc_cuda::memory_usage();
    let (free_after, _) = qwc_cuda::Device::mem_info()?;
    let expected = checkpoint_stats.packed_bytes + checkpoint_stats.scale_bytes;

    println!("\nLoaded");
    println!(
        "  layers                {} ({} linear + {} full)",
        stats.layers, NUM_LINEAR_LAYERS, NUM_FULL_LAYERS
    );
    println!("  quantized projections {}", stats.projections);
    println!(
        "  NVFP4 weights + scales{:>7.2} GB",
        stats.quantized_bytes as f64 / 1e9
    );
    println!(
        "                        {:>7.2} GB in checkpoint (diff {:+.0} MB — scale-layout padding)",
        expected as f64 / 1e9,
        (stats.quantized_bytes as f64 - expected as f64) / 1e6
    );
    println!(
        "  norms, conv1d, gates  {:>7.2} MB",
        stats.plain_bytes as f64 / 1e6
    );
    println!(
        "  embed + lm_head fp8   {:>7.2} GB  (bf16 in checkpoint {:.2} GB)",
        stats.vocab_bytes as f64 / 1e9,
        (checkpoint_stats.embed_bytes + checkpoint_stats.lm_head_bytes) as f64 / 1e9
    );
    println!(
        "  total in VRAM         {:>7.2} GB",
        stats.resident_bytes() as f64 / 1e9
    );
    match &mtp {
        Some(head) => println!(
            "  + MTP head            {:>7.2} GB  -> {:.2} GB with it",
            head.resident_bytes() as f64 / 1e9,
            (stats.resident_bytes() + head.resident_bytes()) as f64 / 1e9
        ),
        None => println!("  MTP head              not loaded (--mtp)"),
    }

    println!("\nMemory");
    println!(
        "  counted by the cap    {:>7.2} GB of {:.2} GB",
        usage.used as f64 / 1e9,
        usage.limit as f64 / 1e9
    );
    let occupied = (free_before - free_after) as f64;
    println!(
        "  used on the card      {:>7.2} GB  (free {:.2} -> {:.2} GB)",
        occupied / 1e9,
        free_before as f64 / 1e9,
        free_after as f64 / 1e9
    );
    // The cap counts only our buffers; the difference is the CUDA context and
    // the driver allocator's granularity, which the engine doesn't control.
    println!(
        "  CUDA context          {:>7.2} GB",
        (occupied - usage.used as f64) / 1e9
    );
    println!("  peak host RAM         {:>7.2} GB", peak_rss_bytes() / 1e9);

    println!("\nTime");
    println!(
        "  load                  {:>7.1} s  ({:.2} GB/s)",
        elapsed,
        stats.resident_bytes() as f64 / 1e9 / elapsed
    );

    verify_first_projection(&checkpoint, &weights)?;
    verify_vocab(&checkpoint, &weights)?;
    Ok(())
}

/// A byte-for-byte check of one loaded projection against the checkpoint: without it the report
/// confirms only the size, not that the right weights are in VRAM.
fn verify_first_projection(
    checkpoint: &Checkpoint,
    weights: &ModelWeights,
) -> Result<(), Box<dyn std::error::Error>> {
    let layer = &weights.layers[0];
    let Mixer::Linear(mixer) = &layer.mixer else {
        return Err("layer 0 must be linear-attention".into());
    };
    let name = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight_packed";
    let expected = checkpoint
        .bytes(name)
        .ok_or("missing in_proj_qkv.weight_packed")?;
    // qkv lies at the start of the fused input projection: the check also verifies
    // that the concatenation didn't swap parts.
    let fused = mixer.in_proj.linear.packed_to_host()?;
    let actual = fused
        .get(..expected.len())
        .ok_or("fused projection is shorter than in_proj_qkv")?;
    if actual != expected {
        return Err(format!(
            "in_proj_qkv of layer 0 diverged from the checkpoint ({} bytes)",
            actual.len()
        )
        .into());
    }
    println!(
        "\nCheck: in_proj_qkv of layer 0 matches the start of the fused projection \
         byte for byte ({:.1} MB of {:.1} MB)",
        actual.len() as f64 / 1e6,
        fused.len() as f64 / 1e6
    );
    Ok(())
}

/// The error of converting the vocab matrices to FP8 on real weights: without these
/// numbers the 2.54 GB saving is a promise, not a measured trade-off.
fn verify_vocab(
    checkpoint: &Checkpoint,
    weights: &ModelWeights,
) -> Result<(), Box<dyn std::error::Error>> {
    const TOKENS: [u32; 6] = [0, 1, 151_643, 1000, 100_000, 248_319];
    /// This many lm_head rows are checked against exact BF16 arithmetic on CPU.
    const CHECKED_ROWS: usize = 2048;

    let stream = Stream::new()?;
    let embed_source = checkpoint
        .bytes("model.language_model.embed_tokens.weight")
        .ok_or("missing embed_tokens")?;
    let head_source = checkpoint.bytes("lm_head.weight").ok_or("missing lm_head")?;

    // Embedding lookup vs BF16 from the checkpoint.
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
    println!("\nFP8 vs BF16 (real weights)");
    println!(
        "  embed: max {:.2}% of row range, mean {:.3}%",
        worst_embed * 100.0,
        sum_embed / (TOKENS.len() * HIDDEN_SIZE) as f64 * 100.0
    );

    // Logits: the reference is computed in BF16 arithmetic on a subset of rows.
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
        "  lm_head: max logit deviation {:.4} at range {:.2} ({:.2}%), rows checked {CHECKED_ROWS}",
        worst,
        magnitude,
        worst / magnitude * 100.0
    );

    let top_exact = argmax(&expected);
    let top_fp8 = argmax(&actual[..CHECKED_ROWS]);
    println!(
        "  argmax on this subset: exact {top_exact}, fp8 {top_fp8}{}",
        if top_exact == top_fp8 {
            ""
        } else {
            "  <- diverged"
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

/// A row of a BF16 matrix from mmap as f32.
fn bf16_row(raw: &[u8], row: usize, cols: usize) -> Vec<f32> {
    let start = row * cols * 2;
    raw[start..start + cols * 2]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| bf16::to_f32(u16::from_le_bytes(*pair)))
        .collect()
}

/// VmHWM — the process's peak resident memory since start.
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

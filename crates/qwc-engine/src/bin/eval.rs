//! Machine-readable differential-eval runner for the native executor.
//!
//! Stdout is JSONL. Human progress is written to stderr so callers can save
//! stdout directly as an artifact.

use qwc_core::arch::VOCAB_SIZE;
use qwc_cuda::delta_net::DeltaStateMode;
use qwc_cuda::paged_attention::KvCacheDtype;
use qwc_engine::{
    DecodeLinearMode, EmbeddingDtype, Executor, ExecutorConfig, LmHeadDtype, LoadConfig,
    ModelWeights,
};
use qwc_model::Checkpoint;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::Instant;

const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Deserialize)]
struct CorpusCase {
    id: String,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    prompt_token_ids: Vec<u32>,
    #[serde(default)]
    repeat: Option<RepeatTokens>,
    #[serde(default = "default_max_new_tokens")]
    max_new_tokens: usize,
    #[serde(default)]
    tags: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct RepeatTokens {
    token_ids: Vec<u32>,
    length: usize,
}

impl CorpusCase {
    fn expanded_tokens(&self) -> Result<Vec<u32>, String> {
        match (&self.prompt_token_ids[..], &self.repeat) {
            ([], Some(repeat)) => {
                if repeat.token_ids.is_empty() {
                    return Err(format!("{}: repeat.token_ids is empty", self.id));
                }
                Ok(repeat
                    .token_ids
                    .iter()
                    .copied()
                    .cycle()
                    .take(repeat.length)
                    .collect())
            }
            (tokens @ [_, ..], None) => Ok(tokens.to_vec()),
            ([], None) => Err(format!("{}: no prompt tokens", self.id)),
            (_, Some(_)) => Err(format!(
                "{}: specify prompt_token_ids or repeat, not both",
                self.id
            )),
        }
    }
}

fn default_max_new_tokens() -> usize {
    16
}

#[derive(Serialize)]
struct RunRecord<'a> {
    record_type: &'static str,
    schema_version: u32,
    engine: &'static str,
    engine_version: &'a str,
    model: String,
    top_k: usize,
    batch: usize,
    logits: &'static str,
    embedding: &'static str,
    lm_head: &'static str,
    kv_cache: &'static str,
    decode_linear: &'static str,
    delta_state: &'static str,
}

#[derive(Serialize)]
struct CaseRecord<'a> {
    record_type: &'static str,
    case_id: &'a str,
    prompt: &'a Option<String>,
    prompt_token_ids: &'a [u32],
    tags: &'a [String],
    status: &'static str,
    steps: Vec<StepRecord>,
    timing: TimingRecord,
}

#[derive(Serialize)]
struct StepRecord {
    index: usize,
    token_id: u32,
    top_tokens: Vec<TokenScore>,
}

#[derive(Serialize)]
struct TokenScore {
    token_id: u32,
    logit: f32,
    logprob: f32,
    rank: usize,
}

#[derive(Serialize)]
struct TimingRecord {
    prefill_ms: f64,
    decode_ms: f64,
}

struct Args {
    model: PathBuf,
    corpus: PathBuf,
    top_k: usize,
    batch: usize,
    max_context: Option<usize>,
    embedding: EmbeddingDtype,
    lm_head: LmHeadDtype,
    kv_cache: KvCacheDtype,
    decode_linear: DecodeLinearMode,
    delta_state: DeltaStateMode,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    let cases = read_corpus(&args.corpus)?;
    let expanded: Vec<Vec<u32>> = cases
        .iter()
        .map(CorpusCase::expanded_tokens)
        .collect::<Result<_, _>>()?;
    for (case, tokens) in cases.iter().zip(&expanded) {
        if tokens.iter().any(|&token| token as usize >= VOCAB_SIZE) {
            return Err(format!("{}: token outside vocabulary", case.id).into());
        }
        if case.max_new_tokens == 0 {
            return Err(format!("{}: max_new_tokens must be positive", case.id).into());
        }
    }
    let needed_context = cases
        .iter()
        .zip(&expanded)
        .map(|(case, tokens)| tokens.len() + case.max_new_tokens)
        .max()
        .ok_or("empty corpus")?;
    let max_context = args.max_context.unwrap_or(needed_context);
    if max_context < needed_context {
        return Err(
            format!("--context {max_context} is smaller than required {needed_context}").into(),
        );
    }

    qwc_cuda::Device::init(0)?;
    qwc_cuda::set_memory_limit(24 * 1_000_000_000)?;
    let checkpoint = Checkpoint::open(&args.model)?;
    let load_started = Instant::now();
    let weights = ModelWeights::load_with_config(
        &checkpoint,
        LoadConfig {
            embedding: args.embedding,
            lm_head: args.lm_head,
            decode_linear: args.decode_linear,
        },
    )?;
    eprintln!(
        "qwc: loaded {:.2} GB in {:.1} s",
        weights.stats().resident_bytes() as f64 / 1e9,
        load_started.elapsed().as_secs_f64()
    );
    let mut executor = Executor::new_with_options(
        ExecutorConfig {
            max_batch: args.batch,
            max_context,
        },
        args.kv_cache,
        args.decode_linear,
        args.delta_state,
    )?;

    print_json(&RunRecord {
        record_type: "run",
        schema_version: SCHEMA_VERSION,
        engine: "qwc",
        engine_version: env!("CARGO_PKG_VERSION"),
        model: canonical_display(&args.model),
        top_k: args.top_k,
        batch: args.batch,
        logits: "raw_full_vocab",
        embedding: args.embedding.as_str(),
        lm_head: args.lm_head.as_str(),
        kv_cache: args.kv_cache.as_str(),
        decode_linear: args.decode_linear.as_str(),
        delta_state: args.delta_state.as_str(),
    })?;

    for (case, prompt_tokens) in cases.iter().zip(&expanded) {
        eprintln!(
            "qwc: {} ({} prompt + {} generated tokens, batch {})",
            case.id,
            prompt_tokens.len(),
            case.max_new_tokens,
            args.batch
        );
        let prefill_started = Instant::now();
        for slot in 0..args.batch {
            executor.prefill_sequence(&weights, prompt_tokens, 0, slot)?;
        }
        let prefill_ms = prefill_started.elapsed().as_secs_f64() * 1e3;

        let decode_started = Instant::now();
        let mut steps = Vec::with_capacity(case.max_new_tokens);
        for step in 0..case.max_new_tokens {
            let logits = executor.logits_to_host(0)?;
            let top_tokens = summarize_logits(&logits, args.top_k);
            let gpu_tokens = executor.argmax_to_host(args.batch)?;
            let token_id = gpu_tokens[0];
            if top_tokens[0].token_id != token_id {
                return Err(format!(
                    "{} step {step}: GPU argmax {} != host argmax {}",
                    case.id, token_id, top_tokens[0].token_id
                )
                .into());
            }
            steps.push(StepRecord {
                index: step,
                token_id,
                top_tokens,
            });
            if step + 1 < case.max_new_tokens {
                let position = (prompt_tokens.len() + step) as u32;
                executor.decode(
                    &weights,
                    &vec![token_id; args.batch],
                    &vec![position; args.batch],
                )?;
            }
        }
        let decode_ms = decode_started.elapsed().as_secs_f64() * 1e3;
        print_json(&CaseRecord {
            record_type: "case",
            case_id: &case.id,
            prompt: &case.prompt,
            prompt_token_ids: prompt_tokens,
            tags: &case.tags,
            status: "ok",
            steps,
            timing: TimingRecord {
                prefill_ms,
                decode_ms,
            },
        })?;
    }
    Ok(())
}

fn summarize_logits(logits: &[f32], top_k: usize) -> Vec<TokenScore> {
    assert!(!logits.is_empty());
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f64 = logits
        .iter()
        .map(|&value| f64::from(value - maximum).exp())
        .sum();
    let logsumexp = maximum + sum_exp.ln() as f32;
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.select_nth_unstable_by(top_k - 1, |&left, &right| {
        logits[right]
            .total_cmp(&logits[left])
            .then_with(|| left.cmp(&right))
    });
    indices.truncate(top_k);
    indices.sort_unstable_by(|&left, &right| {
        logits[right]
            .total_cmp(&logits[left])
            .then_with(|| left.cmp(&right))
    });
    indices
        .into_iter()
        .enumerate()
        .map(|(rank, token)| TokenScore {
            token_id: token as u32,
            logit: logits[token],
            logprob: logits[token] - logsumexp,
            rank: rank + 1,
        })
        .collect()
}

fn read_corpus(path: &Path) -> Result<Vec<CorpusCase>, Box<dyn std::error::Error>> {
    let reader = BufReader::new(File::open(path)?);
    let mut cases = Vec::new();
    for (index, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() || line.trim_start().starts_with('#') {
            continue;
        }
        let case: CorpusCase = serde_json::from_str(&line)
            .map_err(|error| format!("{}:{}: {error}", path.display(), index + 1))?;
        cases.push(case);
    }
    Ok(cases)
}

fn print_json(value: &impl Serialize) -> Result<(), serde_json::Error> {
    println!("{}", serde_json::to_string(value)?);
    Ok(())
}

fn parse_args() -> Result<Args, Box<dyn std::error::Error>> {
    let mut model = None;
    let mut corpus = None;
    let mut top_k = 20usize;
    let mut batch = 1usize;
    let mut max_context = None;
    let mut embedding = EmbeddingDtype::Fp8;
    let mut lm_head = LmHeadDtype::Fp8;
    let mut delta_state = DeltaStateMode::Bf16;
    let mut kv_cache = KvCacheDtype::Fp8;
    let mut decode_linear = DecodeLinearMode::Auto;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model = Some(PathBuf::from(args.next().ok_or("--model needs a path")?)),
            "--corpus" => corpus = Some(PathBuf::from(args.next().ok_or("--corpus needs a path")?)),
            "--top-k" => top_k = args.next().ok_or("--top-k needs a number")?.parse()?,
            "--batch" => batch = args.next().ok_or("--batch needs a number")?.parse()?,
            "--context" => {
                max_context = Some(args.next().ok_or("--context needs a number")?.parse()?)
            }
            "--lm-head" => {
                lm_head = match args.next().ok_or("--lm-head needs fp8 or bf16")?.as_str() {
                    "fp8" => LmHeadDtype::Fp8,
                    "bf16" => LmHeadDtype::Bf16,
                    other => return Err(format!("unsupported --lm-head value: {other}").into()),
                }
            }
            "--delta-state" => {
                delta_state = match args
                    .next()
                    .ok_or("--delta-state needs bf16, fp32 or wy")?
                    .as_str()
                {
                    "bf16" => DeltaStateMode::Bf16,
                    "fp32" => DeltaStateMode::Fp32,
                    "wy" => DeltaStateMode::Wy,
                    other => return Err(format!("unsupported --delta-state value: {other}").into()),
                };
            }
            "--embedding" => {
                embedding = match args.next().ok_or("--embedding needs fp8 or bf16")?.as_str() {
                    "fp8" => EmbeddingDtype::Fp8,
                    "bf16" => EmbeddingDtype::Bf16,
                    other => return Err(format!("unsupported --embedding value: {other}").into()),
                }
            }
            "--kv-cache" => {
                kv_cache = match args.next().ok_or("--kv-cache needs fp8 or bf16")?.as_str() {
                    "fp8" => KvCacheDtype::Fp8,
                    "bf16" => KvCacheDtype::Bf16,
                    other => return Err(format!("unsupported --kv-cache value: {other}").into()),
                }
            }
            "--decode-linear" => {
                decode_linear = match args
                    .next()
                    .ok_or("--decode-linear needs auto or w4a4")?
                    .as_str()
                {
                    "auto" => DecodeLinearMode::Auto,
                    "w4a4" => DecodeLinearMode::W4A4,
                    other => {
                        return Err(format!("unsupported --decode-linear value: {other}").into());
                    }
                }
            }
            "-h" | "--help" => {
                println!(
                    "usage: qwc-eval --model PATH --corpus FILE [--top-k 20] [--batch 1] [--context N] [--embedding fp8|bf16] [--lm-head fp8|bf16] [--kv-cache fp8|bf16] [--decode-linear auto|w4a4] [--delta-state bf16|fp32|wy]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }
    let model = model.unwrap_or_else(|| home().join("models/Qwen3.8-27B-QUASAR-NVFP4"));
    let corpus = corpus.ok_or("--corpus is required")?;
    if top_k == 0 || top_k > VOCAB_SIZE {
        return Err(format!("--top-k must be in 1..={VOCAB_SIZE}").into());
    }
    if !(1..=qwc_engine::executor::MAX_BATCH).contains(&batch) {
        return Err(format!("--batch must be in 1..={}", qwc_engine::executor::MAX_BATCH).into());
    }
    Ok(Args {
        model,
        corpus,
        top_k,
        batch,
        max_context,
        embedding,
        lm_head,
        delta_state,
        kv_cache,
        decode_linear,
    })
}

fn canonical_display(path: &Path) -> String {
    path.canonicalize()
        .unwrap_or_else(|_| path.to_path_buf())
        .display()
        .to_string()
}

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/root".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logits_are_normalized_and_ties_use_smaller_token() {
        let top = summarize_logits(&[1.0, 2.0, 2.0, -1.0], 3);
        assert_eq!(
            top.iter().map(|item| item.token_id).collect::<Vec<_>>(),
            [1, 2, 0]
        );
        let probability_sum: f32 = [1.0_f32, 2.0, 2.0, -1.0]
            .iter()
            .map(|value| {
                (*value - (1.0_f32.exp() + 2.0_f32.exp() * 2.0 + (-1.0_f32).exp()).ln()).exp()
            })
            .sum();
        assert!((probability_sum - 1.0).abs() < 1e-5);
        assert!((top[0].logprob - top[1].logprob).abs() < f32::EPSILON);
    }
}

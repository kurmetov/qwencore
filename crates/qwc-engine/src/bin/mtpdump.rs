//! Выгрузка скрытых состояний под замер acceptance MTP-головы.
//!
//! `cargo run --release -p qwc-engine --bin mtpdump -- --corpus bench/corpus/core.jsonl --max-new 128 --out /tmp/mtp`
//!
//! Draft-голова принимает на вход пару (h_t, эмбеддинг токена t+1) и должна
//! предсказать токен t+2. Здесь снимается левая часть этой пары: финальные
//! скрытые состояния всех позиций и сами токены. Дальше
//! `bench/mtp_acceptance.py` считает голову и сверяет её argmax с тем, что
//! выдала настоящая модель.
//!
//! Формат файла: "QWCMTP01", четыре u32 (hidden, позиций, длина промпта,
//! токенов), затем токены u32 и скрытые состояния bf16.

use qwc_core::arch::{HIDDEN_SIZE, VOCAB_SIZE};
use qwc_engine::{Executor, ExecutorConfig, ModelWeights, PREFILL_CHUNK_SIZE};
use qwc_model::Checkpoint;
use std::io::Write;
use std::path::PathBuf;

struct Case {
    id: String,
    prompt: Vec<u32>,
}

fn parse_cases(text: &str) -> Vec<Case> {
    // Корпус — jsonl с полями id и prompt_token_ids. Полноценный парсер json
    // тут не нужен: формат наш собственный и стабильный.
    let mut cases = Vec::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let id = field(line, "\"id\":").unwrap_or_else(|| "case".into());
        // Корпус знает две формы промпта: явный список токенов и повтор
        // короткого куска до заданной длины.
        let prompt = match (list(line, "\"prompt_token_ids\":"), list(line, "\"token_ids\":")) {
            (Some(ids), _) => ids,
            (None, Some(unit)) => {
                let length: usize = line
                    .split("\"length\":")
                    .nth(1)
                    .and_then(|rest| rest.split([',', '}']).next())
                    .and_then(|x| x.trim().parse().ok())
                    .expect("в форме repeat нет length");
                unit.iter().cycle().take(length).copied().collect()
            }
            _ => panic!("в строке корпуса нет ни prompt_token_ids, ни token_ids"),
        };
        cases.push(Case { id, prompt });
    }
    cases
}

fn list(line: &str, key: &str) -> Option<Vec<u32>> {
    let rest = line.split(key).nth(1)?;
    let inside = rest.trim_start().strip_prefix('[')?.split(']').next()?;
    inside
        .split(',')
        .map(|x| x.trim().parse::<u32>().ok())
        .collect()
}

fn field(line: &str, key: &str) -> Option<String> {
    let rest = line.split(key).nth(1)?;
    let rest = rest.trim_start().strip_prefix('"')?;
    Some(rest.split('"').next()?.to_string())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model = home().join("models/Qwen3.8-27B-QUASAR-NVFP4");
    let mut corpus = PathBuf::from("bench/corpus/core.jsonl");
    let mut out = PathBuf::from("/tmp/mtp");
    let mut max_new = 128usize;
    let mut context = 4096usize;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model = PathBuf::from(args.next().ok_or("--model")?),
            "--corpus" => corpus = PathBuf::from(args.next().ok_or("--corpus")?),
            "--out" => out = PathBuf::from(args.next().ok_or("--out")?),
            "--max-new" => max_new = args.next().ok_or("--max-new")?.parse()?,
            "--context" => context = args.next().ok_or("--context")?.parse()?,
            other => return Err(format!("неизвестный аргумент: {other}").into()),
        }
    }

    let cases = parse_cases(&std::fs::read_to_string(&corpus)?);
    assert!(!cases.is_empty(), "корпус пуст");
    std::fs::create_dir_all(&out)?;

    let checkpoint = Checkpoint::open(&model)?;
    let weights = ModelWeights::load(&checkpoint)?;
    let mut executor = Executor::new(ExecutorConfig {
        max_batch: 1,
        max_context: context,
    })?;

    for case in &cases {
        assert!(
            case.prompt.len() <= PREFILL_CHUNK_SIZE,
            "{}: промпт длиннее одного чанка, скрытые состояния остались бы только от последнего",
            case.id
        );
        assert!(case.prompt.len() + max_new <= context, "{}: не влезает в контекст", case.id);
        assert!(case.prompt.iter().all(|&t| (t as usize) < VOCAB_SIZE));

        let mut tokens = case.prompt.clone();
        let mut hidden: Vec<u16> = Vec::with_capacity((case.prompt.len() + max_new) * HIDDEN_SIZE);

        executor.prefill_sequence(&weights, &case.prompt, 0, 0)?;
        hidden.extend_from_slice(&executor.prefill_hidden_to_host(case.prompt.len())?);
        let mut next = executor.argmax_to_host(1)?;

        for step in 0..max_new {
            tokens.push(next[0]);
            let position = (case.prompt.len() + step) as u32;
            executor.decode(&weights, &next, &[position])?;
            hidden.extend_from_slice(&executor.decode_hidden_to_host(1)?);
            next = executor.argmax_to_host(1)?;
        }
        tokens.push(next[0]);

        let positions = case.prompt.len() + max_new;
        assert_eq!(hidden.len(), positions * HIDDEN_SIZE);
        assert_eq!(tokens.len(), positions + 1);

        let path = out.join(format!("{}.bin", case.id));
        let mut file = std::io::BufWriter::new(std::fs::File::create(&path)?);
        file.write_all(b"QWCMTP01")?;
        for value in [
            HIDDEN_SIZE as u32,
            positions as u32,
            case.prompt.len() as u32,
            tokens.len() as u32,
        ] {
            file.write_all(&value.to_le_bytes())?;
        }
        for token in &tokens {
            file.write_all(&token.to_le_bytes())?;
        }
        for value in &hidden {
            file.write_all(&value.to_le_bytes())?;
        }
        file.flush()?;
        println!(
            "{:12} промпт {:4}, позиций {:5} -> {}",
            case.id,
            case.prompt.len(),
            positions,
            path.display()
        );
    }
    Ok(())
}

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME"))
}

//! Сверка чекпоинта с архитектурой и реальный план VRAM.
//! `cargo run -p qwc-model --bin inspect -- <каталог>`

use qwc_core::dtype::Dtype;
use qwc_core::memory::{CacheConfig, GB, rtx5090};
use qwc_model::Checkpoint;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| dirs_home().join("models/Qwen3.8-27B-QUASAR-NVFP4"));

    println!("чекпоинт: {}\n", dir.display());
    let ck = Checkpoint::open(&dir)?;
    let (problems, st) = ck.validate();

    if problems.is_empty() {
        println!("сверка с архитектурой: OK");
    } else {
        println!("сверка с архитектурой: {} расхождений", problems.len());
        for p in problems.iter().take(15) {
            println!("  {p}");
        }
        if problems.len() > 15 {
            println!("  ... ещё {}", problems.len() - 15);
        }
    }

    let g = |b: u64| b as f64 / 1e9;
    println!("\nтензоров в чекпоинте: {}", ck.tensor_count());
    println!("из них разобрано:     {}", st.tensors);
    println!("квантованных linear:  {}", st.quantized_linears);

    println!("\nРаскладка чекпоинта");
    println!("  веса NVFP4 (packed)   {:>7.2} GB", g(st.packed_bytes));
    println!("  поблочные шкалы fp8   {:>7.2} GB", g(st.scale_bytes));
    println!("  нормы и мелочь        {:>7.2} GB", g(st.plain_bytes));
    println!("  embed_tokens bf16     {:>7.2} GB", g(st.embed_bytes));
    println!("  lm_head bf16          {:>7.2} GB", g(st.lm_head_bytes));
    println!("  MTP-голова bf16       {:>7.2} GB", g(st.mtp_bytes));
    println!(
        "  не нужно (vision)     {:>7.2} GB  ({} тензоров)",
        g(st.unused_bytes),
        st.unused_tensors
    );
    println!(
        "  итого на диске        {:>7.2} GB",
        g(st.engine_bytes() + st.unused_bytes)
    );

    // Наши преобразования при загрузке.
    let head_fp8 = Dtype::Fp8E4m3.bytes(qwc_core::arch::LM_HEAD_PARAMS);
    let embed_fp8 = Dtype::Fp8E4m3.bytes(qwc_core::arch::EMBED_PARAMS);
    let ours =
        st.packed_bytes + st.scale_bytes + st.plain_bytes + st.mtp_bytes + head_fp8 + embed_fp8;

    println!("\nЧто грузим мы (text-only, lm_head и embed в fp8)");
    println!(
        "  vLLM грузит           {:>7.2} GB",
        g(st.engine_bytes() + st.unused_bytes)
    );
    println!("  мы грузим             {:>7.2} GB", g(ours));
    println!(
        "  фора                  {:>7.2} GB",
        g(st.engine_bytes() + st.unused_bytes - ours)
    );

    let cfg = CacheConfig::default();
    let b = rtx5090(ours, 2 * GB);
    println!(
        "\nПод кэш остаётся        {:>7.2} GB",
        g(b.cache_available())
    );
    println!("  max concurrency @4K   {}", b.max_concurrency(&cfg, 4096));
    println!("  max concurrency @32K  {}", b.max_concurrency(&cfg, 32768));
    Ok(())
}

fn dirs_home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/root".into()))
}

//! Checks the checkpoint against the architecture and prints the real VRAM plan.
//! `cargo run -p qwc-model --bin inspect -- <dir>`

use qwc_core::dtype::Dtype;
use qwc_core::memory::{CacheConfig, GB, rtx5090};
use qwc_model::Checkpoint;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| dirs_home().join("models/Qwen3.8-27B-QUASAR-NVFP4"));

    println!("checkpoint: {}\n", dir.display());
    let ck = Checkpoint::open(&dir)?;
    let (problems, st) = ck.validate();

    if problems.is_empty() {
        println!("architecture check: OK");
    } else {
        println!("architecture check: {} mismatches", problems.len());
        for p in problems.iter().take(15) {
            println!("  {p}");
        }
        if problems.len() > 15 {
            println!("  ... {} more", problems.len() - 15);
        }
    }

    let g = |b: u64| b as f64 / 1e9;
    println!("\ntensors in checkpoint: {}", ck.tensor_count());
    println!("of them parsed:        {}", st.tensors);
    println!("quantized linears:     {}", st.quantized_linears);

    println!("\nCheckpoint layout");
    println!("  NVFP4 weights (packed) {:>7.2} GB", g(st.packed_bytes));
    println!("  fp8 block scales       {:>7.2} GB", g(st.scale_bytes));
    println!("  norms and small stuff  {:>7.2} GB", g(st.plain_bytes));
    println!("  embed_tokens bf16     {:>7.2} GB", g(st.embed_bytes));
    println!("  lm_head bf16          {:>7.2} GB", g(st.lm_head_bytes));
    println!("  MTP head bf16          {:>7.2} GB", g(st.mtp_bytes));
    println!(
        "  not needed (vision)    {:>7.2} GB  ({} tensors)",
        g(st.unused_bytes),
        st.unused_tensors
    );
    println!(
        "  total on disk          {:>7.2} GB",
        g(st.engine_bytes() + st.unused_bytes)
    );

    // Our transformations at load time.
    let head_fp8 = Dtype::Fp8E4m3.bytes(qwc_core::arch::LM_HEAD_PARAMS);
    let embed_fp8 = Dtype::Fp8E4m3.bytes(qwc_core::arch::EMBED_PARAMS);
    let ours =
        st.packed_bytes + st.scale_bytes + st.plain_bytes + st.mtp_bytes + head_fp8 + embed_fp8;

    println!("\nWhat we load (text-only, lm_head and embed in fp8)");
    println!(
        "  vLLM loads             {:>7.2} GB",
        g(st.engine_bytes() + st.unused_bytes)
    );
    println!("  we load                {:>7.2} GB", g(ours));
    println!(
        "  head start             {:>7.2} GB",
        g(st.engine_bytes() + st.unused_bytes - ours)
    );

    let cfg = CacheConfig::default();
    let b = rtx5090(ours, 2 * GB);
    println!(
        "\nLeft for cache          {:>7.2} GB",
        g(b.cache_available())
    );
    println!("  max concurrency @4K   {}", b.max_concurrency(&cfg, 4096));
    println!("  max concurrency @32K  {}", b.max_concurrency(&cfg, 32768));
    Ok(())
}

fn dirs_home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/root".into()))
}

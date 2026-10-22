//! Prints the RTX 5090 VRAM plan for different cache configurations.
//! `cargo run -p qwc-core --bin plan`

use qwc_core::arch::*;
use qwc_core::dtype::Dtype;
use qwc_core::memory::*;
use qwc_core::roofline::*;

fn gb(b: u64) -> f64 {
    b as f64 / 1e9
}

fn main() {
    println!("Qwen3.8-27B / RTX 5090 32GB / text-only\n");

    println!("Architecture");
    println!(
        "  layers: {NUM_LAYERS}  ({NUM_LINEAR_LAYERS} DeltaNet + {NUM_FULL_LAYERS} full attention)"
    );
    println!("  hidden {HIDDEN_SIZE}, intermediate {INTERMEDIATE_SIZE}, vocab {VOCAB_SIZE}");
    println!(
        "  attention: {NUM_ATTN_HEADS}q / {NUM_KV_HEADS}kv, head_dim {ATTN_HEAD_DIM}, RoPE on {ROPE_DIM} of {ATTN_HEAD_DIM}"
    );
    println!(
        "  DeltaNet: {LA_NUM_K_HEADS}k / {LA_NUM_V_HEADS}v, dim {LA_K_HEAD_DIM}, conv {LA_CONV_KERNEL}"
    );
    println!("  parameters: {:.2} B\n", TEXT_PARAMS as f64 / 1e9);

    let ours = WeightPlan::default();
    let shipped = WeightPlan::as_shipped();
    println!("Weights");
    println!(
        "  checkpoint as-is (vLLM loads):           {:>6.2} GB",
        gb(shipped.bytes())
    );
    println!(
        "  our load (text-only, fp8 head/embed):    {:>6.2} GB",
        gb(ours.bytes())
    );
    println!(
        "  read by plain decode b=1:                {:>6.2} GB",
        gb(ours.decode_weight_bytes(1))
    );
    println!("  head start: {:>6.2} GB\n", gb(shipped.bytes() - ours.bytes()));

    let cfg = CacheConfig::default();
    println!(
        "Cache (kv {}, state {})",
        cfg.kv_dtype.name(),
        cfg.state_dtype.name()
    );
    println!(
        "  KV per token (16 layers):      {:>8} B",
        cfg.kv_bytes_per_token()
    );
    println!(
        "  state per slot (48 layers):    {:>8.1} MB",
        cfg.state_bytes_per_slot() as f64 / 1e6
    );
    println!(
        "  crossover point:               {:>8} tokens\n",
        cfg.crossover_tokens()
    );

    let b = rtx5090(ours.bytes(), 2 * GB);
    println!("Budget");
    println!("  total VRAM      {:>6.2} GB", gb(b.total_vram));
    println!(
        "  reserved        {:>6.2} GB (desktop + CUDA context)",
        gb(b.reserved)
    );
    println!("  weights         {:>6.2} GB", gb(b.weights));
    println!("  workspace       {:>6.2} GB", gb(b.workspace));
    println!("  for cache       {:>6.2} GB\n", gb(b.cache_available()));

    println!("Max concurrency by context length");
    println!(
        "  {:>8} | {:>10} | {:>10} | {:>12}",
        "context", "fp8 KV", "nvfp4 KV", "vLLM-like fp8"
    );
    println!("  {:->8}-+-{:->10}-+-{:->10}-+-{:->12}", "", "", "", "");
    let fp4 = CacheConfig {
        kv_dtype: Dtype::Nvfp4,
        ..cfg
    };
    let base = rtx5090(shipped.bytes(), 2 * GB);
    for ctx in [4096usize, 8192, 16384, 32768, 65536, 131072] {
        println!(
            "  {:>8} | {:>10} | {:>10} | {:>12}",
            ctx,
            b.max_concurrency(&cfg, ctx),
            b.max_concurrency(&fp4, ctx),
            base.max_concurrency(&cfg, ctx),
        );
    }

    println!("\nTarget benchmark matrix");
    for (name, conc, inp, out) in [
        ("single latency", 1usize, 512usize, 256usize),
        ("long context", 1, 16 * 1024, 512),
        ("light", 4, 2048, 512),
        ("medium", 8, 2048, 512),
        ("high", 16, 2048, 512),
        ("saturation", 32, 2048, 512),
    ] {
        let total = inp + out;
        let used = cfg.bytes_per_seq(total) * conc as u64;
        println!(
            "  {:<15} c={:<3} ctx={:<6} {:>6.2} GB  {}",
            name,
            conc,
            total,
            gb(used),
            if b.fits(&cfg, conc, total) {
                "ok"
            } else {
                "OOM"
            }
        );
    }
    println!("\nDecode ceiling (memory-bound: each step reads all weights)");
    println!(
        "  {:>6} | {:>7} | {:>12} | {:>12} | {:>12}",
        "batch", "ctx", "ITL ideal", "ITL @53%", "tok/s ideal"
    );
    println!(
        "  {:->6}-+-{:->7}-+-{:->12}-+-{:->12}-+-{:->12}",
        "", "", "", "", ""
    );
    for (batch, ctx) in [
        (1usize, 2048usize),
        (1, 32768),
        (8, 2048),
        (16, 2048),
        (32, 2048),
        (32, 8192),
    ] {
        let ideal = DecodeStep {
            batch,
            context_len: ctx,
            bandwidth_efficiency: 1.0,
        };
        let real = DecodeStep {
            bandwidth_efficiency: 0.53,
            ..ideal
        };
        let decode_weights = ours.decode_weight_bytes(batch);
        println!(
            "  {:>6} | {:>7} | {:>9.2} ms | {:>9.2} ms | {:>9.0} t/s",
            batch,
            ctx,
            ideal.itl_ms(decode_weights, &cfg),
            real.itl_ms(decode_weights, &cfg),
            ideal.tokens_per_sec(decode_weights, &cfg),
        );
    }

    println!("\nWhat a decode step consists of (batch=32, ctx=2048)");
    let step = DecodeStep {
        batch: 32,
        context_len: 2048,
        bandwidth_efficiency: 1.0,
    };
    let c = step.cost(ours.decode_weight_bytes(step.batch), &cfg);
    for (name, bytes) in [
        ("weights", c.weight_bytes),
        ("DeltaNet state", c.state_bytes),
        ("KV cache", c.kv_bytes),
    ] {
        println!(
            "  {:<20} {:>7.2} GB  {:>4.0}%",
            name,
            gb(bytes),
            100.0 * bytes as f64 / c.total() as f64
        );
    }
    println!(
        "  compute-bound transition at batch ~{:.0}",
        compute_bound_batch(Dtype::Nvfp4)
    );
}

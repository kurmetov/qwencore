//! Печатает план VRAM для RTX 5090 под разные конфигурации кэша.
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

    println!("Архитектура");
    println!("  слоёв: {NUM_LAYERS}  ({NUM_LINEAR_LAYERS} DeltaNet + {NUM_FULL_LAYERS} full attention)");
    println!("  hidden {HIDDEN_SIZE}, intermediate {INTERMEDIATE_SIZE}, vocab {VOCAB_SIZE}");
    println!("  attention: {NUM_ATTN_HEADS}q / {NUM_KV_HEADS}kv, head_dim {ATTN_HEAD_DIM}, RoPE на {ROPE_DIM} из {ATTN_HEAD_DIM}");
    println!("  DeltaNet: {LA_NUM_K_HEADS}k / {LA_NUM_V_HEADS}v, dim {LA_K_HEAD_DIM}, conv {LA_CONV_KERNEL}");
    println!("  параметров: {:.2} B\n", TEXT_PARAMS as f64 / 1e9);

    let ours = WeightPlan::default();
    let shipped = WeightPlan::as_shipped();
    println!("Веса");
    println!("  чекпоинт как есть (грузит vLLM): {:>6.2} GB", gb(shipped.bytes()));
    println!("  наша загрузка (text-only, fp8 head/embed): {:>6.2} GB", gb(ours.bytes()));
    println!("  фора: {:>6.2} GB\n", gb(shipped.bytes() - ours.bytes()));

    let cfg = CacheConfig::default();
    println!("Кэш (kv {}, state {})", cfg.kv_dtype.name(), cfg.state_dtype.name());
    println!("  KV на токен (16 слоёв):        {:>8} B", cfg.kv_bytes_per_token());
    println!("  состояние на слот (48 слоёв):  {:>8.1} MB", cfg.state_bytes_per_slot() as f64 / 1e6);
    println!("  точка пересечения:             {:>8} токенов\n", cfg.crossover_tokens());

    let b = rtx5090(ours.bytes(), 2 * GB);
    println!("Бюджет");
    println!("  всего VRAM      {:>6.2} GB", gb(b.total_vram));
    println!("  зарезервировано {:>6.2} GB (десктоп + CUDA-контекст)", gb(b.reserved));
    println!("  веса            {:>6.2} GB", gb(b.weights));
    println!("  workspace       {:>6.2} GB", gb(b.workspace));
    println!("  под кэш         {:>6.2} GB\n", gb(b.cache_available()));

    println!("Максимальный concurrency по длине контекста");
    println!("  {:>8} | {:>10} | {:>10} | {:>12}", "контекст", "fp8 KV", "nvfp4 KV", "vLLM-like fp8");
    println!("  {:->8}-+-{:->10}-+-{:->10}-+-{:->12}", "", "", "", "");
    let fp4 = CacheConfig { kv_dtype: Dtype::Nvfp4, ..cfg };
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

    println!("\nЦелевая матрица бенчмарков");
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
            if b.fits(&cfg, conc, total) { "ok" } else { "OOM" }
        );
    }
    println!("\nПотолок decode (memory-bound: за шаг читаются все веса)");
    println!("  {:>6} | {:>7} | {:>12} | {:>12} | {:>12}", "batch", "ctx", "ITL идеал", "ITL @53%", "tok/s идеал");
    println!("  {:->6}-+-{:->7}-+-{:->12}-+-{:->12}-+-{:->12}", "", "", "", "", "");
    for (batch, ctx) in [(1usize, 2048usize), (1, 32768), (8, 2048), (16, 2048), (32, 2048), (32, 8192)] {
        let ideal = DecodeStep { batch, context_len: ctx, bandwidth_efficiency: 1.0 };
        let real = DecodeStep { bandwidth_efficiency: 0.53, ..ideal };
        println!(
            "  {:>6} | {:>7} | {:>9.2} ms | {:>9.2} ms | {:>9.0} t/s",
            batch, ctx,
            ideal.itl_ms(ours.bytes(), &cfg),
            real.itl_ms(ours.bytes(), &cfg),
            ideal.tokens_per_sec(ours.bytes(), &cfg),
        );
    }

    println!("\nИз чего состоит шаг decode (batch=32, ctx=2048)");
    let step = DecodeStep { batch: 32, context_len: 2048, bandwidth_efficiency: 1.0 };
    let c = step.cost(ours.bytes(), &cfg);
    for (name, bytes) in [("веса", c.weight_bytes), ("состояние DeltaNet", c.state_bytes), ("KV-кэш", c.kv_bytes)] {
        println!("  {:<20} {:>7.2} GB  {:>4.0}%", name, gb(bytes), 100.0 * bytes as f64 / c.total() as f64);
    }
    println!("  переход в compute-bound при batch ~{:.0}", compute_bound_batch(Dtype::Nvfp4));
}

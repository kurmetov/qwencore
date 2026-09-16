//! Внимание на префилле: три пути на одной форме.
//! `cargo run --release -p qwc-cuda --bin prefillattnbench [bf16|fp8]`
//!
//! Форма взята из движка: чанк строк запроса на своей позиции в промпте,
//! то есть каждая строка видит свой причинный префикс. Слои держат отдельные
//! кэши, иначе один буфер осел бы в L2 и замер бы поехал.
//!
//! Второй аргумент задаёт размер чанка (по умолчанию 511). Медленные пути
//! decode и тайловый считаются только на чанках до 512 строк: на больших
//! decode идёт десятки секунд и меряет не то.

use qwc_core::arch::{ATTN_HEAD_DIM, NUM_ATTN_HEADS, NUM_FULL_LAYERS, NUM_KV_HEADS};
use qwc_cuda::paged_attention::{self, KvCacheDtype, PAGE_SIZE, PagedAttentionWorkspace};
use qwc_cuda::{Device, DeviceBuffer, Event, Stream, bf16};


#[derive(Clone, Copy)]
enum Path {
    Decode,
    Tiled,
    Mma,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::init(0)?;
    let stream = Stream::new()?;
    let dtype = match std::env::args().nth(1).as_deref() {
        Some("bf16") => KvCacheDtype::Bf16,
        _ => KvCacheDtype::Fp8,
    };
    #[allow(non_snake_case)]
    let ROWS: usize = std::env::args()
        .nth(2)
        .map(|a| a.parse().expect("число строк чанка"))
        .unwrap_or(511);
    let slow_paths = ROWS <= 512;

    println!(
        "Внимание на префилле, RTX 5090 ({} SM), {} слоёв, KV {:?}, {ROWS} строк чанка\n",
        device.sm_count, NUM_FULL_LAYERS, dtype
    );
    println!(
        "  {:>8} | {:>12} | {:>12} | {:>12} | {:>8} | {:>11}",
        "позиция", "decode", "тайловое", "MMA", "выигрыш", "MMA/токен"
    );
    println!(
        "  {:->8}-+-{:->12}-+-{:->12}-+-{:->12}-+-{:->8}-+-{:->11}",
        "", "", "", "", "", ""
    );

    for start in [0usize, 1_024, 3_584, 7_680] {
        let max_context = start + ROWS;
        let max_blocks = max_context.div_ceil(PAGE_SIZE);
        let num_blocks = max_blocks;
        let cache_elements = num_blocks * NUM_KV_HEADS * PAGE_SIZE * ATTN_HEAD_DIM;
        let cache_bytes = cache_elements * dtype.bytes_per_element();

        let caches: Vec<(DeviceBuffer<u8>, DeviceBuffer<u8>)> = (0..NUM_FULL_LAYERS)
            .map(|layer| {
                let key: Vec<u8> = (0..cache_bytes)
                    .map(|i| ((i * 7 + layer * 13) % 200 + 16) as u8)
                    .collect();
                let value: Vec<u8> = (0..cache_bytes)
                    .map(|i| ((i * 5 + layer * 11) % 200 + 16) as u8)
                    .collect();
                (
                    DeviceBuffer::from_slice(&key).unwrap(),
                    DeviceBuffer::from_slice(&value).unwrap(),
                )
            })
            .collect();

        let contexts: Vec<u32> = (0..ROWS).map(|row| (start + row + 1) as u32).collect();
        let one_table: Vec<u32> = (0..max_blocks as u32).collect();
        let tables: Vec<u32> = (0..ROWS).flat_map(|_| one_table.clone()).collect();
        let query: Vec<u16> = (0..ROWS * NUM_ATTN_HEADS * ATTN_HEAD_DIM)
            .map(|i| bf16::from_f32(((i * 13 % 29) as f32 - 14.0) / 28.0))
            .collect();
        let gate = vec![0u16; query.len() * 2];

        let device_query = DeviceBuffer::from_slice(&query)?;
        let device_gate = DeviceBuffer::from_slice(&gate)?;
        let device_tables = DeviceBuffer::from_slice(&tables)?;
        let device_lengths = DeviceBuffer::from_slice(&contexts)?;
        let mut output = DeviceBuffer::<u16>::zeroed(query.len())?;
        // Workspace нужен только пути decode, а он считается лишь на малых
        // чанках: его разметка ограничена 1024 строками.
        let mut workspace = PagedAttentionWorkspace::new(ROWS.min(1024), max_context)?;

        let mut run = |path: Path| -> qwc_cuda::Result<()> {
            for (key, value) in &caches {
                match path {
                    Path::Decode => paged_attention::decode_gated(
                        &device_query, &device_gate, key, value, num_blocks,
                        &device_tables, &device_lengths, max_blocks, &mut output,
                        &mut workspace, ROWS, max_context, dtype, &stream,
                    )?,
                    Path::Tiled => paged_attention::prefill_gated(
                        &device_query, &device_gate, key, value, num_blocks,
                        &device_tables, &device_lengths, max_blocks, &mut output,
                        ROWS, 0, dtype, &stream,
                    )?,
                    Path::Mma => paged_attention::prefill_gated_mma(
                        &device_query, &device_gate, key, value, num_blocks,
                        &device_tables, &device_lengths, max_blocks, &mut output,
                        ROWS, 0, dtype, &stream,
                    )?,
                }
            }
            Ok(())
        };

        let mut time = |path: Path| -> Result<f64, Box<dyn std::error::Error>> {
            run(path)?;
            stream.synchronize()?;
            let (begin, end) = (Event::new()?, Event::new()?);
            let iterations = 5;
            begin.record(&stream)?;
            for _ in 0..iterations {
                run(path)?;
            }
            end.record(&stream)?;
            end.synchronize()?;
            Ok(Event::elapsed_ms(&begin, &end)? as f64 / iterations as f64)
        };

        let decode_ms = if slow_paths { time(Path::Decode)? } else { f64::NAN };
        let tiled_ms = if slow_paths { time(Path::Tiled)? } else { f64::NAN };
        let mma_ms = time(Path::Mma)?;
        let cell = |ms: f64| {
            if ms.is_nan() {
                format!("{:>12}", "-")
            } else {
                format!("{:>9.3} мс", ms)
            }
        };
        println!(
            "  {:>8} | {} | {} | {:>9.3} мс | {:>7} | {:>8.1} us",
            start,
            cell(decode_ms),
            cell(tiled_ms),
            mma_ms,
            if decode_ms.is_nan() {
                "-".to_string()
            } else {
                format!("{:.2}x", decode_ms / mma_ms)
            },
            mma_ms * 1e3 / ROWS as f64,
        );
    }
    Ok(())
}

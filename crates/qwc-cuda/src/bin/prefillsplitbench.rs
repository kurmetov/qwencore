//! Разбиение контекста в префилльном MMA-внимании: свип числа партиций.
//! `cargo run --release -p qwc-cuda --bin prefillsplitbench`
//!
//! Форма — сегмент одной последовательности за длинной историей: проверка
//! черновиков (4 строки), хвост промпта после кэша префиксов (сотни строк),
//! малый чанк. Слои держат отдельные кэши, иначе контекст осел бы в L2.
//! Строка «авто» — то, что выберет `PrefillAttentionWorkspace::new`.

use qwc_core::arch::{ATTN_HEAD_DIM, NUM_ATTN_HEADS, NUM_FULL_LAYERS, NUM_KV_HEADS};
use qwc_cuda::paged_attention::{self, KvCacheDtype, PAGE_SIZE, PrefillAttentionWorkspace};
use qwc_cuda::{Device, DeviceBuffer, Event, Stream, bf16};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::init(0)?;
    let stream = Stream::new()?;
    let dtype = KvCacheDtype::Fp8;
    let forced = [2usize, 4, 7, 14, 28, 56];
    println!(
        "MMA-префилл с разбиением контекста, RTX 5090 ({} SM), {} слоёв, KV {:?}; мкс на слой\n",
        device.sm_count, NUM_FULL_LAYERS, dtype
    );
    print!("  {:>5} | {:>6} | {:>8} | {:>12}", "строк", "старт", "целиком", "авто");
    for partitions in forced {
        print!(" | {:>7}", format!("P={partitions}"));
    }
    println!();

    for start in [8_192usize, 30_000, 60_000] {
        let rows_max = 512usize;
        let max_context = start + rows_max;
        let max_blocks = max_context.div_ceil(PAGE_SIZE);
        let cache_bytes =
            max_blocks * NUM_KV_HEADS * PAGE_SIZE * ATTN_HEAD_DIM * dtype.bytes_per_element();
        let caches: Vec<(DeviceBuffer<u8>, DeviceBuffer<u8>)> = (0..NUM_FULL_LAYERS)
            .map(|_| {
                (
                    DeviceBuffer::zeroed(cache_bytes).unwrap(),
                    DeviceBuffer::zeroed(cache_bytes).unwrap(),
                )
            })
            .collect();
        for rows in [1usize, 4, 8, 64, 128, 256, 512] {
            let contexts: Vec<u32> = (0..rows).map(|row| (start + row + 1) as u32).collect();
            let context = start + rows;
            let one_table: Vec<u32> = (0..max_blocks as u32).collect();
            let tables: Vec<u32> = (0..rows).flat_map(|_| one_table.clone()).collect();
            let query: Vec<u16> = (0..rows * NUM_ATTN_HEADS * ATTN_HEAD_DIM)
                .map(|i| bf16::from_f32(((i * 13 % 29) as f32 - 14.0) / 28.0))
                .collect();
            let gate = vec![0u16; query.len() * 2];
            let device_query = DeviceBuffer::from_slice(&query)?;
            let device_gate = DeviceBuffer::from_slice(&gate)?;
            let device_tables = DeviceBuffer::from_slice(&tables)?;
            let device_lengths = DeviceBuffer::from_slice(&contexts)?;
            let mut output = DeviceBuffer::<u16>::zeroed(query.len())?;

            let mut time = |workspace: Option<&mut PrefillAttentionWorkspace>|
             -> Result<f64, Box<dyn std::error::Error>> {
                let mut workspace = workspace;
                let mut run = |workspace: &mut Option<&mut PrefillAttentionWorkspace>|
                 -> qwc_cuda::Result<()> {
                    for (key, value) in &caches {
                        match workspace {
                            Some(workspace) => paged_attention::prefill_gated_mma_split(
                                &device_query, &device_gate, key, value, max_blocks,
                                &device_tables, &device_lengths, max_blocks, &mut output,
                                rows, 0, context, workspace, dtype, &stream,
                            )?,
                            None => paged_attention::prefill_gated_mma(
                                &device_query, &device_gate, key, value, max_blocks,
                                &device_tables, &device_lengths, max_blocks, &mut output,
                                rows, 0, dtype, &stream,
                            )?,
                        }
                    }
                    Ok(())
                };
                run(&mut workspace)?;
                stream.synchronize()?;
                let (begin, end) = (Event::new()?, Event::new()?);
                let iterations = 5;
                begin.record(&stream)?;
                for _ in 0..iterations {
                    run(&mut workspace)?;
                }
                end.record(&stream)?;
                end.synchronize()?;
                let per_layer = Event::elapsed_ms(&begin, &end)? as f64 * 1e3
                    / (iterations * NUM_FULL_LAYERS) as f64;
                Ok(per_layer)
            };

            let whole = time(None)?;
            let mut auto = PrefillAttentionWorkspace::new(rows, context)?;
            let (planned, _) = auto.plan_for(rows, context);
            let automatic = time(Some(&mut auto))?;
            print!(
                "  {:>5} | {:>6} | {:>8.1} | {:>12}",
                rows,
                start,
                whole,
                format!("{automatic:.1} (P={planned})")
            );
            for partitions in forced {
                let mut workspace =
                    PrefillAttentionWorkspace::with_partitions(rows, context, partitions)?;
                print!(" | {:>7.1}", time(Some(&mut workspace))?);
            }
            println!();
        }
    }
    Ok(())
}

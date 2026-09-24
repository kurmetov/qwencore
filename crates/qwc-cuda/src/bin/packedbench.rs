//! Упакованное внимание против прежних путей на формах `flashinfer_attention.py`.
//! `cargo run --release -p qwc-cuda --bin packedbench [-- --sweep]`
//!
//! Сегмент — `rows` строк одной последовательности за историей `start`:
//! проверка черновиков, хвост промпта, чанк. Прежний путь движка для него —
//! MMA-префилл без разбиения (`prefill_gated_mma`). Decode — `batch`
//! строк разных последовательностей с контекстом `context`, прежний путь —
//! `decode_gated`. Слои держат отдельные кэши, иначе контекст осел бы в L2.
//! `--sweep` добавляет свип числа партиций упакованного ядра.

use qwc_core::arch::{ATTN_HEAD_DIM, NUM_ATTN_HEADS, NUM_FULL_LAYERS, NUM_KV_HEADS};
use qwc_cuda::paged_attention::{
    self, KvCacheDtype, PAGE_SIZE, PackedAttentionWorkspace, PackedShape,
    PagedAttentionWorkspace,
};
use qwc_cuda::{Device, DeviceBuffer, Event, Stream, bf16};

type Caches = Vec<(DeviceBuffer<u8>, DeviceBuffer<u8>)>;

fn caches(num_blocks: usize) -> Caches {
    let bytes = num_blocks * NUM_KV_HEADS * PAGE_SIZE * ATTN_HEAD_DIM;
    (0..NUM_FULL_LAYERS)
        .map(|_| (DeviceBuffer::zeroed(bytes).unwrap(), DeviceBuffer::zeroed(bytes).unwrap()))
        .collect()
}

/// Мкс на слой: медиана семи замеров по пять проходов всех слоёв.
fn time(
    stream: &Stream,
    mut run: impl FnMut(&DeviceBuffer<u8>, &DeviceBuffer<u8>) -> qwc_cuda::Result<()>,
    caches: &Caches,
) -> f64 {
    let mut pass = || -> qwc_cuda::Result<()> {
        for (key, value) in caches {
            run(key, value)?;
        }
        Ok(())
    };
    pass().unwrap();
    stream.synchronize().unwrap();
    let iterations = 5;
    let mut samples = [0.0f64; 7];
    for sample in &mut samples {
        let (begin, end) = (Event::new().unwrap(), Event::new().unwrap());
        begin.record(stream).unwrap();
        for _ in 0..iterations {
            pass().unwrap();
        }
        end.record(stream).unwrap();
        end.synchronize().unwrap();
        *sample = Event::elapsed_ms(&begin, &end).unwrap() as f64 * 1e3
            / (iterations * NUM_FULL_LAYERS) as f64;
    }
    samples.sort_by(f64::total_cmp);
    samples[3]
}

struct Inputs {
    query: DeviceBuffer<u16>,
    gate: DeviceBuffer<u16>,
    tables: DeviceBuffer<u32>,
    lengths: DeviceBuffer<u32>,
    output: DeviceBuffer<u16>,
}

fn inputs(contexts: &[u32], tables: &[u32]) -> Inputs {
    let rows = contexts.len();
    let query: Vec<u16> = (0..rows * NUM_ATTN_HEADS * ATTN_HEAD_DIM)
        .map(|i| bf16::from_f32(((i * 13 % 29) as f32 - 14.0) / 28.0))
        .collect();
    Inputs {
        gate: DeviceBuffer::zeroed(query.len() * 2).unwrap(),
        output: DeviceBuffer::zeroed(query.len()).unwrap(),
        query: DeviceBuffer::from_slice(&query).unwrap(),
        tables: DeviceBuffer::from_slice(tables).unwrap(),
        lengths: DeviceBuffer::from_slice(contexts).unwrap(),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sweep = std::env::args().any(|arg| arg == "--sweep");
    let forced = [2usize, 4, 8, 16, 24, 32, 42, 64];
    let device = Device::init(0)?;
    let stream = Stream::new()?;
    println!(
        "Упакованное внимание, RTX 5090 ({} SM), {} слоёв, KV fp8; мкс на слой\n",
        device.sm_count, NUM_FULL_LAYERS
    );

    println!("Сегмент одной последовательности");
    print!("  {:>5} | {:>6} | {:>14} | {:>14}", "строк", "старт", "MMA", "упакованное");
    if sweep {
        for partitions in forced {
            print!(" | {:>7}", format!("P={partitions}"));
        }
    }
    println!();
    for start in [8_192usize, 30_000, 60_000] {
        let max_context = start + 2_048;
        let max_blocks = max_context.div_ceil(PAGE_SIZE);
        let caches = caches(max_blocks);
        for rows in [1usize, 4, 8, 64, 128, 256, 512, 2_048] {
            let contexts: Vec<u32> = (0..rows).map(|row| (start + row + 1) as u32).collect();
            let context = start + rows;
            let one_table: Vec<u32> = (0..max_blocks as u32).collect();
            let tables: Vec<u32> = (0..rows).flat_map(|_| one_table.clone()).collect();
            let mut data = inputs(&contexts, &tables);

            let old = time(
                &stream,
                |key, value| {
                    paged_attention::prefill_gated_mma(
                        &data.query, &data.gate, key, value, max_blocks, &data.tables,
                        &data.lengths, max_blocks, &mut data.output, rows, 0, KvCacheDtype::Fp8,
                        &stream,
                    )
                },
                &caches,
            );
            let mut run_packed = |workspace: &mut PackedAttentionWorkspace| {
                time(
                    &stream,
                    |key, value| {
                        paged_attention::gated_packed(
                            &data.query, &data.gate, key, value, max_blocks, &data.tables,
                            &data.lengths, max_blocks, &mut data.output, PackedShape::Segment,
                            rows, 0, context, workspace, &stream,
                        )
                    },
                    &caches,
                )
            };
            let mut packed = PackedAttentionWorkspace::new(&[PackedShape::Segment], rows, context)?;
            let plan = packed.plan_for(PackedShape::Segment, rows, context);
            let new = run_packed(&mut packed);
            print!(
                "  {:>5} | {:>6} | {:>14} | {:>14}",
                rows,
                start,
                format!("{old:.1}"),
                format!("{new:.1} (P={})", plan.partitions)
            );
            if sweep {
                for partitions in forced {
                    if partitions > context.div_ceil(PAGE_SIZE) {
                        print!(" | {:>7}", "-");
                        continue;
                    }
                    let mut workspace =
                        PackedAttentionWorkspace::with_partitions(rows, context, partitions)?;
                    print!(" | {:>7.1}", run_packed(&mut workspace));
                }
            }
            println!();
        }
    }

    println!("\nDecode: строки разных последовательностей");
    print!("  {:>5} | {:>7} | {:>14} | {:>14}", "batch", "контекст", "decode_gated", "упакованное");
    if sweep {
        for partitions in forced {
            print!(" | {:>7}", format!("P={partitions}"));
        }
    }
    println!();
    for (batch, context) in [
        (1usize, 8_192usize),
        (1, 32_768),
        (1, 60_000),
        (2, 32_768),
        (4, 8_192),
        (8, 8_192),
        (32, 2_048),
        (32, 8_192),
    ] {
        let blocks_per_sequence = context.div_ceil(PAGE_SIZE);
        let num_blocks = batch * blocks_per_sequence;
        let caches = caches(num_blocks);
        let contexts = vec![context as u32; batch];
        let tables: Vec<u32> = (0..batch * blocks_per_sequence).map(|block| block as u32).collect();
        let mut data = inputs(&contexts, &tables);

        let mut decode = PagedAttentionWorkspace::new(batch, context)?;
        let (kernel, decode_parts) = decode.plan_for(batch);
        let old = time(
            &stream,
            |key, value| {
                paged_attention::decode_gated(
                    &data.query, &data.gate, key, value, num_blocks, &data.tables,
                    &data.lengths, blocks_per_sequence, &mut data.output, &mut decode, batch,
                    context, KvCacheDtype::Fp8, &stream,
                )
            },
            &caches,
        );
        let mut run_packed = |workspace: &mut PackedAttentionWorkspace| {
            time(
                &stream,
                |key, value| {
                    paged_attention::gated_packed(
                        &data.query, &data.gate, key, value, num_blocks, &data.tables,
                        &data.lengths, blocks_per_sequence, &mut data.output, PackedShape::Rows,
                        batch, 0, context, workspace, &stream,
                    )
                },
                &caches,
            )
        };
        let mut packed = PackedAttentionWorkspace::new(&[PackedShape::Rows], batch, context)?;
        let plan = packed.plan_for(PackedShape::Rows, batch, context);
        let new = run_packed(&mut packed);
        let kernel = match kernel {
            paged_attention::DecodeKernel::QueryHead => "q",
            paged_attention::DecodeKernel::SharedKv => "kv",
        };
        print!(
            "  {:>5} | {:>7} | {:>14} | {:>14}",
            batch,
            context,
            format!("{old:.1} ({kernel}/{decode_parts})"),
            format!("{new:.1} (P={})", plan.partitions)
        );
        if sweep {
            for partitions in forced {
                if partitions > context.div_ceil(PAGE_SIZE) {
                    print!(" | {:>7}", "-");
                    continue;
                }
                let mut workspace =
                    PackedAttentionWorkspace::with_partitions(batch, context, partitions)?;
                print!(" | {:>7.1}", run_packed(&mut workspace));
            }
        }
        println!();
    }
    Ok(())
}

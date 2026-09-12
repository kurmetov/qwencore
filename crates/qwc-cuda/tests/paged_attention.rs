//! Correctness of direct and split-context FP8 paged attention.

use qwc_core::arch::{ATTN_HEAD_DIM, NUM_ATTN_HEADS, NUM_KV_HEADS};
use qwc_cuda::paged_attention::{
    self, DecodeKernel, KvCacheDtype, PAGE_SIZE, PagedAttentionWorkspace,
};
use qwc_cuda::{DeviceBuffer, Stream, bf16};

fn run_case(context: usize, expect_split: bool) {
    const BATCH: usize = 1;
    let max_blocks = context.div_ceil(PAGE_SIZE);
    let num_blocks = max_blocks;
    let cache_elements = num_blocks * NUM_KV_HEADS * PAGE_SIZE * ATTN_HEAD_DIM;
    let key_codes = [0x18u8, 0x98, 0x20, 0xa0, 0x28, 0xa8, 0x00];
    let value_codes = [0x30u8, 0xb0, 0x28, 0xa8, 0x20, 0xa0, 0x00];
    let key_cache: Vec<u8> = (0..cache_elements)
        .map(|index| key_codes[(index * 5 + index / ATTN_HEAD_DIM) % key_codes.len()])
        .collect();
    let value_cache: Vec<u8> = (0..cache_elements)
        .map(|index| value_codes[(index * 3 + index / 17) % value_codes.len()])
        .collect();
    // Reverse physical pages so the test cannot accidentally ignore the table.
    let block_tables: Vec<u32> = (0..max_blocks as u32).rev().collect();
    let context_lengths = vec![context as u32];
    let query: Vec<u16> = (0..NUM_ATTN_HEADS * ATTN_HEAD_DIM)
        .map(|index| {
            let value = ((index * 13 % 29) as f32 - 14.0) / 28.0;
            bf16::from_f32(value)
        })
        .collect();

    let mut expected = vec![0.0f32; query.len()];
    paged_attention::reference::decode_fp8(
        &query,
        &key_cache,
        &value_cache,
        &block_tables,
        &context_lengths,
        max_blocks,
        &mut expected,
        BATCH,
    );

    let stream = Stream::new().unwrap();
    let device_query = DeviceBuffer::from_slice(&query).unwrap();
    let device_key = DeviceBuffer::from_slice(&key_cache).unwrap();
    let device_value = DeviceBuffer::from_slice(&value_cache).unwrap();
    let device_tables = DeviceBuffer::from_slice(&block_tables).unwrap();
    let device_lengths = DeviceBuffer::from_slice(&context_lengths).unwrap();
    let mut query_gate = vec![0u16; BATCH * NUM_ATTN_HEADS * ATTN_HEAD_DIM * 2];
    for head in 0..NUM_ATTN_HEADS {
        let base = head * ATTN_HEAD_DIM * 2 + ATTN_HEAD_DIM;
        for dimension in 0..ATTN_HEAD_DIM {
            query_gate[base + dimension] = bf16::from_f32((dimension % 17) as f32 * 0.1 - 0.8);
        }
    }
    let device_query_gate = DeviceBuffer::from_slice(&query_gate).unwrap();
    for kernel in [DecodeKernel::QueryHead, DecodeKernel::SharedKv] {
        for gated in [false, true] {
            let mut output = DeviceBuffer::<u16>::zeroed(query.len()).unwrap();
            let mut workspace =
                PagedAttentionWorkspace::with_kernel(kernel, BATCH, context).unwrap();
            assert_eq!(workspace.partitions() > 1, expect_split);
            if gated {
                paged_attention::decode_fp8_gated(
                    &device_query,
                    &device_query_gate,
                    &device_key,
                    &device_value,
                    num_blocks,
                    &device_tables,
                    &device_lengths,
                    max_blocks,
                    &mut output,
                    &mut workspace,
                    BATCH,
                    context,
                    &stream,
                )
            } else {
                paged_attention::decode_fp8(
                    &device_query,
                    &device_key,
                    &device_value,
                    num_blocks,
                    &device_tables,
                    &device_lengths,
                    max_blocks,
                    &mut output,
                    &mut workspace,
                    BATCH,
                    context,
                    &stream,
                )
            }
            .unwrap();
            stream.synchronize().unwrap();

            for (index, (&actual, &expected)) in
                output.to_vec().unwrap().iter().zip(&expected).enumerate()
            {
                let actual = bf16::to_f32(actual);
                let expected = if gated {
                    let head = index / ATTN_HEAD_DIM;
                    let dimension = index % ATTN_HEAD_DIM;
                    let raw = bf16::to_f32(
                        query_gate[head * ATTN_HEAD_DIM * 2 + ATTN_HEAD_DIM + dimension],
                    );
                    expected / (1.0 + (-raw).exp())
                } else {
                    expected
                };
                assert!(
                    (actual - expected).abs() <= 0.004 + expected.abs() * 0.004,
                    "kernel={kernel:?}, gated={gated}, context={context}, index={index}: GPU={actual}, CPU={expected}"
                );
            }
        }
    }
}

#[test]
fn direct_partition_matches_cpu() {
    run_case(23, false);
}

#[test]
fn split_context_and_paging_match_cpu() {
    run_case(257, true);
}

#[test]
fn bf16_cache_matches_cpu_for_direct_and_split_contexts() {
    const BATCH: usize = 1;
    for context in [23usize, 257] {
        let max_blocks = context.div_ceil(PAGE_SIZE);
        let cache_elements = max_blocks * NUM_KV_HEADS * PAGE_SIZE * ATTN_HEAD_DIM;
        let key: Vec<u16> = (0..cache_elements)
            .map(|index| bf16::from_f32((index % 31) as f32 / 20.0 - 0.75))
            .collect();
        let value: Vec<u16> = (0..cache_elements)
            .map(|index| bf16::from_f32((index % 23) as f32 / 16.0 - 0.6))
            .collect();
        let key_bytes: Vec<u8> = key.iter().flat_map(|item| item.to_ne_bytes()).collect();
        let value_bytes: Vec<u8> = value.iter().flat_map(|item| item.to_ne_bytes()).collect();
        let tables: Vec<u32> = (0..max_blocks as u32).rev().collect();
        let lengths = [context as u32];
        let query: Vec<u16> = (0..NUM_ATTN_HEADS * ATTN_HEAD_DIM)
            .map(|index| bf16::from_f32((index % 29) as f32 / 24.0 - 0.6))
            .collect();
        let mut expected = vec![0.0f32; query.len()];
        paged_attention::reference::decode_bf16(
            &query,
            &key,
            &value,
            &tables,
            &lengths,
            max_blocks,
            &mut expected,
            BATCH,
        );

        let stream = Stream::new().unwrap();
        let device_query = DeviceBuffer::from_slice(&query).unwrap();
        let device_key = DeviceBuffer::from_slice(&key_bytes).unwrap();
        let device_value = DeviceBuffer::from_slice(&value_bytes).unwrap();
        let device_tables = DeviceBuffer::from_slice(&tables).unwrap();
        let device_lengths = DeviceBuffer::from_slice(&lengths).unwrap();
        let gate =
            DeviceBuffer::from_slice(&vec![0u16; NUM_ATTN_HEADS * ATTN_HEAD_DIM * 2]).unwrap();
        let mut output = DeviceBuffer::<u16>::zeroed(query.len()).unwrap();
        let mut workspace = PagedAttentionWorkspace::new(BATCH, context).unwrap();
        paged_attention::decode_gated(
            &device_query,
            &gate,
            &device_key,
            &device_value,
            max_blocks,
            &device_tables,
            &device_lengths,
            max_blocks,
            &mut output,
            &mut workspace,
            BATCH,
            context,
            KvCacheDtype::Bf16,
            &stream,
        )
        .unwrap();
        stream.synchronize().unwrap();
        for (actual, expected) in output.to_vec().unwrap().iter().zip(expected) {
            // Zero gate means sigmoid(0) = 0.5.
            let actual = bf16::to_f32(*actual);
            let expected = expected * 0.5;
            assert!((actual - expected).abs() <= 0.004 + expected.abs() * 0.004);
        }
    }
}

/// Тайл строк префилла должен совпасть с построчным decode: каждая строка
/// видит ровно свой причинный префикс, а общий проход по KV не должен
/// смешивать строки между собой.
#[test]
fn prefill_tile_matches_rowwise_decode() {
    // Строк больше, чем варпов в блоке, чтобы проверить и неполный хвостовой
    // тайл, и границу страницы внутри одного тайла.
    for rows in [19usize, 127, 200] {
        prefill_case(rows);
    }
}

fn prefill_case(rows: usize) {
    #[allow(non_snake_case)]
    let ROWS = rows;
    let first_context = 1usize;
    let contexts: Vec<u32> = (0..ROWS).map(|row| (first_context + row) as u32).collect();
    let max_context = *contexts.iter().max().unwrap() as usize;
    let max_blocks = max_context.div_ceil(PAGE_SIZE);
    let num_blocks = max_blocks;
    let cache_elements = num_blocks * NUM_KV_HEADS * PAGE_SIZE * ATTN_HEAD_DIM;

    let key_codes = [0x18u8, 0x98, 0x20, 0xa0, 0x28, 0xa8, 0x00];
    let value_codes = [0x30u8, 0xb0, 0x28, 0xa8, 0x20, 0xa0, 0x00];
    let key_cache: Vec<u8> = (0..cache_elements)
        .map(|index| key_codes[(index * 5 + index / ATTN_HEAD_DIM) % key_codes.len()])
        .collect();
    let value_cache: Vec<u8> = (0..cache_elements)
        .map(|index| value_codes[(index * 3 + index / 17) % value_codes.len()])
        .collect();

    // Одна последовательность: таблица блоков у всех строк одна и та же.
    // Порядок страниц перевёрнут, чтобы тест нельзя было пройти, игнорируя её.
    let one_table: Vec<u32> = (0..max_blocks as u32).rev().collect();
    let block_tables: Vec<u32> = (0..ROWS).flat_map(|_| one_table.clone()).collect();

    let query: Vec<u16> = (0..ROWS * NUM_ATTN_HEADS * ATTN_HEAD_DIM)
        .map(|index| bf16::from_f32(((index * 13 % 29) as f32 - 14.0) / 28.0))
        .collect();
    let mut query_gate = vec![0u16; ROWS * NUM_ATTN_HEADS * ATTN_HEAD_DIM * 2];
    for row in 0..ROWS {
        for head in 0..NUM_ATTN_HEADS {
            let base = (row * NUM_ATTN_HEADS + head) * ATTN_HEAD_DIM * 2 + ATTN_HEAD_DIM;
            for dimension in 0..ATTN_HEAD_DIM {
                query_gate[base + dimension] =
                    bf16::from_f32(((dimension + row) % 17) as f32 * 0.1 - 0.8);
            }
        }
    }

    let mut expected = vec![0.0f32; query.len()];
    paged_attention::reference::decode_fp8(
        &query,
        &key_cache,
        &value_cache,
        &block_tables,
        &contexts,
        max_blocks,
        &mut expected,
        ROWS,
    );

    let stream = Stream::new().unwrap();
    let device_query = DeviceBuffer::from_slice(&query).unwrap();
    let device_gate = DeviceBuffer::from_slice(&query_gate).unwrap();
    let device_key = DeviceBuffer::from_slice(&key_cache).unwrap();
    let device_value = DeviceBuffer::from_slice(&value_cache).unwrap();
    let device_tables = DeviceBuffer::from_slice(&block_tables).unwrap();
    let device_lengths = DeviceBuffer::from_slice(&contexts).unwrap();
    let mut output = DeviceBuffer::<u16>::zeroed(query.len()).unwrap();

    paged_attention::prefill_gated(
        &device_query,
        &device_gate,
        &device_key,
        &device_value,
        num_blocks,
        &device_tables,
        &device_lengths,
        max_blocks,
        &mut output,
        ROWS,
        0,
        KvCacheDtype::Fp8,
        &stream,
    )
    .unwrap();
    stream.synchronize().unwrap();

    for (index, (&actual, &reference)) in output.to_vec().unwrap().iter().zip(&expected).enumerate()
    {
        let actual = bf16::to_f32(actual);
        let raw = bf16::to_f32(query_gate[(index / ATTN_HEAD_DIM) * ATTN_HEAD_DIM * 2
            + ATTN_HEAD_DIM
            + index % ATTN_HEAD_DIM]);
        let expected = reference / (1.0 + (-raw).exp());
        assert!(
            (actual - expected).abs() <= 0.004 + expected.abs() * 0.004,
            "строка {}, индекс {index}: GPU={actual}, CPU={expected}",
            index / (NUM_ATTN_HEADS * ATTN_HEAD_DIM)
        );
    }
}

/// Тот же вход, но прежним путём: decode-ядро на строках префилла. Если оно
/// с эталоном не сходится, значит расхождение в eval — не регресс нового
/// ядра, а исправление старого.
#[test]
fn rowwise_decode_on_prefill_rows_matches_cpu() {
    const ROWS: usize = 19;
    let contexts: Vec<u32> = (0..ROWS).map(|row| (1 + row) as u32).collect();
    let max_context = *contexts.iter().max().unwrap() as usize;
    let max_blocks = max_context.div_ceil(PAGE_SIZE).max(1);
    let num_blocks = max_blocks;
    let cache_elements = num_blocks * NUM_KV_HEADS * PAGE_SIZE * ATTN_HEAD_DIM;

    let key_codes = [0x18u8, 0x98, 0x20, 0xa0, 0x28, 0xa8, 0x00];
    let value_codes = [0x30u8, 0xb0, 0x28, 0xa8, 0x20, 0xa0, 0x00];
    let key_cache: Vec<u8> = (0..cache_elements)
        .map(|index| key_codes[(index * 5 + index / ATTN_HEAD_DIM) % key_codes.len()])
        .collect();
    let value_cache: Vec<u8> = (0..cache_elements)
        .map(|index| value_codes[(index * 3 + index / 17) % value_codes.len()])
        .collect();
    let one_table: Vec<u32> = (0..max_blocks as u32).rev().collect();
    let block_tables: Vec<u32> = (0..ROWS).flat_map(|_| one_table.clone()).collect();
    let query: Vec<u16> = (0..ROWS * NUM_ATTN_HEADS * ATTN_HEAD_DIM)
        .map(|index| bf16::from_f32(((index * 13 % 29) as f32 - 14.0) / 28.0))
        .collect();

    let mut expected = vec![0.0f32; query.len()];
    paged_attention::reference::decode_fp8(
        &query,
        &key_cache,
        &value_cache,
        &block_tables,
        &contexts,
        max_blocks,
        &mut expected,
        ROWS,
    );

    let stream = Stream::new().unwrap();
    let device_query = DeviceBuffer::from_slice(&query).unwrap();
    let device_key = DeviceBuffer::from_slice(&key_cache).unwrap();
    let device_value = DeviceBuffer::from_slice(&value_cache).unwrap();
    let device_tables = DeviceBuffer::from_slice(&block_tables).unwrap();
    let device_lengths = DeviceBuffer::from_slice(&contexts).unwrap();
    let mut output = DeviceBuffer::<u16>::zeroed(query.len()).unwrap();
    let mut workspace = PagedAttentionWorkspace::new(ROWS, max_context).unwrap();

    paged_attention::decode_fp8(
        &device_query,
        &device_key,
        &device_value,
        num_blocks,
        &device_tables,
        &device_lengths,
        max_blocks,
        &mut output,
        &mut workspace,
        ROWS,
        max_context,
        &stream,
    )
    .unwrap();
    stream.synchronize().unwrap();

    let mut worst = 0.0f32;
    let mut worst_index = 0usize;
    for (index, (&actual, &reference)) in output.to_vec().unwrap().iter().zip(&expected).enumerate()
    {
        let difference = (bf16::to_f32(actual) - reference).abs();
        if difference > worst {
            worst = difference;
            worst_index = index;
        }
    }
    let reference = expected[worst_index];
    assert!(
        worst <= 0.004 + reference.abs() * 0.004,
        "строка {}, индекс {worst_index}: расхождение {worst}, эталон {reference}",
        worst_index / (NUM_ATTN_HEADS * ATTN_HEAD_DIM)
    );
}

/// Прямое сравнение путей между собой: эталон обоих устраивает с допуском,
/// но движок гоняет через 64 слоя, и важно, насколько они расходятся друг с
/// другом, а не насколько каждый близок к CPU.
#[test]
fn prefill_tile_and_rowwise_decode_agree_within_bf16() {
    for bf16_cache in [false, true] {
        agreement_case(bf16_cache);
    }
}

fn agreement_case(bf16_cache: bool) {
    const ROWS: usize = 127;
    let contexts: Vec<u32> = (0..ROWS).map(|row| (1 + row) as u32).collect();
    let max_context = *contexts.iter().max().unwrap() as usize;
    let max_blocks = max_context.div_ceil(PAGE_SIZE).max(1);
    let num_blocks = max_blocks;
    let cache_elements = num_blocks * NUM_KV_HEADS * PAGE_SIZE * ATTN_HEAD_DIM;

    let key_codes = [0x18u8, 0x98, 0x20, 0xa0, 0x28, 0xa8, 0x00];
    let value_codes = [0x30u8, 0xb0, 0x28, 0xa8, 0x20, 0xa0, 0x00];
    let bytes_per_element = if bf16_cache { 2 } else { 1 };
    let key_cache: Vec<u8> = (0..cache_elements * bytes_per_element)
        .map(|index| key_codes[(index * 5 + index / ATTN_HEAD_DIM) % key_codes.len()])
        .collect();
    let value_cache: Vec<u8> = (0..cache_elements * bytes_per_element)
        .map(|index| value_codes[(index * 3 + index / 17) % value_codes.len()])
        .collect();
    let one_table: Vec<u32> = (0..max_blocks as u32).rev().collect();
    let block_tables: Vec<u32> = (0..ROWS).flat_map(|_| one_table.clone()).collect();
    let query: Vec<u16> = (0..ROWS * NUM_ATTN_HEADS * ATTN_HEAD_DIM)
        .map(|index| bf16::from_f32(((index * 13 % 29) as f32 - 14.0) / 28.0))
        .collect();

    let stream = Stream::new().unwrap();
    let device_query = DeviceBuffer::from_slice(&query).unwrap();
    let device_key = DeviceBuffer::from_slice(&key_cache).unwrap();
    let device_value = DeviceBuffer::from_slice(&value_cache).unwrap();
    let device_tables = DeviceBuffer::from_slice(&block_tables).unwrap();
    let device_lengths = DeviceBuffer::from_slice(&contexts).unwrap();
    let gate = vec![0u16; ROWS * NUM_ATTN_HEADS * ATTN_HEAD_DIM * 2];
    let device_gate = DeviceBuffer::from_slice(&gate).unwrap();

    let dtype = if bf16_cache { KvCacheDtype::Bf16 } else { KvCacheDtype::Fp8 };
    let mut tiled = DeviceBuffer::<u16>::zeroed(query.len()).unwrap();
    paged_attention::prefill_gated(
        &device_query, &device_gate, &device_key, &device_value, num_blocks,
        &device_tables, &device_lengths, max_blocks, &mut tiled, ROWS, 0,
        dtype, &stream,
    )
    .unwrap();

    let mut rowwise = DeviceBuffer::<u16>::zeroed(query.len()).unwrap();
    // Движок создаёт workspace под сконфигурированный максимум контекста, а не
    // под фактический, поэтому на коротком промпте старый путь всё равно шёл
    // через партиционированное ядро. Воспроизводим именно это.
    let mut workspace = PagedAttentionWorkspace::new(ROWS, 4096).unwrap();
    paged_attention::decode_gated(
        &device_query, &device_gate, &device_key, &device_value, num_blocks,
        &device_tables, &device_lengths, max_blocks, &mut rowwise, &mut workspace,
        ROWS, max_context, dtype, &stream,
    )
    .unwrap();
    stream.synchronize().unwrap();

    let (a, b) = (tiled.to_vec().unwrap(), rowwise.to_vec().unwrap());
    let mut worst = 0.0f32;
    let mut worst_index = 0usize;
    for (index, (&x, &y)) in a.iter().zip(&b).enumerate() {
        let difference = (bf16::to_f32(x) - bf16::to_f32(y)).abs();
        if difference > worst {
            worst = difference;
            worst_index = index;
        }
    }
    println!(
        "bf16_cache={bf16_cache}: макс. расхождение путей {worst:.3e}, строка {}, канал {}",
        worst_index / (NUM_ATTN_HEADS * ATTN_HEAD_DIM),
        worst_index % ATTN_HEAD_DIM
    );
    assert!(worst < 0.02, "пути разошлись на {worst}");
}

/// Кто ближе к эталону на форме, где движок показал расхождение: мало строк,
/// короткий контекст, а workspace создан под сконфигурированный максимум,
/// поэтому старый путь дробит контекст на партиции и сливает их.
#[test]
fn prefill_tile_is_not_less_accurate_than_rowwise_decode() {
    for rows in [5usize, 14, 63] {
        let contexts: Vec<u32> = (0..rows).map(|row| (1 + row) as u32).collect();
        let max_context = *contexts.iter().max().unwrap() as usize;
        let max_blocks = max_context.div_ceil(PAGE_SIZE).max(1);
        let num_blocks = max_blocks;
        let cache_elements = num_blocks * NUM_KV_HEADS * PAGE_SIZE * ATTN_HEAD_DIM;
        let key_codes = [0x18u8, 0x98, 0x20, 0xa0, 0x28, 0xa8, 0x00];
        let value_codes = [0x30u8, 0xb0, 0x28, 0xa8, 0x20, 0xa0, 0x00];
        let key_cache: Vec<u8> = (0..cache_elements)
            .map(|index| key_codes[(index * 5 + index / ATTN_HEAD_DIM) % key_codes.len()])
            .collect();
        let value_cache: Vec<u8> = (0..cache_elements)
            .map(|index| value_codes[(index * 3 + index / 17) % value_codes.len()])
            .collect();
        let one_table: Vec<u32> = (0..max_blocks as u32).rev().collect();
        let block_tables: Vec<u32> = (0..rows).flat_map(|_| one_table.clone()).collect();
        let query: Vec<u16> = (0..rows * NUM_ATTN_HEADS * ATTN_HEAD_DIM)
            .map(|index| bf16::from_f32(((index * 13 % 29) as f32 - 14.0) / 28.0))
            .collect();

        let mut expected = vec![0.0f32; query.len()];
        paged_attention::reference::decode_fp8(
            &query, &key_cache, &value_cache, &block_tables, &contexts,
            max_blocks, &mut expected, rows,
        );

        let stream = Stream::new().unwrap();
        let device_query = DeviceBuffer::from_slice(&query).unwrap();
        let device_key = DeviceBuffer::from_slice(&key_cache).unwrap();
        let device_value = DeviceBuffer::from_slice(&value_cache).unwrap();
        let device_tables = DeviceBuffer::from_slice(&block_tables).unwrap();
        let device_lengths = DeviceBuffer::from_slice(&contexts).unwrap();
        let gate = vec![0u16; rows * NUM_ATTN_HEADS * ATTN_HEAD_DIM * 2];
        let device_gate = DeviceBuffer::from_slice(&gate).unwrap();

        let mut tiled = DeviceBuffer::<u16>::zeroed(query.len()).unwrap();
        paged_attention::prefill_gated(
            &device_query, &device_gate, &device_key, &device_value, num_blocks,
            &device_tables, &device_lengths, max_blocks, &mut tiled, rows, 0,
            KvCacheDtype::Fp8, &stream,
        )
        .unwrap();

        let mut rowwise = DeviceBuffer::<u16>::zeroed(query.len()).unwrap();
        let mut workspace = PagedAttentionWorkspace::new(rows, 4096).unwrap();
        paged_attention::decode_fp8_gated(
            &device_query, &device_gate, &device_key, &device_value, num_blocks,
            &device_tables, &device_lengths, max_blocks, &mut rowwise, &mut workspace,
            rows, max_context, &stream,
        )
        .unwrap();
        stream.synchronize().unwrap();

        let (a, b) = (tiled.to_vec().unwrap(), rowwise.to_vec().unwrap());
        let mut tile_error = 0.0f32;
        let mut row_error = 0.0f32;
        let mut between = 0.0f32;
        for index in 0..a.len() {
            let (x, y) = (bf16::to_f32(a[index]), bf16::to_f32(b[index]));
            tile_error = tile_error.max((x - expected[index]).abs());
            row_error = row_error.max((y - expected[index]).abs());
            between = between.max((x - y).abs());
        }
        println!(
            "строк {rows:3}: партиций {}, ошибка тайла {tile_error:.3e}, \
             ошибка построчного {row_error:.3e}, между собой {between:.3e}",
            workspace.partitions()
        );
        assert!(
            tile_error <= row_error * 1.5 + 1e-6,
            "тайловый путь хуже эталона: {tile_error} против {row_error}"
        );
    }
}

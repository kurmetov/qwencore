//! FP8-словарь: квантизация строк, выборка эмбеддингов и логиты.

use qwc_cuda::nvfp4::reference::e4m3;
use qwc_cuda::vocab::{Bf16Vocab, Fp8Vocab, MAX_LOGITS_BATCH, reference};
use qwc_cuda::{DeviceBuffer, Stream, bf16};

struct Rng(u64);

impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as u32 as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
}

/// Строки с разным масштабом: смысл шкалы на строку в том, что «тихая» строка
/// не теряет точность из-за «громкой» соседней.
fn table(rows: usize, cols: usize) -> Vec<u16> {
    let mut rng = Rng(0x5090_0806);
    (0..rows * cols)
        .map(|index| {
            let row_scale = match (index / cols) % 4 {
                0 => 1.0,
                1 => 0.01,
                2 => 40.0,
                _ => 0.0005,
            };
            bf16::from_f32(rng.next_f32() * row_scale)
        })
        .collect()
}

fn upload(rows: usize, cols: usize, host: &[u16], chunk_rows: usize) -> (Fp8Vocab, Stream) {
    let stream = Stream::new().expect("поток");
    let mut vocab = Fp8Vocab::zeroed(rows, cols).expect("матрица");
    let mut staging = DeviceBuffer::<u8>::zeroed(chunk_rows * cols * 2).expect("staging");
    let bytes: Vec<u8> = host.iter().flat_map(|value| value.to_le_bytes()).collect();
    let mut first = 0;
    while first < rows {
        let count = chunk_rows.min(rows - first);
        vocab
            .quantize_rows(
                first,
                &bytes[first * cols * 2..(first + count) * cols * 2],
                &mut staging,
                &stream,
            )
            .expect("квантизация");
        first += count;
    }
    stream.synchronize().expect("синхронизация");
    (vocab, stream)
}

#[test]
fn per_row_scale_keeps_e4m3_error_local() {
    const ROWS: usize = 133;
    const COLS: usize = 5120;
    let host = table(ROWS, COLS);
    let (vocab, _stream) = upload(ROWS, COLS, &host, 32);
    let (data, scales) = vocab.to_host().expect("копия на хост");

    for (row, &scale) in scales.iter().enumerate().take(ROWS) {
        let base = row * COLS;
        let source = &host[base..base + COLS];
        let absolute_max = source
            .iter()
            .map(|&value| bf16::to_f32(value).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(
            scale,
            absolute_max / 448.0,
            "строка {row}: шкала не равна absmax/448"
        );

        // E4M3 держит 3 бита мантиссы: относительная ошибка нормального
        // числа не больше 2^-4, а денормалов — абсолютная в 2^-9 шкалы.
        for column in 0..COLS {
            let expected = bf16::to_f32(source[column]);
            let actual = e4m3(data[base + column]) * scale;
            let bound = expected.abs() * 0.0625 + scale * 2f32.powi(-9);
            assert!(
                (actual - expected).abs() <= bound,
                "строка {row}, столбец {column}: {actual} против {expected}"
            );
        }
    }
}

#[test]
fn gather_matches_dequantized_table() {
    const ROWS: usize = 97;
    const COLS: usize = 5120;
    let host = table(ROWS, COLS);
    let (vocab, stream) = upload(ROWS, COLS, &host, 16);
    let (data, scales) = vocab.to_host().expect("копия на хост");

    let ids: Vec<u32> = vec![0, 96, 5, 5, 42, 13];
    let tokens = DeviceBuffer::from_slice(&ids).expect("идентификаторы");
    let mut output = DeviceBuffer::<u16>::zeroed(ids.len() * COLS).expect("выход");
    vocab
        .gather(&tokens, &mut output, &stream)
        .expect("выборка");
    stream.synchronize().expect("синхронизация");

    let mut expected = vec![0u16; ids.len() * COLS];
    reference::gather(&data, &scales, &ids, COLS, &mut expected);
    assert_eq!(output.to_vec().expect("копия"), expected);
}

#[test]
fn logits_match_dequantized_oracle_including_batch_split() {
    const ROWS: usize = 517;
    const COLS: usize = 1280;
    let host = table(ROWS, COLS);
    let (vocab, stream) = upload(ROWS, COLS, &host, 64);
    let (data, scales) = vocab.to_host().expect("копия на хост");

    // batch 12 обязательно режется: потолок одного запуска — 8.
    for batch in [1usize, MAX_LOGITS_BATCH, 12] {
        let mut rng = Rng(0x4242 + batch as u64);
        let hidden: Vec<u16> = (0..batch * COLS)
            .map(|_| bf16::from_f32(rng.next_f32()))
            .collect();
        let device_hidden = DeviceBuffer::from_slice(&hidden).expect("скрытое состояние");
        let mut device_logits = DeviceBuffer::<f32>::zeroed(batch * ROWS).expect("логиты");
        vocab
            .logits(&device_hidden, &mut device_logits, batch, &stream)
            .expect("логиты");
        stream.synchronize().expect("синхронизация");
        let actual = device_logits.to_vec().expect("копия");

        let mut expected = vec![0.0f32; batch * ROWS];
        reference::logits(&data, &scales, &hidden, batch, COLS, ROWS, &mut expected);

        let magnitude = expected.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        for (index, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() <= magnitude * 2e-3,
                "batch {batch}, элемент {index}: {got} против {want}"
            );
        }
    }
}

#[test]
fn bf16_lm_head_matches_cpu_oracle_including_batch_split() {
    const ROWS: usize = 97;
    const COLS: usize = 513;
    let host = table(ROWS, COLS);
    let bytes: Vec<u8> = host.iter().flat_map(|value| value.to_le_bytes()).collect();
    let mut head = Bf16Vocab::zeroed(ROWS, COLS).expect("bf16 lm_head");
    head.copy_rows(0, &bytes).expect("загрузка bf16 lm_head");
    assert_eq!(head.to_host().expect("копия"), bytes);
    let stream = Stream::new().expect("поток");

    for batch in [1usize, MAX_LOGITS_BATCH, 12] {
        let mut rng = Rng(0xb16f + batch as u64);
        let hidden: Vec<u16> = (0..batch * COLS)
            .map(|_| bf16::from_f32(rng.next_f32()))
            .collect();
        let device_hidden = DeviceBuffer::from_slice(&hidden).expect("скрытое состояние");
        let mut device_logits = DeviceBuffer::<f32>::zeroed(batch * ROWS).expect("логиты");
        head.logits(&device_hidden, &mut device_logits, batch, &stream)
            .expect("логиты");
        stream.synchronize().expect("синхронизация");
        let actual = device_logits.to_vec().expect("копия");

        for b in 0..batch {
            for row in 0..ROWS {
                let expected: f32 = (0..COLS)
                    .map(|column| {
                        bf16::to_f32(host[row * COLS + column])
                            * bf16::to_f32(hidden[b * COLS + column])
                    })
                    .sum();
                let got = actual[b * ROWS + row];
                let tolerance = expected.abs().max(1.0) * 2e-4;
                assert!(
                    (got - expected).abs() <= tolerance,
                    "batch {batch}, строка {row}: {got} против {expected}"
                );
            }
        }
    }
}

#[test]
fn bf16_embedding_gather_is_bit_exact() {
    const ROWS: usize = 41;
    const COLS: usize = 513;
    let host = table(ROWS, COLS);
    let bytes: Vec<u8> = host.iter().flat_map(|value| value.to_le_bytes()).collect();
    let mut table = Bf16Vocab::zeroed(ROWS, COLS).expect("bf16 embedding");
    table.copy_rows(0, &bytes).expect("загрузка");
    let stream = Stream::new().expect("поток");
    let ids = [40u32, 0, 17, 17];
    let device_ids = DeviceBuffer::from_slice(&ids).expect("токены");
    let mut output = DeviceBuffer::<u16>::zeroed(ids.len() * COLS).expect("выход");
    table
        .gather(&device_ids, &mut output, &stream)
        .expect("gather");
    stream.synchronize().expect("синхронизация");
    let actual = output.to_vec().expect("копия");
    for (index, &token) in ids.iter().enumerate() {
        assert_eq!(
            &actual[index * COLS..(index + 1) * COLS],
            &host[token as usize * COLS..(token as usize + 1) * COLS]
        );
    }
}

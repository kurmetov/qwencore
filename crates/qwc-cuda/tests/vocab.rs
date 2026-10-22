//! FP8 vocab: row quantization, embedding lookup and logits.

use qwc_cuda::nvfp4::reference::e4m3;
use qwc_cuda::vocab::{Bf16Vocab, Fp8Vocab, MAX_BATCHED_LOGITS, MAX_LOGITS_BATCH, reference};
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

/// Rows with different scales: the point of a per-row scale is that a "quiet" row
/// doesn't lose precision because of a "loud" neighbor.
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
    let stream = Stream::new().expect("stream");
    let mut vocab = Fp8Vocab::zeroed(rows, cols).expect("matrix");
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
            .expect("quantization");
        first += count;
    }
    stream.synchronize().expect("synchronization");
    (vocab, stream)
}

#[test]
fn per_row_scale_keeps_e4m3_error_local() {
    const ROWS: usize = 133;
    const COLS: usize = 5120;
    let host = table(ROWS, COLS);
    let (vocab, _stream) = upload(ROWS, COLS, &host, 32);
    let (data, scales) = vocab.to_host().expect("copy to host");

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
            "row {row}: scale is not absmax/448"
        );

        // E4M3 holds 3 mantissa bits: the relative error of a normal
        // number is at most 2^-4, and for denormals it's absolute at 2^-9 of the scale.
        for column in 0..COLS {
            let expected = bf16::to_f32(source[column]);
            let actual = e4m3(data[base + column]) * scale;
            let bound = expected.abs() * 0.0625 + scale * 2f32.powi(-9);
            assert!(
                (actual - expected).abs() <= bound,
                "row {row}, column {column}: {actual} vs {expected}"
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
    let (data, scales) = vocab.to_host().expect("copy to host");

    let ids: Vec<u32> = vec![0, 96, 5, 5, 42, 13];
    let tokens = DeviceBuffer::from_slice(&ids).expect("ids");
    let mut output = DeviceBuffer::<u16>::zeroed(ids.len() * COLS).expect("output");
    vocab
        .gather(&tokens, &mut output, &stream)
        .expect("lookup");
    stream.synchronize().expect("synchronization");

    let mut expected = vec![0u16; ids.len() * COLS];
    reference::gather(&data, &scales, &ids, COLS, &mut expected);
    assert_eq!(output.to_vec().expect("copy"), expected);
}

#[test]
fn logits_match_dequantized_oracle_including_batch_split() {
    const ROWS: usize = 517;
    const COLS: usize = 1280;
    let host = table(ROWS, COLS);
    let (vocab, stream) = upload(ROWS, COLS, &host, 64);
    let (data, scales) = vocab.to_host().expect("copy to host");

    // 1 and 8 go through the narrow kernel, beyond that the batched one: 12 and 33 land on
    // an incomplete tile, 96 on the max one, 100 is split into groups again.
    for batch in [1usize, MAX_LOGITS_BATCH, 12, 16, 33, 64, MAX_BATCHED_LOGITS, 100] {
        let mut rng = Rng(0x4242 + batch as u64);
        let hidden: Vec<u16> = (0..batch * COLS)
            .map(|_| bf16::from_f32(rng.next_f32()))
            .collect();
        let device_hidden = DeviceBuffer::from_slice(&hidden).expect("hidden state");
        let mut device_logits = DeviceBuffer::<f32>::zeroed(batch * ROWS).expect("logits");
        vocab
            .logits(&device_hidden, &mut device_logits, batch, &stream)
            .expect("logits");
        stream.synchronize().expect("synchronization");
        let actual = device_logits.to_vec().expect("copy");

        let mut expected = vec![0.0f32; batch * ROWS];
        reference::logits(&data, &scales, &hidden, batch, COLS, ROWS, &mut expected);

        let magnitude = expected.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        for (index, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() <= magnitude * 2e-3,
                "batch {batch}, element {index}: {got} vs {want}"
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
    head.copy_rows(0, &bytes).expect("loading bf16 lm_head");
    assert_eq!(head.to_host().expect("copy"), bytes);
    let stream = Stream::new().expect("stream");

    for batch in [1usize, MAX_LOGITS_BATCH, 12] {
        let mut rng = Rng(0xb16f + batch as u64);
        let hidden: Vec<u16> = (0..batch * COLS)
            .map(|_| bf16::from_f32(rng.next_f32()))
            .collect();
        let device_hidden = DeviceBuffer::from_slice(&hidden).expect("hidden state");
        let mut device_logits = DeviceBuffer::<f32>::zeroed(batch * ROWS).expect("logits");
        head.logits(&device_hidden, &mut device_logits, batch, &stream)
            .expect("logits");
        stream.synchronize().expect("synchronization");
        let actual = device_logits.to_vec().expect("copy");

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
                    "batch {batch}, row {row}: {got} vs {expected}"
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
    table.copy_rows(0, &bytes).expect("loading");
    let stream = Stream::new().expect("stream");
    let ids = [40u32, 0, 17, 17];
    let device_ids = DeviceBuffer::from_slice(&ids).expect("tokens");
    let mut output = DeviceBuffer::<u16>::zeroed(ids.len() * COLS).expect("output");
    table
        .gather(&device_ids, &mut output, &stream)
        .expect("gather");
    stream.synchronize().expect("synchronization");
    let actual = output.to_vec().expect("copy");
    for (index, &token) in ids.iter().enumerate() {
        assert_eq!(
            &actual[index * COLS..(index + 1) * COLS],
            &host[token as usize * COLS..(token as usize + 1) * COLS]
        );
    }
}

/// The shortlist must give the same logits as the full pass — otherwise the draft
/// lies not because the list is incomplete, but because of arithmetic.
///
/// The list is deliberately unsorted and takes rows from different tiles: the kernel
/// reads a row by index, and the order must not affect anything.
#[test]
fn subset_logits_match_full_pass_rows() {
    const ROWS: usize = 517;
    const COLS: usize = 1280;
    let host = table(ROWS, COLS);
    let (vocab, stream) = upload(ROWS, COLS, &host, 64);

    let mut rng = Rng(0x51_0715);
    let hidden: Vec<u16> = (0..COLS).map(|_| bf16::from_f32(rng.next_f32())).collect();
    let device_hidden = DeviceBuffer::from_slice(&hidden).expect("hidden state");

    let mut full = DeviceBuffer::<f32>::zeroed(ROWS).expect("full logits");
    vocab
        .logits(&device_hidden, &mut full, 1, &stream)
        .expect("full pass");

    let ids: Vec<u32> = [516u32, 0, 300, 1, 255, 256, 7, 129, 511, 64]
        .into_iter()
        .cycle()
        .take(70)
        .collect();
    let device_ids = DeviceBuffer::from_slice(&ids).expect("row list");
    let mut subset = DeviceBuffer::<f32>::zeroed(ids.len()).expect("shortlist logits");
    vocab
        .logits_subset(&device_hidden, &device_ids, &mut subset, ids.len(), &stream)
        .expect("shortlist");
    stream.synchronize().expect("synchronization");

    let full = full.to_vec().expect("copy");
    let subset = subset.to_vec().expect("copy");
    let magnitude = full.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
    for (slot, &row) in ids.iter().enumerate() {
        let want = full[row as usize];
        let got = subset[slot];
        assert!(
            (got - want).abs() <= magnitude * 2e-3,
            "row {row} in slot {slot}: {got} vs {want}"
        );
    }
}

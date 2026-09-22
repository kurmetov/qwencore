//! Свип геометрии W4A16 по всем decode-формам Qwen3.8 на batch 1..4.
//!
//! Прошлый свип (`nvfp4bench`) шёл по gate/up, down и q+gate, а `stepprofile`
//! показал, что сильнее всех проседают `k/v` и `la_out` — их там не было.
//! Здесь перебираются и раскладка (`rows_per_thread`), и геометрия блока.
//!
//! `cargo run --release -p qwc-cuda --bin nvfp4narrow`

use qwc_core::arch::{HIDDEN_SIZE, INTERMEDIATE_SIZE, KV_PROJ_DIM, LA_V_PROJ_DIM, Q_PROJ_DIM};
use qwc_core::roofline::ACHIEVABLE_BANDWIDTH;
use qwc_cuda::nvfp4::{Linear, QuantizedActivation, W4A4Workspace};
use qwc_cuda::{Device, DeviceBuffer, Event, Stream, bf16};

/// Working set обязан быть больше 96 MiB L2, иначе меряется кэш, а не DRAM.
const WORKING_SET: usize = 200_000_000;

/// (kRows, kKThreads, kKTile) — те, что пережили первый свип. Остальные
/// проигрывали на всех формах и только удлиняли прогон.
const GEOMETRIES: [(usize, usize, usize); 5] = [
    (4, 32, 512),
    (4, 32, 1024),
    (8, 16, 256),
    (8, 16, 512),
    (16, 8, 256),
];

/// Ширины, на которых сравниваются оба пути. Выше восьми W4A16 не
/// инстанцирован: `kMaxBatch` в ядре.
const BATCHES: [usize; 7] = [1, 2, 3, 4, 5, 6, 8];

/// Медиана пяти замеров по четыре прохода: короткий kernel иначе попадает в
/// период разгона частот.
fn time(
    stream: &Stream,
    call: &mut impl FnMut(&Stream) -> qwc_cuda::Result<()>,
) -> Result<f64, Box<dyn std::error::Error>> {
    for _ in 0..2 {
        call(stream)?;
    }
    stream.synchronize()?;
    let mut samples = [0.0f32; 5];
    for elapsed in &mut samples {
        let (start, end) = (Event::new()?, Event::new()?);
        start.record(stream)?;
        for _ in 0..4 {
            call(stream)?;
        }
        end.record(stream)?;
        end.synchronize()?;
        *elapsed = Event::elapsed_ms(&start, &end)?;
    }
    samples.sort_by(f32::total_cmp);
    Ok(samples[samples.len() / 2] as f64)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    Device::init(0)?;
    let stream = Stream::new()?;

    // Формы ровно те, что зовёт исполнитель на decode-шаге.
    let shapes = [
        ("la_in", 16480, HIDDEN_SIZE),
        ("down", HIDDEN_SIZE, INTERMEDIATE_SIZE),
        ("q+gate", 2 * Q_PROJ_DIM, HIDDEN_SIZE),
        ("k или v", KV_PROJ_DIM, HIDDEN_SIZE),
        ("la_out", HIDDEN_SIZE, LA_V_PROJ_DIM),
        ("attn_out", HIDDEN_SIZE, Q_PROJ_DIM),
    ];

    println!(
        "W4A16, свип геометрии по decode-формам; потолок {:.0} ГБ/с\n",
        ACHIEVABLE_BANDWIDTH / 1e9
    );

    for (name, out_features, in_features) in shapes {
        let one = Linear::zeroed(out_features, in_features)?;
        let copies = (WORKING_SET / one.weight_bytes()).max(2);
        let weight_bytes = one.weight_bytes();
        drop(one);
        let linears: Vec<Linear> = (0..copies)
            .map(|_| Linear::zeroed(out_features, in_features))
            .collect::<Result<_, _>>()?;

        println!("## {name}: [{out_features}, {in_features}], {copies} копий");
        println!(
            "  {:>5} | {:>9} | {:>22} | {:>9} | {:>9} | {:>6}",
            "batch", "W4A16", "лучшая геометрия", "W4A4", "W4A4 M=64", "итог"
        );

        for batch in BATCHES {
            // W4A4 в исполнителе округляет узкий batch до трёх строк, так что
            // буферы берутся по этой же мерке и оба пути делят их.
            let rows = batch.max(3);
            let input = DeviceBuffer::from_slice(&vec![bf16::from_f32(0.5); rows * in_features])?;
            let mut output = DeviceBuffer::<u16>::zeroed(rows * out_features)?;

            let mut best = (0.0f64, String::new());
            for (k_rows, k_threads, k_tile) in GEOMETRIES {
                if !in_features.is_multiple_of(k_tile) {
                    continue;
                }
                for rows_per_thread in 1..=2 {
                    let mut call = |stream: &Stream| -> qwc_cuda::Result<()> {
                        for linear in &linears {
                            linear.forward_w4a16_tuned(
                                &input,
                                &mut output,
                                batch,
                                k_rows,
                                k_threads,
                                k_tile,
                                rows_per_thread,
                                stream,
                            )?;
                        }
                        Ok(())
                    };
                    let ms = time(&stream, &mut call)?;
                    let gbps = weight_bytes as f64 * (4 * copies) as f64 / (ms * 1e-3) / 1e9;
                    if gbps > best.0 {
                        best = (
                            gbps,
                            format!("{k_rows}x{k_threads}x{k_tile} / {rows_per_thread}"),
                        );
                    }
                }
            }

            // W4A4: активация квантуется один раз на слой, как в исполнителе.
            let mut quantized = QuantizedActivation::zeroed(128.0, rows, in_features)?;
            let mut workspace = W4A4Workspace::new(rows, out_features, in_features)?;
            quantized.quantize_bf16(&input, &stream)?;
            let mut call = |stream: &Stream| -> qwc_cuda::Result<()> {
                for linear in &linears {
                    linear.forward_w4a4_quantized(
                        &quantized,
                        &mut output,
                        &mut workspace,
                        stream,
                    )?;
                }
                Ok(())
            };
            let w4a4_ms = time(&stream, &mut call)?;
            let w4a4 = weight_bytes as f64 * (4 * copies) as f64 / (w4a4_ms * 1e-3) / 1e9;

            // W4A4 с M, добитым до 64: тайл M у CUTLASS всё равно 128, но
            // время почему-то падает с ростом M до 64 при том же трафике.
            let padded_rows = rows.max(64);
            let padded_input =
                DeviceBuffer::from_slice(&vec![bf16::from_f32(0.5); padded_rows * in_features])?;
            let mut padded_output = DeviceBuffer::<u16>::zeroed(padded_rows * out_features)?;
            let mut padded = QuantizedActivation::zeroed(128.0, padded_rows, in_features)?;
            let mut padded_workspace = W4A4Workspace::new(padded_rows, out_features, in_features)?;
            padded.quantize_bf16_rows(&padded_input, 128.0, rows, &stream)?;
            let mut call = |stream: &Stream| -> qwc_cuda::Result<()> {
                for linear in &linears {
                    linear.forward_w4a4_quantized(
                        &padded,
                        &mut padded_output,
                        &mut padded_workspace,
                        stream,
                    )?;
                }
                Ok(())
            };
            let padded_ms = time(&stream, &mut call)?;
            let padded_gbps = weight_bytes as f64 * (4 * copies) as f64 / (padded_ms * 1e-3) / 1e9;

            println!(
                "  {:>5} | {:>6.0} ГБ/с | {:>22} | {:>6.0} ГБ/с | {:>6.0} ГБ/с | {:>6}",
                batch,
                best.0,
                best.1,
                w4a4,
                padded_gbps,
                if best.0 >= w4a4.max(padded_gbps) {
                    "W4A16"
                } else if padded_gbps > w4a4 {
                    "W4A4/64"
                } else {
                    "W4A4"
                }
            );
        }
        println!();
    }
    Ok(())
}

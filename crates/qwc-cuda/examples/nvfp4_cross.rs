//! Cross-check QWC's NVFP4 activation quantizer and W4A4 GEMM against the
//! reference engine on identical inputs.
//!
//! The reference (vLLM) runs `CutlassNvFp4LinearKernel` for every linear layer,
//! so this isolates the same arithmetic outside the model, where no recurrent
//! trajectory can hide or amplify a difference.
//!
//! ```
//! cargo run --release -p qwc-cuda --example nvfp4_cross -- MODEL ACTS OUT_PREFIX
//! ```
//!
//! `ACTS` is raw little-endian BF16 of shape `[rows, HIDDEN_SIZE]`. Written
//! next to `OUT_PREFIX`: `.codes` (packed E2M1, two per byte), `.scales`
//! (E4M3 block scales in logical row-major order), `.w4a4` and `.w4a16`
//! (FP32 projection outputs).

use qwc_core::arch::{HIDDEN_SIZE, INTERMEDIATE_SIZE};
use qwc_cuda::nvfp4::{self, Linear, QuantizedActivation, W4A4Workspace};
use qwc_cuda::{DeviceBuffer, Stream, bf16};
use qwc_model::Checkpoint;
use std::path::{Path, PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let model = args
        .next()
        .map(PathBuf::from)
        .ok_or("usage: MODEL ACTS OUT")?;
    let acts_path = args
        .next()
        .map(PathBuf::from)
        .ok_or("usage: MODEL ACTS OUT")?;
    let out_prefix = args
        .next()
        .map(PathBuf::from)
        .ok_or("usage: MODEL ACTS OUT")?;
    let tensor_prefix = args
        .next()
        .map(|value| value.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model.language_model.layers.0.mlp.gate_proj".to_string());
    let out_features: usize = match args.next() {
        Some(value) => value.to_string_lossy().parse()?,
        None => INTERMEDIATE_SIZE,
    };

    let raw = std::fs::read(&acts_path)?;
    if raw.len() % (2 * HIDDEN_SIZE) != 0 {
        return Err(format!("{}: not a whole number of rows", acts_path.display()).into());
    }
    let rows = raw.len() / (2 * HIDDEN_SIZE);
    let host: Vec<u16> = raw
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u16::from_le_bytes(*pair))
        .collect();

    let checkpoint = Checkpoint::open(&model)?;
    let weight = checkpoint
        .quant_linear(&tensor_prefix, out_features, HIDDEN_SIZE)
        .map_err(|error| format!("{tensor_prefix}: {error}"))?;
    println!("rows {rows}");
    println!("out_features {out_features}");
    println!("tensor {tensor_prefix}");
    println!("input_global_scale {:.9e}", weight.input_global_scale);
    println!("weight_global_scale {:.9e}", weight.weight_global_scale);

    let stream = Stream::new()?;
    let linear = Linear::from_host(
        weight.packed,
        weight.block_scales,
        weight.weight_global_scale,
        out_features,
        HIDDEN_SIZE,
    )?;
    let input = DeviceBuffer::from_slice(&host)?;

    let mut quantized = QuantizedActivation::zeroed(weight.input_global_scale, rows, HIDDEN_SIZE)?;
    quantized.quantize_bf16(&input, &stream)?;
    stream.synchronize()?;
    let (codes, scales) = quantized.to_host_logical()?;
    std::fs::write(with_suffix(&out_prefix, "codes"), &codes)?;
    std::fs::write(with_suffix(&out_prefix, "scales"), &scales)?;

    let mut workspace = W4A4Workspace::new(rows, out_features, HIDDEN_SIZE)?;
    let mut w4a4 = DeviceBuffer::<u16>::zeroed(rows * out_features)?;
    linear.forward_w4a4_quantized(&quantized, &mut w4a4, &mut workspace, &stream)?;
    stream.synchronize()?;
    write_bf16_as_f32(&with_suffix(&out_prefix, "w4a4"), &w4a4.to_vec()?)?;

    // W4A16 is the decode kernel and only covers small batches; compare the
    // rows it can take so both QWC paths are measured on the same inputs.
    let w4a16_rows = rows.min(nvfp4::MAX_W4A16_BATCH);
    let mut w4a16 = DeviceBuffer::<u16>::zeroed(rows * out_features)?;
    linear.forward_w4a16(&input, &mut w4a16, w4a16_rows, &stream)?;
    stream.synchronize()?;
    write_bf16_as_f32(&with_suffix(&out_prefix, "w4a16"), &w4a16.to_vec()?)?;
    println!("w4a16_rows {w4a16_rows}");

    println!(
        "written {}.{{codes,scales,w4a4,w4a16}}",
        out_prefix.display()
    );
    Ok(())
}

fn with_suffix(prefix: &Path, suffix: &str) -> PathBuf {
    let mut name = prefix.as_os_str().to_owned();
    name.push(".");
    name.push(suffix);
    PathBuf::from(name)
}

fn write_bf16_as_f32(path: &Path, values: &[u16]) -> std::io::Result<()> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for &value in values {
        bytes.extend_from_slice(&bf16::to_f32(value).to_le_bytes());
    }
    std::fs::write(path, bytes)
}

//! A mismatch with the architecture is a load error, not a reason to adapt.
//! Shape checks happen before the first allocation, so the test doesn't touch the GPU.

use qwc_engine::{LoadError, ModelWeights};
use qwc_model::Checkpoint;
use std::path::{Path, PathBuf};

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("qwc-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temporary directory");
        Self(dir)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A checkpoint of the listed tensors: (name, dtype, shape). The data is zero —
/// the loader in these tests never gets to reading it.
fn write_checkpoint(dir: &Path, tensors: &[(&str, &str, &[usize])]) {
    let mut entries = Vec::new();
    let mut data = Vec::new();
    for (name, dtype, shape) in tensors {
        let size: usize = shape.iter().product::<usize>() * if *dtype == "F32" { 4 } else { 1 };
        let (start, end) = (data.len(), data.len() + size);
        data.resize(end, 0u8);
        entries.push(format!(
            r#""{name}":{{"dtype":"{dtype}","shape":{shape:?},"data_offsets":[{start},{end}]}}"#
        ));
    }
    let header = format!("{{{}}}", entries.join(","));
    let mut shard = Vec::with_capacity(8 + header.len() + data.len());
    shard.extend_from_slice(&(header.len() as u64).to_le_bytes());
    shard.extend_from_slice(header.as_bytes());
    shard.extend_from_slice(&data);
    std::fs::write(dir.join("model-00001.safetensors"), &shard).expect("shard");

    let map: Vec<String> = tensors
        .iter()
        .map(|(name, _, _)| format!(r#""{name}":"model-00001.safetensors""#))
        .collect();
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        format!(r#"{{"weight_map":{{{}}}}}"#, map.join(",")),
    )
    .expect("index");
}

fn load_error(tensors: &[(&str, &str, &[usize])], tag: &str) -> String {
    let dir = TempDir::new(tag);
    write_checkpoint(dir.path(), tensors);
    let checkpoint = Checkpoint::open(dir.path()).expect("checkpoint opens");
    match ModelWeights::load(&checkpoint) {
        Err(LoadError::Checkpoint(message)) => message,
        Err(other) => panic!("expected a checkpoint error, got: {other}"),
        Ok(_) => panic!("an empty checkpoint must not load"),
    }
}

#[test]
fn missing_projection_is_named_in_the_error() {
    // An index without a single layer weight: gate_proj of layer 0 is asked first.
    let message = load_error(
        &[("model.language_model.norm.weight", "BF16", &[5120])],
        "loader-missing",
    );
    assert!(
        message.contains("model.language_model.layers.0.mlp.gate_proj.weight_packed"),
        "the error doesn't name the missing tensor: {message}"
    );
}

#[test]
fn wrong_shape_is_rejected_before_any_upload() {
    // The tensor exists, but the shape is wrong: 17408 x 5120/2 vs the substituted 16 x 8.
    let message = load_error(
        &[(
            "model.language_model.layers.0.mlp.gate_proj.weight_packed",
            "U8",
            &[16, 8],
        )],
        "loader-shape",
    );
    assert!(
        message.contains("expected [17408, 2560]"),
        "the error doesn't show the expected shape: {message}"
    );
}

//! Расхождение с архитектурой — ошибка загрузки, а не повод подстроиться.
//! Проверки формы идут до первой аллокации, поэтому тест не трогает GPU.

use qwc_engine::{LoadError, ModelWeights};
use qwc_model::Checkpoint;
use std::path::{Path, PathBuf};

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("qwc-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("временный каталог");
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

/// Чекпоинт из перечисленных тензоров: (имя, dtype, форма). Данные нулевые —
/// до их чтения загрузчик в этих тестах не доходит.
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
    std::fs::write(dir.join("model-00001.safetensors"), &shard).expect("шард");

    let map: Vec<String> = tensors
        .iter()
        .map(|(name, _, _)| format!(r#""{name}":"model-00001.safetensors""#))
        .collect();
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        format!(r#"{{"weight_map":{{{}}}}}"#, map.join(",")),
    )
    .expect("индекс");
}

fn load_error(tensors: &[(&str, &str, &[usize])], tag: &str) -> String {
    let dir = TempDir::new(tag);
    write_checkpoint(dir.path(), tensors);
    let checkpoint = Checkpoint::open(dir.path()).expect("чекпоинт открывается");
    match ModelWeights::load(&checkpoint) {
        Err(LoadError::Checkpoint(message)) => message,
        Err(other) => panic!("ожидалась ошибка чекпоинта, получено: {other}"),
        Ok(_) => panic!("пустой чекпоинт не должен грузиться"),
    }
}

#[test]
fn missing_projection_is_named_in_the_error() {
    // Индекс без единого веса слоя: первым спрашивается gate_proj слоя 0.
    let message = load_error(
        &[("model.language_model.norm.weight", "BF16", &[5120])],
        "loader-missing",
    );
    assert!(
        message.contains("model.language_model.layers.0.mlp.gate_proj.weight_packed"),
        "ошибка не называет отсутствующий тензор: {message}"
    );
}

#[test]
fn wrong_shape_is_rejected_before_any_upload() {
    // Тензор есть, но форма не та: 17408 x 5120/2 против подсунутых 16 x 8.
    let message = load_error(
        &[(
            "model.language_model.layers.0.mlp.gate_proj.weight_packed",
            "U8",
            &[16, 8],
        )],
        "loader-shape",
    );
    assert!(
        message.contains("ожидалась [17408, 2560]"),
        "ошибка не показывает ожидаемую форму: {message}"
    );
}

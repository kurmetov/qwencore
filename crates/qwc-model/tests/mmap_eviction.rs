//! Вытеснение страниц чекпоинта: RSS падает, отображение остаётся живым.

use qwc_model::Checkpoint;
use std::path::{Path, PathBuf};

/// Полезная нагрузка заметно больше страницы, иначе падение RSS утонет в шуме.
const PAYLOAD: usize = 8 << 20;

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

/// Резидентные байты процесса по /proc/self/statm (второе поле — страницы).
fn resident_bytes() -> usize {
    let statm = std::fs::read_to_string("/proc/self/statm").expect("/proc/self/statm");
    let pages: usize = statm
        .split_whitespace()
        .nth(1)
        .and_then(|p| p.parse().ok())
        .expect("поле resident");
    pages * 4096
}

/// Однотензорный шард: [8 байт длины заголовка][JSON][данные].
fn write_checkpoint(dir: &Path, payload: &[u8]) {
    let header = format!(
        r#"{{"weights":{{"dtype":"U8","shape":[{}],"data_offsets":[0,{}]}}}}"#,
        payload.len(),
        payload.len()
    );
    let mut shard = Vec::with_capacity(8 + header.len() + payload.len());
    shard.extend_from_slice(&(header.len() as u64).to_le_bytes());
    shard.extend_from_slice(header.as_bytes());
    shard.extend_from_slice(payload);
    std::fs::write(dir.join("model-00001.safetensors"), &shard).expect("шард");
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        r#"{"weight_map":{"weights":"model-00001.safetensors"}}"#,
    )
    .expect("индекс");
}

#[test]
fn eviction_drops_resident_pages_and_keeps_tensors_readable() {
    let payload: Vec<u8> = (0..PAYLOAD).map(|i| (i % 251) as u8).collect();
    let dir = TempDir::new("mmap-eviction");
    write_checkpoint(dir.path(), &payload);

    let checkpoint = Checkpoint::open(dir.path()).expect("чекпоинт открывается");
    let before_touch = resident_bytes();

    // Аналог фазы загрузки: страницы прочитаны и попали в working set.
    assert_eq!(checkpoint.bytes("weights").expect("тензор"), &payload[..]);
    let touched = resident_bytes();
    assert!(
        touched >= before_touch + PAYLOAD / 2,
        "чтение тензора не подняло RSS: {before_touch} -> {touched}"
    );

    checkpoint.evict_file_pages().expect("вытеснение страниц");
    let evicted = resident_bytes();
    assert!(
        evicted <= touched - PAYLOAD / 2,
        "RSS не упал после вытеснения: {touched} -> {evicted}"
    );

    // Отображение остаётся валидным: страницы подгружаются из файла заново.
    assert_eq!(checkpoint.bytes("weights").expect("тензор"), &payload[..]);
}

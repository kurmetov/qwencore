//! Checkpoint page eviction: RSS drops, the mapping stays alive.

use qwc_model::Checkpoint;
use std::path::{Path, PathBuf};

/// The payload is well above a page, otherwise the RSS drop drowns in noise.
const PAYLOAD: usize = 8 << 20;

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("qwc-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");
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

/// Resident bytes of the process from /proc/self/statm (the second field is pages).
fn resident_bytes() -> usize {
    let statm = std::fs::read_to_string("/proc/self/statm").expect("/proc/self/statm");
    let pages: usize = statm
        .split_whitespace()
        .nth(1)
        .and_then(|p| p.parse().ok())
        .expect("resident field");
    pages * 4096
}

/// A single-tensor shard: [8-byte header length][JSON][data].
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
    std::fs::write(dir.join("model-00001.safetensors"), &shard).expect("shard");
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        r#"{"weight_map":{"weights":"model-00001.safetensors"}}"#,
    )
    .expect("index");
}

#[test]
fn eviction_drops_resident_pages_and_keeps_tensors_readable() {
    let payload: Vec<u8> = (0..PAYLOAD).map(|i| (i % 251) as u8).collect();
    let dir = TempDir::new("mmap-eviction");
    write_checkpoint(dir.path(), &payload);

    let checkpoint = Checkpoint::open(dir.path()).expect("checkpoint opens");
    let before_touch = resident_bytes();

    // Like the load phase: pages are read and enter the working set.
    assert_eq!(checkpoint.bytes("weights").expect("tensor"), &payload[..]);
    let touched = resident_bytes();
    assert!(
        touched >= before_touch + PAYLOAD / 2,
        "reading the tensor didn't raise RSS: {before_touch} -> {touched}"
    );

    checkpoint.evict_file_pages().expect("page eviction");
    let evicted = resident_bytes();
    assert!(
        evicted <= touched - PAYLOAD / 2,
        "RSS didn't drop after eviction: {touched} -> {evicted}"
    );

    // The mapping stays valid: pages are reloaded from the file.
    assert_eq!(checkpoint.bytes("weights").expect("tensor"), &payload[..]);
}

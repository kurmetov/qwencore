//! Reading safetensors via mmap.
//!
//! Format: [8-byte LE header length][JSON header][data].
//! Offsets in the header are relative to the start of the data block.

use memmap2::{Advice, Mmap, UncheckedAdvice};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StDtype {
    Bf16,
    F32,
    F8E4m3,
    U8,
}

impl StDtype {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "BF16" => Self::Bf16,
            "F32" => Self::F32,
            "F8_E4M3" => Self::F8E4m3,
            "U8" => Self::U8,
            _ => return None,
        })
    }

    pub const fn size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::Bf16 => 2,
            Self::F8E4m3 | Self::U8 => 1,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::F32 => "f32",
            Self::F8E4m3 => "f8_e4m3",
            Self::U8 => "u8",
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dtype: StDtype,
    pub shape: Vec<usize>,
    /// Offsets within the shard's data block.
    pub start: usize,
    pub end: usize,
}

impl TensorInfo {
    pub fn elems(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn bytes(&self) -> usize {
        self.end - self.start
    }
}

pub struct Shard {
    mmap: Mmap,
    data_offset: usize,
    tensors: HashMap<String, TensorInfo>,
}

impl Shard {
    pub fn open(path: &Path) -> std::io::Result<Self> {
        let file = File::open(path)?;
        // SAFETY: the checkpoint file doesn't change while the engine runs.
        let mmap = unsafe { Mmap::map(&file)? };
        mmap.advise(Advice::Sequential)?;

        if mmap.len() < 8 {
            return Err(bad("shard shorter than header"));
        }
        let hdr_len = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
        let data_offset = 8 + hdr_len;
        if data_offset > mmap.len() {
            return Err(bad("header length goes past the end of the file"));
        }

        let json: serde_json::Value = serde_json::from_slice(&mmap[8..data_offset])
            .map_err(|e| bad(&format!("header doesn't parse: {e}")))?;

        let mut tensors = HashMap::new();
        for (name, v) in json.as_object().ok_or_else(|| bad("header is not an object"))?
        {
            if name == "__metadata__" {
                continue;
            }
            let dtype_str = v["dtype"].as_str().ok_or_else(|| bad("no dtype"))?;
            let Some(dtype) = StDtype::parse(dtype_str) else {
                return Err(bad(&format!("unsupported type {dtype_str} for {name}")));
            };
            let shape: Vec<usize> = v["shape"]
                .as_array()
                .ok_or_else(|| bad("no shape"))?
                .iter()
                .map(|x| x.as_u64().unwrap_or(0) as usize)
                .collect();
            let off = v["data_offsets"]
                .as_array()
                .ok_or_else(|| bad("no data_offsets"))?;
            let start = off[0].as_u64().unwrap_or(0) as usize;
            let end = off[1].as_u64().unwrap_or(0) as usize;
            tensors.insert(
                name.clone(),
                TensorInfo {
                    dtype,
                    shape,
                    start,
                    end,
                },
            );
        }

        Ok(Self {
            mmap,
            data_offset,
            tensors,
        })
    }

    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// Raw tensor bytes. No copy — it's a slice of the mmap.
    pub fn bytes(&self, name: &str) -> Option<&[u8]> {
        let t = self.tensors.get(name)?;
        Some(&self.mmap[self.data_offset + t.start..self.data_offset + t.end])
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }

    /// Drops clean file-backed pages after their tensors have been uploaded.
    /// The virtual mapping stays valid and pages can be faulted in again.
    pub fn evict_pages(&self) -> std::io::Result<()> {
        // SAFETY: the mapping is read-only and file-backed, no dirty pages —
        // MADV_DONTNEED here only drops clean pages; the data
        // will be reloaded from the file on next access.
        unsafe { self.mmap.unchecked_advise(UncheckedAdvice::DontNeed) }
    }
}

fn bad(msg: &str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, msg.to_string())
}

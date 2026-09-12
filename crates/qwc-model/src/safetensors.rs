//! Чтение safetensors через mmap.
//!
//! Формат: [8 байт длины заголовка LE][JSON-заголовок][данные].
//! Смещения в заголовке отсчитываются от начала блока данных.

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
    /// Смещения внутри блока данных шарда.
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
        // SAFETY: файл чекпоинта не изменяется во время работы движка.
        let mmap = unsafe { Mmap::map(&file)? };
        mmap.advise(Advice::Sequential)?;

        if mmap.len() < 8 {
            return Err(bad("шард короче заголовка"));
        }
        let hdr_len = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
        let data_offset = 8 + hdr_len;
        if data_offset > mmap.len() {
            return Err(bad("длина заголовка выходит за пределы файла"));
        }

        let json: serde_json::Value = serde_json::from_slice(&mmap[8..data_offset])
            .map_err(|e| bad(&format!("заголовок не разбирается: {e}")))?;

        let mut tensors = HashMap::new();
        for (name, v) in json.as_object().ok_or_else(|| bad("заголовок не объект"))?
        {
            if name == "__metadata__" {
                continue;
            }
            let dtype_str = v["dtype"].as_str().ok_or_else(|| bad("нет dtype"))?;
            let Some(dtype) = StDtype::parse(dtype_str) else {
                return Err(bad(&format!("неподдерживаемый тип {dtype_str} у {name}")));
            };
            let shape: Vec<usize> = v["shape"]
                .as_array()
                .ok_or_else(|| bad("нет shape"))?
                .iter()
                .map(|x| x.as_u64().unwrap_or(0) as usize)
                .collect();
            let off = v["data_offsets"]
                .as_array()
                .ok_or_else(|| bad("нет data_offsets"))?;
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

    /// Сырые байты тензора. Копирования нет — это срез из mmap.
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
        // SAFETY: отображение read-only и file-backed, грязных страниц нет —
        // MADV_DONTNEED здесь только сбрасывает чистые страницы, данные
        // подгрузятся из файла заново при следующем обращении.
        unsafe { self.mmap.unchecked_advise(UncheckedAdvice::DontNeed) }
    }
}

fn bad(msg: &str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, msg.to_string())
}

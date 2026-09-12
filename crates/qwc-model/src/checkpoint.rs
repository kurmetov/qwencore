//! Открытие чекпоинта и сверка его с архитектурой, зашитой в compile-time.
//!
//! Движок специализирован под одну модель, поэтому любое расхождение формы —
//! это ошибка загрузки, а не повод подстроиться. Универсальный движок молча
//! примет что угодно и упадёт позже, в кернеле.

use crate::names;
use crate::safetensors::{Shard, StDtype, TensorInfo};
use qwc_core::arch::*;
use qwc_core::dtype::NVFP4_BLOCK;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub struct Checkpoint {
    shards: Vec<Shard>,
    /// Имя тензора -> индекс шарда.
    location: HashMap<String, usize>,
}

/// Квантованный linear-слой: упакованные веса, поблочные шкалы и два глобальных.
pub struct QuantLinear<'a> {
    pub packed: &'a [u8],
    /// Шкалы fp8_e4m3, по одной на блок из 16 элементов вдоль K.
    pub block_scales: &'a [u8],
    pub weight_global_scale: f32,
    pub input_global_scale: f32,
    pub out_features: usize,
    pub in_features: usize,
}

#[derive(Debug, Default)]
pub struct Stats {
    pub tensors: usize,
    pub quantized_linears: usize,
    pub packed_bytes: u64,
    pub scale_bytes: u64,
    pub plain_bytes: u64,
    pub embed_bytes: u64,
    pub lm_head_bytes: u64,
    pub mtp_bytes: u64,
    pub unused_bytes: u64,
    pub unused_tensors: usize,
}

impl Stats {
    /// Байт, которые движок реально грузит в VRAM без переквантизации.
    pub fn engine_bytes(&self) -> u64 {
        self.packed_bytes
            + self.scale_bytes
            + self.plain_bytes
            + self.embed_bytes
            + self.lm_head_bytes
            + self.mtp_bytes
    }
}

impl Checkpoint {
    pub fn open(dir: &Path) -> std::io::Result<Self> {
        let index_path = dir.join("model.safetensors.index.json");
        let index: serde_json::Value = serde_json::from_slice(&std::fs::read(&index_path)?)?;
        let map = index["weight_map"]
            .as_object()
            .ok_or_else(|| err("в индексе нет weight_map"))?;

        // Уникальные файлы шардов в стабильном порядке.
        let mut files: Vec<String> = map
            .values()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        files.sort();
        files.dedup();

        let mut shards = Vec::with_capacity(files.len());
        let mut file_idx = HashMap::new();
        for (i, f) in files.iter().enumerate() {
            shards.push(Shard::open(&PathBuf::from(dir).join(f))?);
            file_idx.insert(f.clone(), i);
        }

        let mut location = HashMap::with_capacity(map.len());
        for (name, file) in map {
            if let Some(f) = file.as_str()
                && let Some(&i) = file_idx.get(f)
            {
                location.insert(name.clone(), i);
            }
        }

        Ok(Self { shards, location })
    }

    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.shards[*self.location.get(name)?].info(name)
    }

    pub fn bytes(&self, name: &str) -> Option<&[u8]> {
        self.shards[*self.location.get(name)?].bytes(name)
    }

    fn scalar_f32(&self, name: &str) -> Option<f32> {
        let b = self.bytes(name)?;
        (b.len() >= 4).then(|| f32::from_le_bytes(b[..4].try_into().unwrap()))
    }

    /// Собирает квантованный слой по префиксу, проверяя формы против архитектуры.
    pub fn quant_linear(
        &self,
        prefix: &str,
        out: usize,
        inp: usize,
    ) -> Result<QuantLinear<'_>, String> {
        let packed_name = format!("{prefix}.weight_packed");
        let scale_name = format!("{prefix}.weight_scale");

        let packed_info = self
            .info(&packed_name)
            .ok_or(format!("нет {packed_name}"))?;
        // Два значения fp4 в байте.
        let want_packed = vec![out, inp / 2];
        if packed_info.shape != want_packed {
            return Err(format!(
                "{packed_name}: форма {:?}, ожидалась {want_packed:?}",
                packed_info.shape
            ));
        }
        if packed_info.dtype != StDtype::U8 {
            return Err(format!(
                "{packed_name}: тип {}, ожидался u8",
                packed_info.dtype.name()
            ));
        }

        let scale_info = self.info(&scale_name).ok_or(format!("нет {scale_name}"))?;
        let want_scale = vec![out, inp / NVFP4_BLOCK];
        if scale_info.shape != want_scale {
            return Err(format!(
                "{scale_name}: форма {:?}, ожидалась {want_scale:?}",
                scale_info.shape
            ));
        }
        if scale_info.dtype != StDtype::F8E4m3 {
            return Err(format!(
                "{scale_name}: тип {}, ожидался f8_e4m3",
                scale_info.dtype.name()
            ));
        }

        Ok(QuantLinear {
            packed: self.bytes(&packed_name).unwrap(),
            block_scales: self.bytes(&scale_name).unwrap(),
            weight_global_scale: self
                .scalar_f32(&format!("{prefix}.weight_global_scale"))
                .ok_or(format!("нет {prefix}.weight_global_scale"))?,
            input_global_scale: self
                .scalar_f32(&format!("{prefix}.input_global_scale"))
                .ok_or(format!("нет {prefix}.input_global_scale"))?,
            out_features: out,
            in_features: inp,
        })
    }

    /// Полная сверка с архитектурой. Возвращает список расхождений и статистику.
    pub fn validate(&self) -> (Vec<String>, Stats) {
        let mut problems = Vec::new();
        let mut st = Stats::default();
        let mut seen: Vec<String> = Vec::new();

        let check_plain =
            |name: &str, want: &[usize], st: &mut Stats, problems: &mut Vec<String>| match self
                .info(name)
            {
                None => problems.push(format!("нет {name}")),
                Some(t) => {
                    if t.shape != want {
                        problems.push(format!("{name}: форма {:?}, ожидалась {want:?}", t.shape));
                    }
                    st.plain_bytes += t.bytes() as u64;
                    st.tensors += 1;
                }
            };

        check_plain(names::FINAL_NORM, &[HIDDEN_SIZE], &mut st, &mut problems);
        seen.push(names::FINAL_NORM.into());

        for (name, field) in [(names::EMBED, 0usize), (names::LM_HEAD, 1)] {
            match self.info(name) {
                None => problems.push(format!("нет {name}")),
                Some(t) => {
                    if t.shape != [VOCAB_SIZE, HIDDEN_SIZE] {
                        problems.push(format!("{name}: форма {:?}", t.shape));
                    }
                    let b = t.bytes() as u64;
                    if field == 0 {
                        st.embed_bytes += b
                    } else {
                        st.lm_head_bytes += b
                    }
                    st.tensors += 1;
                }
            }
            seen.push(name.into());
        }

        for i in 0..NUM_LAYERS {
            for (name, want) in names::plain_tensors(i) {
                check_plain(&name, &want, &mut st, &mut problems);
                seen.push(name);
            }
            for (prefix, out, inp) in names::quant_linears(i) {
                match self.quant_linear(&prefix, out, inp) {
                    Err(e) => problems.push(e),
                    Ok(q) => {
                        st.quantized_linears += 1;
                        st.packed_bytes += q.packed.len() as u64;
                        st.scale_bytes += q.block_scales.len() as u64;
                        st.tensors += 4;
                    }
                }
                for suf in [
                    "weight_packed",
                    "weight_scale",
                    "weight_global_scale",
                    "input_global_scale",
                ] {
                    seen.push(format!("{prefix}.{suf}"));
                }
            }
        }

        for (name, want) in names::mtp_tensors() {
            match self.info(&name) {
                None => problems.push(format!("нет {name}")),
                Some(t) => {
                    if t.shape != want {
                        problems.push(format!("{name}: форма {:?}, ожидалась {want:?}", t.shape));
                    }
                    st.mtp_bytes += t.bytes() as u64;
                    st.tensors += 1;
                }
            }
            seen.push(name);
        }

        // Всё, что мы не тронули: vision tower и прочее, чего в text-only нет.
        let seen: std::collections::HashSet<&str> = seen.iter().map(|s| s.as_str()).collect();
        for name in self.location.keys() {
            if !seen.contains(name.as_str())
                && let Some(t) = self.info(name)
            {
                st.unused_bytes += t.bytes() as u64;
                st.unused_tensors += 1;
            }
        }

        (problems, st)
    }

    pub fn tensor_count(&self) -> usize {
        self.location.len()
    }

    /// Releases checkpoint mmap pages from the process working set. A loader
    /// calls this between upload phases to honor a small host-RAM envelope.
    pub fn evict_file_pages(&self) -> std::io::Result<()> {
        for shard in &self.shards {
            shard.evict_pages()?;
        }
        Ok(())
    }
}

fn err(msg: &str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, msg.to_string())
}

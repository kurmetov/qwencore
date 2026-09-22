//! Форматы хранения весов и кэша.
//!
//! NVFP4 — это не просто «4 бита»: на каждый блок из 16 элементов идёт
//! отдельная шкала float8_e4m3, что даёт 4.5 бита на элемент.
//! Игнорировать это при планировании памяти нельзя — на 24B параметров
//! забытые шкалы дают промах в 1.5 GB.

pub const NVFP4_BLOCK: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dtype {
    /// Веса и активации в bfloat16.
    Bf16,
    /// float8_e4m3 — шкалы блоков, KV-кэш, при желании эмбеддинги.
    Fp8E4m3,
    /// NVFP4: 4 бита на элемент + fp8-шкала на блок из 16.
    Nvfp4,
    /// int8 с масштабом f32 на строку состояния (128 элементов). Формат
    /// хранения рекуррентного состояния DeltaNet: строки плоские, и на них
    /// он вчетверо точнее fp8 при том же байте
    /// (`bench/results/delta-state-8bit.md`).
    Int8Row128,
    Fp32,
}

impl Dtype {
    /// Байт на `n` элементов. Целочисленная арифметика: для NVFP4 это
    /// n/2 байт данных плюс n/16 байт шкал.
    pub const fn bytes(self, n: usize) -> u64 {
        match self {
            Dtype::Fp32 => (n * 4) as u64,
            Dtype::Bf16 => (n * 2) as u64,
            Dtype::Fp8E4m3 => n as u64,
            Dtype::Nvfp4 => (n / 2 + n.div_ceil(NVFP4_BLOCK)) as u64,
            Dtype::Int8Row128 => (n + n.div_ceil(128) * 4) as u64,
        }
    }

    /// Эффективная разрядность, включая накладные расходы на шкалы.
    pub const fn effective_bits(self) -> f32 {
        match self {
            Dtype::Fp32 => 32.0,
            Dtype::Bf16 => 16.0,
            Dtype::Fp8E4m3 => 8.0,
            Dtype::Nvfp4 => 4.5,
            Dtype::Int8Row128 => 8.25,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Dtype::Fp32 => "fp32",
            Dtype::Bf16 => "bf16",
            Dtype::Fp8E4m3 => "fp8_e4m3",
            Dtype::Nvfp4 => "nvfp4",
            Dtype::Int8Row128 => "int8_row128",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvfp4_includes_block_scales() {
        // Ровно один блок: 8 байт данных + 1 байт шкалы.
        assert_eq!(Dtype::Nvfp4.bytes(16), 9);
        // 4.5 бита на элемент на большом тензоре.
        let n = 1_000_000;
        let bits = Dtype::Nvfp4.bytes(n) as f64 * 8.0 / n as f64;
        assert!((bits - 4.5).abs() < 0.01, "{bits} бит/элемент");
    }

    #[test]
    fn quantized_body_size() {
        use crate::arch::QUANTIZED_PARAMS;
        let gb = Dtype::Nvfp4.bytes(QUANTIZED_PARAMS) as f64 / 1e9;
        assert!((13.0..14.5).contains(&gb), "тело модели в NVFP4: {gb} GB");
    }
}

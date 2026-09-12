//! Движок: подъём весов в VRAM и исполнение шага decode.
//!
//! Крейт лежит выше `qwc-cuda` (кернелы), `qwc-model` (чекпоинт) и
//! `qwc-runtime` (планирование), потому что только здесь эти три границы
//! встречаются. Ниже они друг о друге не знают: runtime остаётся
//! CPU-тестируемым, а qwc-model не тянет CUDA.

pub mod executor;
pub mod weights;

pub use executor::{Executor, ExecutorConfig, PREFILL_CHUNK_SIZE};
pub use weights::{
    DecodeLinearMode, EmbeddingDtype, LmHeadDtype, LoadConfig, LoadError, LoadStats, ModelWeights,
};

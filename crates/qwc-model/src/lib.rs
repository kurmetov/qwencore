//! Загрузка чекпоинта Qwen3.8-27B-NVFP4.
//!
//! Загрузчик знает архитектуру на этапе компиляции и сверяет с ней каждый
//! тензор. Расхождение формы — ошибка загрузки, а не повод подстроиться.

pub mod checkpoint;
pub mod names;
pub mod safetensors;

pub use checkpoint::{Checkpoint, QuantLinear, Stats};

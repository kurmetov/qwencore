//! CUDA-слой движка: FFI, память, потоки, кернелы.
//!
//! Целевая архитектура ровно одна — sm_120a (RTX 5090). Это не настройка
//! сборки, а часть специализации: кернелы пишутся под 99 KB shared memory,
//! 170 SM и block-scaled MMA для NVFP4, и на другом железе смысла не имеют.

pub mod bandwidth;
pub mod device;
pub mod error;
pub(crate) mod ffi;
pub mod memory;
pub mod stream;

pub use device::Device;
pub use error::{CudaError, Result};
pub use memory::DeviceBuffer;
pub use stream::{Event, Stream};

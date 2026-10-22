//! The engine: loading weights into VRAM and executing the decode step.
//!
//! The crate sits above `qwc-cuda` (kernels), `qwc-model` (checkpoint) and
//! `qwc-runtime` (scheduling), because only here do these three boundaries
//! meet. Below, they don't know about each other: runtime stays
//! CPU-testable, and qwc-model doesn't pull in CUDA.

pub mod executor;
pub mod mtp;
pub mod weights;

pub use executor::{Executor, ExecutorConfig, PREFILL_CHUNK_SIZE};
pub use weights::{
    DecodeLinearMode, EmbeddingDtype, LmHeadDtype, LoadConfig, LoadError, LoadStats, ModelWeights,
};

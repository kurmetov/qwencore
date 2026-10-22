//! The qwencore core: the Qwen3.8-27B architecture as compile-time constants
//! and memory-planning arithmetic for a single RTX 5090.
//!
//! There's not a single runtime-configurable model dimension here — this is
//! deliberate specialization. Everything that depends on the load (concurrency,
//! context length, cache dtype) is configurable; everything that depends on the model
//! is baked into constants and checked by the compiler.

pub mod arch;
pub mod dtype;
pub mod memory;
pub mod roofline;

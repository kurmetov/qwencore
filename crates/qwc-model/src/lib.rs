//! Loading the Qwen3.8-27B-NVFP4 checkpoint.
//!
//! The loader knows the architecture at compile time and checks every
//! tensor against it. A shape mismatch is a load error, not a reason to adapt.

pub mod checkpoint;
pub mod names;
pub mod safetensors;

pub use checkpoint::{Checkpoint, QuantLinear, Stats};

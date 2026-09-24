//! Рантайм движка: управление кэшем, планирование, батчинг.

pub mod cache;
pub mod layout;
pub mod scheduler;

pub use cache::{Admission, CacheManager, PrefixRestore, PrefixStats, Pressure, Rejected, SeqId};
pub use layout::{BatchLayout, LayoutError};
pub use scheduler::{
    Batch, Completion, FinishReason, PrefillChunk, PrefixRestoreOp, PrefixSaveOp, Request, Scheduler, SchedulerConfig,
    SchedulerError, SubmitError,
};

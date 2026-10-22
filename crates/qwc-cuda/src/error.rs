//! CUDA errors.

use std::ffi::CStr;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaError(pub i32);

pub type Result<T> = std::result::Result<T, CudaError>;

impl CudaError {
    pub fn message(self) -> String {
        // SAFETY: cudaGetErrorString returns a static string for any
        // code, including an unknown one.
        let s = unsafe { crate::ffi::cudaGetErrorString(self.0) };
        if s.is_null() {
            return format!("unknown CUDA error {}", self.0);
        }
        unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned()
    }
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CUDA {}: {}", self.0, self.message())
    }
}

impl std::error::Error for CudaError {}

pub(crate) fn check(code: i32) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(CudaError(code))
    }
}

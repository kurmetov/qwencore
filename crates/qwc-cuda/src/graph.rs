//! Minimal CUDA graph lifetime wrapper for stable-address executor paths.

use crate::error::{Result, check};
use crate::{Stream, ffi};

const CAPTURE_MODE_THREAD_LOCAL: i32 = 1;

pub struct CudaGraph {
    executable: ffi::GraphExec,
}

impl CudaGraph {
    pub fn begin(stream: &Stream) -> Result<()> {
        check(unsafe { ffi::cudaStreamBeginCapture(stream.raw(), CAPTURE_MODE_THREAD_LOCAL) })
    }

    pub fn end(stream: &Stream) -> Result<Self> {
        let mut graph: ffi::Graph = std::ptr::null_mut();
        check(unsafe { ffi::cudaStreamEndCapture(stream.raw(), &mut graph) })?;
        let mut executable: ffi::GraphExec = std::ptr::null_mut();
        let instantiated =
            check(unsafe { ffi::cudaGraphInstantiateWithFlags(&mut executable, graph, 0) });
        // Instantiation retains everything needed by the executable graph.
        let destroyed = check(unsafe { ffi::cudaGraphDestroy(graph) });
        instantiated?;
        destroyed?;
        Ok(Self { executable })
    }

    pub fn launch(&self, stream: &Stream) -> Result<()> {
        check(unsafe { ffi::cudaGraphLaunch(self.executable, stream.raw()) })
    }
}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        if !self.executable.is_null() {
            unsafe {
                ffi::cudaGraphExecDestroy(self.executable);
            }
        }
    }
}

//! Device properties. Values are read from the hardware, not taken from the spec:
//! the denominator in the "share of achieved bandwidth" metric must be
//! measurable, otherwise the metric is meaningless.

use crate::error::{Result, check};
use crate::ffi;
use std::ffi::c_int;

// Values from driver_types.h.
const ATTR_CLOCK_RATE: c_int = 13;
const ATTR_SM_COUNT: c_int = 16;
const ATTR_MEMORY_CLOCK_RATE: c_int = 36;
const ATTR_MEMORY_BUS_WIDTH: c_int = 37;
const ATTR_L2_CACHE_SIZE: c_int = 38;
const ATTR_MAX_THREADS_PER_SM: c_int = 39;
const ATTR_MAX_REGISTERS_PER_SM: c_int = 82;
const ATTR_MAX_SHARED_MEM_OPTIN: c_int = 97;

#[derive(Debug, Clone, Copy)]
pub struct Device {
    pub sm_count: u32,
    pub sm_clock_hz: f64,
    pub memory_clock_hz: f64,
    pub memory_bus_bits: u32,
    pub l2_bytes: u64,
    pub max_threads_per_sm: u32,
    pub max_registers_per_sm: u32,
    /// Shared memory cap per block on explicit opt-in. For sm_120 it's 99 KB,
    /// four times less than datacenter Blackwell. Determines the attention tile.
    pub max_shared_mem_optin: u32,
}

impl Device {
    pub fn init(index: i32) -> Result<Self> {
        check(unsafe { ffi::cudaSetDevice(index) })?;
        let a = |attr| -> Result<i32> {
            let mut v = 0;
            check(unsafe { ffi::cudaDeviceGetAttribute(&mut v, attr, index) })?;
            Ok(v)
        };
        Ok(Self {
            sm_count: a(ATTR_SM_COUNT)? as u32,
            sm_clock_hz: a(ATTR_CLOCK_RATE)? as f64 * 1e3,
            memory_clock_hz: a(ATTR_MEMORY_CLOCK_RATE)? as f64 * 1e3,
            memory_bus_bits: a(ATTR_MEMORY_BUS_WIDTH)? as u32,
            l2_bytes: a(ATTR_L2_CACHE_SIZE)? as u64,
            max_threads_per_sm: a(ATTR_MAX_THREADS_PER_SM)? as u32,
            max_registers_per_sm: a(ATTR_MAX_REGISTERS_PER_SM)? as u32,
            max_shared_mem_optin: a(ATTR_MAX_SHARED_MEM_OPTIN)? as u32,
        })
    }

    /// Spec peak: double the memory clock times the bus width.
    pub fn peak_bandwidth(&self) -> f64 {
        2.0 * self.memory_clock_hz * (self.memory_bus_bits as f64 / 8.0)
    }

    /// Free and total VRAM bytes.
    pub fn mem_info() -> Result<(u64, u64)> {
        let (mut free, mut total) = (0usize, 0usize);
        check(unsafe { ffi::cudaMemGetInfo(&mut free, &mut total) })?;
        Ok((free as u64, total as u64))
    }

    pub fn synchronize() -> Result<()> {
        check(unsafe { ffi::cudaDeviceSynchronize() })
    }
}

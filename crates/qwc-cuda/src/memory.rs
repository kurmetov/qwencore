//! Device memory with RAII.
//!
//! The engine runs on preallocated buffers: allocations during inference
//! are not allowed. This type is the foundation of a static memory plan, not
//! a general-purpose allocator.

use crate::error::{Result, check};
use crate::ffi;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};

const CUDA_ERROR_MEMORY_ALLOCATION: i32 = 2;
static MEMORY_LIMIT: AtomicUsize = AtomicUsize::new(usize::MAX);
static MEMORY_USED: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryUsage {
    pub used: usize,
    pub limit: usize,
}

impl MemoryUsage {
    pub fn remaining(self) -> usize {
        self.limit.saturating_sub(self.used)
    }
}

/// Headroom for what our accounting doesn't see: cudaMalloc rounds every allocation
/// up, and across hundreds of buffers the difference adds up to hundreds of megabytes.
const FREE_MARGIN: usize = 512 * 1024 * 1024;

/// Sets a hard ceiling for allocations owned by this engine process. Configure
/// it after creating the CUDA context and before allocating any model buffer.
///
/// The ceiling is clipped to the card's actually free memory: other processes
/// live alongside, and the requested budget may simply not exist. Without this
/// the engine gets halfway through loading and fails on cudaMalloc, whereas vLLM
/// at the same point profiles free memory and takes less than nominal.
/// Returns the limit actually set.
pub fn set_memory_limit(bytes: usize) -> std::result::Result<usize, &'static str> {
    if bytes == 0 {
        return Err("VRAM limit must be positive");
    }
    if MEMORY_USED.load(Ordering::Acquire) != 0 {
        return Err("VRAM limit must be configured before the first allocation");
    }
    let effective = match device_free_bytes() {
        Some(free) => bytes.min(free.saturating_sub(FREE_MARGIN)),
        None => bytes,
    };
    if effective == 0 {
        return Err("no free memory on the card for the requested budget");
    }
    MEMORY_LIMIT.store(effective, Ordering::Release);
    Ok(effective)
}

/// Free bytes on the device, or `None` if the context isn't created yet and
/// there's nobody to ask — then the limit stays what the caller set.
///
/// This is the only honest quantity when sharing the card: our own accounting
/// counts requested bytes, not what the driver rounded them to, and doesn't see
/// the context, graphs, or other processes.
pub fn device_free_bytes() -> Option<usize> {
    let (mut free, mut total) = (0usize, 0usize);
    // SAFETY: both references are valid; the call doesn't touch our buffers.
    let status = unsafe { ffi::cudaMemGetInfo(&mut free, &mut total) };
    (status == 0 && total > 0).then_some(free)
}

pub fn memory_usage() -> MemoryUsage {
    MemoryUsage {
        used: MEMORY_USED.load(Ordering::Acquire),
        limit: MEMORY_LIMIT.load(Ordering::Acquire),
    }
}

fn reserve(bytes: usize) -> Result<()> {
    MEMORY_USED
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
            used.checked_add(bytes)
                .filter(|&next| next <= MEMORY_LIMIT.load(Ordering::Acquire))
        })
        .map(|_| ())
        .map_err(|_| crate::CudaError(CUDA_ERROR_MEMORY_ALLOCATION))
}

fn release(bytes: usize) {
    MEMORY_USED.fetch_sub(bytes, Ordering::AcqRel);
}

pub struct DeviceBuffer<T> {
    ptr: *mut c_void,
    len: usize,
    allocation_bytes: usize,
    _marker: PhantomData<T>,
}

// A device pointer can be sent between threads: it belongs to the context,
// not the thread. Access synchronization is provided by CUDA streams.
unsafe impl<T: Send> Send for DeviceBuffer<T> {}
unsafe impl<T: Sync> Sync for DeviceBuffer<T> {}

impl<T> DeviceBuffer<T> {
    pub fn zeroed(len: usize) -> Result<Self> {
        let bytes = std::mem::size_of::<T>() * len;
        let allocation_bytes = bytes.max(1);
        reserve(allocation_bytes)?;
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let allocated = unsafe { ffi::cudaMalloc(&mut ptr, allocation_bytes) };
        if allocated != 0 {
            release(allocation_bytes);
            return Err(crate::CudaError(allocated));
        }
        let cleared = unsafe { ffi::cudaMemset(ptr, 0, allocation_bytes) };
        if cleared != 0 {
            unsafe { ffi::cudaFree(ptr) };
            release(allocation_bytes);
            return Err(crate::CudaError(cleared));
        }
        Ok(Self {
            ptr,
            len,
            allocation_bytes,
            _marker: PhantomData,
        })
    }

    pub fn from_slice(data: &[T]) -> Result<Self> {
        let bytes = std::mem::size_of_val(data);
        let allocation_bytes = bytes.max(1);
        reserve(allocation_bytes)?;
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let allocated = unsafe { ffi::cudaMalloc(&mut ptr, allocation_bytes) };
        if allocated != 0 {
            release(allocation_bytes);
            return Err(crate::CudaError(allocated));
        }
        if bytes > 0 {
            let copied = unsafe {
                ffi::cudaMemcpy(ptr, data.as_ptr().cast(), bytes, ffi::MEMCPY_HOST_TO_DEVICE)
            };
            if copied != 0 {
                unsafe { ffi::cudaFree(ptr) };
                release(allocation_bytes);
                return Err(crate::CudaError(copied));
            }
        }
        Ok(Self {
            ptr,
            len: data.len(),
            allocation_bytes,
            _marker: PhantomData,
        })
    }

    /// Copies a host slice to the start of the buffer without recreating the allocation:
    /// on the load path one staging buffer is reused dozens of times.
    pub fn copy_from_slice(&mut self, data: &[T]) -> Result<()> {
        self.copy_from_slice_at(0, data)
    }

    /// Copies a host slice to the given offset without an intermediate
    /// host copy. This allows streaming large mmap tensors.
    pub fn copy_from_slice_at(&mut self, offset: usize, data: &[T]) -> Result<()> {
        assert!(offset <= self.len && data.len() <= self.len - offset);
        if data.is_empty() {
            return Ok(());
        }
        let byte_offset = offset * std::mem::size_of::<T>();
        // SAFETY: element range checked by the assertion above.
        let destination = unsafe { (self.ptr as *mut u8).add(byte_offset).cast() };
        check(unsafe {
            ffi::cudaMemcpy(
                destination,
                data.as_ptr().cast(),
                std::mem::size_of_val(data),
                ffi::MEMCPY_HOST_TO_DEVICE,
            )
        })
    }

    /// A copy of an element range from another buffer, without going through the host.
    ///
    /// Needed where a kernel needs a separate buffer, but the data already sits
    /// as a row inside someone else's arena: the hidden-state row for the MTP draft,
    /// the DeltaNet state snapshot for speculation rollback.
    pub fn copy_from_device_at(
        &mut self,
        offset: usize,
        source: &Self,
        source_offset: usize,
        len: usize,
        stream: &crate::Stream,
    ) -> Result<()> {
        assert!(offset + len <= self.len && source_offset + len <= source.len);
        if len == 0 {
            return Ok(());
        }
        let size = std::mem::size_of::<T>();
        // SAFETY: both ranges checked by the assertion above.
        let destination = unsafe { (self.ptr as *mut u8).add(offset * size).cast() };
        let from = unsafe { (source.ptr as *const u8).add(source_offset * size).cast() };
        check(unsafe {
            ffi::cudaMemcpyAsync(
                destination,
                from,
                len * size,
                ffi::MEMCPY_DEVICE_TO_DEVICE,
                stream.raw(),
            )
        })
    }

    /// A copy of a range within one buffer: a state-slot snapshot for
    /// speculation rollback and putting the accepted state back.
    pub fn copy_within(
        &mut self,
        destination_offset: usize,
        source_offset: usize,
        len: usize,
        stream: &crate::Stream,
    ) -> Result<()> {
        assert!(destination_offset + len <= self.len && source_offset + len <= self.len);
        if len == 0 {
            return Ok(());
        }
        let size = std::mem::size_of::<T>();
        // SAFETY: both ranges are inside the allocation, checked above.
        let destination = unsafe { (self.ptr as *mut u8).add(destination_offset * size).cast() };
        let from = unsafe { (self.ptr as *const u8).add(source_offset * size).cast() };
        check(unsafe {
            ffi::cudaMemcpyAsync(
                destination,
                from,
                len * size,
                ffi::MEMCPY_DEVICE_TO_DEVICE,
                stream.raw(),
            )
        })
    }

    /// Zeroes an element range in place without a host-sized staging buffer.
    pub fn zero_range(&mut self, start: usize, len: usize) -> Result<()> {
        assert!(start <= self.len && len <= self.len - start);
        if len == 0 {
            return Ok(());
        }
        let byte_offset = start * std::mem::size_of::<T>();
        let bytes = len * std::mem::size_of::<T>();
        // SAFETY: the asserted element range lies inside this allocation.
        let pointer = unsafe { (self.ptr as *mut u8).add(byte_offset).cast() };
        check(unsafe { ffi::cudaMemset(pointer, 0, bytes) })
    }

    pub fn to_vec(&self) -> Result<Vec<T>>
    where
        T: Copy + Default,
    {
        let mut host = vec![T::default(); self.len];
        check(unsafe {
            ffi::cudaMemcpy(
                host.as_mut_ptr().cast(),
                self.ptr,
                self.bytes(),
                ffi::MEMCPY_DEVICE_TO_HOST,
            )
        })?;
        Ok(host)
    }

    /// Elements `[start, start + len)` to host.
    pub fn to_vec_range(&self, start: usize, len: usize) -> Result<Vec<T>>
    where
        T: Copy + Default,
    {
        assert!(start <= self.len && len <= self.len - start);
        let mut host = vec![T::default(); len];
        if len == 0 {
            return Ok(host);
        }
        // SAFETY: the asserted element range lies inside this allocation.
        let source = unsafe { (self.ptr as *const u8).add(start * std::mem::size_of::<T>()) };
        check(unsafe {
            ffi::cudaMemcpy(
                host.as_mut_ptr().cast(),
                source.cast(),
                len * std::mem::size_of::<T>(),
                ffi::MEMCPY_DEVICE_TO_HOST,
            )
        })?;
        Ok(host)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn bytes(&self) -> usize {
        std::mem::size_of::<T>() * self.len
    }

    pub fn as_ptr(&self) -> *const c_void {
        self.ptr
    }

    pub fn as_mut_ptr(&mut self) -> *mut c_void {
        self.ptr
    }
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::cudaFree(self.ptr) };
            release(self.allocation_bytes);
        }
    }
}

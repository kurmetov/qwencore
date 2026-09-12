//! Device-память с RAII.
//!
//! Движок работает на преаллоцированных буферах: аллокации во время инференса
//! не допускаются. Этот тип — фундамент для статического плана памяти, а не
//! аллокатор общего назначения.

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

/// Sets a hard ceiling for allocations owned by this engine process. Configure
/// it after creating the CUDA context and before allocating any model buffer.
pub fn set_memory_limit(bytes: usize) -> std::result::Result<(), &'static str> {
    if bytes == 0 {
        return Err("VRAM limit must be positive");
    }
    if MEMORY_USED.load(Ordering::Acquire) != 0 {
        return Err("VRAM limit must be configured before the first allocation");
    }
    MEMORY_LIMIT.store(bytes, Ordering::Release);
    Ok(())
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

// Device-указатель можно передавать между потоками: он принадлежит контексту,
// а не потоку. Синхронизацию доступа обеспечивают потоки CUDA.
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

    /// Копирует срез хоста в начало буфера, не пересоздавая аллокацию:
    /// на пути загрузки один staging-буфер переиспользуется десятки раз.
    pub fn copy_from_slice(&mut self, data: &[T]) -> Result<()> {
        self.copy_from_slice_at(0, data)
    }

    /// Копирует срез хоста в заданное смещение без промежуточной
    /// host-копии. Это позволяет потоково загружать большие mmap-тензоры.
    pub fn copy_from_slice_at(&mut self, offset: usize, data: &[T]) -> Result<()> {
        assert!(offset <= self.len && data.len() <= self.len - offset);
        if data.is_empty() {
            return Ok(());
        }
        let byte_offset = offset * std::mem::size_of::<T>();
        // SAFETY: диапазон элементов проверен assertion выше.
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

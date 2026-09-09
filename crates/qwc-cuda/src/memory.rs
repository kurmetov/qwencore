//! Device-память с RAII.
//!
//! Движок работает на преаллоцированных буферах: аллокации во время инференса
//! не допускаются. Этот тип — фундамент для статического плана памяти, а не
//! аллокатор общего назначения.

use crate::error::{Result, check};
use crate::ffi;
use std::ffi::c_void;
use std::marker::PhantomData;

pub struct DeviceBuffer<T> {
    ptr: *mut c_void,
    len: usize,
    _marker: PhantomData<T>,
}

// Device-указатель можно передавать между потоками: он принадлежит контексту,
// а не потоку. Синхронизацию доступа обеспечивают потоки CUDA.
unsafe impl<T: Send> Send for DeviceBuffer<T> {}
unsafe impl<T: Sync> Sync for DeviceBuffer<T> {}

impl<T> DeviceBuffer<T> {
    pub fn zeroed(len: usize) -> Result<Self> {
        let bytes = std::mem::size_of::<T>() * len;
        let mut ptr: *mut c_void = std::ptr::null_mut();
        check(unsafe { ffi::cudaMalloc(&mut ptr, bytes) })?;
        check(unsafe { ffi::cudaMemset(ptr, 0, bytes) })?;
        Ok(Self { ptr, len, _marker: PhantomData })
    }

    pub fn from_slice(data: &[T]) -> Result<Self> {
        let bytes = std::mem::size_of_val(data);
        let mut ptr: *mut c_void = std::ptr::null_mut();
        check(unsafe { ffi::cudaMalloc(&mut ptr, bytes) })?;
        check(unsafe {
            ffi::cudaMemcpy(ptr, data.as_ptr().cast(), bytes, ffi::MEMCPY_HOST_TO_DEVICE)
        })?;
        Ok(Self { ptr, len: data.len(), _marker: PhantomData })
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
        }
    }
}

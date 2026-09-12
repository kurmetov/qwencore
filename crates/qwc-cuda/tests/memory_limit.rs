//! Process-level VRAM ceiling: reservation, refusal and release on drop.

use qwc_cuda::{CudaError, DeviceBuffer, memory_usage, set_memory_limit};

/// cudaErrorMemoryAllocation — the ceiling reports the same code the driver
/// would report, so callers need no second error path.
const OUT_OF_MEMORY: CudaError = CudaError(2);

/// The ceiling is process-global state, so the whole contract is checked in a
/// single test: cargo runs tests of one binary in parallel threads.
#[test]
fn ceiling_bounds_allocations_and_is_released_on_drop() {
    assert_eq!(memory_usage().used, 0);
    assert_eq!(memory_usage().limit, usize::MAX);
    assert!(set_memory_limit(0).is_err(), "нулевой лимит бессмыслен");

    const LIMIT: usize = 4096;
    set_memory_limit(LIMIT).expect("лимит ставится до первой аллокации");

    let head = DeviceBuffer::<u8>::zeroed(3072).expect("влезает под потолок");
    assert_eq!(memory_usage().used, 3072);
    assert_eq!(memory_usage().remaining(), 1024);

    // Отказ происходит до cudaMalloc и не меняет учёт.
    assert_eq!(DeviceBuffer::<u8>::zeroed(1025).err(), Some(OUT_OF_MEMORY));
    assert_eq!(memory_usage().used, 3072);
    // Считаются байты, а не элементы: 257 * 4 > 1024.
    assert_eq!(DeviceBuffer::<f32>::zeroed(257).err(), Some(OUT_OF_MEMORY));
    assert_eq!(memory_usage().used, 3072);

    // from_slice учитывается тем же потолком.
    let data = vec![7u8; 1024];
    let tail = DeviceBuffer::from_slice(&data).expect("ровно добирает потолок");
    assert_eq!(memory_usage().used, LIMIT);
    assert_eq!(memory_usage().remaining(), 0);
    assert_eq!(tail.to_vec().expect("копия обратно на хост"), data);

    // Пока живы буферы, лимит не переставить — иначе учёт разошёлся бы с VRAM.
    assert!(set_memory_limit(LIMIT * 2).is_err());

    drop(tail);
    assert_eq!(memory_usage().used, 3072);
    drop(head);
    assert_eq!(memory_usage().used, 0);

    // Освобождённое место снова доступно целиком.
    let whole = DeviceBuffer::<u8>::zeroed(LIMIT).expect("потолок доступен заново");
    assert_eq!(memory_usage().used, LIMIT);
    drop(whole);
    assert_eq!(memory_usage().used, 0);
}

//! GPU argmax correctness and deterministic ties.

use qwc_cuda::sampling::Argmax;
use qwc_cuda::{DeviceBuffer, Stream};

#[test]
fn gpu_argmax_matches_cpu_for_capacity_sized_buffers() {
    let max_batch = 4;
    let batch = 3;
    let vocab = 10_003;
    let mut logits = vec![f32::NEG_INFINITY; max_batch * vocab];
    let expected = [9_999u32, 17, 8_008];
    for row in 0..batch {
        for column in 0..vocab {
            logits[row * vocab + column] = ((column * 37 + row * 101) % 997) as f32 * 0.01 - 4.0;
        }
        logits[row * vocab + expected[row] as usize] = 100.0;
    }
    // Equal maxima choose the smaller vocabulary index.
    logits[vocab + 29] = 100.0;

    let stream = Stream::new().unwrap();
    let device_logits = DeviceBuffer::from_slice(&logits).unwrap();
    let mut sampler = Argmax::new(max_batch, vocab).unwrap();
    sampler.sample(&device_logits, batch, &stream).unwrap();
    stream.synchronize().unwrap();
    assert_eq!(sampler.to_host(batch).unwrap(), expected);
}

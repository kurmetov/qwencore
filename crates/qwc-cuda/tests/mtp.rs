//! Сверка BF16-линейки draft-головы с эталоном на CPU.

use qwc_cuda::mtp;
use qwc_cuda::{DeviceBuffer, Stream, bf16};

struct Rng(u64);

impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
}

/// Формы взяты из головы как она лежит в чекпоинте: fc [5120, 10240],
/// q_proj [12288, 5120], down_proj [5120, 17408]. Строк до четырёх — столько
/// несёт проверка черновиков.
#[test]
fn bf16_linear_matches_cpu_on_head_shapes() {
    let stream = Stream::new().unwrap();
    for (rows, k, n) in [(1usize, 5120usize, 10240usize), (4, 10240, 5120), (2, 17408, 5120)] {
        let mut rng = Rng(0x0BF1_6000 ^ ((k as u64) << 8) ^ n as u64);
        let weights: Vec<u16> = (0..k * n)
            .map(|_| bf16::from_f32(rng.next_f32() * 0.05))
            .collect();
        let input: Vec<u16> = (0..rows * k)
            .map(|_| bf16::from_f32(rng.next_f32()))
            .collect();

        let device_weights = DeviceBuffer::from_slice(&weights).unwrap();
        let device_input = DeviceBuffer::from_slice(&input).unwrap();
        let mut device_output = DeviceBuffer::<u16>::zeroed(rows * n).unwrap();
        mtp::bf16_linear(
            &device_weights,
            &device_input,
            &mut device_output,
            rows,
            k,
            n,
            &stream,
        )
        .unwrap();
        stream.synchronize().unwrap();
        let actual = device_output.to_vec().unwrap();

        let mut worst = 0.0f32;
        let mut scale = 0.0f32;
        for row in 0..rows {
            for column in 0..n {
                let mut sum = 0.0f32;
                for index in 0..k {
                    sum += bf16::to_f32(input[row * k + index])
                        * bf16::to_f32(weights[column * k + index]);
                }
                let got = bf16::to_f32(actual[row * n + column]);
                worst = worst.max((got - sum).abs());
                scale = scale.max(sum.abs());
            }
        }
        // Выход кладётся в BF16, поэтому порог — его собственный шаг
        // округления, около 0.4% величины.
        assert!(
            worst <= 6e-3 * scale,
            "формы {rows}x{k}x{n}: расхождение {worst:.3e} при масштабе {scale:.3e}"
        );
        println!("{rows}x{k}x{n}: max {worst:.3e} при масштабе {scale:.3e}");
    }
}

#[test]
fn swiglu_and_concat_match_cpu() {
    let stream = Stream::new().unwrap();
    let mut rng = Rng(0x5117_0000);
    let rows = 3usize;
    let width = 512usize;
    let gate: Vec<u16> = (0..rows * width).map(|_| bf16::from_f32(rng.next_f32() * 2.0)).collect();
    let up: Vec<u16> = (0..rows * width).map(|_| bf16::from_f32(rng.next_f32())).collect();

    let device_gate = DeviceBuffer::from_slice(&gate).unwrap();
    let device_up = DeviceBuffer::from_slice(&up).unwrap();
    let mut device_out = DeviceBuffer::<u16>::zeroed(rows * width).unwrap();
    mtp::swiglu(&device_gate, &device_up, &mut device_out, rows * width, &stream).unwrap();
    let mut device_concat = DeviceBuffer::<u16>::zeroed(rows * 2 * width).unwrap();
    mtp::concat(&device_gate, &device_up, &mut device_concat, rows, width, &stream).unwrap();
    stream.synchronize().unwrap();

    let out = device_out.to_vec().unwrap();
    for index in 0..rows * width {
        let g = bf16::to_f32(gate[index]);
        let want = g / (1.0 + (-g).exp()) * bf16::to_f32(up[index]);
        let got = bf16::to_f32(out[index]);
        assert!(
            (got - want).abs() <= 6e-3 * want.abs().max(1e-2),
            "swiglu[{index}]: {got} против {want}"
        );
    }
    let joined = device_concat.to_vec().unwrap();
    for row in 0..rows {
        assert_eq!(&joined[row * 2 * width..][..width], &gate[row * width..][..width]);
        assert_eq!(&joined[row * 2 * width + width..][..width], &up[row * width..][..width]);
    }
}

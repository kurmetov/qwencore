//! NVFP4 decode-проекции.
//!
//! Первый production-кандидат — W4A16 для batch 1..4: checkpoint-веса
//! используются напрямую, активация остаётся BF16. Это сохраняет качество
//! относительно W4A4 и убирает activation-quantization launch.

use crate::error::{Result, check};
use crate::{DeviceBuffer, Stream, bf16, ffi};

pub const GROUP_SIZE: usize = 16;
pub const MAX_W4A16_BATCH: usize = 4;
pub const MAX_W4A4_BATCH: usize = 128;
pub const MAX_W4A4_ROWS: usize = 1024;
const K_ALIGNMENT: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeKernel {
    W4A16,
    W4A4,
}

/// Измеренная на RTX 5090 граница для форм Qwen3.8.
///
/// B<=3 остаётся на W4A16 ради latency и отсутствия activation-quantization.
/// При B=4 tensor cores выигрывают на широких проекциях, но down_proj
/// с N<K всё ещё быстрее в W4A16. Начиная с B=5 повторный W4A16-проход
/// перечитал бы веса несколько раз, поэтому всегда выбирается W4A4.
pub fn select_decode_kernel(batch: usize, out_features: usize, in_features: usize) -> DecodeKernel {
    assert!((1..=MAX_W4A4_BATCH).contains(&batch));
    if batch <= 3 || (batch == 4 && out_features <= in_features) {
        DecodeKernel::W4A16
    } else {
        DecodeKernel::W4A4
    }
}

/// NVFP4-матрица в исходной раскладке compressed-tensors.
pub struct Linear {
    packed: DeviceBuffer<u8>,
    cutlass_scales: DeviceBuffer<u8>,
    weight_global_scale: f32,
    out_features: usize,
    in_features: usize,
}

impl Linear {
    /// Загружает матрицу без репака 4-битных весов и scale-матрицы.
    /// `weight_global_scale` — значение из checkpoint; там это делитель.
    pub fn from_host(
        packed: &[u8],
        scales: &[u8],
        weight_global_scale: f32,
        out_features: usize,
        in_features: usize,
    ) -> Result<Self> {
        assert_shape(
            packed.len(),
            scales.len(),
            weight_global_scale,
            out_features,
            in_features,
        );
        let cutlass_scales = swizzle_block_scales(scales, out_features, in_features);
        Ok(Self {
            packed: DeviceBuffer::from_slice(packed)?,
            cutlass_scales: DeviceBuffer::from_slice(&cutlass_scales)?,
            weight_global_scale,
            out_features,
            in_features,
        })
    }

    /// Матрица с нулевыми данными для bandwidth-бенчмарка реальных форм.
    pub fn zeroed(out_features: usize, in_features: usize) -> Result<Self> {
        assert_shape(
            out_features * in_features / 2,
            out_features * in_features / GROUP_SIZE,
            1.0,
            out_features,
            in_features,
        );
        Ok(Self {
            packed: DeviceBuffer::zeroed(out_features * in_features / 2)?,
            cutlass_scales: DeviceBuffer::zeroed(block_scale_storage_len(
                out_features,
                in_features,
            ))?,
            weight_global_scale: 1.0,
            out_features,
            in_features,
        })
    }

    /// Shared-weight W4A16 GEMV для decode batch 1..4.
    pub fn forward_w4a16(
        &self,
        input: &DeviceBuffer<u16>,
        output: &mut DeviceBuffer<u16>,
        batch: usize,
        stream: &Stream,
    ) -> Result<()> {
        assert!((1..=MAX_W4A16_BATCH).contains(&batch));
        assert!(input.len() >= batch * self.in_features);
        assert!(output.len() >= batch * self.out_features);
        check(unsafe {
            ffi::qwc_nvfp4_w4a16(
                self.packed.as_ptr(),
                self.cutlass_scales.as_ptr(),
                input.as_ptr(),
                output.as_mut_ptr(),
                self.out_features as i32,
                self.in_features as i32,
                batch as i32,
                self.weight_global_scale,
                stream.raw(),
            )
        })
    }

    pub fn out_features(&self) -> usize {
        self.out_features
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Полный DRAM-трафик весов, без активаций и выхода.
    pub fn weight_bytes(&self) -> usize {
        self.packed.bytes() + self.out_features * self.in_features / GROUP_SIZE
    }

    /// Реальная постоянная VRAM с учётом padding scale-layout.
    pub fn resident_bytes(&self) -> usize {
        self.packed.bytes() + self.cutlass_scales.bytes()
    }

    /// Копия упакованных весов обратно на хост — сверка загрузки с чекпоинтом
    /// без доступа к внутреннему буферу.
    pub fn packed_to_host(&self) -> Result<Vec<u8>> {
        self.packed.to_vec()
    }

    /// SM120 W4A4 tensor-core GEMM. Веса остаются в checkpoint layout;
    /// переставленная копия требуется только для 8-битных scale-факторов.
    pub fn forward_w4a4_quantized(
        &self,
        input: &QuantizedActivation,
        output: &mut DeviceBuffer<u16>,
        workspace: &mut W4A4Workspace,
        stream: &Stream,
    ) -> Result<()> {
        assert!((3..=MAX_W4A4_ROWS).contains(&input.batch));
        assert_eq!(input.in_features, self.in_features);
        assert!(output.len() >= input.batch * self.out_features);
        assert!(self.out_features.is_multiple_of(8));
        workspace.assert_shape(input.batch, self.out_features, self.in_features)?;

        let alpha = 1.0 / (input.global_scale * self.weight_global_scale);
        check(unsafe {
            ffi::qwc_nvfp4_w4a4(
                input.packed.as_ptr(),
                self.packed.as_ptr(),
                input.cutlass_scales.as_ptr(),
                self.cutlass_scales.as_ptr(),
                output.as_mut_ptr(),
                workspace.storage.as_mut_ptr(),
                workspace.bytes,
                input.batch as i32,
                self.out_features as i32,
                self.in_features as i32,
                alpha,
                stream.raw(),
            )
        })
    }
}

/// Уже квантованная decode-активация. Конструктор нужен также для тестирования
/// GEMM независимо от будущего fused BF16->NVFP4 quantizer.
pub struct QuantizedActivation {
    pub(crate) packed: DeviceBuffer<u8>,
    pub(crate) cutlass_scales: DeviceBuffer<u8>,
    pub(crate) global_scale: f32,
    pub(crate) batch: usize,
    capacity: usize,
    pub(crate) in_features: usize,
}

impl QuantizedActivation {
    /// `scales` передаются в логическом row-major `[batch, K/16]` порядке.
    pub fn from_host(
        packed: &[u8],
        scales: &[u8],
        global_scale: f32,
        batch: usize,
        in_features: usize,
    ) -> Result<Self> {
        assert!((1..=MAX_W4A4_ROWS).contains(&batch));
        assert!(in_features > 0 && in_features.is_multiple_of(K_ALIGNMENT));
        assert_eq!(packed.len(), batch * in_features / 2);
        assert_eq!(scales.len(), batch * in_features / GROUP_SIZE);
        assert!(global_scale.is_finite() && global_scale > 0.0);
        let cutlass_scales = swizzle_block_scales(scales, batch, in_features);
        Ok(Self {
            packed: DeviceBuffer::from_slice(packed)?,
            cutlass_scales: DeviceBuffer::from_slice(&cutlass_scales)?,
            global_scale,
            batch,
            capacity: batch,
            in_features,
        })
    }

    /// Создаёт переиспользуемые output-буферы quantizer. Padded scale slots
    /// обнуляются здесь один раз и остаются нулевыми между decode-шагами.
    pub fn zeroed(global_scale: f32, batch: usize, in_features: usize) -> Result<Self> {
        assert!((1..=MAX_W4A4_ROWS).contains(&batch));
        assert!(in_features > 0 && in_features.is_multiple_of(K_ALIGNMENT));
        assert!(global_scale.is_finite() && global_scale > 0.0);
        Ok(Self {
            packed: DeviceBuffer::zeroed(batch * in_features / 2)?,
            cutlass_scales: DeviceBuffer::zeroed(block_scale_storage_len(batch, in_features))?,
            global_scale,
            batch,
            capacity: batch,
            in_features,
        })
    }

    /// Квантует BF16 `[batch,K]` прямо в формат SM120 GEMM.
    pub fn quantize_bf16(&mut self, input: &DeviceBuffer<u16>, stream: &Stream) -> Result<()> {
        self.quantize_bf16_with_scale(input, self.global_scale, stream)
    }

    /// Reuses the same resident buffers with another checkpoint activation
    /// scale. Projections sharing an input shape generally have distinct
    /// `input_global_scale` values, so prefill can quantize into one scratch
    /// allocation instead of keeping a copy per projection.
    pub fn quantize_bf16_with_scale(
        &mut self,
        input: &DeviceBuffer<u16>,
        global_scale: f32,
        stream: &Stream,
    ) -> Result<()> {
        assert_eq!(input.len(), self.batch * self.in_features);
        assert!(global_scale.is_finite() && global_scale > 0.0);
        self.global_scale = global_scale;
        check(unsafe {
            ffi::qwc_nvfp4_quantize_bf16(
                input.as_ptr(),
                self.packed.as_mut_ptr(),
                self.cutlass_scales.as_mut_ptr(),
                self.batch as i32,
                self.in_features as i32,
                global_scale,
                stream.raw(),
            )
        })
    }

    /// Quantizes the live prefix of a capacity-sized activation buffer.
    pub fn quantize_bf16_rows(
        &mut self,
        input: &DeviceBuffer<u16>,
        global_scale: f32,
        rows: usize,
        stream: &Stream,
    ) -> Result<()> {
        assert!(rows > 0 && rows <= self.capacity);
        assert!(input.len() >= rows * self.in_features);
        assert!(global_scale.is_finite() && global_scale > 0.0);
        self.batch = rows;
        self.global_scale = global_scale;
        check(unsafe {
            ffi::qwc_nvfp4_quantize_bf16(
                input.as_ptr(),
                self.packed.as_mut_ptr(),
                self.cutlass_scales.as_mut_ptr(),
                rows as i32,
                self.in_features as i32,
                global_scale,
                stream.raw(),
            )
        })
    }

    /// Диагностический readback в логическом `[batch,K/16]` порядке.
    pub fn to_host_logical(&self) -> Result<(Vec<u8>, Vec<u8>)> {
        let packed = self.packed.to_vec()?;
        let swizzled = self.cutlass_scales.to_vec()?;
        let scales = unswizzle_block_scales(&swizzled, self.batch, self.in_features);
        Ok((packed, scales))
    }
}

/// Предвыделенный scratch CUTLASS. На decode hot path аллокаций нет.
pub struct W4A4Workspace {
    storage: DeviceBuffer<u8>,
    bytes: usize,
    max_batch: usize,
}

impl W4A4Workspace {
    pub fn new(batch: usize, out_features: usize, in_features: usize) -> Result<Self> {
        Self::for_shapes(batch, &[(out_features, in_features)])
    }

    /// One allocation large enough for every projection shape in a fixed-M
    /// prefill chunk. CUTLASS currently needs zero bytes for the selected
    /// schedule, but keeping the capacity check here makes that an
    /// implementation detail rather than an executor assumption.
    pub fn for_shapes(batch: usize, shapes: &[(usize, usize)]) -> Result<Self> {
        assert!(batch > 0);
        assert!(!shapes.is_empty());
        // Требование не монотонно по числу строк: узкая задача режется по K,
        // и чем меньше M, тем больше блоков участвует в редукции. Арена
        // предъявляет любое M до `batch`, поэтому берётся максимум по всем.
        let mut bytes = 0usize;
        for &(out_features, in_features) in shapes {
            for rows in 1..=batch {
                let mut needed = 0usize;
                check(unsafe {
                    ffi::qwc_nvfp4_w4a4_workspace_size(
                        rows as i32,
                        out_features as i32,
                        in_features as i32,
                        &mut needed,
                    )
                })?;
                bytes = bytes.max(needed);
            }
        }
        Ok(Self {
            // cudaMalloc(0) не имеет переносимой семантики; указатель при
            // нулевом workspace всё равно не разыменовывается CUTLASS.
            storage: DeviceBuffer::zeroed(bytes.max(1))?,
            bytes,
            max_batch: batch,
        })
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    fn assert_shape(&self, batch: usize, out_features: usize, in_features: usize) -> Result<()> {
        assert!(batch > 0 && batch <= self.max_batch);
        let mut needed = 0usize;
        check(unsafe {
            ffi::qwc_nvfp4_w4a4_workspace_size(
                batch as i32,
                out_features as i32,
                in_features as i32,
                &mut needed,
            )
        })?;
        assert!(needed <= self.bytes, "W4A4 workspace capacity is too small");
        Ok(())
    }
}

/// Elementwise `silu(gate) * up` for the two W4A4 MLP projections.
pub fn swiglu_bf16(
    gate: &DeviceBuffer<u16>,
    up: &DeviceBuffer<u16>,
    output: &mut DeviceBuffer<u16>,
    batch: usize,
    features: usize,
    stream: &Stream,
) -> Result<()> {
    assert!(batch > 0 && features > 0);
    assert!(gate.len() >= batch * features);
    assert!(up.len() >= batch * features);
    assert!(output.len() >= batch * features);
    check(unsafe {
        ffi::qwc_swiglu_bf16(
            gate.as_ptr(),
            up.as_ptr(),
            output.as_mut_ptr(),
            (batch * features) as i32,
            stream.raw(),
        )
    })
}

fn round_up(value: usize, multiple: usize) -> usize {
    value.div_ceil(multiple) * multiple
}

fn block_scale_storage_len(rows: usize, in_features: usize) -> usize {
    round_up(rows, 128) * round_up(in_features / GROUP_SIZE, 4)
}

/// compressed-tensors `[row, K/16]` -> SM120 block-scale 128x4 swizzle.
pub fn swizzle_block_scales(scales: &[u8], rows: usize, in_features: usize) -> Vec<u8> {
    assert_eq!(scales.len(), rows * in_features / GROUP_SIZE);
    let groups = in_features / GROUP_SIZE;
    let group_blocks = round_up(groups, 4) / 4;
    let mut output = vec![0u8; block_scale_storage_len(rows, in_features)];

    for row in 0..rows {
        let row_block = row / 128;
        let row_in_block = row % 128;
        let row_quad = row_in_block / 32;
        let row_in_quad = row_in_block % 32;
        for group in 0..groups {
            let group_block = group / 4;
            let group_in_block = group % 4;
            let dst = ((((row_block * group_blocks + group_block) * 32 + row_in_quad) * 4
                + row_quad)
                * 4)
                + group_in_block;
            output[dst] = scales[row * groups + group];
        }
    }
    output
}

fn unswizzle_block_scales(scales: &[u8], rows: usize, in_features: usize) -> Vec<u8> {
    assert_eq!(scales.len(), block_scale_storage_len(rows, in_features));
    let groups = in_features / GROUP_SIZE;
    let group_blocks = round_up(groups, 4) / 4;
    let mut output = vec![0u8; rows * groups];
    for row in 0..rows {
        let row_block = row / 128;
        let row_in_block = row % 128;
        let row_quad = row_in_block / 32;
        let row_in_quad = row_in_block % 32;
        for group in 0..groups {
            let dst =
                ((((row_block * group_blocks + group / 4) * 32 + row_in_quad) * 4 + row_quad) * 4)
                    + group % 4;
            output[row * groups + group] = scales[dst];
        }
    }
    output
}

/// Fused `silu(gate(x)) * up(x)` для Qwen MLP decode.
pub fn swiglu_w4a16(
    gate: &Linear,
    up: &Linear,
    input: &DeviceBuffer<u16>,
    output: &mut DeviceBuffer<u16>,
    batch: usize,
    stream: &Stream,
) -> Result<()> {
    assert_eq!(gate.out_features, up.out_features);
    assert_eq!(gate.in_features, up.in_features);
    assert!((1..=MAX_W4A16_BATCH).contains(&batch));
    assert!(input.len() >= batch * gate.in_features);
    assert!(output.len() >= batch * gate.out_features);
    check(unsafe {
        ffi::qwc_nvfp4_swiglu_w4a16(
            gate.packed.as_ptr(),
            gate.cutlass_scales.as_ptr(),
            up.packed.as_ptr(),
            up.cutlass_scales.as_ptr(),
            input.as_ptr(),
            output.as_mut_ptr(),
            gate.out_features as i32,
            gate.in_features as i32,
            batch as i32,
            gate.weight_global_scale,
            up.weight_global_scale,
            stream.raw(),
        )
    })
}

fn assert_shape(
    packed_len: usize,
    scale_len: usize,
    weight_global_scale: f32,
    out_features: usize,
    in_features: usize,
) {
    assert!(out_features > 0);
    assert!(in_features > 0 && in_features.is_multiple_of(K_ALIGNMENT));
    assert_eq!(packed_len, out_features * in_features / 2);
    assert_eq!(scale_len, out_features * in_features / GROUP_SIZE);
    assert!(weight_global_scale.is_finite() && weight_global_scale > 0.0);
}

/// CPU-oracle для bit-exact проверки layout и численной проверки GPU.
pub mod reference {
    use super::*;

    const E2M1: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];

    pub fn e2m1(raw: u8) -> f32 {
        E2M1[(raw & 0x0f) as usize]
    }

    /// NVIDIA E4M3 finite (без infinity; 0x7f/0xff — NaN).
    pub fn e4m3(raw: u8) -> f32 {
        let sign = if raw & 0x80 == 0 { 1.0 } else { -1.0 };
        let exponent = (raw >> 3) & 0x0f;
        let mantissa = raw & 0x07;
        if exponent == 0 {
            return sign * (mantissa as f32) * 2.0f32.powi(-9);
        }
        if exponent == 0x0f && mantissa == 0x07 {
            return f32::NAN;
        }
        sign * (1.0 + mantissa as f32 / 8.0) * 2.0f32.powi(exponent as i32 - 7)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gemv_w4a16(
        packed: &[u8],
        scales: &[u8],
        weight_global_scale: f32,
        input: &[u16],
        output: &mut [f32],
        out_features: usize,
        in_features: usize,
        batch: usize,
    ) {
        assert_shape(
            packed.len(),
            scales.len(),
            weight_global_scale,
            out_features,
            in_features,
        );
        assert_eq!(input.len(), batch * in_features);
        assert_eq!(output.len(), batch * out_features);

        for b in 0..batch {
            for row in 0..out_features {
                let mut acc = 0.0f32;
                for group in 0..in_features / GROUP_SIZE {
                    let scale = e4m3(scales[row * (in_features / GROUP_SIZE) + group])
                        / weight_global_scale;
                    let mut partial = 0.0f32;
                    for j in 0..GROUP_SIZE {
                        let k = group * GROUP_SIZE + j;
                        let byte = packed[row * (in_features / 2) + k / 2];
                        let nibble = if k & 1 == 0 { byte & 0x0f } else { byte >> 4 };
                        partial += e2m1(nibble) * bf16::to_f32(input[b * in_features + k]);
                    }
                    acc += partial * scale;
                }
                output[b * out_features + row] = acc;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gemm_w4a4(
        packed_input: &[u8],
        input_scales: &[u8],
        input_global_scale: f32,
        packed_weight: &[u8],
        weight_scales: &[u8],
        weight_global_scale: f32,
        output: &mut [f32],
        batch: usize,
        out_features: usize,
        in_features: usize,
    ) {
        assert_eq!(packed_input.len(), batch * in_features / 2);
        assert_eq!(input_scales.len(), batch * in_features / GROUP_SIZE);
        assert_shape(
            packed_weight.len(),
            weight_scales.len(),
            weight_global_scale,
            out_features,
            in_features,
        );
        assert_eq!(output.len(), batch * out_features);
        let groups = in_features / GROUP_SIZE;
        let alpha = 1.0 / (input_global_scale * weight_global_scale);

        for b in 0..batch {
            for row in 0..out_features {
                let mut acc = 0.0f32;
                for group in 0..groups {
                    let scale = e4m3(input_scales[b * groups + group])
                        * e4m3(weight_scales[row * groups + group]);
                    let mut partial = 0.0f32;
                    for j in 0..GROUP_SIZE {
                        let k = group * GROUP_SIZE + j;
                        let a_byte = packed_input[b * (in_features / 2) + k / 2];
                        let w_byte = packed_weight[row * (in_features / 2) + k / 2];
                        let a = if k & 1 == 0 {
                            a_byte & 0x0f
                        } else {
                            a_byte >> 4
                        };
                        let w = if k & 1 == 0 {
                            w_byte & 0x0f
                        } else {
                            w_byte >> 4
                        };
                        partial += e2m1(a) * e2m1(w);
                    }
                    acc += partial * scale;
                }
                output[b * out_features + row] = acc * alpha;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::reference::{e2m1, e4m3};

    #[test]
    fn decodes_e2m1_table() {
        assert_eq!(e2m1(0x0), 0.0);
        assert_eq!(e2m1(0x1), 0.5);
        assert_eq!(e2m1(0x7), 6.0);
        assert_eq!(e2m1(0xf), -6.0);
    }

    #[test]
    fn decodes_e4m3_boundaries() {
        assert_eq!(e4m3(0x00), 0.0);
        assert_eq!(e4m3(0x01), 2.0f32.powi(-9));
        assert_eq!(e4m3(0x38), 1.0);
        assert_eq!(e4m3(0x7e), 448.0);
        assert_eq!(e4m3(0xb8), -1.0);
        assert!(e4m3(0x7f).is_nan());
    }

    #[test]
    fn swizzles_128_by_4_scale_tiles() {
        let rows = 129;
        let k = 256;
        let groups = k / super::GROUP_SIZE;
        let input: Vec<u8> = (0..rows * groups).map(|i| i as u8).collect();
        let got = super::swizzle_block_scales(&input, rows, k);
        assert_eq!(got.len(), 256 * groups);

        let index = |row: usize, group: usize| {
            let rb = row / 128;
            let rq = row % 128 / 32;
            let ri = row % 32;
            let gb = group / 4;
            let gi = group % 4;
            ((((rb * (groups / 4) + gb) * 32 + ri) * 4 + rq) * 4) + gi
        };
        for &(row, group) in &[(0, 0), (31, 3), (32, 4), (127, 15), (128, 7)] {
            assert_eq!(got[index(row, group)], input[row * groups + group]);
        }
    }

    #[test]
    fn decode_dispatch_uses_measured_shape_boundary() {
        assert_eq!(
            super::select_decode_kernel(3, 17_408, 5_120),
            super::DecodeKernel::W4A16
        );
        assert_eq!(
            super::select_decode_kernel(4, 17_408, 5_120),
            super::DecodeKernel::W4A4
        );
        assert_eq!(
            super::select_decode_kernel(4, 5_120, 17_408),
            super::DecodeKernel::W4A16
        );
        assert_eq!(
            super::select_decode_kernel(5, 5_120, 17_408),
            super::DecodeKernel::W4A4
        );
    }
}

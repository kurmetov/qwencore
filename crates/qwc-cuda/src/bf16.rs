//! Минимальный bf16 без внешних зависимостей.
//! bf16 — это старшие 16 бит f32, поэтому преобразование тривиально;
//! важно лишь округлять так же, как `__float2bfloat16` на GPU (к ближайшему чётному).

pub fn from_f32(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        return 0x7fc0;
    }
    let lsb = (bits >> 16) & 1;
    ((bits + 0x7fff + lsb) >> 16) as u16
}

pub fn to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

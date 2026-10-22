//! Minimal bf16 without external dependencies.
//! bf16 is the upper 16 bits of an f32, so conversion is trivial;
//! what matters is rounding the same way as `__float2bfloat16` on the GPU (round to nearest even).

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

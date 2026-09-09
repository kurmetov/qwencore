//! Свойства GPU и развёртка достижимой пропускной способности.
//! `cargo run -p qwc-cuda --bin gpuinfo`

use qwc_cuda::{Device, bandwidth};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let d = Device::init(0)?;
    let (free, total) = Device::mem_info()?;

    println!("Устройство");
    println!("  SM                       {}", d.sm_count);
    println!("  частота SM               {:.2} GHz", d.sm_clock_hz / 1e9);
    println!("  шина памяти              {} bit @ {:.1} Gbps", d.memory_bus_bits, d.memory_clock_hz * 2.0 / 1e9);
    println!("  паспортный пик           {:.0} GB/s", d.peak_bandwidth() / 1e9);
    println!("  L2                       {:.1} MiB", d.l2_bytes as f64 / (1024.0 * 1024.0));
    println!("  shared mem / блок optin  {:.1} KB", d.max_shared_mem_optin as f64 / 1024.0);
    println!("  регистров / SM           {}", d.max_registers_per_sm);
    println!("  VRAM свободно            {:.2} / {:.2} GB", free as f64 / 1e9, total as f64 / 1e9);

    let l2 = d.l2_bytes as f64;
    println!("\nДостижимая пропускная способность (чтение)");
    println!("  {:>9} | {:>10} | {:>7} | {}", "буфер", "GB/s", "% пика", "");
    println!("  {:->9}-+-{:->10}-+-{:->7}-+-", "", "", "");
    for mb in [8usize, 32, 64, 96, 128, 256, 1024, 4096] {
        let bytes = mb * 1024 * 1024;
        let iters = if mb <= 128 { 200 } else { 30 };
        let m = bandwidth::read(bytes, iters)?;
        let marker = if (bytes as f64) < l2 { "помещается в L2" } else { "" };
        println!(
            "  {:>6} MiB | {:>10.0} | {:>6.0}% | {}",
            mb,
            m.gb_per_sec,
            100.0 * m.gb_per_sec * 1e9 / d.peak_bandwidth(),
            marker
        );
    }

    println!("\nДостижимая пропускная способность (копирование, чтение+запись)");
    for mb in [64usize, 1024, 4096] {
        let bytes = mb * 1024 * 1024;
        let m = bandwidth::copy(bytes, if mb <= 128 { 200 } else { 30 })?;
        println!(
            "  {:>6} MiB | {:>10.0} GB/s | {:>6.0}% пика",
            mb,
            m.gb_per_sec,
            100.0 * m.gb_per_sec * 1e9 / d.peak_bandwidth()
        );
    }
    Ok(())
}

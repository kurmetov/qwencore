//! Сборка CUDA-кернелов.
//!
//! Никаких build-зависимостей: nvcc вызывается напрямую. Целевая архитектура
//! одна — sm_120a (RTX 5090), это часть специализации движка, а не настройка.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, fs};

/// Единственная поддерживаемая архитектура. `a` — вариант с
/// архитектурно-специфичными инструкциями, без него недоступен
/// block-scaled MMA для NVFP4.
const ARCH: &str = "sm_120a";

fn main() {
    println!("cargo:rerun-if-changed=cuda");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");

    let cuda = cuda_root();
    let nvcc = cuda.join("bin/nvcc");
    assert!(
        nvcc.exists(),
        "не найден nvcc в {}. Задайте CUDA_HOME.",
        cuda.display()
    );

    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let mut includes: Vec<PathBuf> = Vec::new();

    // Баг CUDA 13.1 + glibc >= 2.41: конфликт объявлений rsqrt.
    // Проверяем компиляцией, а не версией, чтобы обход сам отключился
    // на CUDA >= 13.3, где NVIDIA это исправила.
    if !probe_compiles(&nvcc, &out, &[]) {
        let shim = out.join("cuda-shim");
        generate_shim(&cuda, &shim);
        assert!(
            probe_compiles(&nvcc, &out, &[shim.clone()]),
            "не удалось обойти конфликт заголовков CUDA/glibc даже с шимом"
        );
        println!("cargo:warning=применён обход бага CUDA/glibc (см. docs/02-toolchain.md)");
        includes.push(shim);
    }

    let sources: Vec<PathBuf> = fs::read_dir("cuda")
        .expect("нет каталога cuda/")
        .filter_map(|e| {
            let p = e.ok()?.path();
            (p.extension()? == "cu").then_some(p)
        })
        .collect();
    assert!(!sources.is_empty(), "в cuda/ нет ни одного .cu");

    let lib = out.join("libqwc_kernels.a");
    let mut cmd = Command::new(&nvcc);
    cmd.args(["-arch", ARCH])
        .args(["-std", "c++17"])
        .arg("-O3")
        .arg("--lib")
        .arg("-Xcompiler")
        .arg("-fPIC")
        // 20012/20014: __host__/__device__ на defaulted-функциях в CUTLASS.
        .args(["-diag-suppress", "20012,20014"])
        .arg("--expt-relaxed-constexpr");
    for inc in &includes {
        cmd.arg("-I").arg(inc);
    }
    cmd.arg("-o").arg(&lib).args(&sources);

    let status = cmd.status().expect("не удалось запустить nvcc");
    assert!(status.success(), "nvcc завершился с ошибкой");

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=qwc_kernels");
    println!("cargo:rustc-link-search=native={}", cuda.join("lib64").display());
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

fn cuda_root() -> PathBuf {
    if let Ok(h) = env::var("CUDA_HOME") {
        return PathBuf::from(h);
    }
    for c in ["/usr/local/cuda", "/usr/local/cuda-13.1"] {
        if Path::new(c).join("bin/nvcc").exists() {
            return PathBuf::from(c);
        }
    }
    panic!("не найден CUDA toolkit; задайте CUDA_HOME");
}

/// Компилирует пробник, который включает проблемные заголовки.
fn probe_compiles(nvcc: &Path, out: &Path, includes: &[PathBuf]) -> bool {
    let src = out.join("probe.cu");
    fs::write(
        &src,
        "#include <cstdio>\n#include <mutex>\n#include <cuda_runtime.h>\n__global__ void p(){}\n",
    )
    .unwrap();
    let mut cmd = Command::new(nvcc);
    cmd.args(["-arch", ARCH]).args(["-std", "c++17"]).arg("-c");
    for inc in includes {
        cmd.arg("-I").arg(inc);
    }
    cmd.arg("-o").arg(out.join("probe.o")).arg(&src);
    cmd.output().map(|o| o.status.success()).unwrap_or(false)
}

/// Теневое дерево include: симлинки на настоящий toolkit плюс одна
/// пропатченная копия crt/math_functions.h. Работает потому, что CUDA
/// подключает его как "crt/math_functions.h" — относительно каталога
/// включающего файла, а он здесь наш.
fn generate_shim(cuda: &Path, shim: &Path) {
    let inc = cuda.join("targets/x86_64-linux/include");
    let _ = fs::remove_dir_all(shim);
    fs::create_dir_all(shim.join("crt")).unwrap();

    for entry in fs::read_dir(&inc).unwrap().flatten() {
        let name = entry.file_name();
        if name == "crt" {
            continue;
        }
        let _ = std::os::unix::fs::symlink(entry.path(), shim.join(&name));
    }
    for entry in fs::read_dir(inc.join("crt")).unwrap().flatten() {
        let name = entry.file_name();
        if name == "math_functions.h" {
            continue;
        }
        let _ = std::os::unix::fs::symlink(entry.path(), shim.join("crt").join(&name));
    }

    let orig = fs::read_to_string(inc.join("crt/math_functions.h")).unwrap();
    let patched = orig
        .replace("double                 rsqrt(double x);", "double                 rsqrt(double x) noexcept;")
        .replace("float                  rsqrtf(float x);", "float                  rsqrtf(float x) noexcept;");
    assert!(
        patched.contains("rsqrt(double x) noexcept;"),
        "патч не применился: заголовок CUDA изменился"
    );
    fs::write(shim.join("crt/math_functions.h"), patched).unwrap();
}

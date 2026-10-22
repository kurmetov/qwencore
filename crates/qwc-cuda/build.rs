//! Building the CUDA kernels.
//!
//! No build dependencies: nvcc is invoked directly. There is one target architecture
//! — sm_120a (RTX 5090); it's part of the engine's specialization, not a setting.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, fs};

/// The only supported architecture. `a` is the variant with
/// architecture-specific instructions; without it
/// block-scaled MMA for NVFP4 is unavailable.
const ARCH: &str = "sm_120a";

fn main() {
    println!("cargo:rerun-if-changed=cuda");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");

    let cuda = cuda_root();
    let nvcc = cuda.join("bin/nvcc");
    assert!(
        nvcc.exists(),
        "nvcc not found in {}. Set CUDA_HOME.",
        cuda.display()
    );

    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let mut includes: Vec<PathBuf> = Vec::new();

    // CUTLASS is used only as a header-only SM120 MMA generator.
    // At runtime it adds neither dependencies nor hidden allocations.
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let cutlass = manifest.join("../../third_party/cutlass");
    assert!(
        cutlass.join("include/cutlass/cutlass.h").exists(),
        "vendored CUTLASS not found in {}",
        cutlass.display()
    );
    includes.push(cutlass.join("include"));
    includes.push(cutlass.join("tools/util/include"));

    // CUDA 13.1 + glibc >= 2.41 bug: conflicting rsqrt declarations.
    // Detected by compiling, not by version, so the workaround turns itself off
    // on CUDA >= 13.3, where NVIDIA fixed it.
    if !probe_compiles(&nvcc, &out, &[]) {
        let shim = out.join("cuda-shim");
        generate_shim(&cuda, &shim);
        assert!(
            probe_compiles(&nvcc, &out, std::slice::from_ref(&shim)),
            "failed to work around the CUDA/glibc header conflict even with the shim"
        );
        println!("cargo:warning=applied CUDA/glibc bug workaround (see docs/02-toolchain.md)");
        includes.push(shim);
    }

    let sources: Vec<PathBuf> = fs::read_dir("cuda")
        .expect("no cuda/ directory")
        .filter_map(|e| {
            let p = e.ok()?.path();
            (p.extension()? == "cu").then_some(p)
        })
        .collect();
    assert!(!sources.is_empty(), "no .cu files in cuda/");

    let lib = out.join("libqwc_kernels.a");
    let mut cmd = Command::new(&nvcc);
    cmd.args(["-arch", ARCH])
        .args(["-std", "c++17"])
        .arg("-O3")
        .arg("--lib")
        .arg("-Xcompiler")
        .arg("-fPIC")
        // 20012/20014: __host__/__device__ on defaulted functions in CUTLASS.
        .args(["-diag-suppress", "20012,20014"])
        .arg("--expt-relaxed-constexpr");
    for inc in &includes {
        cmd.arg("-I").arg(inc);
    }
    cmd.arg("-o").arg(&lib).args(&sources);

    let status = cmd.status().expect("failed to launch nvcc");
    assert!(status.success(), "nvcc failed");

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=qwc_kernels");
    println!(
        "cargo:rustc-link-search=native={}",
        cuda.join("lib64").display()
    );
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
    panic!("CUDA toolkit not found; set CUDA_HOME");
}

/// Compiles a probe that includes the problematic headers.
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

/// A shadow include tree: symlinks to the real toolkit plus one
/// patched copy of crt/math_functions.h. Works because CUDA
/// includes it as "crt/math_functions.h" — relative to the directory
/// of the including file, which here is ours.
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
        .replace(
            "double                 rsqrt(double x);",
            "double                 rsqrt(double x) noexcept;",
        )
        .replace(
            "float                  rsqrtf(float x);",
            "float                  rsqrtf(float x) noexcept;",
        );
    assert!(
        patched.contains("rsqrt(double x) noexcept;"),
        "patch didn't apply: the CUDA header changed"
    );
    fs::write(shim.join("crt/math_functions.h"), patched).unwrap();
}

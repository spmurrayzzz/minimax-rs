use cudaforge::{KernelBuilder, Result};
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/compatibility.cuh");
    println!("cargo::rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo::rerun-if-changed=src/binary_op_macros.cuh");

    // Build for PTX
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("ptx.rs");
    let bindings = KernelBuilder::new()
        .source_dir("src") // Scan src/ for .cu files
        .exclude(&[
            "moe_*.cu",
            "mmvq_gguf.cu",
            "mmq_*.cu",
            "decode_attention.cu",
        ]) // Exclude statically compiled kernels from ptx build
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3")
        .build_ptx()?;

    bindings.write(&ptx_path)?;
    // CUDA 13.3 emits PTX 9.3, while the box driver advertises CUDA 13.2.
    // Keep the kernels restricted to the sm_120 instruction set and lower the
    // PTX declaration so the 595 driver can load them.
    for entry in std::fs::read_dir(&out_dir)? {
        let path = entry?.path();
        if path.extension().is_some_and(|e| e == "ptx") {
            let text = std::fs::read_to_string(&path)?
                .replace(".version 9.3", ".version 9.2")
                .replace(".target sm_120a", ".target sm_120")
                .replace(".target sm_120f", ".target sm_120");
            std::fs::write(path, text)?;
        }
    }

    let mut moe_builder = KernelBuilder::default()
        .source_files(vec![
            "src/moe/moe_gguf.cu",
            "src/moe/moe_wmma.cu",
            "src/moe/moe_wmma_gguf.cu",
            "src/mmvq_gguf.cu",
            "src/decode_attention.cu",
            "src/mmq_gguf/mmq_quantize.cu",
            "src/mmq_gguf/mmq_instance_q4_0.cu",
            "src/mmq_gguf/mmq_instance_q4_1.cu",
            "src/mmq_gguf/mmq_instance_q5_0.cu",
            "src/mmq_gguf/mmq_instance_q5_1.cu",
            "src/mmq_gguf/mmq_instance_q8_0.cu",
            "src/mmq_gguf/mmq_instance_q2_k.cu",
            "src/mmq_gguf/mmq_instance_q3_k.cu",
            "src/mmq_gguf/mmq_instance_q4_k.cu",
            "src/mmq_gguf/mmq_instance_q5_k.cu",
            "src/mmq_gguf/mmq_instance_q6_k.cu",
        ])
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");

    // Disable bf16 WMMA kernels on GPUs older than sm_80 (Ampere).
    // bf16 WMMA fragments require compute capability >= 8.0.
    let compute_cap = cudaforge::detect_compute_cap()
        .map(|arch| arch.base())
        .unwrap_or(80);
    if compute_cap < 80 {
        moe_builder = moe_builder.arg("-DNO_BF16_KERNEL");
    }

    let mut is_target_msvc = false;
    if let Ok(target) = std::env::var("TARGET") {
        if target.contains("msvc") {
            is_target_msvc = true;
            moe_builder = moe_builder.arg("-D_USE_MATH_DEFINES");
        }
    }

    if !is_target_msvc {
        moe_builder = moe_builder.arg("-Xcompiler").arg("-fPIC");
    }

    moe_builder.build_lib(out_dir.join("libmoe.a"))?;
    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib=moe");
    println!("cargo:rustc-link-lib=dylib=cudart");
    if !is_target_msvc {
        println!("cargo:rustc-link-lib=stdc++");
    }
    Ok(())
}

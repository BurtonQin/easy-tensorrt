use regex::Regex;
use std::env;
use std::path::PathBuf;
use std::process::Command;

fn find_dir(
    env_key: &'static str,
    candidates: Vec<&'static str>,
    file_to_find: &'static str,
) -> Option<PathBuf> {
    match env::var_os(env_key) {
        Some(val) => Some(PathBuf::from(&val)),
        _ => {
            for candidate in candidates {
                let path = PathBuf::from(candidate);
                let file_path = path.join(file_to_find);
                if file_path.exists() {
                    return Some(path);
                }
            }

            None
        }
    }
}

fn extract_tensorrt_version(output: &str) -> Option<String> {
    let re = Regex::new(r"\[TensorRT (v\d+)\]").unwrap();
    // Find the first capture group (version)
    re.captures(output)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().to_string())
}

fn main() {
    let output = Command::new("trtexec")
        .arg("--help")
        .output()
        .expect("Failed to execute trtexec command. Is TensorRT installed and in PATH?");
    let output_str = String::from_utf8_lossy(&output.stdout);
    let first_line = output_str.lines().next().unwrap_or("");
    let version = extract_tensorrt_version(first_line)
        .expect("Failed to extract TensorRT version from trtexec output");
    let (dll_name, flag) = if std::env::var_os("CARGO_CFG_WINDOWS").is_some() {
        ("libnvinfer.dll", "/std:c++17")
    } else {
        ("libnvinfer.so", "-std=c++17")
    };
    let cuda_include_dir = find_dir(
        "CUDA_INCLUDE_PATH",
        vec!["/opt/cuda/include", "/usr/local/cuda/include"],
        "cuda.h",
    )
    .expect("Could not find CUDA include path");

    let tensorrt_include_dir = find_dir(
        "TENSORRT_INCLUDE_PATH",
        vec!["/usr/local/include", "/usr/include/x86_64-linux-gnu"],
        "NvInfer.h",
    )
    .expect("Could not find TensorRT include path");

    let tensorrt_library_dir = find_dir(
        "TENSORRT_LIB_PATH",
        vec!["/usr/local/lib", "/usr/lib/x86_64-linux-gnu"],
        dll_name,
    )
    .expect("Could not find TensorRT library path");

    let include_files = vec!["cxx/include/logger.h", "cxx/include/runtime.h"];
    let cpp_files = vec!["cxx/src/logger.cpp", "cxx/src/runtime.cpp"];
    let rust_files = vec!["src/lib.rs"];

    cxx_build::bridges(&rust_files)
        .include(cuda_include_dir)
        .include(tensorrt_include_dir)
        .include("cxx/include")
        .files(&cpp_files)
        .define("FMT_HEADER_ONLY", None)
        .flag_if_supported(flag)
        .compile("easy-tensorrt-sys-cxxbridge");

    println!(
        "cargo:rustc-link-search={}",
        tensorrt_library_dir.to_string_lossy()
    );

    println!("cargo:rustc-env=TENSORRT_VERSION={}", version);

    let libraries = match version.as_str() {
        "v8601" => vec!["nvinfer", "nvinfer_plugin", "nvparsers"],
        "v101000" => vec!["nvinfer_10", "nvinfer_plugin_10"],
        _ => panic!("Unsupported TensorRT version: {}", version),
    };

    for library in libraries {
        println!("cargo:rustc-link-lib={}", library);
    }

    for file in include_files {
        println!("cargo:rerun-if-changed={}", file);
    }

    for file in cpp_files {
        println!("cargo:rerun-if-changed={}", file);
    }

    for file in rust_files {
        println!("cargo:rerun-if-changed={}", file);
    }
}

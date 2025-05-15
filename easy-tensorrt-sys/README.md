# easy-tensorrt-sys

Rust binding to [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt).

This is a fork of [tensorrt-rs-sys](https://github.com/vivym/tensorrt-rs/tree/main/tensorrt-rs-sys), which seems no longer activately maintained.

## What's new

1. Replaced the dependency [cuda-rs](https://github.com/vivym/cuda-rs) with [cudarc](https://github.com/coreylowman/cudarc) for cuda interaction, which is actively maintained and offers better compatibility.
2. Add `init_libnv_infer_plugins` to address initialization issues with TensorRT plugins.

## Installation

To use `easy-tensorrt-sys`, ensure you have the following installed:

- **CUDA Toolkit** (e.g., CUDA 11.8)
- **TensorRT SDK** (e.g., TensorRT 8.6.1.6)
- **Rust** (with `cargo`)

### Linux

1. Install CUDA and TensorRT (e.g., via `.deb` packages or tarballs).
2. Set environment variables if your install paths are non-standard:

```bash
export CUDA_INCLUDE_PATH=/usr/local/cuda/include
export TENSORRT_INCLUDE_PATH=/usr/local/TensorRT-8.6.1.6/include
export TENSORRT_LIB_PATH=/usr/local/TensorRT-8.6.1.6/lib
```

### Windows

Install CUDA and TensorRT from NVIDIA’s website.

Set environment variables:

```powershell

$env:CUDA_INCLUDE_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include"
$env:TENSORRT_INCLUDE_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT\include"
$env:TENSORRT_LIB_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT\lib"
```

## Notes

- Tested on Linux, with TensorRT 8.6.1.6, CUDA 11.8.

- Other platforms or versions are not guaranteed to work.

## Contributions

Contributions are welcome! Feel free to open issues or submit pull requests.

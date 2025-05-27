# easy-tensorrt

Seamless TensorRT integration with PyTorch or cudarc in Rust

## Installation

See [easy-tensorrt-sys/README](easy-tensorrt-sys/README.md) for installation.

For PyTorch integration, see [easy-tensorrt-torch/README](easy-tensorrt-torch/README.md) for installation.

## easy-tensort-sys

A binding for TensorRT's C++ API. Forked from [tensorrt-rs-sys](https://github.com/vivym/tensorrt-rs/tree/main/tensorrt-rs-sys)

Major changes include:

1. Replace cuda-rs with cudarc to interact with CUDA APIs directly.
2. Add binding to support TensorRT plugin initialization.

See the README in easy-tensorrt-sys for more details.

## easy-tensorrt-core

A Rust wrapper for easy-tensorrt-sys. Reference implementation in [tensorrt-rs](https://github.com/vivym/tensorrt-rs)

Major changes include:

1. Replace Tensor with AbstractTensor trait.
2. Add Init module, changes to APIs.

See the README in easy-tensorrt-core for more details.

See the examples in easy-tensorrt-core for usage.

## easy-tensorrt-torch

Integrate TensorRT with PyTorch Tensor. Call TensorRT APIs from Python.

Note:

To avoid binding conflicts in different versions of torch-sys, I have to separately build easy-tensorrt-tch2x0 for each PyTorch version (2.0.x to 2.7.x).

Because I tried to use features to conditionally compile the correct tch version for different PyTorch versions, but Cargo does not support it.

Implement AbstractTensor trait for newtyped PyTorch Tensor.

See python/test.py for usage.

See the README in easy-tensorrt-torch for more details.
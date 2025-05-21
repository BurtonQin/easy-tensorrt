# easy-tensorrt-core

A safe wrapper around easy-tensorrt-sys.
Provide the core functions to build and run TensorRT engine.

Inspired by [tensorrt-rs](https://github.com/vivym/tensorrt-rs).


## Major changes from tensorrt-rs:

1. Replace cuda-rs with cudarc.
2. Replace Tensor with AbstractTensor trait. Any tensor (e.g. tch::Tensor) that implements AbstractTensor can be used.
3. Add init module to faciitate the initialization of tensorrt plugins and cuda context.
4. Init engine with CuStream and avoid passing in another CuStream when inference.
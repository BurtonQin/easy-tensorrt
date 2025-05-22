# easy-tensorrt-tch200

Integrate TensorRT with PyTorch 2.3.x

## Installation

### 1. Set up your env vars for tch-rs:

```bash
export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1
```

Add the above commands to your env var (e.g., ~/.bashrc):

### 2. Create a conda env YOUR_ENV

Install PyTorch 2.3.x with CUDA support.

I have not tested with PyTorch 2.3.x.

### 3. Install Rust and Maturin

Install Rust from [rust-lang](https://www.rust-lang.org)

Install maturin:

```bash
conda activate YOUR_ENV
pip install maturin
```

### 4. Build and install the package

```bash
cd easy-tensorrt-tch230
conda activate YOUR_ENV
maturin develop --release
```

## Rust Example

Run the example in Rust.
The python script sets up the env var before calling the examples/tch_trt_engine.rs
```bash
python scripts/run_example.py
```

## Python Example

Run the example in Python.
```bash
python scripts/test.py
```


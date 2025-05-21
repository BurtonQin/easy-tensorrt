import os
import torch

# Check if CUDA is available
if not torch.cuda.is_available():
    raise RuntimeError("easy-tensorrt-tch240 requires a CUDA-enabled PyTorch installation.")

# Handle DLL path for TensorRT on Windows
if os.name == "nt":
    tensorrt_lib_path = os.getenv("TENSORRT_LIB_PATH")
    if tensorrt_lib_path is not None:
        os.add_dll_directory(tensorrt_lib_path)
    else:
        print("TENSORRT_LIB_PATH is not set, please set it to the path of TensorRT libs.")

# Import the compiled module
from .easy_tensorrt_tch240 import *

# Forward module-level docstring and __all__ if present
import easy_tensorrt_tch240 as _mod

__doc__ = _mod.__doc__
if hasattr(_mod, "__all__"):
    __all__ = _mod.__all__
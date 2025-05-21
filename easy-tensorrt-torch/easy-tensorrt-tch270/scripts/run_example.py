#!/usr/bin/env python3
import os
import sys
import subprocess

def main():
    # Get python executable from environment or fallback
    python = os.environ.get("PYTHON_SYS_EXECUTABLE", sys.executable)

    # Query LIBDIR and Python version
    libdir, pyver = subprocess.check_output([
        python, "-c",
        "import sysconfig, sys; "
        "print(sysconfig.get_config_var('LIBDIR')); "
        "print(f'python{sys.version_info.major}.{sys.version_info.minor}')"
    ], text=True).splitlines()

    # Try to locate PyTorch's lib directory (where libtorch_cuda.so lives)
    torch_lib_dir = subprocess.check_output([
        python, "-c",
        "import torch, os; "
        "print(os.path.join(torch.__path__[0], 'lib'))"
    ], text=True).strip()


    # Set environment variables
    old_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    new_ld_path = f"{libdir}:{torch_lib_dir}:{old_ld_path}" if old_ld_path else f"{libdir}:{torch_lib_dir}"
    os.environ["LD_LIBRARY_PATH"] = new_ld_path
    os.environ["RUSTFLAGS"] = f"-L {libdir} -l{pyver}"

    print(f"🔧 Using Python lib from: {libdir}")
    print(f"🔗 LD_LIBRARY_PATH={os.environ['LD_LIBRARY_PATH']}")
    print(f"⚙️  RUSTFLAGS={os.environ['RUSTFLAGS']}")

    # Run example
    cmd = ["cargo", "run", "--example", "tch_trt_engine"]
    print(f"🚀 Running example: {' '.join(cmd)}")
    os.execvp(cmd[0], cmd)

if __name__ == "__main__":
    main()


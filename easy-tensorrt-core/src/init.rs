use crate::init_libnv_infer_plugins;
use cudarc::driver::{result::DriverError, CudaContext, CudaStream};
use std::sync::Arc;

/// Initialize TensorRT plugins.
pub fn init_cuda_tensorrt() -> Result<(), DriverError> {
    if init_libnv_infer_plugins() {
        Ok(())
    } else {
        Err(DriverError(
            cudarc::driver::sys::CUresult::CUDA_ERROR_NOT_INITIALIZED,
        ))
    }
}

/// A wrapper around CudaContext.
pub struct TRTContextGuard {
    device: Arc<CudaContext>,
}

impl TRTContextGuard {
    /// Cuda Index of the context.
    pub fn cuda_idx(&self) -> u32 {
        self.device.ordinal() as u32
    }

    /// Cuda context.
    pub fn device(&self) -> Arc<CudaContext> {
        self.device.clone()
    }

    /// The default cuda stream for the context.
    pub fn default_stream(&self) -> Arc<CudaStream> {
        self.device.default_stream()
    }

    /// Create a new cuda stream for the context.
    pub fn new_stream(&self) -> Result<Arc<CudaStream>, DriverError> {
        self.device.new_stream()
    }
}

/// Hold the returned guard as long as cuda or TRT is in use.
pub fn tensorrt_context_guard(device_ordinal: u32) -> Result<TRTContextGuard, DriverError> {
    let device = CudaContext::new(device_ordinal as usize)?;
    Ok(TRTContextGuard { device })
}

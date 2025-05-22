use crate::error::TRTError;
use cudarc::driver::CudaStream;
use easy_tensorrt_sys::runtime::DataType;
use std::ffi::c_void;
use std::sync::Arc;

/// Trait for tensor abstraction.
/// Any Tensor (e.g., tch::Tensor) that implements this trait can be used in TRTEngine.
pub trait AbstractTensor {
    /// Creates a new tensor with zeros.
    fn zeros(shape: &[i32], dtype: DataType, stream: &Arc<CudaStream>) -> Result<Self, TRTError>
    where
        Self: Sized;
    /// Data type of the tensor.
    fn dtype(&self) -> DataType;
    /// Shape of the tensor.
    fn shape(&self) -> Vec<i32>;
    /// Element number of the tensor.
    fn size(&self) -> usize;
    /// The ptr to the underlying cuda memory of the tensor.
    fn data_ptr(&self) -> *mut c_void;
    /// Clone the data from the provided tensor.
    fn clone_from(&mut self, tensor: &Self);
}

use thiserror::Error;

#[derive(Error, Debug)]
pub enum TRTError {
    #[error("IO error: {0}")]
    IOError(#[from] std::io::Error),
    #[error("Cuda error: {0}")]
    CudaError(#[from] cudarc::driver::DriverError),
    #[error("TensorRT runtime creation error")]
    RuntimeCreationError,
    #[error("TensorRT engine deserialization error")]
    EngineDeserializationError,
    #[error("TensorRT engine creation error")]
    EngineCreationError,
    #[error("TensorRT execution context not initialized")]
    ExecutionContextNotInitialized,
    #[error("TensorRT execution context creation error")]
    ExecutionContextCreationError,
    #[error("TensorRT mismatch shape: input {0:?}, requires {0:?}")]
    ShapeMisMatchError(Vec<i32>, Vec<i32>),
    #[error("TensorRT invalid shape: {0:?}")]
    ShapeError(Vec<i32>),
    #[error("TensorRT invalid address")]
    InvalidAddress,
    #[error("TensorRT enqueue error")]
    EnqueueError,
    #[error("TensorRT reset shapes error")]
    ResetShapesError,
    #[error("TensorRT shape mismatch")]
    ShapeMismatch,
    #[error("TensorRT dtype mismatch")]
    DTypeMismatch,
    #[error("Input Tensor num mismatch: input {0} tensors, requires {1} tensors")]
    IoTensorNumMismatch(usize, usize),
    #[error("tch Error: {0:?}")]
    TchError(String),
}

pub type TRTResult<T> = Result<T, TRTError>;

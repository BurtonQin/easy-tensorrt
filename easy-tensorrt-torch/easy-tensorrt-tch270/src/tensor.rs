use cudarc::driver::CudaStream;
use easy_tensorrt_core::error::TRTError;
use easy_tensorrt_core::tensor::AbstractTensor;
use easy_tensorrt_core::DataType;
use std::ffi::c_void;
use std::sync::Arc;

pub struct Tensor(tch::Tensor);

impl AbstractTensor for Tensor {
    fn zeros(shape: &[i32], dtype: DataType, stream: &Arc<CudaStream>) -> Result<Self, TRTError> {
        let device_idx = stream.context().cu_device() as usize;
        let tch_size = to_tch_size(shape);
        let tch_kind = to_tch_kind(dtype);
        let tch_dev = tch::Device::Cuda(device_idx);
        let tensor = tch::Tensor::zeros(tch_size, (tch_kind, tch_dev));
        Self::new(tensor)
    }

    fn dtype(&self) -> DataType {
        match self.0.kind() {
            tch::Kind::Float => DataType::FLOAT,
            tch::Kind::Half => DataType::HALF,
            tch::Kind::Int8 => DataType::INT8,
            tch::Kind::Int => DataType::INT32,
            tch::Kind::Bool => DataType::BOOL,
            tch::Kind::Uint8 => DataType::UINT8,
            _ => unreachable!(), // checked when constructing Tensor
        }
    }

    fn shape(&self) -> Vec<i32> {
        self.0.size().into_iter().map(|dim| dim as i32).collect()
    }

    fn size(&self) -> usize {
        self.0.numel()
    }

    fn data_ptr(&self) -> *mut c_void {
        self.0.data_ptr()
    }

    fn clone_from(&mut self, tensor: &Self) {
        self.0.copy_(&tensor.0);
    }
}

impl Tensor {
    pub fn new(tensor: tch::Tensor) -> Result<Self, TRTError> {
        if !tensor.is_contiguous() {
            return Err(TRTError::TchError(format!("{tensor:?} is not contiguous")));
        }
        Self::check_tch_tensor_kind(tensor.kind())?;
        Ok(Self(tensor))
    }

    #[inline]
    fn check_tch_tensor_kind(kind: tch::Kind) -> Result<(), TRTError> {
        match kind {
            tch::Kind::Float
            | tch::Kind::Half
            | tch::Kind::Int8
            | tch::Kind::Int
            | tch::Kind::Bool
            | tch::Kind::Uint8 => Ok(()),
            _ => Err(TRTError::TchError(format!(
                "{kind:?} not supported in TensorRT"
            ))),
        }
    }

    pub fn into_inner(self) -> tch::Tensor {
        self.0
    }

    pub fn shallow_clone(&self) -> tch::Tensor {
        self.0.shallow_clone()
    }
}

pub fn to_tch_kind(dtype: DataType) -> tch::Kind {
    match dtype {
        DataType::FLOAT => tch::Kind::Float,
        DataType::HALF => tch::Kind::Half,
        DataType::INT8 => tch::Kind::Int8,
        DataType::INT32 => tch::Kind::Int,
        DataType::BOOL => tch::Kind::Bool,
        DataType::UINT8 => tch::Kind::Uint8,
        _ => panic!("Unsupported DataType"),
    }
}

pub fn to_tch_size(shape: &[i32]) -> Vec<i64> {
    shape.iter().map(|i| *i as i64).collect::<Vec<_>>()
}

use cudarc::driver::DevicePtr;
use cudarc::driver::{CudaSlice, CudaStream};
use easy_tensorrt_core::{init, AbstractTensor, TRTEngine};
use easy_tensorrt_core::{DataType, TRTError};
use half;
use std::sync::Arc;

pub enum CustomTensorStorage {
    Float(CudaSlice<f32>),
    Half(CudaSlice<half::f16>),
    Int32(CudaSlice<i32>),
    Bool(CudaSlice<bool>),
    Uint8(CudaSlice<u8>),
    Fp8(CudaSlice<u8>),
}

pub struct CustomTensor {
    storage: CustomTensorStorage,
    shape: Vec<i32>,
}

impl AbstractTensor for CustomTensor {
    fn zeros(shape: &[i32], dtype: DataType, stream: &Arc<CudaStream>) -> Result<Self, TRTError>
    where
        Self: Sized,
    {
        let len = shape.iter().product::<i32>() as usize;
        match dtype {
            DataType::FLOAT => {
                let storage = CustomTensorStorage::Float(stream.alloc_zeros::<f32>(len)?);
                return Ok(Self {
                    storage,
                    shape: shape.to_vec(),
                });
            }
            _ => return Err(TRTError::DTypeMismatch),
        }
    }

    fn dtype(&self) -> DataType {
        use CustomTensorStorage::*;
        match &self.storage {
            Float(_) => DataType::FLOAT,
            _ => panic!("Unsupported data type"),
        }
    }

    fn shape(&self) -> Vec<i32> {
        self.shape.clone()
    }
    fn size(&self) -> usize {
        use CustomTensorStorage::*;
        match &self.storage {
            Float(data) => data.len(),
            _ => panic!("Unsupported data type"),
        }
    }
    fn data_ptr(&self) -> *mut std::ffi::c_void {
        use CustomTensorStorage::*;
        match &self.storage {
            Float(data) => {
                let stream = data.stream();
                let (ptr, _sync_on_drop) = data.device_ptr(stream);
                ptr as *mut std::ffi::c_void
            }
            _ => panic!("Unsupported data type"),
        }
    }

    fn clone_from(&mut self, tensor: &Self) {
        use CustomTensorStorage::*;
        match (&mut self.storage, &tensor.storage) {
            (Float(dst), Float(src)) => {
                dst.clone_from(src);
            }
            _ => panic!("Unsupported data type"),
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA and TensorRT runtime (only once per process)
    init::init_cuda_tensorrt()?;

    let device_ordinal = 0;
    let guard = init::tensorrt_context_guard(device_ordinal)?;
    let stream = guard.default_stream();

    let mut engine =
        TRTEngine::<CustomTensor>::new(&r"../model/model.trt".to_owned(), stream.clone())?;
    engine.activate()?;
    engine.allocate_io_tensors()?;
    println!("{}", engine.info()?);

    let input_tensor = CustomTensor::zeros(&[2, 3, 512, 512], DataType::FLOAT, &stream)?;
    let output_tensors = engine.inference(&[input_tensor])?;
    for (i, output) in output_tensors.iter().enumerate() {
        println!("Output {}: {:?}", i, output.shape());
    }
    Ok(())
}

use crate::error::{TRTError, TRTResult};
use crate::tensor::AbstractTensor;

use cudarc::driver::result::stream;
use cudarc::driver::CudaStream;
use easy_tensorrt_sys::{
    logger::Severity,
    runtime::{CudaEngine, ExecutionContext, Runtime},
};
use std::fmt::Write;
use std::sync::Arc;
use std::{fs, path::Path};

pub struct TRTEngine<T> {
    runtime: Option<Runtime>,
    engine: Option<CudaEngine>,
    context: Option<ExecutionContext>,
    stream: Arc<CudaStream>,
    tensors: Vec<T>,
}

impl<T: AbstractTensor> TRTEngine<T> {
    /// Create a new TRTEngine from an engine file.
    /// The stream is used to execute the engine.
    pub fn new<P: AsRef<Path>>(engine_path: &P, stream: Arc<CudaStream>) -> TRTResult<Self> {
        let mut runtime = match Runtime::new() {
            Some(runtime) => runtime,
            None => return Err(TRTError::RuntimeCreationError),
        };

        let data = fs::read(engine_path)?;

        let engine = match runtime.deserialize(data.as_slice()) {
            Some(engine) => engine,
            None => return Err(TRTError::EngineDeserializationError),
        };

        Ok(Self {
            runtime: Some(runtime),
            engine: Some(engine),
            context: None,
            stream,
            tensors: Vec::new(),
        })
    }

    /// Create the context
    /// TODO: reuse device memory
    pub fn activate(&mut self) -> TRTResult<()> {
        let engine = match self.engine.as_mut() {
            Some(engine) => engine,
            None => return Err(TRTError::EngineCreationError),
        };

        self.context = match engine.create_execution_context() {
            Some(context) => Some(context),
            None => return Err(TRTError::ExecutionContextCreationError),
        };

        Ok(())
    }

    /// Allocate input and output tensors for the engine.
    pub fn allocate_io_tensors(&mut self) -> TRTResult<()> {
        let engine = match self.engine.as_mut() {
            Some(engine) => engine,
            None => return Err(TRTError::EngineCreationError),
        };

        let context: &mut ExecutionContext = match self.context.as_mut() {
            Some(context) => context,
            None => return Err(TRTError::ExecutionContextNotInitialized),
        };

        let num_io_tensors = engine.get_num_io_tensors();
        for i in 0..num_io_tensors {
            let name = engine.get_io_tensor_name(i);
            let shape = engine.get_tensor_shape(name);
            if shape.iter().any(|&dim| dim < 0) {
                return Err(TRTError::ShapeError(shape.clone()));
            }
            if engine.get_tensor_io_mode(name).is_input()
                && !context.set_input_shape(name, shape.as_slice())
            {
                return Err(TRTError::ShapeError(shape.clone()));
            }
            let dtype = engine.get_tensor_dtype(name);
            let tensor = T::zeros(&shape, dtype, &self.stream)?;
            let ptr = tensor.data_ptr();
            self.tensors.push(tensor);
            if !context.set_tensor_address(name, ptr as _) {
                return Err(TRTError::InvalidAddress);
            }
        }

        Ok(())
    }

    /// Print the info of all input and output tensors.
    pub fn info(&mut self) -> TRTResult<String> {
        let engine = match self.engine.as_mut() {
            Some(engine) => engine,
            None => return Err(TRTError::EngineCreationError),
        };

        let _context: &mut ExecutionContext = match self.context.as_mut() {
            Some(context) => context,
            None => return Err(TRTError::ExecutionContextNotInitialized),
        };

        let num_io_tensors = engine.get_num_io_tensors();
        let mut output = String::new();
        let cuda_idx = self.stream.context().ordinal();
        for i in 0..num_io_tensors {
            let name = engine.get_io_tensor_name(i);
            let shape = engine.get_tensor_shape(name);
            if shape.iter().any(|&dim| dim < 0) {
                return Err(TRTError::ShapeError(shape.clone()));
            }
            let is_input = engine.get_tensor_io_mode(name).is_input();
            let is_input_output = if is_input { "is_input" } else { "is_output" };
            let dtype = engine.get_tensor_dtype(name);
            writeln!(
                output,
                "{name}: {is_input_output}\t{shape:?}\t{dtype:?}\tCuda({})",
                cuda_idx
            )
            .unwrap();
        }

        Ok(output)
    }

    /// Run inference on the engine.
    /// TODO: use cuda graph
    pub fn inference(&mut self, feed_tensors: &[T]) -> TRTResult<&[T]> {
        let context: &mut ExecutionContext = match self.context.as_mut() {
            Some(context) => context,
            None => return Err(TRTError::ExecutionContextNotInitialized),
        };
        let stream = &self.stream.cu_stream();

        let input_len = feed_tensors.len();

        for (input_tensor, tensor) in feed_tensors.iter().zip(self.tensors.iter_mut()) {
            tensor.clone_from(input_tensor);
        }
        if !context.enqueue_v3(stream) {
            return Err(TRTError::EnqueueError);
        }

        unsafe {
            stream::synchronize(*stream)?;
        }
        Ok(&self.tensors[input_len..])
    }

    pub fn log(&mut self, level: Severity, msg: &str) {
        self.runtime.as_mut().unwrap().logger().log(level, msg);
    }
}

impl<T> Drop for TRTEngine<T> {
    fn drop(&mut self) {
        if let Some(context) = self.context.take() {
            std::mem::drop(context);
        }

        if let Some(engine) = self.engine.take() {
            std::mem::drop(engine);
        }

        if let Some(runtime) = self.runtime.take() {
            std::mem::drop(runtime);
        }
    }
}

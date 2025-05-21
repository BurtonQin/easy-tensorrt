use crate::Tensor;
use cudarc::driver::CudaStream;
use easy_tensorrt_core::{error::TRTResult, TRTEngine};
use std::path::Path;
use std::sync::Arc;

pub struct TchTrtEngine(TRTEngine<crate::Tensor>);

impl TchTrtEngine {
    pub fn new<P: AsRef<Path>>(engine_path: &P, stream: Arc<CudaStream>) -> TRTResult<Self> {
        let mut engine = TRTEngine::new(engine_path, stream)?;
        engine.activate()?;
        engine.allocate_io_tensors()?;
        Ok(Self(engine))
    }

    pub fn info(&mut self) -> TRTResult<String> {
        self.0.info()
    }

    pub fn inference(&mut self, feed_tensors: &[Tensor]) -> TRTResult<&[Tensor]> {
        self.0.inference(feed_tensors)
    }
}

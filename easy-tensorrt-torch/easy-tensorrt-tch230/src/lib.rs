pub mod engine;
pub mod tensor;
use crate::init::TRTContextGuard;
use easy_tensorrt_core::init;
pub use engine::TchTrtEngine;
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::PyModule;
use pyo3::{pyclass, pymethods, pymodule, PyResult, Python};
use pyo3_tch::PyTensor;
pub use tensor::Tensor;

#[pyclass(unsendable)]
pub struct CudaTRTEnvWrapper(TRTContextGuard);

#[pymethods]
impl CudaTRTEnvWrapper {
    #[new]
    fn new(device_ordinal: u32) -> PyResult<Self> {
        init::init_cuda_tensorrt().map_err(|_| PyRuntimeError::new_err("init tensorrt failed"))?;
        let guard = init::tensorrt_context_guard(device_ordinal)
            .map_err(|e| PyRuntimeError::new_err(format!("{e:?}")))?;
        Ok(Self(guard))
    }
}

#[pyclass(unsendable)]
pub struct TchTrtEngineWrapper(TchTrtEngine);

#[pymethods]
impl TchTrtEngineWrapper {
    #[new]
    pub fn new(env: &CudaTRTEnvWrapper, path: &str) -> PyResult<Self> {
        let stream = env.0.default_stream();
        let engine = TchTrtEngine::new(&path.to_owned(), stream)
            .map_err(|e| PyRuntimeError::new_err(format!("{e:?}")))?;
        Ok(Self(engine))
    }

    pub fn info(&mut self) -> PyResult<String> {
        self.0
            .info()
            .map_err(|e| PyRuntimeError::new_err(format!("{e:?}")))
    }

    pub fn inference(&mut self, inputs: Vec<PyTensor>) -> PyResult<Vec<PyTensor>> {
        let inputs = inputs
            .iter()
            .map(|x| Tensor::new(x.0.shallow_clone()).unwrap())
            .collect::<Vec<_>>();
        let outputs = self
            .0
            .inference(&inputs)
            .map_err(|e| PyRuntimeError::new_err(format!("{e:?}")))?;
        Ok(outputs
            .iter()
            .map(|x| PyTensor(x.shallow_clone()))
            .collect())
    }
}

#[pymodule]
fn easy_tensorrt_tch230(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<CudaTRTEnvWrapper>()?;
    m.add_class::<TchTrtEngineWrapper>()?;
    Ok(())
}

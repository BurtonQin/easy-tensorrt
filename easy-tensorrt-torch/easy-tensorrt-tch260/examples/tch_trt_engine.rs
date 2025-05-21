use easy_tensorrt_core::init;
use easy_tensorrt_tch260::TchTrtEngine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA and TensorRT runtime (only once per process)
    init::init_cuda_tensorrt()?;

    let device_ordinal = 0;
    let guard = init::tensorrt_context_guard(device_ordinal)?;
    let stream = guard.default_stream();

    let mut engine = TchTrtEngine::new(&r"model.trt".to_owned(), stream)?;

    println!("{}", engine.info()?);

    Ok(())
}

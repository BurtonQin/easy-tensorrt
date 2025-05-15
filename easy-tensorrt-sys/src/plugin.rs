use crate::ffi;

pub type PluginLibraryHandle = usize;

pub fn load_library(plugin_path: &str) -> PluginLibraryHandle {
    ffi::load_library(plugin_path)
}

pub fn unload_library(handle: PluginLibraryHandle) {
    ffi::unload_library(handle)
}

pub fn init_libnv_infer_plugins() -> bool {
    ffi::init_libnv_infer_plugins()
}

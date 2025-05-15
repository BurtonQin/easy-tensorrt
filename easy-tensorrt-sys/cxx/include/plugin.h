#pragma once

#include <memory>
#include <NvInferRuntime.h>
#include <NvInferPlugin.h>
#include "rust/cxx.h"

namespace trt_rs::plugin {

using nvinfer1::IPluginRegistry;

inline size_t load_library(rust::Str plugin_path) noexcept {
    const auto path = std::string(plugin_path);
    const auto handle = getPluginRegistry()->loadLibrary(path.c_str());
    return reinterpret_cast<size_t>(handle);
}

inline void unload_library(size_t handle) noexcept {
    getPluginRegistry()->deregisterLibrary(
        reinterpret_cast<IPluginRegistry::PluginLibraryHandle>(handle));
}

inline bool init_libnv_infer_plugins() noexcept {
    return initLibNvInferPlugins(nullptr, "");
}

} // namespace trt_rs::plugin

if (NOT SCRAPGPU_BUILD_OPENCL11)
    message(FATAL_ERROR "SCRAPGPU_BUILD_ORT_EP requires SCRAPGPU_BUILD_OPENCL11=ON")
endif()

if (NOT ORT_ROOT)
    message(FATAL_ERROR "SCRAPGPU_BUILD_ORT_EP requires -DORT_ROOT=... (ONNX Runtime C API package)")
endif()

set(_ort_inc "${ORT_ROOT}/include")
set(_ort_lib "${ORT_ROOT}/lib")

if (NOT EXISTS "${_ort_inc}/onnxruntime_c_api.h")
    message(FATAL_ERROR "ORT_ROOT does not contain include/onnxruntime_c_api.h: ${ORT_ROOT}")
endif()

add_library(scrapgpu_ort_ep SHARED
    scrapgpu_ort_ep.cpp)
target_include_directories(scrapgpu_ort_ep PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
    "${_ort_inc}")
target_link_libraries(scrapgpu_ort_ep PRIVATE scrapgpu_opencl11_api)
target_compile_features(scrapgpu_ort_ep PRIVATE cxx_std_17)
set_target_properties(scrapgpu_ort_ep PROPERTIES
    CXX_VISIBILITY_PRESET default
    VISIBILITY_INLINES_HIDDEN ON
    OUTPUT_NAME scrapgpu_ort_ep
    PREFIX lib)
# The plugin is loaded into the host onnxruntime process; do not link libonnxruntime (ABI comes via CreateEpFactories).
if (APPLE)
    target_link_options(scrapgpu_ort_ep PRIVATE "-Wl,-undefined,dynamic_lookup")
endif()

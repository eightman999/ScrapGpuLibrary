cmake_minimum_required(VERSION 3.14)
project(ScrapGpuLibrary C CXX)

option(SCRAPGPU_BUILD_CORE "Build the ScrapGpu core library without ggml or OpenCL" ON)
option(SCRAPGPU_BUILD_OPENCL11 "Build the OpenCL 1.1 backend (Phase 2)" OFF)
option(SCRAPGPU_BUILD_GGML_ADAPTER "Build the existing ggml CL11 adapter" OFF)
option(SCRAPGPU_BUILD_PYTORCH "Build the PyTorch adapter (Stage A custom ops)" OFF)
option(SCRAPGPU_BUILD_TESTS "Build ScrapGpu contract tests" ON)
option(SCRAPGPU_BUILD_EXAMPLES "Build the examples (needs SCRAPGPU_BUILD_OPENCL11)" ${SCRAPGPU_BUILD_OPENCL11})

if (SCRAPGPU_BUILD_CORE OR SCRAPGPU_BUILD_OPENCL11)
    # the ggml-independent tensor descriptors and their math (no OpenCL, no ggml)
    add_library(scrapgpu_tensor STATIC
        src/tensor/tensor.cpp)
    target_include_directories(scrapgpu_tensor PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_compile_features(scrapgpu_tensor PUBLIC cxx_std_17)
    set_target_properties(scrapgpu_tensor PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        POSITION_INDEPENDENT_CODE ON)
endif()

if (SCRAPGPU_BUILD_CORE)
    add_library(scrapgpu_core
        src/core/op.cpp
        src/core/ref/f16.cpp
        src/core/ref/ops.cpp
        src/core/ref/quant.cpp
        src/core/registry.cpp
        src/core/status.cpp
        src/core/tensor.cpp)
    target_include_directories(scrapgpu_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_link_libraries(scrapgpu_core PRIVATE scrapgpu_tensor)
    target_compile_features(scrapgpu_core PUBLIC cxx_std_17)
    set_target_properties(scrapgpu_core PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON)
endif()

if (SCRAPGPU_BUILD_OPENCL11)
    find_package(OpenCL REQUIRED)
    add_library(scrapgpu_opencl11 STATIC
        src/backends/opencl11/buffer.cpp
        src/backends/opencl11/context.cpp
        src/backends/opencl11/device.cpp
        src/backends/opencl11/device_state.cpp
        src/backends/opencl11/kernels.cpp
        src/backends/opencl11/ops.cpp
        src/backends/opencl11/ops_matmul.cpp
        src/backends/opencl11/ops_moe.cpp
        src/backends/opencl11/ops_support.cpp
        src/backends/opencl11/r32.cpp)
    target_include_directories(scrapgpu_opencl11 PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/src/backends"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/backends/opencl11"
        ${OpenCL_INCLUDE_DIRS})
    target_link_libraries(scrapgpu_opencl11 PUBLIC ${OpenCL_LIBRARIES} scrapgpu_tensor)
    target_compile_features(scrapgpu_opencl11 PUBLIC cxx_std_17)
    target_compile_definitions(scrapgpu_opencl11 PUBLIC
        CL_TARGET_OPENCL_VERSION=110
        CL_USE_DEPRECATED_OPENCL_1_1_APIS)
    set_target_properties(scrapgpu_opencl11 PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        POSITION_INDEPENDENT_CODE ON)

    include("${CMAKE_CURRENT_SOURCE_DIR}/src/backends/opencl11/opencl11_kernels.cmake")
    scrapgpu_opencl11_embed_kernels(scrapgpu_opencl11 "${CMAKE_CURRENT_SOURCE_DIR}/src/backends/opencl11" "${CMAKE_CURRENT_BINARY_DIR}")

    # the ScrapGpu C API on the OpenCL 1.1 backend (scrapgpu_opencl11_register); not part of the ggml plugin
    add_library(scrapgpu_opencl11_api STATIC
        src/backends/opencl11/api.cpp)
    target_include_directories(scrapgpu_opencl11_api PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(scrapgpu_opencl11_api PUBLIC scrapgpu_core scrapgpu_opencl11)
    target_compile_features(scrapgpu_opencl11_api PUBLIC cxx_std_17)
    set_target_properties(scrapgpu_opencl11_api PROPERTIES CXX_VISIBILITY_PRESET hidden)
endif()

if (SCRAPGPU_BUILD_GGML_ADAPTER)
    add_subdirectory(adapters/ggml)
endif()

if (SCRAPGPU_BUILD_PYTORCH)
    if (NOT SCRAPGPU_BUILD_OPENCL11)
        message(FATAL_ERROR "SCRAPGPU_BUILD_PYTORCH requires SCRAPGPU_BUILD_OPENCL11=ON")
    endif ()
    add_subdirectory(adapters/pytorch)
endif ()

if (SCRAPGPU_BUILD_TESTS AND SCRAPGPU_BUILD_CORE)
    enable_testing()

    add_executable(scrapgpu_core_contract_tests tests/core/core_contract_tests.cpp)
    target_link_libraries(scrapgpu_core_contract_tests PRIVATE scrapgpu_core)
    add_test(NAME scrapgpu_core_contract_tests COMMAND scrapgpu_core_contract_tests)

    add_executable(scrapgpu_public_headers_c tests/core/public_headers_c.c)
    target_link_libraries(scrapgpu_public_headers_c PRIVATE scrapgpu_core)
    set_target_properties(scrapgpu_public_headers_c PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED ON)
    add_test(NAME scrapgpu_public_headers_c COMMAND scrapgpu_public_headers_c)

    add_executable(scrapgpu_public_headers_cpp tests/core/public_headers_cpp.cpp)
    target_link_libraries(scrapgpu_public_headers_cpp PRIVATE scrapgpu_core)
    add_test(NAME scrapgpu_public_headers_cpp COMMAND scrapgpu_public_headers_cpp)

    add_executable(scrapgpu_tensor_tests tests/tensor/tensor_tests.cpp)
    target_link_libraries(scrapgpu_tensor_tests PRIVATE scrapgpu_tensor)
    target_include_directories(scrapgpu_tensor_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/backends")
    add_test(NAME scrapgpu_tensor_tests COMMAND scrapgpu_tensor_tests)

    add_executable(scrapgpu_r32_tests tests/r32/r32_tests.cpp)
    target_link_libraries(scrapgpu_r32_tests PRIVATE scrapgpu_tensor)
    target_include_directories(scrapgpu_r32_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/backends")
    add_test(NAME scrapgpu_r32_tests COMMAND scrapgpu_r32_tests)

    add_executable(scrapgpu_ref_ops_tests tests/ref/ref_ops_tests.cpp)
    target_link_libraries(scrapgpu_ref_ops_tests PRIVATE scrapgpu_core)
    add_test(NAME scrapgpu_ref_ops_tests COMMAND scrapgpu_ref_ops_tests)

    add_executable(scrapgpu_ref_quant_tests tests/ref/ref_quant_tests.cpp)
    target_link_libraries(scrapgpu_ref_quant_tests PRIVATE scrapgpu_core)
    add_test(NAME scrapgpu_ref_quant_tests COMMAND scrapgpu_ref_quant_tests)

    # GPU tests of the ops through the public C API only (SKIPPED when no OpenCL device exists)
    if (SCRAPGPU_BUILD_OPENCL11)
        add_executable(scrapgpu_ops_tests tests/ops/ops_tests.cpp)
        target_link_libraries(scrapgpu_ops_tests PRIVATE scrapgpu_opencl11_api)
        add_test(NAME scrapgpu_ops_tests COMMAND scrapgpu_ops_tests)
        set_tests_properties(scrapgpu_ops_tests PROPERTIES SKIP_RETURN_CODE 77)
    endif()

    add_test(NAME scrapgpu_dependency_check
        COMMAND "${CMAKE_COMMAND}" -DROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}" -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_core_dependencies.cmake")
endif()

if (SCRAPGPU_BUILD_EXAMPLES)
    if (NOT SCRAPGPU_BUILD_OPENCL11)
        message(FATAL_ERROR "SCRAPGPU_BUILD_EXAMPLES needs SCRAPGPU_BUILD_OPENCL11=ON")
    endif()
    add_executable(scrapgpu_standalone_matmul examples/standalone_matmul.c)
    target_link_libraries(scrapgpu_standalone_matmul PRIVATE scrapgpu_opencl11_api)
    set_target_properties(scrapgpu_standalone_matmul PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED ON)
endif()

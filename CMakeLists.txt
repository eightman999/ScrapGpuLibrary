cmake_minimum_required(VERSION 3.14)
project(ScrapGpuLibrary C CXX)

option(SCRAPGPU_BUILD_CORE "Build the ScrapGpu core library without ggml or OpenCL" ON)
option(SCRAPGPU_BUILD_OPENCL11 "Build the OpenCL 1.1 backend (Phase 2)" OFF)
option(SCRAPGPU_BUILD_GGML_ADAPTER "Build the existing ggml CL11 adapter" OFF)
option(SCRAPGPU_BUILD_TESTS "Build ScrapGpu contract tests" ON)

if (SCRAPGPU_BUILD_CORE)
    add_library(scrapgpu_core
        src/core/op.cpp
        src/core/registry.cpp
        src/core/status.cpp
        src/core/tensor.cpp)
    target_include_directories(scrapgpu_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
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
        src/backends/opencl11/kernels.cpp)
    target_include_directories(scrapgpu_opencl11 PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/src/backends"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/backends/opencl11"
        ${OpenCL_INCLUDE_DIRS})
    target_link_libraries(scrapgpu_opencl11 PUBLIC ${OpenCL_LIBRARIES})
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
endif()

if (SCRAPGPU_BUILD_GGML_ADAPTER)
    add_subdirectory(adapters/ggml)
endif()

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

    add_test(NAME scrapgpu_dependency_check
        COMMAND "${CMAKE_COMMAND}" -DROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}" -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_core_dependencies.cmake")
endif()

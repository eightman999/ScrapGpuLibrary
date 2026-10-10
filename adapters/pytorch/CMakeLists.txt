cmake_minimum_required(VERSION 3.14)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
    RESULT_VARIABLE _torch_import_status
    OUTPUT_VARIABLE _torch_cmake_prefix
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if (NOT _torch_import_status EQUAL 0)
    message(FATAL_ERROR "The selected Python3_EXECUTABLE must have PyTorch installed")
endif ()
list(APPEND CMAKE_PREFIX_PATH "${_torch_cmake_prefix}")
find_package(Torch REQUIRED)

set(SCRAPGPU_TORCH_EXT_NAME "_C")

add_library(${SCRAPGPU_TORCH_EXT_NAME} SHARED src/torch_api.cpp)
target_compile_features(${SCRAPGPU_TORCH_EXT_NAME} PRIVATE cxx_std_20)
target_include_directories(${SCRAPGPU_TORCH_EXT_NAME} PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
    ${TORCH_INCLUDE_DIRS})
target_link_libraries(${SCRAPGPU_TORCH_EXT_NAME} PRIVATE
    ${TORCH_LIBRARIES})
if (SCRAPGPU_BUILD_OPENCL11)
    target_link_libraries(${SCRAPGPU_TORCH_EXT_NAME} PRIVATE scrapgpu_opencl11_api)
    target_compile_definitions(${SCRAPGPU_TORCH_EXT_NAME} PRIVATE SCRAPGPU_TORCH_HAS_OPENCL11=1)
endif ()
target_link_libraries(${SCRAPGPU_TORCH_EXT_NAME} PRIVATE scrapgpu_core)
separate_arguments(_torch_cxx_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(${SCRAPGPU_TORCH_EXT_NAME} PRIVATE ${_torch_cxx_flags})

# Place the extension next to the Python package (import scrapgpu_torch then load()).
set_target_properties(${SCRAPGPU_TORCH_EXT_NAME} PROPERTIES
    PREFIX ""
    OUTPUT_NAME "${SCRAPGPU_TORCH_EXT_NAME}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/scrapgpu_torch"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/scrapgpu_torch")

if (APPLE)
    set_target_properties(${SCRAPGPU_TORCH_EXT_NAME} PROPERTIES SUFFIX ".so")
endif ()

if (SCRAPGPU_BUILD_TESTS)
    enable_testing()
    foreach (_test ops attention backend_config)
        add_test(NAME scrapgpu_pytorch_${_test}_cpu
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_${_test}.py")
        set_tests_properties(scrapgpu_pytorch_${_test}_cpu PROPERTIES
            ENVIRONMENT "SCRAPGPU_TORCH_BACKEND=cpu;SCRAPGPU_TORCH_DEVICE=0;SCRAPGPU_TORCH_ALLOW_CPU_FALLBACK=0"
            TIMEOUT 120)
    endforeach ()
endif ()

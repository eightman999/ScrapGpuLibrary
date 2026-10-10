cmake_minimum_required(VERSION 3.14)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE _torch_cmake_prefix
    OUTPUT_STRIP_TRAILING_WHITESPACE)
list(APPEND CMAKE_PREFIX_PATH "${_torch_cmake_prefix}")
find_package(Torch REQUIRED)

set(SCRAPGPU_TORCH_EXT_NAME "_C")

add_library(${SCRAPGPU_TORCH_EXT_NAME} SHARED src/torch_api.cpp)
target_compile_features(${SCRAPGPU_TORCH_EXT_NAME} PRIVATE cxx_std_20)
target_include_directories(${SCRAPGPU_TORCH_EXT_NAME} PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
    ${TORCH_INCLUDE_DIRS})
target_link_libraries(${SCRAPGPU_TORCH_EXT_NAME} PRIVATE
    ${TORCH_LIBRARIES}
    scrapgpu_opencl11_api)
target_compile_options(${SCRAPGPU_TORCH_EXT_NAME} PRIVATE ${TORCH_CXX_FLAGS})

# Place the extension next to the Python package (import scrapgpu_torch then load()).
set_target_properties(${SCRAPGPU_TORCH_EXT_NAME} PROPERTIES
    PREFIX ""
    OUTPUT_NAME "${SCRAPGPU_TORCH_EXT_NAME}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/scrapgpu_torch"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/scrapgpu_torch")

if (APPLE)
    set_target_properties(${SCRAPGPU_TORCH_EXT_NAME} PROPERTIES SUFFIX ".so")
endif ()

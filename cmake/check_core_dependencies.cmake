if (NOT DEFINED ROOT_DIR)
    message(FATAL_ERROR "ROOT_DIR is required")
endif()

file(GLOB_RECURSE SCRAPGPU_PUBLIC_HEADERS
    "${ROOT_DIR}/include/scrapgpu/*.h")
file(GLOB_RECURSE SCRAPGPU_CORE_SOURCES
    "${ROOT_DIR}/src/core/*.h"
    "${ROOT_DIR}/src/core/*.hpp"
    "${ROOT_DIR}/src/core/*.c"
    "${ROOT_DIR}/src/core/*.cpp")

set(FORBIDDEN_PATTERN "#[ \t]*include[ \t]*[<\"]([^>\"]*(ggml|CL/cl\\.h|OpenCL/opencl\\.h|torch|onnxruntime)[^>\"]*)[>\"]")
foreach (FILE IN LISTS SCRAPGPU_PUBLIC_HEADERS SCRAPGPU_CORE_SOURCES)
    file(READ "${FILE}" CONTENT)
    if (CONTENT MATCHES "${FORBIDDEN_PATTERN}")
        message(FATAL_ERROR "Forbidden dependency include in ${FILE}: ${CMAKE_MATCH_1}")
    endif()
endforeach()

message(STATUS "ScrapGpu core dependency check passed")

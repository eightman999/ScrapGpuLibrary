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
file(GLOB_RECURSE SCRAPGPU_TENSOR_SOURCES
    "${ROOT_DIR}/src/tensor/*.h"
    "${ROOT_DIR}/src/tensor/*.hpp"
    "${ROOT_DIR}/src/tensor/*.c"
    "${ROOT_DIR}/src/tensor/*.cpp")
file(GLOB_RECURSE SCRAPGPU_CORE_TESTS
    "${ROOT_DIR}/tests/ref/*.h"
    "${ROOT_DIR}/tests/ref/*.hpp"
    "${ROOT_DIR}/tests/ref/*.c"
    "${ROOT_DIR}/tests/ref/*.cpp"
    "${ROOT_DIR}/tests/ops/*.h"
    "${ROOT_DIR}/tests/ops/*.hpp"
    "${ROOT_DIR}/tests/ops/*.c"
    "${ROOT_DIR}/tests/ops/*.cpp"
    "${ROOT_DIR}/tests/remote/*.h"
    "${ROOT_DIR}/tests/remote/*.hpp"
    "${ROOT_DIR}/tests/remote/*.c"
    "${ROOT_DIR}/tests/remote/*.cpp"
    "${ROOT_DIR}/examples/*.h"
    "${ROOT_DIR}/examples/*.hpp"
    "${ROOT_DIR}/examples/*.c"
    "${ROOT_DIR}/examples/*.cpp")
file(GLOB_RECURSE SCRAPGPU_OPENCL11_SOURCES
    "${ROOT_DIR}/src/backends/opencl11/*.h"
    "${ROOT_DIR}/src/backends/opencl11/*.hpp"
    "${ROOT_DIR}/src/backends/opencl11/*.c"
    "${ROOT_DIR}/src/backends/opencl11/*.cpp")
file(GLOB_RECURSE SCRAPGPU_VULKAN_SOURCES
    "${ROOT_DIR}/src/backends/vulkan/*.h"
    "${ROOT_DIR}/src/backends/vulkan/*.hpp"
    "${ROOT_DIR}/src/backends/vulkan/*.c"
    "${ROOT_DIR}/src/backends/vulkan/*.cpp")

# The pattern must match a real include *line*: anchored at a line start (the first line, or one
# beginning after a newline) with nothing but whitespace before the '#', so commented-out or
# mid-line text like `// #include <ggml.h>` does not count as a violation.
set(FORBIDDEN_PATTERN "(^|\n)[ \t]*#[ \t]*include[ \t]*[<\"]([^>\"]*(ggml|CL/cl\\.h|OpenCL/opencl\\.h|torch|onnxruntime)[^>\"]*)[>\"]")
foreach (FILE IN LISTS SCRAPGPU_PUBLIC_HEADERS SCRAPGPU_CORE_SOURCES SCRAPGPU_TENSOR_SOURCES SCRAPGPU_CORE_TESTS)
    file(READ "${FILE}" CONTENT)
    if (CONTENT MATCHES "${FORBIDDEN_PATTERN}")
        message(FATAL_ERROR "Forbidden dependency include in ${FILE}: ${CMAKE_MATCH_2}")
    endif()
endforeach()

set(FORBIDDEN_OPENCL11_PATTERN "(^|\n)[ \t]*#[ \t]*include[ \t]*[<\"]([^>\"]*(ggml|torch|onnxruntime)[^>\"]*)[>\"]")
foreach (FILE IN LISTS SCRAPGPU_OPENCL11_SOURCES SCRAPGPU_VULKAN_SOURCES)
    file(READ "${FILE}" CONTENT)
    if (CONTENT MATCHES "${FORBIDDEN_OPENCL11_PATTERN}")
        message(FATAL_ERROR "Forbidden adapter dependency include in ${FILE}: ${CMAKE_MATCH_2}")
    endif()
endforeach()

message(STATUS "ScrapGpu dependency check passed")

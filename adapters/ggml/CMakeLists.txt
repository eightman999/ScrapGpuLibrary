# CL11: OpenCL 1.1 backend (no OpenCL >= 1.2 host API, no OpenCL C > 1.1, no Python needed)
#
# Two ways to build this file:
#
#  1. in-tree: the directory is copied to <llama.cpp>/ggml/src/ggml-cl11/ (see llama.cpp/install-in-tree.sh) and
#     included by ggml/src/CMakeLists.txt (ggml_add_backend(CL11), option GGML_CL11).
#  2. out-of-tree plugin: a stand-alone CMake project, needs a llama.cpp checkout (headers) and a llama.cpp build
#     done with -DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON (libggml-base):
#       cmake -S adapters/ggml -B build -DLLAMA_CPP_SOURCE_DIR=<llama.cpp> -DLLAMA_CPP_BUILD_DIR=<llama.cpp>/build
#     The result is libggml-cl11.{so,dylib}, loaded with GGML_BACKEND_PATH=/path/to/it.
#
# The ggml backend API is not stable: this code is written for llama.cpp b11384 (commit 0faee50).

if (COMMAND ggml_add_backend_library)
    set(CL11_IN_TREE ON)
else()
    set(CL11_IN_TREE OFF)
    cmake_minimum_required(VERSION 3.14)
    if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
        project(ggml-cl11 C CXX)
    endif()
endif()

find_package(OpenCL REQUIRED)

set(TARGET_NAME ggml-cl11)

if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../src/backends/opencl11/device.cpp")
    get_filename_component(CL11_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../src" ABSOLUTE)
    get_filename_component(CL11_PUBLIC_INC "${CMAKE_CURRENT_SOURCE_DIR}/../../include" ABSOLUTE)
    set(CL11_OPENCL11_DIR "${CL11_SRC_DIR}/backends/opencl11")
else()
    # in-tree: install-in-tree.sh copied the sources next to this file (opencl11/, tensor/, scrapgpu/)
    set(CL11_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    set(CL11_PUBLIC_INC "${CMAKE_CURRENT_SOURCE_DIR}")
    set(CL11_OPENCL11_DIR "${CMAKE_CURRENT_SOURCE_DIR}/opencl11")
endif()

# the ggml-independent tensor descriptors and their math
if (NOT TARGET scrapgpu_tensor)
    add_library(scrapgpu_tensor STATIC
                "${CL11_SRC_DIR}/tensor/tensor.cpp")
    target_include_directories(scrapgpu_tensor PUBLIC "${CL11_SRC_DIR}" "${CL11_PUBLIC_INC}")
    target_compile_features(scrapgpu_tensor PUBLIC cxx_std_17)
    set_target_properties(scrapgpu_tensor PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        POSITION_INDEPENDENT_CODE ON)
endif()

if (NOT TARGET scrapgpu_opencl11)
    add_library(scrapgpu_opencl11 STATIC
                "${CL11_OPENCL11_DIR}/buffer.cpp"
                "${CL11_OPENCL11_DIR}/context.cpp"
                "${CL11_OPENCL11_DIR}/device.cpp"
                "${CL11_OPENCL11_DIR}/device_state.cpp"
                "${CL11_OPENCL11_DIR}/kernels.cpp")
    get_filename_component(CL11_BACKENDS_DIR "${CL11_OPENCL11_DIR}/.." ABSOLUTE)
    target_include_directories(scrapgpu_opencl11 PUBLIC "${CL11_BACKENDS_DIR}" "${CL11_OPENCL11_DIR}")
    target_link_libraries(scrapgpu_opencl11 PUBLIC ${OpenCL_LIBRARIES})
    target_include_directories(scrapgpu_opencl11 PUBLIC ${OpenCL_INCLUDE_DIRS})
    target_compile_features(scrapgpu_opencl11 PUBLIC cxx_std_17)
    target_compile_definitions(scrapgpu_opencl11 PUBLIC
        CL_TARGET_OPENCL_VERSION=110
        CL_USE_DEPRECATED_OPENCL_1_1_APIS)
    set_target_properties(scrapgpu_opencl11 PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        POSITION_INDEPENDENT_CODE ON)
endif()

if (CL11_IN_TREE)
    ggml_add_backend_library(${TARGET_NAME}
                             ggml-cl11.cpp
                             ../../include/ggml-cl11.h)
else()
    # ---- out-of-tree plugin -------------------------------------------------------------------------------------
    set(CL11_LLAMA_CPP_COMMIT   "0faee50"  CACHE INTERNAL "llama.cpp commit this backend is written for")
    set(CL11_LLAMA_CPP_VERSION  "b11384"   CACHE INTERNAL "llama.cpp release this backend is written for")
    set(CL11_GGML_VERSION       "0.25.3"   CACHE INTERNAL "ggml version of that release")

    set(LLAMA_CPP_SOURCE_DIR "" CACHE PATH "llama.cpp source tree (ggml/include and ggml/src are used)")
    set(LLAMA_CPP_BUILD_DIR  "" CACHE PATH "llama.cpp build tree (configured with -DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON)")
    set(GGML_BASE_LIBRARY    "" CACHE FILEPATH "libggml-base (default: searched in LLAMA_CPP_BUILD_DIR)")

    if (NOT LLAMA_CPP_SOURCE_DIR OR NOT EXISTS "${LLAMA_CPP_SOURCE_DIR}/ggml/src/ggml-backend-impl.h")
        message(FATAL_ERROR "set -DLLAMA_CPP_SOURCE_DIR=<llama.cpp checkout> (ggml/src/ggml-backend-impl.h not found)")
    endif()
    get_filename_component(LLAMA_CPP_SOURCE_DIR "${LLAMA_CPP_SOURCE_DIR}" ABSOLUTE)

    # version check: the backend API (ggml-backend-impl.h) changes between llama.cpp releases
    set(_ggml_version "")
    file(STRINGS "${LLAMA_CPP_SOURCE_DIR}/ggml/CMakeLists.txt" _ggml_ver_lines REGEX "^set\\(GGML_VERSION_(MAJOR|MINOR|PATCH) [0-9]+\\)")
    foreach (_l ${_ggml_ver_lines})
        string(REGEX REPLACE "^set\\(GGML_VERSION_[A-Z]+ ([0-9]+)\\)" "\\1" _n "${_l}")
        list(APPEND _ggml_version "${_n}")
    endforeach()
    string(REPLACE ";" "." _ggml_version "${_ggml_version}")
    set(_api_version "?")
    file(STRINGS "${LLAMA_CPP_SOURCE_DIR}/ggml/src/ggml-backend-impl.h" _api_line REGEX "#define GGML_BACKEND_API_VERSION ")
    if (_api_line MATCHES "GGML_BACKEND_API_VERSION ([0-9]+)")
        set(_api_version "${CMAKE_MATCH_1}")
    endif()
    set(_commit "unknown")
    find_package(Git QUIET)
    if (GIT_FOUND)
        execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${LLAMA_CPP_SOURCE_DIR}" rev-parse --short=7 HEAD
                        OUTPUT_VARIABLE _commit OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    endif()
    message(STATUS "CL11: llama.cpp source ${LLAMA_CPP_SOURCE_DIR}: ggml ${_ggml_version}, backend API ${_api_version}, commit ${_commit}")
    if (NOT _ggml_version STREQUAL CL11_GGML_VERSION OR NOT _api_version STREQUAL "3")
        message(WARNING "CL11 is written for llama.cpp ${CL11_LLAMA_CPP_VERSION} (commit ${CL11_LLAMA_CPP_COMMIT}, ggml ${CL11_GGML_VERSION}, "
                        "backend API 3); this tree has ggml ${_ggml_version} / backend API ${_api_version}. "
                        "The ggml backend API is not stable, expect compile errors or a plugin that refuses to load.")
    elseif (NOT _commit STREQUAL CL11_LLAMA_CPP_COMMIT)
        message(WARNING "CL11 is written for llama.cpp commit ${CL11_LLAMA_CPP_COMMIT} (${CL11_LLAMA_CPP_VERSION}), this tree is at ${_commit}. "
                        "The ggml backend API is not stable; if the build or the test-backend-ops run fails, use the matching commit.")
    endif()

    # libggml-base from the llama.cpp build
    if (NOT GGML_BASE_LIBRARY)
        if (NOT LLAMA_CPP_BUILD_DIR)
            message(FATAL_ERROR "set -DLLAMA_CPP_BUILD_DIR=<llama.cpp build dir> (built with -DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON) "
                                "or -DGGML_BASE_LIBRARY=<path to libggml-base>")
        endif()
        get_filename_component(LLAMA_CPP_BUILD_DIR "${LLAMA_CPP_BUILD_DIR}" ABSOLUTE)
        unset(CL11_GGML_BASE_LIB CACHE)
        find_library(CL11_GGML_BASE_LIB NAMES ggml-base
                     HINTS "${LLAMA_CPP_BUILD_DIR}/bin" "${LLAMA_CPP_BUILD_DIR}/lib" "${LLAMA_CPP_BUILD_DIR}/ggml/src"
                     NO_DEFAULT_PATH)
        if (NOT CL11_GGML_BASE_LIB)
            message(FATAL_ERROR "libggml-base not found in ${LLAMA_CPP_BUILD_DIR}/{bin,lib,ggml/src}: build llama.cpp with "
                                "-DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON first (or pass -DGGML_BASE_LIBRARY=...)")
        endif()
        set(GGML_BASE_LIBRARY "${CL11_GGML_BASE_LIB}")
    endif()
    message(STATUS "CL11: linking ${GGML_BASE_LIBRARY}")
    get_filename_component(_base_dir "${GGML_BASE_LIBRARY}" DIRECTORY)

    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
        set(CMAKE_BUILD_TYPE Release)
    endif()

    add_library(${TARGET_NAME} MODULE
                ggml-cl11.cpp
                include/ggml-cl11.h)
    # same definitions as ggml_add_backend_library (ggml/src/CMakeLists.txt) with GGML_BACKEND_DL and BUILD_SHARED_LIBS
    target_compile_definitions(${TARGET_NAME} PRIVATE GGML_BACKEND_DL GGML_BACKEND_BUILD GGML_BACKEND_SHARED)
    target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
    # llama.cpp's headers as system headers: their static helpers would warn (-Wunused-function) in every user
    target_include_directories(${TARGET_NAME} SYSTEM PRIVATE
        "${LLAMA_CPP_SOURCE_DIR}/ggml/include"
        "${LLAMA_CPP_SOURCE_DIR}/ggml/src")
    target_link_libraries(${TARGET_NAME} PRIVATE "${GGML_BASE_LIBRARY}")
    # find libggml-base next to the llama.cpp binaries (it is the same image that the host already loaded)
    set_target_properties(${TARGET_NAME} PROPERTIES
        BUILD_RPATH "${_base_dir}"
        INSTALL_RPATH "${_base_dir}")
    install(TARGETS ${TARGET_NAME} LIBRARY DESTINATION lib)
endif()

target_link_libraries(${TARGET_NAME} PRIVATE scrapgpu_opencl11 scrapgpu_tensor ${OpenCL_LIBRARIES})
target_include_directories(${TARGET_NAME} PRIVATE ${OpenCL_INCLUDE_DIRS})

# the host code only uses the OpenCL 1.1 API (Khronos headers: hide everything newer)
target_compile_definitions(${TARGET_NAME} PRIVATE
    CL_TARGET_OPENCL_VERSION=110
    CL_USE_DEPRECATED_OPENCL_1_1_APIS)

include("${CL11_OPENCL11_DIR}/opencl11_kernels.cmake")
scrapgpu_opencl11_embed_kernels(scrapgpu_opencl11 "${CL11_OPENCL11_DIR}" "${CMAKE_CURRENT_BINARY_DIR}")

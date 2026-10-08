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
    target_include_directories(${TARGET_NAME} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${LLAMA_CPP_SOURCE_DIR}/ggml/include"
        "${LLAMA_CPP_SOURCE_DIR}/ggml/src")
    target_link_libraries(${TARGET_NAME} PRIVATE "${GGML_BASE_LIBRARY}")
    # find libggml-base next to the llama.cpp binaries (it is the same image that the host already loaded)
    set_target_properties(${TARGET_NAME} PROPERTIES
        BUILD_RPATH "${_base_dir}"
        INSTALL_RPATH "${_base_dir}")
    install(TARGETS ${TARGET_NAME} LIBRARY DESTINATION lib)
endif()

target_link_libraries(${TARGET_NAME} PRIVATE ${OpenCL_LIBRARIES})
target_include_directories(${TARGET_NAME} PRIVATE ${OpenCL_INCLUDE_DIRS})

# the host code only uses the OpenCL 1.1 API (Khronos headers: hide everything newer)
target_compile_definitions(${TARGET_NAME} PRIVATE
    CL_TARGET_OPENCL_VERSION=110
    CL_USE_DEPRECATED_OPENCL_1_1_APIS)

# Embed the kernel sources (.cl) as C++ raw string literals into an autogenerated header.
# The header is regenerated at configure time, and cmake re-runs when a .cl file changes.
set(GGML_CL11_KERNEL_FILES
    common.cl
    elementwise.cl
    norm_rope.cl
    get_set_rows.cl
    mul_mv.cl
    mul_mv_r.cl
    mul_mv_k.cl
    attn.cl
    mul_mv_xcommon.cl
    mvx_defs_staged.cl
    mul_mv_x.cl
    mvx_defs_global.cl
    mul_mv_x.cl
    mul_mm.cl
)

set(GGML_CL11_KERNEL_HDR "${CMAKE_CURRENT_BINARY_DIR}/autogenerated/ggml-cl11-kernels.h")
set(GGML_CL11_HDR_CONTENT "// generated by CMake from kernels/*.cl, do not edit\n")
set(GGML_CL11_SRC_LIST "")
set(GGML_CL11_NAME_LIST "")
set(KINDEX 0)
foreach (KFILE ${GGML_CL11_KERNEL_FILES})
    set(KPATH "${CMAKE_CURRENT_SOURCE_DIR}/kernels/${KFILE}")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${KPATH}")
    file(READ "${KPATH}" KCONTENT)
    # split the source into one raw string literal per line (keeps every literal short, which some compilers require)
    string(REPLACE "\n" ")CL11SRC\"\nR\"CL11SRC(\n" KCONTENT "${KCONTENT}")
    math(EXPR KINDEX "${KINDEX} + 1")
    string(MAKE_C_IDENTIFIER "${KFILE}" KID)
    set(KID "${KINDEX}_${KID}")
    string(APPEND GGML_CL11_HDR_CONTENT "static const char * const ggml_cl11_src_${KID} =\nR\"CL11SRC(${KCONTENT})CL11SRC\";\n\n")
    string(APPEND GGML_CL11_SRC_LIST "    ggml_cl11_src_${KID},\n")
    string(APPEND GGML_CL11_NAME_LIST "    \"${KFILE}\",\n")
endforeach()
string(APPEND GGML_CL11_HDR_CONTENT "static const char * ggml_cl11_sources[] = {\n${GGML_CL11_SRC_LIST}};\n")
string(APPEND GGML_CL11_HDR_CONTENT "static const char * const ggml_cl11_source_names[] = {\n${GGML_CL11_NAME_LIST}};\n")

set(GGML_CL11_OLD_CONTENT "")
if (EXISTS "${GGML_CL11_KERNEL_HDR}")
    file(READ "${GGML_CL11_KERNEL_HDR}" GGML_CL11_OLD_CONTENT)
endif()
if (NOT "${GGML_CL11_OLD_CONTENT}" STREQUAL "${GGML_CL11_HDR_CONTENT}")
    file(WRITE "${GGML_CL11_KERNEL_HDR}" "${GGML_CL11_HDR_CONTENT}")
endif()
target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/autogenerated")

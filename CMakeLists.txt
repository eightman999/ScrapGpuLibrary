# Thin wrapper so that `cmake -S . -B build -DLLAMA_CPP_SOURCE_DIR=... -DLLAMA_CPP_BUILD_DIR=...` builds
# the CL11 ggml plugin (adapters/ggml). See adapters/ggml/llama.cpp/README.md.
cmake_minimum_required(VERSION 3.14)
project(ScrapGpuLibrary C CXX)
add_subdirectory(adapters/ggml)

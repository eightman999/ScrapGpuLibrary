#!/bin/bash
set -u
cd /home/eightman/dev/tools/ScrapGpuLibrary-heterogeneous-20261011
mkdir -p results-final
lock=/home/eightman/bench/gpu-bench.lock
plugin="$PWD/build/adapters/ggml/libggml-cl11.so"
model=/home/eightman/bench/models/qwen3-0.6b-base-q4_k_m.gguf
run() {
    local tag=$1; shift
    flock "$lock" timeout 180 "$@" >"results-final/$tag.log" 2>&1
    local rc=$?
    printf '%s\t%s\n' "$tag" "$rc" | tee -a results-final/status.tsv
}
run cpu-probe ./build/scrapgpu_compat_probe cpu
run cpu-boundary ./build/scrapgpu_cpu_boundary_tests
run cpu-attention ./build/scrapgpu_attention_tests
run cpu-generation env -u GGML_BACKEND_PATH ./llama-build/bin/llama-completion -m "$model" -ngl 0 -t 2 -b 64 -ub 64 -c 256 -n 16 --temp 0 --seed 1 -no-cnv --no-display-prompt --simple-io -p 'The capital of France is'
for spec in 'gt730|GT 730|NVIDIA' 'gt430|GT 430|NVIDIA' 'gt710|GT 710|NVIDIA' 'wx2100|WX 2100 rusticl|rusticl' 'rx6400|RX 6400 rusticl|rusticl'; do
    IFS='|' read -r tag gpu platform <<< "$spec"
    wrapper=(env "GGML_CL11_PLATFORMS=$platform" ./src/adapters/ggml/tools/kepler-sweep/gpu_exec.sh "$gpu")
    run "$tag-probe" "${wrapper[@]}" ./build/scrapgpu_compat_probe opencl11
    run "$tag-attention" "${wrapper[@]}" env SCRAPGPU_TEST_BACKEND=opencl11 ./build/scrapgpu_attention_tests
    run "$tag-ops" "${wrapper[@]}" ./build/scrapgpu_ops_tests
    run "$tag-llama-add" "${wrapper[@]}" env "GGML_BACKEND_PATH=$plugin" ./llama-build/bin/test-backend-ops -b CL11_0 -o ADD -p 'type=f32.*ne=\[32,1,1,1\]'
    run "$tag-generation" "${wrapper[@]}" env "GGML_BACKEND_PATH=$plugin" ./llama-build/bin/llama-completion -m "$model" -dev CL11_0 -ngl 99 -t 2 -b 64 -ub 64 -c 256 -n 16 --temp 0 --seed 1 -fa off -no-cnv --no-display-prompt --simple-io -p 'The capital of France is'
    run "$tag-bench" "${wrapper[@]}" env "GGML_BACKEND_PATH=$plugin" ./llama-build/bin/llama-bench -m "$model" -dev CL11_0 -ngl 99 -t 2 -b 64 -ub 64 -p 32 -n 16 -r 2 -fa off -o json
 done

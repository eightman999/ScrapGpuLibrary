#!/bin/bash
set -u
cd /home/eightman/dev/tools/ScrapGpuLibrary-heterogeneous-20261011
for spec in 'gt730|GT 730|NVIDIA' 'gt430|GT 430|NVIDIA' 'gt710|GT 710|NVIDIA' 'wx2100|WX 2100 rusticl|rusticl' 'rx6400|RX 6400 rusticl|rusticl'; do
 IFS='|' read -r tag gpu platform <<< "$spec"
 for op in ADD MUL_MAT; do
  if [ "$op" = ADD ]; then filter='type=f32.*nf=1,'; else filter='type_a=(f32|f16|q4_0|q8_0|q4_K|q6_K),type_b=f32,m=16,n=(1|4),k=256,'; fi
  flock /home/eightman/bench/gpu-bench.lock timeout 180 env GGML_CL11_PLATFORMS="$platform" GGML_BACKEND_PATH="$PWD/build/adapters/ggml/libggml-cl11.so" ./src/adapters/ggml/tools/kepler-sweep/gpu_exec.sh "$gpu" ./llama-build/bin/test-backend-ops -b CL11_0 -o "$op" -p "$filter" >"results-final/$tag-llama-$op.log" 2>&1
  printf '%s-llama-%s\t%s\n' "$tag" "$op" "$?" | tee -a results-final/ops-status.tsv
 done
 done

#!/bin/bash
# kepler-sweep: build the plugin of this branch and measure everything that is new on the real GPUs (GT 730 / GT 710 / GT 430 of the TB250):
#   stage build   the plugin (cmake) and the tools (mm_bench, mvm_check, attn_check)
#   stage test    test-backend-ops (all ops: default tuning and the fusions forced; -o MUL_MAT for every mat-mat variant), the graph checks
#                 (mvm_check: multi mat-vec + paired SET_ROWS; attn_check)                                                     -> test.csv
#   stage greedy  64 greedy tokens of Qwen3-0.6B Q8_0 (and Q4_K_M if there is a model) vs the CPU output, kernels per token      -> greedy.csv
#   stage mm      mat-mat throughput of every variant: tools/mm_bench (n=128 and n=512 shapes of the models and of the perf suite) -> mm.csv,
#                 `test-backend-ops perf -o MUL_MAT` (n=512 of the suite)                                                       -> perf.csv
#   stage bench   llama-bench pp128 / tg64: main's plugin (before), this plugin with the fusions off / each one / both, and the best mat-mat
#                 variants of the mm stage (fusions on)                                                                          -> bench.csv
#   stage dn_test  GT 430 only: the Fermi "denormal products" mat-vec (docs/CL11.md, GT 430 section): test-backend-ops (MUL_MAT and all ops) and the graph
#                 checks with DN on / off (GGML_CL11_MVR_DN=1 / 0)                                                               -> test.csv
#   stage dn_mv    mat-vec GB/s of Q4_0 / Q4_K / Q6_K (and Q8_0 as the control, which has no DN kernel), DN on / off              -> dn_mv.csv
#   stage dn_greedy 64 greedy tokens, DN on / off (Q8_0 vs the CPU; Q4_K_M and the optional Q4_0 model vs the output with DN off)  -> greedy.csv
#   stage dn_kl    KL divergence of the Q4_K_M model against the CPU of the official llama.cpp (llama-perplexity --kl-divergence-base /
#                 --kl-divergence, -c 512 --chunks 8; tolerance mean <= 0.0070, 99% <= 0.033, Same top p >= 95.0), DN on / off      -> kl.csv
#   stage dn_bench llama-bench Q8_0 / Q4_K_M (/ Q4_0), DN on / off, main's plugin as the "before"                                -> bench.csv
#                 (`dn` = the five dn_* stages)
#   stage regress every GPU (GT 430, GT 730, GT 710, WX 2100 Clover, WX 2100 rusticl, RX 6400 rusticl), default configuration: test-backend-ops,
#                 greedy (Q8_0 vs the CPU, Q4_K_M vs main's plugin), llama-bench Q8_0 / Q4_K_M, compared with baseline.csv (the reference
#                 table of README.md and docs/CL11.md; more than the tolerance below it = REGRESSION)                           -> regress.csv
#   stage summary summary.md from the csv files
#
# Usage (on the TB250, the branch checked out in ~/bench/scrapgpu-kepler):
#   cd ~/bench/scrapgpu-kepler && adapters/ggml/tools/kepler-sweep/sweep.sh                  # everything, the three GPUs, one after the other
#   GPUS="GT 710" STAGES="build test greedy" adapters/ggml/tools/kepler-sweep/sweep.sh       # a part
#   STAGES="build dn summary" adapters/ggml/tools/kepler-sweep/sweep.sh                      # the GT 430 denormal-products checks (DN_GPUS="GT 430")
#   STAGES="build regress summary" adapters/ggml/tools/kepler-sweep/sweep.sh                 # the regression check of all the GPUs
#   DRY_RUN=1 adapters/ggml/tools/kepler-sweep/sweep.sh                                      # prints every GPU / CPU command (and $OUT/dryrun.txt), runs nothing
# Every GPU command runs as `flock $LOCK gpu_exec.sh "<GPU>" ...`: it waits for the lock (another job may use the GPUs), waits while the GPU is
# at THERMAL_MAX (85) Celsius or more, and selects the one GPU with GGML_CL11_DEVICES. The results are in $OUT (default: sweep-out/<time> in the checkout).
#
# Environment (defaults in parentheses):
#   ROOT           the checkout of this branch (the repository that holds this script)
#   LLAMA          pristine llama.cpp b11384 with the GGML_BACKEND_DL=ON build in $LLAMA/build-dl (~/bench/scrapgpu/llama.cpp)
#   BASE_PLUGIN    the plugin of main, the "before" of the bench stage (~/bench/scrapgpu/build-cl11/libggml-cl11.so; skipped if missing)
#   MODEL_Q8       Qwen3-0.6B-Q8_0.gguf (searched in ~/bench/*/ and ~/models/);  MODEL_Q4K  Qwen3-0.6B-Q4_K_M.gguf (optional)
#   REF_Q8         the CPU output of the greedy test (/tmp/cl11_out_none.txt)
#   GPUS           comma separated substrings of the OpenCL device names ("GT 730,GT 710,GT 430")
#   STAGES         ("build test greedy mm bench dn regress summary")
#   DN_GPUS        the GPUs of the dn stages ("GT 430"; the denormal products are tuned for Fermi, other GPUs run the same comparison as a control)
#   REGRESS_GPUS   comma separated targets of the regress stage ("GT 430,GT 730,GT 710,WX 2100 clover,WX 2100 rusticl,RX 6400 rusticl"). A Radeon is
#                  "<name> clover|rusticl": rusticl = RUSTICL_ENABLE=radeonsi and `-dev CL11_0` for llama.cpp (Clover is also visible then, as CL11_1),
#                  clover = RUSTICL_ENABLE unset. The RX 6400 hangs on Clover and is refused there. Use `GPUS=...` names without a suffix elsewhere.
#   REGRESS_TOL    allowed loss in percent against baseline.csv (its tol_pct column when there is one)
#   MODEL_Q40      an optional Q4_0 model for the dn stages (Qwen2-0.5B-ft Q4_0 or Qwen3-0.6B-Q4_0.gguf, searched like the others)
#   PPL_TEXT       the text of the KL divergence (default: README.md + docs/*.md of the checkout, the first 4096 tokens are used by -c 512 --chunks 8)
#   KL_BASE        the CPU logits (default $OUT/kl/base.kld; an existing file is reused), KL_ARGS  extra llama-perplexity arguments ("-b 128 -ub 128 -t 2")
#   KL_MEAN_MAX / KL_P99_MAX / KL_TOP_MIN  the tolerance (0.0070 / 0.033 / 95.0)
#   DRY_RUN        1: print the commands instead of running them
#   MM_VARIANTS    mat-mat variants (presets of cl11_mmx_presets or key=value lists, see docs/CL11.md), space separated; `base` = k_mm_*
#   TEST_VARIANTS  the variants whose MUL_MAT is run through test-backend-ops (MM_VARIANTS without base)
#   GREEDY_VARIANTS  the variants of the greedy stage ("x88 x164 x168 x88k x88d")
#   BENCH_TOP      how many of the best variants per GPU the bench stage takes from mm.csv (3)
#   BENCH_REPS     llama-bench repetitions (3);  JOBS  compiler jobs (2);  LOCK  (~/bench/gpu-bench.lock);  THERMAL_MAX (85)
set -u

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${ROOT:-$(cd "$HERE/../../../.." && pwd)}
LLAMA=${LLAMA:-$HOME/bench/scrapgpu/llama.cpp}
LLAMA_BIN=${LLAMA_BIN:-$LLAMA/build-dl/bin}
BUILD=${BUILD:-$ROOT/build-cl11}
PLUGIN=${PLUGIN:-$BUILD/libggml-cl11.so}
BASE_PLUGIN=${BASE_PLUGIN:-$HOME/bench/scrapgpu/build-cl11/libggml-cl11.so}
LOCK=${LOCK:-$HOME/bench/gpu-bench.lock}
GPUS=${GPUS:-"GT 730,GT 710,GT 430"}
STAGES=${STAGES:-"build test greedy mm bench dn regress summary"}
DRY_RUN=${DRY_RUN:-0}
DN_GPUS=${DN_GPUS:-"GT 430"}
REGRESS_GPUS=${REGRESS_GPUS:-"GT 430,GT 730,GT 710,WX 2100 clover,WX 2100 rusticl,RX 6400 rusticl"}
KL_ARGS=${KL_ARGS:--b 128 -ub 128 -t 2}
OUT=${OUT:-$ROOT/sweep-out/$(date +%Y%m%d-%H%M%S)}
BIN=${BIN:-$ROOT/sweep-bin}
JOBS=${JOBS:-2}
BENCH_REPS=${BENCH_REPS:-3}
BENCH_TOP=${BENCH_TOP:-3}
REF_Q8=${REF_Q8:-/tmp/cl11_out_none.txt}
MM_VARIANTS=${MM_VARIANTS:-"base x84 x84p x88 x88w x88s x88k x88d x164 x168 x168s x168w x88g x88,ku=4 x88,maxrreg=128 x88,maxrreg=168 x168,maxrreg=168 x88w,pf=2"}
TEST_VARIANTS=${TEST_VARIANTS:-$(for v in $MM_VARIANTS; do [ "$v" = base ] || printf '%s ' "$v"; done)}
GREEDY_VARIANTS=${GREEDY_VARIANTS:-"x88 x164 x168 x88k x88d"}
# the models: pass the files explicitly (MODEL_Q8, MODEL_Q4K, MODEL_Q40); the search below only finds the usual names, in any letter case
find_model() { local pat; for pat in "$@"; do find "$HOME/bench" "$HOME/models" -maxdepth 2 -iname "$pat" 2>/dev/null | head -1; done | head -1; }
MODEL_Q8=${MODEL_Q8:-$(find_model 'Qwen3-0.6B-Q8_0.gguf')}
MODEL_Q4K=${MODEL_Q4K:-$(find_model 'Qwen3-0.6B-Q4_K_M.gguf' 'Qwen3-0.6B-Base-Q4_K_M.gguf')}
MODEL_Q40=${MODEL_Q40:-$(find_model 'Qwen3-0.6B-Q4_0.gguf' '*-Q4_0.gguf')}
PROMPT="The capital of France is Paris. The capital of Japan is"
if [ "$DRY_RUN" = 1 ]; then # the commands are printed, the files need not exist
    MODEL_Q8=${MODEL_Q8:-"~/bench/models/Qwen3-0.6B-Q8_0.gguf"}; MODEL_Q4K=${MODEL_Q4K:-"~/bench/models/Qwen3-0.6B-Q4_K_M.gguf"}; MODEL_Q40=${MODEL_Q40:-"~/bench/models/Qwen2-0.5B-ft-Q4_0.gguf"}
else # a stage without its model would silently produce empty rows (it happened): stop here unless ALLOW_MISSING_MODELS=1
    for mv in MODEL_Q8 MODEL_Q4K MODEL_Q40; do
        if [ ! -s "${!mv}" ] && [ "${ALLOW_MISSING_MODELS:-0}" != 1 ]; then
            echo "sweep.sh: $mv is not set or not a file ('${!mv}'); pass the model files (MODEL_Q8, MODEL_Q4K, MODEL_Q40) or set ALLOW_MISSING_MODELS=1" >&2
            exit 1
        fi
    done
fi

# the shapes of the mat-mat sweep: the matrices of Qwen3-0.6B with a 128-column prompt batch, and the suite of test-backend-ops perf (m=4096, k=14336, n=512)
MM_SHAPES_N128="3072:128:1024 1024:128:3072"
MM_SHAPES_N512="4096:512:14336"
MM_TYPES="q8_0 q4_0 q4_k q6_k f16 f32"
# the matrices of the dn_mv stage (m:k): the suite's mat-vec (m=4096, k=14336) and the FFN gate / down of Qwen3-0.6B
DN_MV_SHAPES="4096:14336 3072:1024 1024:3072"

if command -v flock >/dev/null 2>&1; then LOCKCMD=(flock "$LOCK"); else LOCKCMD=(); [ "$DRY_RUN" = 1 ] || echo "sweep: flock not found, running without the lock" >&2; fi
if [ "$DRY_RUN" = 1 ]; then OUT=${OUT_DRY:-$OUT-dry}; else mkdir -p "$(dirname "$LOCK")"; fi
mkdir -p "$OUT/logs" "$BIN"
log() { echo "[sweep $(date +%H:%M:%S)] $*" >&2; }
has_stage() { case " $STAGES " in *" $1 "*) return 0;; esac; return 1; }
safe() { echo "$1" | tr -c 'A-Za-z0-9._\n-' '_'; }
csv_header() { [ -s "$1" ] || echo "$2" > "$1"; }
csvq() { local v=${1//\"/\"\"}; printf '"%s"' "$v"; }   # a CSV field that may contain commas (variant names such as "x88,ku=4")

# run a command on one GPU: lock, cool down, GGML_CL11_DEVICES. usage: run_gpu <gpu> <env assignments (a string)> <command...>
run_gpu() {
    local gpu="$1" envs="$2"; shift 2
    if [ "$DRY_RUN" = 1 ]; then
        echo "flock $LOCK gpu_exec.sh '$gpu' env GGML_BACKEND_PATH=${PLUGIN_USED:-$PLUGIN} GGML_LIB_DIR=$LLAMA_BIN $envs $*" | tee -a "$OUT/dryrun.txt" >&2
        return 0
    fi
    # shellcheck disable=SC2086
    ${LOCKCMD[@]+"${LOCKCMD[@]}"} "$HERE/gpu_exec.sh" "$gpu" env GGML_BACKEND_PATH="${PLUGIN_USED:-$PLUGIN}" GGML_LIB_DIR="$LLAMA_BIN" $envs "$@"
}

# a command that uses no GPU (the CPU reference of the KL divergence), under the same lock
run_cpu() {
    if [ "$DRY_RUN" = 1 ]; then
        echo "flock $LOCK $*" | tee -a "$OUT/dryrun.txt" >&2
        return 0
    fi
    ${LOCKCMD[@]+"${LOCKCMD[@]}"} "$@"
}

# a file the stage needs: always "there" in a dry run (the commands are printed)
have_file() {
    [ -n "$1" ] || return 1
    [ "$DRY_RUN" = 1 ] && return 0
    [ -s "$1" ]
}

# llama.cpp device argument of a target: with RUSTICL_ENABLE=radeonsi Mesa Clover is also visible (CL11_1), rusticl is CL11_0
devargs_for() { case "$1" in *" rusticl") echo "-dev CL11_0";; *) echo "";; esac; }
# the name of the GPU without the platform suffix
gpu_name() { local g="${1% rusticl}"; echo "${g% clover}"; }

# the environment of a configuration for a GPU (Fermi: the constant-memory src1 path of the mat-vec excludes the fusion, so it is switched off there)
cfg_env() {
    local name="$1" gpu="$2" extra=""
    case "$gpu" in *430*) extra="GGML_CL11_MVR_CONST_X=0";; esac
    case "$name" in
        default|main) echo "";;
        dn)      echo "GGML_CL11_MVR_DN=1";;
        nodn)    echo "GGML_CL11_MVR_DN=0";;
        base)    echo "GGML_CL11_MULTI_MV=0 GGML_CL11_FUSE_SET_ROWS=0 GGML_CL11_MM_VARIANT=base";;
        multi)   echo "GGML_CL11_MULTI_MV=1 GGML_CL11_FUSE_SET_ROWS=0 GGML_CL11_MM_VARIANT=base $extra";;
        setrows) echo "GGML_CL11_MULTI_MV=0 GGML_CL11_FUSE_SET_ROWS=1 GGML_CL11_MM_VARIANT=base";;
        fused)   echo "GGML_CL11_MULTI_MV=1 GGML_CL11_FUSE_SET_ROWS=1 GGML_CL11_MM_VARIANT=base $extra";;
        mm:*)    echo "GGML_CL11_MULTI_MV=1 GGML_CL11_FUSE_SET_ROWS=1 GGML_CL11_MM_VARIANT=${name#mm:} $extra";;
        *)       echo "sweep: unknown configuration $name" >&2; return 1;;
    esac
}

#
# build
#
stage_build() {
    if [ "$DRY_RUN" = 1 ]; then log "build: (dry run) cmake -S $ROOT/adapters/ggml -B $BUILD, tools mm_bench mvm_check attn_check -> $BIN"; return 0; fi
    log "build: plugin in $BUILD"
    cmake -S "$ROOT/adapters/ggml" -B "$BUILD" -DLLAMA_CPP_SOURCE_DIR="$LLAMA" -DLLAMA_CPP_BUILD_DIR="$LLAMA/build-dl" -DCMAKE_BUILD_TYPE=Release > "$OUT/logs/build.log" 2>&1 \
        && cmake --build "$BUILD" -j"$JOBS" >> "$OUT/logs/build.log" 2>&1 || { log "build FAILED, see $OUT/logs/build.log"; tail -20 "$OUT/logs/build.log" >&2; exit 1; }
    for t in mm_bench mvm_check attn_check; do
        c++ -std=c++17 -O1 -I"$LLAMA/ggml/include" "$ROOT/adapters/ggml/tools/$t.cpp" -L"$LLAMA_BIN" -lggml -lggml-base -Wl,-rpath,"$LLAMA_BIN" -o "$BIN/$t" >> "$OUT/logs/build.log" 2>&1 \
            || { log "build of $t FAILED, see $OUT/logs/build.log"; exit 1; }
    done
    log "build: done ($PLUGIN)"
}

#
# test: test-backend-ops and the graph checks
#
test_row() { # gpu config log
    local pt
    pt=$("$HERE/summarize.py" test "$3")
    echo "$1,$(csvq "$2"),${pt% *},${pt#* }" >> "$OUT/test.csv"
    log "test $1 $2: ${pt% *}/${pt#* }"
}

stage_test() {
    csv_header "$OUT/test.csv" "gpu,config,passed,total"
    local gpu
    IFS=',' read -ra gpu_list <<< "$GPUS"
    for gpu in "${gpu_list[@]}"; do
        local g; g=$(safe "$gpu")
        local cfg lf
        for cfg in default fused; do
            lf="$OUT/logs/test_${g}_${cfg}.log"
            run_gpu "$gpu" "$(cfg_env $cfg "$gpu")" "$LLAMA_BIN/test-backend-ops" test -b CL11_0 > "$lf" 2>&1
            test_row "$gpu" "test-backend-ops $cfg" "$lf"
        done
        for cfg in default fused; do
            lf="$OUT/logs/test_${g}_mvm_check_${cfg}.log"
            run_gpu "$gpu" "$(cfg_env $cfg "$gpu")" "$BIN/mvm_check" > "$lf" 2>&1
            echo "$gpu,mvm_check $cfg,$(grep -c ' OK$' "$lf"),$(grep -cE ' (OK|BAD)$' "$lf")" >> "$OUT/test.csv"
            log "test $gpu mvm_check $cfg: $(grep -c ' OK$' "$lf") OK, $(grep -c 'BAD$' "$lf") BAD"
        done
        lf="$OUT/logs/test_${g}_attn_check.log"
        run_gpu "$gpu" "" "$BIN/attn_check" > "$lf" 2>&1
        echo "$gpu,attn_check,$(grep -c ' OK$' "$lf"),$(grep -cE ' (OK|BAD)$' "$lf")" >> "$OUT/test.csv"
        local v
        for v in $TEST_VARIANTS; do
            lf="$OUT/logs/test_${g}_mm_$(safe "$v").log"
            run_gpu "$gpu" "$(cfg_env "mm:$v" "$gpu")" "$LLAMA_BIN/test-backend-ops" test -b CL11_0 -o MUL_MAT > "$lf" 2>&1
            test_row "$gpu" "MUL_MAT variant $v" "$lf"
        done
    done
}

#
# greedy: 64 tokens, the same command as in docs/CL11.md
#
greedy_run() { # gpu model_path model_label config ref
    local gpu="$1" model="$2" label="$3" cfg="$4" ref="$5" g; g=$(safe "$gpu")
    local o="$OUT/logs/greedy_${g}_${label}_$(safe "$cfg").out" e="$OUT/logs/greedy_${g}_${label}_$(safe "$cfg").err"
    run_gpu "$gpu" "$(cfg_env "$cfg" "$gpu") GGML_CL11_LAUNCH_LOG=1" "$LLAMA_BIN/llama-completion" $(devargs_for "$gpu") -m "$model" -n 64 --temp 0 -s 1 -ngl 99 -no-cnv --no-warmup -c 512 -b 128 -ub 128 -p "$PROMPT" > "$o" 2> "$e"
    local same=no
    if [ "$ref" = /dev/null ]; then same=reference; elif [ -s "$ref" ] && cmp -s "$o" "$ref"; then same=yes; fi
    [ "$DRY_RUN" = 1 ] && same="(dry run)"
    GREEDY_LAST=$same
    echo "$gpu,$label,$(csvq "$cfg"),$same,$("$HERE/summarize.py" kernels "$e")" >> "$OUT/greedy.csv"
    log "greedy $gpu $label $cfg: identical=$same"
}

stage_greedy() {
    csv_header "$OUT/greedy.csv" "gpu,model,config,identical,kernels_per_token"
    local gpu v cfg
    IFS=',' read -ra gpu_list <<< "$GPUS"
    for gpu in "${gpu_list[@]}"; do
        if have_file "$MODEL_Q8" && have_file "$REF_Q8"; then
            for cfg in default base fused; do greedy_run "$gpu" "$MODEL_Q8" q8_0 "$cfg" "$REF_Q8"; done
            for v in $GREEDY_VARIANTS; do greedy_run "$gpu" "$MODEL_Q8" q8_0 "mm:$v" "$REF_Q8"; done
        else
            log "greedy: MODEL_Q8 ('$MODEL_Q8') or REF_Q8 ('$REF_Q8') missing, skipped"
        fi
        if have_file "$MODEL_Q4K"; then
            # the CPU quantizes the activations, so the K-quant text is compared with this GPU's output without the fusions
            greedy_run "$gpu" "$MODEL_Q4K" q4_k_m base /dev/null
            cp "$OUT/logs/greedy_$(safe "$gpu")_q4_k_m_base.out" "$OUT/logs/ref_q4k_$(safe "$gpu").out"
            for cfg in default fused; do greedy_run "$gpu" "$MODEL_Q4K" q4_k_m "$cfg" "$OUT/logs/ref_q4k_$(safe "$gpu").out"; done
            for v in $GREEDY_VARIANTS; do greedy_run "$gpu" "$MODEL_Q4K" q4_k_m "mm:$v" "$OUT/logs/ref_q4k_$(safe "$gpu").out"; done
        fi
    done
}

#
# mm: mat-mat throughput of the variants
#
stage_mm() {
    csv_header "$OUT/mm.csv" "gpu,variant,type,m,n,k,iters,ms,gflops"
    csv_header "$OUT/perf.csv" "gpu,variant,type,m,n,k,gflops"
    local gpu v t shapes
    IFS=',' read -ra gpu_list <<< "$GPUS"
    for gpu in "${gpu_list[@]}"; do
        local g; g=$(safe "$gpu")
        for v in $MM_VARIANTS; do
            local jobs=()
            for t in $MM_TYPES; do
                for shapes in $MM_SHAPES_N128 $MM_SHAPES_N512; do jobs+=("$t:$shapes"); done
            done
            local lf="$OUT/logs/mm_${g}_$(safe "$v").csv"
            run_gpu "$gpu" "$(cfg_env "mm:$v" "$gpu")" "$BIN/mm_bench" "${jobs[@]}" > "$lf" 2> "$lf.err"
            while IFS=, read -r ty m n k iters ms gf; do
                [ -n "$gf" ] && echo "$gpu,$(csvq "$v"),$ty,$m,$n,$k,$iters,$ms,$gf" >> "$OUT/mm.csv"
            done < "$lf"
            log "mm $gpu $v: $(wc -l < "$lf") results"
            # the official perf suite (n=512 only)
            local pf="$OUT/logs/perf_${g}_$(safe "$v").log"
            run_gpu "$gpu" "$(cfg_env "mm:$v" "$gpu")" "$LLAMA_BIN/test-backend-ops" perf -b CL11_0 -o MUL_MAT -p "n=512," > "$pf" 2>&1
            "$HERE/summarize.py" perf "$pf" "$gpu" "$v" >> "$OUT/perf.csv"
        done
    done
}

#
# bench: llama-bench
#
bench_run() { # gpu model_path label config plugin
    local gpu="$1" model="$2" label="$3" cfg="$4" plugin="$5" g; g=$(safe "$gpu")
    local csvf="$OUT/logs/bench_${g}_${label}_$(safe "$cfg").csv"
    PLUGIN_USED="$plugin" run_gpu "$gpu" "$(cfg_env "${cfg%%@*}" "$gpu")" "$LLAMA_BIN/llama-bench" $(devargs_for "$gpu") -m "$model" -ngl 99 -ub 128 -b 128 -p 128 -n 64 -r "$BENCH_REPS" -t 2 -o csv > "$csvf" 2> "$csvf.err"
    local r; r=$("$HERE/summarize.py" bench "$csvf")
    BENCH_PP=${r% *}; BENCH_TG=${r#* }
    echo "$gpu,$label,$(csvq "$cfg"),${r% *},${r#* }" >> "$OUT/bench.csv"
    log "bench $gpu $label $cfg: pp128 ${r% *} tg64 ${r#* }"
}

stage_bench() {
    csv_header "$OUT/bench.csv" "gpu,model,config,pp128,tg64"
    local gpu label model
    IFS=',' read -ra gpu_list <<< "$GPUS"
    for gpu in "${gpu_list[@]}"; do
        for entry in "q8_0:$MODEL_Q8" "q4_k_m:$MODEL_Q4K"; do
            label=${entry%%:*}; model=${entry#*:}
            have_file "$model" || { log "bench: no model for $label, skipped"; continue; }
            if have_file "$BASE_PLUGIN"; then bench_run "$gpu" "$model" "$label" "main" "$BASE_PLUGIN"; fi
            bench_run "$gpu" "$model" "$label" base "$PLUGIN"
            local cfg
            for cfg in multi setrows fused default; do bench_run "$gpu" "$model" "$label" "$cfg" "$PLUGIN"; done
            local v
            for v in $("$HERE/summarize.py" best "$OUT/mm.csv" "$gpu" 128 "$BENCH_TOP"); do
                bench_run "$gpu" "$model" "$label" "mm:$v" "$PLUGIN"
            done
        done
    done
}

#
# dn_*: the Fermi "denormal products" kernels (cfg.mvr_dn, GGML_CL11_MVR_DN) on the GT 430, switched on (dn) and off (nodn)
#
stage_dn_test() {
    csv_header "$OUT/test.csv" "gpu,config,passed,total"
    local gpu lf g cfg
    IFS=',' read -ra gpu_list <<< "$DN_GPUS"
    for gpu in "${gpu_list[@]}"; do
        g=$(safe "$gpu")
        for cfg in dn nodn; do
            lf="$OUT/logs/dn_test_${g}_${cfg}_mulmat.log"
            run_gpu "$gpu" "$(cfg_env $cfg "$gpu")" "$LLAMA_BIN/test-backend-ops" test -b CL11_0 -o MUL_MAT > "$lf" 2>&1
            test_row "$gpu" "MUL_MAT $cfg" "$lf"
            lf="$OUT/logs/dn_test_${g}_${cfg}_full.log"
            run_gpu "$gpu" "$(cfg_env $cfg "$gpu")" "$LLAMA_BIN/test-backend-ops" test -b CL11_0 > "$lf" 2>&1
            test_row "$gpu" "test-backend-ops $cfg (all ops)" "$lf"
            lf="$OUT/logs/dn_test_${g}_${cfg}_mvm_check.log"
            run_gpu "$gpu" "$(cfg_env $cfg "$gpu")" "$BIN/mvm_check" > "$lf" 2>&1
            echo "$gpu,mvm_check $cfg,$(grep -c ' OK$' "$lf"),$(grep -cE ' (OK|BAD)$' "$lf")" >> "$OUT/test.csv"
            lf="$OUT/logs/dn_test_${g}_${cfg}_attn_check.log"
            run_gpu "$gpu" "$(cfg_env $cfg "$gpu")" "$BIN/attn_check" > "$lf" 2>&1
            echo "$gpu,attn_check $cfg,$(grep -c ' OK$' "$lf"),$(grep -cE ' (OK|BAD)$' "$lf")" >> "$OUT/test.csv"
        done
    done
}

# mat-vec of one column (n=1): GB/s of the weights. Q8_0 has no DN kernel (the control: the same with dn and nodn).
stage_dn_mv() {
    csv_header "$OUT/dn_mv.csv" "gpu,config,type,m,k,ms,gbs"
    local gpu g cfg t shp jobs lf
    IFS=',' read -ra gpu_list <<< "$DN_GPUS"
    for gpu in "${gpu_list[@]}"; do
        g=$(safe "$gpu")
        jobs=()
        for t in q4_0 q4_k q6_k q8_0; do
            for shp in $DN_MV_SHAPES; do jobs+=("$t:${shp%%:*}:1:${shp#*:}"); done
        done
        for cfg in dn nodn; do
            lf="$OUT/logs/dn_mv_${g}_${cfg}.csv"
            run_gpu "$gpu" "$(cfg_env $cfg "$gpu")" "$BIN/mm_bench" "${jobs[@]}" > "$lf" 2> "$lf.err"
            "$HERE/summarize.py" mvgbs "$lf" "$gpu" "$cfg" >> "$OUT/dn_mv.csv"
            log "dn_mv $gpu $cfg: $(wc -l < "$lf") results"
        done
    done
}

stage_dn_greedy() {
    csv_header "$OUT/greedy.csv" "gpu,model,config,identical,kernels_per_token"
    local gpu g
    IFS=',' read -ra gpu_list <<< "$DN_GPUS"
    for gpu in "${gpu_list[@]}"; do
        g=$(safe "$gpu")
        if have_file "$MODEL_Q8" && have_file "$REF_Q8"; then
            greedy_run "$gpu" "$MODEL_Q8" q8_0 nodn "$REF_Q8"
            greedy_run "$gpu" "$MODEL_Q8" q8_0 dn "$REF_Q8"
        else
            log "dn_greedy: MODEL_Q8 ('$MODEL_Q8') or REF_Q8 ('$REF_Q8') missing, Q8_0 skipped"
        fi
        # the CPU quantizes the activations to 8 bits, so the K-quant / Q4_0 text is compared with this GPU's output with DN off (the Q4_K_M KL stage compares with the CPU)
        local entry label model
        for entry in "q4_k_m:$MODEL_Q4K" "q4_0:$MODEL_Q40"; do
            label=${entry%%:*}; model=${entry#*:}
            have_file "$model" || { log "dn_greedy: no model for $label, skipped"; continue; }
            greedy_run "$gpu" "$model" "$label" nodn /dev/null
            cp "$OUT/logs/greedy_${g}_${label}_nodn.out" "$OUT/logs/ref_dn_${label}_${g}.out"
            greedy_run "$gpu" "$model" "$label" dn "$OUT/logs/ref_dn_${label}_${g}.out"
        done
    done
}

# the text of the KL divergence: PPL_TEXT, or the documentation of the checkout (4096 tokens are used by -c 512 --chunks 8)
kl_text() {
    if [ -n "${PPL_TEXT:-}" ]; then echo "$PPL_TEXT"; return; fi
    mkdir -p "$OUT/kl"
    if [ "$DRY_RUN" != 1 ] && [ ! -s "$OUT/kl/text.txt" ]; then cat "$ROOT/README.md" "$ROOT"/docs/*.md > "$OUT/kl/text.txt"; fi
    echo "$OUT/kl/text.txt"
}

stage_dn_kl() {
    csv_header "$OUT/kl.csv" "gpu,model,config,ppl,mean_kld,kld99,kld999,max_dp,same_top_p,verdict"
    if ! have_file "$MODEL_Q4K"; then log "dn_kl: no Q4_K_M model, skipped"; return 0; fi
    if [ "$DRY_RUN" != 1 ] && [ ! -x "$LLAMA_BIN/llama-perplexity" ]; then log "dn_kl: $LLAMA_BIN/llama-perplexity is missing, skipped"; return 0; fi
    local text base gpu g cfg lf
    text=$(kl_text)
    mkdir -p "$OUT/kl"
    base=${KL_BASE:-$OUT/kl/base.kld}
    if [ "$DRY_RUN" = 1 ] || [ ! -s "$base" ]; then
        # the official llama.cpp b11384 on the CPU (no plugin), once
        # shellcheck disable=SC2086
        run_cpu env -u GGML_BACKEND_PATH "$LLAMA_BIN/llama-perplexity" -m "$MODEL_Q4K" -f "$text" -c 512 --chunks 8 $KL_ARGS -ngl 0 -dev none \
            --kl-divergence-base "$base" > "$OUT/logs/kl_base_cpu.log" 2>&1
    fi
    IFS=',' read -ra gpu_list <<< "$DN_GPUS"
    for gpu in "${gpu_list[@]}"; do
        g=$(safe "$gpu")
        for cfg in dn nodn; do
            lf="$OUT/logs/kl_${g}_${cfg}.log"
            # shellcheck disable=SC2086
            run_gpu "$gpu" "$(cfg_env $cfg "$gpu")" "$LLAMA_BIN/llama-perplexity" $(devargs_for "$gpu") -m "$MODEL_Q4K" -f "$text" -c 512 --chunks 8 $KL_ARGS -ngl 99 \
                --kl-divergence-base "$base" --kl-divergence > "$lf" 2>&1
            echo "$gpu,q4_k_m,$cfg,$("$HERE/summarize.py" kl "$lf")" >> "$OUT/kl.csv"
            log "dn_kl $gpu $cfg: $(tail -1 "$OUT/kl.csv")"
        done
    done
}

stage_dn_bench() {
    csv_header "$OUT/bench.csv" "gpu,model,config,pp128,tg64"
    local gpu entry label model cfg
    IFS=',' read -ra gpu_list <<< "$DN_GPUS"
    for gpu in "${gpu_list[@]}"; do
        for entry in "q8_0:$MODEL_Q8" "q4_k_m:$MODEL_Q4K" "q4_0:$MODEL_Q40"; do
            label=${entry%%:*}; model=${entry#*:}
            have_file "$model" || { log "dn_bench: no model for $label, skipped"; continue; }
            if have_file "$BASE_PLUGIN"; then bench_run "$gpu" "$model" "$label" main "$BASE_PLUGIN"; fi
            for cfg in nodn dn; do bench_run "$gpu" "$model" "$label" "$cfg" "$PLUGIN"; done
        done
    done
}

#
# regress: every GPU with the default configuration, against the reference table (baseline.csv)
#
stage_regress() {
    csv_header "$OUT/regress.csv" "gpu,test_passed,test_total,platform,greedy_q8,greedy_q4k,q8_pp,q8_tg,q4k_pp,q4k_tg"
    local gpu g lf pt plat want
    IFS=',' read -ra gpu_list <<< "$REGRESS_GPUS"
    for gpu in "${gpu_list[@]}"; do
        g=$(safe "$gpu")
        lf="$OUT/logs/regress_${g}_test.log"
        run_gpu "$gpu" "" "$LLAMA_BIN/test-backend-ops" test -b CL11_0 > "$lf" 2>&1
        pt=$("$HERE/summarize.py" test "$lf")
        # the platform that really ran (the device version is in the log of the plugin)
        case "$gpu" in
            *" rusticl") want="OpenCL 3.0";;
            *" clover")  want="OpenCL 1.1";;
            *)           want="";;
        esac
        if [ -z "$want" ]; then plat="-"; elif [ "$DRY_RUN" = 1 ]; then plat="(dry run)"; elif grep -q "$want" "$lf"; then plat="ok ($want)"; else plat="WRONG (no '$want' in the log)"; fi
        local gq8="-" gq4="-" qpp qtg
        if have_file "$MODEL_Q8" && have_file "$REF_Q8"; then greedy_run "$gpu" "$MODEL_Q8" q8_0 default "$REF_Q8"; gq8=$GREEDY_LAST; fi
        if have_file "$MODEL_Q4K"; then
            if have_file "$BASE_PLUGIN"; then
                PLUGIN_USED="$BASE_PLUGIN" greedy_run "$gpu" "$MODEL_Q4K" q4_k_m main /dev/null
                cp "$OUT/logs/greedy_${g}_q4_k_m_main.out" "$OUT/logs/ref_regress_q4k_${g}.out"
                greedy_run "$gpu" "$MODEL_Q4K" q4_k_m default "$OUT/logs/ref_regress_q4k_${g}.out"; gq4=$GREEDY_LAST
            else
                greedy_run "$gpu" "$MODEL_Q4K" q4_k_m default /dev/null; gq4="no main plugin"
            fi
        fi
        local b8pp="-" b8tg="-" bkpp="-" bktg="-"
        if have_file "$MODEL_Q8"; then bench_run "$gpu" "$MODEL_Q8" q8_0 default "$PLUGIN"; b8pp=$BENCH_PP; b8tg=$BENCH_TG; fi
        if have_file "$MODEL_Q4K"; then bench_run "$gpu" "$MODEL_Q4K" q4_k_m default "$PLUGIN"; bkpp=$BENCH_PP; bktg=$BENCH_TG; fi
        echo "$gpu,${pt% *},${pt#* },$plat,$gq8,$gq4,$b8pp,$b8tg,$bkpp,$bktg" >> "$OUT/regress.csv"
        log "regress $gpu: test ${pt% *}/${pt#* }, platform $plat, greedy $gq8 / $gq4, Q8_0 $b8pp/$b8tg, Q4_K_M $bkpp/$bktg"
    done
}

stage_dn() { stage_dn_test; stage_dn_mv; stage_dn_greedy; stage_dn_kl; stage_dn_bench; }

stage_summary() {
    "$HERE/summarize.py" summary "$OUT"
}

log "output: $OUT  (GPUs: $GPUS, stages: $STAGES)"
for s in build test greedy mm bench dn dn_test dn_mv dn_greedy dn_kl dn_bench regress summary; do
    if has_stage "$s"; then "stage_$s"; fi
done
log "done: $OUT"

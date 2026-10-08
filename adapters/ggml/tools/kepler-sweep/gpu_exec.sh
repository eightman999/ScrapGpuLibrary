#!/bin/bash
# kepler-sweep helper: wait until the GPU is cool, then run a command on that GPU only.
#   gpu_exec.sh "<GPU>" command args...
# <GPU> is a substring of the OpenCL device name ("GT 730"), optionally followed by the Mesa platform for the Radeons:
#   "WX 2100 rusticl"   RUSTICL_ENABLE=radeonsi (Mesa rusticl, OpenCL 3.0; with it Clover is also visible, rusticl is CL11_0: pass `-dev CL11_0` to llama.cpp)
#   "WX 2100 clover"    RUSTICL_ENABLE unset (Mesa Clover, OpenCL 1.1; the only platform of the card)
#   a Radeon without a platform keeps the environment as it is. The RX 6400 hangs on Clover (docs/HARDWARE.md): "RX 6400 clover" is refused.
# sweep.sh calls it under `flock $LOCK`, so the temperature is read while nobody else uses the GPUs.
#   THERMAL_MAX       wait while the GPU is at this temperature (Celsius) or above (default 85), until it is below THERMAL_RESUME (default 78)
#   THERMAL_WAIT_MAX  give up waiting after this many seconds (default 1800)
# Temperature: nvidia-smi for NVIDIA GPUs; for a Radeon the amdgpu hwmon (/sys/class/drm/card*/device/hwmon/*/temp1_input; the card is found by its
# PCI device id: WX 2100 = 6995, RX 6400 = 743f, otherwise the hottest amdgpu card counts).
GPU="$1"; shift
PLAT=""
case "$GPU" in
    *" rusticl") PLAT=rusticl; GPU="${GPU% rusticl}";;
    *" clover")  PLAT=clover;  GPU="${GPU% clover}";;
esac
if [ "$PLAT" = clover ] && [[ "$GPU" == *"RX 6400"* ]]; then
    echo "gpu_exec: Mesa Clover hangs the RX 6400 (docs/HARDWARE.md), refusing" >&2
    exit 3
fi
MAXT=${THERMAL_MAX:-85}

amd_temp() { # hottest matching amdgpu card in Celsius, empty if none
    local id="" best="" h c t dev
    case "$GPU" in *"WX 2100"*) id=6995;; *"RX 6400"*) id=743f;; esac
    for h in /sys/class/drm/card*/device/hwmon/hwmon*; do
        [ -r "$h/temp1_input" ] || continue
        [ "$(cat "$h/name" 2>/dev/null)" = amdgpu ] || continue
        c=${h%/hwmon/*}
        dev=$(cat "$c/device" 2>/dev/null)
        [ -z "$id" ] || [ "${dev#0x}" = "$id" ] || continue
        t=$(( $(cat "$h/temp1_input") / 1000 ))
        [ -z "$best" ] || [ "$t" -gt "$best" ] && best=$t
    done
    echo "$best"
}

gpu_temp() {
    local t=""
    case "$GPU" in
        *WX*|*RX*|*Radeon*|*AMD*) t=$(amd_temp);;
        *) if command -v nvidia-smi >/dev/null 2>&1; then
               # "name, temperature": the line of the GPU whose name contains the substring
               t=$(nvidia-smi --query-gpu=name,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | grep -i -- "$GPU" | head -1 | awk -F', ' '{print $NF}')
           fi;;
    esac
    echo "$t"
}

wait_cool() {
    # at THERMAL_MAX or above: wait until the GPU is below THERMAL_RESUME (hysteresis, default 78)
    local waited=0 t limit=$MAXT
    while :; do
        t=$(gpu_temp)
        [[ "$t" =~ ^[0-9]+$ ]] || return 0
        (( t < limit )) && return 0
        limit=${THERMAL_RESUME:-78}
        if (( waited >= ${THERMAL_WAIT_MAX:-1800} )); then
            echo "gpu_exec: $GPU is still at $t C after ${waited} s, going on" >&2
            return 0
        fi
        echo "gpu_exec: $GPU at $t C (>= $MAXT C), waiting for < $limit C" >&2
        sleep 10
        waited=$((waited + 10))
    done
}
wait_cool
case "$PLAT" in
    rusticl) exec env GGML_CL11_DEVICES="$GPU" RUSTICL_ENABLE=radeonsi "$@";;
    clover)  exec env -u RUSTICL_ENABLE GGML_CL11_DEVICES="$GPU" "$@";;
    *)       exec env GGML_CL11_DEVICES="$GPU" "$@";;
esac

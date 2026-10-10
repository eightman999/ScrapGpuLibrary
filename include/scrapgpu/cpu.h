#ifndef SCRAPGPU_CPU_H
#define SCRAPGPU_CPU_H

#include "scrapgpu/api.h"
#include "scrapgpu/status.h"

#ifdef __cplusplus
extern "C" {
#endif

// Registers the CPU backend (src/backends/cpu): one device named "CPU" whose buffers live in
// system RAM and whose ops run synchronously through scrapgpu_ref_execute (the same public
// device / buffer / stream / plan API as the GPU backends). The device is portable C++17;
// nothing beyond the compiler baseline (SSE2 on x86-64) is required, AVX in particular is not.
// Only the natural layout is supported. Safe to call more than once.
SCRAPGPU_API scrapgpu_status scrapgpu_cpu_register(void);

#ifdef __cplusplus
}
#endif

#endif

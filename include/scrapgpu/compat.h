#ifndef SCRAPGPU_COMPAT_H
#define SCRAPGPU_COMPAT_H

#include "scrapgpu/op.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum scrapgpu_execution_path {
    SCRAPGPU_EXECUTION_NONE = 0,
    SCRAPGPU_EXECUTION_NATIVE = 1,
    SCRAPGPU_EXECUTION_CPU_FALLBACK = 2,
} scrapgpu_execution_path;

typedef struct scrapgpu_compat_options {
    size_t struct_size;
    int allow_cpu_fallback;
    /* Total host tensor staging allocation. Zero means no staging allowed.
     * Does not include the CPU kernel's own scratch (e.g. matmul panels). */
    uint64_t max_staging_bytes;
} scrapgpu_compat_options;

typedef struct scrapgpu_execution_report {
    size_t struct_size;
    scrapgpu_execution_path path;
    uint64_t staging_bytes;
    uint64_t downloaded_bytes;
    uint64_t uploaded_bytes;
} scrapgpu_execution_report;

/* Native support is queried before dispatch. Only an UNSUPPORTED result from
 * that query permits fallback; native execution errors are never retried.
 * Fallback is synchronous, explicit opt-in, and bounded. It stages contiguous
 * tensors in their natural layout (R32 inputs are unpacked by the backend).
 * Outputs are published only after reference computation succeeds. Multiple
 * output publication is not transactional on a device transfer failure.
 * Unknown ops/dtypes are rejected. This is not a CUDA API emulator.
 * Pass NULL options for native-only execution, NULL report to omit reporting. */
SCRAPGPU_API scrapgpu_status scrapgpu_op_execute_compatible(
    scrapgpu_device_t * device, scrapgpu_stream_t * stream,
    const scrapgpu_op_desc * desc, const scrapgpu_compat_options * options,
    scrapgpu_execution_report * report);

#ifdef __cplusplus
}
#endif
#endif

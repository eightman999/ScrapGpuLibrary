#ifndef SCRAPGPU_OPENCL11_DEVICE_STATE_H
#define SCRAPGPU_OPENCL11_DEVICE_STATE_H

#include "buffer.h"
#include "context.h"
#include "executor.h"
#include "kernels.h"

#include <mutex>
#include <string>
#include <vector>

// The runtime state of one OpenCL 1.1 device: the OpenCL context and queue, the programs and the kernel handles, the tuning and the
// caches / scratch buffers of the executor. The host-side wrapper of an API (the ggml adapter until Phase 3) keeps only its own objects
// and points to (or owns) an instance of this struct.
struct cl11_device {
    cl11_device_info info;
    int              index = 0;
    std::string      name; // "CL11_<index>"

    // protects the lazily created state and the kernel objects (clSetKernelArg is not thread safe)
    std::mutex mutex;

    // created on first use
    bool             context_ready = false;
    bool             program_ready = false;
    cl_program       program_k = nullptr; // the Q4_K / Q6_K mat-vec kernels when they are apart (cl11_ensure_kprogram)
    bool             program_k_ready = false;
    cl_program       program_mm = nullptr; // the mat-mat kernels when they are apart (split_programs)
    bool             split_programs = false; // the program is split into the decode kernels, the mat-mat kernels and the K-quant kernels
    bool             k_separate = false;
    bool             failed        = false;
    cl_context       context = nullptr;
    cl_command_queue queue   = nullptr;
    cl_program       program = nullptr;
    cl_kernel        kernels[K_COUNT] = {};
    cl11_config      cfg;

    // mat-vec is used for src1 with at most this many columns (GGML_CL11_MV_MAX_COLS)
    int mv_max_cols = 1;
    // MUL_MATs with the same src1 in one launch (cl11_tuning::multi_mv, GGML_CL11_MULTI_MV=0/1); mvm_wgs: the work-group limit of k_mvm_<type>
    bool   multi_mv = false;
    size_t mvm_wgs[CL11_RK_COUNT] = {};
    bool   fuse_set_rows = false; // cl11_tuning::fuse_set_rows, GGML_CL11_FUSE_SET_ROWS=0/1
    // tuning / debugging (environment variables)
    int  mv_tpr     = 0;      // GGML_CL11_MV_TPR: threads per row of the generic mat-vec kernels (0 = automatic)
    bool mvs_enabled = true;  // GGML_CL11_MV_STAGED=0: use the generic mat-vec kernels only
    int  mvs_min_k   = 256;   // minimum row length for the x mat-vec kernels
    bool use_nv_asm  = false; // NVIDIA: inline PTX in the kernels (GGML_CL11_NO_ASM=1 disables it)
    bool use_nv_ldg  = false; // NVIDIA sm_35+: ld.global.nc (read-only data cache) for src1 in the mat-vec kernels (GGML_CL11_NO_LDG=1 disables it)
    bool no_cpy_t    = false; // GGML_CL11_NO_CPY_T=1: no transposing copy kernel
    bool no_mvt      = false; // GGML_CL11_NO_MVT=1: do not fuse CONT + MUL_MAT
    bool no_mvq      = false; // GGML_CL11_NO_MVQ=1: no special kernel for the K cache
    bool no_fuse     = false; // GGML_CL11_NO_FUSE=1: no fusion of consecutive nodes (RMS_NORM + MUL)
    int  mvs_family  = 0;     // 0: automatic, 1: k_mvx (src1 in local memory), 2: k_mvw (src1 through the caches)
    bool mvr_const_x = false; // GGML_CL11_MVR_CONST_X=0/1
    size_t max_const = 0;     // CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE
    bool strict_alloc_dev = false; // cl11_tuning::strict_alloc: like GGML_CL11_STRICT_ALLOC=1, and supports_op rejects tensors above the limit

    // sub-buffers of src1 for the constant memory (key: parent, offset, size); released with the parent buffer
    std::vector<cl11_const_sub_entry> const_subs;

    // src1 prepared by k_xprep for the "denormal products" mat-vec kernels: one scratch buffer per pattern (Q4_0, Q4_K, Q6_K; scaled copy of the
    // column + sums of groups of 16) and the column each one holds now
    struct xprep_slot {
        cl_mem buf   = nullptr;
        bool   valid = false;
        cl_mem src   = nullptr;   // the src1 column it was made from: buffer, byte offset, bytes
        size_t off   = 0;
        size_t bytes = 0;
    };
    xprep_slot xprep[3];

    // MUL_MAT_ID, prompt processing (cl11_op_mul_mat_id_sync): scratch buffers (columns of src1 of the (slot, token) pairs sorted by expert, the
    // result columns, the pairs), grown when needed
    struct mid_buffer {
        cl_mem buf  = nullptr;
        size_t size = 0;
    };
    mid_buffer mid_gather, mid_out, mid_pairs;
    // the ids read back by the last MUL_MAT_ID of the prompt method: the three MUL_MAT_IDs of a layer (up, gate, down) read the same ids, the queue
    // is finished once. Invalid at the start of every graph and when a node writes the buffer range (cl11_xprep_note_write).
    struct mid_ids_cache {
        bool                 valid = false;
        cl_mem               mem   = nullptr;
        size_t               off   = 0;
        std::vector<uint8_t> raw;
    };
    mid_ids_cache mid_ids;
    int mid_max_tokens = -1; // GGML_CL11_MID_MAX_TOKENS: up to this many tokens MUL_MAT_ID reads the expert numbers on the GPU (no synchronization); -1: while n_used * n_tokens <= n_expert

    cl11_executor_state exec;
};

// creates cl_context and the command queue and reads the runtime tuning / debugging environment variables, the caller holds dev->mutex
bool cl11_ensure_context(cl11_device * dev);

// builds the program(s) and creates the kernels (the work-group sizes start from what the device allows and are reduced, with a rebuild,
// until every kernel can be launched with them), the caller holds dev->mutex
bool cl11_ensure_program(cl11_device * dev);

// builds the program of the Q4_K / Q6_K mat-vec kernels (only when cl11_ensure_program put them apart), the caller holds dev->mutex
bool cl11_ensure_kprogram(cl11_device * dev);

// the 64 KiB scratch buffer of k_xprep for the pattern `slot` (one at most 64 KiB column of src1 scaled + the sums of its groups of 16),
// created on first use, the caller holds dev->mutex
cl_mem cl11_xprep_scratch(cl11_device * dev, int slot);

// grows a MUL_MAT_ID scratch buffer to `bytes` when needed, the caller holds dev->mutex
bool cl11_mid_ensure(cl11_device * dev, cl11_device::mid_buffer & b, size_t bytes);

#endif

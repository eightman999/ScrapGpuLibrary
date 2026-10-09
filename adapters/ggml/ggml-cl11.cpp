// CL11: ggml backend that only needs OpenCL 1.1 (host API and OpenCL C 1.1).
//
// Target: old GPUs such as the NVIDIA GeForce GT 430 (Fermi, "OpenCL 1.1 CUDA"):
//   - no sub-groups, no cl_khr_fp16, no doubles in the kernels,
//   - clCreateBuffer is limited to CL_DEVICE_MAX_MEM_ALLOC_SIZE (1/4 of the memory),
//   - PCIe transfers are slow: the whole graph is enqueued on one in-order queue and the host only
//     synchronizes in synchronize() and get_tensor().
//
// Only the OpenCL 1.1 host API is used (CL_TARGET_OPENCL_VERSION 110): no clEnqueueFillBuffer,
// no clEnqueueMarkerWithWaitList, no clCreateProgramWithBuiltInKernels, no clCreateImage, no SVM.

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 110
#endif

#ifdef __APPLE__
#  ifndef CL_SILENCE_DEPRECATION
#    define CL_SILENCE_DEPRECATION
#  endif
#  include <OpenCL/opencl.h>
#else
#  include <CL/cl.h>
#endif

#include "ggml-cl11.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "opencl11/buffer.h"
#include "opencl11/context.h"
#include "opencl11/device.h"
#include "opencl11/device_state.h"
#include "opencl11/executor.h"
#include "opencl11/kernels.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <mutex>
#include <string>
#include <vector>

// ggml-alloc asks for the base address of a buffer: the tensors of a CL11 buffer are addressed as
// offsets from this dummy (non-null, page aligned) pointer.
#define CL11_BASE ((void *) 0x1000)

// Device information, context creation, the programs / kernels and the executor state live in src/backends/opencl11/
// (cl11_device). ggml keeps the API objects: the buffer wrapper, the op translation / dispatch and the registry.

// tensor descriptor passed to the kernels by value, must match `tdesc` in src/backends/opencl11/kernels/common.cl
struct cl11_td {
    int32_t  ne[4];
    int32_t  nb[4];
    int32_t  off;
    uint32_t fm[4]; // fast division by ne[i]
    uint32_t fl[4];
};
static_assert(sizeof(cl11_td) == 17 * sizeof(int32_t), "cl11_td layout");

// magic numbers m, l such that n / d == (mulhi(n, m) + n) >> l for 0 <= n < 2^31
static void cl11_fastdiv_init(uint32_t d, uint32_t & m, uint32_t & l) {
    m = 0;
    l = 0;
    if (d == 0) {
        return;
    }
    while (l < 32 && ((uint64_t) 1 << l) < d) {
        ++l;
    }
    m = (uint32_t) ((((uint64_t) 1 << 32) * (((uint64_t) 1 << l) - d)) / d + 1);
}

// must match `bcast` in src/backends/opencl11/kernels/common.cl
struct cl11_bcast {
    int32_t  r2, r3;
    uint32_t m2, l2, m3, l3;
    int32_t  tiles;
    uint32_t mt, lt;
    int32_t  tiles2;
    uint32_t mt2, lt2;
};
static_assert(sizeof(cl11_bcast) == 12 * sizeof(int32_t), "cl11_bcast layout");

static cl11_bcast cl11_make_bcast(int64_t r2, int64_t r3, int64_t tiles, int64_t tiles2) {
    cl11_bcast b = {};
    b.r2 = (int32_t) r2;
    b.r3 = (int32_t) r3;
    cl11_fastdiv_init((uint32_t) r2, b.m2, b.l2);
    cl11_fastdiv_init((uint32_t) r3, b.m3, b.l3);
    b.tiles = (int32_t) tiles;
    cl11_fastdiv_init((uint32_t) tiles, b.mt, b.lt);
    b.tiles2 = (int32_t) tiles2;
    cl11_fastdiv_init((uint32_t) tiles2, b.mt2, b.lt2);
    return b;
}

// The ggml device wrapper: only the ggml objects (buffer type, device, registry) and the runtime state of the device, which lives in
// src/backends/opencl11 (cl11_device: OpenCL context / queue, programs / kernels, tuning, caches, scratch buffers, executor).
struct cl11_device_context {
    std::unique_ptr<cl11_device> device;
    ggml_backend_buffer_type buft = {};
    ggml_backend_device      dev  = {};
};

static std::vector<std::unique_ptr<cl11_device_context>> g_cl11_devices;

static bool cl11_debug() {
    static const bool debug = getenv("GGML_CL11_DEBUG") != nullptr;
    return debug;
}

#define CL11_CHECK(expr)                                                                   \
    do {                                                                                   \
        const cl_int _err = (expr);                                                        \
        if (_err != CL_SUCCESS) {                                                          \
            GGML_LOG_ERROR("%s: %s failed: %s (%d)\n", __func__, #expr, cl11_err_str(_err), (int) _err); \
            return false;                                                                  \
        }                                                                                  \
    } while (0)

//
// device enumeration compatibility shim
//

static void cl11_probe_devices() {
    (void) cl11_probe_device_infos([](const cl11_device_info & info, bool strict_alloc) {
        auto ctx = std::make_unique<cl11_device_context>();
        ctx->device = std::make_unique<cl11_device>();
        ctx->device->info = info;
        ctx->device->strict_alloc_dev = strict_alloc;
        ctx->device->index = (int) g_cl11_devices.size();
        ctx->device->name = "CL11_" + std::to_string(ctx->device->index);
        g_cl11_devices.push_back(std::move(ctx));
    });
}

//
// buffers
//

struct cl11_buffer_context {
    cl11_device * dev = nullptr;
    cl_mem        mem = nullptr;
    size_t        size = 0; // allocated size
};

static ggml_backend_buffer_t cl11_tensor_buffer(const ggml_tensor * t) {
    return t->view_src ? t->view_src->buffer : t->buffer;
}

// byte offset of the tensor data inside of its cl_mem
static size_t cl11_tensor_offset(const ggml_tensor * t) {
    return (size_t) ((const char *) t->data - (const char *) CL11_BASE);
}

struct cl11_tref {
    cl_mem  mem;
    cl11_td td;
};

static cl11_tref cl11_ref(const ggml_tensor * t) {
    cl11_tref r;
    auto * bctx = (cl11_buffer_context *) cl11_tensor_buffer(t)->context;
    r.mem = bctx->mem;
    for (int i = 0; i < 4; ++i) {
        r.td.ne[i] = (int32_t) t->ne[i];
        r.td.nb[i] = (int32_t) t->nb[i];
    }
    r.td.off = (int32_t) cl11_tensor_offset(t);
    for (int i = 0; i < 4; ++i) {
        cl11_fastdiv_init((uint32_t) t->ne[i], r.td.fm[i], r.td.fl[i]);
    }
    return r;
}

//
// Repacked layout "R32" of the Q4_0 / Q8_0 / Q4_K / Q6_K tensors
//
// The blocks of Q8_0 (34 bytes), Q4_0 (18 bytes) and Q6_K (210 bytes) are only 2-byte aligned, and the blocks of all of them have
// a lot of fields: the natural layout needs 16-bit loads and a lot of address arithmetic on an old GPU. set_tensor therefore
// stores these tensors (when cl11_repacked() is true, that is a property of the tensor alone, so supports_op / the kernels /
// get_tensor agree) in a layout made for coalesced 128-bit loads, get_tensor converts back. The size is the same as
// ggml_nbytes(). For every slice of the tensor (ne2 * ne3 slices of M = ne1 rows, nb = ne0 / block size blocks per row), the rows
// are grouped by 32 (rs = number of rows of the group, 32 except for the last group). The bytes of a block are split into up to
// 3 areas, one after the other in the slice (area k starts at M * nb * (bytes of the areas before it per block)), see
// `cl11_rp_areas`. Inside of an area of 16 * U bytes per block the group g starts at g * 32 * nb * 16 * U and the 16-byte unit j
// (0 <= j < U) of the block b of the row `lane` of the group is the unit (b * U + j) * rs + lane. An area of 2 bytes (the half
// scale) is an array of halves at b * rs + lane:
//   Q8_0: quants (32 bytes = 2 units, xor 0x80), d              Q4_0: nibbles (16 bytes = 1 unit), d
//   Q4_K: quants (the 128 bytes of nibbles, 8 units), header (d, dmin, 12 bytes of scales: 1 unit)
//   Q6_K: ql + qh (192 bytes, 12 units), scales (16 bytes, xor 0x80, 1 unit), d
// (see src/backends/opencl11/kernels/mul_mv_r.cl, src/backends/opencl11/kernels/mul_mv_k.cl, dqr_* in common.cl)
//

static bool cl11_rp_type(ggml_type type) {
    return type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q6_K;
}

static int cl11_rkind(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0: return CL11_RK_Q4_0;
        case GGML_TYPE_Q8_0: return CL11_RK_Q8_0;
        case GGML_TYPE_Q4_K: return CL11_RK_Q4_K;
        default:             return CL11_RK_Q6_K;
    }
}

static bool cl11_repacked(const ggml_tensor * t) {
    if (t == nullptr || !cl11_rp_type(t->type) || t->view_src != nullptr || !ggml_is_contiguous(t)) {
        return false;
    }
    // the slices must keep the 16-byte alignment of the 128-bit loads
    return t->ne[2] * t->ne[3] == 1 || (t->nb[2] % 16 == 0 && t->nb[3] % 16 == 0);
}

// a view of a repacked tensor cannot be addressed (the kernels and the transfers know the layout of the whole tensor only)
static void cl11_rp_check_view(const ggml_tensor * t) {
    if (t->view_src != nullptr && cl11_repacked(t->view_src)) {
        GGML_ABORT("CL11: the tensor %s is a view of a repacked quantized tensor (%s), which is not supported\n", t->name, t->view_src->name);
    }
}

struct cl11_rp_area {
    int     src;  // first byte of the area in the natural block
    int     len;  // bytes per block (2: halves, otherwise a multiple of 16)
    uint8_t xr;   // xor mask applied to the bytes
};

struct cl11_rp_geom {
    int          nb;      // blocks per row
    int          bs;      // bytes of a block in the natural layout
    int          na;      // number of areas
    cl11_rp_area area[3];
    int64_t      M;       // rows per slice
    size_t       rowb;    // bytes of a row in the natural layout
};

static cl11_rp_geom cl11_rp_geometry(const ggml_tensor * t) {
    cl11_rp_geom g = {};
    g.M    = t->ne[1];
    g.bs   = (int) ggml_type_size(t->type);
    g.nb   = (int) (t->ne[0] / ggml_blck_size(t->type));
    g.rowb = (size_t) g.nb * g.bs;
    switch (t->type) {
        case GGML_TYPE_Q8_0: g.na = 2; g.area[0] = {  2, 32, 0x80 }; g.area[1] = {   0,   2, 0 }; break;
        case GGML_TYPE_Q4_0: g.na = 2; g.area[0] = {  2, 16, 0    }; g.area[1] = {   0,   2, 0 }; break;
        case GGML_TYPE_Q4_K: g.na = 2; g.area[0] = { 16, 128, 0   }; g.area[1] = {   0,  16, 0 }; break;
        default:             g.na = 3; g.area[0] = {  0, 192, 0   }; g.area[1] = { 192, 16, 0x80 }; g.area[2] = { 208, 2, 0 }; break;
    }
    return g;
}

// bytes per block of the areas before the area k
static size_t cl11_rp_area_before(const cl11_rp_geom & G, int k) {
    size_t n = 0;
    for (int i = 0; i < k; ++i) {
        n += (size_t) G.area[i].len;
    }
    return n;
}

// natural -> repacked: `nrows` consecutive rows (a multiple of 32, except at the end of a slice) -> the buffers of the areas
static void cl11_rp_pack(const cl11_rp_geom & G, const uint8_t * nat, int64_t nrows, uint8_t * const * out) {
    for (int64_t g0 = 0; g0 < nrows; g0 += 32) {
        const int rs = (int) std::min<int64_t>(32, nrows - g0);
        for (int lane = 0; lane < rs; ++lane) {
            const uint8_t * row = nat + (size_t) (g0 + lane) * G.rowb;
            for (int b = 0; b < G.nb; ++b) {
                const uint8_t * p = row + (size_t) b * G.bs;
                for (int k = 0; k < G.na; ++k) {
                    const cl11_rp_area & A = G.area[k];
                    uint8_t * og = out[k] + (size_t) (g0 / 32) * 32 * G.nb * A.len;
                    if (A.len == 2) {
                        memcpy(og + ((size_t) b * rs + lane) * 2, p + A.src, 2);
                    } else {
                        const int U = A.len / 16;
                        for (int j = 0; j < U; ++j) {
                            uint8_t * o = og + (((size_t) b * U + j) * rs + lane) * 16;
                            const uint8_t * s = p + A.src + 16 * j;
                            for (int i = 0; i < 16; ++i) {
                                o[i] = s[i] ^ A.xr;
                            }
                        }
                    }
                }
            }
        }
    }
}

// repacked -> natural
static void cl11_rp_unpack(const cl11_rp_geom & G, const uint8_t * const * in, int64_t nrows, uint8_t * nat) {
    for (int64_t g0 = 0; g0 < nrows; g0 += 32) {
        const int rs = (int) std::min<int64_t>(32, nrows - g0);
        for (int lane = 0; lane < rs; ++lane) {
            uint8_t * row = nat + (size_t) (g0 + lane) * G.rowb;
            for (int b = 0; b < G.nb; ++b) {
                uint8_t * p = row + (size_t) b * G.bs;
                for (int k = 0; k < G.na; ++k) {
                    const cl11_rp_area & A = G.area[k];
                    const uint8_t * ig = in[k] + (size_t) (g0 / 32) * 32 * G.nb * A.len;
                    if (A.len == 2) {
                        memcpy(p + A.src, ig + ((size_t) b * rs + lane) * 2, 2);
                    } else {
                        const int U = A.len / 16;
                        for (int j = 0; j < U; ++j) {
                            const uint8_t * s = ig + (((size_t) b * U + j) * rs + lane) * 16;
                            uint8_t * o = p + A.src + 16 * j;
                            for (int i = 0; i < 16; ++i) {
                                o[i] = s[i] ^ A.xr;
                            }
                        }
                    }
                }
            }
        }
    }
}

// copy `size` bytes at byte `offset` of the (natural layout of the) tensor from / to the device. The conversion is done
// per chunk of whole groups of rows of a slice; a chunk that is only partly written is read, patched and written back.
static bool cl11_rp_transfer(cl11_device * c, cl_mem mem, const ggml_tensor * t, uint8_t * host, size_t offset, size_t size, bool to_dev) {
    const cl11_rp_geom G = cl11_rp_geometry(t);
    const size_t slice_nat = (size_t) G.M * G.rowb;
    const size_t end = offset + size;
    const size_t base = cl11_tensor_offset(t);
    const int64_t chunk_groups = std::max<int64_t>(1, (int64_t) ((size_t) 24 * 1024 * 1024 / (32 * G.rowb)));

    std::vector<uint8_t> abuf[3], nbuf;
    for (size_t s = offset / slice_nat; s <= (end - 1) / slice_nat; ++s) {
        const size_t slice_lo = s * slice_nat;
        const size_t lo = std::max(offset, slice_lo) - slice_lo;
        const size_t hi = std::min(end, slice_lo + slice_nat) - slice_lo;
        const int64_t g_first = (int64_t) (lo / G.rowb) / 32;
        const int64_t g_last  = (int64_t) ((hi - 1) / G.rowb) / 32;
        const size_t q_slice = base + slice_lo;

        for (int64_t gc = g_first; gc <= g_last; gc += chunk_groups) {
            const int64_t rA = gc * 32;
            const int64_t rB = std::min<int64_t>(G.M, std::min(g_last + 1, gc + chunk_groups) * 32);
            const int64_t nrows = rB - rA;
            const size_t nat_lo = (size_t) rA * G.rowb, nat_hi = (size_t) rB * G.rowb;
            const size_t ov_lo = std::max(lo, nat_lo), ov_hi = std::min(hi, nat_hi);
            const bool full = ov_lo == nat_lo && ov_hi == nat_hi;
            uint8_t * hp = full ? host + (slice_lo + nat_lo - offset) : nullptr; // the chunk lies inside of the host range

            uint8_t * ap[3] = { nullptr, nullptr, nullptr };
            size_t   an[3] = { 0, 0, 0 }, ao[3] = { 0, 0, 0 };
            for (int k = 0; k < G.na; ++k) {
                an[k] = (size_t) nrows * G.nb * G.area[k].len;
                ao[k] = q_slice + (size_t) G.M * G.nb * cl11_rp_area_before(G, k) + (size_t) rA * G.nb * G.area[k].len;
                abuf[k].resize(an[k]);
                ap[k] = abuf[k].data();
            }
            const uint8_t * const * cap = (const uint8_t * const *) ap;
            if (to_dev) {
                if (full) {
                    cl11_rp_pack(G, hp, nrows, ap);
                } else {
                    nbuf.resize((size_t) nrows * G.rowb);
                    for (int k = 0; k < G.na; ++k) {
                        CL11_CHECK(clEnqueueReadBuffer(c->queue, mem, CL_TRUE, ao[k], an[k], ap[k], 0, nullptr, nullptr));
                    }
                    cl11_rp_unpack(G, cap, nrows, nbuf.data());
                    memcpy(nbuf.data() + (ov_lo - nat_lo), host + (slice_lo + ov_lo - offset), ov_hi - ov_lo);
                    cl11_rp_pack(G, nbuf.data(), nrows, ap);
                }
                for (int k = 0; k < G.na; ++k) {
                    CL11_CHECK(clEnqueueWriteBuffer(c->queue, mem, CL_TRUE, ao[k], an[k], ap[k], 0, nullptr, nullptr));
                }
            } else {
                for (int k = 0; k < G.na; ++k) {
                    CL11_CHECK(clEnqueueReadBuffer(c->queue, mem, CL_TRUE, ao[k], an[k], ap[k], 0, nullptr, nullptr));
                }
                if (full) {
                    cl11_rp_unpack(G, cap, nrows, hp);
                } else {
                    nbuf.resize((size_t) nrows * G.rowb);
                    cl11_rp_unpack(G, cap, nrows, nbuf.data());
                    memcpy(host + (slice_lo + ov_lo - offset), nbuf.data() + (ov_lo - nat_lo), ov_hi - ov_lo);
                }
            }
        }
    }
    return true;
}

static void ggml_backend_cl11_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto * bctx = (cl11_buffer_context *) buffer->context;
    {
        std::lock_guard<std::mutex> lock(bctx->dev->mutex);
        cl11_release_const_subs_for_parent(bctx->dev->const_subs, bctx->mem);
    }
    if (bctx->mem) {
        clReleaseMemObject(bctx->mem);
    }
    delete bctx;
}

static void * ggml_backend_cl11_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return CL11_BASE;
}

static void ggml_backend_cl11_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    auto * bctx = (cl11_buffer_context *) buffer->context;
    if (size == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(bctx->dev->mutex);
    if (!cl11_ensure_program(bctx->dev) ||
            !cl11_fill(bctx->dev->queue, bctx->dev->exec, bctx->dev->kernels[K_FILL_U32], bctx->dev->kernels[K_FILL_U8],
                       bctx->dev->cfg, bctx->mem, cl11_tensor_offset(tensor) + offset, size, value)) {
        GGML_ABORT("%s: memset failed\n", __func__);
    }
}

static void ggml_backend_cl11_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    auto * bctx = (cl11_buffer_context *) buffer->context;
    if (size == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(bctx->dev->mutex);
    cl11_rp_check_view(tensor);
    if (cl11_repacked(tensor)) {
        if (!cl11_rp_transfer(bctx->dev, bctx->mem, tensor, (uint8_t *) const_cast<void *>(data), offset, size, true)) {
            GGML_ABORT("%s: transfer of the repacked tensor %s failed\n", __func__, tensor->name);
        }
        return;
    }
    // blocking write: it also waits for the kernels that are already enqueued
    const cl_int err = clEnqueueWriteBuffer(bctx->dev->queue, bctx->mem, CL_TRUE, cl11_tensor_offset(tensor) + offset, size, data, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        GGML_ABORT("%s: clEnqueueWriteBuffer failed: %s\n", __func__, cl11_err_str(err));
    }
}

static void ggml_backend_cl11_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    auto * bctx = (cl11_buffer_context *) buffer->context;
    if (size == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(bctx->dev->mutex);
    cl11_rp_check_view(tensor);
    if (cl11_repacked(tensor)) {
        if (!cl11_rp_transfer(bctx->dev, bctx->mem, tensor, (uint8_t *) data, offset, size, false)) {
            GGML_ABORT("%s: transfer of the repacked tensor %s failed\n", __func__, tensor->name);
        }
        return;
    }
    const cl_int err = clEnqueueReadBuffer(bctx->dev->queue, bctx->mem, CL_TRUE, cl11_tensor_offset(tensor) + offset, size, data, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        GGML_ABORT("%s: clEnqueueReadBuffer failed: %s\n", __func__, cl11_err_str(err));
    }
}

static bool ggml_backend_cl11_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    auto * dctx_buf = (cl11_buffer_context *) buffer->context;
    cl11_device * c = dctx_buf->dev;
    const size_t size = ggml_nbytes(src);
    if (size == 0) {
        return true;
    }

    ggml_backend_buffer_t src_buf = cl11_tensor_buffer(src);
    if (src_buf == nullptr) {
        return false;
    }

    if (ggml_backend_buffer_is_host(src_buf)) {
        // host -> device
        std::lock_guard<std::mutex> lock(c->mutex);
        if (cl11_repacked(dst)) {
            return cl11_rp_transfer(c, dctx_buf->mem, dst, (uint8_t *) src->data, 0, size, true);
        }
        return clEnqueueWriteBuffer(c->queue, dctx_buf->mem, CL_TRUE, cl11_tensor_offset(dst), size, src->data, 0, nullptr, nullptr) == CL_SUCCESS;
    }

    if (src_buf->iface.free_buffer == ggml_backend_cl11_buffer_free_buffer) {
        if (cl11_repacked(src) != cl11_repacked(dst)) {
            return false; // different layouts: let ggml copy through the host
        }
        auto * sctx = (cl11_buffer_context *) src_buf->context;
        if (sctx->dev != c) {
            return false; // another cl_context: let ggml copy through the host
        }
        const size_t so = cl11_tensor_offset(src);
        const size_t d_o = cl11_tensor_offset(dst);
        if (sctx->mem == dctx_buf->mem && so < d_o + size && d_o < so + size) {
            return false; // overlapping regions are not allowed in clEnqueueCopyBuffer
        }
        std::lock_guard<std::mutex> lock(c->mutex);
        return clEnqueueCopyBuffer(c->queue, sctx->mem, dctx_buf->mem, so, d_o, size, 0, nullptr, nullptr) == CL_SUCCESS;
    }

    return false;
}

static void ggml_backend_cl11_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * bctx = (cl11_buffer_context *) buffer->context;
    std::lock_guard<std::mutex> lock(bctx->dev->mutex);
    if (!cl11_ensure_program(bctx->dev) ||
            !cl11_fill(bctx->dev->queue, bctx->dev->exec, bctx->dev->kernels[K_FILL_U32], bctx->dev->kernels[K_FILL_U8],
                       bctx->dev->cfg, bctx->mem, 0, bctx->size, value)) {
        GGML_ABORT("%s: clear failed\n", __func__);
    }
}

static const ggml_backend_buffer_i ggml_backend_cl11_buffer_i = {
    /* .free_buffer   = */ ggml_backend_cl11_buffer_free_buffer,
    /* .get_base      = */ ggml_backend_cl11_buffer_get_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ ggml_backend_cl11_buffer_memset_tensor,
    /* .set_tensor    = */ ggml_backend_cl11_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_cl11_buffer_get_tensor,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ ggml_backend_cl11_buffer_cpy_tensor,
    /* .clear         = */ ggml_backend_cl11_buffer_clear,
    /* .reset         = */ nullptr,
};

//
// buffer type
//

static const char * ggml_backend_cl11_buft_get_name(ggml_backend_buffer_type_t buft) {
    auto * c = (cl11_device_context *) buft->context;
    return c->device->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_cl11_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto * c = (cl11_device_context *) buft->context;
    cl11_device * dev = c->device.get();
    std::lock_guard<std::mutex> lock(dev->mutex);
    if (!cl11_ensure_context(dev)) {
        return nullptr;
    }

    cl_mem mem = nullptr;
    size_t alloc_size = 0;
    if (!cl11_alloc_buffer(dev->context, dev->queue, dev->info, dev->name.c_str(), __func__, dev->strict_alloc_dev, size, mem, alloc_size)) {
        return nullptr;
    }

    auto * bctx = new cl11_buffer_context{ dev, mem, alloc_size };
    return ggml_backend_buffer_init(buft, ggml_backend_cl11_buffer_i, bctx, size);
}

static size_t ggml_backend_cl11_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    auto * c = (cl11_device_context *) buft->context;
    return cl11_buffer_alignment(c->device->info);
}

static size_t ggml_backend_cl11_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    auto * c = (cl11_device_context *) buft->context;
    return cl11_max_buffer_size(c->device->info);
}

static bool ggml_backend_cl11_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return false;
}

static const ggml_backend_buffer_type_i ggml_backend_cl11_buft_i = {
    /* .get_name         = */ ggml_backend_cl11_buft_get_name,
    /* .alloc_buffer     = */ ggml_backend_cl11_buft_alloc_buffer,
    /* .alloc_buffer_n   = */ nullptr,
    /* .get_alignment    = */ ggml_backend_cl11_buft_get_alignment,
    /* .get_max_size     = */ ggml_backend_cl11_buft_get_max_size,
    /* .get_alloc_size   = */ nullptr,
    /* .get_alloc_size_n = */ nullptr,
    /* .is_host          = */ ggml_backend_cl11_buft_is_host,
};

//
// supported operations
//

static int cl11_qtype_index(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:  return CL11_QT_F32;
        case GGML_TYPE_F16:  return CL11_QT_F16;
        case GGML_TYPE_Q4_0: return CL11_QT_Q4_0;
        case GGML_TYPE_Q8_0: return CL11_QT_Q8_0;
        case GGML_TYPE_Q4_K: return CL11_QT_Q4_K;
        case GGML_TYPE_Q6_K: return CL11_QT_Q6_K;
        default:             return -1;
    }
}

static bool cl11_is_float(ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16;
}

// the offsets and strides are passed as 32-bit integers
static bool cl11_fits_int(const ggml_tensor * t) {
    if (t == nullptr) {
        return true;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (t->ne[i] > INT_MAX || t->nb[i] > INT_MAX) {
            return false;
        }
    }
    // (keep a margin: the kernels compute group_index * work_group_size + local_id in 32 bits)
    return ggml_nelements(t) <= INT_MAX - 65536 && ggml_nbytes(t) <= (size_t) INT_MAX - 65536;
}

// ARGSORT / TOP_K: a row is sorted in local memory (2 x 4 bytes per element, padded to a power of 2)
static int64_t cl11_sort_npad(int64_t ncols) {
    int64_t n = 1;
    while (n < ncols) {
        n <<= 1;
    }
    return n;
}

static bool cl11_sort_fits(const cl11_device * c, int64_t ncols) {
    return ncols >= 1 && cl11_sort_npad(ncols) <= cl11_sort_nmax(c->info.local_mem);
}

// GGML_CL11_NO_OPS=NAME,NAME,...: supports_op is false for these operations (ggml_op_desc names, e.g. MUL_MAT_ID,ARGSORT); debugging of mixed CPU / GPU graphs
static bool cl11_op_disabled(const ggml_tensor * op) {
    static const std::string list = [] {
        const char * env = getenv("GGML_CL11_NO_OPS");
        return env != nullptr ? "," + std::string(env) + "," : std::string();
    }();
    return !list.empty() && list.find("," + std::string(ggml_op_desc(op)) + ",") != std::string::npos;
}

static bool cl11_supports_op_impl(const cl11_device * c, const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    const ggml_tensor * src2 = op->src[2];

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        default:
            break;
    }

    if (cl11_op_disabled(op)) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (!cl11_fits_int(op->src[i])) {
            return false;
        }
    }
    if (!cl11_fits_int(op)) {
        return false;
    }
    if (c->strict_alloc_dev) {
        // the driver cannot allocate a tensor larger than CL_DEVICE_MAX_MEM_ALLOC_SIZE
        if (ggml_nbytes(op) > cl11_max_buffer_size(c->info)) {
            return false;
        }
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            if (op->src[i] != nullptr && ggml_nbytes(op->src[i]) > cl11_max_buffer_size(c->info)) {
                return false;
            }
        }
    }

    switch (op->op) {
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            return src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   ggml_are_same_shape(src0, op);
        case GGML_OP_SCALE:
        case GGML_OP_CLAMP:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
            return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_TANH:
                    return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
                default:
                    return false;
            }
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_REGLU:
                    if (src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                        return false;
                    }
                    if (src1 != nullptr) {
                        return src1->type == GGML_TYPE_F32 && ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, op);
                    }
                    return src0->ne[0] == 2 * op->ne[0];
                default:
                    return false;
            }
        case GGML_OP_RMS_NORM:
            return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   src0->nb[0] == sizeof(float) && op->nb[0] == sizeof(float);
        case GGML_OP_SOFT_MAX: {
            if (src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32 || src2 != nullptr) {
                return false; // attention sinks are not supported
            }
            float max_bias = 0.0f;
            memcpy(&max_bias, (const int32_t *) op->op_params + 1, sizeof(float));
            if (max_bias != 0.0f) {
                return false; // ALiBi is not supported
            }
            if (src0->nb[0] != sizeof(float) || op->nb[0] != sizeof(float)) {
                return false;
            }
            if (src1 != nullptr) {
                if (!cl11_is_float(src1->type) || src1->nb[0] != ggml_type_size(src1->type) ||
                    src1->ne[0] != src0->ne[0] || src1->ne[1] < src0->ne[1]) {
                    return false;
                }
            }
            return true;
        }
        case GGML_OP_ROPE: {
            const int n_dims = ((const int32_t *) op->op_params)[1];
            const int mode   = ((const int32_t *) op->op_params)[2];
            const int n_offs = ((const int32_t *) op->op_params)[15];
            if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) {
                return false; // no mrope / vision
            }
            if (!cl11_is_float(src0->type) || op->type != src0->type || src1 == nullptr || src1->type != GGML_TYPE_I32) {
                return false;
            }
            if (src2 != nullptr && (src2->type != GGML_TYPE_F32 || src2->ne[0] < n_dims / 2 || src2->nb[0] != sizeof(float))) {
                return false;
            }
            return n_dims > 0 && n_dims % 2 == 0 && op->ne[0] % 2 == 0 && n_offs >= 0 && n_offs % 2 == 0 &&
                   n_offs + n_dims <= op->ne[0] &&
                   src0->nb[0] == ggml_type_size(src0->type) && op->nb[0] == ggml_type_size(op->type) &&
                   src1->ne[0] >= op->ne[2];
        }
        case GGML_OP_GET_ROWS:
            return cl11_qtype_index(src0->type) >= 0 && (!cl11_rp_type(src0->type) || cl11_repacked(src0)) &&
                   src1->type == GGML_TYPE_I32 && op->type == GGML_TYPE_F32 &&
                   src0->nb[0] == ggml_type_size(src0->type) &&
                   src0->ne[0] == op->ne[0];
        case GGML_OP_SET_ROWS:
            return cl11_is_float(src0->type) && cl11_is_float(op->type) &&
                   (src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32) &&
                   src0->nb[0] == ggml_type_size(src0->type) && op->nb[0] == ggml_type_size(op->type) &&
                   src0->ne[0] == op->ne[0] && src0->ne[2] % src1->ne[1] == 0 && src0->ne[3] % src1->ne[2] == 0;
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT:
            return cl11_is_float(src0->type) && cl11_is_float(op->type) && ggml_nelements(src0) == ggml_nelements(op);
        case GGML_OP_MUL_MAT:
            return cl11_qtype_index(src0->type) >= 0 && (!cl11_rp_type(src0->type) || cl11_repacked(src0)) &&
                   src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   src0->nb[0] == ggml_type_size(src0->type) &&
                   src1->nb[0] == sizeof(float) && op->nb[0] == sizeof(float) &&
                   src0->ne[0] == src1->ne[0] &&
                   src1->ne[2] % src0->ne[2] == 0 && src1->ne[3] % src0->ne[3] == 0;
        case GGML_OP_ADD_ID:
            return src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && src2 != nullptr && src2->type == GGML_TYPE_I32 && op->type == GGML_TYPE_F32 &&
                   src0->nb[0] == sizeof(float) && src1->nb[0] == sizeof(float) && src2->nb[0] == sizeof(int32_t) && op->nb[0] == sizeof(float) &&
                   src0->ne[0] == src1->ne[0] && src0->ne[3] == 1 && src2->ne[0] == src0->ne[1] && src2->ne[1] == src0->ne[2] && ggml_are_same_shape(src0, op);
        case GGML_OP_SSM_CONV:
            return src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   src0->nb[0] == sizeof(float) && src1->nb[0] == sizeof(float) && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
                   src0->ne[1] == src1->ne[1] && src0->ne[0] == src1->ne[0] - 1 + op->ne[1] && op->ne[0] == src0->ne[1];
        case GGML_OP_CONCAT: {
            const int dim = ggml_get_op_params_i32(op, 0);
            return src0->type == src1->type && src0->type == op->type && !ggml_is_quantized(src0->type) &&
                   (ggml_type_size(src0->type) == 4 || ggml_type_size(src0->type) == 2) && ggml_blck_size(src0->type) == 1 &&
                   src0->type != GGML_TYPE_F64 && dim >= 0 && dim < GGML_MAX_DIMS;
        }
        case GGML_OP_PAD:
            return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && src0->nb[0] == sizeof(float) && op->nb[0] == sizeof(float);
        case GGML_OP_SUM_ROWS:
            return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && op->ne[0] == 1;
        case GGML_OP_ARGSORT:
        case GGML_OP_TOP_K:
            // the whole row is sorted in local memory
            return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_I32 && src0->nb[0] == sizeof(float) && op->nb[0] == sizeof(int32_t) &&
                   cl11_sort_fits(c, src0->ne[0]);
        case GGML_OP_MUL_MAT_ID: {
            // src0: experts [K, M, n_expert], src1: [K, 1 or n_used, n_tokens], src2 (ids): [n_used, n_tokens], dst [M, n_used, n_tokens]
            if (src2 == nullptr || src2->type != GGML_TYPE_I32 || src1->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                return false;
            }
            if (cl11_qtype_index(src0->type) < 0 || (cl11_rp_type(src0->type) && !cl11_repacked(src0))) {
                return false;
            }
            if (src0->nb[0] != ggml_type_size(src0->type) || src1->nb[0] != sizeof(float) || op->nb[0] != sizeof(float) || src2->nb[0] != sizeof(int32_t) ||
                src0->ne[3] != 1 || src1->ne[3] != 1 || src2->ne[2] != 1 || src2->ne[3] != 1 || src0->ne[0] != src1->ne[0]) {
                return false;
            }
            if (src1->ne[2] != src2->ne[1] || (src1->ne[1] != 1 && src1->ne[1] != src2->ne[0]) ||
                op->ne[0] != src0->ne[1] || op->ne[1] != src2->ne[0] || op->ne[2] != src2->ne[1]) {
                return false;
            }
            // the scratch buffers of the prompt processing path
            const size_t pairs = (size_t) (src2->ne[0] * src2->ne[1]);
            return pairs * (size_t) std::max(src0->ne[0], src0->ne[1]) * sizeof(float) <= cl11_max_buffer_size(c->info);
        }
        default:
            return false;
    }
}

//
// graph compute
//

static bool cl11_op_binary(cl11_device * c, const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    cl11_kernel_id kid;
    switch (dst->op) {
        case GGML_OP_ADD: kid = K_ADD; break;
        case GGML_OP_SUB: kid = K_SUB; break;
        case GGML_OP_MUL: kid = K_MUL; break;
        default:          kid = K_DIV; break;
    }
    const cl11_tref s0 = cl11_ref(src0), s1 = cl11_ref(src1), d = cl11_ref(dst);
    const int32_t n = (int32_t) ggml_nelements(dst);

    cl11_args a(c->kernels[kid]);
    a.set(s0.mem).set(s0.td).set(s1.mem).set(s1.td).set(d.mem).set(d.td).set(n);
    return cl11_launch(c->queue, c->exec, a, c->cfg.wg_ew, cl11_div_up(n, c->cfg.wg_ew));
}

// UNARY, SCALE, CLAMP, SQR, SQRT: op codes of k_unary
static bool cl11_op_unary(cl11_device * c, const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    int32_t code = 0;
    float p0 = 0.0f, p1 = 0.0f;
    switch (dst->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_SILU:    code = 0; break;
                case GGML_UNARY_OP_RELU:    code = 1; break;
                case GGML_UNARY_OP_NEG:     code = 2; break;
                case GGML_UNARY_OP_ABS:     code = 3; break;
                case GGML_UNARY_OP_SIGMOID: code = 4; break;
                case GGML_UNARY_OP_TANH:    code = 5; break;
                default: return false;
            }
            break;
        case GGML_OP_SQR:   code = 6; break;
        case GGML_OP_SQRT:  code = 7; break;
        case GGML_OP_SCALE:
            code = 8;
            memcpy(&p0, (const int32_t *) dst->op_params + 0, sizeof(float));
            memcpy(&p1, (const int32_t *) dst->op_params + 1, sizeof(float));
            break;
        case GGML_OP_CLAMP:
            code = 9;
            memcpy(&p0, (const int32_t *) dst->op_params + 0, sizeof(float));
            memcpy(&p1, (const int32_t *) dst->op_params + 1, sizeof(float));
            break;
        default:
            return false;
    }

    const cl11_tref s = cl11_ref(src0), d = cl11_ref(dst);
    const int32_t n = (int32_t) ggml_nelements(dst);

    cl11_args a(c->kernels[K_UNARY]);
    a.set(s.mem).set(s.td).set(d.mem).set(d.td).set(n).set(code).set(p0).set(p1);
    return cl11_launch(c->queue, c->exec, a, c->cfg.wg_ew, cl11_div_up(n, c->cfg.wg_ew));
}

static bool cl11_op_glu(cl11_device * c, const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    const int32_t code    = ggml_get_glu_op(dst) == GGML_GLU_OP_SWIGLU ? 0 : 1;
    const int32_t split   = src1 != nullptr ? 1 : 0;
    const int32_t swapped = ggml_get_op_params_i32(dst, 1);
    const int32_t nc      = (int32_t) dst->ne[0];

    const cl11_tref s0 = cl11_ref(src0), s1 = cl11_ref(src1 ? src1 : src0), d = cl11_ref(dst);
    const int32_t n = (int32_t) ggml_nelements(dst);

    cl11_args a(c->kernels[K_GLU]);
    a.set(s0.mem).set(s0.td).set(s1.mem).set(s1.td).set(d.mem).set(d.td).set(n).set(code).set(split).set(swapped).set(nc);
    return cl11_launch(c->queue, c->exec, a, c->cfg.wg_ew, cl11_div_up(n, c->cfg.wg_ew));
}

static bool cl11_op_rms_norm(cl11_device * c, const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    float eps = 0.0f;
    memcpy(&eps, (const int32_t *) dst->op_params + 0, sizeof(float));

    const cl11_tref s = cl11_ref(src0), d = cl11_ref(dst);
    cl11_args a(c->kernels[K_RMS_NORM]);
    a.set(s.mem).set(s.td).set(d.mem).set(d.td).set(eps);
    return cl11_launch(c->queue, c->exec, a, c->cfg.wg_row, (size_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]));
}

// RMS_NORM (F32) followed by a MUL with a weight vector: the weights `w` (ne0 contiguous floats) of the MUL `mul` whose other operand is the
// RMS_NORM node i, which has no other user
static const ggml_tensor * cl11_norm_mul_weights(const cl11_device * c, const ggml_cgraph * cgraph, int i) {
    if (c->no_fuse || i + 1 >= cgraph->n_nodes) {
        return nullptr;
    }
    const ggml_tensor * norm = cgraph->nodes[i];
    const ggml_tensor * mul  = cgraph->nodes[i + 1];
    if (norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL || (mul->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
        !ggml_node_has_n_uses(cgraph, i, 1) || norm->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) {
        return nullptr;
    }
    const ggml_tensor * w = mul->src[0] == norm ? mul->src[1] : (mul->src[1] == norm ? mul->src[0] : nullptr);
    if (w == nullptr || w == norm || w->type != GGML_TYPE_F32 || ggml_nelements(w) != norm->ne[0] || w->ne[0] != norm->ne[0] ||
        w->nb[0] != sizeof(float) || !ggml_are_same_shape(mul, norm) || mul->nb[0] != sizeof(float) || norm->src[0]->nb[0] != sizeof(float) ||
        norm->src[0]->type != GGML_TYPE_F32 || !cl11_fits_int(w)) {
        return nullptr;
    }
    return w;
}

static bool cl11_op_rms_norm_mul(cl11_device * c, const ggml_tensor * norm, const ggml_tensor * w, const ggml_tensor * mul) {
    const ggml_tensor * src0 = norm->src[0];

    float eps = 0.0f;
    memcpy(&eps, (const int32_t *) norm->op_params + 0, sizeof(float));

    const cl11_tref s = cl11_ref(src0), wr = cl11_ref(w), d = cl11_ref(mul);
    cl11_args a(c->kernels[K_RMS_NORM_MUL]);
    a.set(s.mem).set(s.td).set(wr.mem).set(wr.td).set(d.mem).set(d.td).set(eps);
    return cl11_launch(c->queue, c->exec, a, c->cfg.wg_row, (size_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]));
}

// ADD (F32, one row, same shapes, no broadcast) + RMS_NORM + MUL: nodes i, i + 1, i + 2 (the ADD result has other users)
static bool cl11_can_fuse_add_norm_mul(const cl11_device * c, const ggml_cgraph * cgraph, int i, const ggml_tensor ** w) {
    if (c->no_fuse || i + 2 >= cgraph->n_nodes) {
        return false;
    }
    const ggml_tensor * add = cgraph->nodes[i];
    const ggml_tensor * norm = cgraph->nodes[i + 1];
    if (add->op != GGML_OP_ADD || norm->op != GGML_OP_RMS_NORM || norm->src[0] != add || (add->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
        (norm->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        return false;
    }
    const ggml_tensor * a = add->src[0], * b = add->src[1];
    if (a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 || !ggml_are_same_shape(a, b) || !ggml_are_same_shape(a, add) ||
        ggml_nelements(add) != add->ne[0] || a->nb[0] != sizeof(float) || b->nb[0] != sizeof(float) || add->nb[0] != sizeof(float) ||
        !cl11_fits_int(a) || !cl11_fits_int(b) || !cl11_fits_int(add)) {
        return false;
    }
    *w = cl11_norm_mul_weights(c, cgraph, i + 1);
    return *w != nullptr && cgraph->nodes[i + 2]->ne[0] == add->ne[0] && ggml_nelements(cgraph->nodes[i + 2]) == add->ne[0];
}

static bool cl11_op_add_rms_norm_mul(cl11_device * c, const ggml_tensor * add, const ggml_tensor * norm, const ggml_tensor * w, const ggml_tensor * mul) {
    float eps = 0.0f;
    memcpy(&eps, (const int32_t *) norm->op_params + 0, sizeof(float));

    const cl11_tref a = cl11_ref(add->src[0]), b = cl11_ref(add->src[1]), wr = cl11_ref(w), sr = cl11_ref(add), d = cl11_ref(mul);
    cl11_args args(c->kernels[K_ADD_RMS_NORM_MUL]);
    args.set(a.mem).set(a.td).set(b.mem).set(b.td).set(wr.mem).set(wr.td).set(sr.mem).set(sr.td).set(d.mem).set(d.td).set(eps);
    return cl11_launch(c->queue, c->exec, args, c->cfg.wg_row, 1);
}

static bool cl11_op_soft_max(cl11_device * c, const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1]; // mask

    float scale = 1.0f;
    memcpy(&scale, (const int32_t *) dst->op_params + 0, sizeof(float));

    const cl11_tref s = cl11_ref(src0), d = cl11_ref(dst);
    const cl11_tref m = src1 ? cl11_ref(src1) : s;
    cl11_tref mm = m;
    if (!src1) {
        for (int i = 0; i < 4; ++i) {
            mm.td.ne[i] = 1;
            mm.td.nb[i] = 0;
            cl11_fastdiv_init(1, mm.td.fm[i], mm.td.fl[i]);
        }
        mm.td.off = 0;
    }
    const int32_t has_mask = src1 ? 1 : 0;
    const int32_t mask_f16 = (src1 && src1->type == GGML_TYPE_F16) ? 1 : 0;

    cl11_args a(c->kernels[K_SOFT_MAX]);
    a.set(s.mem).set(s.td).set(mm.mem).set(mm.td).set(has_mask).set(mask_f16).set(d.mem).set(d.td).set(scale);
    return cl11_launch(c->queue, c->exec, a, c->cfg.wg_row, (size_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]));
}

static bool cl11_op_rope(cl11_device * c, const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1]; // positions
    const ggml_tensor * src2 = dst->src[2]; // freq factors

    const int32_t * op = (const int32_t *) dst->op_params;
    const int n_dims     = op[1];
    const int mode       = op[2];
    const int n_ctx_orig = op[4];
    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    memcpy(&freq_base,   op +  5, sizeof(float));
    memcpy(&freq_scale,  op +  6, sizeof(float));
    memcpy(&ext_factor,  op +  7, sizeof(float));
    memcpy(&attn_factor, op +  8, sizeof(float));
    memcpy(&beta_fast,   op +  9, sizeof(float));
    memcpy(&beta_slow,   op + 10, sizeof(float));
    const int n_offs = op[15];

    const float theta_scale = powf(freq_base, -2.0f / n_dims);
    float corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    const cl11_tref s = cl11_ref(src0), p = cl11_ref(src1), d = cl11_ref(dst);
    const cl11_tref f = src2 ? cl11_ref(src2) : p;
    const int32_t has_ff = src2 ? 1 : 0;
    const int32_t neox   = mode == GGML_ROPE_TYPE_NEOX ? 1 : 0;
    const int32_t nd = n_dims, no = n_offs;

    cl11_args a(c->kernels[src0->type == GGML_TYPE_F16 ? K_ROPE_F16 : K_ROPE_F32]);
    a.set(s.mem).set(s.td).set(p.mem).set(p.td).set(f.mem).set(f.td).set(has_ff).set(d.mem).set(d.td)
     .set(nd).set(neox).set(no)
     .set(theta_scale).set(freq_scale).set(ext_factor).set(attn_factor).set(corr_dims[0]).set(corr_dims[1]);

    // one work-group per row
    return cl11_launch(c->queue, c->exec, a, c->cfg.wg_row, (size_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]));
}

static bool cl11_op_get_rows(cl11_device * c, const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    const cl11_tref s0 = cl11_ref(src0), s1 = cl11_ref(src1), d = cl11_ref(dst);
    cl11_args a(c->kernels[K_GET_ROWS + cl11_qtype_index(src0->type)]);
    a.set(s0.mem).set(s0.td).set(s1.mem).set(s1.td).set(d.mem).set(d.td);
    return cl11_launch(c->queue, c->exec, a, c->cfg.wg_row, (size_t) (src1->ne[0] * src1->ne[1] * src1->ne[2]));
}

static bool cl11_op_set_rows(cl11_device * c, const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    const int kid = K_SET_ROWS + (src0->type == GGML_TYPE_F16 ? 2 : 0) + (dst->type == GGML_TYPE_F16 ? 1 : 0);
    const cl11_tref s0 = cl11_ref(src0), s1 = cl11_ref(src1), d = cl11_ref(dst);
    const int32_t n = (int32_t) ggml_nelements(src0);
    const int32_t idx64 = src1->type == GGML_TYPE_I64 ? 1 : 0;

    cl11_args a(c->kernels[kid]);
    a.set(s0.mem).set(s0.td).set(s1.mem).set(s1.td).set(d.mem).set(d.td).set(n).set(idx64);
    return cl11_launch(c->queue, c->exec, a, c->cfg.wg_ew, cl11_div_up(n, c->cfg.wg_ew));
}

// two SET_ROWS with the same types in one launch (k_set_rows2_*)
static bool cl11_op_set_rows2(cl11_device * c, const ggml_tensor * da, const ggml_tensor * db) {
    const int kid = K_SET_ROWS2 + (da->src[0]->type == GGML_TYPE_F16 ? 2 : 0) + (da->type == GGML_TYPE_F16 ? 1 : 0);
    const cl11_tref a0 = cl11_ref(da->src[0]), a1 = cl11_ref(da->src[1]), ad = cl11_ref(da);
    const cl11_tref b0 = cl11_ref(db->src[0]), b1 = cl11_ref(db->src[1]), bd = cl11_ref(db);
    const int32_t na = (int32_t) ggml_nelements(da->src[0]);
    const int32_t nb = (int32_t) ggml_nelements(db->src[0]);
    const int32_t groups_a = (int32_t) cl11_div_up(na, c->cfg.wg_ew);

    cl11_args a(c->kernels[kid]);
    a.set(a0.mem).set(a0.td).set(a1.mem).set(a1.td).set(ad.mem).set(ad.td).set(na).set((int32_t) (da->src[1]->type == GGML_TYPE_I64 ? 1 : 0));
    a.set(b0.mem).set(b0.td).set(b1.mem).set(b1.td).set(bd.mem).set(bd.td).set(nb).set((int32_t) (db->src[1]->type == GGML_TYPE_I64 ? 1 : 0));
    a.set(groups_a);
    return cl11_launch(c->queue, c->exec, a, c->cfg.wg_ew, groups_a + cl11_div_up(nb, c->cfg.wg_ew));
}

// can the SET_ROWS `b` be computed in the same launch as `a` (which comes before it, only nodes without computation in between)?
static bool cl11_can_pair_set_rows(const cl11_device * c, const ggml_tensor * a, const ggml_tensor * b) {
    if (!c->fuse_set_rows || b->op != GGML_OP_SET_ROWS || (b->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
        c->kernels[K_SET_ROWS2 + (a->src[0]->type == GGML_TYPE_F16 ? 2 : 0) + (a->type == GGML_TYPE_F16 ? 1 : 0)] == nullptr) {
        return false;
    }
    if (a->src[0]->type != b->src[0]->type || a->type != b->type) {
        return false;
    }
    auto base = [](const ggml_tensor * t) { return t->view_src ? t->view_src : t; };
    // b must not read what a writes, and they must not write the same tensor
    if (base(a) == base(b)) {
        return false;
    }
    for (int k = 0; k < 2; ++k) {
        if (b->src[k] == a || base(b->src[k]) == base(a)) {
            return false;
        }
    }
    return true;
}

static bool cl11_op_cpy(cl11_device * c, const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    const cl11_tref s = cl11_ref(src0), d = cl11_ref(dst);

    // transposing copy of a tensor whose dim 1 is contiguous into a contiguous tensor of the same type and shape
    if (src0->type == dst->type && ggml_are_same_shape(src0, dst) && ggml_is_contiguous(dst) &&
        src0->nb[1] == ggml_type_size(src0->type) && src0->nb[0] > src0->nb[1] && src0->ne[0] > 1 && src0->ne[1] > 1 &&
        c->cfg.gt >= 8 && !c->no_cpy_t) {
        const int64_t tiles0 = (dst->ne[0] + 31) / 32;
        const int64_t tiles1 = (dst->ne[1] + 31) / 32;
        const cl11_bcast bc = cl11_make_bcast(1, 1, tiles0, tiles1);
        cl11_args a(c->kernels[src0->type == GGML_TYPE_F16 ? K_CPY_T_F16 : K_CPY_T_F32]);
        a.set(s.mem).set(s.td).set(d.mem).set(d.td).set(bc);
        return cl11_launch(c->queue, c->exec, a, (size_t) c->cfg.gt * c->cfg.gt, (size_t) (tiles0 * tiles1 * dst->ne[2] * dst->ne[3]));
    }

    const int kid = K_CPY_F32_F32 + (src0->type == GGML_TYPE_F16 ? 2 : 0) + (dst->type == GGML_TYPE_F16 ? 1 : 0);
    const int32_t n = (int32_t) ggml_nelements(src0);

    cl11_args a(c->kernels[kid]);
    a.set(s.mem).set(s.td).set(d.mem).set(d.td).set(n);
    return cl11_launch(c->queue, c->exec, a, c->cfg.wg_ew, cl11_div_up(n, c->cfg.wg_ew));
}

// a node that writes into a column the scratch buffers were made from makes them stale
static void cl11_xprep_note_write(cl11_device * c, const ggml_tensor * node) {
    if (node->data == nullptr) {
        return;
    }
    const cl11_tref d = cl11_ref(node);
    if (c->mid_ids.valid && d.mem == c->mid_ids.mem) {
        const size_t b = (size_t) d.td.off, e = b + ggml_nbytes(node);
        if (b < c->mid_ids.off + c->mid_ids.raw.size() && c->mid_ids.off < e) {
            c->mid_ids.valid = false;
        }
    }
    for (auto & x : c->xprep) {
        if (x.valid && d.mem == x.src) {
            const size_t b = (size_t) d.td.off, e = b + ggml_nbytes(node);
            if (b < x.off + x.bytes && x.off < e) {
                x.valid = false;
            }
        }
    }
}

static int cl11_log2_exact(int v) {
    int l = 0;
    while ((1 << l) < v) {
        ++l;
    }
    return l;
}

// conditions of the attention kernels of attn.cl (K cache: k_mvq_f16, transposed V cache: k_mvt_f16): the heads of a position are
// contiguous in the cache (row = ne02 heads of `hd` halves), 16-byte aligned rows, a work-item per 16 bytes of a row, at most 8 query
// heads per head of the cache, src1 (q / x) with 16-byte aligned float4 loads
static bool cl11_attn_ok(const cl11_device * c, const cl11_td & ta, const cl11_td & tb, int64_t hd, int64_t ne02, int64_t r2, int64_t k_len, bool v) {
    const int64_t max_lanes = v ? c->cfg.wg_att : c->cfg.wg_row;
    if (c->no_mvq || c->cfg.wg_row % 32 != 0 || r2 < 1 || r2 > 8 || hd % 8 != 0 || k_len % 8 != 0) {
        return false;
    }
    if (ta.nb[2] != hd * 2 || ne02 * (hd / 8) > max_lanes) {
        return false;
    }
    if (ta.off % 16 != 0 || ta.nb[3] % 16 != 0 || tb.off % 16 != 0 || tb.nb[2] % 16 != 0 || tb.nb[3] % 16 != 0) {
        return false;
    }
    return v ? ta.nb[0] % 16 == 0 : ta.nb[1] % 16 == 0;
}

// The product of the tensors (type `type` of src0) described by s0, s1 and d. `moe`: the operands are slices / scratch columns of MUL_MAT_ID
// (no K cache kernel, and no constant-memory / prepared src1: the scratch columns change from one product to the next).
static bool cl11_mul_mat_core(cl11_device * c, ggml_type type, const cl11_tref & s0, const cl11_tref & s1, const cl11_tref & d, bool moe) {
    const int qt = cl11_qtype_index(type);

    const int64_t ne00 = s0.td.ne[0];
    const int64_t ne01 = s0.td.ne[1];
    const int64_t ne02 = s0.td.ne[2];
    const int64_t ne03 = s0.td.ne[3];
    const int64_t ne11 = s1.td.ne[1];
    const int64_t ne12 = s1.td.ne[2];
    const int64_t ne13 = s1.td.ne[3];
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    const bool rp = cl11_rp_type(type); // repacked weights (R32 layout): the mat-vec kernel is k_mvr_*
    const int  rk = rp ? cl11_rkind(type) : 0;
    if (rp && ne11 <= c->mv_max_cols && c->cfg.wg_mvr[rk] % 32 == 0) {
        if (rk >= CL11_RK_Q4_K && !cl11_ensure_kprogram(c)) {
            return false;
        }
        const cl11_td & ta = s0.td;
        const cl11_td & tb = s1.td;
        const bool a16 = ta.off % 16 == 0 && ta.nb[2] % 16 == 0 && ta.nb[3] % 16 == 0;
        const bool b16 = tb.off % 16 == 0 && tb.nb[1] % 16 == 0 && tb.nb[2] % 16 == 0 && tb.nb[3] % 16 == 0;
        if (a16 && b16) {
            const int64_t rows_wg = 32 * c->cfg.mvr_nrr[rk];
            const int64_t tiles = (ne01 + rows_wg - 1) / rows_wg;
            const cl11_bcast bc = cl11_make_bcast(r2, r3, tiles, 1);
            const size_t x_bytes = (size_t) ne00 * sizeof(float);
            cl_mem xsub = nullptr;
            if (!moe && c->mvr_const_x && ne11 == 1 && ne12 == 1 && ne13 == 1 && x_bytes <= std::min<size_t>(c->max_const, 65536)) {
                xsub = cl11_const_sub(c->const_subs, c->info, s1.mem, (size_t) tb.off, x_bytes);
            }
            if (xsub != nullptr && c->cfg.mvr_dn[rk]) {
                // "denormal products" (Fermi): src1 is scaled and its group sums are computed once (k_xprep) into a scratch buffer, the mat-vec reads it
                // from constant memory; the same column used by the next mat-vec (Q, K, V; gate, up) is not prepared again
                const size_t total = x_bytes + x_bytes / 16;
                const int slot = rk == CL11_RK_Q4_0 ? 1 : rk == CL11_RK_Q4_K ? 2 : 0;
                cl_mem scratch = (total <= std::min<size_t>(c->max_const, 65536) && c->kernels[K_XPREP] != nullptr && (ne00 & 15) == 0) ? cl11_xprep_scratch(c, slot) : nullptr;
                cl_mem dsub = scratch != nullptr ? cl11_const_sub(c->const_subs, c->info, scratch, 0, total) : nullptr;
                if (dsub != nullptr) {
                    const int hbit = rk == CL11_RK_Q4_0 ? 4 : rk == CL11_RK_Q4_K ? 8 : 0;
                    auto & xs = c->xprep[slot];
                    if (!(xs.valid && xs.src == s1.mem && xs.off == (size_t) tb.off && xs.bytes == x_bytes)) {
                        const size_t wgp = std::min<size_t>(64, (size_t) c->cfg.wg_ew);
                        cl11_args ap(c->kernels[K_XPREP]);
                        ap.set(s1.mem).set(s1.td).set(scratch).set(hbit);
                        if (!cl11_launch(c->queue, c->exec, ap, wgp, cl11_div_up((size_t) ne00 / 16, wgp))) {
                            return false;
                        }
                        xs.valid = true;
                        xs.src   = s1.mem;
                        xs.off   = (size_t) tb.off;
                        xs.bytes = x_bytes;
                    }
                    cl11_args a(c->kernels[K_MVRC + rk]);
                    a.set(s0.mem).set(s0.td).set(dsub).set(d.mem).set(d.td).set(bc);
                    return cl11_launch(c->queue, c->exec, a, c->cfg.wg_mvr[rk], (size_t) tiles);
                }
                xsub = nullptr; // too long for the constant memory with the sums: the general kernels
            }
            if (xsub != nullptr && !c->cfg.mvr_dn[rk]) {
                cl11_args a(c->kernels[K_MVRC + rk]);
                a.set(s0.mem).set(s0.td).set(xsub).set(d.mem).set(d.td).set(bc);
                return cl11_launch(c->queue, c->exec, a, c->cfg.wg_mvr[rk], (size_t) tiles);
            }
            cl11_args a(c->kernels[K_MVR + rk]);
            a.set(s0.mem).set(s0.td).set(s1.mem).set(s1.td).set(d.mem).set(d.td).set(bc);
            return cl11_launch(c->queue, c->exec, a, c->cfg.wg_mvr[rk], (size_t) tiles * (size_t) (ne11 * ne12 * ne13));
        }
    }

    // K cache of the attention (F16, grouped query attention): whole rows of the cache per work-group, see attn.cl
    if (!moe && type == GGML_TYPE_F16 && ne11 == 1 && ne13 == ne03 && cl11_attn_ok(c, s0.td, s1.td, ne00, ne02, r2, ne00, false)) {
        const int64_t pw = (c->cfg.wg_row / (ne02 * (ne00 / 8))) * c->cfg.att_p;   // positions per work-group
        const int64_t tiles = (ne01 + pw - 1) / pw;
        const cl11_bcast bc = cl11_make_bcast(r2, 1, tiles, 1);
        cl11_args a(c->kernels[K_MVQ_F16]);
        a.set(s0.mem).set(s0.td).set(s1.mem).set(s1.td).set(d.mem).set(d.td).set(bc);
        return cl11_launch(c->queue, c->exec, a, c->cfg.wg_row, (size_t) (tiles * ne03));
    }

    if (ne11 <= c->mv_max_cols && !rp) {
        const cl11_td & ta = s0.td;
        const cl11_td & tb = s1.td;

        // staged kernels (src1 is copied to local memory, several rows per work-group):
        // long rows, 4-byte aligned rows of src0 and a 16-byte aligned src1
        const bool a4  = ta.off % 4 == 0 && ta.nb[1] % 4 == 0 && ta.nb[2] % 4 == 0 && ta.nb[3] % 4 == 0;
        const bool a16 = ta.off % 16 == 0 && ta.nb[1] % 16 == 0 && ta.nb[2] % 16 == 0 && ta.nb[3] % 16 == 0;
        const bool b16 = tb.off % 16 == 0 && tb.nb[1] % 16 == 0 && tb.nb[2] % 16 == 0 && tb.nb[3] % 16 == 0;
        bool staged = c->mvs_enabled && c->cfg.wg_mv % 32 == 0 && b16 && ne00 >= c->mvs_min_k;
        switch (type) {
            case GGML_TYPE_F16:  staged = staged && a4 && ne00 % 8 == 0;   break;
            default:             staged = staged && a16 && ne00 % 8 == 0;  break;
        }

        if (staged) {
            const int rows_wg = c->cfg.wg_mv / 32 * c->cfg.mvs_nrw;
            const int64_t tiles = (ne01 + rows_wg - 1) / rows_wg;
            const cl11_bcast bc = cl11_make_bcast(r2, r3, tiles, 1);
            // measured on a GT 430: the x copy in local memory did not pay off for F32 / F16
            const int family = c->mvs_family != 0 ? c->mvs_family : 2;
            cl11_args a(c->kernels[(family == 2 ? K_MVW : K_MVX) + qt]);
            a.set(s0.mem).set(s0.td).set(s1.mem).set(s1.td).set(d.mem).set(d.td).set(bc);
            return cl11_launch(c->queue, c->exec, a, c->cfg.wg_mv, (size_t) tiles * (size_t) (ne11 * ne12 * ne13));
        }

        // generic mat-vec: `tpr` threads per output, the number of units of work per row bounds it
        const int64_t units = (ne00 + 3) / 4;
        // short rows (K cache, V cache): several units per thread, fewer threads (and reduction steps) per row
        int tpr = cl11_pow2_floor((size_t) std::max<int64_t>(units, 1));
        if (units <= 128) {
            tpr = std::min(tpr, std::max(4, cl11_pow2_floor((size_t) std::max<int64_t>(units / 4, 1))));
        }
        if (c->mv_tpr > 0) {
            tpr = std::min(tpr, cl11_pow2_floor((size_t) c->mv_tpr));
        }
        tpr = std::min(tpr, c->cfg.wg_mv);
        const size_t total = (size_t) (ne01 * ne11 * ne12 * ne13);
        const size_t rows_per_group = (size_t) (c->cfg.wg_mv / tpr);
        const cl11_bcast bc = cl11_make_bcast(r2, r3, 1, 1);

        cl11_args a(c->kernels[K_MV + qt]);
        a.set(s0.mem).set(s0.td).set(s1.mem).set(s1.td).set(d.mem).set(d.td).set(bc).set((int32_t) tpr).set((int32_t) cl11_log2_exact(tpr));
        return cl11_launch(c->queue, c->exec, a, c->cfg.wg_mv, cl11_div_up(total, rows_per_group));
    }

    // parameterized mat-mat kernels (mul_mm_x.cl), when a variant is selected
    if (c->cfg.mmx.on && c->kernels[K_MMX + qt] != nullptr) {
        const cl11_mmx & mx = c->cfg.mmx;
        const int64_t tiles_m = (ne01 + mx.bm - 1) / mx.bm;
        const int64_t tiles_n = (ne11 + mx.bn - 1) / mx.bn;
        const size_t groups = (size_t) (tiles_m * tiles_n * ne12 * ne13);
        const cl11_bcast bc = cl11_make_bcast(r2, r3, tiles_m, tiles_n);
        cl11_args a(c->kernels[K_MMX + qt]);
        a.set(s0.mem).set(s0.td).set(s1.mem).set(s1.td).set(d.mem).set(d.td).set(bc);
        return cl11_launch(c->queue, c->exec, a, (size_t) mx.nt(), groups);
    }

    // mat-mat: (4*gt) x (mm_bn*gt) tiles (src0 rows x src1 rows)
    const int64_t tile_m = 4 * c->cfg.gt;
    const int64_t tile_n = (int64_t) c->cfg.mm_bn * c->cfg.gt;
    const int64_t tiles_m = (ne01 + tile_m - 1) / tile_m;
    const int64_t tiles_n = (ne11 + tile_n - 1) / tile_n;
    const size_t groups = (size_t) (tiles_m * tiles_n * ne12 * ne13);
    const cl11_bcast bc = cl11_make_bcast(r2, r3, tiles_m, tiles_n);

    cl11_args a(c->kernels[K_MM + qt]);
    a.set(s0.mem).set(s0.td).set(s1.mem).set(s1.td).set(d.mem).set(d.td).set(bc);
    return cl11_launch(c->queue, c->exec, a, (size_t) (c->cfg.gt / 2) * (c->cfg.mm_bn * c->cfg.gt / 4), groups);
}

static bool cl11_op_mul_mat(cl11_device * c, const ggml_tensor * dst) {
    return cl11_mul_mat_core(c, dst->src[0]->type, cl11_ref(dst->src[0]), cl11_ref(dst->src[1]), cl11_ref(dst), false);
}

// ---- the operations of the hybrid / MoE graphs (LFM2, LFM2-MoE): moe_ops.cl ----

static bool cl11_op_add_id(cl11_device * c, const ggml_tensor * dst) {
    const cl11_tref s0 = cl11_ref(dst->src[0]), s1 = cl11_ref(dst->src[1]), s2 = cl11_ref(dst->src[2]), d = cl11_ref(dst);
    const int32_t n = (int32_t) ggml_nelements(dst);
    cl11_args a(c->kernels[K_ADD_ID]);
    a.set(s0.mem).set(s0.td).set(s1.mem).set(s1.td).set(s2.mem).set(s2.td).set(d.mem).set(d.td).set(n);
    return cl11_launch(c->queue, c->exec, a, c->cfg.wg_ew, cl11_div_up(n, c->cfg.wg_ew));
}

static bool cl11_op_ssm_conv(cl11_device * c, const ggml_tensor * dst) {
    const cl11_tref s0 = cl11_ref(dst->src[0]), s1 = cl11_ref(dst->src[1]), d = cl11_ref(dst);
    const int32_t n = (int32_t) ggml_nelements(dst);
    cl11_args a(c->kernels[K_SSM_CONV]);
    a.set(s0.mem).set(s0.td).set(s1.mem).set(s1.td).set(d.mem).set(d.td).set(n);
    return cl11_launch(c->queue, c->exec, a, c->cfg.wg_ew, cl11_div_up(n, c->cfg.wg_ew));
}

static bool cl11_op_concat(cl11_device * c, const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const cl11_tref s0 = cl11_ref(src0), s1 = cl11_ref(dst->src[1]), d = cl11_ref(dst);
    const int32_t n   = (int32_t) ggml_nelements(dst);
    const int32_t dim = ggml_get_op_params_i32(dst, 0);
    const int32_t o   = (int32_t) src0->ne[dim];
    cl11_args a(c->kernels[ggml_type_size(src0->type) == 4 ? K_CONCAT4 : K_CONCAT2]);
    a.set(s0.mem).set(s0.td).set(s1.mem).set(s1.td).set(d.mem).set(d.td).set(n).set(dim).set(o);
    return cl11_launch(c->queue, c->exec, a, c->cfg.wg_ew, cl11_div_up(n, c->cfg.wg_ew));
}

// must match `padp` in src/backends/opencl11/kernels/moe_ops.cl
struct cl11_padp {
    int32_t lp[4];
    int32_t rp[4];
    int32_t w[4];
    int32_t circ;
};
static_assert(sizeof(cl11_padp) == 13 * sizeof(int32_t), "cl11_padp layout");

static bool cl11_op_pad(cl11_device * c, const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    cl11_padp pp = {};
    pp.circ = ggml_get_op_params_i32(dst, 8) != 0;
    for (int i = 0; i < 4; ++i) {
        pp.lp[i] = ggml_get_op_params_i32(dst, 2 * i);
        pp.rp[i] = ggml_get_op_params_i32(dst, 2 * i + 1);
        // circular: i - lp + w >= 0 with w a multiple of ne (the kernel has no integer division)
        const int64_t ne = std::max<int64_t>(src0->ne[i], 1);
        pp.w[i] = (int32_t) (cl11_div_up((size_t) std::max<int32_t>(pp.lp[i], 0), (size_t) ne) * (size_t) ne);
    }
    const cl11_tref s0 = cl11_ref(src0), d = cl11_ref(dst);
    const int32_t n = (int32_t) ggml_nelements(dst);
    cl11_args a(c->kernels[K_PAD]);
    a.set(s0.mem).set(s0.td).set(d.mem).set(d.td).set(n).set(pp);
    return cl11_launch(c->queue, c->exec, a, c->cfg.wg_ew, cl11_div_up(n, c->cfg.wg_ew));
}

static bool cl11_op_sum_rows(cl11_device * c, const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const cl11_tref s0 = cl11_ref(src0), d = cl11_ref(dst);
    const size_t lsz = (size_t) std::min(c->cfg.wg_row, std::max(32, 1 << cl11_log2_exact((int) std::min<int64_t>(src0->ne[0], 1 << 20))));
    cl11_args a(c->kernels[K_SUM_ROWS]);
    a.set(s0.mem).set(s0.td).set(d.mem).set(d.td);
    return cl11_launch(c->queue, c->exec, a, lsz, (size_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]));
}

static bool cl11_op_argsort(cl11_device * c, const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const cl11_tref s0 = cl11_ref(src0), d = cl11_ref(dst);
    const int32_t desc = dst->op == GGML_OP_TOP_K || ggml_get_op_params_i32(dst, 0) == GGML_SORT_ORDER_DESC ? 1 : 0;
    const int32_t npad = (int32_t) cl11_sort_npad(src0->ne[0]);
    const size_t  lsz  = (size_t) std::min<int64_t>(c->cfg.wg_row, std::max<int64_t>(32, npad / 2));
    cl11_args a(c->kernels[K_ARGSORT]);
    a.set(s0.mem).set(s0.td).set(d.mem).set(d.td).set(npad).set(desc);   // no local-memory argument: see k_argsort (Mesa Clover)
    return cl11_launch(c->queue, c->exec, a, lsz, (size_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]));
}

// ---- MUL_MAT_ID: dst[:, i, t] = src0[:, :, ids[i, t]] * src1[:, i % ne11, t] ----

// A descriptor of the sub-tensor made of `ne2` slices of t starting at slice `first`, or of the 2-D tensor [ne0, ne1] at the byte offset `off`
static cl11_td cl11_td_matrix(int64_t ne0, int64_t ne1, size_t nb0, size_t nb1, size_t off) {
    cl11_td t = {};
    t.ne[0] = (int32_t) ne0;
    t.ne[1] = (int32_t) ne1;
    t.ne[2] = 1;
    t.ne[3] = 1;
    t.nb[0] = (int32_t) nb0;
    t.nb[1] = (int32_t) nb1;
    t.nb[2] = (int32_t) (nb1 * (size_t) ne1);
    t.nb[3] = t.nb[2];
    t.off   = (int32_t) off;
    for (int i = 0; i < 4; ++i) {
        cl11_fastdiv_init((uint32_t) t.ne[i], t.fm[i], t.fl[i]);
    }
    return t;
}

// Prompt processing (several tokens): the ids are read back (the queue is finished), the (slot, token) pairs are sorted by expert, the columns of
// src1 of all the pairs are gathered into one scratch buffer in that order, every expert that is used multiplies its range of columns (the normal
// MUL_MAT kernels, one mat-mat or mat-vec per expert) and the result columns are scattered into dst.
static bool cl11_op_mul_mat_id_sync(cl11_device * c, const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    const int64_t K = src0->ne[0], M = src0->ne[1], n_exp = src0->ne[2];
    const int64_t n_used = ids->ne[0], n_tok = ids->ne[1];
    const int64_t P = n_used * n_tok;

    const cl11_tref s0 = cl11_ref(src0), s1 = cl11_ref(src1), d = cl11_ref(dst), ir = cl11_ref(ids);

    auto & cache = c->mid_ids;
    const size_t ids_bytes = (size_t) ((n_tok - 1) * ids->nb[1] + n_used * sizeof(int32_t));
    cl_int err = CL_SUCCESS;
    if (!(cache.valid && cache.mem == ir.mem && cache.off == (size_t) ir.td.off && cache.raw.size() == ids_bytes)) {
        cache.raw.resize(ids_bytes);
        err = clEnqueueReadBuffer(c->queue, ir.mem, CL_TRUE, (size_t) ir.td.off, ids_bytes, cache.raw.data(), 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            cache.valid = false;
            GGML_LOG_ERROR("%s: reading the ids failed: %s\n", __func__, cl11_err_str(err));
            return false;
        }
        cache.valid = true;
        cache.mem   = ir.mem;
        cache.off   = (size_t) ir.td.off;
    }
    const std::vector<uint8_t> & raw = cache.raw;
    std::vector<int32_t> start((size_t) n_exp + 1, 0);
    for (int64_t t = 0; t < n_tok; ++t) {
        for (int64_t i = 0; i < n_used; ++i) {
            int32_t e;
            memcpy(&e, raw.data() + t * ids->nb[1] + i * sizeof(int32_t), sizeof(e));
            if (e < 0 || e >= n_exp) {
                GGML_LOG_ERROR("%s: expert %d out of range (0..%lld)\n", __func__, e, (long long) n_exp - 1);
                return false;
            }
            ++start[(size_t) e + 1];
        }
    }
    for (int64_t e = 0; e < n_exp; ++e) {
        start[(size_t) e + 1] += start[(size_t) e];
    }
    std::vector<int32_t> pairs((size_t) P * 2);
    {
        std::vector<int32_t> pos(start.begin(), start.end() - 1);
        for (int64_t t = 0; t < n_tok; ++t) {
            for (int64_t i = 0; i < n_used; ++i) {
                int32_t e;
                memcpy(&e, raw.data() + t * ids->nb[1] + i * sizeof(int32_t), sizeof(e));
                const int32_t p = pos[(size_t) e]++;
                pairs[(size_t) p * 2]     = (int32_t) i;
                pairs[(size_t) p * 2 + 1] = (int32_t) t;
            }
        }
    }

    if (!cl11_mid_ensure(c, c->mid_gather, (size_t) P * K * sizeof(float)) || !cl11_mid_ensure(c, c->mid_out, (size_t) P * M * sizeof(float)) ||
        !cl11_mid_ensure(c, c->mid_pairs, (size_t) P * 2 * sizeof(int32_t))) {
        return false;
    }
    err = clEnqueueWriteBuffer(c->queue, c->mid_pairs.buf, CL_TRUE, 0, (size_t) P * 2 * sizeof(int32_t), pairs.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        GGML_LOG_ERROR("%s: writing the pairs failed: %s\n", __func__, cl11_err_str(err));
        return false;
    }

    {
        cl11_args a(c->kernels[K_MID_GATHER]);
        a.set(s1.mem).set(s1.td).set(c->mid_gather.buf).set(c->mid_pairs.buf).set((int32_t) K).set((int32_t) src1->ne[1]);
        if (!cl11_launch(c->queue, c->exec, a, (size_t) c->cfg.wg_row, (size_t) P)) {
            return false;
        }
    }
    for (int64_t e = 0; e < n_exp; ++e) {
        const int64_t first = start[(size_t) e], cnt = start[(size_t) e + 1] - first;
        if (cnt == 0) {
            continue;
        }
        cl11_tref xs0 = s0;
        xs0.td = cl11_td_matrix(K, M, src0->nb[0], src0->nb[1], (size_t) s0.td.off + (size_t) e * src0->nb[2]);
        // (the repacked layout only depends on K, M and the type: the strides of the descriptor are those of the tensor)
        xs0.td.nb[2] = s0.td.nb[2];
        xs0.td.nb[3] = s0.td.nb[3];
        cl11_tref xs1, xd;
        xs1.mem = c->mid_gather.buf;
        xs1.td  = cl11_td_matrix(K, cnt, sizeof(float), K * sizeof(float), (size_t) first * K * sizeof(float));
        xd.mem  = c->mid_out.buf;
        xd.td   = cl11_td_matrix(M, cnt, sizeof(float), M * sizeof(float), (size_t) first * M * sizeof(float));
        if (!cl11_mul_mat_core(c, src0->type, xs0, xs1, xd, true)) {
            return false;
        }
    }
    {
        cl11_args a(c->kernels[K_MID_SCATTER]);
        a.set(c->mid_out.buf).set(d.mem).set(d.td).set(c->mid_pairs.buf).set((int32_t) M);
        if (!cl11_launch(c->queue, c->exec, a, (size_t) c->cfg.wg_row, (size_t) P)) {
            return false;
        }
    }
    return true;
}

// Generation (few tokens): the kernels read the expert numbers from the ids on the GPU, one launch for all the selected experts (no
// synchronization). 1: computed, 0: not applicable (alignment, ...), -1: error
static int cl11_op_mul_mat_id_gen(cl11_device * c, const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    if (c->mv_max_cols < 1) {
        return 0;
    }
    const cl11_tref s0 = cl11_ref(src0), s1 = cl11_ref(src1), d = cl11_ref(dst), ir = cl11_ref(ids);
    cl11_td tb = s1.td;
    if (src1->ne[1] == 1) {
        tb.nb[1] = 0; // src1 is shared by the slots
    }
    const int64_t M = src0->ne[1];
    const int64_t n_cols = ids->ne[0] * ids->ne[1];
    const int qt = cl11_qtype_index(src0->type);

    if (cl11_rp_type(src0->type)) {
        const int rk = cl11_rkind(src0->type);
        if (c->cfg.wg_mvr[rk] % 32 != 0) {
            return 0;
        }
        const cl11_td & ta = s0.td;
        const bool a16 = ta.off % 16 == 0 && ta.nb[2] % 16 == 0;
        const bool b16 = tb.off % 16 == 0 && tb.nb[1] % 16 == 0 && tb.nb[2] % 16 == 0;
        if (!a16 || !b16) {
            return 0;
        }
        if (rk >= CL11_RK_Q4_K && !cl11_ensure_kprogram(c)) {
            return -1;
        }
        if (c->kernels[K_MVRI + rk] == nullptr) {
            return 0;
        }
        const int64_t rows_wg = 32 * c->cfg.mvr_nrr[rk];
        const int64_t tiles = (M + rows_wg - 1) / rows_wg;
        const cl11_bcast bc = cl11_make_bcast(1, 1, tiles, 1);
        cl11_args a(c->kernels[K_MVRI + rk]);
        a.set(s0.mem).set(s0.td).set(s1.mem).set(tb).set(d.mem).set(d.td).set(bc).set(ir.mem).set(ir.td);
        return cl11_launch(c->queue, c->exec, a, c->cfg.wg_mvr[rk], (size_t) tiles * (size_t) n_cols) ? 1 : -1;
    }

    // F32 / F16 experts: the generic mat-vec, `tpr` threads per output
    const int64_t ne00 = src0->ne[0];
    const int64_t units = (ne00 + 3) / 4;
    int tpr = cl11_pow2_floor((size_t) std::max<int64_t>(units, 1));
    if (units <= 128) {
        tpr = std::min(tpr, std::max(4, cl11_pow2_floor((size_t) std::max<int64_t>(units / 4, 1))));
    }
    if (c->mv_tpr > 0) {
        tpr = std::min(tpr, cl11_pow2_floor((size_t) c->mv_tpr));
    }
    tpr = std::min(tpr, c->cfg.wg_mv);
    const size_t total = (size_t) (M * n_cols);
    const size_t rows_per_group = (size_t) (c->cfg.wg_mv / tpr);
    const cl11_bcast bc = cl11_make_bcast(1, 1, 1, 1);
    cl11_args a(c->kernels[K_MVI + qt]);
    a.set(s0.mem).set(s0.td).set(s1.mem).set(tb).set(d.mem).set(d.td).set(bc).set((int32_t) tpr).set((int32_t) cl11_log2_exact(tpr)).set(ir.mem).set(ir.td);
    return cl11_launch(c->queue, c->exec, a, c->cfg.wg_mv, cl11_div_up(total, rows_per_group)) ? 1 : -1;
}

static bool cl11_op_mul_mat_id(cl11_device * c, const ggml_tensor * dst) {
    // generation method: it reads one slice of the experts per (slot, token), so up to n_expert pairs it reads no more weights than the prompt method
    const ggml_tensor * ids = dst->src[2];
    const bool gen = c->mid_max_tokens >= 0 ? ids->ne[1] <= c->mid_max_tokens : ids->ne[0] * ids->ne[1] <= dst->src[0]->ne[2];
    if (gen) {
        const int r = cl11_op_mul_mat_id_gen(c, dst);
        if (r != 0) {
            return r > 0;
        }
    }
    return cl11_op_mul_mat_id_sync(c, dst);
}

// ---- several MUL_MATs of the same src1 in one launch (k_mvm_*, mul_mv_multi.cl) ----

// static properties only (types and shapes, no buffers: the graph is also looked at before it is allocated, see graph_optimize):
// a MUL_MAT of a repacked weight matrix with a single column of src1 and no batch. The launch-time checks (alignment, work-group limits) are in cl11_op_mul_mat_multi.
static bool cl11_mvm_candidate(const cl11_device * c, const ggml_tensor * node) {
    if (!c->multi_mv || c->mvr_const_x || c->no_fuse || node->op != GGML_OP_MUL_MAT || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        return false;
    }
    const ggml_tensor * a = node->src[0];
    const ggml_tensor * b = node->src[1];
    if (a == nullptr || b == nullptr || !cl11_rp_type(a->type) || a->op != GGML_OP_NONE || a->view_src != nullptr || b->type != GGML_TYPE_F32) {
        return false;
    }
    return c->mv_max_cols >= 1 && b->ne[1] == 1 && b->ne[2] == 1 && b->ne[3] == 1 && a->ne[2] == 1 && a->ne[3] == 1 &&
           node->type == GGML_TYPE_F32 && node->ne[2] == 1 && node->ne[3] == 1 && c->cfg.wg_mvr[cl11_rkind(a->type)] % 32 == 0;
}

// can `other` be computed in the same launch as `head`? (both are candidates)
static bool cl11_mvm_same_group(const ggml_tensor * head, const ggml_tensor * other) {
    return other->src[1] == head->src[1] && other->src[0]->type == head->src[0]->type && other->src[0]->ne[0] == head->src[0]->ne[0] &&
           other != head && other->src[0] != head->src[0];
}

// 1: computed, 0: not applicable (the caller computes the nodes one by one), -1: error
static int cl11_op_mul_mat_multi(cl11_device * c, ggml_tensor * const * nodes, int n) {
    GGML_ASSERT(n >= 2 && n <= 3);
    const ggml_tensor * a0 = nodes[0]->src[0];
    const int rk = cl11_rkind(a0->type);
    if (rk >= CL11_RK_Q4_K && !cl11_ensure_kprogram(c)) {
        return -1;
    }
    if (c->kernels[K_MVM + rk] == nullptr || c->mvm_wgs[rk] < (size_t) c->cfg.wg_mvr[rk]) {
        return 0;
    }
    const cl11_tref s1 = cl11_ref(nodes[0]->src[1]);
    if (s1.td.off % 16 != 0) {
        return 0;
    }
    cl11_tref sa[3], dd[3];
    cl11_bcast bc[3];
    const int64_t rows_wg = 32 * c->cfg.mvr_nrr[rk];
    size_t groups = 0;
    for (int j = 0; j < 3; ++j) {
        const ggml_tensor * node = nodes[j < n ? j : 0];
        sa[j] = cl11_ref(node->src[0]);
        dd[j] = cl11_ref(node);
        if (j < n && sa[j].td.off % 16 != 0) {
            return 0;
        }
        const int64_t tiles = j < n ? (node->src[0]->ne[1] + rows_wg - 1) / rows_wg : 0;
        bc[j] = cl11_make_bcast(1, 1, tiles, 1);
        groups += (size_t) tiles;
    }
    cl11_args a(c->kernels[K_MVM + rk]);
    a.set(sa[0].mem).set(sa[1].mem).set(sa[2].mem).set(dd[0].mem).set(dd[1].mem).set(dd[2].mem);
    a.set(sa[0].td).set(sa[1].td).set(sa[2].td).set(dd[0].td).set(dd[1].td).set(dd[2].td);
    a.set(bc[0]).set(bc[1]).set(bc[2]);
    a.set(s1.mem).set(s1.td);
    return cl11_launch(c->queue, c->exec, a, (size_t) c->cfg.wg_mvr[rk], groups) ? 1 : -1;
}

// MUL_MAT(CONT(transposed F16 view), x) with a single column x: the CONT is skipped, see k_mvt_f16
static bool cl11_op_mul_mat_t(cl11_device * c, const ggml_tensor * dst, const ggml_tensor * a) {
    const ggml_tensor * src1 = dst->src[1];

    const cl11_tref s0 = cl11_ref(a), s1 = cl11_ref(src1), d = cl11_ref(dst);
    if (c->cfg.mvt_groups) {
        // tiles of 32 outputs, WG_ROW / 32 slices of the positions per output
        const int64_t tiles = (a->ne[1] + 31) / 32;
        const cl11_bcast bcg = cl11_make_bcast(src1->ne[2] / a->ne[2], src1->ne[3] / a->ne[3], tiles, 1);
        cl11_args ag(c->kernels[K_MVT_G_F16]);
        ag.set(s0.mem).set(s0.td).set(s1.mem).set(s1.td).set(d.mem).set(d.td).set(bcg);
        return cl11_launch(c->queue, c->exec, ag, c->cfg.wg_row, (size_t) tiles * (size_t) (src1->ne[1] * src1->ne[2] * src1->ne[3]));
    }
    const cl11_bcast bc = cl11_make_bcast(src1->ne[2] / a->ne[2], 1, 1, 1);

    cl11_args args(c->kernels[K_MVT_F16]);
    args.set(s0.mem).set(s0.td).set(s1.mem).set(s1.td).set(d.mem).set(d.td).set(bc);
    return cl11_launch(c->queue, c->exec, args, c->cfg.wg_att, (size_t) a->ne[3]);
}

// can the CONT node i be skipped because the (only) MUL_MAT that uses it reads the transposed view directly?
static bool cl11_can_skip_cont(const cl11_device * c, const ggml_cgraph * cgraph, int i) {
    if (c->no_mvt || c->cfg.wg_row % 32 != 0) {
        return false;
    }
    const ggml_tensor * cont = cgraph->nodes[i];
    if (cont->op != GGML_OP_CONT || !ggml_node_has_n_uses(cgraph, i, 1)) {
        return false;
    }
    const ggml_tensor * a = cont->src[0];
    if (a == nullptr || a->type != GGML_TYPE_F16 || cont->type != GGML_TYPE_F16 || !ggml_are_same_shape(a, cont) ||
        a->nb[1] != ggml_type_size(GGML_TYPE_F16) || a->nb[0] <= a->nb[1] || !cl11_fits_int(a)) {
        return false;
    }
    // the consumer
    for (int j = i + 1; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * mm = cgraph->nodes[j];
        if (mm->src[1] == cont) {
            return false; // used as src1
        }
        if (mm->src[0] != cont) {
            continue;
        }
        const ggml_tensor * src1 = mm->src[1];
        return mm->op == GGML_OP_MUL_MAT && (mm->flags & GGML_TENSOR_FLAG_COMPUTE) != 0 &&
               src1->type == GGML_TYPE_F32 && src1->ne[1] == 1 && src1->nb[0] == sizeof(float) &&
               mm->type == GGML_TYPE_F32 && mm->nb[0] == sizeof(float) &&
               a->ne[0] == src1->ne[0] && src1->ne[2] % a->ne[2] == 0 &&
               (c->cfg.mvt_groups ? src1->ne[3] % a->ne[3] == 0
                                  : src1->ne[3] == a->ne[3] &&
                                    cl11_attn_ok(c, cl11_ref(a).td, cl11_ref(src1).td, a->ne[1], a->ne[2], src1->ne[2] / a->ne[2], a->ne[0], true));
    }
    return false;
}

static bool cl11_compute_node(cl11_device * c, const ggml_tensor * node) {
    switch (node->op) {
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:       return cl11_op_binary(c, node);
        case GGML_OP_UNARY:
        case GGML_OP_SCALE:
        case GGML_OP_CLAMP:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:      return cl11_op_unary(c, node);
        case GGML_OP_GLU:       return cl11_op_glu(c, node);
        case GGML_OP_RMS_NORM:  return cl11_op_rms_norm(c, node);
        case GGML_OP_SOFT_MAX:  return cl11_op_soft_max(c, node);
        case GGML_OP_ROPE:      return cl11_op_rope(c, node);
        case GGML_OP_GET_ROWS:  return cl11_op_get_rows(c, node);
        case GGML_OP_SET_ROWS:  return cl11_op_set_rows(c, node);
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT:      return cl11_op_cpy(c, node);
        case GGML_OP_MUL_MAT:   return cl11_op_mul_mat(c, node);
        case GGML_OP_MUL_MAT_ID: return cl11_op_mul_mat_id(c, node);
        case GGML_OP_ADD_ID:    return cl11_op_add_id(c, node);
        case GGML_OP_SSM_CONV:  return cl11_op_ssm_conv(c, node);
        case GGML_OP_CONCAT:    return cl11_op_concat(c, node);
        case GGML_OP_PAD:       return cl11_op_pad(c, node);
        case GGML_OP_SUM_ROWS:  return cl11_op_sum_rows(c, node);
        case GGML_OP_ARGSORT:
        case GGML_OP_TOP_K:     return cl11_op_argsort(c, node);
        default:
            GGML_LOG_ERROR("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
            return false;
    }
}

//
// backend
//

struct cl11_backend_context {
    cl11_device * dev;
    std::string   name;
};

static const char * ggml_backend_cl11_get_name(ggml_backend_t backend) {
    auto * bctx = (cl11_backend_context *) backend->context;
    return bctx->name.c_str();
}

static void ggml_backend_cl11_free(ggml_backend_t backend) {
    auto * bctx = (cl11_backend_context *) backend->context;
    if (bctx->dev->exec.profile) {
        std::lock_guard<std::mutex> lock(bctx->dev->mutex);
        if (bctx->dev->queue) {
            clFinish(bctx->dev->queue);
            cl11_prof_collect(bctx->dev->exec);
        }
        cl11_prof_print(bctx->dev->exec, bctx->dev->name.c_str());
    }
    delete bctx;
    delete backend;
}

static void ggml_backend_cl11_synchronize(ggml_backend_t backend) {
    auto * bctx = (cl11_backend_context *) backend->context;
    cl11_device * c = bctx->dev;
    std::lock_guard<std::mutex> lock(c->mutex);
    if (c->queue) {
        const cl_int err = clFinish(c->queue);
        if (err != CL_SUCCESS) {
            GGML_LOG_ERROR("%s: clFinish failed: %s\n", __func__, cl11_err_str(err));
        }
    }
}

static enum ggml_status ggml_backend_cl11_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto * bctx = (cl11_backend_context *) backend->context;
    cl11_device * c = bctx->dev;

    std::lock_guard<std::mutex> lock(c->mutex);
    if (!cl11_ensure_program(c)) {
        return GGML_STATUS_FAILED;
    }

    if (getenv("GGML_CL11_TRACE_GRAPH")) {
        static int traced = 0;
        if (traced++ == 1) { // the first decode graph
            for (int i = 0; i < cgraph->n_nodes && i < 80; ++i) {
                const ggml_tensor * n = cgraph->nodes[i];
                fprintf(stderr, "node %3d %-10s %-28s [%lld,%lld,%lld,%lld]%s src0=%s src1=%s\n", i, ggml_op_name(n->op), n->name, (long long) n->ne[0], (long long) n->ne[1], (long long) n->ne[2], (long long) n->ne[3],
                        (n->flags & GGML_TENSOR_FLAG_COMPUTE) ? "" : " (skip)", n->src[0] ? n->src[0]->name : "-", n->src[1] ? n->src[1]->name : "-");
            }
        }
    }
    for (auto & x : c->xprep) {
        x.valid = false;
    }
    c->mid_ids.valid = false;
    std::vector<const ggml_tensor *> skipped_conts;
    const size_t launches_before = c->exec.n_launches;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_tensor * node = cgraph->nodes[i];

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        if (ggml_is_empty(node) || ggml_op_is_empty(node->op)) {
            continue;
        }
        cl11_xprep_note_write(c, node);
        if (node->op == GGML_OP_ADD && i + 2 < cgraph->n_nodes) {
            cl11_xprep_note_write(c, cgraph->nodes[i + 1]);
            cl11_xprep_note_write(c, cgraph->nodes[i + 2]);
        } else if (node->op == GGML_OP_RMS_NORM && i + 1 < cgraph->n_nodes) {
            cl11_xprep_note_write(c, cgraph->nodes[i + 1]);
        }

        // CONT of a transposed view that only feeds a mat-vec: skipped, the MUL_MAT reads the view (k_mvt_f16)
        if (node->op == GGML_OP_CONT && cl11_can_skip_cont(c, cgraph, i)) {
            skipped_conts.push_back(node);
            continue;
        }
        if (node->op == GGML_OP_MUL_MAT && std::find(skipped_conts.begin(), skipped_conts.end(), node->src[0]) != skipped_conts.end()) {
            if (c->exec.profile) {
                c->exec.prof_key = cl11_prof_key(c->exec, c->cfg.mvt_groups ? "MUL_MAT f16 transposed (CONT skipped, many groups)" : "MUL_MAT f16 transposed (CONT skipped)");
                c->exec.prof_stats[c->exec.prof_key].count++;
            }
            if (!cl11_op_mul_mat_t(c, node, node->src[0]->src[0])) {
                GGML_LOG_ERROR("%s: failed to compute node %d (%s, %s)\n", __func__, i, ggml_op_desc(node), node->name);
                return GGML_STATUS_FAILED;
            }
            continue;
        }

        // MUL_MATs of the same src1 and the same type of weights that follow each other (the nodes are made adjacent by graph_optimize): one launch
        if (node->op == GGML_OP_MUL_MAT && cl11_mvm_candidate(c, node)) {
            ggml_tensor * group[3] = { node, nullptr, nullptr };
            int n_group = 1;
            while (n_group < 3 && i + n_group < cgraph->n_nodes) {
                ggml_tensor * next = cgraph->nodes[i + n_group];
                if (next->op != GGML_OP_MUL_MAT || !cl11_mvm_candidate(c, next) || !cl11_mvm_same_group(node, next)) {
                    break;
                }
                group[n_group++] = next;
            }
            if (n_group >= 2) {
                if (c->exec.profile) {
                    std::string key = std::string("MUL_MAT ") + ggml_type_name(node->src[0]->type) + " mv multi x" + std::to_string(n_group) + " (fused)";
                    c->exec.prof_key = cl11_prof_key(c->exec, key);
                    c->exec.prof_stats[c->exec.prof_key].count++;
                    for (int j = 0; j < n_group; ++j) {
                        c->exec.prof_stats[c->exec.prof_key].bytes += (double) ggml_nbytes(group[j]->src[0]);
                    }
                }
                const int r = cl11_op_mul_mat_multi(c, group, n_group);
                if (r < 0) {
                    GGML_LOG_ERROR("%s: failed to compute node %d (%s, %s) with %d fused mat-vec\n", __func__, i, ggml_op_desc(node), node->name, n_group);
                    return GGML_STATUS_FAILED;
                }
                if (r > 0) {
                    for (int j = 1; j < n_group; ++j) {
                        cl11_xprep_note_write(c, group[j]);
                    }
                    i += n_group - 1;
                    continue;
                }
                if (c->exec.profile) {
                    c->exec.prof_stats[c->exec.prof_key].count--; // not fused after all
                }
            }
        }

        // the SET_ROWS of the K cache and the one of the V cache (only views in between): one launch
        if (node->op == GGML_OP_SET_ROWS && c->fuse_set_rows) {
            int j = i + 1;
            while (j < cgraph->n_nodes && ((cgraph->nodes[j]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || ggml_is_empty(cgraph->nodes[j]) || ggml_op_is_empty(cgraph->nodes[j]->op))) {
                ++j;
            }
            if (j < cgraph->n_nodes && cl11_can_pair_set_rows(c, node, cgraph->nodes[j])) {
                if (c->exec.profile) {
                    c->exec.prof_key = cl11_prof_key(c->exec, std::string("SET_ROWS ") + ggml_type_name(node->src[0]->type) + "->" + ggml_type_name(node->type) + " x2 (fused)");
                    c->exec.prof_stats[c->exec.prof_key].count++;
                }
                if (!cl11_op_set_rows2(c, node, cgraph->nodes[j])) {
                    GGML_LOG_ERROR("%s: failed to compute node %d (%s, %s) with the next SET_ROWS\n", __func__, i, ggml_op_desc(node), node->name);
                    return GGML_STATUS_FAILED;
                }
                cl11_xprep_note_write(c, cgraph->nodes[j]);
                i = j;
                continue;
            }
        }

        // ADD + RMS_NORM + MUL (norm weights) in one kernel
        if (node->op == GGML_OP_ADD) {
            const ggml_tensor * w = nullptr;
            if (cl11_can_fuse_add_norm_mul(c, cgraph, i, &w)) {
                if (c->exec.profile) {
                    c->exec.prof_key = cl11_prof_key(c->exec, "ADD+RMS_NORM+MUL (fused)");
                    c->exec.prof_stats[c->exec.prof_key].count++;
                }
                if (!cl11_op_add_rms_norm_mul(c, node, cgraph->nodes[i + 1], w, cgraph->nodes[i + 2])) {
                    GGML_LOG_ERROR("%s: failed to compute node %d (%s, %s)\n", __func__, i, ggml_op_desc(node), node->name);
                    return GGML_STATUS_FAILED;
                }
                i += 2;
                continue;
            }
        }

        // RMS_NORM + MUL (norm weights) in one kernel
        if (node->op == GGML_OP_RMS_NORM) {
            if (const ggml_tensor * w = cl11_norm_mul_weights(c, cgraph, i)) {
                if (c->exec.profile) {
                    c->exec.prof_key = cl11_prof_key(c->exec, "RMS_NORM+MUL (fused)");
                    c->exec.prof_stats[c->exec.prof_key].count++;
                }
                if (!cl11_op_rms_norm_mul(c, node, w, cgraph->nodes[i + 1])) {
                    GGML_LOG_ERROR("%s: failed to compute node %d (%s, %s)\n", __func__, i, ggml_op_desc(node), node->name);
                    return GGML_STATUS_FAILED;
                }
                ++i;
                continue;
            }
        }

        if (c->exec.profile) {
            std::string key = ggml_op_desc(node);
            if (node->op == GGML_OP_MUL_MAT) {
                key += std::string(" ") + ggml_type_name(node->src[0]->type) + (node->src[1]->ne[1] <= c->mv_max_cols ? " mv" : " mm");
                key += " n=" + std::to_string(node->src[1]->ne[1] <= c->mv_max_cols ? 1 : node->src[1]->ne[1]);
            } else if (node->op == GGML_OP_GET_ROWS || node->op == GGML_OP_CPY || node->op == GGML_OP_SET_ROWS) {
                key += std::string(" ") + ggml_type_name(node->src[0]->type) + "->" + ggml_type_name(node->type);
            }
            if (getenv("GGML_CL11_PROFILE_SHAPES")) {
                const ggml_tensor * s0 = node->src[0];
                char buf[160];
                snprintf(buf, sizeof(buf), " src0[%lld,%lld,%lld,%lld] nb[%zu,%zu,%zu,%zu]", s0 ? (long long) s0->ne[0] : 0, s0 ? (long long) s0->ne[1] : 0,
                        s0 ? (long long) s0->ne[2] : 0, s0 ? (long long) s0->ne[3] : 0, s0 ? s0->nb[0] : 0, s0 ? s0->nb[1] : 0, s0 ? s0->nb[2] : 0, s0 ? s0->nb[3] : 0);
                key += buf;
            }
            c->exec.prof_key = cl11_prof_key(c->exec, key);
            c->exec.prof_stats[c->exec.prof_key].count++;
            if (node->op == GGML_OP_MUL_MAT) {
                c->exec.prof_stats[c->exec.prof_key].bytes += (double) ggml_nbytes(node->src[0]);
            }
        }

        if (!cl11_compute_node(c, node)) {
            GGML_LOG_ERROR("%s: failed to compute node %d (%s, %s)\n", __func__, i, ggml_op_desc(node), node->name);
            return GGML_STATUS_FAILED;
        }
    }

    if (c->exec.launch_log) {
        // the number of kernels of a graph does not depend on the GPU: the same count can be compared on any device
        int n_mul_mat = 0;
        int64_t n_cols = 0;
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            if (cgraph->nodes[i]->op == GGML_OP_MUL_MAT && (cgraph->nodes[i]->flags & GGML_TENSOR_FLAG_COMPUTE)) {
                ++n_mul_mat;
                n_cols = std::max<int64_t>(n_cols, cgraph->nodes[i]->src[1]->ne[1]);
            }
        }
        fprintf(stderr, "cl11 launches: nodes %d, mul_mat %d (max columns %lld), kernels %zu\n", cgraph->n_nodes, n_mul_mat, (long long) n_cols,
                c->exec.n_launches - launches_before);
    }

    // submit the commands, but do not wait for them
    clFlush(c->queue);
    if (c->exec.profile) {
        clFinish(c->queue);
        cl11_prof_collect(c->exec);
    }
    return GGML_STATUS_SUCCESS;
}

// Makes the MUL_MATs of the same src1 and the same type of weights adjacent (Q / K / V of an attention layer: the nodes of the RoPE / norm of Q lie
// between them) so that graph_compute can compute them in one launch. This runs before the graph is allocated (ggml_backend_sched): the buffers of
// the moved nodes are chosen for the new order, and a MUL_MAT whose inputs are a weight tensor and a tensor that the first node of the group reads too
// can be computed at any position after that tensor.
static void ggml_backend_cl11_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    GGML_UNUSED(params);
    auto * bctx = (cl11_backend_context *) backend->context;
    cl11_device * c = bctx->dev;

    std::lock_guard<std::mutex> lock(c->mutex);
    if (!cl11_ensure_program(c) || !c->multi_mv || c->no_fuse) {
        return;
    }
    const int n = cgraph->n_nodes;
    const int window = 48; // how far behind the first node the others are looked for
    int moved = 0;
    for (int i = 0; i < n; ++i) {
        ggml_tensor * head = cgraph->nodes[i];
        if (!cl11_mvm_candidate(c, head)) {
            continue;
        }
        int last = i;
        int members = 1;
        for (int j = i + 1; j < n && j < i + window && members < 3; ++j) {
            ggml_tensor * cand = cgraph->nodes[j];
            if (!cl11_mvm_candidate(c, cand) || !cl11_mvm_same_group(head, cand)) {
                continue;
            }
            if (j != last + 1) {
                for (int k = j; k > last + 1; --k) {
                    cgraph->nodes[k] = cgraph->nodes[k - 1];
                }
                cgraph->nodes[last + 1] = cand;
                ++moved;
            }
            ++last;
            ++members;
        }
        i = last;
    }
    if (moved > 0 && cl11_debug()) {
        GGML_LOG_INFO("%s: moved %d mat-vec nodes next to the nodes that read the same src1\n", __func__, moved);
    }
}

static const ggml_backend_i ggml_backend_cl11_i = {
    /* .get_name                = */ ggml_backend_cl11_get_name,
    /* .free                    = */ ggml_backend_cl11_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ ggml_backend_cl11_synchronize,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_cl11_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ ggml_backend_cl11_graph_optimize,
};

static ggml_guid_t ggml_backend_cl11_guid() {
    static ggml_guid guid = { 0x43, 0x4c, 0x31, 0x31, 0x6f, 0x70, 0x65, 0x6e, 0x63, 0x6c, 0x31, 0x2e, 0x31, 0x67, 0x67, 0x6d };
    return &guid;
}

//
// device
//

static cl11_device_context * cl11_dev_ctx(ggml_backend_dev_t dev) {
    return (cl11_device_context *) dev->context;
}

static const char * ggml_backend_cl11_device_get_name(ggml_backend_dev_t dev) {
    return cl11_dev_ctx(dev)->device->name.c_str();
}

static const char * ggml_backend_cl11_device_get_description(ggml_backend_dev_t dev) {
    return cl11_dev_ctx(dev)->device->info.name.c_str();
}

static void ggml_backend_cl11_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    const cl11_device * c = cl11_dev_ctx(dev)->device.get();
    // OpenCL 1.1 cannot report the free memory: assume that the system (display, ...) keeps some of it
    *total = c->info.global_mem;
    const size_t reserve = std::min<size_t>(128u * 1024 * 1024, *total / 4);
    *free = *total - reserve;
}

static enum ggml_backend_dev_type ggml_backend_cl11_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_cl11_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_cl11_device_get_name(dev);
    props->description = ggml_backend_cl11_device_get_description(dev);
    props->type        = ggml_backend_cl11_device_get_type(dev);
    props->device_id   = nullptr;
    ggml_backend_cl11_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ false,
        /* .events               = */ false,
        /* .mmap_support         = */ false,
    };
}

static ggml_backend_t ggml_backend_cl11_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    cl11_device * c = cl11_dev_ctx(dev)->device.get();

    {
        std::lock_guard<std::mutex> lock(c->mutex);
        if (!cl11_ensure_context(c)) {
            return nullptr;
        }
    }

    auto * bctx = new cl11_backend_context{ c, c->name };
    return new ggml_backend {
        /* .guid    = */ ggml_backend_cl11_guid(),
        /* .iface   = */ ggml_backend_cl11_i,
        /* .device  = */ dev,
        /* .context = */ bctx,
    };
}

static ggml_backend_buffer_type_t ggml_backend_cl11_device_get_buffer_type(ggml_backend_dev_t dev) {
    return &cl11_dev_ctx(dev)->buft;
}

static bool ggml_backend_cl11_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    const bool ok = cl11_supports_op_impl(cl11_dev_ctx(dev)->device.get(), op);
    // GGML_CL11_LOG_UNSUPPORTED=1: print every operation (once per name / shape) that is left to another backend
    if (!ok && op->op != GGML_OP_NONE && getenv("GGML_CL11_LOG_UNSUPPORTED") != nullptr) {
        static std::mutex mtx;
        static std::vector<std::string> seen;
        char buf[256];
        snprintf(buf, sizeof(buf), "%s %s [%lld,%lld,%lld,%lld] src0 %s", ggml_op_desc(op), ggml_type_name(op->type), (long long) op->ne[0], (long long) op->ne[1], (long long) op->ne[2],
                (long long) op->ne[3], op->src[0] ? ggml_type_name(op->src[0]->type) : "-");
        std::lock_guard<std::mutex> lock(mtx);
        if (std::find(seen.begin(), seen.end(), buf) == seen.end()) {
            seen.push_back(buf);
            fprintf(stderr, "cl11: not supported: %s (%s)\n", buf, op->name);
        }
    }
    return ok;
}

static bool ggml_backend_cl11_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    // only the buffers of the same device: a cl_mem cannot be used with another cl_context
    return buft->iface.get_name == ggml_backend_cl11_buft_get_name && buft->context == dev->context;
}

static const ggml_backend_device_i ggml_backend_cl11_device_i = {
    /* .get_name             = */ ggml_backend_cl11_device_get_name,
    /* .get_description      = */ ggml_backend_cl11_device_get_description,
    /* .get_memory           = */ ggml_backend_cl11_device_get_memory,
    /* .get_type             = */ ggml_backend_cl11_device_get_type,
    /* .get_props            = */ ggml_backend_cl11_device_get_props,
    /* .init_backend         = */ ggml_backend_cl11_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_cl11_device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_cl11_device_supports_op,
    /* .supports_buft        = */ ggml_backend_cl11_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

//
// registry
//

static const char * ggml_backend_cl11_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_CL11_NAME;
}

static size_t ggml_backend_cl11_reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return g_cl11_devices.size();
}

static ggml_backend_dev_t ggml_backend_cl11_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index < g_cl11_devices.size());
    GGML_UNUSED(reg);
    return &g_cl11_devices[index]->dev;
}

static const ggml_backend_reg_i ggml_backend_cl11_reg_i = {
    /* .get_name         = */ ggml_backend_cl11_reg_get_name,
    /* .get_device_count = */ ggml_backend_cl11_reg_get_device_count,
    /* .get_device       = */ ggml_backend_cl11_reg_get_device,
    /* .get_proc_address = */ nullptr,
};

ggml_backend_reg_t ggml_backend_cl11_reg(void) {
    static std::mutex mutex;
    static ggml_backend_reg reg;
    static bool initialized = false;

    std::lock_guard<std::mutex> lock(mutex);
    if (initialized) {
        return &reg;
    }
    initialized = true;

    reg = ggml_backend_reg {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_cl11_reg_i,
        /* .context     = */ nullptr,
    };

    cl11_probe_devices();

    for (auto & c : g_cl11_devices) {
        c->buft = ggml_backend_buffer_type {
            /* .iface   = */ ggml_backend_cl11_buft_i,
            /* .device  = */ &c->dev,
            /* .context = */ c.get(),
        };
        c->dev = ggml_backend_device {
            /* .iface   = */ ggml_backend_cl11_device_i,
            /* .reg     = */ &reg,
            /* .context = */ c.get(),
        };
    }

    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cl11_reg)

//
// public API
//

int ggml_backend_cl11_get_device_count(void) {
    return (int) ggml_backend_reg_dev_count(ggml_backend_cl11_reg());
}

void ggml_backend_cl11_get_device_description(int device, char * description, size_t description_size) {
    ggml_backend_reg_t reg = ggml_backend_cl11_reg();
    if (device < 0 || (size_t) device >= ggml_backend_reg_dev_count(reg) || description_size == 0) {
        if (description_size > 0) {
            description[0] = '\0';
        }
        return;
    }
    snprintf(description, description_size, "%s", ggml_backend_dev_description(ggml_backend_reg_dev_get(reg, (size_t) device)));
}

ggml_backend_t ggml_backend_cl11_init(int device) {
    ggml_backend_reg_t reg = ggml_backend_cl11_reg();
    if (device < 0 || (size_t) device >= ggml_backend_reg_dev_count(reg)) {
        GGML_LOG_ERROR("%s: invalid device %d\n", __func__, device);
        return nullptr;
    }
    return ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, (size_t) device), nullptr);
}

bool ggml_backend_is_cl11(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_cl11_guid());
}

ggml_backend_buffer_type_t ggml_backend_cl11_buffer_type(int device) {
    ggml_backend_reg_t reg = ggml_backend_cl11_reg();
    if (device < 0 || (size_t) device >= ggml_backend_reg_dev_count(reg)) {
        return nullptr;
    }
    return ggml_backend_dev_buffer_type(ggml_backend_reg_dev_get(reg, (size_t) device));
}

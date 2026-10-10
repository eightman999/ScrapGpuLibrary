// MUL_MAT_ID of the CL11 backend (ops.h): dst[:, i, t] = s0[:, :, ids[i, t]] * s1[:, i % ne11, t].
// The prompt method reads the ids back (a synchronization point, see docs/API_CONTRACT.md), sorts the
// (slot, token) pairs by expert and runs one product per expert on scratch columns; the generation
// method reads the expert numbers on the GPU, one launch for all the selected experts.

#include "ops.h"

#include "executor.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// Prompt processing (several tokens): the ids are read back (the queue is finished), the (slot, token) pairs are sorted by expert, the columns of
// src1 of all the pairs are gathered into one scratch buffer in that order, every expert that is used multiplies its range of columns (the normal
// MUL_MAT kernels, one mat-mat or mat-vec per expert) and the result columns are scattered into dst.
static bool cl11_op_mul_mat_id_sync(cl11_device * c, const cl11_tensor & vd, const cl11_tensor & v0, const cl11_tensor & v1, const cl11_tensor & vi) {
    const int64_t K = v0.d.ne[0], M = v0.d.ne[1], n_exp = v0.d.ne[2];
    const int64_t n_used = vi.d.ne[0], n_tok = vi.d.ne[1];
    const int64_t P = n_used * n_tok;
    // the scratch products below must not wrap (the same bound as cl11_supports_mul_mat_id,
    // defensively — audit M-4)
    if (P < 0 || sgt::mul_sat_u64((uint64_t) P,
                     sgt::mul_sat_u64((uint64_t) std::max<int64_t>(std::max(K, M), 0),
                                      (uint64_t) sizeof(float))) > cl11_max_buffer_size(c->info)) {
        fprintf(stderr, "%s: the pair scratch does not fit the buffer limit\n", __func__);
        return false;
    }

    const cl11_tref s1 = cl11_tref_of(v1), d = cl11_tref_of(vd), ir = cl11_tref_of(vi);

    auto & cache = c->mid_ids;
    const size_t ids_bytes = (size_t) ((n_tok - 1) * vi.d.nb[1] + n_used * sizeof(int32_t));
    cl_int err = CL_SUCCESS;
    if (!(cache.valid && cache.mem == ir.mem && cache.off == (size_t) ir.td.off && cache.raw.size() == ids_bytes)) {
        cache.raw.resize(ids_bytes);
        err = clEnqueueReadBuffer(c->queue, ir.mem, CL_TRUE, (size_t) ir.td.off, ids_bytes, cache.raw.data(), 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            cache.valid = false;
            fprintf(stderr, "%s: reading the ids failed: %s\n", __func__, cl11_err_str(err));
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
            memcpy(&e, raw.data() + t * vi.d.nb[1] + i * sizeof(int32_t), sizeof(e));
            if (e < 0 || e >= n_exp) {
                fprintf(stderr, "%s: expert %d out of range (0..%lld)\n", __func__, e, (long long) n_exp - 1);
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
                memcpy(&e, raw.data() + t * vi.d.nb[1] + i * sizeof(int32_t), sizeof(e));
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
        fprintf(stderr, "%s: writing the pairs failed: %s\n", __func__, cl11_err_str(err));
        return false;
    }

    {
        cl11_args a(c->kernels[K_MID_GATHER]);
        a.set(s1.mem).set(s1.td).set(c->mid_gather.buf).set(c->mid_pairs.buf).set((int32_t) K).set((int32_t) v1.d.ne[1]);
        if (!cl11_launch(c->queue, c->exec, a, (size_t) c->cfg.wg_row, (size_t) P)) {
            return false;
        }
    }
    for (int64_t e = 0; e < n_exp; ++e) {
        const int64_t first = start[(size_t) e], cnt = start[(size_t) e + 1] - first;
        if (cnt == 0) {
            continue;
        }
        // (the repacked layout only depends on K, M and the type: the strides of the descriptor are those of the tensor)
        cl11_tensor xs0 = v0;
        xs0.d = cl11_matrix_desc(K, M, v0.d.nb[0], v0.d.nb[1], v0.d.offset + (size_t) e * v0.d.nb[2]);
        xs0.d.nb[2]  = v0.d.nb[2];
        xs0.d.nb[3]  = v0.d.nb[3];
        xs0.d.dtype  = v0.d.dtype;
        xs0.d.layout = v0.d.layout;
        cl11_tensor xs1 = {}, xd = {};
        xs1.mem     = c->mid_gather.buf;
        xs1.d       = cl11_matrix_desc(K, cnt, sizeof(float), K * sizeof(float), (size_t) first * K * sizeof(float));
        xs1.d.dtype = SCRAPGPU_DTYPE_F32;
        xd.mem      = c->mid_out.buf;
        xd.d        = cl11_matrix_desc(M, cnt, sizeof(float), M * sizeof(float), (size_t) first * M * sizeof(float));
        xd.d.dtype  = SCRAPGPU_DTYPE_F32;
        if (!cl11_mul_mat_core(c, xs0, xs1, xd, true)) {
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
static int cl11_op_mul_mat_id_gen(cl11_device * c, const cl11_tensor & vd, const cl11_tensor & v0, const cl11_tensor & v1, const cl11_tensor & vi) {
    if (c->mv_max_cols < 1) {
        return 0;
    }
    const cl11_tref s0 = cl11_tref_of(v0), s1 = cl11_tref_of(v1), d = cl11_tref_of(vd), ir = cl11_tref_of(vi);
    cl11_td tb = s1.td;
    if (v1.d.ne[1] == 1) {
        tb.nb[1] = 0; // src1 is shared by the slots
    }
    const int64_t M = v0.d.ne[1];
    const int64_t n_cols = vi.d.ne[0] * vi.d.ne[1];
    const int qt = cl11_qt_of(v0.d.dtype);

    if (v0.repacked) {
        const int rk = cl11_rk_of(v0.d.dtype);
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
    const int64_t ne00 = v0.d.ne[0];
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

bool cl11_op_mul_mat_id(cl11_device * c, const cl11_tensor & dst, const cl11_tensor & s0, const cl11_tensor & s1, const cl11_tensor & ids) {
    if (ids.d.ne[0] <= 0 || ids.d.ne[1] <= 0) {
        return true; // an empty id set writes nothing (a 0-size launch would be a driver error)
    }
    // generation method: it reads one slice of the experts per (slot, token), so up to n_expert pairs it reads no more weights than the prompt method
    const bool gen = c->mid_max_tokens >= 0 ? ids.d.ne[1] <= c->mid_max_tokens : ids.d.ne[0] * ids.d.ne[1] <= s0.d.ne[2];
    if (gen) {
        const int r = cl11_op_mul_mat_id_gen(c, dst, s0, s1, ids);
        if (r != 0) {
            return r > 0;
        }
    }
    return cl11_op_mul_mat_id_sync(c, dst, s0, s1, ids);
}

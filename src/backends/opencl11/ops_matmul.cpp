// MUL_MAT and the fused mat-vec forms of the CL11 backend (ops.h): the kernel selection
// (repacked mat-vec / K cache / staged / generic mat-vec / parameterized and default mat-mat), the
// constant-memory and prepared src1 handling (cl11_const_sub, k_xprep scratch) and the scratch /
// readback cache invalidation on writes.

#include "ops.h"

#include "buffer.h"
#include "executor.h"

#include <algorithm>
#include <cstdint>

// conditions of the attention kernels of attn.cl (K cache: k_mvq_f16, transposed V cache: k_mvt_f16): the heads of a position are
// contiguous in the cache (row = ne02 heads of `hd` halves), 16-byte aligned rows, a work-item per 16 bytes of a row, at most 8 query
// heads per head of the cache, src1 (q / x) with 16-byte aligned float4 loads
bool cl11_attn_ok(const cl11_device * c, const cl11_td & ta, const cl11_td & tb, int64_t hd, int64_t ne02, int64_t r2, int64_t k_len, bool v) {
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

// a write into a column the scratch buffers were made from makes them stale
void cl11_xprep_note_write(cl11_device * c, cl_mem mem, size_t offset, size_t bytes) {
    if (c->mid_ids.valid && mem == c->mid_ids.mem) {
        const size_t b = offset, e = b + bytes;
        if (b < c->mid_ids.off + c->mid_ids.raw.size() && c->mid_ids.off < e) {
            c->mid_ids.valid = false;
        }
    }
    for (auto & x : c->xprep) {
        if (x.valid && mem == x.src) {
            const size_t b = offset, e = b + bytes;
            if (b < x.off + x.bytes && x.off < e) {
                x.valid = false;
            }
        }
    }
}

// The product of the tensors (dtype `s0.d.dtype` of src0) described by s0, s1 and d. `moe`: the operands are slices / scratch columns of
// MUL_MAT_ID (no K cache kernel, and no constant-memory / prepared src1: the scratch columns change from one product to the next).
bool cl11_mul_mat_core(cl11_device * c, const cl11_tensor & v0, const cl11_tensor & v1, const cl11_tensor & vd, bool moe) {
    const scrapgpu_dtype type = v0.d.dtype;
    const int qt = cl11_qt_of(type);

    const cl11_tref s0 = cl11_tref_of(v0), s1 = cl11_tref_of(v1), d = cl11_tref_of(vd);

    const int64_t ne00 = s0.td.ne[0];
    const int64_t ne01 = s0.td.ne[1];
    const int64_t ne02 = s0.td.ne[2];
    const int64_t ne03 = s0.td.ne[3];
    const int64_t ne11 = s1.td.ne[1];
    const int64_t ne12 = s1.td.ne[2];
    const int64_t ne13 = s1.td.ne[3];
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    const bool rp = v0.repacked; // repacked weights (R32 layout): the mat-vec kernel is k_mvr_*
    const int  rk = rp ? cl11_rk_of(type) : 0;
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
    if (!moe && type == SCRAPGPU_DTYPE_F16 && ne11 == 1 && ne13 == ne03 && cl11_attn_ok(c, s0.td, s1.td, ne00, ne02, r2, ne00, false)) {
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
            case SCRAPGPU_DTYPE_F16:  staged = staged && a4 && ne00 % 8 == 0;   break;
            default:                  staged = staged && a16 && ne00 % 8 == 0;  break;
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

bool cl11_op_mul_mat(cl11_device * c, const cl11_tensor & dst, const cl11_tensor & s0, const cl11_tensor & s1) {
    return cl11_mul_mat_core(c, s0, s1, dst, false);
}

// ---- several MUL_MATs of the same src1 in one launch (k_mvm_*, mul_mv_multi.cl) ----

// 1: computed, 0: not applicable (the caller computes the nodes one by one), -1: error.
// The caller guarantees 2 <= n <= 3 (adjacent nodes of the same src1 / weight type).
int cl11_op_mul_mat_multi(cl11_device * c, const cl11_tensor dsts[3], const cl11_tensor srcs0[3], const cl11_tensor & s1v, int n) {
    if (n < 2 || n > 3) {
        return -1;
    }
    const int rk = cl11_rk_of(srcs0[0].d.dtype);
    if (rk >= CL11_RK_Q4_K && !cl11_ensure_kprogram(c)) {
        return -1;
    }
    if (c->kernels[K_MVM + rk] == nullptr || c->mvm_wgs[rk] < (size_t) c->cfg.wg_mvr[rk]) {
        return 0;
    }
    const cl11_tref s1 = cl11_tref_of(s1v);
    if (s1.td.off % 16 != 0) {
        return 0;
    }
    cl11_tref sa[3], dd[3];
    cl11_bcast bc[3];
    const int64_t rows_wg = 32 * c->cfg.mvr_nrr[rk];
    size_t groups = 0;
    for (int j = 0; j < 3; ++j) {
        const int k = j < n ? j : 0;
        sa[j] = cl11_tref_of(srcs0[k]);
        dd[j] = cl11_tref_of(dsts[k]);
        if (j < n && sa[j].td.off % 16 != 0) {
            return 0;
        }
        const int64_t tiles = j < n ? (srcs0[j].d.ne[1] + rows_wg - 1) / rows_wg : 0;
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
bool cl11_op_mul_mat_t(cl11_device * c, const cl11_tensor & dst, const cl11_tensor & a, const cl11_tensor & s1v) {
    const cl11_tref s0 = cl11_tref_of(a), s1 = cl11_tref_of(s1v), d = cl11_tref_of(dst);
    if (c->cfg.mvt_groups) {
        // tiles of 32 outputs, WG_ROW / 32 slices of the positions per output
        const int64_t tiles = (a.d.ne[1] + 31) / 32;
        const cl11_bcast bcg = cl11_make_bcast(s1v.d.ne[2] / a.d.ne[2], s1v.d.ne[3] / a.d.ne[3], tiles, 1);
        cl11_args ag(c->kernels[K_MVT_G_F16]);
        ag.set(s0.mem).set(s0.td).set(s1.mem).set(s1.td).set(d.mem).set(d.td).set(bcg);
        return cl11_launch(c->queue, c->exec, ag, c->cfg.wg_row, (size_t) tiles * (size_t) (s1v.d.ne[1] * s1v.d.ne[2] * s1v.d.ne[3]));
    }
    const cl11_bcast bc = cl11_make_bcast(s1v.d.ne[2] / a.d.ne[2], 1, 1, 1);

    cl11_args args(c->kernels[K_MVT_F16]);
    args.set(s0.mem).set(s0.td).set(s1.mem).set(s1.td).set(d.mem).set(d.td).set(bc);
    return cl11_launch(c->queue, c->exec, args, c->cfg.wg_att, (size_t) a.d.ne[3]);
}

// The elementwise and row ops of the CL11 backend (see ops.h): each enqueues one kernel on the
// in-order queue of the device. No synchronization, no allocation, no ggml.

#include "ops.h"

#include "executor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

bool cl11_op_binary(cl11_device * c, const cl11_tensor & dst, const cl11_tensor & s0, const cl11_tensor & s1, cl11_binary_kind kind) {
    cl11_kernel_id kid;
    switch (kind) {
        case CL11_BINARY_ADD: kid = K_ADD; break;
        case CL11_BINARY_SUB: kid = K_SUB; break;
        case CL11_BINARY_MUL: kid = K_MUL; break;
        default:              kid = K_DIV; break;
    }
    const cl11_tref sr0 = cl11_tref_of(s0), sr1 = cl11_tref_of(s1), d = cl11_tref_of(dst);
    const int32_t n = (int32_t) sgt::nelements(dst.d);

    cl11_args a(c->kernels[kid]);
    a.set(sr0.mem).set(sr0.td).set(sr1.mem).set(sr1.td).set(d.mem).set(d.td).set(n);
    return cl11_launch(c->queue, c->exec, a, c->cfg.wg_ew, cl11_div_up(n, c->cfg.wg_ew));
}

// UNARY, SCALE, CLAMP, SQR, SQRT: op codes of k_unary
bool cl11_op_unary(cl11_device * c, const cl11_tensor & dst, const cl11_tensor & s0, int32_t code, float p0, float p1) {
    const cl11_tref s = cl11_tref_of(s0), d = cl11_tref_of(dst);
    const int32_t n = (int32_t) sgt::nelements(dst.d);

    cl11_args a(c->kernels[K_UNARY]);
    a.set(s.mem).set(s.td).set(d.mem).set(d.td).set(n).set(code).set(p0).set(p1);
    return cl11_launch(c->queue, c->exec, a, c->cfg.wg_ew, cl11_div_up(n, c->cfg.wg_ew));
}

bool cl11_op_glu(cl11_device * c, const cl11_tensor & dst, const cl11_tensor & s0, const cl11_tensor * s1, const scrapgpu_glu_params & p) {
    const int32_t code    = p.kind == SCRAPGPU_GLU_SWIGLU ? 0 : 1;
    const int32_t split   = s1 != nullptr ? 1 : 0;
    const int32_t swapped = p.swapped;
    const int32_t nc      = (int32_t) dst.d.ne[0];

    const cl11_tref sr0 = cl11_tref_of(s0), sr1 = cl11_tref_of(s1 ? *s1 : s0), d = cl11_tref_of(dst);
    const int32_t n = (int32_t) sgt::nelements(dst.d);

    cl11_args a(c->kernels[K_GLU]);
    a.set(sr0.mem).set(sr0.td).set(sr1.mem).set(sr1.td).set(d.mem).set(d.td).set(n).set(code).set(split).set(swapped).set(nc);
    return cl11_launch(c->queue, c->exec, a, c->cfg.wg_ew, cl11_div_up(n, c->cfg.wg_ew));
}

bool cl11_op_rms_norm(cl11_device * c, const cl11_tensor & dst, const cl11_tensor & s0, const scrapgpu_rms_norm_params & p) {
    const cl11_tref s = cl11_tref_of(s0), d = cl11_tref_of(dst);
    cl11_args a(c->kernels[K_RMS_NORM]);
    a.set(s.mem).set(s.td).set(d.mem).set(d.td).set(p.eps);
    return cl11_launch(c->queue, c->exec, a, c->cfg.wg_row, (size_t) (s0.d.ne[1] * s0.d.ne[2] * s0.d.ne[3]));
}

// RMS_NORM (F32) followed by a MUL with a weight vector `w` (the fusion itself is decided by the caller)
bool cl11_op_rms_norm_mul(cl11_device * c, const cl11_tensor & dst, const cl11_tensor & s0, const cl11_tensor & w, const scrapgpu_rms_norm_params & p) {
    const cl11_tref s = cl11_tref_of(s0), wr = cl11_tref_of(w), d = cl11_tref_of(dst);
    cl11_args a(c->kernels[K_RMS_NORM_MUL]);
    a.set(s.mem).set(s.td).set(wr.mem).set(wr.td).set(d.mem).set(d.td).set(p.eps);
    return cl11_launch(c->queue, c->exec, a, c->cfg.wg_row, (size_t) (s0.d.ne[1] * s0.d.ne[2] * s0.d.ne[3]));
}

// ADD (F32, one row, same shapes, no broadcast) + RMS_NORM + MUL in one kernel
bool cl11_op_add_rms_norm_mul(cl11_device * c, const cl11_tensor & dst, const cl11_tensor & a, const cl11_tensor & b,
                              const cl11_tensor & w, const cl11_tensor & add, const scrapgpu_rms_norm_params & p) {
    const cl11_tref ar = cl11_tref_of(a), br = cl11_tref_of(b), wr = cl11_tref_of(w), sr = cl11_tref_of(add), d = cl11_tref_of(dst);
    cl11_args args(c->kernels[K_ADD_RMS_NORM_MUL]);
    args.set(ar.mem).set(ar.td).set(br.mem).set(br.td).set(wr.mem).set(wr.td).set(sr.mem).set(sr.td).set(d.mem).set(d.td).set(p.eps);
    return cl11_launch(c->queue, c->exec, args, c->cfg.wg_row, 1);
}

bool cl11_op_soft_max(cl11_device * c, const cl11_tensor & dst, const cl11_tensor & s0, const cl11_tensor * mask, float scale) {
    const cl11_tref s = cl11_tref_of(s0), d = cl11_tref_of(dst);
    const cl11_tref m = mask ? cl11_tref_of(*mask) : s;
    cl11_tref mm = m;
    if (!mask) {
        for (int i = 0; i < 4; ++i) {
            mm.td.ne[i] = 1;
            mm.td.nb[i] = 0;
            sgt::fastdiv_init(1, mm.td.fm[i], mm.td.fl[i]);
        }
        mm.td.off = 0;
    }
    const int32_t has_mask = mask ? 1 : 0;
    const int32_t mask_f16 = (mask && mask->d.dtype == SCRAPGPU_DTYPE_F16) ? 1 : 0;

    cl11_args a(c->kernels[K_SOFT_MAX]);
    a.set(s.mem).set(s.td).set(mm.mem).set(mm.td).set(has_mask).set(mask_f16).set(d.mem).set(d.td).set(scale);
    return cl11_launch(c->queue, c->exec, a, c->cfg.wg_row, (size_t) (s0.d.ne[1] * s0.d.ne[2] * s0.d.ne[3]));
}

bool cl11_op_rope(cl11_device * c, const cl11_tensor & dst, const cl11_tensor & s0, const cl11_tensor & pos, const cl11_tensor * ff,
                  const cl11_rope_params & p) {
    const float theta_scale = powf(p.freq_base, -2.0f / p.n_dims);

    const cl11_tref s = cl11_tref_of(s0), pr = cl11_tref_of(pos), d = cl11_tref_of(dst);
    const cl11_tref f = ff ? cl11_tref_of(*ff) : pr;
    const int32_t has_ff = ff ? 1 : 0;
    const int32_t neox   = p.mode == SCRAPGPU_ROPE_MODE_NEOX ? 1 : 0;
    const int32_t nd = p.n_dims, no = p.n_offs;

    cl11_args a(c->kernels[s0.d.dtype == SCRAPGPU_DTYPE_F16 ? K_ROPE_F16 : K_ROPE_F32]);
    a.set(s.mem).set(s.td).set(pr.mem).set(pr.td).set(f.mem).set(f.td).set(has_ff).set(d.mem).set(d.td)
     .set(nd).set(neox).set(no)
     .set(theta_scale).set(p.freq_scale).set(p.ext_factor).set(p.attn_factor).set(p.corr_dims[0]).set(p.corr_dims[1]);

    // one work-group per row
    return cl11_launch(c->queue, c->exec, a, c->cfg.wg_row, (size_t) (dst.d.ne[1] * dst.d.ne[2] * dst.d.ne[3]));
}

bool cl11_op_get_rows(cl11_device * c, const cl11_tensor & dst, const cl11_tensor & s0, const cl11_tensor & ids) {
    const cl11_tref sr0 = cl11_tref_of(s0), sr1 = cl11_tref_of(ids), d = cl11_tref_of(dst);
    cl11_args a(c->kernels[K_GET_ROWS + cl11_qt_of(s0.d.dtype)]);
    a.set(sr0.mem).set(sr0.td).set(sr1.mem).set(sr1.td).set(d.mem).set(d.td);
    return cl11_launch(c->queue, c->exec, a, c->cfg.wg_row, (size_t) (ids.d.ne[0] * ids.d.ne[1] * ids.d.ne[2]));
}

bool cl11_op_set_rows(cl11_device * c, const cl11_tensor & dst, const cl11_tensor & s0, const cl11_tensor & ids) {
    const int kid = K_SET_ROWS + (s0.d.dtype == SCRAPGPU_DTYPE_F16 ? 2 : 0) + (dst.d.dtype == SCRAPGPU_DTYPE_F16 ? 1 : 0);
    const cl11_tref sr0 = cl11_tref_of(s0), sr1 = cl11_tref_of(ids), d = cl11_tref_of(dst);
    const int32_t n = (int32_t) sgt::nelements(s0.d);
    const int32_t idx64 = ids.d.dtype == SCRAPGPU_DTYPE_I64 ? 1 : 0;

    cl11_args a(c->kernels[kid]);
    a.set(sr0.mem).set(sr0.td).set(sr1.mem).set(sr1.td).set(d.mem).set(d.td).set(n).set(idx64);
    return cl11_launch(c->queue, c->exec, a, c->cfg.wg_ew, cl11_div_up(n, c->cfg.wg_ew));
}

// two SET_ROWS with the same types in one launch (k_set_rows2_*)
bool cl11_op_set_rows2(cl11_device * c, const cl11_tensor a[3], const cl11_tensor b[3]) {
    const int kid = K_SET_ROWS2 + (a[0].d.dtype == SCRAPGPU_DTYPE_F16 ? 2 : 0) + (a[2].d.dtype == SCRAPGPU_DTYPE_F16 ? 1 : 0);
    const cl11_tref a0 = cl11_tref_of(a[0]), a1 = cl11_tref_of(a[1]), ad = cl11_tref_of(a[2]);
    const cl11_tref b0 = cl11_tref_of(b[0]), b1 = cl11_tref_of(b[1]), bd = cl11_tref_of(b[2]);
    const int32_t na = (int32_t) sgt::nelements(a[0].d);
    const int32_t nb = (int32_t) sgt::nelements(b[0].d);
    const int32_t groups_a = (int32_t) cl11_div_up(na, c->cfg.wg_ew);

    cl11_args args(c->kernels[kid]);
    args.set(a0.mem).set(a0.td).set(a1.mem).set(a1.td).set(ad.mem).set(ad.td).set(na).set((int32_t) (a[1].d.dtype == SCRAPGPU_DTYPE_I64 ? 1 : 0));
    args.set(b0.mem).set(b0.td).set(b1.mem).set(b1.td).set(bd.mem).set(bd.td).set(nb).set((int32_t) (b[1].d.dtype == SCRAPGPU_DTYPE_I64 ? 1 : 0));
    args.set(groups_a);
    return cl11_launch(c->queue, c->exec, args, c->cfg.wg_ew, groups_a + cl11_div_up(nb, c->cfg.wg_ew));
}

bool cl11_op_cpy(cl11_device * c, const cl11_tensor & dst, const cl11_tensor & s0) {
    const cl11_tref s = cl11_tref_of(s0), d = cl11_tref_of(dst);

    // transposing copy of a tensor whose dim 1 is contiguous into a contiguous tensor of the same type and shape
    if (s0.d.dtype == dst.d.dtype && sgt::same_shape(s0.d, dst.d) && sgt::is_contiguous(dst.d) &&
        s0.d.nb[1] == sgt::dtype_type_size(s0.d.dtype) && sgt::is_transposed(s0.d) && s0.d.ne[0] > 1 && s0.d.ne[1] > 1 &&
        c->cfg.gt >= 8 && !c->no_cpy_t) {
        const int64_t tiles0 = (dst.d.ne[0] + 31) / 32;
        const int64_t tiles1 = (dst.d.ne[1] + 31) / 32;
        const cl11_bcast bc = cl11_make_bcast(1, 1, tiles0, tiles1);
        cl11_args a(c->kernels[s0.d.dtype == SCRAPGPU_DTYPE_F16 ? K_CPY_T_F16 : K_CPY_T_F32]);
        a.set(s.mem).set(s.td).set(d.mem).set(d.td).set(bc);
        return cl11_launch(c->queue, c->exec, a, (size_t) c->cfg.gt * c->cfg.gt, (size_t) (tiles0 * tiles1 * dst.d.ne[2] * dst.d.ne[3]));
    }

    const int kid = K_CPY_F32_F32 + (s0.d.dtype == SCRAPGPU_DTYPE_F16 ? 2 : 0) + (dst.d.dtype == SCRAPGPU_DTYPE_F16 ? 1 : 0);
    const int32_t n = (int32_t) sgt::nelements(s0.d);

    cl11_args a(c->kernels[kid]);
    a.set(s.mem).set(s.td).set(d.mem).set(d.td).set(n);
    return cl11_launch(c->queue, c->exec, a, c->cfg.wg_ew, cl11_div_up(n, c->cfg.wg_ew));
}

// ---- the operations of the hybrid / MoE graphs (LFM2, LFM2-MoE): moe_ops.cl ----

bool cl11_op_add_id(cl11_device * c, const cl11_tensor & dst, const cl11_tensor & s0, const cl11_tensor & s1, const cl11_tensor & ids) {
    const cl11_tref sr0 = cl11_tref_of(s0), sr1 = cl11_tref_of(s1), sr2 = cl11_tref_of(ids), d = cl11_tref_of(dst);
    const int32_t n = (int32_t) sgt::nelements(dst.d);
    cl11_args a(c->kernels[K_ADD_ID]);
    a.set(sr0.mem).set(sr0.td).set(sr1.mem).set(sr1.td).set(sr2.mem).set(sr2.td).set(d.mem).set(d.td).set(n);
    return cl11_launch(c->queue, c->exec, a, c->cfg.wg_ew, cl11_div_up(n, c->cfg.wg_ew));
}

bool cl11_op_ssm_conv(cl11_device * c, const cl11_tensor & dst, const cl11_tensor & s0, const cl11_tensor & s1) {
    const cl11_tref sr0 = cl11_tref_of(s0), sr1 = cl11_tref_of(s1), d = cl11_tref_of(dst);
    const int32_t n = (int32_t) sgt::nelements(dst.d);
    cl11_args a(c->kernels[K_SSM_CONV]);
    a.set(sr0.mem).set(sr0.td).set(sr1.mem).set(sr1.td).set(d.mem).set(d.td).set(n);
    return cl11_launch(c->queue, c->exec, a, c->cfg.wg_ew, cl11_div_up(n, c->cfg.wg_ew));
}

bool cl11_op_concat(cl11_device * c, const cl11_tensor & dst, const cl11_tensor & s0, const cl11_tensor & s1, const scrapgpu_concat_params & p) {
    const cl11_tref sr0 = cl11_tref_of(s0), sr1 = cl11_tref_of(s1), d = cl11_tref_of(dst);
    const int32_t n   = (int32_t) sgt::nelements(dst.d);
    const int32_t dim = (int32_t) p.dim;
    const int32_t o   = (int32_t) s0.d.ne[dim];
    cl11_args a(c->kernels[sgt::dtype_type_size(s0.d.dtype) == 4 ? K_CONCAT4 : K_CONCAT2]);
    a.set(sr0.mem).set(sr0.td).set(sr1.mem).set(sr1.td).set(d.mem).set(d.td).set(n).set(dim).set(o);
    return cl11_launch(c->queue, c->exec, a, c->cfg.wg_ew, cl11_div_up(n, c->cfg.wg_ew));
}

bool cl11_op_pad(cl11_device * c, const cl11_tensor & dst, const cl11_tensor & s0, const scrapgpu_pad_params & p) {
    cl11_padp pp = {};
    pp.circ = p.circular != 0;
    for (int i = 0; i < 4; ++i) {
        pp.lp[i] = (int32_t) p.before[i];
        pp.rp[i] = (int32_t) p.after[i];
        // circular: i - lp + w >= 0 with w a multiple of ne (the kernel has no integer division)
        const int64_t ne = std::max<int64_t>(s0.d.ne[i], 1);
        pp.w[i] = (int32_t) (cl11_div_up((size_t) std::max<int32_t>(pp.lp[i], 0), (size_t) ne) * (size_t) ne);
    }
    const cl11_tref sr0 = cl11_tref_of(s0), d = cl11_tref_of(dst);
    const int32_t n = (int32_t) sgt::nelements(dst.d);
    cl11_args a(c->kernels[K_PAD]);
    a.set(sr0.mem).set(sr0.td).set(d.mem).set(d.td).set(n).set(pp);
    return cl11_launch(c->queue, c->exec, a, c->cfg.wg_ew, cl11_div_up(n, c->cfg.wg_ew));
}

bool cl11_op_sum_rows(cl11_device * c, const cl11_tensor & dst, const cl11_tensor & s0) {
    const cl11_tref sr0 = cl11_tref_of(s0), d = cl11_tref_of(dst);
    const size_t lsz = (size_t) std::min(c->cfg.wg_row, std::max(32, 1 << cl11_log2_exact((int) std::min<int64_t>(s0.d.ne[0], 1 << 20))));
    cl11_args a(c->kernels[K_SUM_ROWS]);
    a.set(sr0.mem).set(sr0.td).set(d.mem).set(d.td);
    return cl11_launch(c->queue, c->exec, a, lsz, (size_t) (s0.d.ne[1] * s0.d.ne[2] * s0.d.ne[3]));
}

bool cl11_op_argsort(cl11_device * c, const cl11_tensor & dst, const cl11_tensor & s0, const scrapgpu_argsort_params & p, bool top_k) {
    const cl11_tref sr0 = cl11_tref_of(s0), d = cl11_tref_of(dst);
    const int32_t desc = top_k || p.order == SCRAPGPU_ARGSORT_DESC ? 1 : 0;
    const int32_t npad = (int32_t) cl11_sort_npad(s0.d.ne[0]);
    const size_t  lsz  = (size_t) std::min<int64_t>(c->cfg.wg_row, std::max<int64_t>(32, npad / 2));
    cl11_args a(c->kernels[K_ARGSORT]);
    a.set(sr0.mem).set(sr0.td).set(d.mem).set(d.td).set(npad).set(desc);   // no local-memory argument: see k_argsort (Mesa Clover)
    return cl11_launch(c->queue, c->exec, a, lsz, (size_t) (s0.d.ne[1] * s0.d.ne[2] * s0.d.ne[3]));
}

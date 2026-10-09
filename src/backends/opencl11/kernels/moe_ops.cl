#ifndef CL11_NO_BASE
// CL11 backend: the operations that the hybrid / MoE graphs (LFM2, LFM2-MoE) need besides the matrix products:
// SSM_CONV (short convolution), CONCAT, PAD, SUM_ROWS, ARGSORT / TOP_K and the gather / scatter of MUL_MAT_ID.

// ---------------------------------------------------------------------------------------------
// ADD_ID (bias of an expert): dst[i0, i1, i2] = src0[i0, i1, i2] + src1[i0, ids[i1, i2]]. F32, ids I32. One work-item per element.
// ---------------------------------------------------------------------------------------------
kernel void k_add_id(global uchar * p0, tdesc t0, global uchar * p1, tdesc t1, global uchar * p2, tdesc t2, global uchar * pd, tdesc td, int n) {
    const int idx = LGID() * WG_EW + (int)get_local_id(0);
    if (idx >= n) return;
    int i0, i1, i2, i3;
    SPLIT4(idx, td, i0, i1, i2, i3);
    const int e = clamp(*(global const int *)(p2 + t2.off + i1 * t2.nb0 + i2 * t2.nb1), 0, t1.ne1 - 1);
    ST_F32(pd + OFF4(td, i0, i1, i2, i3), LD_F32(p0 + OFF4(t0, i0, i1, i2, i3)) + LD_F32(p1 + t1.off + i0 * t1.nb0 + e * t1.nb1));
}

// ---------------------------------------------------------------------------------------------
// SSM_CONV: dst[i0, i1, i2] = sum_j src0[i1 + j, i0, i2] * src1[j, i0]
// src0 [d_conv - 1 + n_t, d_inner, n_s] (the window slides by one element), src1 [d_conv, d_inner], dst [d_inner, n_t, n_s]. F32.
// One work-item per element of dst.
// ---------------------------------------------------------------------------------------------
kernel void k_ssm_conv(global uchar * p0, tdesc t0, global uchar * p1, tdesc t1, global uchar * pd, tdesc td, int n) {
    const int idx = LGID() * WG_EW + (int)get_local_id(0);
    if (idx >= n) return;
    int i0, i1, i2, i3;
    SPLIT4(idx, td, i0, i1, i2, i3);
    global const uchar * s = p0 + t0.off + i0 * t0.nb1 + i1 * t0.nb0 + i2 * t0.nb2;
    global const uchar * c = p1 + t1.off + i0 * t1.nb1;
    float sum = 0.0f;
    for (int j = 0; j < t1.ne0; ++j) {
        sum += LD_F32(s + j * t0.nb0) * LD_F32(c + j * t1.nb0);
    }
    ST_F32(pd + OFF4(td, i0, i1, i2, i3), sum);
}

// ---------------------------------------------------------------------------------------------
// CONCAT along `dim`: the elements of src0 first, then (offset `o` = src0->ne[dim]) those of src1. 4-byte (F32, I32) and 2-byte
// (F16, BF16, I16) elements, any strides.
// ---------------------------------------------------------------------------------------------
#define DEF_CONCAT(NAME, T)                                                                                        \
kernel void NAME(global uchar * p0, tdesc t0, global uchar * p1, tdesc t1, global uchar * pd, tdesc td, int n, int dim, int o) { \
    const int idx = LGID() * WG_EW + (int)get_local_id(0);                                                         \
    if (idx >= n) return;                                                                                          \
    int i0, i1, i2, i3;                                                                                            \
    SPLIT4(idx, td, i0, i1, i2, i3);                                                                               \
    global const uchar * s;                                                                                        \
    if (i0 < t0.ne0 && i1 < t0.ne1 && i2 < t0.ne2 && i3 < t0.ne3) {                                               \
        s = p0 + OFF4(t0, i0, i1, i2, i3);                                                                         \
    } else {                                                                                                       \
        s = p1 + OFF4(t1, i0 - (dim == 0 ? o : 0), i1 - (dim == 1 ? o : 0), i2 - (dim == 2 ? o : 0), i3 - (dim == 3 ? o : 0)); \
    }                                                                                                              \
    *(global T *)(pd + OFF4(td, i0, i1, i2, i3)) = *(global const T *)s;                                           \
}

DEF_CONCAT(k_concat_4, uint)
DEF_CONCAT(k_concat_2, ushort)

// ---------------------------------------------------------------------------------------------
// PAD (ggml_pad_ext, F32): dst[i] = src[i - lp] inside of the source, 0 around it. circ != 0 (ggml_pad_ext_circular): the source is
// repeated (wrap around) instead. w0..w3: a multiple of ne of src0 that is >= lp, so that i - lp + w is not negative (no integer division).
// ---------------------------------------------------------------------------------------------
typedef struct {
    int lp0, lp1, lp2, lp3;
    int rp0, rp1, rp2, rp3;
    int w0, w1, w2, w3;
    int circ;
} padp;

kernel void k_pad(global uchar * p0, tdesc t0, global uchar * pd, tdesc td, int n, padp pp) {
    const int idx = LGID() * WG_EW + (int)get_local_id(0);
    if (idx >= n) return;
    int i0, i1, i2, i3;
    SPLIT4(idx, td, i0, i1, i2, i3);
    float v = 0.0f;
    if (pp.circ != 0) {
        const int a0 = i0 - pp.lp0 + pp.w0, a1 = i1 - pp.lp1 + pp.w1, a2 = i2 - pp.lp2 + pp.w2, a3 = i3 - pp.lp3 + pp.w3;
        v = LD_F32(p0 + OFF4(t0, FMOD(a0, t0, 0), FMOD(a1, t0, 1), FMOD(a2, t0, 2), FMOD(a3, t0, 3)));
    } else if (i0 >= pp.lp0 && i0 < td.ne0 - pp.rp0 && i1 >= pp.lp1 && i1 < td.ne1 - pp.rp1 &&
               i2 >= pp.lp2 && i2 < td.ne2 - pp.rp2 && i3 >= pp.lp3 && i3 < td.ne3 - pp.rp3) {
        v = LD_F32(p0 + OFF4(t0, i0 - pp.lp0, i1 - pp.lp1, i2 - pp.lp2, i3 - pp.lp3));
    }
    ST_F32(pd + OFF4(td, i0, i1, i2, i3), v);
}

// ---------------------------------------------------------------------------------------------
// SUM_ROWS (F32): dst[0, i1, i2, i3] = sum_k src0[k, i1, i2, i3]. One work-group per row (the host picks a power of 2 as its size).
// ---------------------------------------------------------------------------------------------
kernel void k_sum_rows(global uchar * p0, tdesc t0, global uchar * pd, tdesc td) {
    local float sh[WG_ROW];
    const int lid = (int)get_local_id(0);
    const int lsz = (int)get_local_size(0);
    const int row = LGID();
    if (row >= t0.ne1 * t0.ne2 * t0.ne3) return; // uniform for the work-group
    const int q  = FDIV(row, t0, 1);
    const int i1 = row - q * t0.ne1;
    const int i3 = FDIV(q, t0, 2);
    const int i2 = q - i3 * t0.ne2;
    global const uchar * s = p0 + t0.off + i1 * t0.nb1 + i2 * t0.nb2 + i3 * t0.nb3;
    float sum = 0.0f;
    for (int k = lid; k < t0.ne0; k += lsz) {
        sum += LD_F32(s + k * t0.nb0);
    }
    sh[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int st = lsz >> 1; st > 0; st >>= 1) {
        if (lid < st) {
            sh[lid] += sh[lid + st];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        ST_F32(pd + td.off + i1 * td.nb1 + i2 * td.nb2 + i3 * td.nb3, sh[0]);
    }
}

// ---------------------------------------------------------------------------------------------
// ARGSORT / TOP_K (F32 keys, I32 indices): one work-group per row, the row (padded to `npad` = a power of 2) is sorted in local memory with a
// bitonic network. Keys equal: the smaller index first (a total order, so the padding at the end never mixes with the row). desc != 0: largest
// first. dst has td.ne0 entries per row: the whole row (ARGSORT) or the first k (TOP_K).
// ---------------------------------------------------------------------------------------------
#define SORT_BEFORE(ka, ia, kb, ib, desc) ((desc) ? ((ka) > (kb) || ((ka) == (kb) && (ia) < (ib))) : ((ka) < (kb) || ((ka) == (kb) && (ia) < (ib))))

// The keys and indices are static local arrays of SORT_NMAX entries (set by the host from the local memory of the device, see
// cl11_sort_nmax), not local-memory kernel arguments: Mesa Clover passes the kernel arguments at wrong offsets when a local-memory
// argument follows by-value struct arguments (tdesc), the kernel then reads garbage pointers (WX 2100, Mesa 25.0.7).
kernel void k_argsort(global uchar * p0, tdesc t0, global uchar * pd, tdesc td, int npad, int desc) {
    local float keys[SORT_NMAX];
    local int   idxs[SORT_NMAX];
    const int lid = (int)get_local_id(0);
    const int lsz = (int)get_local_size(0);
    const int row = LGID();
    if (row >= t0.ne1 * t0.ne2 * t0.ne3) return; // uniform for the work-group
    const int q  = FDIV(row, t0, 1);
    const int i1 = row - q * t0.ne1;
    const int i3 = FDIV(q, t0, 2);
    const int i2 = q - i3 * t0.ne2;
    global const uchar * s = p0 + t0.off + i1 * t0.nb1 + i2 * t0.nb2 + i3 * t0.nb3;
    const int nc = t0.ne0;
    const float pad = desc ? -INFINITY : INFINITY;
    for (int i = lid; i < npad; i += lsz) {
        keys[i] = i < nc ? LD_F32(s + i * t0.nb0) : pad;
        idxs[i] = i;
    }
    const int half_n = npad >> 1;
    for (int k = 2; k <= npad; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int p = lid; p < half_n; p += lsz) {
                const int a = ((p & ~(j - 1)) << 1) | (p & (j - 1));
                const int b = a + j;
                const float ka = keys[a], kb = keys[b];
                const int   ia = idxs[a], ib = idxs[b];
                const bool up = (a & k) == 0;
                // up: a must come before b; otherwise b before a
                const bool swap = up ? SORT_BEFORE(kb, ib, ka, ia, desc) : SORT_BEFORE(ka, ia, kb, ib, desc);
                if (swap) {
                    keys[a] = kb; keys[b] = ka;
                    idxs[a] = ib; idxs[b] = ia;
                }
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    global uchar * d = pd + td.off + i1 * td.nb1 + i2 * td.nb2 + i3 * td.nb3;
    for (int i = lid; i < td.ne0; i += lsz) {
        *(global int *)(d + i * td.nb0) = idxs[i];
    }
}

// ---------------------------------------------------------------------------------------------
// MUL_MAT_ID, prompt processing: the host sorts the (slot, token) pairs by expert; the columns of src1 of the pairs are gathered into a
// contiguous buffer (p-th column = pair p), the products of every expert are computed on a range of it by the normal MUL_MAT kernels and
// the columns of the result are scattered into dst. One work-group per pair.
// ---------------------------------------------------------------------------------------------
kernel void k_mid_gather(global uchar * pb, tdesc tb, global float * pg, global const int * pairs, int K, int ne11) {
    const int p = LGID();
    const int slot = pairs[2 * p], tok = pairs[2 * p + 1];
    global const uchar * s = pb + tb.off + (ne11 == 1 ? 0 : slot) * tb.nb1 + tok * tb.nb2;
    global float * d = pg + p * K;
    for (int k = (int)get_local_id(0); k < K; k += (int)get_local_size(0)) {
        d[k] = LD_F32(s + k * tb.nb0);
    }
}

kernel void k_mid_scatter(global const float * po, global uchar * pd, tdesc td, global const int * pairs, int M) {
    const int p = LGID();
    const int slot = pairs[2 * p], tok = pairs[2 * p + 1];
    global uchar * d = pd + td.off + slot * td.nb1 + tok * td.nb2;
    global const float * s = po + p * M;
    for (int m = (int)get_local_id(0); m < M; m += (int)get_local_size(0)) {
        ST_F32(d + m * td.nb0, s[m]);
    }
}
#endif // CL11_NO_BASE

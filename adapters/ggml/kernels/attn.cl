// CL11 backend: the two MUL_MAT of the attention of a single token (F16 KV cache, F32 src1 with one column).
//
// The cache of llama.cpp keeps the heads of a position together (row of position p: all the heads, ne02 * ne00 halves, contiguous),
// the generic mat-vec kernels read 256 bytes out of every 2 KiB (one head) with a few lanes per output and were slow (227 / 130 us
// per call for 0.5 MiB of cache on a GT 710, the DRAM needs 45 us): these kernels read whole rows of the cache with 16-byte loads
// and compute the r2 query heads (grouped query attention) that share a head of the cache together.

#ifndef CL11_NO_BASE

#ifdef CL11_NV_ASM
// two halves of a 32-bit word -> floats (one cvt each)
float2 cl11_h2(uint x) {
    float a, b;
    asm("{ .reg .b16 l, h; mov.b32 {l, h}, %2; cvt.f32.f16 %0, l; cvt.f32.f16 %1, h; }" : "=f"(a), "=f"(b) : "r"(x));
    return (float2)(a, b);
}
// 8 halves (16 bytes at p, 16-byte aligned) -> floats
float8 cl11_ld_h8(global const uchar * p) {
    const uint4 w = *(global const uint4 *)p;
    const float2 a = cl11_h2(w.x), b = cl11_h2(w.y), c = cl11_h2(w.z), d = cl11_h2(w.w);
    return (float8)(a.x, a.y, b.x, b.y, c.x, c.y, d.x, d.y);
}
#else
float8 cl11_ld_h8(global const uchar * p) {
    return vload_half8(0, (global const half *)p);
}
#endif

// ---------------------------------------------------------------------------------------------
// K cache: dst[i0, 0, i2] = sum_k K[k, i0, i02] * q[k, 0, i2], i02 = i2 / r2 (r2 <= 8, ne13 == ne03).
// Host conditions (cl11_attn_ok): ne00 % 8 == 0, the heads of a position are contiguous (nb2 == 2 * ne00), LANES = ne02 * ne00 / 8
// <= WG_ROW, 16-byte aligned rows.
// The work-items of a group of LANES read a row of the cache with 16 bytes each (coalesced: the whole row, 2 KiB for Qwen3-0.6B, per
// instruction): lane l = piece c = l % C of the head h = l / C (C = ne00 / 8 pieces per head); they multiply it with the same piece
// of the vectors q of the r2 query heads of the head h (in registers, ATT_JP of them at a time); the C partial sums of a
// (position, head, q) are added through local memory. A work-group has R = WG_ROW / LANES such groups, every group handles ATT_PL
// consecutive positions, two rows at a time (more loads in flight at once are slower on a GT 710: DRAM page conflicts):
// PW = R * ATT_PL positions per work-group. bc.tiles: groups of PW positions, bc.r2 = r2.
// ---------------------------------------------------------------------------------------------
#ifndef ATT_PL
#define ATT_PL 4      // positions of K per group of lanes (even)
#endif
#define ATT_JP 2      // query heads per pass (registers, local memory: many small work-groups hide the latency)

kernel void k_mvq_f16(global uchar * pa, tdesc ta, global uchar * pb, tdesc tb, global uchar * pc, tdesc tc, bcast bc) {
    local float red[ATT_JP * ATT_PL * WG_ROW];
    const int lid = (int)get_local_id(0);
    int g = LGID();
    if (g >= bc.tiles * ta.ne3) return;   // uniform for the work-group
    const int gq = FDIV_RAW(g, bc.mt, bc.lt);
    const int tile = g - gq * bc.tiles;
    const int i03 = gq;
    const int r2 = bc.r2;
    const int C = ta.ne0 >> 3;
    const int lanes = ta.ne2 * C;
    const int R = WG_ROW / lanes;
    const int PW = R * ATT_PL;
    const int s = lid / lanes;
    const int ll = lid - s * lanes;
    const bool act = s < R;
    const int h = ll / C;
    const int c = ll - h * C;
    const int i00 = tile * PW;
    const int ip0 = i00 + s * ATT_PL;       // first position of this group of lanes
    global const uchar * rowp = pa + ta.off + i03 * ta.nb3 + ll * 16;

    for (int jg = 0; jg < r2; jg += ATT_JP) {
        // pieces of the vectors q of this lane
        float8 q[ATT_JP];
#pragma unroll
        for (int j = 0; j < ATT_JP; ++j) {
            global const float4 * qp = (global const float4 *)(pb + tb.off + (h * r2 + min(jg + j, r2 - 1)) * tb.nb2 + i03 * tb.nb3) + 2 * c;
            q[j] = (float8)(qp[0], qp[1]);
        }
        for (int p = 0; p < ATT_PL; p += 2) {
            const float8 a0 = (act && ip0 + p < ta.ne1) ? cl11_ld_h8(rowp + (ip0 + p) * ta.nb1) : (float8)(0.0f);
            const float8 a1 = (act && ip0 + p + 1 < ta.ne1) ? cl11_ld_h8(rowp + (ip0 + p + 1) * ta.nb1) : (float8)(0.0f);
            if (act) {
#pragma unroll
                for (int j = 0; j < ATT_JP; ++j) {
                    red[(j * PW + s * ATT_PL + p) * lanes + ll] = (dot(a0.s0123, q[j].s0123) + dot(a0.s4567, q[j].s4567));
                    red[(j * PW + s * ATT_PL + p + 1) * lanes + ll] = (dot(a1.s0123, q[j].s0123) + dot(a1.s4567, q[j].s4567));
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        const int nout = min(ATT_JP, r2 - jg) * PW * ta.ne2;
        for (int t = lid; t < nout; t += WG_ROW) {
            const int jp = t / ta.ne2;
            const int hh = t - jp * ta.ne2;
            const int j = jp / PW;
            const int pr = jp - j * PW;
            if (i00 + pr < ta.ne1) {
                local const float * rp = red + jp * lanes + hh * C;
                float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
                int cc = 0;
                for (; cc + 3 < C; cc += 4) {
                    s0 += rp[cc]; s1 += rp[cc + 1]; s2 += rp[cc + 2]; s3 += rp[cc + 3];
                }
                for (; cc < C; ++cc) {
                    s0 += rp[cc];
                }
                const float sum = (s0 + s1) + (s2 + s3);
                ST_F32(pc + tc.off + (i00 + pr) * tc.nb0 + (hh * r2 + jg + j) * tc.nb2 + i03 * tc.nb3, sum);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

// ---------------------------------------------------------------------------------------------
// V cache (transposed): dst[i0, 0, i2] = sum_k A[k, i0, i02] * x[k, 0, i2], i02 = i2 / r2 (r2 <= 8, ne13 == ne03).
// It replaces the CONT of the transposed V cache followed by the MUL_MAT of the attention (the CONT is not needed): the elements of A
// along i0 are contiguous (ta.nb1 == 2) and the ones along k (the positions) are strided (ta.nb0).
// Host conditions (cl11_attn_ok): ne1 % 8 == 0, the heads of a position are contiguous (nb2 == 2 * ne1), LANES = ne2 * ne1 / 8
// <= WG_ATT, K = ne0 % 8 == 0, 16-byte aligned rows, x with contiguous elements.
// One work-group per batch: lane l (< LANES) reads the 16 bytes l of a row of A (all the heads of a position: the 2 KiB of the row in
// one instruction) = 8 consecutive outputs of the head h = l / (ne1 / 8), for the r2 vectors x of the head, ATT_JP of them at a time;
// the S = WG_ATT / LANES slices of the positions (slice s: [s * K / S, (s + 1) * K / S)) are added through local memory.
// Two rows at a time (more loads in flight at once are slower on a GT 710: DRAM page conflicts).
// ---------------------------------------------------------------------------------------------
kernel void k_mvt_f16(global uchar * pa, tdesc ta, global uchar * pb, tdesc tb, global uchar * pc, tdesc tc, bcast bc) {
    local float red[WG_ATT * 8];
    const int lid = (int)get_local_id(0);
    const int i03 = LGID();
    if (i03 >= ta.ne3) return;
    const int r2 = bc.r2;
    const int C = tc.ne0 >> 3;           // pieces of 8 outputs per head
    const int lanes = ta.ne2 * C;
    const int S = WG_ATT / lanes;
    const int s = lid / lanes;
    const int l = lid - s * lanes;
    const bool act = s < S;
    const int h = l / C;
    const int c = l - h * C;
    const int K = ta.ne0;
    const int kper = ((K / S + 7) >> 3) << 3;            // positions per slice (multiple of 8)
    const int k0 = act ? min(s * kper, K) : K;
    const int k1 = act ? min(k0 + kper, K) : K;
    global const uchar * a = pa + ta.off + i03 * ta.nb3 + l * 16;
    global const uchar * xb = pb + tb.off + (h * r2) * tb.nb2 + i03 * tb.nb3;
    for (int jg = 0; jg < r2; jg += ATT_JP) {
        float8 acc[ATT_JP];
        for (int j = 0; j < ATT_JP; ++j) {
            acc[j] = (float8)(0.0f);
        }
        for (int k = k0; k < k1; k += 2) {
            const float8 v0 = cl11_ld_h8(a + k * ta.nb0);
            const float8 v1 = cl11_ld_h8(a + (k + 1) * ta.nb0);
#pragma unroll
            for (int j = 0; j < ATT_JP; ++j) {
                if (jg + j < r2) {
                    const float2 p = vload2(0, (global const float *)(xb + (jg + j) * tb.nb2) + k);
                    acc[j] = CL11_FMA(v0, (float8)(p.x), acc[j]);
                    acc[j] = CL11_FMA(v1, (float8)(p.y), acc[j]);
                }
            }
        }
#pragma unroll
        for (int j = 0; j < ATT_JP; ++j) {
            if (jg + j < r2) {
                // add the S slices: red[s * lanes * 8 + l * 8 + i]
                barrier(CLK_LOCAL_MEM_FENCE);
                if (act) {
                    vstore8(acc[j], 0, red + (s * lanes + l) * 8);
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                for (int t = lid; t < lanes * 8; t += WG_ATT) {
                    float sum = 0.0f;
                    for (int ss = 0; ss < S; ++ss) {
                        sum += red[ss * lanes * 8 + t];
                    }
                    const int ll = t >> 3, i = t & 7;
                    const int hh = ll / C, cc = ll - hh * C;
                    ST_F32(pc + tc.off + (cc * 8 + i) * tc.nb0 + (hh * r2 + jg + j) * tc.nb2 + i03 * tc.nb3, sum);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// V cache (transposed), many work-groups: the same operation as k_mvt_f16 (the CONT of the transposed view is skipped), for the devices whose
// work-group is too small for the single-group kernel (AMD: 256 work-items; the single group of k_mvt_f16 then runs on one of the 8 compute
// units: 307 us instead of 38 us per call on a Radeon Pro WX 2100). Also the general case: any ne1 / ne2 / ne3 of src1 and the broadcast along i3.
// mat-vec with a transposed F16 src0: dst[i0] = sum_k A[k, i0] * x[k], where the elements along i0 are contiguous
// (ta.nb1 == 2) and the elements along k are strided (ta.nb0).
// A work-group computes 32 consecutive outputs i0 (the work-items of a warp read consecutive halves of the rows of A:
// coalesced) with WG_ROW/32 slices of k per output, the slices are added through local memory. src1 (x) is F32.
// bc.tiles: number of groups of 32 outputs.
// ---------------------------------------------------------------------------------------------
kernel void k_mvt_g_f16(global uchar * pa, tdesc ta, global uchar * pb, tdesc tb, global uchar * pc, tdesc tc, bcast bc) {
    local float part[WG_ROW];
    const int lid = (int)get_local_id(0);
    const int o  = lid & 31;
    const int ks = lid >> 5;
    const int nks = WG_ROW >> 5;
    int g = LGID();
    if (g >= bc.tiles * tc.ne1 * tc.ne2 * tc.ne3) return; // uniform for the work-group
    int gq = FDIV_RAW(g, bc.mt, bc.lt);
    const int tile = g - gq * bc.tiles;
    g = gq;
    gq = FDIV(g, tc, 1);
    const int i1 = g - gq * tc.ne1;
    g = gq;
    const int i3 = FDIV(g, tc, 2);
    const int i2 = g - i3 * tc.ne2;
    const int i02 = FDIV_RAW(i2, bc.m2, bc.l2);
    const int i03 = FDIV_RAW(i3, bc.m3, bc.l3);

    const int i0 = tile * 32 + o;
    const bool valid = i0 < tc.ne0;
    global const uchar * a = pa + ta.off + (valid ? i0 : 0) * ta.nb1 + i02 * ta.nb2 + i03 * ta.nb3;
    global const uchar * x = pb + tb.off + i1 * tb.nb1 + i2 * tb.nb2 + i3 * tb.nb3;
    const int ne00 = ta.ne0;
    float acc0 = 0.0f, acc1 = 0.0f;
    int k = ks;
    if (valid) {
        for (; k + nks < ne00; k += 2 * nks) {
            acc0 += vload_half(0, (global const half *)(a + k * ta.nb0)) * LD_F32(x + k * tb.nb0);
            acc1 += vload_half(0, (global const half *)(a + (k + nks) * ta.nb0)) * LD_F32(x + (k + nks) * tb.nb0);
        }
        if (k < ne00) {
            acc0 += vload_half(0, (global const half *)(a + k * ta.nb0)) * LD_F32(x + k * tb.nb0);
        }
    }
    part[lid] = acc0 + acc1;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (ks == 0 && valid) {
        float sum = part[o];
        for (int j = 1; j < nks; ++j) {
            sum += part[j * 32 + o];
        }
        ST_F32(pc + tc.off + i0 * tc.nb0 + i1 * tc.nb1 + i2 * tc.nb2 + i3 * tc.nb3, sum);
    }
}

#endif // CL11_NO_BASE

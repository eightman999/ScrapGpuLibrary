// CL11 backend: mat-vec for the repacked Q4_K / Q6_K weights (layout "R32", see cl11_rp_* in ggml-cl11.cpp).
//
// Same idea as mul_mv_r.cl: groups of 32 consecutive rows (rs = number of rows of the group, 32 or fewer for the last one), the
// lane of a warp is the row, so the 128-bit loads of a warp are 512 contiguous bytes and there is no reduction over the lanes.
// nb = ne0 / 256 super-blocks per row. Areas of a slice (all made of 16-byte units, group g starts at g * 32 * nb * <bytes of
// the area per super-block>, inside of it the unit of (super-block b, piece j) of the row lane is (b * U + j) * rs + lane):
//   Q4_K: area 0 "quants" (U = 8): the 128 bytes of nibbles of the super-block (piece j = bytes 16j .. 16j+15)
//         area 1 "header" (U = 1): the first 16 bytes of the block: half d, half dmin, 12 bytes of 6-bit scales / mins
//   Q6_K: area 0 "quants" (U = 12): ql (128 bytes, pieces 0..7), qh (64 bytes, pieces 8..11)
//         area 1 "scales" (U = 1): the 16 int8 scales, xor 0x80
//         area 2 "d" (half, at (g * 32 * nb + b * rs + lane) * 2)
// Area k starts at M * nb * (bytes of the previous areas per super-block).

// A work-group of NWR warps computes NRR groups of 32 rows (NRR rows per lane); the warps split the super-blocks of the rows
// (warp s: s, s + NWR, ...), the partial sums are added through local memory at the end (MVR_END of mul_mv_r.cl).

// QU: 16-byte units of the quants of a super-block
#define MVK_BEGIN(QU, XT, XINIT)                                                                       \
    const int lid  = (int)get_local_id(0);                                                            \
    const int lane = lid & 31;                                                                        \
    const int sp   = lid >> 5;                                                                        \
    int g = LGID();                                                                                   \
    if (g >= bc.tiles * tc.ne1 * tc.ne2 * tc.ne3) return; /* uniform for the work-group */            \
    int gq = FDIV_RAW(g, bc.mt, bc.lt);                                                               \
    const int tile = g - gq * bc.tiles;                                                               \
    g = gq;                                                                                           \
    gq = FDIV(g, tc, 1);                                                                              \
    const int i1 = g - gq * tc.ne1;                                                                   \
    g = gq;                                                                                           \
    const int i3 = FDIV(g, tc, 2);                                                                    \
    const int i2 = g - i3 * tc.ne2;                                                                   \
    const int i02 = FDIV_RAW(i2, bc.m2, bc.l2);                                                       \
    const int i03 = FDIV_RAW(i3, bc.m3, bc.l3);                                                       \
    global const uchar * abase = pa + ta.off + i02 * ta.nb2 + i03 * ta.nb3;                           \
    const int M   = ta.ne1;                                                                           \
    const int nb  = ta.ne0 >> 8;                                                                      \
    const int ngrp = (M + 31) >> 5;                                                                   \
    const int gb  = tile * NRR;                                                                       \
    global const uchar * hbase = abase + M * nb * (16 * (QU));                                        \
    global const uchar * dbase = hbase + M * nb * 16;                                                 \
    global const uint4 * qp[NRR];                                                                     \
    global const uint4 * hp[NRR];                                                                     \
    global const uchar * dp[NRR];                                                                     \
    int   gr[NRR];                                                                                    \
    int   rsz[NRR];                                                                                   \
    float acc[NRR];                                                                                   \
    bool  full = true;                                                                                \
    for (int j = 0; j < NRR; ++j) {                                                                   \
        const int gj  = (gb + j < ngrp) ? gb + j : 0;                                                 \
        const int rs  = min(32, M - gj * 32);                                                         \
        const int lc  = min(lane, rs - 1);                                                            \
        gr[j]  = gb + j;                                                                              \
        rsz[j] = rs;                                                                                  \
        full  &= (rs == 32);                                                                          \
        qp[j]  = (global const uint4 *)(abase + gj * (32 * nb * 16 * (QU))) + lc + sp * (QU) * rs;    \
        hp[j]  = (global const uint4 *)(hbase + gj * (32 * nb * 16)) + lc + sp * rs;                  \
        dp[j]  = dbase + (gj * 32 * nb + sp * rs + lc) * 2;                                           \
        acc[j] = 0.0f;                                                                                \
    }                                                                                                 \
    XT const float4 * xp = (XINIT) + sp * 64;

// high nibbles without the shift: the byte is 16 * q (the factor 1/16 is applied to the sum of the sub-block)
float4 f4_h16(uint w) { return f4_mag(w & 0xF0F0F0F0u) - (float4)(8388608.0f); }

#define V4(v, i) ((i) == 0 ? (v).x : (i) == 1 ? (v).y : (i) == 2 ? (v).z : (v).w)

// one chunk (64 weights, 32 bytes = 2 pieces of 16 bytes) of a Q4_K super-block of the NRR rows:
// the low nibbles are the sub-block 2c, the high nibbles the sub-block 2c + 1;
// tl[j] += sc * sum(q x) (low), th[j] += sc * sum(16 q x) (high), uu[j] += min * sum(x)
// scv[j], mnv[j]: the 4 scales / mins of the sub-blocks 4 (c / 2) .. 4 (c / 2) + 3
#define MVK_Q4_CHUNK_WIDE(C, RS)                                                                      \
    {                                                                                                 \
        uint4 qa[NRR], qb[NRR];                                                                       \
        for (int j = 0; j < NRR; ++j) {                                                               \
            qa[j] = qp[j][(2 * (C)) * RS(j)];                                                         \
            qb[j] = qp[j][(2 * (C) + 1) * RS(j)];                                                     \
        }                                                                                             \
        {                                                                                             \
            const float4 x0 = XLO(16 * (C) + 0), x1 = XLO(16 * (C) + 1), x2 = XLO(16 * (C) + 2), x3 = XLO(16 * (C) + 3); \
            const float4 x4 = XLO(16 * (C) + 4), x5 = XLO(16 * (C) + 5), x6 = XLO(16 * (C) + 6), x7 = XLO(16 * (C) + 7); \
            const float4 sv = ((x0 + x1) + (x2 + x3)) + ((x4 + x5) + (x6 + x7));                      \
            const float sx = (sv.x + sv.y) + (sv.z + sv.w);                                           \
            for (int j = 0; j < NRR; ++j) {                                                           \
                float4 a = f4_lo4(qa[j].x) * x0;                                                      \
                a = CL11_FMA(f4_lo4(qa[j].y), x1, a); a = CL11_FMA(f4_lo4(qa[j].z), x2, a); a = CL11_FMA(f4_lo4(qa[j].w), x3, a); \
                a = CL11_FMA(f4_lo4(qb[j].x), x4, a); a = CL11_FMA(f4_lo4(qb[j].y), x5, a); a = CL11_FMA(f4_lo4(qb[j].z), x6, a); a = CL11_FMA(f4_lo4(qb[j].w), x7, a); \
                const float s = (a.x + a.y) + (a.z + a.w);                                            \
                tl[j] = CL11_FMA(V4(scv[j], (2 * (C)) & 3), s, tl[j]);                                     \
                uu[j] = CL11_FMA(V4(mnv[j], (2 * (C)) & 3), sx, uu[j]);                                    \
            }                                                                                         \
        }                                                                                             \
        {                                                                                             \
            const float4 x0 = XLO(16 * (C) + 8), x1 = XLO(16 * (C) + 9), x2 = XLO(16 * (C) + 10), x3 = XLO(16 * (C) + 11); \
            const float4 x4 = XLO(16 * (C) + 12), x5 = XLO(16 * (C) + 13), x6 = XLO(16 * (C) + 14), x7 = XLO(16 * (C) + 15); \
            const float4 sv = ((x0 + x1) + (x2 + x3)) + ((x4 + x5) + (x6 + x7));                      \
            const float sx = (sv.x + sv.y) + (sv.z + sv.w);                                           \
            for (int j = 0; j < NRR; ++j) {                                                           \
                float4 a = f4_h16(qa[j].x) * x0;                                                      \
                a = CL11_FMA(f4_h16(qa[j].y), x1, a); a = CL11_FMA(f4_h16(qa[j].z), x2, a); a = CL11_FMA(f4_h16(qa[j].w), x3, a); \
                a = CL11_FMA(f4_h16(qb[j].x), x4, a); a = CL11_FMA(f4_h16(qb[j].y), x5, a); a = CL11_FMA(f4_h16(qb[j].z), x6, a); a = CL11_FMA(f4_h16(qb[j].w), x7, a); \
                const float s = (a.x + a.y) + (a.z + a.w);                                            \
                th[j] = CL11_FMA(V4(scv[j], (2 * (C) + 1) & 3), s, th[j]);                                 \
                uu[j] = CL11_FMA(V4(mnv[j], (2 * (C) + 1) & 3), sx, uu[j]);                                \
            }                                                                                         \
        }                                                                                             \
    }

#define MVK_Q4_HALF(C, NIB, F4, X0, SC, TT, QOFF)                                                     \
        {                                                                                             \
            float4 av[NRR];                                                                           \
            float4 sv;                                                                                \
            {                                                                                         \
                const float4 x0 = XLO(X0 + 0), x1 = XLO(X0 + 1), x2 = XLO(X0 + 2), x3 = XLO(X0 + 3); \
                sv = (x0 + x1) + (x2 + x3);                                                           \
                for (int j = 0; j < NRR; ++j) {                                                       \
                    av[j] = F4(qa[j].x) * x0;                                                         \
                    av[j] = CL11_FMA(F4(qa[j].y), x1, av[j]); av[j] = CL11_FMA(F4(qa[j].z), x2, av[j]); av[j] = CL11_FMA(F4(qa[j].w), x3, av[j]); \
                }                                                                                     \
            }                                                                                         \
            {                                                                                         \
                const float4 x4 = XLO(X0 + 4), x5 = XLO(X0 + 5), x6 = XLO(X0 + 6), x7 = XLO(X0 + 7); \
                sv += (x4 + x5) + (x6 + x7);                                                          \
                for (int j = 0; j < NRR; ++j) {                                                       \
                    av[j] = CL11_FMA(F4(qb[j].x), x4, av[j]); av[j] = CL11_FMA(F4(qb[j].y), x5, av[j]); av[j] = CL11_FMA(F4(qb[j].z), x6, av[j]); av[j] = CL11_FMA(F4(qb[j].w), x7, av[j]); \
                }                                                                                     \
            }                                                                                         \
            const float sx = (sv.x + sv.y) + (sv.z + sv.w);                                           \
            for (int j = 0; j < NRR; ++j) {                                                           \
                const float s = (av[j].x + av[j].y) + (av[j].z + av[j].w);                            \
                TT[j] = CL11_FMA(V4(scv[j], SC), s, TT[j]);                                                \
                uu[j] = CL11_FMA(V4(mnv[j], SC), sx, uu[j]);                                               \
            }                                                                                         \
        }

#define MVK_Q4_CHUNK_SPLIT(C, RS)                                                                           \
    {                                                                                                 \
        uint4 qa[NRR], qb[NRR];                                                                       \
        for (int j = 0; j < NRR; ++j) {                                                               \
            qa[j] = qp[j][(2 * (C)) * RS(j)];                                                         \
            qb[j] = qp[j][(2 * (C) + 1) * RS(j)];                                                     \
        }                                                                                             \
        MVK_Q4_HALF(C, 0, f4_lo4, 16 * (C) + 0, (2 * (C)) & 3, tl, 0)                                 \
        MVK_Q4_HALF(C, 1, f4_h16, 16 * (C) + 8, (2 * (C) + 1) & 3, th, 0)                             \
    }

// the x values of a half chunk are loaded 4 at a time when there is a single row per lane (Fermi: 63 registers per work-item), all 8 otherwise
#if NRR_Q4_K == 1
#define MVK_Q4_CHUNK MVK_Q4_CHUNK_SPLIT
#else
#define MVK_Q4_CHUNK MVK_Q4_CHUNK_WIDE
#endif

#define MVK_Q4_LOOP(RS)                                                                               \
    for (int b = sp; b < nb; b += NWR) {                                                              \
        uint4 hd[NRR];                                                                                \
        float4 scv[NRR], mnv[NRR];                                                                    \
        float tl[NRR], th[NRR], uu[NRR];                                                              \
        for (int j = 0; j < NRR; ++j) {                                                               \
            hd[j] = hp[j][0];                                                                       \
            tl[j] = 0.0f;                                                                             \
            th[j] = 0.0f;                                                                             \
            uu[j] = 0.0f;                                                                             \
            scv[j] = f4_mag(hd[j].y & 0x3F3F3F3Fu) - (float4)(8388608.0f);                            \
            mnv[j] = f4_mag(hd[j].z & 0x3F3F3F3Fu) - (float4)(8388608.0f);                            \
        }                                                                                             \
        MVK_Q4_CHUNK(0, RS)                                                                           \
        MVK_Q4_CHUNK(1, RS)                                                                           \
        for (int j = 0; j < NRR; ++j) {                                                               \
            scv[j] = f4_mag((hd[j].w & 0x0F0F0F0Fu) | ((hd[j].y >> 2) & 0x30303030u)) - (float4)(8388608.0f); \
            mnv[j] = f4_mag(((hd[j].w >> 4) & 0x0F0F0F0Fu) | ((hd[j].z >> 2) & 0x30303030u)) - (float4)(8388608.0f); \
        }                                                                                             \
        MVK_Q4_CHUNK(2, RS)                                                                           \
        MVK_Q4_CHUNK(3, RS)                                                                           \
        for (int j = 0; j < NRR; ++j) {                                                               \
            const float d    = vload_half(0, (global const half *)hp[j]);                            \
            const float dmin = vload_half(1, (global const half *)hp[j]);                            \
            acc[j] = CL11_FMA(d, CL11_FMA(0.0625f, th[j], tl[j]), CL11_FMA(-dmin, uu[j], acc[j]));                   \
            qp[j] += NWR * 8 * RS(j);                                                                 \
            hp[j] += NWR * RS(j);                                                                     \
        }                                                                                             \
        xp += NWR * 64;                                                                               \
    }

// Fermi variant (DN_Q4_K, "denormal products", see mul_mv_r.cl; src1 prepared by k_xprep with the pattern hbit = 8, the sums of x of the groups of 16
// weights are at psp[]): nibbles masked in place are floats. One word (4 bytes = 4 low + 4 high nibbles) of the NRR rows: XA / XB = the
// float4 of xs of the low / high nibbles.
#define DNK_WORD(Q, W, XA, XB)                                                                        \
    {                                                                                                 \
        const float4 xl = XLO(XA), xh = XLO(XB);                                                      \
        for (int j = 0; j < NRR; ++j) {                                                               \
            const uint w = Q[j].W;                                                                    \
            const uint t = w >> 24;                                                                   \
            sl[j] = fma(as_float(w & 0x0000000Fu), xl.x, sl[j]);                                      \
            sh[j] = fma(as_float(w & 0x000000F0u), xh.x, sh[j]);                                      \
            sl[j] = fma(as_float(w & 0x00000F00u), xl.y, sl[j]);                                      \
            sh[j] = fma(as_float(w & 0x0000F000u), xh.y, sh[j]);                                      \
            sl[j] = fma(as_float(w & 0x000F0000u), xl.z, sl[j]);                                      \
            sh[j] = fma(as_float(w & 0x00F00000u), xh.z, sh[j]);                                      \
            sl[j] = fma(as_float(t & 0x0Fu), xl.w, sl[j]);                                            \
            sh[j] = fma(as_float(t & 0xF0u), xh.w, sh[j]);                                            \
        }                                                                                             \
    }

#define DNK_CHUNK(C, RS)                                                                              \
    {                                                                                                 \
        uint4 qa[NRR], qb[NRR];                                                                       \
        float sl[NRR], sh[NRR];                                                                       \
        for (int j = 0; j < NRR; ++j) {                                                               \
            qa[j] = qp[j][(2 * (C)) * RS(j)];                                                         \
            qb[j] = qp[j][(2 * (C) + 1) * RS(j)];                                                     \
            sl[j] = 0.0f;                                                                             \
            sh[j] = 0.0f;                                                                             \
        }                                                                                             \
        DNK_WORD(qa, x, 16 * (C) + 0, 16 * (C) + 8)  DNK_WORD(qa, y, 16 * (C) + 1, 16 * (C) + 9)      \
        DNK_WORD(qa, z, 16 * (C) + 2, 16 * (C) + 10) DNK_WORD(qa, w, 16 * (C) + 3, 16 * (C) + 11)     \
        DNK_WORD(qb, x, 16 * (C) + 4, 16 * (C) + 12) DNK_WORD(qb, y, 16 * (C) + 5, 16 * (C) + 13)     \
        DNK_WORD(qb, z, 16 * (C) + 6, 16 * (C) + 14) DNK_WORD(qb, w, 16 * (C) + 7, 16 * (C) + 15)     \
        const float sxl = psp[4 * (C)] + psp[4 * (C) + 1];                                            \
        const float sxh = psp[4 * (C) + 2] + psp[4 * (C) + 3];                                        \
        for (int j = 0; j < NRR; ++j) {                                                               \
            tl[j] = fma(V4(scv[j], (2 * (C)) & 3), sl[j], tl[j]);                                      \
            th[j] = fma(V4(scv[j], (2 * (C) + 1) & 3), sh[j], th[j]);                                  \
            uu[j] = fma(V4(mnv[j], (2 * (C)) & 3), sxl, fma(V4(mnv[j], (2 * (C) + 1) & 3), sxh, uu[j])); \
        }                                                                                             \
    }

#define MVK_Q4_DN_LOOP(RS)                                                                            \
    for (int b = sp; b < nb; b += NWR) {                                                              \
        uint4 hd[NRR];                                                                                \
        float4 scv[NRR], mnv[NRR];                                                                    \
        float tl[NRR], th[NRR], uu[NRR];                                                              \
        for (int j = 0; j < NRR; ++j) {                                                               \
            hd[j] = hp[j][0];                                                                         \
            tl[j] = 0.0f;                                                                             \
            th[j] = 0.0f;                                                                             \
            uu[j] = 0.0f;                                                                             \
            scv[j] = f4_mag(hd[j].y & 0x3F3F3F3Fu) - (float4)(8388608.0f);                            \
            mnv[j] = f4_mag(hd[j].z & 0x3F3F3F3Fu) - (float4)(8388608.0f);                            \
        }                                                                                             \
        DNK_CHUNK(0, RS)                                                                              \
        DNK_CHUNK(1, RS)                                                                              \
        for (int j = 0; j < NRR; ++j) {                                                               \
            scv[j] = f4_mag((hd[j].w & 0x0F0F0F0Fu) | ((hd[j].y >> 2) & 0x30303030u)) - (float4)(8388608.0f); \
            mnv[j] = f4_mag(((hd[j].w >> 4) & 0x0F0F0F0Fu) | ((hd[j].z >> 2) & 0x30303030u)) - (float4)(8388608.0f); \
        }                                                                                             \
        DNK_CHUNK(2, RS)                                                                              \
        DNK_CHUNK(3, RS)                                                                              \
        for (int j = 0; j < NRR; ++j) {                                                               \
            const float d    = vload_half(0, (global const half *)hp[j]);                            \
            const float dmin = vload_half(1, (global const half *)hp[j]);                            \
            acc[j] = fma(d * P2(149 - DN_E), tl[j] + th[j], fma(-dmin, uu[j], acc[j]));               \
            qp[j] += NWR * 8 * RS(j);                                                                 \
            hp[j] += NWR * RS(j);                                                                     \
        }                                                                                             \
        xp += NWR * 64;                                                                               \
        psp += NWR * 16;                                                                              \
    }

// Q6_K: weights (qd, l) of a half h (0, 1) of the super-block: q = (nibble | field << 4) - 32, the nibble is the low (qd 0, 1) or
// high (qd 2, 3) half of the ql byte (h * 64 + (qd & 1) * 32 + l), the field the bits 2 * qd of the qh byte (h * 32 + l).
// A work-item handles 16 consecutive l (lh = 0, 1) of the 4 quadrants: pieces A = ql[h * 64 + 16 lh], A2 = ql[h * 64 + 32 + 16 lh],
// C = qh[h * 32 + 16 lh], 4 words each (l = 16 lh + 4 k + 0..3), the scale of (h, qd, lh) is scales[h * 8 + lh + 2 qd].
// x of (h, qd, l = 16 lh + 4 k) is the float4 h * 32 + qd * 8 + 4 lh + k of the super-block.

// the 4 words of the values of the quadrant qd (formulas of the 4 quadrants, W = x y z w)
#define V6_0(W) ((pa_[j].W & 0x0F0F0F0Fu) | ((pc_[j].W << 4) & 0x30303030u))
#define V6_1(W) ((pb_[j].W & 0x0F0F0F0Fu) | ((pc_[j].W << 2) & 0x30303030u))
#define V6_2(W) (((pa_[j].W >> 4) & 0x0F0F0F0Fu) | (pc_[j].W & 0x30303030u))
#define V6_3(W) (((pb_[j].W >> 4) & 0x0F0F0F0Fu) | ((pc_[j].W >> 2) & 0x30303030u))

#define MVK_Q6_Q(H, LH, QD, VF)                                                                       \
    {                                                                                                 \
        const int xo = (H) * 32 + (QD) * 8 + 4 * (LH);                                                \
        const float4 x0 = XLO(xo), x1 = XLO(xo + 1), x2 = XLO(xo + 2), x3 = XLO(xo + 3);                  \
        for (int j = 0; j < NRR; ++j) {                                                               \
            float4 a = f4_q6(VF(x)) * x0;                                                            \
            a = CL11_FMA(f4_q6(VF(y)), x1, a); a = CL11_FMA(f4_q6(VF(z)), x2, a); a = CL11_FMA(f4_q6(VF(w)), x3, a); \
            const float s = (a.x + a.y) + (a.z + a.w);                                                \
            tt[j] = CL11_FMA(sc[j][(H) * 8 + (LH) + 2 * (QD)], s, tt[j]);                                  \
        }                                                                                             \
    }

#define MVK_Q6_PIECE(H, LH, RS)                                                                       \
    {                                                                                                 \
        uint4 pa_[NRR], pb_[NRR], pc_[NRR];                                                           \
        for (int j = 0; j < NRR; ++j) {                                                               \
            pa_[j] = qp[j][(4 * (H) + (LH)) * RS(j)];                                                 \
            pb_[j] = qp[j][(4 * (H) + 2 + (LH)) * RS(j)];                                             \
            pc_[j] = qp[j][(8 + 2 * (H) + (LH)) * RS(j)];                                             \
        }                                                                                             \
        MVK_Q6_Q(H, LH, 0, V6_0)                                                                      \
        MVK_Q6_Q(H, LH, 1, V6_1)                                                                      \
        MVK_Q6_Q(H, LH, 2, V6_2)                                                                      \
        MVK_Q6_Q(H, LH, 3, V6_3)                                                                      \
    }

#define MVK_Q6_LOOP(RS)                                                                               \
    for (int b = sp; b < nb; b += NWR) {                                                              \
        float tt[NRR];                                                                                \
        float sc[NRR][16];                                                                            \
        for (int j = 0; j < NRR; ++j) {                                                               \
            const uint4 sw = hp[j][0];                                                                \
            const float4 s0 = f4_u8(sw.x), s1 = f4_u8(sw.y), s2 = f4_u8(sw.z), s3 = f4_u8(sw.w);      \
            sc[j][0]  = s0.x; sc[j][1]  = s0.y; sc[j][2]  = s0.z; sc[j][3]  = s0.w;                   \
            sc[j][4]  = s1.x; sc[j][5]  = s1.y; sc[j][6]  = s1.z; sc[j][7]  = s1.w;                   \
            sc[j][8]  = s2.x; sc[j][9]  = s2.y; sc[j][10] = s2.z; sc[j][11] = s2.w;                   \
            sc[j][12] = s3.x; sc[j][13] = s3.y; sc[j][14] = s3.z; sc[j][15] = s3.w;                   \
            tt[j] = 0.0f;                                                                             \
        }                                                                                             \
        MVK_Q6_PIECE(0, 0, RS)                                                                        \
        MVK_Q6_PIECE(0, 1, RS)                                                                        \
        MVK_Q6_PIECE(1, 0, RS)                                                                        \
        MVK_Q6_PIECE(1, 1, RS)                                                                        \
        for (int j = 0; j < NRR; ++j) {                                                               \
            const float d = vload_half(0, (global const half *)dp[j]);                                \
            acc[j] = CL11_FMA(d, tt[j], acc[j]);                                                           \
            qp[j] += NWR * 12 * RS(j);                                                                \
            hp[j] += NWR * RS(j);                                                                     \
            dp[j] += NWR * 2 * RS(j);                                                                 \
        }                                                                                             \
        xp += NWR * 64;                                                                               \
    }

// Fermi variant (DN_Q6_K) of the Q6_K loop (pattern hbit = 0 of k_xprep: every byte i of a word has the scale 2^(DN_E - 8 i), byte 3 is shifted): the
// 6-bit values (q = nibble | field << 4, one per byte) are combined as before, then each byte is masked in place (the float q * 2^(8 i - 149))
// and multiplied by xs; the offset 32 is removed with the sum of x of the 16 weights of a scale (psp[group], group = scale index).
#define DNQ6_WORD(VF, W, XI)                                                                          \
    {                                                                                                 \
        const float4 xs = XLO(XI);                                                                    \
        for (int j = 0; j < NRR; ++j) {                                                               \
            const uint c = VF(W);                                                                     \
            sg[j] = fma(as_float(c & 0x000000FFu), xs.x, sg[j]);                                      \
            sg[j] = fma(as_float(c & 0x0000FF00u), xs.y, sg[j]);                                      \
            sg[j] = fma(as_float(c & 0x00FF0000u), xs.z, sg[j]);                                      \
            sg[j] = fma(as_float(c >> 24), xs.w, sg[j]);                                              \
        }                                                                                             \
    }

#define DNQ6_GROUP(H, LH, QD, VF)                                                                     \
    {                                                                                                 \
        const int xo = (H) * 32 + (QD) * 8 + 4 * (LH);                                                \
        const int si = (H) * 8 + (LH) + 2 * (QD);                                                     \
        for (int j = 0; j < NRR; ++j) {                                                               \
            sg[j] = 0.0f;                                                                             \
        }                                                                                             \
        DNQ6_WORD(VF, x, xo + 0) DNQ6_WORD(VF, y, xo + 1) DNQ6_WORD(VF, z, xo + 2) DNQ6_WORD(VF, w, xo + 3) \
        const float sx = psp[si];                                                                     \
        for (int j = 0; j < NRR; ++j) {                                                               \
            const uint sw = (si >> 2) == 0 ? hw[j].x : (si >> 2) == 1 ? hw[j].y : (si >> 2) == 2 ? hw[j].z : hw[j].w; \
            const float sc = V4(f4_u8(sw), (si & 3));                                                   \
            tt[j] = fma(sc, fma(-P2(DN_E - 144), sx, sg[j]), tt[j]);                                   \
        }                                                                                             \
    }

#define DNQ6_PIECE(H, LH, RS)                                                                         \
    {                                                                                                 \
        uint4 pa_[NRR], pb_[NRR], pc_[NRR];                                                           \
        for (int j = 0; j < NRR; ++j) {                                                               \
            pa_[j] = qp[j][(4 * (H) + (LH)) * RS(j)];                                                 \
            pb_[j] = qp[j][(4 * (H) + 2 + (LH)) * RS(j)];                                             \
            pc_[j] = qp[j][(8 + 2 * (H) + (LH)) * RS(j)];                                             \
        }                                                                                             \
        DNQ6_GROUP(H, LH, 0, V6_0)                                                                    \
        DNQ6_GROUP(H, LH, 1, V6_1)                                                                    \
        DNQ6_GROUP(H, LH, 2, V6_2)                                                                    \
        DNQ6_GROUP(H, LH, 3, V6_3)                                                                    \
    }

#define MVK_Q6_DN_LOOP(RS)                                                                            \
    for (int b = sp; b < nb; b += NWR) {                                                              \
        float tt[NRR], sg[NRR];                                                                       \
        uint4 hw[NRR];                                                                                \
        for (int j = 0; j < NRR; ++j) {                                                               \
            hw[j] = hp[j][0];                                                                         \
            tt[j] = 0.0f;                                                                             \
        }                                                                                             \
        DNQ6_PIECE(0, 0, RS)                                                                          \
        DNQ6_PIECE(0, 1, RS)                                                                          \
        DNQ6_PIECE(1, 0, RS)                                                                          \
        DNQ6_PIECE(1, 1, RS)                                                                          \
        for (int j = 0; j < NRR; ++j) {                                                               \
            const float d = vload_half(0, (global const half *)dp[j]);                                \
            acc[j] = fma(d * P2(149 - DN_E), tt[j], acc[j]);                                          \
            qp[j] += NWR * 12 * RS(j);                                                                \
            hp[j] += NWR * RS(j);                                                                     \
            dp[j] += NWR * 2 * RS(j);                                                                 \
        }                                                                                             \
        xp += NWR * 64;                                                                               \
        psp += NWR * 16;                                                                              \
    }

#if !defined(CL11_NO_K) && !defined(CL11_ONLY_MM)
#define WG_MVR WG_MVR_Q4_K
#define XLDG   LDG_Q4_K
#define NRR    NRR_Q4_K
#define NWR    (WG_MVR / 32)
#define XLO XLO_G
kernel void k_mvr_q4_k(MVR_ARGS_G) {
    MVK_BEGIN(8, global, MVR_XG)
    if (full) { MVK_Q4_LOOP(RS_FULL) } else { MVK_Q4_LOOP(RS_VAR) }
    MVR_END
}

#undef XLO
#define XLO XLO_C
#if defined(DN_Q4_K) && DN_Q4_K
kernel void k_mvrc_q4_k(MVR_ARGS_C) {
    MVK_BEGIN(8, constant, MVR_XC)
    constant const float * psp = (constant const float *)px + ta.ne0 + sp * 16;
    if (full) { MVK_Q4_DN_LOOP(RS_FULL) } else { MVK_Q4_DN_LOOP(RS_VAR) }
    MVR_END
}
#else
kernel void k_mvrc_q4_k(MVR_ARGS_C) {
    MVK_BEGIN(8, constant, MVR_XC)
    if (full) { MVK_Q4_LOOP(RS_FULL) } else { MVK_Q4_LOOP(RS_VAR) }
    MVR_END
}
#endif
#undef XLO
#undef XLDG
#undef NWR
#undef NRR
#undef WG_MVR

#define WG_MVR WG_MVR_Q6_K
#define XLDG   LDG_Q6_K
#define NRR    NRR_Q6_K
#define NWR    (WG_MVR / 32)
#define XLO XLO_G
kernel void k_mvr_q6_k(MVR_ARGS_G) {
    MVK_BEGIN(12, global, MVR_XG)
    if (full) { MVK_Q6_LOOP(RS_FULL) } else { MVK_Q6_LOOP(RS_VAR) }
    MVR_END
}

#undef XLO
#define XLO XLO_C
#if defined(DN_Q6_K) && DN_Q6_K
kernel void k_mvrc_q6_k(MVR_ARGS_C) {
    MVK_BEGIN(12, constant, MVR_XC)
    constant const float * psp = (constant const float *)px + ta.ne0 + sp * 16;
    if (full) { MVK_Q6_DN_LOOP(RS_FULL) } else { MVK_Q6_DN_LOOP(RS_VAR) }
    MVR_END
}
#else
kernel void k_mvrc_q6_k(MVR_ARGS_C) {
    MVK_BEGIN(12, constant, MVR_XC)
    if (full) { MVK_Q6_LOOP(RS_FULL) } else { MVK_Q6_LOOP(RS_VAR) }
    MVR_END
}
#endif
#undef XLO
#undef XLDG
#undef NWR
#undef NRR
#undef WG_MVR
#endif // CL11_NO_K

// CL11 backend: mat-vec for the repacked Q8_0 / Q4_0 weights (layout "R32", see cl11_rp_* in ggml-cl11.cpp).
//
// Layout of a slice (ne1 = M rows, nb = ne0 / 32 blocks per row) of a repacked tensor, groups of 32 consecutive rows,
// rs = number of rows of the group (32, fewer for the last group):
//   quants: group g at g * 32 * nb * QB bytes (QB = 32 for Q8_0, 16 for Q4_0), inside of it, 16-byte units:
//             Q8_0: unit ((b * 2 + h) * rs + lane): bytes 16 * h .. 16 * h + 15 of the quants of block b of row lane, xor 0x80
//             Q4_0: unit (b * rs + lane): the 16 bytes of nibbles of block b of row lane
//   scales: half, at M * nb * QB + 2 * (g * 32 * nb + b * rs + lane)
// A warp reads 512 contiguous bytes per load (one 16-byte unit per lane): the lane is the row, so there is no
// reduction over the lanes and the vector src1 is the same for all the lanes of a warp.
//
// A work-group of S = NWR warps computes the NRR groups (NRR rows per lane) tile * NRR ...: the S warps split the
// blocks of the rows (warp s: blocks s, s + S, ...), the partial sums are added through local memory at the end.

float4 f4_u8(uint w) { return f4_mag(w) - (float4)(8388736.0f); }                               // 4 bytes stored xor 0x80 -> signed values
float4 f4_n0(uint w) { return f4_mag(w & 0x0F0F0F0Fu) - (float4)(8388616.0f); }                 // low nibbles - 8
float4 f4_n1(uint w) { return f4_mag((w >> 4) & 0x0F0F0F0Fu) - (float4)(8388616.0f); }          // high nibbles - 8

// The macros below use WG_MVR (work-items of a work-group), NWR = WG_MVR / 32 (warps, they split the blocks of the rows) and NRR (groups
// of 32 rows per work-item = rows per lane); every kernel family defines them (from WG_MVR_<TYPE> / NRR_<TYPE>, set by the host)
// before its kernels.

#define MVR_BEGIN(QB, XT, XINIT)                                                                               \
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
    const int nb  = ta.ne0 >> 5;                                                                      \
    const int ngrp = (M + 31) >> 5;                                                                   \
    const int gb  = tile * NRR;                                                                       \
    global const uchar * dbase = abase + M * nb * (QB);                                               \
    global const uint4 * qp[NRR];                                                                     \
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
        qp[j]  = (global const uint4 *)(abase + gj * (32 * nb * (QB))) + lc + sp * ((QB) / 16) * rs;  \
        dp[j]  = dbase + (gj * 32 * nb + sp * rs + lc) * 2;                                           \
        acc[j] = 0.0f;                                                                                \
    }                                                                                                 \
    XT const float4 * xp = (XINIT) + sp * 8;

#define MVR_END                                                                                       \
    local float red[WG_MVR * NRR];                                                                     \
    if (WG_MVR > 32) {                                                                                 \
        for (int j = 0; j < NRR; ++j) {                                                               \
            red[(sp * NRR + j) * 32 + lane] = acc[j];                                                 \
        }                                                                                             \
        barrier(CLK_LOCAL_MEM_FENCE);                                                                 \
        if (sp == 0) {                                                                                \
            for (int j = 0; j < NRR; ++j) {                                                           \
                float t = acc[j];                                                                     \
                for (int s2 = 1; s2 < NWR; ++s2) {                                             \
                    t += red[(s2 * NRR + j) * 32 + lane];                                             \
                }                                                                                     \
                acc[j] = t;                                                                           \
            }                                                                                         \
        }                                                                                             \
    }                                                                                                 \
    if (sp == 0) {                                                                                    \
        for (int j = 0; j < NRR; ++j) {                                                               \
            const int row = gr[j] * 32 + lane;                                                        \
            if (gr[j] < ngrp && row < M) {                                                            \
                ST_F32(pc + tc.off + row * tc.nb0 + i1 * tc.nb1 + i2 * tc.nb2 + i3 * tc.nb3, acc[j]); \
            }                                                                                         \
        }                                                                                             \
    }


// the loops are instantiated twice: RS = 32 (constant offsets) for the groups of 32 rows, RS = rsz[j] for the last group
#define MVR_Q8_LOOP(RS)                                                                               \
    for (int b = sp; b < nb; b += NWR) {                                                              \
        const float4 x0 = XLO(0), x1 = XLO(1), x2 = XLO(2), x3 = XLO(3), x4 = XLO(4), x5 = XLO(5), x6 = XLO(6), x7 = XLO(7); \
        for (int j = 0; j < NRR; ++j) {                                                               \
            const uint4 a = qp[j][0];                                                                 \
            const uint4 c = qp[j][RS(j)];                                                             \
            const float d = vload_half(0, (global const half *)dp[j]);                                \
            const float s0 = dot(f4_u8(a.x), x0) + dot(f4_u8(a.y), x1);                               \
            const float s1 = dot(f4_u8(a.z), x2) + dot(f4_u8(a.w), x3);                               \
            const float s2 = dot(f4_u8(c.x), x4) + dot(f4_u8(c.y), x5);                               \
            const float s3 = dot(f4_u8(c.z), x6) + dot(f4_u8(c.w), x7);                               \
            acc[j] = CL11_FMA(d, (s0 + s1) + (s2 + s3), acc[j]);                                           \
            qp[j] += NWR * 2 * RS(j);                                                                 \
            dp[j] += NWR * 2 * RS(j);                                                                 \
        }                                                                                             \
        xp += NWR * 8;                                                                                \
    }

#define MVR_Q4_LOOP(RS)                                                                               \
    for (int b = sp; b < nb; b += NWR) {                                                              \
        const float4 x0 = XLO(0), x1 = XLO(1), x2 = XLO(2), x3 = XLO(3), x4 = XLO(4), x5 = XLO(5), x6 = XLO(6), x7 = XLO(7); \
        for (int j = 0; j < NRR; ++j) {                                                               \
            const uint4 a = qp[j][0];                                                                 \
            const float d = vload_half(0, (global const half *)dp[j]);                                \
            const float s0 = dot(f4_n0(a.x), x0) + dot(f4_n0(a.y), x1);                               \
            const float s1 = dot(f4_n0(a.z), x2) + dot(f4_n0(a.w), x3);                               \
            const float s2 = dot(f4_n1(a.x), x4) + dot(f4_n1(a.y), x5);                               \
            const float s3 = dot(f4_n1(a.z), x6) + dot(f4_n1(a.w), x7);                               \
            acc[j] = CL11_FMA(d, (s0 + s1) + (s2 + s3), acc[j]);                                           \
            qp[j] += NWR * RS(j);                                                                     \
            dp[j] += NWR * 2 * RS(j);                                                                 \
        }                                                                                             \
        xp += NWR * 8;                                                                                \
    }

// ---------------------------------------------------------------------------------------------------------------------------
// Fermi variant ("denormal products", cfg.mvr_dn, DN_<type>; docs/CL11.md): a nibble or byte masked in place is the float n * 2^(p - 149)
// (p = bit position, bits 0..23: a denormal or one of the smallest normal numbers, exact, and the FFMA of Fermi handles denormals at full
// speed), so the quants need no conversion: one AND and one FFMA per weight (the PRMT of the normal kernels is a 16-per-clock instruction on
// Fermi). The byte 3 of a word (bits 24..31) is shifted down first. src1 is prepared once per mat-vec by k_xprep (xs = x * 2^(DN_E - p) for
// the bit position p of the weight that multiplies x, so every product is n * x * 2^(DN_E - 149), and the sums of x over groups of 16
// weights for the offsets: Q4_0 -8, Q6_K -32), the kernels read both from constant memory (the k_mvrc_* kernels; the general k_mvr_* kernels
// keep the normal arithmetic). Scratch layout: xs[K] floats, then sums[K / 16].
#define P2(k) as_float((uint)(127 + (k)) << 23)
#define DN_E 100
#define DN_SCL4 ((float4)(P2(DN_E), P2(DN_E - 8), P2(DN_E - 16), P2(DN_E)))                     // low nibble / byte i of a word
#define DN_SCH4 ((float4)(P2(DN_E - 4), P2(DN_E - 12), P2(DN_E - 20), P2(DN_E - 4)))            // high nibble of byte i

#if !defined(CL11_NO_BASE) && ((defined(DN_Q4_0) && DN_Q4_0) || (defined(DN_Q4_K) && DN_Q4_K) || (defined(DN_Q6_K) && DN_Q6_K))
// one work-item per group of 16 weights; hbit: the float4 index bit that selects the pattern of the high nibbles (Q4_0: 4, Q4_K: 8, none: 0)
kernel void k_xprep(global uchar * pb, tdesc tb, global float * dst, int hbit) {
    const int g = LGID() * (int)get_local_size(0) + (int)get_local_id(0);
    if (g >= (tb.ne0 >> 4)) return;
    global const float4 * x = (global const float4 *)(pb + tb.off) + 4 * g;
    global float4 * d = (global float4 *)dst + 4 * g;
    float4 s = (float4)(0.0f);
    for (int q = 0; q < 4; ++q) {
        const float4 v = x[q];
        s += v;
        d[q] = v * (((4 * g + q) & hbit) != 0 ? DN_SCH4 : DN_SCL4);
    }
    dst[tb.ne0 + g] = (s.x + s.y) + (s.z + s.w);
}
#endif

#define DN_Q4_WORD(W, I)                                                                              \
    {                                                                                                 \
        const float4 xl = XLO(I), xh = XLO((I) + 4);                                                  \
        for (int j = 0; j < NRR; ++j) {                                                               \
            const uint w = a[j].W;                                                                    \
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

// psp: the sums of x of the groups of 16 weights (2 per block)
#define MVR_Q4_DN_LOOP(RS)                                                                            \
    for (int b = sp; b < nb; b += NWR) {                                                              \
        uint4 a[NRR];                                                                                 \
        float sl[NRR], sh[NRR], dd[NRR];                                                              \
        for (int j = 0; j < NRR; ++j) {                                                               \
            a[j]  = qp[j][0];                                                                         \
            dd[j] = vload_half(0, (global const half *)dp[j]);                                        \
            sl[j] = 0.0f;                                                                             \
            sh[j] = 0.0f;                                                                             \
        }                                                                                             \
        DN_Q4_WORD(x, 0) DN_Q4_WORD(y, 1) DN_Q4_WORD(z, 2) DN_Q4_WORD(w, 3)                           \
        const float sx = psp[0] + psp[1];                                                             \
        for (int j = 0; j < NRR; ++j) {                                                               \
            const float u = fma(-P2(DN_E - 146), sx, sl[j] + sh[j]);                                  \
            acc[j] = fma(dd[j] * P2(149 - DN_E), u, acc[j]);                                          \
            qp[j] += NWR * RS(j);                                                                     \
            dp[j] += NWR * 2 * RS(j);                                                                 \
        }                                                                                             \
        xp += NWR * 8;                                                                                \
        psp += NWR * 2;                                                                               \
    }

// loads of src1: through the read-only data cache (ld.global.nc, sm_35 and newer, host: CL11_NV_LDG) which has its own path and
// relieves the load / store unit of the loads of the weights (faster for the K-quants, slower for Q4_0 / Q8_0: LDG_<type>, set by
// the host); XLO(i) is the i-th float4 of the current block of src1 (xp)
float4 ldg_f4(global const float4 * p) {
#ifdef CL11_NV_LDG
    float a, b, c, d;
    asm("ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(a), "=f"(b), "=f"(c), "=f"(d) : "l"(p));
    return (float4)(a, b, c, d);
#else
    return *p;
#endif
}
#define XLO_G(i) (XLDG ? ldg_f4(xp + (i)) : xp[i])
#define XLO_C(i) xp[i]

#define RS_FULL(j)  32
#define RS_VAR(j)   rsz[j]

// src1 (x) through the caches (any column / batch), or, for a single column, in constant memory: the host passes a sub-buffer
// that starts at x (at most 64 KiB), the loads of x then do not use the load / store units
#define MVR_ARGS_G global uchar * pa, tdesc ta, global uchar * pb, tdesc tb, global uchar * pc, tdesc tc, bcast bc
#define MVR_XG     ((global const float4 *)(pb + tb.off + i1 * tb.nb1 + i2 * tb.nb2 + i3 * tb.nb3))
#define MVR_ARGS_C global uchar * pa, tdesc ta, constant float4 * px, global uchar * pc, tdesc tc, bcast bc
#define MVR_XC     px

#ifndef CL11_NO_BASE
#define WG_MVR WG_MVR_Q8_0
#define XLDG   LDG_Q8_0
#define NRR    NRR_Q8_0
#define NWR    (WG_MVR / 32)
#define XLO XLO_G
kernel void k_mvr_q8_0(MVR_ARGS_G) {
    MVR_BEGIN(32, global, MVR_XG)
    if (full) { MVR_Q8_LOOP(RS_FULL) } else { MVR_Q8_LOOP(RS_VAR) }
    MVR_END
}

#undef XLO
#define XLO XLO_C
kernel void k_mvrc_q8_0(MVR_ARGS_C) {
    MVR_BEGIN(32, constant, MVR_XC)
    if (full) { MVR_Q8_LOOP(RS_FULL) } else { MVR_Q8_LOOP(RS_VAR) }
    MVR_END
}
#undef XLO
#undef XLDG
#undef NWR
#undef NRR
#undef WG_MVR

#define WG_MVR WG_MVR_Q4_0
#define XLDG   LDG_Q4_0
#define NRR    NRR_Q4_0
#define NWR    (WG_MVR / 32)
#define XLO XLO_G
kernel void k_mvr_q4_0(MVR_ARGS_G) {
    MVR_BEGIN(16, global, MVR_XG)
    if (full) { MVR_Q4_LOOP(RS_FULL) } else { MVR_Q4_LOOP(RS_VAR) }
    MVR_END
}

#undef XLO
#define XLO XLO_C
#if defined(DN_Q4_0) && DN_Q4_0
kernel void k_mvrc_q4_0(MVR_ARGS_C) {
    MVR_BEGIN(16, constant, MVR_XC)
    constant const float * psp = (constant const float *)px + ta.ne0 + sp * 2;
    if (full) { MVR_Q4_DN_LOOP(RS_FULL) } else { MVR_Q4_DN_LOOP(RS_VAR) }
    MVR_END
}
#else
kernel void k_mvrc_q4_0(MVR_ARGS_C) {
    MVR_BEGIN(16, constant, MVR_XC)
    if (full) { MVR_Q4_LOOP(RS_FULL) } else { MVR_Q4_LOOP(RS_VAR) }
    MVR_END
}
#endif
#undef XLO
#undef XLDG
#undef NWR
#undef NRR
#undef WG_MVR
#endif // CL11_NO_BASE

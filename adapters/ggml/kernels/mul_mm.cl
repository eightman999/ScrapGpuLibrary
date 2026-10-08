#if !defined(CL11_PROG_K) && !defined(CL11_NO_MM)
// CL11 backend: MUL_MAT with a wide src1 (mat-mat), tiled through local memory.
//
// A work-group of TXN x TYN = (GT/2) x (MMBN*GT/4) work-items computes a (4*GT) x (MMBN*GT) tile of dst (m = src0 row, n = src1 row),
// every work-item accumulates 8 (rows, two float4 of the tile) x 4 (columns, one float4) values, so one k step needs 3
// 128-bit local memory loads for 32 FMAs. The k dimension is processed in steps of BK = 16; src0 is dequantized while it is
// copied to local memory (the repacked Q4_0 / Q8_0 weights: one 16-byte unit per row and step, coalesced over the rows).
//   As[k][m]  (BM + 4 floats per k), Bs[k][n]  (BN + 4 floats per k, the padding spreads the stores over the banks)
// The addresses of the tiles of a work-item are computed once, before the loop over k.

#define BM (4 * GT)
#define BN (MMBN * GT)            // MMBN: 4 or 8 (host: cl11_config.mm_bn)
#define BK 16
#define BMS (BM + 4)          // row stride of As: keeps the float4 alignment and spreads the generic stores over the banks
#define TXN (GT / 2)
#define TYN (MMBN * GT / 4)
#define NT  (TXN * TYN)

// items of the tiles per work-item: A: 8 weights (half of a 16-byte unit) of a row, B: 4 consecutive k of a row of src1
#define AIT ((BM * 2 + NT - 1) / NT)
#define BIT ((BN * (BK / 4) + NT - 1) / NT)

// ---- A tile ----
// generic: the types that are dequantized element by element
#define A_PREP_GENERIC(AEL)
#define A_STAGE_GENERIC(AEL)                                                                                   \
    for (int e = lid; e < BM * BK; e += NT) {                                                                  \
        const int kk = e % BK;                                                                                 \
        const int mm = e / BK;                                                                                 \
        const int m = m0 + mm;                                                                                 \
        const int k = k0 + kk;                                                                                 \
        As[kk * BMS + mm] = (m < tc.ne0 && k < K) ? AEL(abase, m, k) : 0.0f;                                    \
    }

// repacked Q8_0: 16 weights = one 16-byte unit per row and step; the unit of step s is at (s * rs + lane) * 16 of the group
#define A_PREP_Q8_0(AEL)                                                                                       \
    global const uchar * aq[AIT];                                                                              \
    global const uchar * ad[AIT];                                                                              \
    int  ars[AIT];                                                                                             \
    int  aso[AIT];                                                                                             \
    bool aval[AIT];                                                                                            \
    for (int i = 0; i < AIT; ++i) {                                                                            \
        const int e = lid + i * NT;                                                                            \
        const int mm = e % BM;                                                                                 \
        const int m = m0 + mm;                                                                                 \
        aval[i] = e < BM * 2 && m < tc.ne0;                                                                    \
        const int mc = aval[i] ? m : 0;                                                                        \
        const int g = mc >> 5, lane = mc & 31;                                                                 \
        ars[i] = min(32, tc.ne0 - (g << 5));                                                                   \
        aso[i] = (e / BM) * 8;                                                                                 \
        aq[i] = abase + g * (32 * nb * 32) + (lane << 4) + aso[i];                                             \
        ad[i] = abase + tc.ne0 * nb * 32 + ((g * 32 * nb + lane) << 1);                                        \
    }
#define A_STAGE_Q8_0(AEL)                                                                                      \
    for (int i = 0; i < AIT; ++i) {                                                                            \
        const int e = lid + i * NT;                                                                            \
        if (e < BM * 2) {                                                                                      \
            float4 v0 = (float4)(0.0f), v1 = (float4)(0.0f);                                                   \
            if (aval[i]) {                                                                                     \
                const uint2 q = *(global const uint2 *)(aq[i] + (k0 >> 4) * (ars[i] << 4));                    \
                const float d = vload_half(0, (global const half *)(ad[i] + (k0 >> 5) * (ars[i] << 1)));       \
                v0 = f4_u8(q.x) * d;                                                                           \
                v1 = f4_u8(q.y) * d;                                                                           \
            }                                                                                                  \
            local float * o = As + aso[i] * BMS + e % BM;                                                       \
            o[0 * BMS] = v0.x; o[1 * BMS] = v0.y; o[2 * BMS] = v0.z; o[3 * BMS] = v0.w;                            \
            o[4 * BMS] = v1.x; o[5 * BMS] = v1.y; o[6 * BMS] = v1.z; o[7 * BMS] = v1.w;                            \
        }                                                                                                      \
    }

// repacked Q4_0: the 16 bytes of a block hold the weights 0..15 (low nibbles) and 16..31 (high nibbles), two steps per block
#define A_PREP_Q4_0(AEL)                                                                                       \
    global const uchar * aq[AIT];                                                                              \
    global const uchar * ad[AIT];                                                                              \
    int  ars[AIT];                                                                                             \
    int  aso[AIT];                                                                                             \
    bool aval[AIT];                                                                                            \
    for (int i = 0; i < AIT; ++i) {                                                                            \
        const int e = lid + i * NT;                                                                            \
        const int mm = e % BM;                                                                                 \
        const int m = m0 + mm;                                                                                 \
        aval[i] = e < BM * 2 && m < tc.ne0;                                                                    \
        const int mc = aval[i] ? m : 0;                                                                        \
        const int g = mc >> 5, lane = mc & 31;                                                                 \
        ars[i] = min(32, tc.ne0 - (g << 5));                                                                   \
        aso[i] = (e / BM) * 8;                                                                                 \
        aq[i] = abase + g * (32 * nb * 16) + (lane << 4) + aso[i];                                             \
        ad[i] = abase + tc.ne0 * nb * 16 + ((g * 32 * nb + lane) << 1);                                        \
    }
#define A_STAGE_Q4_0(AEL)                                                                                      \
    for (int i = 0; i < AIT; ++i) {                                                                            \
        const int e = lid + i * NT;                                                                            \
        if (e < BM * 2) {                                                                                      \
            float4 v0 = (float4)(0.0f), v1 = (float4)(0.0f);                                                   \
            if (aval[i]) {                                                                                     \
                const uint2 q = *(global const uint2 *)(aq[i] + (k0 >> 5) * (ars[i] << 4));                    \
                const float d = vload_half(0, (global const half *)(ad[i] + (k0 >> 5) * (ars[i] << 1)));       \
                if (k0 & 16) {                                                                                 \
                    v0 = f4_n1(q.x) * d;                                                                       \
                    v1 = f4_n1(q.y) * d;                                                                       \
                } else {                                                                                       \
                    v0 = f4_n0(q.x) * d;                                                                       \
                    v1 = f4_n0(q.y) * d;                                                                       \
                }                                                                                              \
            }                                                                                                  \
            local float * o = As + aso[i] * BMS + e % BM;                                                       \
            o[0 * BMS] = v0.x; o[1 * BMS] = v0.y; o[2 * BMS] = v0.z; o[3 * BMS] = v0.w;                            \
            o[4 * BMS] = v1.x; o[5 * BMS] = v1.y; o[6 * BMS] = v1.z; o[7 * BMS] = v1.w;                            \
        }                                                                                                      \
    }

// repacked Q4_K: a step of 16 weights is half a sub-block of 32 (low or high nibbles of 32 bytes = 2 units of 16 bytes of the quants);
// step s (0..15) of the super-block b: sub-block js = s / 2, chunk c = js / 2, nibble js % 2, unit 2c + s % 2 (see mul_mv_k.cl).
// The header (d, dmin, scales) of the super-block is kept in registers (hdr, dd, dm), reloaded every 256 weights.
#define A_PREP_Q4_K(AEL)                                                                                       \
    global const uchar * aq[AIT];                                                                              \
    global const uchar * ah[AIT];                                                                              \
    int  ars[AIT];                                                                                             \
    int  aso[AIT];                                                                                             \
    bool aval[AIT];                                                                                            \
    uint4 ahd[AIT];                                                                                            \
    float add[AIT], adm[AIT];                                                                                  \
    const int nbk = K >> 8;                                                                                    \
    for (int i = 0; i < AIT; ++i) {                                                                            \
        const int e = lid + i * NT;                                                                            \
        const int mm = e % BM;                                                                                 \
        const int m = m0 + mm;                                                                                 \
        aval[i] = e < BM * 2 && m < tc.ne0;                                                                    \
        const int mc = aval[i] ? m : 0;                                                                        \
        const int g = mc >> 5, lane = mc & 31;                                                                 \
        ars[i] = min(32, tc.ne0 - (g << 5));                                                                   \
        aso[i] = (e / BM) * 8;                                                                                 \
        aq[i] = abase + g * (32 * nbk * 128) + (lane << 4) + aso[i];                                           \
        ah[i] = abase + tc.ne0 * nbk * 128 + ((g * 32 * nbk) << 4) + (lane << 4);                              \
        ahd[i] = (uint4)(0u);                                                                                  \
        add[i] = 0.0f;                                                                                         \
        adm[i] = 0.0f;                                                                                         \
    }
#define A_STAGE_Q4_K(AEL)                                                                                      \
    {                                                                                                          \
        const int kb = k0 >> 8, ks = (k0 >> 4) & 15, js = ks >> 1;                                             \
        const int unit = 2 * (js >> 1) + (ks & 1);                                                             \
        for (int i = 0; i < AIT; ++i) {                                                                        \
            const int e = lid + i * NT;                                                                        \
            if (e < BM * 2) {                                                                                  \
                float4 v0 = (float4)(0.0f), v1 = (float4)(0.0f);                                               \
                if (aval[i]) {                                                                                 \
                    if (ks == 0) {                                                                             \
                        global const uchar * hp = ah[i] + kb * (ars[i] << 4);                                  \
                        ahd[i] = *(global const uint4 *)hp;                                                    \
                        add[i] = vload_half(0, (global const half *)hp);                                       \
                        adm[i] = vload_half(1, (global const half *)hp);                                       \
                    }                                                                                          \
                    const uint2 q = *(global const uint2 *)(aq[i] + (kb * 8 + unit) * (ars[i] << 4));          \
                    float scl, mnv;                                                                            \
                    if (js < 4) {                                                                              \
                        scl = (float)((ahd[i].y >> (8 * js)) & 63u);                                           \
                        mnv = (float)((ahd[i].z >> (8 * js)) & 63u);                                           \
                    } else {                                                                                   \
                        const int sh = 8 * (js - 4);                                                           \
                        scl = (float)(((ahd[i].w >> sh) & 0xFu) | (((ahd[i].y >> (sh + 6)) & 3u) << 4));       \
                        mnv = (float)(((ahd[i].w >> (sh + 4)) & 0xFu) | (((ahd[i].z >> (sh + 6)) & 3u) << 4)); \
                    }                                                                                          \
                    const float dsc = add[i] * scl, dmn = adm[i] * mnv;                                        \
                    if (js & 1) {                                                                              \
                        v0 = CL11_FMA(f4_hi4(q.x), (float4)(dsc), (float4)(-dmn));                                  \
                        v1 = CL11_FMA(f4_hi4(q.y), (float4)(dsc), (float4)(-dmn));                                  \
                    } else {                                                                                   \
                        v0 = CL11_FMA(f4_lo4(q.x), (float4)(dsc), (float4)(-dmn));                                  \
                        v1 = CL11_FMA(f4_lo4(q.y), (float4)(dsc), (float4)(-dmn));                                  \
                    }                                                                                          \
                }                                                                                              \
                local float * o = As + aso[i] * BMS + e % BM;                                                   \
                o[0 * BMS] = v0.x; o[1 * BMS] = v0.y; o[2 * BMS] = v0.z; o[3 * BMS] = v0.w;                    \
                o[4 * BMS] = v1.x; o[5 * BMS] = v1.y; o[6 * BMS] = v1.z; o[7 * BMS] = v1.w;                    \
            }                                                                                                  \
        }                                                                                                      \
    }

// repacked Q6_K: a step of 16 weights (s = 0..15 of the super-block: h = s / 8, quadrant qd = (s / 2) % 4, l = 16 (s % 2) + 0..15)
// is 16 bytes of ql (unit 4h + 2 (qd % 2) + s % 2), 16 bytes of qh (unit 8 + 2h + s % 2) and one scale (byte h * 8 + s % 2 + 2 qd).
#define A_PREP_Q6_K(AEL)                                                                                       \
    global const uchar * aq[AIT];                                                                              \
    global const uchar * as_[AIT];                                                                             \
    global const uchar * ad[AIT];                                                                              \
    int  ars[AIT];                                                                                             \
    int  aso[AIT];                                                                                             \
    bool aval[AIT];                                                                                            \
    float add[AIT];                                                                                            \
    const int nbk = K >> 8;                                                                                    \
    for (int i = 0; i < AIT; ++i) {                                                                            \
        const int e = lid + i * NT;                                                                            \
        const int mm = e % BM;                                                                                 \
        const int m = m0 + mm;                                                                                 \
        aval[i] = e < BM * 2 && m < tc.ne0;                                                                    \
        const int mc = aval[i] ? m : 0;                                                                        \
        const int g = mc >> 5, lane = mc & 31;                                                                 \
        ars[i] = min(32, tc.ne0 - (g << 5));                                                                   \
        aso[i] = (e / BM) * 8;                                                                                 \
        aq[i] = abase + g * (32 * nbk * 192) + (lane << 4) + aso[i];                                           \
        as_[i] = abase + tc.ne0 * nbk * 192 + ((g * 32 * nbk) << 4) + (lane << 4);                              \
        ad[i] = abase + tc.ne0 * nbk * 208 + ((g * 32 * nbk + lane) << 1);                                     \
        add[i] = 0.0f;                                                                                         \
    }
#define A_STAGE_Q6_K(AEL)                                                                                      \
    {                                                                                                          \
        const int kb = k0 >> 8, ks = (k0 >> 4) & 15;                                                           \
        const int hh = ks >> 3, qd = (ks >> 1) & 3, lh = ks & 1;                                               \
        const int ulq = 4 * hh + 2 * (qd & 1) + lh, ulh = 8 + 2 * hh + lh;                                     \
        for (int i = 0; i < AIT; ++i) {                                                                        \
            const int e = lid + i * NT;                                                                        \
            if (e < BM * 2) {                                                                                  \
                float4 v0 = (float4)(0.0f), v1 = (float4)(0.0f);                                               \
                if (aval[i]) {                                                                                 \
                    if (ks == 0) {                                                                             \
                        add[i] = vload_half(0, (global const half *)(ad[i] + kb * (ars[i] << 1)));             \
                    }                                                                                          \
                    const uint2 qa = *(global const uint2 *)(aq[i] + (kb * 12 + ulq) * (ars[i] << 4));         \
                    const uint2 qb = *(global const uint2 *)(aq[i] + (kb * 12 + ulh) * (ars[i] << 4));         \
                    const float sc = (float)((int)as_[i][kb * (ars[i] << 4) + hh * 8 + lh + 2 * qd] - 128);          \
                    const float ds = add[i] * sc;                                                              \
                    const uint x0 = ((qa.x >> (4 * (qd >> 1))) & 0x0F0F0F0Fu) | (((qb.x >> (2 * qd)) & 0x03030303u) << 4); \
                    const uint x1 = ((qa.y >> (4 * (qd >> 1))) & 0x0F0F0F0Fu) | (((qb.y >> (2 * qd)) & 0x03030303u) << 4); \
                    v0 = f4_q6(x0) * ds;                                                                       \
                    v1 = f4_q6(x1) * ds;                                                                       \
                }                                                                                              \
                local float * o = As + aso[i] * BMS + e % BM;                                                   \
                o[0 * BMS] = v0.x; o[1 * BMS] = v0.y; o[2 * BMS] = v0.z; o[3 * BMS] = v0.w;                    \
                o[4 * BMS] = v1.x; o[5 * BMS] = v1.y; o[6 * BMS] = v1.z; o[7 * BMS] = v1.w;                    \
            }                                                                                                  \
        }                                                                                                      \
    }

// ---- B tile: src1 (F32), BN rows x BK elements; float4 loads when the rows are 16-byte aligned and K is a multiple of 4 ----
#define B_PREP                                                                                                 \
    global const uchar * bp[BIT];                                                                              \
    int  bso[BIT];                                                                                             \
    int  bk4[BIT];                                                                                             \
    bool bval[BIT];                                                                                            \
    for (int i = 0; i < BIT; ++i) {                                                                            \
        const int e = lid + i * NT;                                                                            \
        const int kq = e % (BK / 4);                                                                           \
        const int nn = e / (BK / 4);                                                                           \
        const int n = n0 + nn;                                                                                 \
        bval[i] = e < BN * (BK / 4) && n < tc.ne1;                                                             \
        bp[i] = bbase + (bval[i] ? n : 0) * tb.nb1 + kq * 16;                                                  \
        bso[i] = (4 * kq) * (BN + 4) + nn;                                                                     \
        bk4[i] = 4 * kq;                                                                                       \
    }
#define B_STAGE                                                                                                \
    if (bvec) {                                                                                                \
        for (int i = 0; i < BIT; ++i) {                                                                        \
            float4 v = (float4)(0.0f);                                                                         \
            if (bval[i] && k0 + bk4[i] < K) {                                                                      \
                v = *(global const float4 *)(bp[i] + k0 * 4);                                                  \
            }                                                                                                  \
            local float * o = Bs + bso[i];                                                                     \
            o[0 * (BN + 4)] = v.x; o[1 * (BN + 4)] = v.y; o[2 * (BN + 4)] = v.z; o[3 * (BN + 4)] = v.w;        \
        }                                                                                                      \
    } else {                                                                                                   \
        for (int e = lid; e < BN * BK; e += NT) {                                                              \
            const int kk = e % BK;                                                                             \
            const int nn = e / BK;                                                                             \
            const int n = n0 + nn;                                                                             \
            const int k = k0 + kk;                                                                             \
            Bs[kk * (BN + 4) + nn] = (n < tc.ne1 && k < K) ? ((global const float *)(bbase + n * tb.nb1))[k] : 0.0f; \
        }                                                                                                      \
    }

#define DEF_MM(NAME, AEL, A_PREP, A_STAGE)                                                                     \
kernel void NAME(global uchar * pa, tdesc ta, global uchar * pb, tdesc tb, global uchar * pc, tdesc tc, bcast bc) { \
    local float4 As4[BK * (BMS / 4)];                                                                           \
    local float4 Bs4[BK * (BN + 4) / 4];                                                                       \
    local float * As = (local float *)As4;                                                                     \
    local float * Bs = (local float *)Bs4;                                                                     \
    const int lid = (int)get_local_id(0);                                                                      \
    const int tx = lid % TXN;                                                                                  \
    const int ty = lid / TXN;                                                                                  \
    int g = LGID();                                                                                            \
    if (g >= bc.tiles * bc.tiles2 * tc.ne2 * tc.ne3) return; /* uniform for the whole group */                  \
    const int gq  = FDIV_RAW(g, bc.mt, bc.lt);                                                                 \
    const int bm  = g - gq * bc.tiles;                                                                         \
    const int gq2 = FDIV_RAW(gq, bc.mt2, bc.lt2);                                                              \
    const int bn  = gq - gq2 * bc.tiles2;                                                                      \
    const int batch = gq2;                                                                                     \
    const int i3 = FDIV(batch, tc, 2);                                                                         \
    const int i2 = batch - i3 * tc.ne2;                                                                        \
    const int i02 = FDIV_RAW(i2, bc.m2, bc.l2);                                                                \
    const int i03 = FDIV_RAW(i3, bc.m3, bc.l3);                                                                \
    global const uchar * abase = pa + ta.off + i02 * ta.nb2 + i03 * ta.nb3;                                    \
    global const uchar * bbase = pb + tb.off + i2 * tb.nb2 + i3 * tb.nb3;                                      \
    const int m0 = bm * BM;                                                                                    \
    const int n0 = bn * BN;                                                                                    \
    const int K  = ta.ne0;                                                                                     \
    const int nb = K >> 5;                                                                                     \
    /* src1 rows can be read as float4: 16-byte aligned rows and K a multiple of 4 */                          \
    const bool bvec = ((tb.off | tb.nb1 | tb.nb2 | tb.nb3) & 15) == 0 && (K & 3) == 0;                         \
    A_PREP(AEL)                                                                                                \
    B_PREP                                                                                                     \
    float acc[8][4];                                                                                           \
    for (int i = 0; i < 8; ++i) {                                                                              \
        for (int j = 0; j < 4; ++j) {                                                                          \
            acc[i][j] = 0.0f;                                                                                  \
        }                                                                                                      \
    }                                                                                                          \
    for (int k0 = 0; k0 < K; k0 += BK) {                                                                       \
        A_STAGE(AEL)                                                                                           \
        B_STAGE                                                                                                \
        barrier(CLK_LOCAL_MEM_FENCE);                                                                          \
        for (int kk = 0; kk < BK; ++kk) {                                                                      \
            const float4 a0 = As4[kk * (BMS / 4) + tx];                                                         \
            const float4 a1 = As4[kk * (BMS / 4) + TXN + tx];                                                   \
            const float4 b  = Bs4[kk * ((BN + 4) / 4) + ty];                                                   \
            acc[0][0] = CL11_FMA(a0.x, b.x, acc[0][0]); acc[0][1] = CL11_FMA(a0.x, b.y, acc[0][1]);                      \
            acc[0][2] = CL11_FMA(a0.x, b.z, acc[0][2]); acc[0][3] = CL11_FMA(a0.x, b.w, acc[0][3]);                      \
            acc[1][0] = CL11_FMA(a0.y, b.x, acc[1][0]); acc[1][1] = CL11_FMA(a0.y, b.y, acc[1][1]);                      \
            acc[1][2] = CL11_FMA(a0.y, b.z, acc[1][2]); acc[1][3] = CL11_FMA(a0.y, b.w, acc[1][3]);                      \
            acc[2][0] = CL11_FMA(a0.z, b.x, acc[2][0]); acc[2][1] = CL11_FMA(a0.z, b.y, acc[2][1]);                      \
            acc[2][2] = CL11_FMA(a0.z, b.z, acc[2][2]); acc[2][3] = CL11_FMA(a0.z, b.w, acc[2][3]);                      \
            acc[3][0] = CL11_FMA(a0.w, b.x, acc[3][0]); acc[3][1] = CL11_FMA(a0.w, b.y, acc[3][1]);                      \
            acc[3][2] = CL11_FMA(a0.w, b.z, acc[3][2]); acc[3][3] = CL11_FMA(a0.w, b.w, acc[3][3]);                      \
            acc[4][0] = CL11_FMA(a1.x, b.x, acc[4][0]); acc[4][1] = CL11_FMA(a1.x, b.y, acc[4][1]);                      \
            acc[4][2] = CL11_FMA(a1.x, b.z, acc[4][2]); acc[4][3] = CL11_FMA(a1.x, b.w, acc[4][3]);                      \
            acc[5][0] = CL11_FMA(a1.y, b.x, acc[5][0]); acc[5][1] = CL11_FMA(a1.y, b.y, acc[5][1]);                      \
            acc[5][2] = CL11_FMA(a1.y, b.z, acc[5][2]); acc[5][3] = CL11_FMA(a1.y, b.w, acc[5][3]);                      \
            acc[6][0] = CL11_FMA(a1.z, b.x, acc[6][0]); acc[6][1] = CL11_FMA(a1.z, b.y, acc[6][1]);                      \
            acc[6][2] = CL11_FMA(a1.z, b.z, acc[6][2]); acc[6][3] = CL11_FMA(a1.z, b.w, acc[6][3]);                      \
            acc[7][0] = CL11_FMA(a1.w, b.x, acc[7][0]); acc[7][1] = CL11_FMA(a1.w, b.y, acc[7][1]);                      \
            acc[7][2] = CL11_FMA(a1.w, b.z, acc[7][2]); acc[7][3] = CL11_FMA(a1.w, b.w, acc[7][3]);                      \
        }                                                                                                      \
        barrier(CLK_LOCAL_MEM_FENCE);                                                                          \
    }                                                                                                          \
    for (int j = 0; j < 4; ++j) {                                                                              \
        const int n = n0 + 4 * ty + j;                                                                         \
        for (int i = 0; i < 8; ++i) {                                                                          \
            const int m = m0 + (i >> 2) * (BM / 2) + 4 * tx + (i & 3);                                         \
            if (m < tc.ne0 && n < tc.ne1) {                                                                    \
                ST_F32(pc + tc.off + m * tc.nb0 + n * tc.nb1 + i2 * tc.nb2 + i3 * tc.nb3, acc[i][j]);          \
            }                                                                                                  \
        }                                                                                                      \
    }                                                                                                          \
}

// element (m, k) of the slice of src0 that starts at abase
#define AEL_F32(a, m, k)   dq_f32((a) + (m) * ta.nb1, k)
#define AEL_F16(a, m, k)   dq_f16((a) + (m) * ta.nb1, k)
#define AEL_Q4_K(a, m, k)  dqr_q4_k(a, tc.ne0, K >> 8, m, k)
#define AEL_Q6_K(a, m, k)  dqr_q6_k(a, tc.ne0, K >> 8, m, k)

DEF_MM(k_mm_f32,  AEL_F32,  A_PREP_GENERIC, A_STAGE_GENERIC)
DEF_MM(k_mm_f16,  AEL_F16,  A_PREP_GENERIC, A_STAGE_GENERIC)
DEF_MM(k_mm_q4_0, AEL_F32,  A_PREP_Q4_0,    A_STAGE_Q4_0)
DEF_MM(k_mm_q8_0, AEL_F32,  A_PREP_Q8_0,    A_STAGE_Q8_0)
DEF_MM(k_mm_q4_k, AEL_Q4_K, A_PREP_Q4_K,    A_STAGE_Q4_K)
DEF_MM(k_mm_q6_k, AEL_Q6_K, A_PREP_Q6_K,    A_STAGE_Q6_K)
#endif // mat-mat kernels

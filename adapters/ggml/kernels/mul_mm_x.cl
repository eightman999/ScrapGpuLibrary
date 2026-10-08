#if !defined(CL11_PROG_K) && !defined(CL11_NO_MM) && defined(CL11_MMX)
// CL11 backend: MUL_MAT with a wide src1, parameterized tiled kernels ("mmx"), candidates for the Kepler GPUs (compute capability 3.x,
// up to 255 registers per work-item). The kernels of mul_mm.cl (8 x 4 values per work-item, BK = 16) stay the default everywhere else;
// these are selected by GGML_CL11_MM_VARIANT / GGML_CL11_MMX (see cl11_mmx_parse in ggml-cl11.cpp) and compiled only then.
//
// Parameters (compile time, set by the host):
//   MX_BM, MX_BN  tile of dst: BM rows of src0 x BN rows of src1
//   MX_RM, MX_CN  values per work-item: RM rows (8 or 16, float4 groups of rows) x CN columns (4 or 8); the work-group has
//                 (BM / RM) x (BN / CN) work-items. One k step needs RM/4 + CN/4 128-bit local memory loads for RM * CN FMAs
//                 (8 x 4: 3 loads / 32 FMAs, 8 x 8: 4 / 64, 16 x 4: 5 / 64, 16 x 8: 6 / 128)
//   MX_BK         k per step (16 or 32): the barriers per FMA are halved with 32
//   MX_KU         the loop over the k of a step is unrolled by this (1, 2, 4, 8, 16)
//   MX_PF         0: load, convert, store, barrier, compute, barrier (like mul_mm.cl)
//                 1: the global loads of the next step are issued before the compute of this step (registers hold the raw data),
//                    converted and stored to the local memory after it
//                 2: like 1 with two local memory buffers: one barrier per step
//   MX_LDG        src0 / src1 tile loads through ld.global.nc (NVIDIA sm_35+, CL11_NV_LDG)
//   MX_REQD       reqd_work_group_size(NT, 1, 1)
// Layout of the tiles: As[k][m] (BM + 4 floats per k), Bs[k][n] (BN + 4 floats per k); work-item (tx, ty) owns the rows
// jv * (4 * TXN) + 4 * tx + 0..3 (jv < RM / 4) and the columns jn * (4 * TYN) + 4 * ty + 0..3 (jn < CN / 4).
// The accumulators are scalars (c<row>_<col>) so that no compiler has to unroll loops to keep them in registers.

#define XBM  MX_BM
#define XBN  MX_BN
#define XRM  MX_RM
#define XCN  MX_CN
#define XBK  MX_BK
#define XTXN (XBM / XRM)
#define XTYN (XBN / XCN)
#define XNT  (XTXN * XTYN)
#define XAS  (XBM + 4)
#define XBS  (XBN + 4)
#define XSUB (XBK / 16)
#define XBUFSZ (XBK * (XAS + XBS))                 // floats of one buffer (A tile + B tile)
#if MX_PF == 2
#define XNBUF 2
#else
#define XNBUF 1
#endif
// items of the A tile: (row, half of a 16-byte unit = 8 weights, 16-weight sub-step of the step); of the B tile: (row of src1, 4 consecutive k)
#define XAITEMS (XBM * 2 * XSUB)
#define XAIT ((XAITEMS + XNT - 1) / XNT)
#define XBITEMS (XBN * (XBK / 4))
#define XBIT ((XBITEMS + XNT - 1) / XNT)

#if defined(MX_LDG) && MX_LDG && defined(CL11_NV_LDG)
uint2 mx_ldg_u2(global const uchar * p) {
    uint a, b;
    asm("ld.global.nc.v2.u32 {%0, %1}, [%2];" : "=r"(a), "=r"(b) : "l"(p));
    return (uint2)(a, b);
}
float4 mx_ldg_f4(global const uchar * p) {
    float a, b, c, d;
    asm("ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(a), "=f"(b), "=f"(c), "=f"(d) : "l"(p));
    return (float4)(a, b, c, d);
}
#define XLD_U2(p) mx_ldg_u2((global const uchar *)(p))
#define XLD_F4(p) mx_ldg_f4((global const uchar *)(p))
#else
#define XLD_U2(p) (*(global const uint2 *)(p))
#define XLD_F4(p) (*(global const float4 *)(p))
#endif

// 8 consecutive k of the row mm: two float4 -> 8 scalar stores (As is [k][m])
#define XST8(AS, kofs, mm, v0, v1)                                                                             \
    {                                                                                                          \
        local float * o = (AS) + (kofs) * XAS + (mm);                                                          \
        o[0 * XAS] = (v0).x; o[1 * XAS] = (v0).y; o[2 * XAS] = (v0).z; o[3 * XAS] = (v0).w;                    \
        o[4 * XAS] = (v1).x; o[5 * XAS] = (v1).y; o[6 * XAS] = (v1).z; o[7 * XAS] = (v1).w;                    \
    }

// per-item decode shared by the repacked types: row mm, half (0, 1) of the 16-byte unit, 16-weight sub-step of the step
#define X_ITEM_COMMON                                                                                          \
    const int e = lid + i * XNT;                                                                               \
    const int mm = e % XBM;                                                                                    \
    const int rest = e / XBM;                                                                                  \
    const int m = m0 + mm;                                                                                     \
    aval[i] = e < XAITEMS && m < tc.ne0;                                                                       \
    const int mc = aval[i] ? m : 0;                                                                            \
    const int g = mc >> 5, lane = mc & 31;                                                                     \
    ars[i] = min(32, tc.ne0 - (g << 5));                                                                       \
    asu[i] = (rest >> 1) * 16;                                                                                 \
    aso[i] = (rest & 1) * 8 + (rest >> 1) * 16;                                                                \
    amm[i] = mm;

#define X_A_VARS                                                                                               \
    global const uchar * aq[XAIT];                                                                             \
    int  ars[XAIT];                                                                                            \
    int  aso[XAIT];                                                                                            \
    int  asu[XAIT];                                                                                            \
    int  amm[XAIT];                                                                                            \
    bool aval[XAIT];                                                                                           \
    uint2 aqr[XAIT];

// ---- A tile: generic (F32 / F16): element by element, loaded when the tile is stored ----
#define X_A_DECL_GENERIC
#define X_A_LOAD_GENERIC(KK)
#define X_A_CONV_GENERIC(KK, AS, AEL)                                                                          \
    for (int e = lid; e < XBM * XBK; e += XNT) {                                                               \
        const int kk = e % XBK;                                                                                \
        const int mm = e / XBK;                                                                                \
        const int m = m0 + mm;                                                                                 \
        const int k = (KK) + kk;                                                                               \
        (AS)[kk * XAS + mm] = (m < tc.ne0 && k < K) ? AEL(abase, m, k) : 0.0f;                                 \
    }
#define XAEL_F32(a, m, k)  dq_f32((a) + (m) * ta.nb1, k)
#define XAEL_F16(a, m, k)  dq_f16((a) + (m) * ta.nb1, k)

// ---- repacked Q8_0: 16 weights = one 16-byte unit per row and 16-weight step; the unit of step s is at (s * rs + lane) * 16 of the group ----
#define X_A_DECL_Q8_0                                                                                          \
    X_A_VARS                                                                                                   \
    global const uchar * ad[XAIT];                                                                             \
    float adr[XAIT];                                                                                           \
    for (int i = 0; i < XAIT; ++i) {                                                                           \
        X_ITEM_COMMON                                                                                          \
        aq[i] = abase + g * (32 * nb * 32) + (lane << 4) + (rest & 1) * 8;                                     \
        ad[i] = abase + tc.ne0 * nb * 32 + ((g * 32 * nb + lane) << 1);                                        \
        aqr[i] = (uint2)(0u);                                                                                  \
        adr[i] = 0.0f;                                                                                         \
    }
#define X_A_LOAD_Q8_0(KK)                                                                                      \
    for (int i = 0; i < XAIT; ++i) {                                                                           \
        if (aval[i]) {                                                                                         \
            const int kx = (KK) + asu[i];                                                                      \
            aqr[i] = XLD_U2(aq[i] + (kx >> 4) * (ars[i] << 4));                                                \
            adr[i] = vload_half(0, (global const half *)(ad[i] + (kx >> 5) * (ars[i] << 1)));                  \
        }                                                                                                      \
    }
#define X_A_CONV_Q8_0(KK, AS, AEL)                                                                             \
    for (int i = 0; i < XAIT; ++i) {                                                                           \
        if ((XAITEMS == XAIT * XNT) || lid + i * XNT < XAITEMS) {                                              \
            const float4 v0 = f4_u8(aqr[i].x) * adr[i];                                                        \
            const float4 v1 = f4_u8(aqr[i].y) * adr[i];                                                        \
            XST8(AS, aso[i], amm[i], v0, v1)                                                                   \
        }                                                                                                      \
    }

// ---- repacked Q4_0: the 16 bytes of a block hold the weights 0..15 (low nibbles) and 16..31 (high nibbles) ----
#define X_A_DECL_Q4_0                                                                                          \
    X_A_VARS                                                                                                   \
    global const uchar * ad[XAIT];                                                                             \
    float adr[XAIT];                                                                                           \
    bool  ahi[XAIT];                                                                                           \
    for (int i = 0; i < XAIT; ++i) {                                                                           \
        X_ITEM_COMMON                                                                                          \
        aq[i] = abase + g * (32 * nb * 16) + (lane << 4) + (rest & 1) * 8;                                     \
        ad[i] = abase + tc.ne0 * nb * 16 + ((g * 32 * nb + lane) << 1);                                        \
        aqr[i] = (uint2)(0u);                                                                                  \
        adr[i] = 0.0f;                                                                                         \
        ahi[i] = false;                                                                                        \
    }
#define X_A_LOAD_Q4_0(KK)                                                                                      \
    for (int i = 0; i < XAIT; ++i) {                                                                           \
        if (aval[i]) {                                                                                         \
            const int kx = (KK) + asu[i];                                                                      \
            aqr[i] = XLD_U2(aq[i] + (kx >> 5) * (ars[i] << 4));                                                \
            adr[i] = vload_half(0, (global const half *)(ad[i] + (kx >> 5) * (ars[i] << 1)));                  \
            ahi[i] = (kx & 16) != 0;                                                                           \
        }                                                                                                      \
    }
#define X_A_CONV_Q4_0(KK, AS, AEL)                                                                             \
    for (int i = 0; i < XAIT; ++i) {                                                                           \
        if ((XAITEMS == XAIT * XNT) || lid + i * XNT < XAITEMS) {                                              \
            float4 v0, v1;                                                                                     \
            if (ahi[i]) {                                                                                      \
                v0 = f4_n1(aqr[i].x) * adr[i];                                                                 \
                v1 = f4_n1(aqr[i].y) * adr[i];                                                                 \
            } else {                                                                                           \
                v0 = f4_n0(aqr[i].x) * adr[i];                                                                 \
                v1 = f4_n0(aqr[i].y) * adr[i];                                                                 \
            }                                                                                                  \
            XST8(AS, aso[i], amm[i], v0, v1)                                                                   \
        }                                                                                                      \
    }

// ---- repacked Q4_K: a step of 16 weights is half a sub-block of 32; step s (0..15) of the super-block: sub-block js = s / 2,
// unit 2 (js / 2) + s % 2 (see mul_mm.cl). The header (d, dmin, scales) of the super-block is kept in registers, reloaded every 256 weights. ----
#define X_A_DECL_Q4_K                                                                                          \
    X_A_VARS                                                                                                   \
    global const uchar * ah[XAIT];                                                                             \
    uint4 ahd[XAIT];                                                                                           \
    float add[XAIT], adm[XAIT];                                                                                \
    const int nbk = K >> 8;                                                                                    \
    for (int i = 0; i < XAIT; ++i) {                                                                           \
        X_ITEM_COMMON                                                                                          \
        aq[i] = abase + g * (32 * nbk * 128) + (lane << 4) + (rest & 1) * 8;                                   \
        ah[i] = abase + tc.ne0 * nbk * 128 + ((g * 32 * nbk) << 4) + (lane << 4);                              \
        aqr[i] = (uint2)(0u);                                                                                  \
        ahd[i] = (uint4)(0u);                                                                                  \
        add[i] = 0.0f;                                                                                         \
        adm[i] = 0.0f;                                                                                         \
    }
#define X_A_LOAD_Q4_K(KK)                                                                                      \
    for (int i = 0; i < XAIT; ++i) {                                                                           \
        if (aval[i]) {                                                                                         \
            const int kx = (KK) + asu[i];                                                                      \
            const int kb = kx >> 8, ks = (kx >> 4) & 15, js = ks >> 1;                                         \
            const int unit = 2 * (js >> 1) + (ks & 1);                                                         \
            if (((KK) & 255) == 0) {                                                                           \
                global const uchar * hp = ah[i] + kb * (ars[i] << 4);                                          \
                ahd[i] = *(global const uint4 *)hp;                                                            \
                add[i] = vload_half(0, (global const half *)hp);                                               \
                adm[i] = vload_half(1, (global const half *)hp);                                               \
            }                                                                                                  \
            aqr[i] = XLD_U2(aq[i] + (kb * 8 + unit) * (ars[i] << 4));                                          \
        }                                                                                                      \
    }
#define X_A_CONV_Q4_K(KK, AS, AEL)                                                                             \
    for (int i = 0; i < XAIT; ++i) {                                                                           \
        if ((XAITEMS == XAIT * XNT) || lid + i * XNT < XAITEMS) {                                              \
            const int kx = (KK) + asu[i];                                                                      \
            const int ks = (kx >> 4) & 15, js = ks >> 1;                                                       \
            float scl, mnv;                                                                                    \
            if (js < 4) {                                                                                      \
                scl = (float)((ahd[i].y >> (8 * js)) & 63u);                                                   \
                mnv = (float)((ahd[i].z >> (8 * js)) & 63u);                                                   \
            } else {                                                                                           \
                const int sh = 8 * (js - 4);                                                                   \
                scl = (float)(((ahd[i].w >> sh) & 0xFu) | (((ahd[i].y >> (sh + 6)) & 3u) << 4));               \
                mnv = (float)(((ahd[i].w >> (sh + 4)) & 0xFu) | (((ahd[i].z >> (sh + 6)) & 3u) << 4));         \
            }                                                                                                  \
            const float dsc = add[i] * scl, dmn = adm[i] * mnv;                                                \
            float4 v0, v1;                                                                                     \
            if (js & 1) {                                                                                      \
                v0 = CL11_FMA(f4_hi4(aqr[i].x), (float4)(dsc), (float4)(-dmn));                                \
                v1 = CL11_FMA(f4_hi4(aqr[i].y), (float4)(dsc), (float4)(-dmn));                                \
            } else {                                                                                           \
                v0 = CL11_FMA(f4_lo4(aqr[i].x), (float4)(dsc), (float4)(-dmn));                                \
                v1 = CL11_FMA(f4_lo4(aqr[i].y), (float4)(dsc), (float4)(-dmn));                                \
            }                                                                                                  \
            if (!aval[i]) {                                                                                    \
                v0 = (float4)(0.0f);                                                                           \
                v1 = (float4)(0.0f);                                                                           \
            }                                                                                                  \
            XST8(AS, aso[i], amm[i], v0, v1)                                                                   \
        }                                                                                                      \
    }

// ---- repacked Q6_K: a step of 16 weights (s = 0..15 of the super-block: h = s / 8, quadrant qd = (s / 2) % 4, l = 16 (s % 2) + 0..15)
// is 16 bytes of ql (unit 4h + 2 (qd % 2) + s % 2), 16 bytes of qh (unit 8 + 2h + s % 2) and one scale (byte h * 8 + s % 2 + 2 qd). ----
#define X_A_DECL_Q6_K                                                                                          \
    X_A_VARS                                                                                                   \
    global const uchar * as_[XAIT];                                                                            \
    global const uchar * ad[XAIT];                                                                             \
    uint2 aqh[XAIT];                                                                                           \
    float asc[XAIT], add[XAIT];                                                                                \
    const int nbk = K >> 8;                                                                                    \
    for (int i = 0; i < XAIT; ++i) {                                                                           \
        X_ITEM_COMMON                                                                                          \
        aq[i] = abase + g * (32 * nbk * 192) + (lane << 4) + (rest & 1) * 8;                                   \
        as_[i] = abase + tc.ne0 * nbk * 192 + ((g * 32 * nbk) << 4) + (lane << 4);                             \
        ad[i] = abase + tc.ne0 * nbk * 208 + ((g * 32 * nbk + lane) << 1);                                     \
        aqr[i] = (uint2)(0u);                                                                                  \
        aqh[i] = (uint2)(0u);                                                                                  \
        asc[i] = 0.0f;                                                                                         \
        add[i] = 0.0f;                                                                                         \
    }
#define X_A_LOAD_Q6_K(KK)                                                                                      \
    for (int i = 0; i < XAIT; ++i) {                                                                           \
        if (aval[i]) {                                                                                         \
            const int kx = (KK) + asu[i];                                                                      \
            const int kb = kx >> 8, ks = (kx >> 4) & 15;                                                       \
            const int hh = ks >> 3, qd = (ks >> 1) & 3, lh = ks & 1;                                           \
            const int ulq = 4 * hh + 2 * (qd & 1) + lh, ulh = 8 + 2 * hh + lh;                                 \
            if (((KK) & 255) == 0) {                                                                           \
                add[i] = vload_half(0, (global const half *)(ad[i] + kb * (ars[i] << 1)));                     \
            }                                                                                                  \
            aqr[i] = XLD_U2(aq[i] + (kb * 12 + ulq) * (ars[i] << 4));                                          \
            aqh[i] = XLD_U2(aq[i] + (kb * 12 + ulh) * (ars[i] << 4));                                          \
            asc[i] = (float)((int)as_[i][kb * (ars[i] << 4) + hh * 8 + lh + 2 * qd] - 128);                    \
        }                                                                                                      \
    }
#define X_A_CONV_Q6_K(KK, AS, AEL)                                                                             \
    for (int i = 0; i < XAIT; ++i) {                                                                           \
        if ((XAITEMS == XAIT * XNT) || lid + i * XNT < XAITEMS) {                                              \
            const int kx = (KK) + asu[i];                                                                      \
            const int qd = (((kx >> 4) & 15) >> 1) & 3;                                                          \
            const float ds = add[i] * asc[i];                                                                  \
            const uint x0 = ((aqr[i].x >> (4 * (qd >> 1))) & 0x0F0F0F0Fu) | (((aqh[i].x >> (2 * qd)) & 0x03030303u) << 4); \
            const uint x1 = ((aqr[i].y >> (4 * (qd >> 1))) & 0x0F0F0F0Fu) | (((aqh[i].y >> (2 * qd)) & 0x03030303u) << 4); \
            float4 v0 = f4_q6(x0) * ds;                                                                        \
            float4 v1 = f4_q6(x1) * ds;                                                                        \
            if (!aval[i]) {                                                                                    \
                v0 = (float4)(0.0f);                                                                           \
                v1 = (float4)(0.0f);                                                                           \
            }                                                                                                  \
            XST8(AS, aso[i], amm[i], v0, v1)                                                                   \
        }                                                                                                      \
    }

// ---- B tile: src1 (F32), BN rows x BK elements; float4 loads when the rows are 16-byte aligned and K is a multiple of 4 ----
#define X_B_DECL                                                                                               \
    global const uchar * bp[XBIT];                                                                             \
    int    bso[XBIT];                                                                                          \
    int    bk4[XBIT];                                                                                          \
    bool   bval[XBIT];                                                                                         \
    float4 bvr[XBIT];                                                                                          \
    for (int i = 0; i < XBIT; ++i) {                                                                           \
        const int e = lid + i * XNT;                                                                           \
        const int kq = e % (XBK / 4);                                                                          \
        const int nn = e / (XBK / 4);                                                                          \
        const int n = n0 + nn;                                                                                 \
        bval[i] = e < XBITEMS && n < tc.ne1;                                                                   \
        bp[i] = bbase + (bval[i] ? n : 0) * tb.nb1 + kq * 16;                                                  \
        bso[i] = (4 * kq) * XBS + nn;                                                                          \
        bk4[i] = 4 * kq;                                                                                       \
        bvr[i] = (float4)(0.0f);                                                                               \
    }
#define X_B_LOAD(KK)                                                                                           \
    if (bvec) {                                                                                                \
        for (int i = 0; i < XBIT; ++i) {                                                                       \
            bvr[i] = (bval[i] && (KK) + bk4[i] < K) ? XLD_F4(bp[i] + (KK) * 4) : (float4)(0.0f);               \
        }                                                                                                      \
    }
#define X_B_STORE(KK, BS)                                                                                      \
    if (bvec) {                                                                                                \
        for (int i = 0; i < XBIT; ++i) {                                                                       \
            if ((XBITEMS == XBIT * XNT) || lid + i * XNT < XBITEMS) {                                          \
                local float * o = (BS) + bso[i];                                                               \
                o[0 * XBS] = bvr[i].x; o[1 * XBS] = bvr[i].y; o[2 * XBS] = bvr[i].z; o[3 * XBS] = bvr[i].w;    \
            }                                                                                                  \
        }                                                                                                      \
    } else {                                                                                                   \
        for (int e = lid; e < XBN * XBK; e += XNT) {                                                           \
            const int kk = e % XBK;                                                                            \
            const int nn = e / XBK;                                                                            \
            const int n = n0 + nn;                                                                             \
            const int k = (KK) + kk;                                                                           \
            (BS)[kk * XBS + nn] = (n < tc.ne1 && k < K) ? ((global const float *)(bbase + n * tb.nb1))[k] : 0.0f; \
        }                                                                                                      \
    }

// ---- registers: c<row>_<col> ----
#define X_ROWS(M)  M(0) M(1) M(2) M(3) M(4) M(5) M(6) M(7) X_ROWS_HI(M)
#if XRM == 16
#define X_ROWS_HI(M)  M(8) M(9) M(10) M(11) M(12) M(13) M(14) M(15)
#else
#define X_ROWS_HI(M)
#endif

#if XCN == 8
#define X_ACC_ROW(i)  float c##i##_0 = 0.0f, c##i##_1 = 0.0f, c##i##_2 = 0.0f, c##i##_3 = 0.0f, c##i##_4 = 0.0f, c##i##_5 = 0.0f, c##i##_6 = 0.0f, c##i##_7 = 0.0f;
#define X_FMA_ROW(i)                                                                                           \
    c##i##_0 = CL11_FMA(av[i], bv[0], c##i##_0); c##i##_1 = CL11_FMA(av[i], bv[1], c##i##_1);                  \
    c##i##_2 = CL11_FMA(av[i], bv[2], c##i##_2); c##i##_3 = CL11_FMA(av[i], bv[3], c##i##_3);                  \
    c##i##_4 = CL11_FMA(av[i], bv[4], c##i##_4); c##i##_5 = CL11_FMA(av[i], bv[5], c##i##_5);                  \
    c##i##_6 = CL11_FMA(av[i], bv[6], c##i##_6); c##i##_7 = CL11_FMA(av[i], bv[7], c##i##_7);
#define X_EPI_COLS(jv, r0, r1, r2, r3)                                                                         \
    X_EPI_C(jv, 0, c##r0##_0, c##r1##_0, c##r2##_0, c##r3##_0) X_EPI_C(jv, 1, c##r0##_1, c##r1##_1, c##r2##_1, c##r3##_1) \
    X_EPI_C(jv, 2, c##r0##_2, c##r1##_2, c##r2##_2, c##r3##_2) X_EPI_C(jv, 3, c##r0##_3, c##r1##_3, c##r2##_3, c##r3##_3) \
    X_EPI_C(jv, 4, c##r0##_4, c##r1##_4, c##r2##_4, c##r3##_4) X_EPI_C(jv, 5, c##r0##_5, c##r1##_5, c##r2##_5, c##r3##_5) \
    X_EPI_C(jv, 6, c##r0##_6, c##r1##_6, c##r2##_6, c##r3##_6) X_EPI_C(jv, 7, c##r0##_7, c##r1##_7, c##r2##_7, c##r3##_7)
#else
#define X_ACC_ROW(i)  float c##i##_0 = 0.0f, c##i##_1 = 0.0f, c##i##_2 = 0.0f, c##i##_3 = 0.0f;
#define X_FMA_ROW(i)                                                                                           \
    c##i##_0 = CL11_FMA(av[i], bv[0], c##i##_0); c##i##_1 = CL11_FMA(av[i], bv[1], c##i##_1);                  \
    c##i##_2 = CL11_FMA(av[i], bv[2], c##i##_2); c##i##_3 = CL11_FMA(av[i], bv[3], c##i##_3);
#define X_EPI_COLS(jv, r0, r1, r2, r3)                                                                         \
    X_EPI_C(jv, 0, c##r0##_0, c##r1##_0, c##r2##_0, c##r3##_0) X_EPI_C(jv, 1, c##r0##_1, c##r1##_1, c##r2##_1, c##r3##_1) \
    X_EPI_C(jv, 2, c##r0##_2, c##r1##_2, c##r2##_2, c##r3##_2) X_EPI_C(jv, 3, c##r0##_3, c##r1##_3, c##r2##_3, c##r3##_3)
#endif

// the 4 rows (m = m0 + jv * 4 * TXN + 4 tx + 0..3) of column j of the work-item: one float4 store when dst is contiguous in m and aligned
#define X_EPI_C(jv, j, v0, v1, v2, v3)                                                                         \
    {                                                                                                          \
        const int n = n0 + ((j) >> 2) * (4 * XTYN) + 4 * ty + ((j) & 3);                                       \
        const int m = m0 + (jv) * (4 * XTXN) + 4 * tx;                                                         \
        if (n < tc.ne1) {                                                                                      \
            global uchar * dp = cbase + n * tc.nb1;                                                            \
            if (cvec && m + 3 < tc.ne0) {                                                                      \
                *(global float4 *)(dp + m * 4) = (float4)(v0, v1, v2, v3);                                     \
            } else {                                                                                           \
                if (m + 0 < tc.ne0) { ST_F32(dp + (m + 0) * tc.nb0, v0); }                                     \
                if (m + 1 < tc.ne0) { ST_F32(dp + (m + 1) * tc.nb0, v1); }                                     \
                if (m + 2 < tc.ne0) { ST_F32(dp + (m + 2) * tc.nb0, v2); }                                     \
                if (m + 3 < tc.ne0) { ST_F32(dp + (m + 3) * tc.nb0, v3); }                                     \
            }                                                                                                  \
        }                                                                                                      \
    }

#if XRM == 16
#define X_EPILOGUE  X_EPI_COLS(0, 0, 1, 2, 3) X_EPI_COLS(1, 4, 5, 6, 7) X_EPI_COLS(2, 8, 9, 10, 11) X_EPI_COLS(3, 12, 13, 14, 15)
#else
#define X_EPILOGUE  X_EPI_COLS(0, 0, 1, 2, 3) X_EPI_COLS(1, 4, 5, 6, 7)
#endif

// loads of the fragments of step k from the local memory: av[XRM] (rows), bv[XCN] (columns)
#define X_LD_A4(AS4, k, jv, o)                                                                                 \
    { const float4 t = (AS4)[(k) * (XAS / 4) + (jv) * XTXN + tx]; av[o] = t.x; av[o + 1] = t.y; av[o + 2] = t.z; av[o + 3] = t.w; }
#define X_LD_B4(BS4, k, jn, o)                                                                                 \
    { const float4 t = (BS4)[(k) * (XBS / 4) + (jn) * XTYN + ty]; bv[o] = t.x; bv[o + 1] = t.y; bv[o + 2] = t.z; bv[o + 3] = t.w; }
#if XRM == 16
#define X_LD_A(AS4, k)  X_LD_A4(AS4, k, 0, 0) X_LD_A4(AS4, k, 1, 4) X_LD_A4(AS4, k, 2, 8) X_LD_A4(AS4, k, 3, 12)
#else
#define X_LD_A(AS4, k)  X_LD_A4(AS4, k, 0, 0) X_LD_A4(AS4, k, 1, 4)
#endif
#if XCN == 8
#define X_LD_B(BS4, k)  X_LD_B4(BS4, k, 0, 0) X_LD_B4(BS4, k, 1, 4)
#else
#define X_LD_B(BS4, k)  X_LD_B4(BS4, k, 0, 0)
#endif
#define X_STEP(AS4, BS4, k)                                                                                    \
    {                                                                                                          \
        float av[XRM], bv[XCN];                                                                                \
        X_LD_A(AS4, k)                                                                                         \
        X_LD_B(BS4, k)                                                                                         \
        X_ROWS(X_FMA_ROW)                                                                                      \
    }

#if MX_KU == 16
#define X_KSTEPS(AS4, BS4, k)  X_STEP(AS4, BS4, (k)) X_STEP(AS4, BS4, (k) + 1) X_STEP(AS4, BS4, (k) + 2) X_STEP(AS4, BS4, (k) + 3) \
                               X_STEP(AS4, BS4, (k) + 4) X_STEP(AS4, BS4, (k) + 5) X_STEP(AS4, BS4, (k) + 6) X_STEP(AS4, BS4, (k) + 7) \
                               X_STEP(AS4, BS4, (k) + 8) X_STEP(AS4, BS4, (k) + 9) X_STEP(AS4, BS4, (k) + 10) X_STEP(AS4, BS4, (k) + 11) \
                               X_STEP(AS4, BS4, (k) + 12) X_STEP(AS4, BS4, (k) + 13) X_STEP(AS4, BS4, (k) + 14) X_STEP(AS4, BS4, (k) + 15)
#elif MX_KU == 8
#define X_KSTEPS(AS4, BS4, k)  X_STEP(AS4, BS4, (k)) X_STEP(AS4, BS4, (k) + 1) X_STEP(AS4, BS4, (k) + 2) X_STEP(AS4, BS4, (k) + 3) \
                               X_STEP(AS4, BS4, (k) + 4) X_STEP(AS4, BS4, (k) + 5) X_STEP(AS4, BS4, (k) + 6) X_STEP(AS4, BS4, (k) + 7)
#elif MX_KU == 4
#define X_KSTEPS(AS4, BS4, k)  X_STEP(AS4, BS4, (k)) X_STEP(AS4, BS4, (k) + 1) X_STEP(AS4, BS4, (k) + 2) X_STEP(AS4, BS4, (k) + 3)
#elif MX_KU == 2
#define X_KSTEPS(AS4, BS4, k)  X_STEP(AS4, BS4, (k)) X_STEP(AS4, BS4, (k) + 1)
#else
#define X_KSTEPS(AS4, BS4, k)  X_STEP(AS4, BS4, (k))
#endif
#define X_COMPUTE(AS4, BS4)                                                                                    \
    for (int kk = 0; kk < XBK; kk += MX_KU) {                                                                  \
        X_KSTEPS(AS4, BS4, kk)                                                                                 \
    }

#if defined(MX_REQD) && MX_REQD
#define X_ATTR __attribute__((reqd_work_group_size(XNT, 1, 1)))
#else
#define X_ATTR
#endif

#define X_DEF_MM(NAME, AEL, A_DECL, A_LOAD, A_CONV)                                                            \
kernel X_ATTR void NAME(global uchar * pa, tdesc ta, global uchar * pb, tdesc tb, global uchar * pc, tdesc tc, bcast bc) { \
    local float4 Ls4[XNBUF * XBUFSZ / 4];                                                                      \
    local float * Ls = (local float *)Ls4;                                                                     \
    const int lid = (int)get_local_id(0);                                                                      \
    const int tx = lid % XTXN;                                                                                 \
    const int ty = lid / XTXN;                                                                                 \
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
    global uchar * cbase = pc + tc.off + i2 * tc.nb2 + i3 * tc.nb3;                                            \
    const int m0 = bm * XBM;                                                                                   \
    const int n0 = bn * XBN;                                                                                   \
    const int K  = ta.ne0;                                                                                     \
    const int nb = K >> 5;                                                                                     \
    /* src1 rows can be read as float4: 16-byte aligned rows and K a multiple of 4 */                          \
    const bool bvec = ((tb.off | tb.nb1 | tb.nb2 | tb.nb3) & 15) == 0 && (K & 3) == 0;                         \
    /* dst can be written as float4 along m: contiguous and 16-byte aligned */                                 \
    const bool cvec = tc.nb0 == 4 && ((tc.off | tc.nb1 | tc.nb2 | tc.nb3) & 15) == 0;                          \
    A_DECL                                                                                                     \
    X_B_DECL                                                                                                   \
    X_ROWS(X_ACC_ROW)                                                                                          \
    local float4 * As4_0 = Ls4;                                                                                \
    local float4 * Bs4_0 = Ls4 + XBK * XAS / 4;                                                                \
    local float  * As_0  = Ls;                                                                                 \
    local float  * Bs_0  = Ls + XBK * XAS;                                                                     \
    X_MAINLOOP(AEL, A_LOAD, A_CONV)                                                                            \
    X_EPILOGUE                                                                                                 \
}

#if MX_PF == 0
#define X_MAINLOOP(AEL, A_LOAD, A_CONV)                                                                        \
    for (int k0 = 0; k0 < K; k0 += XBK) {                                                                      \
        A_LOAD(k0)                                                                                             \
        X_B_LOAD(k0)                                                                                           \
        A_CONV(k0, As_0, AEL)                                                                                  \
        X_B_STORE(k0, Bs_0)                                                                                    \
        barrier(CLK_LOCAL_MEM_FENCE);                                                                          \
        X_COMPUTE(As4_0, Bs4_0)                                                                                \
        barrier(CLK_LOCAL_MEM_FENCE);                                                                          \
    }
#elif MX_PF == 1
#define X_MAINLOOP(AEL, A_LOAD, A_CONV)                                                                        \
    A_LOAD(0)                                                                                                  \
    X_B_LOAD(0)                                                                                                \
    A_CONV(0, As_0, AEL)                                                                                       \
    X_B_STORE(0, Bs_0)                                                                                         \
    barrier(CLK_LOCAL_MEM_FENCE);                                                                              \
    for (int k0 = 0; k0 < K; k0 += XBK) {                                                                      \
        const bool more = k0 + XBK < K;                                                                        \
        if (more) {                                                                                            \
            A_LOAD(k0 + XBK)                                                                                   \
            X_B_LOAD(k0 + XBK)                                                                                 \
        }                                                                                                      \
        X_COMPUTE(As4_0, Bs4_0)                                                                                \
        barrier(CLK_LOCAL_MEM_FENCE);                                                                          \
        if (more) {                                                                                            \
            A_CONV(k0 + XBK, As_0, AEL)                                                                        \
            X_B_STORE(k0 + XBK, Bs_0)                                                                          \
        }                                                                                                      \
        barrier(CLK_LOCAL_MEM_FENCE);                                                                          \
    }
#else
#define X_MAINLOOP(AEL, A_LOAD, A_CONV)                                                                        \
    A_LOAD(0)                                                                                                  \
    X_B_LOAD(0)                                                                                                \
    A_CONV(0, As_0, AEL)                                                                                       \
    X_B_STORE(0, Bs_0)                                                                                         \
    barrier(CLK_LOCAL_MEM_FENCE);                                                                              \
    for (int k0 = 0, it = 0; k0 < K; k0 += XBK, ++it) {                                                        \
        const bool more = k0 + XBK < K;                                                                        \
        local float4 * Acur = ((it & 1) ? Ls4 + XBUFSZ / 4 : Ls4);                                             \
        local float4 * Bcur = Acur + XBK * XAS / 4;                                                            \
        local float  * Anxt = ((it & 1) ? Ls : Ls + XBUFSZ);                                                   \
        local float  * Bnxt = Anxt + XBK * XAS;                                                                \
        if (more) {                                                                                            \
            A_LOAD(k0 + XBK)                                                                                   \
            X_B_LOAD(k0 + XBK)                                                                                 \
        }                                                                                                      \
        X_COMPUTE(Acur, Bcur)                                                                                  \
        if (more) {                                                                                            \
            A_CONV(k0 + XBK, Anxt, AEL)                                                                        \
            X_B_STORE(k0 + XBK, Bnxt)                                                                          \
        }                                                                                                      \
        barrier(CLK_LOCAL_MEM_FENCE);                                                                          \
    }
#endif

X_DEF_MM(k_mmx_f32,  XAEL_F32, X_A_DECL_GENERIC, X_A_LOAD_GENERIC, X_A_CONV_GENERIC)
X_DEF_MM(k_mmx_f16,  XAEL_F16, X_A_DECL_GENERIC, X_A_LOAD_GENERIC, X_A_CONV_GENERIC)
X_DEF_MM(k_mmx_q4_0, XAEL_F32, X_A_DECL_Q4_0,    X_A_LOAD_Q4_0,    X_A_CONV_Q4_0)
X_DEF_MM(k_mmx_q8_0, XAEL_F32, X_A_DECL_Q8_0,    X_A_LOAD_Q8_0,    X_A_CONV_Q8_0)
X_DEF_MM(k_mmx_q4_k, XAEL_F32, X_A_DECL_Q4_K,    X_A_LOAD_Q4_K,    X_A_CONV_Q4_K)
X_DEF_MM(k_mmx_q6_k, XAEL_F32, X_A_DECL_Q6_K,    X_A_LOAD_Q6_K,    X_A_CONV_Q6_K)
#endif // mat-mat kernels (mmx)

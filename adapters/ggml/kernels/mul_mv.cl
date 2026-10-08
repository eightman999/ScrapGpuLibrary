// CL11 backend: MUL_MAT with a skinny src1 (mat-vec) of F32 / F16 weights, and the helpers that convert the bytes of the quantized weights to floats.
//
//   dst[i0, i1, i2, i3] = sum_k src0[k, i0, i2 / r2, i3 / r3] * src1[k, i1, i2, i3]
//
// src0: F32/F16 (rows contiguous), src1: F32 (rows contiguous), dst: F32. (Q4_0 / Q8_0: mul_mv_r.cl, Q4_K / Q6_K: mul_mv_k.cl)
// `tpr` threads (a power of 2, chosen by the host) cooperate on one output element, and a work-group
// of WG_MV threads computes WG_MV/tpr outputs. The partial sums are reduced through local memory.

#ifndef CL11_NO_BASE
#define MV_BEGIN                                                                              \
    local float sh[WG_MV];                                                                    \
    const int lid  = (int)get_local_id(0);                                                    \
    const int lane = lid & (tpr - 1);                                                         \
    const int r    = LGID() * (WG_MV >> tprl) + (lid >> tprl);                                \
    const bool valid = r < tc.ne0 * tc.ne1 * tc.ne2 * tc.ne3;                                 \
    global const uchar * arow = pa;                                                           \
    global const uchar * brow = pb;                                                           \
    int i0 = 0, i1 = 0, i2 = 0, i3 = 0;                                                       \
    if (valid) {                                                                              \
        SPLIT4(r, tc, i0, i1, i2, i3);                                                        \
        const int i02 = FDIV_RAW(i2, bc.m2, bc.l2);                                           \
        const int i03 = FDIV_RAW(i3, bc.m3, bc.l3);                                           \
        arow = pa + ta.off + i0 * ta.nb1 + i02 * ta.nb2 + i03 * ta.nb3;                       \
        brow = pb + tb.off + i1 * tb.nb1 + i2 * tb.nb2 + i3 * tb.nb3;                         \
    }                                                                                         \
    const int ne00 = ta.ne0;                                                                  \
    global const float * bf = (global const float *)brow;                                     \
    float acc = 0.0f;

#define MV_END                                                                                \
    sh[lid] = acc;                                                                            \
    barrier(CLK_LOCAL_MEM_FENCE);                                                             \
    for (int s = tpr >> 1; s > 0; s >>= 1) {                                                  \
        if (lane < s) {                                                                       \
            sh[lid] += sh[lid + s];                                                           \
        }                                                                                     \
        barrier(CLK_LOCAL_MEM_FENCE);                                                         \
    }                                                                                         \
    if (valid && lane == 0) {                                                                 \
        ST_F32(pc + tc.off + i0 * tc.nb0 + i1 * tc.nb1 + i2 * tc.nb2 + i3 * tc.nb3, sh[lid]); \
    }

#define MV_ARGS global uchar * pa, tdesc ta, global uchar * pb, tdesc tb, global uchar * pc, tdesc tc, bcast bc, int tpr, int tprl

// ---- F32 ----
// the loads of 4 chunks (of 4 elements per thread) are issued together: the rows are short (K cache) and the kernels are latency bound
kernel void k_mv_f32(MV_ARGS) {
    MV_BEGIN
    if (valid) {
        global const float * af = (global const float *)arow;
        int i = lane * 4;
        for (; i + 12 * tpr + 4 <= ne00; i += 16 * tpr) {
            const float4 a0 = vload4(0, af + i), a1 = vload4(0, af + i + 4 * tpr), a2 = vload4(0, af + i + 8 * tpr), a3 = vload4(0, af + i + 12 * tpr);
            const float4 x0 = vload4(0, bf + i), x1 = vload4(0, bf + i + 4 * tpr), x2 = vload4(0, bf + i + 8 * tpr), x3 = vload4(0, bf + i + 12 * tpr);
            acc += (dot(a0, x0) + dot(a1, x1)) + (dot(a2, x2) + dot(a3, x3));
        }
        for (; i < ne00; i += tpr * 4) {
            if (i + 4 <= ne00) {
                acc += dot(vload4(0, af + i), vload4(0, bf + i));
            } else {
                for (int j = i; j < ne00; ++j) {
                    acc += af[j] * bf[j];
                }
            }
        }
    }
    MV_END
}

// ---- F16 ----
kernel void k_mv_f16(MV_ARGS) {
    MV_BEGIN
    if (valid) {
        global const half * ah = (global const half *)arow;
        int i = lane * 4;
        for (; i + 12 * tpr + 4 <= ne00; i += 16 * tpr) {
            const float4 a0 = vload_half4(0, ah + i), a1 = vload_half4(0, ah + i + 4 * tpr), a2 = vload_half4(0, ah + i + 8 * tpr), a3 = vload_half4(0, ah + i + 12 * tpr);
            const float4 x0 = vload4(0, bf + i), x1 = vload4(0, bf + i + 4 * tpr), x2 = vload4(0, bf + i + 8 * tpr), x3 = vload4(0, bf + i + 12 * tpr);
            acc += (dot(a0, x0) + dot(a1, x1)) + (dot(a2, x2) + dot(a3, x3));
        }
        for (; i < ne00; i += tpr * 4) {
            if (i + 4 <= ne00) {
                acc += dot(vload_half4(0, ah + i), vload4(0, bf + i));
            } else {
                for (int j = i; j < ne00; ++j) {
                    acc += vload_half(j, ah) * bf[j];
                }
            }
        }
    }
    MV_END
}
#endif // CL11_NO_BASE

// =============================================================================================
// 4 weights per 32-bit word, converted without the (slow on Fermi) integer to float conversion instructions:
// a byte b (0..255) is the mantissa of the float 2^23 + b, the offset is subtracted afterwards (exact).
#ifdef CL11_NV_ASM
// NVIDIA PTX: prmt picks arbitrary bytes of two 32-bit registers: one instruction builds the float 2^23 + byte k of w
uint cl11_prmt(uint a, uint b, uint sel) {
    uint r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
    return r;
}
float4 f4_mag(uint w) {
    return (float4)(as_float(cl11_prmt(w, 0x4B000000u, 0x7540u)), as_float(cl11_prmt(w, 0x4B000000u, 0x7541u)),
                    as_float(cl11_prmt(w, 0x4B000000u, 0x7542u)), as_float(cl11_prmt(w, 0x4B000000u, 0x7543u)));
}
#else
float4 f4_mag(uint w) {
    return (float4)(as_float(0x4B000000u | (w & 0xFFu)), as_float(0x4B000000u | ((w >> 8) & 0xFFu)),
                    as_float(0x4B000000u | ((w >> 16) & 0xFFu)), as_float(0x4B000000u | (w >> 24)));
}
#endif
// low / high nibbles of the 4 bytes (0..15)
float4 f4_lo4(uint w)   { return f4_mag(w & 0x0F0F0F0Fu) - (float4)(8388608.0f); }
float4 f4_hi4(uint w)   { return f4_mag((w >> 4) & 0x0F0F0F0Fu) - (float4)(8388608.0f); }
// 6-bit values - 32 (q6_K), the 4 bytes hold values 0..63
float4 f4_q6(uint w)    { return f4_mag(w) - (float4)(8388640.0f); }

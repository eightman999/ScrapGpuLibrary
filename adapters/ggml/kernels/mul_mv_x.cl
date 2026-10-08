#ifndef CL11_NO_BASE
// CL11 backend: the mat-vec kernels of mul_mv_xcommon.cl, instantiated by the macros of mvx_defs_staged.cl / mvx_defs_global.cl

// ---- F32: a lane handles the float4 groups lane, lane + 32, ... of a chunk of 1024 elements ----
kernel void KN(f32)(MVS_ARGS) {
    MVX_BEGIN(1024)
    for (int k0 = 0; k0 < ne00; k0 += 1024) {
        XSTAGE(1024, k0)
        XBAR
        for (int t = 0; t < 8; ++t) {
            const int gi = lane + 32 * t;
            if (k0 + 4 * gi < ne00) {
                const float4 x = LDX(gi);
                for (int rr = 0; rr < NRW; ++rr) {
                    acc[rr] += dot(((global const float4 *)rowp[rr])[(k0 >> 2) + gi], x);
                }
            }
        }
        XBAR
    }
    MVS_END
}

// ---- F16 ----
kernel void KN(f16)(MVS_ARGS) {
    MVX_BEGIN(1024)
    for (int k0 = 0; k0 < ne00; k0 += 1024) {
        XSTAGE(1024, k0)
        XBAR
        for (int t = 0; t < 8; ++t) {
            const int gi = lane + 32 * t;
            if (k0 + 4 * gi < ne00) {
                const float4 x = LDX(gi);
                for (int rr = 0; rr < NRW; ++rr) {
                    acc[rr] += dot(vload_half4(0, (global const half *)rowp[rr] + k0 + 4 * gi), x);
                }
            }
        }
        XBAR
    }
    MVS_END
}
#endif // CL11_NO_BASE

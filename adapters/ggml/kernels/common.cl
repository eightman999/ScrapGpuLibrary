// CL11 backend: common definitions.
//
// Everything here must be valid OpenCL C 1.1 (-cl-std=CL1.1):
//   - no cl_khr_fp16 (half is a storage type only: vload_half/vstore_half),
//   - no double, no subgroups, no atomics, no printf,
//   - no static functions, no program-scope variables.
//
// The following macros are defined on the compiler command line by the host:
//   WG_EW   work-group size of the element-wise kernels (power of 2)
//   WG_ROW  work-group size of the row kernels: rms_norm, soft_max, get_rows (power of 2)
//   WG_MV   work-group size of the mat-vec kernels (power of 2)
//   GT      the tiled mat-mat kernels use GT*GT work-items and compute (4*GT)x(4*GT) tiles

//   CL11_MAD  use mad() instead of fma() (CL11_FMA): Mesa rusticl emulates fma() in software on the GCN GPUs (~100x slower than mad),
//             set by the host for the AMD devices (cl11_tuning_for), the other devices keep the fused fma()
#ifdef CL11_MAD
#define CL11_FMA(a, b, c) mad(a, b, c)
#else
#define CL11_FMA(a, b, c) fma(a, b, c)
#endif

// Parts of the program (the host builds one program, or several, see cl11_ensure_program):
//   CL11_NO_K     without the Q4_K / Q6_K mat-vec kernels           CL11_PROG_K   only these (built with another register cap on NVIDIA)
//   CL11_NO_MM    without the mat-mat kernels                      CL11_ONLY_MM  only these
// Mesa Clover pays for every launch in proportion to the size of the program that holds the kernel (empty kernel, WX 2100: 5 us in a program
// of 1 kernel, 24 us with 50 more kernels, 104 us with 200), so the host puts the mat-mat kernels (the largest part) and the K-quant kernels
// apart from the kernels of the decode path there.
#if defined(CL11_PROG_K) || defined(CL11_ONLY_MM)
#define CL11_NO_BASE
#endif

// Tensor descriptor, passed by value. The layout (9 x int) must match struct cl11_td on the host.
// ne: number of elements, nb: strides in bytes, off: byte offset of the first element in the cl_mem.
// fm*/fl*: magic numbers to divide by ne0..ne3 without an integer division (see FDIV): the integer division
// of the 32-bit integers is very slow on the old GPUs (a long instruction sequence on Fermi).
typedef struct {
    int ne0, ne1, ne2, ne3;
    int nb0, nb1, nb2, nb3;
    int off;
    uint fm0, fm1, fm2, fm3;
    uint fl0, fl1, fl2, fl3;
} tdesc;

// broadcast / tiling constants of the mul_mat kernels (the host computes them)
typedef struct {
    int  r2, r3;            // ne12 / ne02, ne13 / ne03
    uint m2, l2, m3, l3;    // fast division by r2, r3
    int  tiles;             // number of work-groups along the rows (mat-vec: row tiles, mat-mat: tiles of dst rows)
    uint mt, lt;            // fast division by tiles
    int  tiles2;            // mat-mat: number of tiles along the columns
    uint mt2, lt2;          // fast division by tiles2
} bcast;

// n / d for 0 <= n < 2^31, given the magic numbers m, l of d (see cl11_fastdiv_init on the host)
#define FDIV_RAW(n, m, l)  ((int)((mul_hi((uint)(n), (uint)(m)) + (uint)(n)) >> (l)))
#define FDIV(n, t, k)      FDIV_RAW(n, (t).fm##k, (t).fl##k)
#define FMOD(n, t, k)      ((n) - FDIV(n, t, k) * (t).ne##k)

// Linear index of the work-group. Launches are always 2-D (x, y) so that no dimension
// exceeds 65535 groups (the grid limit of old NVIDIA GPUs).
#define LGID() ((int)get_group_id(0) + (int)get_group_id(1) * (int)get_num_groups(0))

#define PCHAR(p, o)  ((global uchar *)(p) + (o))

// element accessors on byte pointers
#define LD_F32(p)     (*(global const float *)(p))
#define LD_F16(p)     vload_half(0, (global const half *)(p))
#define ST_F32(p, v)  (*(global float *)(p) = (v))
#define ST_F16(p, v)  vstore_half((v), 0, (global half *)(p))

// split a linear index over the dimensions of a tensor descriptor
#define SPLIT4(idx, t, i0, i1, i2, i3) \
    { int _r = (idx);                  \
      int _q = FDIV(_r, t, 0); i0 = _r - _q * (t).ne0; _r = _q; \
      _q = FDIV(_r, t, 1);     i1 = _r - _q * (t).ne1; _r = _q; \
      _q = FDIV(_r, t, 2);     i2 = _r - _q * (t).ne2; i3 = _q; }

#define OFF4(t, i0, i1, i2, i3) ((t).off + (i0) * (t).nb0 + (i1) * (t).nb1 + (i2) * (t).nb2 + (i3) * (t).nb3)

// ---------------------------------------------------------------------------------------------
// F32 / F16 rows: dequantize a single element k of a row (r points at the first byte of the row).
// The quantized types (Q4_0 / Q8_0 / Q4_K / Q6_K) are stored in a repacked layout, see dqr_* below.
// ---------------------------------------------------------------------------------------------

float dq_f32(global const uchar * r, int k) { return ((global const float *)r)[k]; }
float dq_f16(global const uchar * r, int k) { return vload_half(k, (global const half *)r); }

// Q4_0 and Q8_0 weights are stored in the repacked layout "R32" (groups of 32 rows, see mul_mv_r.cl): the element (m, k) of a
// slice is addressed through the base of the slice, its number of rows M, the number of blocks per row nb = ne0 / 32.
float dqr_q8_0(global const uchar * s, int M, int nb, int m, int k) {
    const int g = m >> 5, lane = m & 31;
    const int rs = min(32, M - (g << 5));
    const int b = k >> 5, kk = k & 31;
    const int q = g * (32 * nb * 32) + ((((b * 2 + (kk >> 4)) * rs + lane) << 4) + (kk & 15));
    const int d = M * nb * 32 + ((g * 32 * nb + b * rs + lane) << 1);
    return vload_half(0, (global const half *)(s + d)) * (float)((int)s[q] - 128);
}

float dqr_q4_0(global const uchar * s, int M, int nb, int m, int k) {
    const int g = m >> 5, lane = m & 31;
    const int rs = min(32, M - (g << 5));
    const int b = k >> 5, kk = k & 31;
    const int q = g * (32 * nb * 16) + (((b * rs + lane) << 4) + (kk & 15));
    const int d = M * nb * 16 + ((g * 32 * nb + b * rs + lane) << 1);
    const int v = (int)s[q];
    return vload_half(0, (global const half *)(s + d)) * (float)(((kk < 16) ? (v & 0xF) : (v >> 4)) - 8);
}

// 6-bit scale / min of q4_K, j in [0, 8)
int k4_sc(int j, global const uchar * q) {
    return (j < 4) ? (q[j] & 63) : ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
}
int k4_mn(int j, global const uchar * q) {
    return (j < 4) ? (q[j + 4] & 63) : ((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
}

// Q4_K / Q6_K weights are repacked too (groups of 32 rows, see mul_mv_k.cl): element (m, k) of a slice, nb = ne0 / 256
float dqr_q4_k(global const uchar * s, int M, int nb, int m, int k) {
    const int g = m >> 5, lane = m & 31;
    const int rs = min(32, M - (g << 5));
    const int b = k >> 8, j = k & 255;
    const int c = j >> 6, l = j & 63;
    const int by = c * 32 + (l & 31);                                  // byte of the nibbles in the 128 bytes of the super-block
    const int q = g * (32 * nb * 128) + ((((b * 8 + (by >> 4)) * rs + lane) << 4) + (by & 15));
    global const uchar * h = s + M * nb * 128 + (((g * 32 * nb + b * rs + lane)) << 4);
    const float d    = vload_half(0, (global const half *)h);
    const float dmin = vload_half(1, (global const half *)h);
    const int is = c * 2 + (l >> 5);
    int v = s[q];
    v = (l < 32) ? (v & 0xF) : (v >> 4);
    return d * (float)k4_sc(is, h + 4) * (float)v - dmin * (float)k4_mn(is, h + 4);
}

float dqr_q6_k(global const uchar * s, int M, int nb, int m, int k) {
    const int g = m >> 5, lane = m & 31;
    const int rs = min(32, M - (g << 5));
    const int b = k >> 8, j = k & 255;
    const int h  = j >> 7;
    const int rr = j & 127;
    const int qd = rr >> 5;
    const int l  = rr & 31;
    const int by1 = h * 64 + (qd & 1) * 32 + l;                        // byte of ql
    const int by2 = 128 + h * 32 + l;                                   // byte of qh
    const int qbase = g * (32 * nb * 192);
    const int ql = s[qbase + ((((b * 12 + (by1 >> 4)) * rs + lane) << 4) + (by1 & 15))];
    const int qh = s[qbase + ((((b * 12 + (by2 >> 4)) * rs + lane) << 4) + (by2 & 15))];
    const int nbl = (qd >> 1) ? (ql >> 4) : (ql & 0xF);
    const int q = (nbl | (((qh >> (qd * 2)) & 3) << 4)) - 32;
    const int sc = (int)s[M * nb * 192 + ((((g * 32 * nb + b * rs + lane)) << 4) + h * 8 + (l >> 4) + qd * 2)] - 128;
    const float d = vload_half(0, (global const half *)(s + M * nb * 208 + ((g * 32 * nb + b * rs + lane) << 1)));
    return d * (float)(sc * q);
}


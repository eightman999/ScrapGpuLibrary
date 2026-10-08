// CL11 backend: "multi mat-vec" (k_mvm_*): up to 3 MUL_MATs that read the same src1 (one column) in ONE launch (Q / K / V projections, FFN gate / up).
//
// The body of every kernel is the one of the mat-vec kernel of the same type (k_mvr_*, mul_mv_r.cl / mul_mv_k.cl; the macros MVR_* / MVK_*
// are reused, nothing of them is changed): a work-group first picks the weight tensor (slot) its group index belongs to
// (slot 0: groups 0 .. tiles0 - 1, slot 1: the next tiles1, slot 2: the rest), takes the weight / dst pointers and descriptors of the slot
// and the tile number inside of it, and then runs the usual code. The host (cl11_op_mul_mat_multi) fuses only MUL_MATs of the same
// repacked type and the same row length with a single column of src1 and no batch (ne2 = ne3 = 1), a slot may have a different number of rows
// (Q 2048, K 1024, V 1024 for Qwen3-0.6B). An unused slot has bc.tiles = 0.
//
// LGID() is redefined for the kernels of this file (the group index relative to the slot) and restored at the end.

#ifndef CL11_NO_MVM
#define MVM_ARGS                                                                                              \
    global uchar * pa0, global uchar * pa1, global uchar * pa2,                                               \
    global uchar * pc0, global uchar * pc1, global uchar * pc2,                                               \
    tdesc ta0, tdesc ta1, tdesc ta2, tdesc tc0, tdesc tc1, tdesc tc2,                                         \
    bcast bc0, bcast bc1, bcast bc2,                                                                          \
    global uchar * pb, tdesc tb

#define MVM_SELECT                                                                                            \
    int mvm_g = (int)get_group_id(0) + (int)get_group_id(1) * (int)get_num_groups(0);                         \
    global uchar * pa;                                                                                        \
    global uchar * pc;                                                                                        \
    tdesc ta;                                                                                                 \
    tdesc tc;                                                                                                 \
    bcast bc;                                                                                                 \
    if (mvm_g < bc0.tiles) {                                                                                  \
        pa = pa0; pc = pc0; ta = ta0; tc = tc0; bc = bc0;                                                     \
    } else if (mvm_g < bc0.tiles + bc1.tiles) {                                                               \
        mvm_g -= bc0.tiles;                                                                                   \
        pa = pa1; pc = pc1; ta = ta1; tc = tc1; bc = bc1;                                                     \
    } else {                                                                                                  \
        mvm_g -= bc0.tiles + bc1.tiles;                                                                       \
        pa = pa2; pc = pc2; ta = ta2; tc = tc2; bc = bc2;                                                     \
    }

#undef LGID
#define LGID() mvm_g

#ifndef CL11_NO_BASE
#define WG_MVR WG_MVR_Q8_0
#define XLDG   LDG_Q8_0
#define NRR    NRR_Q8_0
#define NWR    (WG_MVR / 32)
#define XLO XLO_G
kernel void k_mvm_q8_0(MVM_ARGS) {
    MVM_SELECT
    MVR_BEGIN(32, global, MVR_XG)
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
kernel void k_mvm_q4_0(MVM_ARGS) {
    MVM_SELECT
    MVR_BEGIN(16, global, MVR_XG)
    if (full) { MVR_Q4_LOOP(RS_FULL) } else { MVR_Q4_LOOP(RS_VAR) }
    MVR_END
}
#undef XLO
#undef XLDG
#undef NWR
#undef NRR
#undef WG_MVR
#endif // CL11_NO_BASE

#if !defined(CL11_NO_K) && !defined(CL11_ONLY_MM)
#define WG_MVR WG_MVR_Q4_K
#define XLDG   LDG_Q4_K
#define NRR    NRR_Q4_K
#define NWR    (WG_MVR / 32)
#define XLO XLO_G
kernel void k_mvm_q4_k(MVM_ARGS) {
    MVM_SELECT
    MVK_BEGIN(8, global, MVR_XG)
    if (full) { MVK_Q4_LOOP(RS_FULL) } else { MVK_Q4_LOOP(RS_VAR) }
    MVR_END
}
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
kernel void k_mvm_q6_k(MVM_ARGS) {
    MVM_SELECT
    MVK_BEGIN(12, global, MVR_XG)
    if (full) { MVK_Q6_LOOP(RS_FULL) } else { MVK_Q6_LOOP(RS_VAR) }
    MVR_END
}
#undef XLO
#undef XLDG
#undef NWR
#undef NRR
#undef WG_MVR
#endif // CL11_NO_K

#undef LGID
#define LGID() ((int)get_group_id(0) + (int)get_group_id(1) * (int)get_num_groups(0))
#endif // CL11_NO_MVM

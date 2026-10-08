// CL11 backend: check of the attention kernels at the graph level (K cache MUL_MAT, and CONT of the transposed V cache fused with the MUL_MAT,
// grouped query attention) against a double precision reference. test-backend-ops does not build the fused graph.
//
//   out-of-tree plugin (llama.cpp built with -DGGML_BACKEND_DL=ON, CL11 built as libggml-cl11.so, see llama.cpp/README.md):
//     c++ -std=c++17 -O1 -I<llama.cpp>/ggml/include adapters/ggml/tools/attn_check.cpp -L<llama.cpp>/build-dl/bin -lggml -lggml-base -Wl,-rpath,<llama.cpp>/build-dl/bin -o attn_check
//     GGML_BACKEND_PATH=$PWD/build-cl11/libggml-cl11.so ./attn_check
//   in-tree: the same with -L<llama.cpp>/build-cl11/bin and without GGML_BACKEND_PATH
//   GGML_CL11_DEVICES="GT 710" ./attn_check        # every line must end with OK
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>

static float frand() { return (float) rand() / RAND_MAX - 0.5f; }

static int run(ggml_backend_t be, int hd, int nkvh, int r2, int kv, int npos, std::vector<float> & out, const std::vector<ggml_fp16_t> & vdata, const std::vector<ggml_fp16_t> & kdata, const std::vector<float> & qd, const std::vector<float> & pd) {
    ggml_init_params ip = { 1 << 24, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    const int nemb = hd * nkvh;
    ggml_tensor * vc = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, nemb, kv);     // [nemb, kv]: heads of a position contiguous
    ggml_tensor * kc = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, nemb, kv);
    ggml_tensor * q  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, 1, nkvh * r2);
    ggml_tensor * p  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kv, 1, nkvh * r2);
    // K: view [hd, nkvh, kv] -> permute -> [hd, kv, nkvh]
    ggml_tensor * kv3 = ggml_view_3d(ctx, kc, hd, nkvh, kv, hd * sizeof(ggml_fp16_t), nemb * sizeof(ggml_fp16_t), 0);
    ggml_tensor * kp = ggml_permute(ctx, kv3, 0, 2, 1, 3);
    ggml_tensor * kq = ggml_mul_mat(ctx, kp, q);                              // [kv, 1, nkvh*r2]
    ggml_tensor * vv3 = ggml_view_3d(ctx, vc, hd, nkvh, kv, hd * sizeof(ggml_fp16_t), nemb * sizeof(ggml_fp16_t), 0);
    ggml_tensor * vp = ggml_permute(ctx, vv3, 0, 2, 1, 3);                    // [hd, kv, nkvh]
    ggml_tensor * vt = ggml_cont(ctx, ggml_transpose(ctx, vp));               // [kv, hd, nkvh]
    ggml_tensor * kqv = ggml_mul_mat(ctx, vt, p);                             // [hd, 1, nkvh*r2]
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, kq);
    ggml_build_forward_expand(gf, kqv);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_backend_tensor_set(vc, vdata.data(), 0, ggml_nbytes(vc));
    ggml_backend_tensor_set(kc, kdata.data(), 0, ggml_nbytes(kc));
    ggml_backend_tensor_set(q, qd.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(p, pd.data(), 0, ggml_nbytes(p));
    ggml_backend_graph_compute(be, gf);
    out.resize(ggml_nelements(kq) + ggml_nelements(kqv));
    ggml_backend_tensor_get(kq, out.data(), 0, ggml_nbytes(kq));
    ggml_backend_tensor_get(kqv, out.data() + ggml_nelements(kq), 0, ggml_nbytes(kqv));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return 0;
}

int main() {
    ggml_backend_load_all();
    ggml_backend_t cl = nullptr, cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (strncmp(ggml_backend_dev_name(d), "CL11", 4) == 0) { cl = ggml_backend_dev_init(d, nullptr); break; }
    }
    if (!cl) { printf("no CL11 device\n"); return 1; }
    int bad = 0;
    struct cfg { int hd, nkvh, r2, kv; } cfgs[] = { {128, 8, 2, 256}, {64, 2, 7, 256}, {64, 4, 8, 256}, {128, 8, 1, 64}, {80, 3, 3, 96}, {256, 2, 4, 128}, {128, 8, 2, 40}, {32, 16, 2, 32} };
    for (auto c : cfgs) {
        const int nemb = c.hd * c.nkvh;
        std::vector<ggml_fp16_t> v(nemb * c.kv), k(nemb * c.kv);
        for (auto & x : v) x = ggml_fp32_to_fp16(frand());
        for (auto & x : k) x = ggml_fp32_to_fp16(frand());
        std::vector<float> q(c.hd * c.nkvh * c.r2), p(c.kv * c.nkvh * c.r2);
        for (auto & x : q) x = frand();
        for (auto & x : p) x = frand();
        std::vector<float> a, b;
        run(cl, c.hd, c.nkvh, c.r2, c.kv, c.kv, b, v, k, q, p);
        // double reference: kq[pos, hq] = sum_d K[pos][h][d] * q[hq][d]; kqv[d, hq] = sum_pos V[pos][h][d] * p[hq][pos]
        double me = 0;
        for (int hq = 0; hq < c.nkvh * c.r2; ++hq) {
            const int h = hq / c.r2;
            for (int pos = 0; pos < c.kv; ++pos) {
                double s = 0;
                for (int d = 0; d < c.hd; ++d) s += (double) ggml_fp16_to_fp32(k[(size_t) pos * nemb + h * c.hd + d]) * q[hq * c.hd + d];
                me = std::max(me, fabs(s - b[(size_t) hq * c.kv + pos]));
            }
            for (int d = 0; d < c.hd; ++d) {
                double s = 0;
                for (int pos = 0; pos < c.kv; ++pos) s += (double) ggml_fp16_to_fp32(v[(size_t) pos * nemb + h * c.hd + d]) * p[hq * c.kv + pos];
                me = std::max(me, fabs(s - b[(size_t) c.nkvh * c.r2 * c.kv + (size_t) hq * c.hd + d]));
            }
        }
        printf("hd=%d nkvh=%d r2=%d kv=%d: max err vs double %g %s\n", c.hd, c.nkvh, c.r2, c.kv, me, me < 1e-4 ? "OK" : "BAD");
        bad += me >= 1e-4;
    }
    return bad;
}

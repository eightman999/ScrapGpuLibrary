// CL11 backend: MUL_MAT_ID of one token (the generation method: the kernels read the expert numbers from the ids on the GPU) at the shapes of
// real MoE models, against a double precision reference. test-backend-ops covers the op with small random shapes only.
//
//   c++ -std=c++17 -O1 -I<llama.cpp>/ggml/include adapters/ggml/tools/mid_check.cpp -L<llama.cpp>/build-dl/bin -lggml -lggml-base -Wl,-rpath,<llama.cpp>/build-dl/bin -o mid_check
//   GGML_BACKEND_PATH=$PWD/build-cl11/libggml-cl11.so GGML_CL11_DEVICES="RX 6400" ./mid_check      # every line must end with OK
//   (GGML_LIB_DIR=<llama.cpp>/build-dl/bin if the CPU backend plugin is not next to the tool)
//
// The cases follow build_moe_ffn of llama.cpp: the ids are a view [n_used, n_tokens] of an [n_expert, n_tokens] tensor (so nb[1] = n_expert * 4),
// the gate / up products read one x for all the slots (src1 [K, 1, n_tokens]), the down product one x per slot (src1 [K, n_used, n_tokens]).
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static float frand() { return (float) rand() / (float) RAND_MAX - 0.5f; }

struct cs { const char * name; ggml_type type; int n_exp, n_used, k, m, n_slots_x, n_tok; };

// gpu_ids: the ids are computed in the graph (ARGSORT of random scores, as the router of a MoE model does) instead of being an input
static bool run(ggml_backend_t cl, ggml_backend_t cpu, const cs & c, bool gpu_ids) {
    std::vector<float> f((size_t) c.k * c.m * c.n_exp);
    for (auto & v : f) v = frand();
    const size_t rs = ggml_row_size(c.type, c.k);
    std::vector<uint8_t> q(rs * c.m * c.n_exp);
    ggml_quantize_chunk(c.type, f.data(), q.data(), 0, (int64_t) c.m * c.n_exp, c.k, nullptr);
    std::vector<float> deq(f.size());
    const ggml_type_traits * tr = ggml_get_type_traits(c.type);
    for (int64_t r = 0; r < (int64_t) c.m * c.n_exp; ++r) {
        tr->to_float(q.data() + r * rs, deq.data() + r * c.k, c.k);
    }

    ggml_context * wctx = ggml_init({ 1 << 20, nullptr, true });
    ggml_context * gctx = ggml_init({ 1 << 24, nullptr, true });
    ggml_tensor * w = ggml_new_tensor_3d(wctx, c.type, c.k, c.m, c.n_exp);
    ggml_backend_buffer_t wb = ggml_backend_alloc_ctx_tensors(wctx, cl);
    ggml_backend_buffer_set_usage(wb, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_set(w, q.data(), 0, q.size());

    ggml_tensor * x   = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, c.k, c.n_slots_x, c.n_tok);
    ggml_tensor * all = ggml_new_tensor_2d(gctx, GGML_TYPE_I32, c.n_exp, c.n_tok);
    ggml_set_input(x);
    ggml_set_input(all);
    ggml_tensor * ids = ggml_view_2d(gctx, all, c.n_used, c.n_tok, all->nb[1], 0);
    ggml_tensor * sorted = nullptr;
    if (gpu_ids) {
        all = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, c.n_exp, c.n_tok); // the scores
        ggml_set_input(all);
        ids = ggml_argsort_top_k(gctx, all, c.n_used);
        sorted = ids->view_src ? ids->view_src : ids;
        ggml_set_output(sorted);
    }
    ggml_tensor * y   = ggml_mul_mat_id(gctx, w, x, ids);
    ggml_set_output(y);
    ggml_cgraph * gf = ggml_new_graph(gctx);
    ggml_build_forward_expand(gf, y);

    ggml_backend_t backends[2] = { cl, cpu };
    ggml_backend_buffer_type_t bufts[2] = { ggml_backend_get_default_buffer_type(cl), ggml_backend_get_default_buffer_type(cpu) };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, 2, 256, false, false);
    bool ok = ggml_backend_sched_alloc_graph(sched, gf);
    std::vector<float> xv((size_t) c.k * c.n_slots_x * c.n_tok);
    for (auto & v : xv) v = frand() * 4;
    std::vector<int32_t> iv((size_t) c.n_exp * c.n_tok);
    for (int t = 0; t < c.n_tok; ++t) {
        for (int e = 0; e < c.n_exp; ++e) iv[(size_t) t * c.n_exp + e] = (e * 7 + t * 3 + 5) % c.n_exp; // a permutation when gcd(7, n_exp) == 1
    }
    if (ok) {
        ggml_backend_tensor_set(x, xv.data(), 0, ggml_nbytes(x));
        if (gpu_ids) {
            std::vector<float> sc((size_t) c.n_exp * c.n_tok);
            for (size_t i = 0; i < sc.size(); ++i) sc[i] = (float) ((i * 7919 + 13) % 1009); // distinct per token row
            ggml_backend_tensor_set(all, sc.data(), 0, ggml_nbytes(all));
        } else {
            ggml_backend_tensor_set(all, iv.data(), 0, ggml_nbytes(all));
        }
        ok = ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;
    }
    double me = 0, mag = 0;
    if (ok && gpu_ids) {
        std::vector<int32_t> so((size_t) ggml_nelements(sorted));
        ggml_backend_tensor_get(sorted, so.data(), 0, ggml_nbytes(sorted));
        for (int t = 0; t < c.n_tok; ++t) {
            for (int i = 0; i < c.n_used; ++i) iv[(size_t) t * c.n_exp + i] = so[(size_t) t * c.n_exp + i];
        }
    }
    if (ok) {
        std::vector<float> out(ggml_nelements(y));
        ggml_backend_tensor_get(y, out.data(), 0, ggml_nbytes(y));
        for (int t = 0; t < c.n_tok; ++t) {
            for (int i = 0; i < c.n_used; ++i) {
                const int e = iv[(size_t) t * c.n_exp + i];
                const float * xs = xv.data() + ((size_t) t * c.n_slots_x + (c.n_slots_x == 1 ? 0 : i)) * c.k;
                for (int r = 0; r < c.m; ++r) {
                    double s = 0;
                    const float * wr = deq.data() + ((size_t) e * c.m + r) * c.k;
                    for (int k = 0; k < c.k; ++k) s += (double) wr[k] * xs[k];
                    const double got = out[((size_t) t * c.n_used + i) * c.m + r];
                    me = std::max(me, fabs(s - got));
                    mag = std::max(mag, fabs(s));
                }
            }
        }
    }
    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(wb);
    ggml_free(gctx);
    ggml_free(wctx);
    const bool good = ok && me <= 1e-3 * std::max(mag, 1.0);
    printf("%-44s %s max err %.3g (max |y| %.3g) %s\n", c.name, gpu_ids ? "gpu-ids" : "host-ids", me, mag, good ? "OK" : "BAD");
    return good;
}


// A chain of MoE layers like build_moe_ffn (router, sigmoid, ARGSORT top-k, GET_ROWS of the weights, three MUL_MAT_IDs, swiglu, scaling, sum of the
// slots), one token, everything on the device between the first and the last node. The same graph runs on the CPU backend alone: the outputs must
// agree to the precision of the CPU's 8-bit activations. A launch that starts before the previous one is complete / visible shows up here.
static bool run_chain(ggml_backend_t cl, ggml_backend_t cpu, ggml_type type, int n_layers) {
    const int n_embd = 2048, n_ff = 896, n_exp = 32, n_used = 4;
    auto make = [&](int k, int m, int ne, std::vector<uint8_t> & q) {
        std::vector<float> f((size_t) k * m * ne);
        for (auto & v : f) v = frand() * 0.5f;
        q.resize(ggml_row_size(type, k) * m * ne);
        ggml_quantize_chunk(type, f.data(), q.data(), 0, (int64_t) m * ne, k, nullptr);
    };
    std::vector<uint8_t> qup, qdown, qinp;
    make(n_embd, n_ff, n_exp, qup);
    make(n_ff, n_embd, n_exp, qdown);
    std::vector<float> finp((size_t) n_embd * n_exp), fbias(n_exp);
    for (auto & v : finp) v = frand();
    for (auto & v : fbias) v = frand();
    std::vector<float> xin(n_embd);
    for (auto & v : xin) v = frand() * 4;

    std::vector<float> outs[2];
    for (int pass = 0; pass < 2; ++pass) {
        ggml_backend_t wbk = pass == 0 ? cl : cpu;
        ggml_context * wctx = ggml_init({ 1 << 22, nullptr, true });
        ggml_context * gctx = ggml_init({ 1 << 24, nullptr, true });
        struct L { ggml_tensor *up, *gate, *down, *inp, *bias; };
        std::vector<L> ls(n_layers);
        for (auto & l : ls) {
            l.up   = ggml_new_tensor_3d(wctx, type, n_embd, n_ff, n_exp);
            l.gate = ggml_new_tensor_3d(wctx, type, n_embd, n_ff, n_exp);
            l.down = ggml_new_tensor_3d(wctx, type, n_ff, n_embd, n_exp);
            l.inp  = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, n_embd, n_exp);
            l.bias = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, n_exp);
        }
        ggml_backend_buffer_t wb = ggml_backend_alloc_ctx_tensors(wctx, wbk);
        ggml_backend_buffer_set_usage(wb, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (auto & l : ls) {
            ggml_backend_tensor_set(l.up, qup.data(), 0, qup.size());
            ggml_backend_tensor_set(l.gate, qup.data(), 0, qup.size());
            ggml_backend_tensor_set(l.down, qdown.data(), 0, qdown.size());
            ggml_backend_tensor_set(l.inp, finp.data(), 0, finp.size() * 4);
            ggml_backend_tensor_set(l.bias, fbias.data(), 0, fbias.size() * 4);
        }
        ggml_tensor * x = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, n_embd, 1);
        ggml_set_input(x);
        ggml_tensor * cur = x;
        for (auto & l : ls) {
            ggml_tensor * h = ggml_rms_norm(gctx, cur, 1e-6f);
            ggml_tensor * probs = ggml_sigmoid(gctx, ggml_mul_mat(gctx, l.inp, h));             // [n_exp, 1]
            ggml_tensor * ids = ggml_argsort_top_k(gctx, ggml_add(gctx, probs, l.bias), n_used); // [n_used, 1]
            ggml_tensor * w = ggml_get_rows(gctx, ggml_reshape_3d(gctx, probs, 1, n_exp, 1), ids); // [1, n_used, 1]
            w = ggml_div(gctx, w, ggml_sum_rows(gctx, ggml_reshape_2d(gctx, w, n_used, 1)));
            ggml_tensor * h3 = ggml_reshape_3d(gctx, h, n_embd, 1, 1);
            ggml_tensor * up = ggml_mul_mat_id(gctx, l.up, h3, ids);
            ggml_tensor * gt = ggml_mul_mat_id(gctx, l.gate, h3, ids);
            ggml_tensor * act = ggml_swiglu_split(gctx, gt, up);
            ggml_tensor * dn = ggml_mul_mat_id(gctx, l.down, act, ids);                        // [n_embd, n_used, 1]
            dn = ggml_mul(gctx, dn, ggml_reshape_3d(gctx, w, 1, n_used, 1));
            ggml_tensor * sum = nullptr;
            for (int i = 0; i < n_used; ++i) {
                ggml_tensor * v = ggml_view_2d(gctx, dn, n_embd, 1, dn->nb[2], i * dn->nb[1]);
                sum = sum ? ggml_add(gctx, sum, v) : v;
            }
            cur = ggml_add(gctx, cur, ggml_scale(gctx, sum, 0.25f));
        }
        ggml_set_output(cur);
        ggml_cgraph * gf = ggml_new_graph(gctx);
        ggml_build_forward_expand(gf, cur);
        ggml_backend_t bk[2] = { wbk, cpu };
        ggml_backend_buffer_type_t bt[2] = { ggml_backend_get_default_buffer_type(wbk), ggml_backend_get_default_buffer_type(cpu) };
        ggml_backend_sched_t sched = ggml_backend_sched_new(bk, bt, wbk == cpu ? 1 : 2, 1024, false, false);
        bool ok = ggml_backend_sched_alloc_graph(sched, gf);
        if (ok) {
            ggml_backend_tensor_set(x, xin.data(), 0, n_embd * 4);
            ok = ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;
        }
        if (!ok) { printf("chain graph failed BAD\n"); return false; }
        outs[pass].resize(n_embd);
        ggml_backend_tensor_get(cur, outs[pass].data(), 0, n_embd * 4);
        ggml_backend_sched_free(sched);
        ggml_backend_buffer_free(wb);
        ggml_free(gctx);
        ggml_free(wctx);
    }
    double me = 0, mag = 0;
    for (int i = 0; i < n_embd; ++i) {
        me = std::max(me, (double) fabs(outs[0][i] - outs[1][i]));
        mag = std::max(mag, (double) fabs(outs[1][i]));
    }
    const bool good = me <= 0.05 * mag;
    printf("%-44s chain x%d max diff vs CPU %.3g (max |y| %.3g) %s\n", ggml_type_name(type), n_layers, me, mag, good ? "OK" : "BAD");
    return good;
}

int main() {
    ggml_backend_load_all();
    if (const char * dir = getenv("GGML_LIB_DIR")) {
        ggml_backend_load_all_from_path(dir);
    }
    ggml_backend_t cl = nullptr, cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu) { printf("no CPU backend (set GGML_LIB_DIR)\n"); return 1; }
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (strncmp(ggml_backend_dev_name(d), "CL11", 4) == 0) { cl = ggml_backend_dev_init(d, nullptr); break; }
    }
    if (!cl) { printf("no CL11 device\n"); return 1; }
    const cs cases[] = {
        { "q4_0 gate/up 32e top4 k2048 m1792",   GGML_TYPE_Q4_0, 32, 4, 2048, 1792, 1, 1 },
        { "q4_0 down    32e top4 k1792 m2048",   GGML_TYPE_Q4_0, 32, 4, 1792, 2048, 4, 1 },
        { "q8_0 gate/up 32e top4 k2048 m1792",   GGML_TYPE_Q8_0, 32, 4, 2048, 1792, 1, 1 },
        { "q8_0 down    32e top4 k1792 m2048",   GGML_TYPE_Q8_0, 32, 4, 1792, 2048, 4, 1 },
        { "q4_k gate/up 32e top4 k2048 m1792",   GGML_TYPE_Q4_K, 32, 4, 2048, 1792, 1, 1 },
        { "q6_k down    32e top4 k1792 m2048",   GGML_TYPE_Q6_K, 32, 4, 1792, 2048, 4, 1 },
        { "q4_0 small 8e top2 k256 m128",        GGML_TYPE_Q4_0, 8, 2, 256, 128, 1, 1 },
        { "q4_0 2 tokens 16e top2 k512 m256",    GGML_TYPE_Q4_0, 16, 2, 512, 256, 1, 2 },
    };
    int bad = 0;
    for (const auto & c : cases) {
        bad += run(cl, cpu, c, false) ? 0 : 1;
        bad += run(cl, cpu, c, true) ? 0 : 1;
    }
    bad += run_chain(cl, cpu, GGML_TYPE_Q4_0, 6) ? 0 : 1;
    printf("%s\n", bad ? "FAILED" : "all OK");
    return bad ? 1 : 0;
}

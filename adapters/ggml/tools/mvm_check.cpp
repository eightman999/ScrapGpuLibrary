// CL11 backend: check of the launch fusions at the graph level, against a double precision reference (test-backend-ops builds one node per graph,
// so it never fuses anything):
//  - multi mat-vec (k_mvm_*: MUL_MATs that read the same src1 in one launch, graph_optimize + graph_compute),
//  - two SET_ROWS (K and V cache write) in one launch (k_set_rows2_*).
//
//   out-of-tree plugin (llama.cpp built with -DGGML_BACKEND_DL=ON, CL11 built as libggml-cl11.so, see llama.cpp/README.md):
//     c++ -std=c++17 -O1 -I<llama.cpp>/ggml/include adapters/ggml/tools/mvm_check.cpp -L<llama.cpp>/build-dl/bin -lggml -lggml-base -Wl,-rpath,<llama.cpp>/build-dl/bin -o mvm_check
//     GGML_BACKEND_PATH=$PWD/build-cl11/libggml-cl11.so GGML_CL11_MULTI_MV=1 ./mvm_check      # every line must end with OK
//   GGML_CL11_LAUNCH_LOG=1 prints the number of kernels of every graph; "hash" is the hash of the bits of all the outputs. It can differ between
//   GGML_CL11_MULTI_MV=0 and 1 in the last bits: a repacked matrix whose slice size is not a multiple of 16 bytes (rows 97 / 65 ...) is computed
//   by the mat-mat kernel when it is not fused, and the compiler may contract the multiply-adds of the two kernels differently. MVM_DUMP=<file>
//   appends the raw outputs of every case to a file (to compare two runs).
//   GGML_CL11_DEVICES="GT 710" ./mvm_check  # one device;  GGML_LIB_DIR=<llama.cpp>/build-dl/bin if the CPU backend plugin is not next to the tool
//
// Every case builds the graph of an attention + FFN block the way llama.cpp does (the Q / K / V projections of the same x have other nodes between
// them: a scale and a RMS_NORM of Q and K), runs it through ggml_backend_sched (which calls graph_optimize, then allocates), and compares the
// projections with sum(dequantized weights * x) in double precision. Cases: the four repacked types, rows that are not a multiple of 32, 2 and 3
// weights, different row counts in one group, mixed types (not fused), weights in several buffers.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>
#include <cstdint>
#include <string>

static float frand() { return (float) rand() / (float) RAND_MAX - 0.5f; }

struct weight {
    ggml_type          type;
    int                k, m;
    std::vector<float> deq;     // dequantized, [m][k]
    ggml_tensor *      t = nullptr;
    std::vector<uint8_t> q;
};

static void make_weight(weight & w) {
    std::vector<float> f((size_t) w.k * w.m);
    for (auto & x : f) x = frand();
    const ggml_type_traits * tr = ggml_get_type_traits(w.type);
    w.q.resize(ggml_row_size(w.type, w.k) * w.m);
    ggml_quantize_chunk(w.type, f.data(), w.q.data(), 0, w.m, w.k, nullptr);
    w.deq.resize(f.size());
    for (int r = 0; r < w.m; ++r) {
        tr->to_float(w.q.data() + (size_t) r * ggml_row_size(w.type, w.k), w.deq.data() + (size_t) r * w.k, w.k);
    }
}

struct result {
    std::vector<std::vector<float>> out;
    uint64_t hash = 0;
};

// n_w weights, projections of the same x; `between` puts a SCALE / RMS_NORM of the previous projection between them (as llama.cpp does for Q and K)
static bool run_case(ggml_backend_t cl, ggml_backend_t cpu, std::vector<weight> & ws, bool between, bool split_ctx, const std::vector<float> & x, result & res) {
    const int K = ws[0].k;
    // weights in one or two contexts (buffers)
    ggml_init_params ip = { 1 << 26, nullptr, true };
    ggml_context * wctx0 = ggml_init(ip);
    ggml_context * wctx1 = ggml_init(ip);
    ggml_context * gctx = ggml_init({ 1 << 26, nullptr, true });
    for (size_t i = 0; i < ws.size(); ++i) {
        ws[i].t = ggml_new_tensor_2d(split_ctx && (i & 1) ? wctx1 : wctx0, ws[i].type, ws[i].k, ws[i].m);
    }
    ggml_backend_buffer_t wb0 = ggml_backend_alloc_ctx_tensors(wctx0, cl);
    ggml_backend_buffer_t wb1 = split_ctx ? ggml_backend_alloc_ctx_tensors(wctx1, cl) : nullptr;
    for (auto & w : ws) {
        ggml_backend_tensor_set(w.t, w.q.data(), 0, w.q.size());
    }
    ggml_tensor * xin = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, K, 1);
    ggml_set_input(xin);
    // the nodes of llama.cpp: x = rms_norm(xin) * w (so x is a computed node, not an input)
    ggml_tensor * xn = ggml_rms_norm(gctx, xin, 1e-6f);
    std::vector<ggml_tensor *> proj;
    ggml_cgraph * gf = ggml_new_graph_custom(gctx, 256, false);
    for (size_t i = 0; i < ws.size(); ++i) {
        ggml_tensor * p = ggml_mul_mat(gctx, ws[i].t, xn);
        proj.push_back(p);
        if (between) {
            ggml_tensor * r = ggml_rms_norm(gctx, ggml_scale(gctx, p, 0.5f), 1e-6f);
            ggml_set_output(r);
            ggml_build_forward_expand(gf, r);
        }
    }
    for (auto * p : proj) {
        ggml_set_output(p);
        ggml_build_forward_expand(gf, p);
    }
    ggml_backend_t backends[2] = { cl, cpu };
    ggml_backend_buffer_type_t bufts[2] = { ggml_backend_get_default_buffer_type(cl), ggml_backend_get_default_buffer_type(cpu) };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, 2, 256, false, false);
    bool ok = ggml_backend_sched_alloc_graph(sched, gf);
    if (ok) {
        // xin has no data yet: x is set as the input
        ggml_backend_tensor_set(xin, x.data(), 0, ggml_nbytes(xin));
        ok = ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;
    }
    res.out.clear();
    res.hash = 14695981039346656037ull;
    if (ok) {
        // the reference input: rms_norm(x) in double
        for (auto * p : proj) {
            std::vector<float> o(ggml_nelements(p));
            ggml_backend_tensor_get(p, o.data(), 0, ggml_nbytes(p));
            for (float v : o) {
                uint32_t b;
                memcpy(&b, &v, 4);
                res.hash = (res.hash ^ b) * 1099511628211ull;
            }
            res.out.push_back(std::move(o));
        }
    }
    ggml_backend_sched_free(sched);
    if (wb1) ggml_backend_buffer_free(wb1);
    ggml_backend_buffer_free(wb0);
    ggml_free(gctx);
    ggml_free(wctx1);
    ggml_free(wctx0);
    return ok;
}

// two SET_ROWS into two F16 / F32 caches (one with I64 and one with I32 row indices, other row counts), with a view in between as llama.cpp has
static int check_set_rows(ggml_backend_t cl, ggml_backend_t cpu) {
    struct sc { int emb_a, emb_b, ntok_a, ntok_b, rows; ggml_type dst; bool same_n; };
    const sc cases[] = { {1024, 1024, 1, 1, 512, GGML_TYPE_F16, true}, {300, 300, 5, 5, 64, GGML_TYPE_F16, true}, {128, 256, 12, 7, 100, GGML_TYPE_F16, false},
                         {1024, 1024, 1, 1, 512, GGML_TYPE_F32, true}, {64, 64, 33, 33, 40, GGML_TYPE_F32, true} };
    int bad = 0;
    for (const auto & c : cases) {
        ggml_context * ctx = ggml_init({ 1 << 24, nullptr, true });
        ggml_tensor * da = ggml_new_tensor_2d(ctx, c.dst, c.emb_a, c.rows);
        ggml_tensor * db = ggml_new_tensor_2d(ctx, c.dst, c.emb_b, c.rows);
        ggml_tensor * sa = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.emb_a, c.ntok_a);
        ggml_tensor * sb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.emb_b, c.ntok_b);
        ggml_tensor * ia = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, c.ntok_a);
        ggml_tensor * ib = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, c.ntok_b);
        ggml_tensor * ra = ggml_set_rows(ctx, da, sa, ia);
        ggml_tensor * vb = ggml_view_2d(ctx, sb, c.emb_b, c.ntok_b, sb->nb[1], 0);   // an empty node between the two SET_ROWS
        ggml_tensor * rb = ggml_set_rows(ctx, db, vb, ib);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, ra);
        ggml_build_forward_expand(gf, rb);
        ggml_backend_t backends[2] = { cl, cpu };
        ggml_backend_buffer_type_t bufts[2] = { ggml_backend_get_default_buffer_type(cl), ggml_backend_get_default_buffer_type(cpu) };
        ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, 2, 256, false, false);
        ggml_set_input(sa); ggml_set_input(sb); ggml_set_input(ia); ggml_set_input(ib);
        ggml_set_output(ra); ggml_set_output(rb);
        bool ok = ggml_backend_sched_alloc_graph(sched, gf);
        std::vector<float> xa((size_t) c.emb_a * c.ntok_a), xb((size_t) c.emb_b * c.ntok_b);
        for (auto & v : xa) v = frand();
        for (auto & v : xb) v = frand();
        std::vector<int64_t> pa(c.ntok_a);
        std::vector<int32_t> pb(c.ntok_b);
        for (int t = 0; t < c.ntok_a; ++t) pa[t] = (t * 7 + 3) % c.rows;
        for (int t = 0; t < c.ntok_b; ++t) pb[t] = (t * 3 + 1) % c.rows;
        std::vector<float> ca(ggml_nelements(da), 0.0f), cb(ggml_nelements(db), 0.0f);
        std::vector<float> oa(ca.size()), ob(cb.size());
        if (ok) {
            std::vector<uint8_t> za(ggml_nbytes(da), 0), zb(ggml_nbytes(db), 0);
            ggml_backend_tensor_set(da, za.data(), 0, za.size());
            ggml_backend_tensor_set(db, zb.data(), 0, zb.size());
            ggml_backend_tensor_set(sa, xa.data(), 0, ggml_nbytes(sa));
            ggml_backend_tensor_set(sb, xb.data(), 0, ggml_nbytes(sb));
            ggml_backend_tensor_set(ia, pa.data(), 0, ggml_nbytes(ia));
            ggml_backend_tensor_set(ib, pb.data(), 0, ggml_nbytes(ib));
            ok = ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;
        }
        double me = 0;
        if (ok) {
            std::vector<uint8_t> ra_(ggml_nbytes(da)), rb_(ggml_nbytes(db));
            ggml_backend_tensor_get(da, ra_.data(), 0, ra_.size());
            ggml_backend_tensor_get(db, rb_.data(), 0, rb_.size());
            auto get = [&](const std::vector<uint8_t> & raw, int i) -> float {
                return c.dst == GGML_TYPE_F16 ? ggml_fp16_to_fp32(((const ggml_fp16_t *) raw.data())[i]) : ((const float *) raw.data())[i];
            };
            // expected: rows that are not written stay 0, written rows hold the (rounded) source; later tokens win when two tokens share a row (they do not here)
            for (int t = 0; t < c.ntok_a; ++t) for (int e = 0; e < c.emb_a; ++e) {
                me = std::max(me, (double) fabs(get(ra_, (int) (pa[t] * c.emb_a + e)) - xa[(size_t) t * c.emb_a + e]));
            }
            for (int t = 0; t < c.ntok_b; ++t) for (int e = 0; e < c.emb_b; ++e) {
                me = std::max(me, (double) fabs(get(rb_, (int) (pb[t] * c.emb_b + e)) - xb[(size_t) t * c.emb_b + e]));
            }
        }
        const bool good = ok && me < (c.dst == GGML_TYPE_F16 ? 1e-3 : 1e-7);
        printf("set_rows x2 %s emb %d/%d tokens %d/%d rows %d: max err %.3g %s\n", c.dst == GGML_TYPE_F16 ? "f16" : "f32", c.emb_a, c.emb_b, c.ntok_a, c.ntok_b, c.rows, me, good ? "OK" : "BAD");
        bad += !good;
        ggml_backend_sched_free(sched);
        ggml_free(ctx);
    }
    return bad;
}

int main() {
    ggml_backend_load_all();
    // the CPU backend of a GGML_BACKEND_DL build is a plugin too: GGML_LIB_DIR=<llama.cpp>/build-dl/bin when the tool does not live next to it
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

    struct cs { const char * name; ggml_type t[3]; int n; int k; int m[3]; bool between; bool split; };
    const ggml_type Q4_0 = GGML_TYPE_Q4_0, Q8_0 = GGML_TYPE_Q8_0, Q4_K = GGML_TYPE_Q4_K, Q6_K = GGML_TYPE_Q6_K;
    const cs cases[] = {
        { "q8_0 qkv 2048/1024/1024",      { Q8_0, Q8_0, Q8_0 }, 3, 1024, { 2048, 1024, 1024 }, true,  false },
        { "q8_0 gate/up 3072",            { Q8_0, Q8_0, Q8_0 }, 2, 1024, { 3072, 3072, 0 },    false, false },
        { "q4_0 qkv",                     { Q4_0, Q4_0, Q4_0 }, 3, 512,  { 512, 128, 128 },    true,  false },
        { "q4_k qkv",                     { Q4_K, Q4_K, Q4_K }, 3, 768,  { 768, 256, 256 },    true,  false },
        { "q6_k qkv",                     { Q6_K, Q6_K, Q6_K }, 3, 512,  { 512, 256, 256 },    true,  false },
        { "q8_0 rows not x32 (1000,33,1)", { Q8_0, Q8_0, Q8_0 }, 3, 256,  { 1000, 33, 1 },      true,  false },
        { "q4_0 rows not x32 (97,31,65)", { Q4_0, Q4_0, Q4_0 }, 3, 128,  { 97, 31, 65 },       false, false },
        { "q4_k rows not x32 (70,5)",     { Q4_K, Q4_K, Q4_K }, 2, 512,  { 70, 5, 0 },         true,  false },
        { "q6_k rows not x32 (33,200)",   { Q6_K, Q6_K, Q6_K }, 2, 256,  { 33, 200, 0 },       false, false },
        { "q8_0 two buffers",             { Q8_0, Q8_0, Q8_0 }, 3, 512,  { 300, 200, 100 },    true,  true  },
        { "mixed q4_k/q6_k/q4_k (not fused)", { Q4_K, Q6_K, Q4_K }, 3, 512, { 256, 256, 256 }, true,  false },
        { "mixed q8_0/q4_0 (not fused)",  { Q8_0, Q4_0, Q8_0 }, 3, 256,  { 128, 128, 128 },    false, false },
        { "q8_0 small rows (64)",           { Q8_0, Q8_0, Q8_0 }, 3, 128,  { 64, 64, 64 },       true,  false },
    };
    int bad = 0;
    uint64_t total_hash = 14695981039346656037ull;
    for (const auto & c : cases) {
        std::vector<weight> ws(c.n);
        for (int i = 0; i < c.n; ++i) {
            ws[i].type = c.t[i];
            ws[i].k = c.k;
            ws[i].m = c.m[i];
            make_weight(ws[i]);
        }
        std::vector<float> x(c.k);
        for (auto & v : x) v = frand() * 4;
        result r;
        if (!run_case(cl, cpu, ws, c.between, c.split, x, r)) {
            printf("%-34s: graph failed BAD\n", c.name);
            ++bad;
            continue;
        }
        // reference: y = W * rms_norm(x)
        double ss = 0;
        for (float v : x) ss += (double) v * v;
        const double scale = 1.0 / sqrt(ss / c.k + 1e-6);
        double me = 0, mag = 0;
        for (int i = 0; i < c.n; ++i) {
            for (int row = 0; row < c.m[i]; ++row) {
                double s = 0;
                for (int k = 0; k < c.k; ++k) {
                    s += (double) ws[i].deq[(size_t) row * c.k + k] * ((double) x[k] * scale);
                }
                me = std::max(me, fabs(s - r.out[i][row]));
                mag = std::max(mag, fabs(s));
            }
        }
        if (const char * dump = getenv("MVM_DUMP")) { // debugging: the raw outputs of every case (compare two runs, e.g. with and without the fusion)
            FILE * f = fopen(dump, "ab");
            for (const auto & o : r.out) fwrite(o.data(), sizeof(float), o.size(), f);
            fclose(f);
        }
        const bool ok = me < 1e-3 * std::max(1.0, mag);
        printf("%-34s: max err %.3g (max |y| %.3g) hash %016llx %s\n", c.name, me, mag, (unsigned long long) r.hash, ok ? "OK" : "BAD");
        bad += !ok;
        total_hash = (total_hash ^ r.hash) * 1099511628211ull;
    }
    printf("total hash %016llx\n", (unsigned long long) total_hash);
    bad += check_set_rows(cl, cpu);
    return bad;
}

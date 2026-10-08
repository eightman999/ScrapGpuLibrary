// CL11 backend: throughput of MUL_MAT nodes (m x k weights times n columns) on the CL11 device, as CSV.
// test-backend-ops perf has no n = 128 (the prompt batch of `llama-bench -p 128`) and its output is hard to collect, so this tool times the same node.
//
//   out-of-tree plugin (see adapters/ggml/llama.cpp/README.md):
//     c++ -std=c++17 -O1 -I<llama.cpp>/ggml/include adapters/ggml/tools/mm_bench.cpp -L<llama.cpp>/build-dl/bin -lggml -lggml-base -Wl,-rpath,<llama.cpp>/build-dl/bin -o mm_bench
//     GGML_BACKEND_PATH=$PWD/build-cl11/libggml-cl11.so GGML_CL11_DEVICES="GT 730" ./mm_bench [--iters=N] q8_0:3072:128:1024 q4_0:4096:512:14336 ...
//   one job per argument: type:m:n:k (types: f32 f16 q4_0 q8_0 q4_k q6_k); the kernels are built once per process, so give all the jobs of a
//   configuration to one call.  output (stdout): type,m,n,k,iters,ms_per_iter,gflops   (GFLOPS = 2 m n k / time)
//   The first call of a job (upload, kernel build) is not timed; the queue is synchronized before the clock stops. Without --iters the number of
//   iterations is chosen so that a job lasts about a second on the GPU (one trial of 3 calls).
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static float frand() { return (float) rand() / (float) RAND_MAX - 0.5f; }

static bool run_job(ggml_backend_t cl, const std::string & ts, int64_t m, int64_t n, int64_t k, int iters) {
    ggml_type type;
    if      (ts == "f32")  type = GGML_TYPE_F32;
    else if (ts == "f16")  type = GGML_TYPE_F16;
    else if (ts == "q4_0") type = GGML_TYPE_Q4_0;
    else if (ts == "q8_0") type = GGML_TYPE_Q8_0;
    else if (ts == "q4_k") type = GGML_TYPE_Q4_K;
    else if (ts == "q6_k") type = GGML_TYPE_Q6_K;
    else { fprintf(stderr, "unknown type %s\n", ts.c_str()); return false; }

    ggml_context * ctx = ggml_init({ 1 << 20, nullptr, true });
    ggml_tensor * a = ggml_new_tensor_2d(ctx, type, k, m);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    ggml_tensor * c = ggml_mul_mat(ctx, a, b);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cl);
    if (!buf) {
        fprintf(stderr, "allocation failed for %s %lld x %lld x %lld\n", ts.c_str(), (long long) m, (long long) n, (long long) k);
        ggml_free(ctx);
        return false;
    }
    {
        // one random row quantized and repeated: the values do not matter for the speed, and quantizing 60 MB of floats is slow
        std::vector<float> row((size_t) k);
        for (auto & v : row) v = frand();
        std::vector<uint8_t> qrow(ggml_row_size(type, k));
        if (type == GGML_TYPE_F32) memcpy(qrow.data(), row.data(), qrow.size());
        else if (type == GGML_TYPE_F16) ggml_fp32_to_fp16_row(row.data(), (ggml_fp16_t *) qrow.data(), k);
        else ggml_quantize_chunk(type, row.data(), qrow.data(), 0, 1, k, nullptr);
        std::vector<uint8_t> all(ggml_nbytes(a));
        for (int64_t r = 0; r < m; ++r) memcpy(all.data() + (size_t) r * qrow.size(), qrow.data(), qrow.size());
        ggml_backend_tensor_set(a, all.data(), 0, all.size());
        std::vector<float> x((size_t) k * n);
        for (auto & v : x) v = frand();
        ggml_backend_tensor_set(b, x.data(), 0, ggml_nbytes(b));
    }
    ggml_backend_graph_compute(cl, gf);
    ggml_backend_synchronize(cl);
    auto now = [] { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
    if (iters <= 0) {
        const double t0 = now();
        for (int i = 0; i < 3; ++i) ggml_backend_graph_compute(cl, gf);
        ggml_backend_synchronize(cl);
        const double per = (now() - t0) / 3;
        iters = (int) std::max(3.0, std::min(2000.0, 1.0 / std::max(per, 1e-6)));
    }
    const double t0 = now();
    for (int i = 0; i < iters; ++i) ggml_backend_graph_compute(cl, gf);
    ggml_backend_synchronize(cl);
    const double dt = (now() - t0) / iters;
    printf("%s,%lld,%lld,%lld,%d,%.4f,%.1f\n", ts.c_str(), (long long) m, (long long) n, (long long) k, iters, dt * 1e3, 2.0 * m * n * k / dt / 1e9);
    fflush(stdout);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

int main(int argc, char ** argv) {
    int iters = 0;
    std::vector<std::string> jobs;
    for (int i = 1; i < argc; ++i) {
        if (strncmp(argv[i], "--iters=", 8) == 0) {
            iters = atoi(argv[i] + 8);
        } else {
            jobs.push_back(argv[i]);
        }
    }
    if (jobs.empty()) {
        fprintf(stderr, "usage: %s [--iters=N] type:m:n:k ...\n", argv[0]);
        return 2;
    }

    ggml_backend_load_all();
    ggml_backend_t cl = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (strncmp(ggml_backend_dev_name(d), "CL11", 4) == 0) { cl = ggml_backend_dev_init(d, nullptr); break; }
    }
    if (!cl) { fprintf(stderr, "no CL11 device\n"); return 1; }

    int bad = 0;
    for (const auto & j : jobs) {
        char t[16] = {};
        long long m = 0, n = 0, k = 0;
        if (sscanf(j.c_str(), "%15[^:]:%lld:%lld:%lld", t, &m, &n, &k) != 4) {
            fprintf(stderr, "bad job '%s' (type:m:n:k)\n", j.c_str());
            return 2;
        }
        bad += !run_job(cl, t, m, n, k, iters);
    }
    return bad;
}

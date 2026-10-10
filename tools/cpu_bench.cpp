// Public-API CPU matmul benchmark. Input preparation is outside the timed region,
// so this file can also be built against an older scrapgpu_core for comparison.
#include "scrapgpu/reference.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct DtypeCase {
    const char * name;
    scrapgpu_dtype dtype;
};

const DtypeCase dtypes[] = {
    {"F32", SCRAPGPU_DTYPE_F32}, {"F16", SCRAPGPU_DTYPE_F16},
    {"Q4_0", SCRAPGPU_DTYPE_Q4_0}, {"Q8_0", SCRAPGPU_DTYPE_Q8_0},
    {"Q4_K", SCRAPGPU_DTYPE_Q4_K}, {"Q6_K", SCRAPGPU_DTYPE_Q6_K},
};

scrapgpu_tensor_desc tensor(scrapgpu_dtype dtype, int64_t k, int64_t rows) {
    scrapgpu_tensor_desc t{};
    t.struct_size = sizeof(t);
    t.dtype = dtype;
    t.layout = SCRAPGPU_LAYOUT_NATURAL;
    t.ne[0] = k;
    t.ne[1] = rows;
    t.ne[2] = t.ne[3] = 1;
    t.nb[0] = scrapgpu_dtype_type_size(dtype);
    t.nb[1] = scrapgpu_tensor_row_size(dtype, k);
    t.nb[2] = t.nb[3] = t.nb[1] * static_cast<uint64_t>(rows);
    return t;
}

void check(scrapgpu_status st) {
    if (st != SCRAPGPU_STATUS_OK) {
        throw std::runtime_error(std::string(scrapgpu_status_string(st)) + ": " + scrapgpu_last_error());
    }
}

void run(const DtypeCase & c, int64_t k, int64_t m, int64_t n, int iterations, int warmup) {
    const size_t block = scrapgpu_dtype_block_size(c.dtype);
    if (k % static_cast<int64_t>(block) != 0) {
        throw std::runtime_error(std::string(c.name) + " requires K divisible by " + std::to_string(block));
    }
    scrapgpu_tensor_desc inputs[2] = {tensor(c.dtype, k, m), tensor(SCRAPGPU_DTYPE_F32, k, n)};
    scrapgpu_tensor_desc output = tensor(SCRAPGPU_DTYPE_F32, m, n);
    std::vector<float> weights(static_cast<size_t>(k * m));
    std::vector<float> activations(static_cast<size_t>(k * n));
    std::vector<float> result(static_cast<size_t>(m * n));
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = static_cast<float>(static_cast<int>((i * 37 + 11) % 257) - 128) / 64.0f;
    }
    for (size_t i = 0; i < activations.size(); ++i) {
        activations[i] = static_cast<float>(static_cast<int>((i * 17 + 3) % 251) - 125) / 128.0f;
    }
    std::vector<uint8_t> encoded;
    void * weight_data = weights.data();
    if (c.dtype != SCRAPGPU_DTYPE_F32) {
        encoded.resize(static_cast<size_t>(inputs[0].nb[2]));
        weight_data = encoded.data();
        if (c.dtype == SCRAPGPU_DTYPE_F16) {
            scrapgpu_tensor_desc source = tensor(SCRAPGPU_DTYPE_F32, k, m);
            scrapgpu_cpy_params params{sizeof(params), 0};
            scrapgpu_op_desc copy{};
            copy.struct_size = sizeof(copy);
            copy.op = SCRAPGPU_OP_CPY;
            copy.inputs = &source;
            copy.input_count = 1;
            copy.outputs = &inputs[0];
            copy.output_count = 1;
            copy.params = &params;
            copy.params_size = sizeof(params);
            void * from[] = {weights.data()};
            void * to[] = {encoded.data()};
            check(scrapgpu_ref_execute(&copy, from, to));
        } else {
            check(scrapgpu_ref_quantize(c.dtype, weights.data(), encoded.data(), k * m));
        }
    }
    // Release the temporary F32 source for compressed cases before measuring.
    if (c.dtype != SCRAPGPU_DTYPE_F32) {
        std::vector<float>().swap(weights);
    }
    scrapgpu_mul_mat_params params{sizeof(params), 0};
    scrapgpu_op_desc op{};
    op.struct_size = sizeof(op);
    op.op = SCRAPGPU_OP_MUL_MAT;
    op.inputs = inputs;
    op.input_count = 2;
    op.outputs = &output;
    op.output_count = 1;
    op.params = &params;
    op.params_size = sizeof(params);
    void * in[] = {weight_data, activations.data()};
    void * out[] = {result.data()};
    for (int i = 0; i < warmup; ++i) {
        check(scrapgpu_ref_execute(&op, in, out));
    }
    std::vector<double> ms;
    ms.reserve(static_cast<size_t>(iterations));
    for (int i = 0; i < iterations; ++i) {
        const auto start = std::chrono::steady_clock::now();
        check(scrapgpu_ref_execute(&op, in, out));
        const auto stop = std::chrono::steady_clock::now();
        ms.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
    }
    double checksum = 0.0;
    for (size_t i = 0; i < result.size(); ++i) {
        if (!std::isfinite(result[i])) {
            throw std::runtime_error("non-finite matmul result");
        }
        checksum += static_cast<double>(result[i]) * static_cast<double>(i % 17 + 1);
    }
    double mean = 0.0;
    for (double value : ms) mean += value / iterations;
    std::sort(ms.begin(), ms.end());
    const double median = iterations % 2 ? ms[iterations / 2] : (ms[iterations / 2 - 1] + ms[iterations / 2]) / 2;
    const double gflops = 2.0 * static_cast<double>(k) * static_cast<double>(m) * static_cast<double>(n) / (median * 1e6);
    std::printf("%s,%lld,%lld,%lld,%d,%.6f,%.6f,%.6f,%.6f,%.9g,%llu\n",
                c.name, static_cast<long long>(k), static_cast<long long>(m), static_cast<long long>(n),
                iterations, ms.front(), median, mean, gflops, checksum,
                static_cast<unsigned long long>(inputs[0].nb[2]));
    std::fflush(stdout);
}

int64_t number(const char * value) {
    size_t consumed = 0;
    const std::string text(value);
    const int64_t result = std::stoll(text, &consumed);
    if (consumed != text.size() || result < 1) throw std::runtime_error("expected a positive integer");
    return result;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        int64_t k = 2048, m = 512, n = 0;
        int iterations = 7, warmup = 2;
        std::string selected = "all";
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            if (arg == "--help") {
                std::printf("Usage: %s [--k 2048] [--m 512] [--n N] [--iterations 7] [--warmup 2] [--dtype all|F32|F16|Q4_0|Q8_0|Q4_K|Q6_K]\n"
                            "Without --n, measures N=1 and N=16. Times include per-call scratch allocation and weight conversion.\n"
                            "Quantization/input generation is excluded. Build in Release without fast-math for comparable results.\n", argv[0]);
                return 0;
            }
            if (i + 1 >= argc) throw std::runtime_error("missing option value");
            const char * value = argv[++i];
            if (arg == "--k") k = number(value);
            else if (arg == "--m") m = number(value);
            else if (arg == "--n") n = number(value);
            else if (arg == "--dtype") selected = value;
            else if (arg == "--iterations" || arg == "--warmup") {
                const int64_t count = number(value);
                if (count > 100000) throw std::runtime_error("iteration count is too large");
                if (arg == "--iterations") iterations = static_cast<int>(count);
                else warmup = static_cast<int>(count);
            } else throw std::runtime_error("unknown option: " + arg);
        }
        const int64_t largest_n = n ? n : 16;
        const int64_t max_elements = static_cast<int64_t>((std::numeric_limits<size_t>::max)() / sizeof(float));
        if (k > max_elements / m || k > max_elements / largest_n || m > max_elements / largest_n) {
            throw std::runtime_error("tensor size overflows address space");
        }
        bool matched = false;
        std::printf("dtype,k,m,n,iterations,min_ms,median_ms,mean_ms,gflops,checksum,weights_bytes\n");
        for (const DtypeCase & c : dtypes) {
            if (selected != "all" && selected != c.name) continue;
            matched = true;
            if (n) run(c, k, m, n, iterations, warmup);
            else {
                run(c, k, m, 1, iterations, warmup);
                run(c, k, m, 16, iterations, warmup);
            }
        }
        if (!matched) throw std::runtime_error("unknown dtype: " + selected);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "cpu_bench: %s\n", e.what());
        return 1;
    }
    return 0;
}

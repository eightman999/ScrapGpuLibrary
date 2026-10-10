#include "scrapgpu/attention.h"

#include "scrapgpu/buffer.h"
#include "scrapgpu/device.h"
#include "scrapgpu/opencl11.h"
#include "scrapgpu/status.h"
#include "scrapgpu/stream.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

extern "C" scrapgpu_status scrapgpu_cpu_register(void);

#define CHECK(cond)                                                                                     \
    do {                                                                                                \
        if (!(cond)) {                                                                                  \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond);              \
            std::exit(1);                                                                               \
        }                                                                                               \
    } while (0)

#define CHECK_OK(status)                                                                                \
    do {                                                                                                \
        const scrapgpu_status st_ = (status);                                                           \
        if (st_ != SCRAPGPU_STATUS_OK) {                                                                \
            std::fprintf(stderr, "%s:%d: unexpected status %s: %s\n",                                   \
                         __FILE__, __LINE__, scrapgpu_status_string(st_), scrapgpu_last_error());       \
            std::exit(1);                                                                               \
        }                                                                                               \
    } while (0)

namespace {

constexpr double kTol = 3e-4;

scrapgpu_device_t * g_device = nullptr;
scrapgpu_stream_t * g_stream = nullptr;
int g_run = 0;

size_t idx4(int64_t i0, int64_t i1, int64_t i2, int64_t i3,
            int64_t n0, int64_t n1, int64_t n2) {
    return static_cast<size_t>((((i3 * n2) + i2) * n1 + i1) * n0 + i0);
}

bool is_masked_value(double v) {
    return std::isinf(v) && v < 0.0;
}

scrapgpu_tensor_desc tdesc(int64_t n0, int64_t n1, int64_t n2 = 1, int64_t n3 = 1) {
    scrapgpu_tensor_desc t{};
    t.struct_size = sizeof(t);
    t.dtype = SCRAPGPU_DTYPE_F32;
    t.ne[0] = n0;
    t.ne[1] = n1;
    t.ne[2] = n2;
    t.ne[3] = n3;
    t.nb[0] = sizeof(float);
    t.nb[1] = scrapgpu_tensor_row_size(SCRAPGPU_DTYPE_F32, n0);
    t.nb[2] = t.nb[1] * static_cast<uint64_t>(n1);
    t.nb[3] = t.nb[2] * static_cast<uint64_t>(n2);
    t.layout = SCRAPGPU_LAYOUT_NATURAL;
    return t;
}

struct OwnedTensor {
    scrapgpu_tensor_desc desc{};
    scrapgpu_buffer_t * buffer = nullptr;
    std::vector<float> host;

    OwnedTensor() = default;
    OwnedTensor(const OwnedTensor &) = delete;
    OwnedTensor & operator=(const OwnedTensor &) = delete;
    OwnedTensor(OwnedTensor && other) noexcept
        : desc(other.desc), buffer(other.buffer), host(std::move(other.host)) {
        other.desc = {};
        other.buffer = nullptr;
    }
    OwnedTensor & operator=(OwnedTensor && other) noexcept {
        if (this != &other) {
            scrapgpu_buffer_free(buffer);
            desc = other.desc;
            buffer = other.buffer;
            host = std::move(other.host);
            other.desc = {};
            other.buffer = nullptr;
        }
        return *this;
    }
    ~OwnedTensor() {
        scrapgpu_buffer_free(buffer);
    }
};

OwnedTensor make_tensor(int64_t n0, int64_t n1, int64_t n2, int64_t n3, const std::vector<float> & host) {
    OwnedTensor t;
    t.desc = tdesc(n0, n1, n2, n3);
    t.host = host;
    CHECK(t.host.size() == static_cast<size_t>(n0 * n1 * n2 * n3));
    CHECK_OK(scrapgpu_buffer_alloc(g_device, t.host.size() * sizeof(float), &t.buffer));
    t.desc.buffer = t.buffer;
    CHECK_OK(scrapgpu_tensor_upload(g_stream, &t.desc, t.host.data(), t.host.size() * sizeof(float)));
    return t;
}

OwnedTensor make_output_tensor(int64_t n0, int64_t n1, int64_t n2, int64_t n3) {
    return make_tensor(n0, n1, n2, n3, std::vector<float>(static_cast<size_t>(n0 * n1 * n2 * n3), -99.0f));
}

void download_tensor(OwnedTensor & t) {
    CHECK_OK(scrapgpu_tensor_download(g_stream, &t.desc, t.host.data(), t.host.size() * sizeof(float)));
}

uint64_t workspace_for_tile(int64_t k, int64_t dv, int64_t tile_q, bool need_mask_workspace) {
    const uint64_t per_tile = static_cast<uint64_t>(k) * sizeof(float) * static_cast<uint64_t>(1 + (need_mask_workspace ? 1 : 0));
    return static_cast<uint64_t>(k) * static_cast<uint64_t>(dv) * sizeof(float) +
           per_tile * static_cast<uint64_t>(tile_q);
}

struct AttentionHostInputs {
    int64_t d = 0;
    int64_t q = 0;
    int64_t k = 0;
    int64_t dv = 0;
    int64_t hq = 0;
    int64_t hkv = 0;
    int64_t batches = 0;
    bool causal = false;
    uint64_t query_position_offset = 0;
    int64_t mask_h = 1;
    int64_t mask_b = 1;
    const std::vector<float> * mask = nullptr;
    const std::vector<float> * query = nullptr;
    const std::vector<float> * key = nullptr;
    const std::vector<float> * value = nullptr;
};

std::vector<double> oracle_attention(const AttentionHostInputs & in) {
    const double scale = 1.0 / std::sqrt(static_cast<double>(in.d));
    const int64_t heads_per_kv = in.hq / in.hkv;
    std::vector<double> out(static_cast<size_t>(in.dv * in.q * in.hq * in.batches), 0.0);
    std::vector<double> scores(static_cast<size_t>(in.k), 0.0);
    std::vector<double> probs(static_cast<size_t>(in.k), 0.0);

    for (int64_t b = 0; b < in.batches; ++b) {
        for (int64_t h = 0; h < in.hq; ++h) {
            const int64_t kv_head = h / heads_per_kv;
            for (int64_t qpos = 0; qpos < in.q; ++qpos) {
                bool any_open = false;
                double max_score = -std::numeric_limits<double>::infinity();
                for (int64_t kpos = 0; kpos < in.k; ++kpos) {
                    bool masked = in.causal && static_cast<uint64_t>(kpos) > in.query_position_offset + static_cast<uint64_t>(qpos);
                    double add = 0.0;
                    if (in.mask != nullptr) {
                        const int64_t mh = in.mask_h == 1 ? 0 : h;
                        const int64_t mb = in.mask_b == 1 ? 0 : b;
                        add = (*in.mask)[idx4(kpos, qpos, mh, mb, in.k, in.q, in.mask_h)];
                        if (is_masked_value(add)) {
                            masked = true;
                        }
                    }
                    if (masked) {
                        scores[static_cast<size_t>(kpos)] = -std::numeric_limits<double>::infinity();
                        continue;
                    }
                    double dot = 0.0;
                    for (int64_t d = 0; d < in.d; ++d) {
                        const double qv = (*in.query)[idx4(d, qpos, h, b, in.d, in.q, in.hq)];
                        const double kv = (*in.key)[idx4(d, kpos, kv_head, b, in.d, in.k, in.hkv)];
                        dot += qv * kv;
                    }
                    scores[static_cast<size_t>(kpos)] = dot * scale + add;
                    max_score = std::max(max_score, scores[static_cast<size_t>(kpos)]);
                    any_open = true;
                }
                if (!any_open) {
                    continue;
                }
                double sum = 0.0;
                for (int64_t kpos = 0; kpos < in.k; ++kpos) {
                    if (!std::isfinite(scores[static_cast<size_t>(kpos)])) {
                        probs[static_cast<size_t>(kpos)] = 0.0;
                        continue;
                    }
                    probs[static_cast<size_t>(kpos)] = std::exp(scores[static_cast<size_t>(kpos)] - max_score);
                    sum += probs[static_cast<size_t>(kpos)];
                }
                CHECK(sum > 0.0);
                for (int64_t kpos = 0; kpos < in.k; ++kpos) {
                    probs[static_cast<size_t>(kpos)] /= sum;
                }
                for (int64_t dv = 0; dv < in.dv; ++dv) {
                    double acc = 0.0;
                    for (int64_t kpos = 0; kpos < in.k; ++kpos) {
                        const double vv = (*in.value)[idx4(dv, kpos, kv_head, b, in.dv, in.k, in.hkv)];
                        acc += vv * probs[static_cast<size_t>(kpos)];
                    }
                    out[idx4(dv, qpos, h, b, in.dv, in.q, in.hq)] = acc;
                }
            }
        }
    }
    return out;
}

void expect_close(const std::string & name, const std::vector<float> & actual, const std::vector<double> & expected) {
    CHECK(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        const double diff = std::fabs(static_cast<double>(actual[i]) - expected[i]);
        if (diff > kTol) {
            std::fprintf(stderr, "%s: mismatch at %zu, got %.9g expected %.9g diff %.9g\n",
                         name.c_str(), i, static_cast<double>(actual[i]), expected[i], diff);
            std::exit(1);
        }
    }
}

scrapgpu_status run_attention(const scrapgpu_tensor_desc & query, const scrapgpu_tensor_desc & key,
                              const scrapgpu_tensor_desc & value, const scrapgpu_tensor_desc * mask,
                              const scrapgpu_tensor_desc & output, uint64_t budget,
                              bool causal, uint64_t query_position_offset) {
    scrapgpu_attention_params p{};
    p.struct_size = sizeof(p);
    p.query = &query;
    p.key = &key;
    p.value = &value;
    p.mask = mask;
    p.output = &output;
    p.query_position_offset = query_position_offset;
    p.max_workspace_bytes = budget;
    p.causal = causal ? 1 : 0;
    p.scale = 1.0f / std::sqrt(static_cast<float>(query.ne[0]));
    return scrapgpu_scaled_dot_product_attention(g_device, g_stream, &p);
}

void test_noncausal_gqa_tiled() {
    const int64_t d = 3, q = 5, k = 4, dv = 2, hq = 4, hkv = 2, b = 1;
    std::vector<float> qv(static_cast<size_t>(d * q * hq * b));
    std::vector<float> kv(static_cast<size_t>(d * k * hkv * b));
    std::vector<float> vv(static_cast<size_t>(dv * k * hkv * b));
    for (int64_t batch = 0; batch < b; ++batch) {
        for (int64_t h = 0; h < hq; ++h) {
            for (int64_t qp = 0; qp < q; ++qp) {
                for (int64_t i = 0; i < d; ++i) {
                    qv[idx4(i, qp, h, batch, d, q, hq)] =
                        static_cast<float>(0.11 * (i + 1) + 0.07 * qp + 0.03 * h - 0.02 * batch);
                }
            }
        }
    }
    for (int64_t batch = 0; batch < b; ++batch) {
        for (int64_t h = 0; h < hkv; ++h) {
            for (int64_t kp = 0; kp < k; ++kp) {
                for (int64_t i = 0; i < d; ++i) {
                    kv[idx4(i, kp, h, batch, d, k, hkv)] =
                        static_cast<float>(0.05 * (i + 1) - 0.04 * kp + 0.09 * h + 0.01 * batch);
                }
                for (int64_t j = 0; j < dv; ++j) {
                    vv[idx4(j, kp, h, batch, dv, k, hkv)] =
                        static_cast<float>(0.13 * (j + 1) + 0.02 * kp - 0.03 * h + 0.01 * batch);
                }
            }
        }
    }

    OwnedTensor query = make_tensor(d, q, hq, b, qv);
    OwnedTensor key = make_tensor(d, k, hkv, b, kv);
    OwnedTensor value = make_tensor(dv, k, hkv, b, vv);
    OwnedTensor output = make_output_tensor(dv, q, hq, b);

    CHECK_OK(run_attention(query.desc, key.desc, value.desc, nullptr, output.desc,
                           workspace_for_tile(k, dv, 2, false), false, 0));
    CHECK_OK(scrapgpu_stream_sync(g_stream));
    download_tensor(output);

    AttentionHostInputs in;
    in.d = d; in.q = q; in.k = k; in.dv = dv; in.hq = hq; in.hkv = hkv; in.batches = b;
    in.query = &qv; in.key = &kv; in.value = &vv;
    expect_close("noncausal_gqa_tiled", output.host, oracle_attention(in));
    std::printf("PASS  %-24s\n", "noncausal_gqa_tiled");
    ++g_run;
}

void test_causal_decode_offset() {
    const int64_t d = 4, q = 3, k = 6, dv = 3, hq = 2, hkv = 1, b = 2;
    std::vector<float> qv(static_cast<size_t>(d * q * hq * b));
    std::vector<float> kv(static_cast<size_t>(d * k * hkv * b));
    std::vector<float> vv(static_cast<size_t>(dv * k * hkv * b));
    for (int64_t batch = 0; batch < b; ++batch) {
        for (int64_t h = 0; h < hq; ++h) {
            for (int64_t qp = 0; qp < q; ++qp) {
                for (int64_t i = 0; i < d; ++i) {
                    qv[idx4(i, qp, h, batch, d, q, hq)] =
                        static_cast<float>(0.03 * (i + 1) + 0.05 * qp + 0.07 * h + 0.02 * batch);
                }
            }
        }
    }
    for (int64_t batch = 0; batch < b; ++batch) {
        for (int64_t kp = 0; kp < k; ++kp) {
            for (int64_t i = 0; i < d; ++i) {
                kv[idx4(i, kp, 0, batch, d, k, hkv)] =
                    static_cast<float>(0.04 * (i + 1) - 0.06 * kp + 0.01 * batch);
            }
            for (int64_t j = 0; j < dv; ++j) {
                vv[idx4(j, kp, 0, batch, dv, k, hkv)] =
                    static_cast<float>(0.09 * (j + 1) + 0.03 * kp - 0.02 * batch);
            }
        }
    }

    OwnedTensor query = make_tensor(d, q, hq, b, qv);
    OwnedTensor key = make_tensor(d, k, hkv, b, kv);
    OwnedTensor value = make_tensor(dv, k, hkv, b, vv);
    OwnedTensor output = make_output_tensor(dv, q, hq, b);

    CHECK_OK(run_attention(query.desc, key.desc, value.desc, nullptr, output.desc,
                           workspace_for_tile(k, dv, 2, true), true, 2));
    CHECK_OK(scrapgpu_stream_sync(g_stream));
    download_tensor(output);

    AttentionHostInputs in;
    in.d = d; in.q = q; in.k = k; in.dv = dv; in.hq = hq; in.hkv = hkv; in.batches = b;
    in.causal = true;
    in.query_position_offset = 2;
    in.query = &qv; in.key = &kv; in.value = &vv;
    expect_close("causal_decode_offset", output.host, oracle_attention(in));
    std::printf("PASS  %-24s\n", "causal_decode_offset");
    ++g_run;
}

void test_masked_rows_zero() {
    const int64_t d = 3, q = 4, k = 5, dv = 4, hq = 2, hkv = 2, b = 1;
    std::vector<float> qv(static_cast<size_t>(d * q * hq * b));
    std::vector<float> kv(static_cast<size_t>(d * k * hkv * b));
    std::vector<float> vv(static_cast<size_t>(dv * k * hkv * b));
    std::vector<float> mv(static_cast<size_t>(k * q * hq * b), 0.0f);

    for (int64_t h = 0; h < hq; ++h) {
        for (int64_t qp = 0; qp < q; ++qp) {
            for (int64_t i = 0; i < d; ++i) {
                qv[idx4(i, qp, h, 0, d, q, hq)] = static_cast<float>(0.02 * (i + 1) + 0.06 * qp + 0.08 * h);
            }
        }
    }
    for (int64_t h = 0; h < hkv; ++h) {
        for (int64_t kp = 0; kp < k; ++kp) {
            for (int64_t i = 0; i < d; ++i) {
                kv[idx4(i, kp, h, 0, d, k, hkv)] = static_cast<float>(0.04 * (i + 1) - 0.01 * kp + 0.03 * h);
            }
            for (int64_t j = 0; j < dv; ++j) {
                vv[idx4(j, kp, h, 0, dv, k, hkv)] = static_cast<float>(0.07 * (j + 1) + 0.02 * kp - 0.01 * h);
            }
        }
    }

    for (int64_t kp = 0; kp < k; ++kp) {
        mv[idx4(kp, 1, 0, 0, k, q, hq)] = -std::numeric_limits<float>::infinity();
        mv[idx4(kp, 3, 1, 0, k, q, hq)] = -std::numeric_limits<float>::infinity();
    }
    mv[idx4(4, 0, 0, 0, k, q, hq)] = -std::numeric_limits<float>::infinity();
    mv[idx4(0, 2, 1, 0, k, q, hq)] = -std::numeric_limits<float>::infinity();
    mv[idx4(1, 2, 1, 0, k, q, hq)] = -std::numeric_limits<float>::infinity();

    OwnedTensor query = make_tensor(d, q, hq, b, qv);
    OwnedTensor key = make_tensor(d, k, hkv, b, kv);
    OwnedTensor value = make_tensor(dv, k, hkv, b, vv);
    OwnedTensor mask = make_tensor(k, q, hq, b, mv);
    OwnedTensor output = make_output_tensor(dv, q, hq, b);

    CHECK_OK(run_attention(query.desc, key.desc, value.desc, &mask.desc, output.desc,
                           workspace_for_tile(k, dv, 3, true), false, 0));
    CHECK_OK(scrapgpu_stream_sync(g_stream));
    download_tensor(output);

    AttentionHostInputs in;
    in.d = d; in.q = q; in.k = k; in.dv = dv; in.hq = hq; in.hkv = hkv; in.batches = b;
    in.mask_h = hq;
    in.mask_b = b;
    in.mask = &mv;
    in.query = &qv; in.key = &kv; in.value = &vv;
    expect_close("masked_rows_zero", output.host, oracle_attention(in));

    for (int64_t dv_i = 0; dv_i < dv; ++dv_i) {
        CHECK(std::fabs(output.host[idx4(dv_i, 1, 0, 0, dv, q, hq)]) <= kTol);
        CHECK(std::fabs(output.host[idx4(dv_i, 3, 1, 0, dv, q, hq)]) <= kTol);
    }

    std::printf("PASS  %-24s\n", "masked_rows_zero");
    ++g_run;
}

void test_invalid_inputs_and_budget() {
    const int64_t d = 2, q = 2, k = 3, dv = 2, hq = 4, hkv = 2, b = 1;
    const std::vector<float> qv(static_cast<size_t>(d * q * hq * b), 0.25f);
    const std::vector<float> kv(static_cast<size_t>(d * k * hkv * b), -0.5f);
    const std::vector<float> vv(static_cast<size_t>(dv * k * hkv * b), 0.75f);
    const std::vector<float> mv(static_cast<size_t>(k * q * hq * b), 0.0f);

    OwnedTensor query = make_tensor(d, q, hq, b, qv);
    OwnedTensor key = make_tensor(d, k, hkv, b, kv);
    OwnedTensor value = make_tensor(dv, k, hkv, b, vv);
    OwnedTensor mask = make_tensor(k, q, hq, b, mv);
    OwnedTensor output = make_output_tensor(dv, q, hq, b);

    CHECK(run_attention(query.desc, key.desc, value.desc, nullptr, output.desc,
                        workspace_for_tile(k, dv, 1, false) - 1, false, 0) == SCRAPGPU_STATUS_OUT_OF_MEMORY);

    scrapgpu_tensor_desc bad_query = query.desc;
    bad_query.ne[2] = 3;
    CHECK(run_attention(bad_query, key.desc, value.desc, nullptr, output.desc,
                        workspace_for_tile(k, dv, 1, false), false, 0) == SCRAPGPU_STATUS_INVALID_ARGUMENT);

    scrapgpu_tensor_desc bad_output = output.desc;
    bad_output.ne[0] = dv + 1;
    CHECK(run_attention(query.desc, key.desc, value.desc, nullptr, bad_output,
                        workspace_for_tile(k, dv, 1, false), false, 0) == SCRAPGPU_STATUS_INVALID_ARGUMENT);

    scrapgpu_tensor_desc bad_mask = mask.desc;
    bad_mask.ne[2] = hkv;
    CHECK(run_attention(query.desc, key.desc, value.desc, &bad_mask, output.desc,
                        workspace_for_tile(k, dv, 1, true), false, 0) == SCRAPGPU_STATUS_INVALID_ARGUMENT);

    std::printf("PASS  %-24s\n", "invalid_inputs_budget");
    ++g_run;
}

int register_backend_from_env() {
    const char * backend = std::getenv("SCRAPGPU_TEST_BACKEND");
    if (backend == nullptr || backend[0] == '\0') {
        backend = "cpu";
    }
    if (std::strcmp(backend, "cpu") == 0) {
        if (scrapgpu_cpu_register() != SCRAPGPU_STATUS_OK) {
            std::fprintf(stderr, "scrapgpu_cpu_register failed: %s\n", scrapgpu_last_error());
            return 1;
        }
        return 0;
    }
    if (std::strcmp(backend, "opencl11") == 0) {
#if defined(SCRAPGPU_ATTENTION_TEST_WITH_OPENCL11) || defined(SCRAPGPU_OPS_TEST_WITH_OPENCL11)
        if (scrapgpu_opencl11_register() != SCRAPGPU_STATUS_OK) {
            std::fprintf(stderr, "scrapgpu_opencl11_register failed: %s\n", scrapgpu_last_error());
            return 1;
        }
        return 0;
#else
        std::printf("SKIP: built without the OpenCL 1.1 backend\n");
        return 77;
#endif
    }
    std::fprintf(stderr, "unknown SCRAPGPU_TEST_BACKEND '%s'\n", backend);
    return 1;
}

} // namespace

int main() {
    const int reg = register_backend_from_env();
    if (reg != 0) {
        return reg;
    }
    if (scrapgpu_get_device_count() == 0) {
        std::printf("SKIP: no device\n");
        return 77;
    }
    CHECK_OK(scrapgpu_get_device(0, &g_device));
    CHECK_OK(scrapgpu_stream_create(g_device, &g_stream));

    scrapgpu_device_info info{};
    CHECK_OK(scrapgpu_device_info_init(&info, sizeof(info)));
    CHECK_OK(scrapgpu_device_get_info(g_device, &info));
    std::printf("device: %s (%s)\n", info.name, info.backend_name);

    test_noncausal_gqa_tiled();
    test_causal_decode_offset();
    test_masked_rows_zero();
    test_invalid_inputs_and_budget();

    CHECK_OK(scrapgpu_stream_free(g_stream));
    std::printf("%d attention runs\n", g_run);
    return 0;
}

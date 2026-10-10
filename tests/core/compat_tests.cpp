#include "scrapgpu/compat.h"
#include "scrapgpu/backend.h"
#include "scrapgpu/reference.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "%d: %s (%s)\n", __LINE__, #x, scrapgpu_last_error()); std::exit(1); } } while (0)
namespace {
scrapgpu_backend_vtable backend = {};
scrapgpu_object_header device = {&backend}, stream = {&backend};
struct Buffer { scrapgpu_object_header header{&backend}; std::vector<unsigned char> data; };
scrapgpu_status support_status = SCRAPGPU_STATUS_UNSUPPORTED;
scrapgpu_status execution_status = SCRAPGPU_STATUS_OK;
int executions = 0, transfers = 0;
scrapgpu_device_t * dev() { return reinterpret_cast<scrapgpu_device_t *>(&device); }
scrapgpu_stream_t * queue() { return reinterpret_cast<scrapgpu_stream_t *>(&stream); }
scrapgpu_device_t * stream_device(scrapgpu_stream_t *) { return dev(); }
scrapgpu_status supported(scrapgpu_device_t *, const scrapgpu_op_desc *, const char **) { return support_status; }
scrapgpu_status execute(scrapgpu_stream_t *, const scrapgpu_op_desc *) { ++executions; return execution_status; }
scrapgpu_status transfer(const scrapgpu_tensor_desc * t, void * host, size_t size, bool write) {
    auto * b = reinterpret_cast<Buffer *>(t->buffer);
    if (t->offset > b->data.size() || size > b->data.size() - t->offset) return SCRAPGPU_STATUS_INVALID_ARGUMENT;
    ++transfers;
    if (write) std::memcpy(b->data.data() + t->offset, host, size);
    else std::memcpy(host, b->data.data() + t->offset, size);
    return SCRAPGPU_STATUS_OK;
}
scrapgpu_status download(scrapgpu_stream_t *, const scrapgpu_tensor_desc * t, void * p, size_t n) { return transfer(t, p, n, false); }
scrapgpu_status upload(scrapgpu_stream_t *, const scrapgpu_tensor_desc * t, const void * p, size_t n) { return transfer(t, const_cast<void *>(p), n, true); }
scrapgpu_tensor_desc tensor(scrapgpu_dtype dtype, int64_t k, int64_t m, Buffer & b) {
    scrapgpu_tensor_desc t = {};
    t.struct_size = sizeof(t); t.dtype = dtype; t.ne[0] = k; t.ne[1] = m; t.ne[2] = t.ne[3] = 1;
    t.nb[0] = scrapgpu_dtype_type_size(dtype); t.nb[1] = t.nb[0] * k;
    t.nb[2] = t.nb[3] = t.nb[1] * m; t.buffer = reinterpret_cast<scrapgpu_buffer_t *>(&b);
    b.data.resize(t.nb[2]); return t;
}
float float_bits(uint32_t bits) { float f; std::memcpy(&f, &bits, 4); return f; }
void test_bf16() {
    Buffer unused_a, unused_b;
    auto a = tensor(SCRAPGPU_DTYPE_F32, 8, 1, unused_a);
    auto b = tensor(SCRAPGPU_DTYPE_BF16, 8, 1, unused_b);
    float source[] = {1.f, -0.f, float_bits(0x3f808000), float_bits(0x3f818000),
                      float_bits(0x7f800001), INFINITY, -INFINITY, float_bits(0x00010000)};
    uint16_t target[8] = {};
    void * in[] = {source}; void * out[] = {target};
    scrapgpu_op_desc op = {sizeof(op), SCRAPGPU_OP_CONT, &a, 1, &b, 1, nullptr, 0};
    CHECK(scrapgpu_ref_execute(&op, in, out) == SCRAPGPU_STATUS_OK);
    CHECK(target[0] == 0x3f80 && target[1] == 0x8000 && target[2] == 0x3f80 && target[3] == 0x3f82);
    CHECK((target[4] & 0x7f80) == 0x7f80 && (target[4] & 0x7f));
    CHECK(target[5] == 0x7f80 && target[6] == 0xff80 && target[7] == 1);
    float roundtrip[8] = {};
    op.inputs = &b; op.outputs = &a; in[0] = target; out[0] = roundtrip;
    CHECK(scrapgpu_ref_execute(&op, in, out) == SCRAPGPU_STATUS_OK);
    CHECK(roundtrip[0] == 1.f && std::signbit(roundtrip[1]) && std::isnan(roundtrip[4]));
}
void test_compat() {
    Buffer weights, activation, result;
    scrapgpu_tensor_desc inputs[] = {tensor(SCRAPGPU_DTYPE_BF16, 2, 2, weights), tensor(SCRAPGPU_DTYPE_F32, 2, 1, activation)};
    auto output = tensor(SCRAPGPU_DTYPE_F32, 2, 1, result);
    const uint16_t w[] = {0x3f80, 0x4000, 0x4040, 0x4080}; // [1,2], [3,4]
    const float x[] = {5,6}; std::memcpy(weights.data.data(), w, sizeof(w)); std::memcpy(activation.data.data(), x, sizeof(x));
    scrapgpu_mul_mat_params params = {sizeof(params), 1};
    scrapgpu_op_desc op = {sizeof(op), SCRAPGPU_OP_MUL_MAT, inputs, 2, &output, 1, &params, sizeof(params)};
    scrapgpu_compat_options options = {sizeof(options), 1, 24};
    scrapgpu_execution_report report = {}; report.struct_size = sizeof(report);
    CHECK(scrapgpu_op_execute_compatible(dev(), queue(), &op, nullptr, &report) == SCRAPGPU_STATUS_UNSUPPORTED);
    CHECK(transfers == 0 && report.path == SCRAPGPU_EXECUTION_NONE);
    options.max_staging_bytes = 23;
    CHECK(scrapgpu_op_execute_compatible(dev(), queue(), &op, &options, &report) == SCRAPGPU_STATUS_OUT_OF_MEMORY);
    CHECK(transfers == 0);
    options.max_staging_bytes = 24;
    CHECK(scrapgpu_op_execute_compatible(dev(), queue(), &op, &options, &report) == SCRAPGPU_STATUS_OK);
    float actual[2]; std::memcpy(actual, result.data.data(), sizeof(actual));
    CHECK(actual[0] == 17 && actual[1] == 39);
    CHECK(report.path == SCRAPGPU_EXECUTION_CPU_FALLBACK && report.staging_bytes == 24);
    CHECK(report.downloaded_bytes == 24 && report.uploaded_bytes == 8 && executions == 0);
    // A supported native op which fails is not retried or silently masked.
    support_status = SCRAPGPU_STATUS_OK; execution_status = SCRAPGPU_STATUS_DEVICE_LOST;
    const int before = transfers;
    CHECK(scrapgpu_op_execute_compatible(dev(), queue(), &op, &options, &report) == SCRAPGPU_STATUS_DEVICE_LOST);
    CHECK(executions == 1 && transfers == before && report.path == SCRAPGPU_EXECUTION_NATIVE);
    support_status = SCRAPGPU_STATUS_INVALID_ARGUMENT;
    CHECK(scrapgpu_op_execute_compatible(dev(), queue(), &op, &options, &report) == SCRAPGPU_STATUS_INVALID_ARGUMENT);
    CHECK(transfers == before && executions == 1);
    support_status = SCRAPGPU_STATUS_UNSUPPORTED;
    inputs[0].ne[0] = 0;
    CHECK(scrapgpu_op_execute_compatible(dev(), queue(), &op, &options, &report) == SCRAPGPU_STATUS_INVALID_ARGUMENT);
    inputs[0].ne[0] = 2;
    inputs[0].nb[1] += 2;
    CHECK(scrapgpu_op_execute_compatible(dev(), queue(), &op, &options, &report) == SCRAPGPU_STATUS_UNSUPPORTED);
    inputs[0].nb[1] -= 2;
    inputs[0].offset = UINT64_MAX;
    CHECK(scrapgpu_op_execute_compatible(dev(), queue(), &op, &options, &report) == SCRAPGPU_STATUS_OUT_OF_MEMORY);
    inputs[0].offset = 0;
    scrapgpu_object_header other_device{&backend};
    CHECK(scrapgpu_op_execute_compatible(reinterpret_cast<scrapgpu_device_t *>(&other_device), queue(), &op, &options, &report) == SCRAPGPU_STATUS_INVALID_ARGUMENT);
    CHECK(transfers == before);
    // The real allocation is shorter than the struct, so ASan catches a copy
    // or dtype read performed before checking struct_size.
    const auto * saved_inputs = op.inputs;
    for (size_t bytes : {sizeof(size_t), offsetof(scrapgpu_tensor_desc, layout)}) {
        void * raw = std::calloc(1, bytes);
        CHECK(raw != nullptr);
        *static_cast<size_t *>(raw) = bytes;
        op.inputs = static_cast<scrapgpu_tensor_desc *>(raw);
        op.input_count = 1;
        CHECK(scrapgpu_op_execute_compatible(dev(), queue(), &op, &options, &report) == SCRAPGPU_STATUS_INVALID_ARGUMENT);
        std::free(raw);
    }
    op.inputs = saved_inputs; op.input_count = 2;
    op.op = static_cast<scrapgpu_op>(63); // Unassigned, but within the C++ enum's value range.
    CHECK(scrapgpu_op_execute_compatible(dev(), queue(), &op, &options, &report) == SCRAPGPU_STATUS_INVALID_ARGUMENT);
}
}
int main() {
    backend.struct_size = sizeof(backend); backend.name = "test"; backend.stream_device = stream_device;
    backend.op_supported = supported; backend.op_execute = execute; backend.tensor_download = download; backend.tensor_upload = upload;
    test_bf16(); test_compat(); std::puts("compatibility and BF16 tests passed");
}

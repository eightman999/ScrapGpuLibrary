#include "scrapgpu/cpu.h"
#include "scrapgpu/backend.h"
#include "scrapgpu/buffer.h"
#include "scrapgpu/stream.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL line %d: %s (%s)\n", __LINE__, #x, scrapgpu_last_error()); std::exit(1); } } while (0)
#define OK(x) CHECK((x) == SCRAPGPU_STATUS_OK)

static scrapgpu_tensor_desc tensor(scrapgpu_buffer_t * b) {
    scrapgpu_tensor_desc t{};
    t.struct_size = sizeof(t); t.dtype = SCRAPGPU_DTYPE_F32; t.buffer = b;
    t.ne[0] = t.ne[1] = 2; t.ne[2] = t.ne[3] = 1;
    t.nb[0] = 4; t.nb[1] = 8; t.nb[2] = t.nb[3] = 16;
    return t;
}

int main() {
    OK(scrapgpu_cpu_register());
    OK(scrapgpu_cpu_register());
    CHECK(scrapgpu_get_device_count() == 1);
    scrapgpu_unregister_all_backends();
    CHECK(scrapgpu_get_device_count() == 0);
    OK(scrapgpu_cpu_register());
    CHECK(scrapgpu_get_device_count() == 1);
    scrapgpu_device_t * device = nullptr;
    scrapgpu_stream_t * stream = nullptr;
    OK(scrapgpu_get_device(0, &device));
    OK(scrapgpu_stream_create(device, &stream));

    // Actual short allocations make overreads/overwrites observable under ASan.
    auto * info = static_cast<scrapgpu_device_info *>(std::malloc(sizeof(size_t)));
    CHECK(info != nullptr);
    info->struct_size = sizeof(size_t);
    OK(scrapgpu_device_get_info(device, info));
    CHECK(info->struct_size == sizeof(size_t));
    OK(scrapgpu_device_get_info(device, info));
    std::free(info);
    auto * short_tensor = static_cast<scrapgpu_tensor_desc *>(std::malloc(sizeof(size_t)));
    CHECK(short_tensor != nullptr);
    short_tensor->struct_size = sizeof(size_t);
    auto output = tensor(nullptr);
    scrapgpu_op_desc invalid{sizeof(invalid), SCRAPGPU_OP_SQR, short_tensor, 1, &output, 1, nullptr, 0};
    CHECK(scrapgpu_op_supported(device, &invalid, nullptr) == SCRAPGPU_STATUS_INVALID_ARGUMENT);
    CHECK(scrapgpu_op_execute(stream, &invalid) == SCRAPGPU_STATUS_INVALID_ARGUMENT);
    std::free(short_tensor);

    scrapgpu_buffer_t * x = nullptr, * w = nullptr, * y = nullptr;
    OK(scrapgpu_buffer_alloc(device, 16, &x));
    OK(scrapgpu_buffer_alloc(device, 16, &w));
    OK(scrapgpu_buffer_alloc(device, 16, &y));
    const float values[]{1, 2, 3, 4}, weights[]{2, 3, 100, 100};
    OK(scrapgpu_buffer_write(stream, x, 0, values, 16, 1));
    OK(scrapgpu_buffer_write(stream, w, 0, weights, 16, 1));
    scrapgpu_tensor_desc in[]{tensor(x), tensor(w)};
    auto out = tensor(y);
    scrapgpu_rms_norm_params norm{sizeof(norm), 0};
    scrapgpu_op_desc op{sizeof(op), SCRAPGPU_OP_RMS_NORM_MUL, in, 2, &out, 1, &norm, sizeof(norm)};
    OK(scrapgpu_op_execute(stream, &op));
    out = tensor(w);
    CHECK(scrapgpu_op_execute(stream, &op) == SCRAPGPU_STATUS_INVALID_ARGUMENT);
    float actual[4]{};
    OK(scrapgpu_buffer_read(stream, w, 0, actual, 16, 1));
    for (int i = 0; i < 4; ++i) CHECK(actual[i] == weights[i]);

    // SOFT_MAX may safely reuse its entire input view, as required by Attention.
    auto scores = tensor(x);
    scrapgpu_soft_max_params softmax{sizeof(softmax), 1, nullptr};
    op = {sizeof(op), SCRAPGPU_OP_SOFT_MAX, &scores, 1, &scores, 1, &softmax, sizeof(softmax)};
    OK(scrapgpu_op_execute(stream, &op));
    OK(scrapgpu_buffer_read(stream, x, 0, actual, 16, 1));
    for (int i = 0; i < 4; ++i) CHECK(std::fabs(actual[i] - (i % 2 ? 0.73105858f : 0.26894142f)) < 1e-6f);
    OK(scrapgpu_buffer_free(x)); OK(scrapgpu_buffer_free(w)); OK(scrapgpu_buffer_free(y));
    OK(scrapgpu_stream_free(stream));
    std::puts("PASS: CPU registration, short descriptors, aliasing and in-place softmax");
}

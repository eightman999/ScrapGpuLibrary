#include "scrapgpu/backend.h"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

// CHECK stays active in Release builds (CHECK() is compiled out with NDEBUG, which made these tests pass without checking anything)
#define CHECK(cond)                                                                     \
    do {                                                                                \
        if (!(cond)) {                                                                  \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            std::exit(1);                                                               \
        }                                                                               \
    } while (0)

namespace {

scrapgpu_tensor_desc tensor(scrapgpu_dtype dtype, int64_t n0, int64_t n1, int64_t n2, int64_t n3) {
    scrapgpu_tensor_desc t{};
    t.struct_size = sizeof(t);
    t.dtype = dtype;
    t.ne[0] = n0;
    t.ne[1] = n1;
    t.ne[2] = n2;
    t.ne[3] = n3;
    t.nb[0] = scrapgpu_dtype_type_size(dtype);
    t.nb[1] = scrapgpu_tensor_row_size(dtype, n0);
    t.nb[2] = t.nb[1] * static_cast<uint64_t>(n1);
    t.nb[3] = t.nb[2] * static_cast<uint64_t>(n2);
    t.layout = SCRAPGPU_LAYOUT_NATURAL;
    return t;
}

void expect_ok(scrapgpu_status status) {
    if (status != SCRAPGPU_STATUS_OK) {
        std::fprintf(stderr, "unexpected status %s: %s\n", scrapgpu_status_string(status), scrapgpu_last_error());
    }
    CHECK(status == SCRAPGPU_STATUS_OK);
}

void validate(scrapgpu_op op, const scrapgpu_tensor_desc * inputs, size_t input_count, scrapgpu_tensor_desc * outputs, size_t output_count, const void * params, size_t params_size) {
    scrapgpu_op_desc desc{};
    desc.struct_size = sizeof(desc);
    desc.op = op;
    desc.inputs = inputs;
    desc.input_count = input_count;
    desc.outputs = outputs;
    desc.output_count = output_count;
    desc.params = params;
    desc.params_size = params_size;
    expect_ok(scrapgpu_op_desc_validate(&desc));
}

void test_op_validation() {
    scrapgpu_tensor_desc f32 = tensor(SCRAPGPU_DTYPE_F32, 8, 4, 1, 1);
    scrapgpu_tensor_desc f16 = tensor(SCRAPGPU_DTYPE_F16, 8, 4, 1, 1);
    scrapgpu_tensor_desc q40 = tensor(SCRAPGPU_DTYPE_Q4_0, 32, 4, 1, 1);
    scrapgpu_tensor_desc i32 = tensor(SCRAPGPU_DTYPE_I32, 8, 1, 1, 1);
    scrapgpu_tensor_desc in2[3] = {f32, f32, i32};
    scrapgpu_tensor_desc mm[3] = {q40, f32, i32};
    scrapgpu_tensor_desc rows[2] = {q40, i32};
    scrapgpu_tensor_desc out = f32;
    scrapgpu_tensor_desc sort_out = tensor(SCRAPGPU_DTYPE_I32, 8, 4, 1, 1);

    validate(SCRAPGPU_OP_ADD, in2, 2, &out, 1, nullptr, 0);
    validate(SCRAPGPU_OP_SUB, in2, 2, &out, 1, nullptr, 0);
    validate(SCRAPGPU_OP_MUL, in2, 2, &out, 1, nullptr, 0);
    validate(SCRAPGPU_OP_DIV, in2, 2, &out, 1, nullptr, 0);

    scrapgpu_scale_params scale{sizeof(scale), 2.0f, 1.0f};
    scrapgpu_clamp_params clamp{sizeof(clamp), -1.0f, 1.0f};
    scrapgpu_unary_params unary{sizeof(unary), SCRAPGPU_UNARY_SILU};
    scrapgpu_glu_params glu{sizeof(glu), SCRAPGPU_GLU_SWIGLU, 0, 0};
    scrapgpu_rope_params rope{};
    rope.struct_size = sizeof(rope);
    rope.mode = SCRAPGPU_ROPE_MODE_NEOX;
    rope.n_dims = 8;
    scrapgpu_soft_max_params softmax{sizeof(softmax), 1.0f, nullptr};
    scrapgpu_get_rows_params get_rows{sizeof(get_rows), 0};
    scrapgpu_set_rows_params set_rows{sizeof(set_rows), 1};
    scrapgpu_cpy_params cpy{sizeof(cpy), 1};
    scrapgpu_mul_mat_params mul_mat{sizeof(mul_mat), 1};
    scrapgpu_mul_mat_id_params mid{sizeof(mid), 4, 128};
    scrapgpu_ssm_conv_params ssm{sizeof(ssm), 4};
    scrapgpu_concat_params concat{sizeof(concat), 1};
    scrapgpu_pad_params pad{};
    pad.struct_size = sizeof(pad);
    scrapgpu_argsort_params argsort{sizeof(argsort), SCRAPGPU_ARGSORT_DESC, 0};
    scrapgpu_argsort_params topk{sizeof(topk), SCRAPGPU_ARGSORT_DESC, 4};

    validate(SCRAPGPU_OP_SCALE, &f32, 1, &out, 1, &scale, sizeof(scale));
    validate(SCRAPGPU_OP_CLAMP, &f32, 1, &out, 1, &clamp, sizeof(clamp));
    validate(SCRAPGPU_OP_SQR, &f32, 1, &out, 1, nullptr, 0);
    validate(SCRAPGPU_OP_SQRT, &f32, 1, &out, 1, nullptr, 0);
    validate(SCRAPGPU_OP_UNARY, &f32, 1, &out, 1, &unary, sizeof(unary));
    validate(SCRAPGPU_OP_GLU, &f32, 1, &out, 1, &glu, sizeof(glu));
    validate(SCRAPGPU_OP_RMS_NORM, &f32, 1, &out, 1, nullptr, 0);
    validate(SCRAPGPU_OP_ROPE, &f16, 1, &out, 1, &rope, sizeof(rope));
    validate(SCRAPGPU_OP_SOFT_MAX, &f32, 1, &out, 1, &softmax, sizeof(softmax));
    validate(SCRAPGPU_OP_GET_ROWS, rows, 2, &out, 1, &get_rows, sizeof(get_rows));
    scrapgpu_tensor_desc set_rows_in[2] = {f32, i32};
    validate(SCRAPGPU_OP_SET_ROWS, set_rows_in, 2, &out, 1, &set_rows, sizeof(set_rows));
    validate(SCRAPGPU_OP_CPY, &f16, 1, &out, 1, &cpy, sizeof(cpy));
    validate(SCRAPGPU_OP_DUP, &f16, 1, &out, 1, nullptr, 0);
    validate(SCRAPGPU_OP_CONT, &f16, 1, &out, 1, nullptr, 0);
    validate(SCRAPGPU_OP_MUL_MAT, mm, 2, &out, 1, &mul_mat, sizeof(mul_mat));
    validate(SCRAPGPU_OP_MUL_MAT_ID, mm, 3, &out, 1, &mid, sizeof(mid));
    validate(SCRAPGPU_OP_ADD_ID, mm, 3, &out, 1, nullptr, 0);
    validate(SCRAPGPU_OP_SSM_CONV, in2, 2, &out, 1, &ssm, sizeof(ssm));
    validate(SCRAPGPU_OP_CONCAT, in2, 2, &out, 1, &concat, sizeof(concat));
    validate(SCRAPGPU_OP_PAD, &f32, 1, &out, 1, &pad, sizeof(pad));
    validate(SCRAPGPU_OP_SUM_ROWS, &f32, 1, &out, 1, nullptr, 0);
    validate(SCRAPGPU_OP_ARGSORT, &f32, 1, &sort_out, 1, &argsort, sizeof(argsort));
    validate(SCRAPGPU_OP_TOP_K, &f32, 1, &sort_out, 1, &topk, sizeof(topk));
}

void test_op_desc_generic_validation() {
    // audit H-3: every op — the metadata ones included — must be structurally validated before any
    // early return: a small struct_size, absurd counts and null arrays are refused, so the plan
    // builders that copy desc.inputs / desc.outputs always see a valid descriptor
    scrapgpu_tensor_desc f32 = tensor(SCRAPGPU_DTYPE_F32, 8, 4, 1, 1);
    scrapgpu_tensor_desc out = f32;
    scrapgpu_op_desc desc{};
    desc.struct_size = sizeof(desc);
    desc.op = SCRAPGPU_OP_RESHAPE;
    desc.inputs = &f32;
    desc.input_count = 1;
    desc.outputs = &out;
    desc.output_count = 1;

    expect_ok(scrapgpu_op_desc_validate(&desc)); // a sane metadata op stays valid

    const scrapgpu_op_desc valid = desc;

    desc = valid;
    desc.input_count = 1000000; // past the sanity bound even for metadata ops
    CHECK(scrapgpu_op_desc_validate(&desc) == SCRAPGPU_STATUS_INVALID_ARGUMENT);

    desc = valid;
    desc.output_count = 1000000;
    CHECK(scrapgpu_op_desc_validate(&desc) == SCRAPGPU_STATUS_INVALID_ARGUMENT);

    desc = valid;
    desc.inputs = nullptr; // a null array with a nonzero count (the SIGSEGV of the audit)
    CHECK(scrapgpu_op_desc_validate(&desc) == SCRAPGPU_STATUS_INVALID_ARGUMENT);

    desc = valid;
    desc.outputs = nullptr;
    CHECK(scrapgpu_op_desc_validate(&desc) == SCRAPGPU_STATUS_INVALID_ARGUMENT);

    desc = valid;
    desc.struct_size = offsetof(scrapgpu_op_desc, inputs); // the inputs/outputs/count fields do not exist
    CHECK(scrapgpu_op_desc_validate(&desc) == SCRAPGPU_STATUS_INVALID_ARGUMENT);

    desc = valid;
    desc.params = nullptr;
    desc.params_size = 16; // a size without a buffer
    CHECK(scrapgpu_op_desc_validate(&desc) == SCRAPGPU_STATUS_INVALID_ARGUMENT);

    desc = valid;
    desc.params = &f32;
    desc.params_size = 4097; // past the sanity bound: the plan builders memcpy params_size bytes
    CHECK(scrapgpu_op_desc_validate(&desc) == SCRAPGPU_STATUS_INVALID_ARGUMENT);

    desc = valid;
    desc.params = &f32;
    desc.params_size = 4096; // at the bound: accepted (the bytes are not read)
    expect_ok(scrapgpu_op_desc_validate(&desc));

    // the same malformed descriptors on a compute op
    desc = valid;
    desc.op = SCRAPGPU_OP_ADD;
    desc.input_count = 1000000;
    CHECK(scrapgpu_op_desc_validate(&desc) == SCRAPGPU_STATUS_INVALID_ARGUMENT);

    // a zero input/output count stays acceptable for the metadata ops (they do no work)
    desc = valid;
    desc.op = SCRAPGPU_OP_VIEW;
    desc.inputs = nullptr;
    desc.outputs = nullptr;
    desc.input_count = 0;
    desc.output_count = 0;
    expect_ok(scrapgpu_op_desc_validate(&desc));
}

}

int main() {
    CHECK(scrapgpu_get_abi_version() == ((SCRAPGPU_ABI_VERSION_MAJOR << 16u) | SCRAPGPU_ABI_VERSION_MINOR));
    CHECK(std::strcmp(scrapgpu_status_string(SCRAPGPU_STATUS_OK), "OK") == 0);
    CHECK(std::strcmp(scrapgpu_status_string(SCRAPGPU_STATUS_DEVICE_LOST), "DEVICE_LOST") == 0);

    scrapgpu_set_last_error("main-thread");
    std::thread worker([] {
        CHECK(std::strcmp(scrapgpu_last_error(), "") == 0);
        scrapgpu_set_last_error("worker-thread");
        CHECK(std::strcmp(scrapgpu_last_error(), "worker-thread") == 0);
    });
    worker.join();
    CHECK(std::strcmp(scrapgpu_last_error(), "main-thread") == 0);

    struct old_device_info {
        size_t struct_size;
        const char * name;
        const char * vendor;
    } old_info{};
    expect_ok(scrapgpu_device_info_init(reinterpret_cast<scrapgpu_device_info *>(&old_info), sizeof(old_info)));
    CHECK(old_info.struct_size == sizeof(old_info));
    CHECK(old_info.name == nullptr);
    CHECK(old_info.vendor == nullptr);

    CHECK(scrapgpu_dtype_block_size(SCRAPGPU_DTYPE_Q4_0) == 32);
    CHECK(scrapgpu_dtype_type_size(SCRAPGPU_DTYPE_Q4_0) == 18);
    CHECK(scrapgpu_dtype_block_size(SCRAPGPU_DTYPE_Q8_0) == 32);
    CHECK(scrapgpu_dtype_type_size(SCRAPGPU_DTYPE_Q8_0) == 34);
    CHECK(scrapgpu_dtype_block_size(SCRAPGPU_DTYPE_Q4_K) == 256);
    CHECK(scrapgpu_dtype_type_size(SCRAPGPU_DTYPE_Q4_K) == 144);
    CHECK(scrapgpu_dtype_block_size(SCRAPGPU_DTYPE_Q6_K) == 256);
    CHECK(scrapgpu_dtype_type_size(SCRAPGPU_DTYPE_Q6_K) == 210);
    CHECK(scrapgpu_dtype_type_size(SCRAPGPU_DTYPE_F16) == 2);
    CHECK(scrapgpu_dtype_type_size(SCRAPGPU_DTYPE_F32) == 4);
    CHECK(scrapgpu_tensor_row_size(SCRAPGPU_DTYPE_Q4_0, 64) == 36);
    CHECK(scrapgpu_tensor_row_size(SCRAPGPU_DTYPE_Q8_0, 64) == 68);
    CHECK(scrapgpu_tensor_row_size(SCRAPGPU_DTYPE_Q4_K, 512) == 288);
    CHECK(scrapgpu_tensor_row_size(SCRAPGPU_DTYPE_Q6_K, 512) == 420);

    // scrapgpu_tensor_r32_eligible is the same rule the CL11 backend applies: a repackable dtype,
    // contiguous in the ggml sense, and a single slice or 16-byte aligned slice strides
    scrapgpu_tensor_desc r32 = tensor(SCRAPGPU_DTYPE_Q4_0, 32, 16, 1, 1);
    CHECK(scrapgpu_tensor_is_contiguous(&r32) == 1);
    CHECK(scrapgpu_tensor_r32_eligible(&r32) == 1);
    r32.nb[2] += 8; // a dim of size 1: the strict contiguity fails, the relaxed one holds, and a single slice needs no stride alignment
    CHECK(scrapgpu_tensor_is_contiguous(&r32) == 0);
    CHECK(scrapgpu_tensor_r32_eligible(&r32) == 1);
    r32 = tensor(SCRAPGPU_DTYPE_Q4_0, 32, 3, 1, 1); // a single slice is eligible whatever nb[2] is
    CHECK((r32.nb[2] % 16u) != 0u);
    CHECK(scrapgpu_tensor_r32_eligible(&r32) == 1);
    r32 = tensor(SCRAPGPU_DTYPE_Q4_0, 32, 3, 2, 1); // several slices: nb[2] must stay a multiple of 16
    CHECK((r32.nb[2] % 16u) != 0u);
    CHECK(scrapgpu_tensor_r32_eligible(&r32) == 0);
    r32 = tensor(SCRAPGPU_DTYPE_Q4_0, 32, 16, 2, 1);
    CHECK(scrapgpu_tensor_r32_eligible(&r32) == 1);
    r32 = tensor(SCRAPGPU_DTYPE_Q8_0, 32, 16, 2, 1); // nb[2] = 34*16 = 544 % 16 == 0
    CHECK(scrapgpu_tensor_r32_eligible(&r32) == 1);
    r32 = tensor(SCRAPGPU_DTYPE_F32, 32, 16, 1, 1); // not a repackable dtype
    CHECK(scrapgpu_tensor_r32_eligible(&r32) == 0);
    r32 = tensor(SCRAPGPU_DTYPE_Q4_0, 32, 16, 1, 1);
    r32.nb[1] += 16; // not contiguous in either sense
    CHECK(scrapgpu_tensor_r32_eligible(&r32) == 0);

    test_op_validation();
    test_op_desc_generic_validation();

    scrapgpu_unregister_all_backends();
    CHECK(scrapgpu_get_device_count() == 0);
    scrapgpu_tensor_desc in[2] = {tensor(SCRAPGPU_DTYPE_F32, 4, 4, 1, 1), tensor(SCRAPGPU_DTYPE_F32, 4, 4, 1, 1)};
    scrapgpu_tensor_desc out = tensor(SCRAPGPU_DTYPE_F32, 4, 4, 1, 1);
    scrapgpu_op_desc op{};
    op.struct_size = sizeof(op);
    op.op = SCRAPGPU_OP_ADD;
    op.inputs = in;
    op.input_count = 2;
    op.outputs = &out;
    op.output_count = 1;
    const char * reason = nullptr;
    CHECK(scrapgpu_op_supported(nullptr, &op, &reason) == SCRAPGPU_STATUS_UNSUPPORTED);
    CHECK(reason != nullptr);
    CHECK(std::strstr(reason, "no backend") != nullptr);

    return 0;
}

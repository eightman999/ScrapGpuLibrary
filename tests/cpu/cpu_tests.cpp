// Tests of the CPU backend (src/backends/cpu) through the public C API only: registration,
// device info, buffer/stream ops with bounds and foreign-object checks, tensor transfers, the op
// support matrix mirroring the reference implementation, op_execute / plan semantics (deep-copied
// descriptors and params, aliasing rules, plan-buffer lifetime) and the documented failure
// statuses. Everything is synchronous on this backend.
//
// CHECK (not assert) keeps the tests meaningful with NDEBUG.

#include "scrapgpu/api.h"
#include "scrapgpu/backend.h"
#include "scrapgpu/buffer.h"
#include "scrapgpu/cpu.h"
#include "scrapgpu/device.h"
#include "scrapgpu/op.h"
#include "scrapgpu/reference.h"
#include "scrapgpu/status.h"
#include "scrapgpu/stream.h"
#include "scrapgpu/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#define CHECK(cond)                                                                          \
    do {                                                                                     \
        if (!(cond)) {                                                                       \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond);    \
            std::exit(1);                                                                    \
        }                                                                                    \
    } while (0)

#define CHECK_OK(status)                                                                     \
    do {                                                                                     \
        const scrapgpu_status st_ = (status);                                                \
        if (st_ != SCRAPGPU_STATUS_OK) {                                                     \
            std::fprintf(stderr, "%s:%d: unexpected status %s: %s\n",                        \
                         __FILE__, __LINE__, scrapgpu_status_string(st_), scrapgpu_last_error()); \
            std::exit(1);                                                                    \
        }                                                                                    \
    } while (0)

#define CHECK_ST(status, expected)                                                           \
    do {                                                                                     \
        const scrapgpu_status st_ = (status);                                                \
        if (st_ != (expected)) {                                                             \
            std::fprintf(stderr, "%s:%d: expected status %s, got %s: %s\n",                  \
                         __FILE__, __LINE__, scrapgpu_status_string(expected),               \
                         scrapgpu_status_string(st_), scrapgpu_last_error());                \
            std::exit(1);                                                                    \
        }                                                                                    \
    } while (0)

namespace {

scrapgpu_device_t * g_dev = nullptr;
scrapgpu_stream_t * g_stream = nullptr;
std::vector<scrapgpu_buffer_t *> g_buffers;

scrapgpu_buffer_t * g_alloc(size_t bytes) {
    scrapgpu_buffer_t * buf = nullptr;
    CHECK_OK(scrapgpu_buffer_alloc(g_dev, bytes, &buf));
    g_buffers.push_back(buf);
    return buf;
}

// a contiguous natural-layout desc
scrapgpu_tensor_desc tdesc(scrapgpu_dtype dt, int64_t n0, int64_t n1 = 1, int64_t n2 = 1, int64_t n3 = 1) {
    scrapgpu_tensor_desc t{};
    t.struct_size = sizeof(t);
    t.dtype = dt;
    t.ne[0] = n0;
    t.ne[1] = n1;
    t.ne[2] = n2;
    t.ne[3] = n3;
    t.nb[0] = scrapgpu_dtype_type_size(dt);
    t.nb[1] = scrapgpu_tensor_row_size(dt, n0);
    t.nb[2] = t.nb[1] * (uint64_t) n1;
    t.nb[3] = t.nb[2] * (uint64_t) n2;
    t.layout = SCRAPGPU_LAYOUT_NATURAL;
    return t;
}

// bytes a tensor image spans (the offset of the last byte + 1)
uint64_t tspan(const scrapgpu_tensor_desc & t) {
    const size_t block = scrapgpu_dtype_block_size(t.dtype);
    uint64_t n = block > 1 ? (uint64_t) t.ne[0] * t.nb[0] / block
                         : scrapgpu_dtype_type_size(t.dtype) + (uint64_t) (t.ne[0] - 1) * t.nb[0];
    for (int i = 1; i < 4; ++i) {
        if (t.ne[i] > 1) {
            n += (uint64_t) (t.ne[i] - 1) * t.nb[i];
        }
    }
    return n;
}

// a tensor backed by a CPU buffer of exactly its span, at offset 0
struct TB {
    scrapgpu_tensor_desc desc = {};
    scrapgpu_buffer_t *  buf  = nullptr;
    std::vector<uint8_t> host;
};

TB tmake(const scrapgpu_tensor_desc & d) {
    TB t;
    t.desc = d;
    t.desc.offset = 0;
    t.host.resize((size_t) tspan(d));
    t.buf = g_alloc((size_t) tspan(d));
    t.desc.buffer = t.buf;
    return t;
}

void tfill(TB & t, const std::vector<float> & v) {
    CHECK(v.size() == (size_t) (t.desc.ne[0] * t.desc.ne[1] * t.desc.ne[2] * t.desc.ne[3]));
    CHECK(t.desc.dtype == SCRAPGPU_DTYPE_F32);
    std::memcpy(t.host.data(), v.data(), v.size() * sizeof(float));
    CHECK_OK(scrapgpu_buffer_write(g_stream, t.buf, 0, t.host.data(), t.host.size(), 1));
}

void tfill_i32(TB & t, const std::vector<int32_t> & v) {
    CHECK(v.size() == (size_t) (t.desc.ne[0] * t.desc.ne[1] * t.desc.ne[2] * t.desc.ne[3]));
    CHECK(t.desc.dtype == SCRAPGPU_DTYPE_I32);
    std::memcpy(t.host.data(), v.data(), v.size() * sizeof(int32_t));
    CHECK_OK(scrapgpu_buffer_write(g_stream, t.buf, 0, t.host.data(), t.host.size(), 1));
}

std::vector<float> tread(TB & t) {
    CHECK(t.desc.dtype == SCRAPGPU_DTYPE_F32);
    const size_t n = (size_t) (t.desc.ne[0] * t.desc.ne[1] * t.desc.ne[2] * t.desc.ne[3]);
    std::vector<float> v(n);
    CHECK_OK(scrapgpu_buffer_read(g_stream, t.buf, 0, v.data(), n * sizeof(float), 1));
    return v;
}

scrapgpu_op_desc make_desc(scrapgpu_op op, const scrapgpu_tensor_desc * ins, size_t nin,
                           scrapgpu_tensor_desc * outs, size_t nout, const void * params, size_t params_size) {
    scrapgpu_op_desc d{};
    d.struct_size = sizeof(d);
    d.op = op;
    d.inputs = ins;
    d.input_count = nin;
    d.outputs = outs;
    d.output_count = nout;
    d.params = params;
    d.params_size = params_size;
    return d;
}

scrapgpu_status exec_op(scrapgpu_op op, const scrapgpu_tensor_desc * ins, size_t nin,
                        scrapgpu_tensor_desc * outs, size_t nout, const void * params, size_t params_size) {
    scrapgpu_op_desc d = make_desc(op, ins, nin, outs, nout, params, params_size);
    return scrapgpu_op_execute(g_stream, &d);
}

scrapgpu_status supported(scrapgpu_op op, const scrapgpu_tensor_desc * ins, size_t nin,
                          scrapgpu_tensor_desc * outs, size_t nout, const void * params, size_t params_size) {
    scrapgpu_op_desc d = make_desc(op, ins, nin, outs, nout, params, params_size);
    const char * reason = nullptr;
    const scrapgpu_status st = scrapgpu_op_supported(g_dev, &d, &reason);
    if (st != SCRAPGPU_STATUS_OK) {
        std::fprintf(stderr, "  supported() -> %s: %s\n", scrapgpu_status_string(st),
                     reason != nullptr ? reason : "");
    }
    return st;
}

bool nearly(const std::vector<float> & a, const std::vector<float> & b, float rel) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        const float m = std::max(1.0f, std::fabs(b[i]));
        if (std::fabs(a[i] - b[i]) > rel * m) {
            std::fprintf(stderr, "element %zu: %g != %g\n", i, (double) a[i], (double) b[i]);
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------------------------
// registration / device info
// ---------------------------------------------------------------------------------------------

void test_register() {
    CHECK_OK(scrapgpu_cpu_register());
    CHECK_OK(scrapgpu_cpu_register()); // idempotent
    CHECK(scrapgpu_get_device_count() >= 1);

    // find the CPU device (other backends may be registered by the same process)
    g_dev = nullptr;
    for (size_t i = 0; i < scrapgpu_get_device_count(); ++i) {
        scrapgpu_device_t * dev = nullptr;
        CHECK_OK(scrapgpu_get_device(i, &dev));
        CHECK(dev != nullptr);
        scrapgpu_device_info info{};
        CHECK_OK(scrapgpu_device_info_init(&info, sizeof(info)));
        CHECK_OK(scrapgpu_device_get_info(dev, &info));
        if (info.backend_name != nullptr && std::strcmp(info.backend_name, "cpu") == 0) {
            g_dev = dev;
            break;
        }
    }
    CHECK(g_dev != nullptr);

    scrapgpu_device_info info{};
    CHECK_OK(scrapgpu_device_info_init(&info, sizeof(info)));
    CHECK_OK(scrapgpu_device_get_info(g_dev, &info));
    CHECK(std::strcmp(info.name, "CPU") == 0);
    CHECK(std::strcmp(info.backend_name, "cpu") == 0);
    CHECK(info.total_memory_bytes > 0);
    CHECK(info.compute_units >= 1);
    std::printf("PASS  register: device '%s' backend '%s' ram=%llu MB units=%u\n", info.name,
                info.backend_name, (unsigned long long) (info.total_memory_bytes / 1048576),
                info.compute_units);

    CHECK_OK(scrapgpu_stream_create(g_dev, &g_stream));
    CHECK(g_stream != nullptr);
    CHECK_OK(scrapgpu_stream_sync(g_stream));
}

// ---------------------------------------------------------------------------------------------
// buffers: bounds, fill, overlapping copy
// ---------------------------------------------------------------------------------------------

void test_buffers() {
    scrapgpu_buffer_t * buf = g_alloc(64);

    const uint8_t pattern[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    CHECK_OK(scrapgpu_buffer_write(g_stream, buf, 0, pattern, sizeof(pattern), 1));
    CHECK_OK(scrapgpu_buffer_write(g_stream, buf, 48, pattern, sizeof(pattern), 1));
    CHECK_OK(scrapgpu_buffer_write(g_stream, buf, 64, pattern, 0, 1)); // empty write at the end

    // out of bounds / overflowing writes and reads
    CHECK_ST(scrapgpu_buffer_write(g_stream, buf, 60, pattern, 8, 1), SCRAPGPU_STATUS_INVALID_ARGUMENT);
    CHECK_ST(scrapgpu_buffer_write(g_stream, buf, UINT64_MAX - 2, pattern, 8, 1), SCRAPGPU_STATUS_INVALID_ARGUMENT);
    uint8_t sink[16] = {};
    CHECK_ST(scrapgpu_buffer_read(g_stream, buf, 60, sink, 8, 1), SCRAPGPU_STATUS_INVALID_ARGUMENT);
    CHECK_ST(scrapgpu_buffer_read(g_stream, buf, UINT64_MAX, sink, 1, 1), SCRAPGPU_STATUS_INVALID_ARGUMENT);
    CHECK_OK(scrapgpu_buffer_read(g_stream, buf, 0, sink, sizeof(sink), 1));
    CHECK(std::memcmp(sink, pattern, sizeof(sink)) == 0);

    // fill
    CHECK_OK(scrapgpu_buffer_fill(g_stream, buf, 32, 16, 0xAB, 1));
    CHECK_ST(scrapgpu_buffer_fill(g_stream, buf, 60, 8, 0xAB, 1), SCRAPGPU_STATUS_INVALID_ARGUMENT);
    CHECK_OK(scrapgpu_buffer_read(g_stream, buf, 40, sink, 1, 1));
    CHECK(sink[0] == 0xAB);

    // device-to-device copy, including an overlapping range inside one buffer (memmove semantics)
    scrapgpu_buffer_t * dst = g_alloc(64);
    CHECK_OK(scrapgpu_buffer_copy(g_stream, dst, 0, buf, 0, 64, 1));
    CHECK_ST(scrapgpu_buffer_copy(g_stream, dst, 60, buf, 0, 8, 1), SCRAPGPU_STATUS_INVALID_ARGUMENT);
    CHECK_ST(scrapgpu_buffer_copy(g_stream, dst, 0, buf, 60, 8, 1), SCRAPGPU_STATUS_INVALID_ARGUMENT);

    CHECK_OK(scrapgpu_buffer_fill(g_stream, buf, 0, 64, 0, 1));
    CHECK_OK(scrapgpu_buffer_write(g_stream, buf, 0, pattern, sizeof(pattern), 1));
    // [0,16) -> [8,24): forward overlap; the copy must use the original bytes
    CHECK_OK(scrapgpu_buffer_copy(g_stream, buf, 8, buf, 0, 16, 1));
    uint8_t verify[24] = {};
    CHECK_OK(scrapgpu_buffer_read(g_stream, buf, 0, verify, sizeof(verify), 1));
    for (int i = 0; i < 8; ++i) {
        CHECK(verify[i] == (uint8_t) i);
    }
    for (int i = 8; i < 24; ++i) {
        CHECK(verify[i] == (uint8_t) (i - 8));
    }
    std::printf("PASS  buffers: bounds, fill, overlapping copy\n");
}

// ---------------------------------------------------------------------------------------------
// foreign objects
// ---------------------------------------------------------------------------------------------

void test_foreign_objects() {
    // an object that does not carry our backend header: the dispatch must reject it, not crash
    struct fake_obj {
        scrapgpu_object_header hdr;
        uint64_t pad[8];
    };
    fake_obj fake = {};
    fake.hdr.backend = nullptr;

    auto * fdev = reinterpret_cast<scrapgpu_device_t *>(&fake);
    auto * fbuf = reinterpret_cast<scrapgpu_buffer_t *>(&fake);
    auto * fstr = reinterpret_cast<scrapgpu_stream_t *>(&fake);
    auto * fplan = reinterpret_cast<scrapgpu_plan_t *>(&fake);

    scrapgpu_buffer_t * out_buf = nullptr;
    CHECK_ST(scrapgpu_buffer_alloc(fdev, 16, &out_buf), SCRAPGPU_STATUS_INVALID_ARGUMENT);
    CHECK(out_buf == nullptr);
    scrapgpu_device_info info{};
    CHECK_OK(scrapgpu_device_info_init(&info, sizeof(info)));
    CHECK_ST(scrapgpu_device_get_info(fdev, &info), SCRAPGPU_STATUS_INVALID_ARGUMENT);
    scrapgpu_stream_t * out_str = nullptr;
    CHECK_ST(scrapgpu_stream_create(fdev, &out_str), SCRAPGPU_STATUS_INVALID_ARGUMENT);

    const uint8_t one = 1;
    CHECK_ST(scrapgpu_buffer_write(g_stream, fbuf, 0, &one, 1, 1), SCRAPGPU_STATUS_INVALID_ARGUMENT);
    uint8_t sink = 0;
    CHECK_ST(scrapgpu_buffer_read(g_stream, fbuf, 0, &sink, 1, 1), SCRAPGPU_STATUS_INVALID_ARGUMENT);
    scrapgpu_buffer_t * real = g_alloc(16);
    CHECK_ST(scrapgpu_buffer_copy(g_stream, real, 0, fbuf, 0, 8, 1), SCRAPGPU_STATUS_INVALID_ARGUMENT);
    CHECK_ST(scrapgpu_buffer_fill(g_stream, fbuf, 0, 8, 0, 1), SCRAPGPU_STATUS_INVALID_ARGUMENT);
    CHECK_ST(scrapgpu_buffer_free(fbuf), SCRAPGPU_STATUS_INVALID_ARGUMENT);
    CHECK_ST(scrapgpu_stream_sync(fstr), SCRAPGPU_STATUS_INVALID_ARGUMENT);
    CHECK_ST(scrapgpu_stream_free(fstr), SCRAPGPU_STATUS_INVALID_ARGUMENT);
    CHECK_ST(scrapgpu_plan_free(fplan), SCRAPGPU_STATUS_INVALID_ARGUMENT);

    // a real object of a different registered backend (a dummy backend with an empty vtable)
    static scrapgpu_backend_vtable dummy_vt = {};
    dummy_vt.struct_size = sizeof(dummy_vt);
    dummy_vt.name = "dummy";
    CHECK_OK(scrapgpu_register_backend(&dummy_vt));
    fake.hdr.backend = &dummy_vt;
    CHECK_ST(scrapgpu_buffer_write(g_stream, fbuf, 0, &one, 1, 1), SCRAPGPU_STATUS_INVALID_ARGUMENT);
    CHECK_ST(scrapgpu_buffer_read(g_stream, fbuf, 0, &sink, 1, 1), SCRAPGPU_STATUS_INVALID_ARGUMENT);
    fake.hdr.backend = nullptr;

    // a tensor desc whose buffer is foreign / null must be refused at execute time
    TB a = tmake(tdesc(SCRAPGPU_DTYPE_F32, 4));
    TB d = tmake(tdesc(SCRAPGPU_DTYPE_F32, 4));
    tfill(a, {1, 2, 3, 4});
    scrapgpu_unary_params up{sizeof(up), SCRAPGPU_UNARY_NEG};
    scrapgpu_tensor_desc in = a.desc, out = d.desc;
    in.buffer = fbuf;
    CHECK_ST(exec_op(SCRAPGPU_OP_UNARY, &in, 1, &out, 1, &up, sizeof(up)), SCRAPGPU_STATUS_INVALID_ARGUMENT);
    in.buffer = nullptr;
    CHECK_ST(exec_op(SCRAPGPU_OP_UNARY, &in, 1, &out, 1, &up, sizeof(up)), SCRAPGPU_STATUS_INVALID_ARGUMENT);
    in.buffer = a.desc.buffer;
    CHECK_OK(exec_op(SCRAPGPU_OP_UNARY, &in, 1, &out, 1, &up, sizeof(up)));

    std::printf("PASS  foreign objects rejected\n");
}

// ---------------------------------------------------------------------------------------------
// tensor transfers
// ---------------------------------------------------------------------------------------------

void test_tensor_transfers() {
    TB t = tmake(tdesc(SCRAPGPU_DTYPE_F32, 8, 2));
    std::vector<float> v(16);
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = (float) i + 0.5f;
    }
    CHECK_OK(scrapgpu_tensor_upload(g_stream, &t.desc, v.data(), v.size() * sizeof(float)));
    std::vector<float> back(16, 0.0f);
    CHECK_OK(scrapgpu_tensor_download(g_stream, &t.desc, back.data(), back.size() * sizeof(float)));
    CHECK(back == v);

    // partial upload of the first bytes is fine
    CHECK_OK(scrapgpu_tensor_upload(g_stream, &t.desc, v.data(), 4 * sizeof(float)));

    // oversize, strided and wrong-layout transfers are refused
    CHECK_ST(scrapgpu_tensor_upload(g_stream, &t.desc, v.data(), v.size() * sizeof(float) + 4),
             SCRAPGPU_STATUS_INVALID_ARGUMENT);
    scrapgpu_tensor_desc strided = tdesc(SCRAPGPU_DTYPE_F32, 4, 2);
    strided.nb[0] = 8; // every other float
    strided.buffer = g_alloc(64);
    CHECK_ST(scrapgpu_tensor_upload(g_stream, &strided, v.data(), 32), SCRAPGPU_STATUS_UNSUPPORTED);
    scrapgpu_tensor_desc r32 = t.desc;
    r32.layout = SCRAPGPU_LAYOUT_R32;
    CHECK_ST(scrapgpu_tensor_upload(g_stream, &r32, v.data(), 8), SCRAPGPU_STATUS_UNSUPPORTED);
    scrapgpu_tensor_desc tiny = t.desc;
    tiny.struct_size = offsetof(scrapgpu_tensor_desc, offset); // too small to carry offset
    CHECK_ST(scrapgpu_tensor_download(g_stream, &tiny, back.data(), 8), SCRAPGPU_STATUS_INVALID_ARGUMENT);

    // a view at an offset inside the buffer: upload and download the window only
    TB wide = tmake(tdesc(SCRAPGPU_DTYPE_F32, 8, 4)); // 128 bytes
    scrapgpu_tensor_desc view = tdesc(SCRAPGPU_DTYPE_F32, 8, 2);
    view.buffer = wide.buf;
    view.offset = 64;
    CHECK_OK(scrapgpu_tensor_upload(g_stream, &view, v.data(), v.size() * sizeof(float)));
    std::vector<float> window(16, 0.0f);
    CHECK_OK(scrapgpu_tensor_download(g_stream, &view, window.data(), window.size() * sizeof(float)));
    CHECK(window == v);
    std::printf("PASS  tensor transfers\n");
}

// ---------------------------------------------------------------------------------------------
// elementwise ops: shapes, broadcasting, strides, in-place
// ---------------------------------------------------------------------------------------------

void test_elementwise() {
    TB a = tmake(tdesc(SCRAPGPU_DTYPE_F32, 4, 4));
    TB b = tmake(tdesc(SCRAPGPU_DTYPE_F32, 4, 4));
    TB d = tmake(tdesc(SCRAPGPU_DTYPE_F32, 4, 4));
    std::vector<float> va(16), vb(16);
    for (size_t i = 0; i < 16; ++i) {
        va[i] = (float) i - 8.0f;
        vb[i] = (float) i * 0.5f;
    }
    tfill(a, va);
    tfill(b, vb);
    scrapgpu_tensor_desc in[2] = {a.desc, b.desc};
    scrapgpu_tensor_desc out = d.desc;
    CHECK_OK(supported(SCRAPGPU_OP_ADD, in, 2, &out, 1, nullptr, 0));
    CHECK_OK(exec_op(SCRAPGPU_OP_ADD, in, 2, &out, 1, nullptr, 0));
    std::vector<float> got = tread(d);
    for (size_t i = 0; i < 16; ++i) {
        CHECK(got[i] == va[i] + vb[i]);
    }

    // broadcasting: src1 [4,1] repeats its row over the rows of dst [4,4]
    TB c = tmake(tdesc(SCRAPGPU_DTYPE_F32, 4, 1));
    tfill(c, {100.0f, 200.0f, 300.0f, 400.0f});
    scrapgpu_tensor_desc bin[2] = {a.desc, c.desc};
    CHECK_OK(supported(SCRAPGPU_OP_ADD, bin, 2, &out, 1, nullptr, 0));
    CHECK_OK(exec_op(SCRAPGPU_OP_ADD, bin, 2, &out, 1, nullptr, 0));
    got = tread(d);
    for (int i1 = 0; i1 < 4; ++i1) {
        for (int i0 = 0; i0 < 4; ++i0) {
            CHECK(got[(size_t) i1 * 4 + (size_t) i0] == va[(size_t) i1 * 4 + (size_t) i0] + ((float) i0 + 1.0f) * 100.0f);
        }
    }

    // src1 larger than dst in a dim is INVALID
    TB wide = tmake(tdesc(SCRAPGPU_DTYPE_F32, 4, 8));
    scrapgpu_tensor_desc win[2] = {a.desc, wide.desc};
    CHECK_ST(supported(SCRAPGPU_OP_ADD, win, 2, &out, 1, nullptr, 0), SCRAPGPU_STATUS_INVALID_ARGUMENT);

    // a strided src0 (nb[0] = 8 bytes: every other float) works on the CPU backend
    TB sbuf = tmake(tdesc(SCRAPGPU_DTYPE_F32, 8, 4)); // physical storage: rows of 8
    std::vector<float> shost(32, 0.0f);
    for (int i = 0; i < 32; ++i) {
        shost[(size_t) i] = (float) i;
    }
    CHECK_OK(scrapgpu_buffer_write(g_stream, sbuf.buf, 0, shost.data(), shost.size() * sizeof(float), 1));
    scrapgpu_tensor_desc sv = a.desc; // [4,4] view of the even elements of the storage rows
    sv.nb[0] = 8;
    sv.nb[1] = 32;
    sv.nb[2] = 128;
    sv.nb[3] = 128;
    sv.buffer = sbuf.buf;
    scrapgpu_tensor_desc sin[2] = {sv, b.desc};
    CHECK_OK(supported(SCRAPGPU_OP_ADD, sin, 2, &out, 1, nullptr, 0));
    CHECK_OK(exec_op(SCRAPGPU_OP_ADD, sin, 2, &out, 1, nullptr, 0));
    got = tread(d);
    for (int i1 = 0; i1 < 4; ++i1) {
        for (int i0 = 0; i0 < 4; ++i0) {
            CHECK(got[(size_t) i1 * 4 + (size_t) i0] == shost[(size_t) i1 * 8 + (size_t) i0 * 2] + vb[(size_t) i1 * 4 + (size_t) i0]);
        }
    }

    // exact in-place (dst is the identical view of src0) is allowed for elementwise ops
    scrapgpu_tensor_desc self_in[2] = {a.desc, b.desc};
    scrapgpu_tensor_desc self_out = a.desc;
    CHECK_OK(supported(SCRAPGPU_OP_ADD, self_in, 2, &self_out, 1, nullptr, 0));
    CHECK_OK(exec_op(SCRAPGPU_OP_ADD, self_in, 2, &self_out, 1, nullptr, 0));
    got = tread(a);
    for (size_t i = 0; i < 16; ++i) {
        CHECK(got[i] == va[i] + vb[i]); // a was overwritten with a+b
    }

    // a partial overlap is refused even for elementwise ops
    TB c4 = tmake(tdesc(SCRAPGPU_DTYPE_F32, 4));
    scrapgpu_tensor_desc in4 = c4.desc;   // [4] into the start of the same buffer
    scrapgpu_tensor_desc out4 = c4.desc;  // [4] at byte offset 4: partial overlap
    out4.offset = 4;
    scrapgpu_tensor_desc oin[2] = {in4, c4.desc};
    CHECK_ST(exec_op(SCRAPGPU_OP_ADD, oin, 2, &out4, 1, nullptr, 0), SCRAPGPU_STATUS_INVALID_ARGUMENT);

    // SCALE + params
    scrapgpu_scale_params sp{sizeof(sp), 2.0f, 0.5f};
    scrapgpu_tensor_desc sin1 = a.desc;
    scrapgpu_tensor_desc sout1 = d.desc;
    CHECK_OK(exec_op(SCRAPGPU_OP_SCALE, &sin1, 1, &sout1, 1, &sp, sizeof(sp)));
    got = tread(d);
    for (size_t i = 0; i < 16; ++i) {
        CHECK(got[i] == (va[i] + vb[i]) * 2.0f + 0.5f);
    }

    // UNARY RELU
    scrapgpu_unary_params up{sizeof(up), SCRAPGPU_UNARY_RELU};
    CHECK_OK(exec_op(SCRAPGPU_OP_UNARY, &sin1, 1, &sout1, 1, &up, sizeof(up)));
    got = tread(d);
    for (size_t i = 0; i < 16; ++i) {
        CHECK(got[i] == std::max(0.0f, va[i] + vb[i]));
    }
    std::printf("PASS  elementwise: broadcast, strides, in-place, params\n");
}

// ---------------------------------------------------------------------------------------------
// matmul: F32, Q4_0, batch broadcasting
// ---------------------------------------------------------------------------------------------

void test_mul_mat_f32() {
    // W [K=8, N=4] x X [K=8, M=2] -> dst [4,2]
    TB w = tmake(tdesc(SCRAPGPU_DTYPE_F32, 8, 4));
    TB x = tmake(tdesc(SCRAPGPU_DTYPE_F32, 8, 2));
    TB d = tmake(tdesc(SCRAPGPU_DTYPE_F32, 4, 2));
    std::vector<float> wv(32), xv(16);
    for (size_t i = 0; i < wv.size(); ++i) wv[i] = (float) ((int) (i % 7) - 3) * 0.25f;
    for (size_t i = 0; i < xv.size(); ++i) xv[i] = (float) ((int) (i % 5) - 2) * 0.5f;
    tfill(w, wv);
    tfill(x, xv);
    scrapgpu_mul_mat_params mp{sizeof(mp), 0};
    scrapgpu_tensor_desc in[2] = {w.desc, x.desc};
    scrapgpu_tensor_desc out = d.desc;
    CHECK_OK(supported(SCRAPGPU_OP_MUL_MAT, in, 2, &out, 1, &mp, sizeof(mp)));
    CHECK_OK(exec_op(SCRAPGPU_OP_MUL_MAT, in, 2, &out, 1, &mp, sizeof(mp)));
    std::vector<float> got = tread(d);
    // dst[n, m] = sum_k W[k, n] * X[k, m]  (element n + 4*m of dst, k + 8*n of W, k + 8*m of X)
    for (int n = 0; n < 4; ++n) {
        for (int m = 0; m < 2; ++m) {
            float acc = 0.0f;
            for (int k = 0; k < 8; ++k) {
                acc += wv[(size_t) k + 8 * (size_t) n] * xv[(size_t) k + 8 * (size_t) m];
            }
            CHECK(std::fabs(got[(size_t) n + 4 * (size_t) m] - acc) < 1e-5f);
        }
    }

    // batch broadcasting: W [8,4,2], X [8,2,4] -> dst [4,2,4] with allow_broadcast = 1
    TB wb = tmake(tdesc(SCRAPGPU_DTYPE_F32, 8, 4, 2));
    TB xb = tmake(tdesc(SCRAPGPU_DTYPE_F32, 8, 2, 4));
    TB db = tmake(tdesc(SCRAPGPU_DTYPE_F32, 4, 2, 4));
    std::vector<float> wbv(64), xbv(64);
    for (size_t i = 0; i < wbv.size(); ++i) wbv[i] = (float) ((int) (i % 9) - 4) * 0.125f;
    for (size_t i = 0; i < xbv.size(); ++i) xbv[i] = (float) ((int) (i % 3) - 1) * 0.75f;
    tfill(wb, wbv);
    tfill(xb, xbv);
    scrapgpu_tensor_desc bin[2] = {wb.desc, xb.desc};
    scrapgpu_tensor_desc bout = db.desc;
    CHECK_ST(supported(SCRAPGPU_OP_MUL_MAT, bin, 2, &bout, 1, &mp, sizeof(mp)),
             SCRAPGPU_STATUS_UNSUPPORTED); // allow_broadcast == 0
    scrapgpu_mul_mat_params mpb{sizeof(mpb), 1};
    CHECK_OK(supported(SCRAPGPU_OP_MUL_MAT, bin, 2, &bout, 1, &mpb, sizeof(mpb)));
    CHECK_OK(exec_op(SCRAPGPU_OP_MUL_MAT, bin, 2, &bout, 1, &mpb, sizeof(mpb)));
    got = tread(db);
    for (int i2 = 0; i2 < 4; ++i2) { // dst batch i2 uses W slice i2/2
        const int w2 = i2 / 2;
        for (int m = 0; m < 2; ++m) {
            for (int n = 0; n < 4; ++n) {
                float acc = 0.0f;
                for (int k = 0; k < 8; ++k) {
                    const float wa = wbv[(size_t) k + 8 * ((size_t) n + 4 * (size_t) w2)];
                    const float xb_ = xbv[(size_t) k + 8 * ((size_t) m + 2 * (size_t) i2)];
                    acc += wa * xb_;
                }
                CHECK(std::fabs(got[(size_t) n + 4 * ((size_t) m + 2 * (size_t) i2)] - acc) < 1e-5f);
            }
        }
    }
    std::printf("PASS  MUL_MAT F32 (mat-mat + broadcast)\n");
}

void test_mul_mat_q4() {
    // Q4_0 W [K=32, N=4] natural GGUF blocks (4 rows x 1 block of 18 bytes)
    TB w = tmake(tdesc(SCRAPGPU_DTYPE_Q4_0, 32, 4));
    std::vector<float> wf(128);
    for (size_t i = 0; i < wf.size(); ++i) {
        wf[i] = (float) ((int) (i % 11) - 5) * 0.2f;
    }
    CHECK_OK(scrapgpu_ref_quantize(SCRAPGPU_DTYPE_Q4_0, wf.data(), w.host.data(), (int64_t) wf.size()));
    // the natural blocks are uploaded as raw tensor bytes
    CHECK_OK(scrapgpu_tensor_upload(g_stream, &w.desc, w.host.data(), w.host.size()));

    TB x = tmake(tdesc(SCRAPGPU_DTYPE_F32, 32, 2));
    std::vector<float> xv(64);
    for (size_t i = 0; i < xv.size(); ++i) {
        xv[i] = (float) ((int) (i % 7) - 3) * 0.3f;
    }
    tfill(x, xv);
    TB d = tmake(tdesc(SCRAPGPU_DTYPE_F32, 4, 2));

    scrapgpu_mul_mat_params mp{sizeof(mp), 0};
    scrapgpu_tensor_desc in[2] = {w.desc, x.desc};
    scrapgpu_tensor_desc out = d.desc;
    CHECK_OK(supported(SCRAPGPU_OP_MUL_MAT, in, 2, &out, 1, &mp, sizeof(mp)));
    CHECK_OK(exec_op(SCRAPGPU_OP_MUL_MAT, in, 2, &out, 1, &mp, sizeof(mp)));

    // expected: dequantize the blocks back and run a float matmul against them
    std::vector<float> wq(128, 0.0f);
    CHECK_OK(scrapgpu_ref_dequantize(SCRAPGPU_DTYPE_Q4_0, w.host.data(), wq.data(), (int64_t) wq.size()));
    std::vector<float> got = tread(d);
    std::vector<float> exp(8, 0.0f);
    for (int n = 0; n < 4; ++n) {
        for (int m = 0; m < 2; ++m) {
            float acc = 0.0f;
            for (int k = 0; k < 32; ++k) {
                acc += wq[(size_t) k + 32 * (size_t) n] * xv[(size_t) k + 32 * (size_t) m];
            }
            exp[(size_t) n + 4 * (size_t) m] = acc;
        }
    }
    CHECK(nearly(got, exp, 1e-4f));

    // a quantized src0 whose row is not a whole number of blocks is UNSUPPORTED
    scrapgpu_tensor_desc bad_w = w.desc;
    bad_w.ne[0] = 48;
    scrapgpu_tensor_desc bad_in[2] = {bad_w, x.desc};
    scrapgpu_tensor_desc bad_out = d.desc;
    CHECK_ST(supported(SCRAPGPU_OP_MUL_MAT, bad_in, 2, &bad_out, 1, &mp, sizeof(mp)),
             SCRAPGPU_STATUS_UNSUPPORTED);
    std::printf("PASS  MUL_MAT Q4_0\n");
}

// ---------------------------------------------------------------------------------------------
// the op support matrix (metadata only -- no tensor data is touched)
// ---------------------------------------------------------------------------------------------

void test_supported_matrix() {
    scrapgpu_tensor_desc f4 = tdesc(SCRAPGPU_DTYPE_F32, 4);
    scrapgpu_tensor_desc f8 = tdesc(SCRAPGPU_DTYPE_F32, 8);
    scrapgpu_tensor_desc f44 = tdesc(SCRAPGPU_DTYPE_F32, 4, 4);
    scrapgpu_tensor_desc h44 = tdesc(SCRAPGPU_DTYPE_F16, 4, 4);
    scrapgpu_tensor_desc b44 = tdesc(SCRAPGPU_DTYPE_BF16, 4, 4);
    scrapgpu_tensor_desc i4 = tdesc(SCRAPGPU_DTYPE_I32, 4);
    scrapgpu_tensor_desc i44 = tdesc(SCRAPGPU_DTYPE_I32, 4, 4);

    // metadata ops
    scrapgpu_tensor_desc m_in = f44;
    scrapgpu_tensor_desc m_out = f44;
    CHECK(supported(SCRAPGPU_OP_RESHAPE, &m_in, 1, &m_out, 1, nullptr, 0) == SCRAPGPU_STATUS_OK);
    CHECK(supported(SCRAPGPU_OP_NONE, nullptr, 0, nullptr, 0, nullptr, 0) == SCRAPGPU_STATUS_OK);

    // ADD on F16 tensors is refused by the descriptor validation
    scrapgpu_tensor_desc h_in[2] = {h44, h44};
    CHECK(supported(SCRAPGPU_OP_ADD, h_in, 2, &h44, 1, nullptr, 0) == SCRAPGPU_STATUS_INVALID_ARGUMENT);

    // UNARY kind out of range -> UNSUPPORTED (validation accepts the params)
    scrapgpu_unary_params bad_unary{sizeof(bad_unary), (scrapgpu_unary_kind) 99};
    CHECK(supported(SCRAPGPU_OP_UNARY, &f44, 1, &f44, 1, &bad_unary, sizeof(bad_unary)) == SCRAPGPU_STATUS_UNSUPPORTED);

    // GLU: [8] -> [4] works, [6] -> [4] does not
    scrapgpu_glu_params glu{sizeof(glu), SCRAPGPU_GLU_SWIGLU, 0, 0};
    scrapgpu_tensor_desc g8 = tdesc(SCRAPGPU_DTYPE_F32, 8);
    CHECK(supported(SCRAPGPU_OP_GLU, &g8, 1, &f4, 1, &glu, sizeof(glu)) == SCRAPGPU_STATUS_OK);
    scrapgpu_tensor_desc g6 = tdesc(SCRAPGPU_DTYPE_F32, 6);
    CHECK(supported(SCRAPGPU_OP_GLU, &g6, 1, &f4, 1, &glu, sizeof(glu)) == SCRAPGPU_STATUS_INVALID_ARGUMENT);

    // RMS_NORM / RMS_NORM_MUL / ADD_RMS_NORM_MUL
    scrapgpu_rms_norm_params rms{sizeof(rms), 1e-5f};
    CHECK(supported(SCRAPGPU_OP_RMS_NORM, &f44, 1, &f44, 1, &rms, sizeof(rms)) == SCRAPGPU_STATUS_OK);
    scrapgpu_tensor_desc rnm_in[2] = {f44, f4};
    CHECK(supported(SCRAPGPU_OP_RMS_NORM_MUL, rnm_in, 2, &f44, 1, &rms, sizeof(rms)) == SCRAPGPU_STATUS_OK);
    scrapgpu_tensor_desc arnm_in[3] = {f44, f44, f4};
    scrapgpu_tensor_desc arnm_out[2] = {f44, f44};
    CHECK(supported(SCRAPGPU_OP_ADD_RMS_NORM_MUL, arnm_in, 3, arnm_out, 2, &rms, sizeof(rms)) == SCRAPGPU_STATUS_OK);

    // SOFT_MAX with and without mask
    scrapgpu_soft_max_params sm{sizeof(sm), 1.0f, nullptr};
    CHECK(supported(SCRAPGPU_OP_SOFT_MAX, &f44, 1, &f44, 1, &sm, sizeof(sm)) == SCRAPGPU_STATUS_OK);
    scrapgpu_tensor_desc sm_in[2] = {f44, h44}; // F16 mask
    scrapgpu_soft_max_params smm{sizeof(smm), 1.0f, &sm_in[1]};
    CHECK(supported(SCRAPGPU_OP_SOFT_MAX, sm_in, 2, &f44, 1, &smm, sizeof(smm)) == SCRAPGPU_STATUS_OK);
    smm.mask = &sm_in[0]; // non-null mask pointer but only one input
    CHECK(supported(SCRAPGPU_OP_SOFT_MAX, sm_in, 1, &f44, 1, &smm, sizeof(smm)) == SCRAPGPU_STATUS_UNSUPPORTED);
    scrapgpu_tensor_desc bad_mask = tdesc(SCRAPGPU_DTYPE_F32, 2, 4); // mask ne0 too short
    scrapgpu_tensor_desc bm_in[2] = {f44, bad_mask};
    smm.mask = nullptr;
    CHECK(supported(SCRAPGPU_OP_SOFT_MAX, bm_in, 2, &f44, 1, &smm, sizeof(smm)) == SCRAPGPU_STATUS_INVALID_ARGUMENT);

    // ROPE
    scrapgpu_tensor_desc rope_in[3] = {f44, i4, f4};
    scrapgpu_rope_params rp{};
    rp.struct_size = sizeof(rp);
    rp.mode = SCRAPGPU_ROPE_MODE_NEOX;
    rp.n_dims = 4;
    CHECK(supported(SCRAPGPU_OP_ROPE, rope_in, 2, &f44, 1, &rp, sizeof(rp)) == SCRAPGPU_STATUS_OK);
    rp.n_dims = 3; // odd
    CHECK(supported(SCRAPGPU_OP_ROPE, rope_in, 2, &f44, 1, &rp, sizeof(rp)) == SCRAPGPU_STATUS_INVALID_ARGUMENT);
    rp.n_dims = 4;
    scrapgpu_rope_params rpo = rp;
    rpo.n_offs = &rope_in[0]; // n_offs is not implemented by the reference
    CHECK(supported(SCRAPGPU_OP_ROPE, rope_in, 2, &f44, 1, &rpo, sizeof(rpo)) == SCRAPGPU_STATUS_UNSUPPORTED);
    scrapgpu_rope_params rpf = rp;
    rpf.freq_factors = &rope_in[2]; // a non-null pointer requires the third input
    CHECK(supported(SCRAPGPU_OP_ROPE, rope_in, 2, &f44, 1, &rpf, sizeof(rpf)) == SCRAPGPU_STATUS_UNSUPPORTED);
    CHECK(supported(SCRAPGPU_OP_ROPE, rope_in, 3, &f44, 1, &rpf, sizeof(rpf)) == SCRAPGPU_STATUS_OK);

    // GET_ROWS (F32 source here; BF16 is not a valid row source)
    scrapgpu_get_rows_params gr{sizeof(gr), 0};
    scrapgpu_tensor_desc gr_in[2] = {f44, i4};
    scrapgpu_tensor_desc gr_out = tdesc(SCRAPGPU_DTYPE_F32, 4, 4);
    CHECK(supported(SCRAPGPU_OP_GET_ROWS, gr_in, 2, &gr_out, 1, &gr, sizeof(gr)) == SCRAPGPU_STATUS_OK);
    scrapgpu_tensor_desc gr_b[2] = {b44, i4};
    CHECK(supported(SCRAPGPU_OP_GET_ROWS, gr_b, 2, &gr_out, 1, &gr, sizeof(gr)) == SCRAPGPU_STATUS_UNSUPPORTED);

    // SET_ROWS
    scrapgpu_set_rows_params sr{sizeof(sr), 0};
    scrapgpu_tensor_desc sr_in[2] = {f44, i4};
    scrapgpu_tensor_desc sr_out = tdesc(SCRAPGPU_DTYPE_F32, 4, 8);
    CHECK(supported(SCRAPGPU_OP_SET_ROWS, sr_in, 2, &sr_out, 1, &sr, sizeof(sr)) == SCRAPGPU_STATUS_OK);

    // CPY / DUP / CONT: F32<->F16<->BF16 conversions
    scrapgpu_cpy_params cp{sizeof(cp), 0};
    CHECK(supported(SCRAPGPU_OP_CPY, &f44, 1, &f44, 1, &cp, sizeof(cp)) == SCRAPGPU_STATUS_OK);
    CHECK(supported(SCRAPGPU_OP_CPY, &f44, 1, &h44, 1, &cp, sizeof(cp)) == SCRAPGPU_STATUS_OK); // F32 -> F16
    CHECK(supported(SCRAPGPU_OP_CPY, &f44, 1, &b44, 1, &cp, sizeof(cp)) == SCRAPGPU_STATUS_OK); // F32 -> BF16
    CHECK(supported(SCRAPGPU_OP_CPY, &f44, 1, &i44, 1, &cp, sizeof(cp)) == SCRAPGPU_STATUS_INVALID_ARGUMENT); // F32 -> I32

    // MUL_MAT: BF16 weights are valid; mismatched K is not
    scrapgpu_mul_mat_params mm{sizeof(mm), 1};
    scrapgpu_tensor_desc mm_in[2] = {b44, f44};
    CHECK(supported(SCRAPGPU_OP_MUL_MAT, mm_in, 2, &f44, 1, &mm, sizeof(mm)) == SCRAPGPU_STATUS_OK);
    scrapgpu_tensor_desc mm_bad = tdesc(SCRAPGPU_DTYPE_F32, 8, 4);
    scrapgpu_tensor_desc mm_in2[2] = {mm_bad, f44};
    CHECK(supported(SCRAPGPU_OP_MUL_MAT, mm_in2, 2, &f44, 1, &mm, sizeof(mm)) == SCRAPGPU_STATUS_INVALID_ARGUMENT);

    // MUL_MAT_ID: [K=4, M=2, E=2] experts, [K=4, n_used=2, tokens=2] activations, I32 [2,2] ids
    scrapgpu_mul_mat_id_params mid{sizeof(mid), 2, 4};
    scrapgpu_tensor_desc e_w = tdesc(SCRAPGPU_DTYPE_F32, 4, 2, 2);
    scrapgpu_tensor_desc e_x = tdesc(SCRAPGPU_DTYPE_F32, 4, 2, 2);
    scrapgpu_tensor_desc e_ids = tdesc(SCRAPGPU_DTYPE_I32, 2, 2);
    scrapgpu_tensor_desc e_out = tdesc(SCRAPGPU_DTYPE_F32, 2, 2, 2);
    scrapgpu_tensor_desc mid_in[3] = {e_w, e_x, e_ids};
    CHECK(supported(SCRAPGPU_OP_MUL_MAT_ID, mid_in, 3, &e_out, 1, &mid, sizeof(mid)) == SCRAPGPU_STATUS_OK);

    // ADD_ID: the ids must cover the expert dims of src0 ([4,4] -> ids [4,1]); [2,1] ids are refused
    scrapgpu_tensor_desc aid_x = tdesc(SCRAPGPU_DTYPE_F32, 4, 8);
    scrapgpu_tensor_desc aid_ids = tdesc(SCRAPGPU_DTYPE_I32, 2, 1);
    scrapgpu_tensor_desc aid_in[3] = {f44, aid_x, aid_ids};
    CHECK(supported(SCRAPGPU_OP_ADD_ID, aid_in, 3, &f44, 1, nullptr, 0) == SCRAPGPU_STATUS_INVALID_ARGUMENT);
    scrapgpu_tensor_desc aid_ids2 = tdesc(SCRAPGPU_DTYPE_I32, 4, 1);
    scrapgpu_tensor_desc aid_in2[3] = {f44, aid_x, aid_ids2};
    CHECK(supported(SCRAPGPU_OP_ADD_ID, aid_in2, 3, &f44, 1, nullptr, 0) == SCRAPGPU_STATUS_OK);

    // SSM_CONV: F32 [8,4] state and [4,4] filter -> [4,5]
    scrapgpu_ssm_conv_params ssm{sizeof(ssm), 4};
    scrapgpu_tensor_desc conv_in[2] = {tdesc(SCRAPGPU_DTYPE_F32, 8, 4), f44};
    scrapgpu_tensor_desc conv_out = tdesc(SCRAPGPU_DTYPE_F32, 4, 5);
    CHECK(supported(SCRAPGPU_OP_SSM_CONV, conv_in, 2, &conv_out, 1, &ssm, sizeof(ssm)) == SCRAPGPU_STATUS_OK);

    // CONCAT along dim 0
    scrapgpu_concat_params cc{sizeof(cc), 0};
    scrapgpu_tensor_desc cc_in[2] = {f4, f4};
    CHECK(supported(SCRAPGPU_OP_CONCAT, cc_in, 2, &f8, 1, &cc, sizeof(cc)) == SCRAPGPU_STATUS_OK);

    // PAD
    scrapgpu_pad_params pd{};
    pd.struct_size = sizeof(pd);
    pd.before[0] = 1;
    pd.after[0] = 1;
    scrapgpu_tensor_desc pd_out = tdesc(SCRAPGPU_DTYPE_F32, 6);
    CHECK(supported(SCRAPGPU_OP_PAD, &f4, 1, &pd_out, 1, &pd, sizeof(pd)) == SCRAPGPU_STATUS_OK);
    CHECK(supported(SCRAPGPU_OP_PAD, &f4, 1, &f8, 1, &pd, sizeof(pd)) == SCRAPGPU_STATUS_INVALID_ARGUMENT);

    // SUM_ROWS
    scrapgpu_tensor_desc srow_bad = tdesc(SCRAPGPU_DTYPE_F32, 1, 1);
    CHECK(supported(SCRAPGPU_OP_SUM_ROWS, &f44, 1, &srow_bad, 1, nullptr, 0) == SCRAPGPU_STATUS_INVALID_ARGUMENT);
    scrapgpu_tensor_desc srow = tdesc(SCRAPGPU_DTYPE_F32, 1, 4);
    CHECK(supported(SCRAPGPU_OP_SUM_ROWS, &f44, 1, &srow, 1, nullptr, 0) == SCRAPGPU_STATUS_OK);

    // ARGSORT / TOP_K
    scrapgpu_argsort_params as{sizeof(as), SCRAPGPU_ARGSORT_ASC, 0};
    CHECK(supported(SCRAPGPU_OP_ARGSORT, &f44, 1, &i44, 1, &as, sizeof(as)) == SCRAPGPU_STATUS_OK);
    scrapgpu_argsort_params tk{sizeof(tk), SCRAPGPU_ARGSORT_DESC, 2};
    scrapgpu_tensor_desc tk_out = tdesc(SCRAPGPU_DTYPE_I32, 2, 4);
    CHECK(supported(SCRAPGPU_OP_TOP_K, &f44, 1, &tk_out, 1, &tk, sizeof(tk)) == SCRAPGPU_STATUS_OK);
    tk.top_k = 3;
    CHECK(supported(SCRAPGPU_OP_TOP_K, &f44, 1, &tk_out, 1, &tk, sizeof(tk)) == SCRAPGPU_STATUS_INVALID_ARGUMENT);

    // ops the reference does not implement -> UNSUPPORTED
    scrapgpu_tensor_desc pair_in[4] = {f44, i44, f44, i44};
    scrapgpu_tensor_desc pair_out[2] = {f44, f44};
    CHECK(supported(SCRAPGPU_OP_SET_ROWS_PAIR, pair_in, 4, pair_out, 2, nullptr, 0) == SCRAPGPU_STATUS_UNSUPPORTED);
    scrapgpu_tensor_desc multi_in[3] = {f44, f44, f44};
    scrapgpu_tensor_desc multi_out[2] = {f44, f44};
    CHECK(supported(SCRAPGPU_OP_MUL_MAT_MULTI, multi_in, 3, multi_out, 2, nullptr, 0) == SCRAPGPU_STATUS_UNSUPPORTED);

    // non-natural layout and malformed shapes
    scrapgpu_tensor_desc r32_in[2] = {f44, f44};
    r32_in[0].layout = SCRAPGPU_LAYOUT_R32;
    CHECK(supported(SCRAPGPU_OP_ADD, r32_in, 2, &f44, 1, nullptr, 0) == SCRAPGPU_STATUS_UNSUPPORTED);
    scrapgpu_tensor_desc zero_in[2] = {f44, f44};
    zero_in[0].ne[0] = 0;
    CHECK(supported(SCRAPGPU_OP_ADD, zero_in, 2, &f44, 1, nullptr, 0) == SCRAPGPU_STATUS_INVALID_ARGUMENT);
    zero_in[0].ne[0] = -3;
    CHECK(supported(SCRAPGPU_OP_ADD, zero_in, 2, &f44, 1, nullptr, 0) == SCRAPGPU_STATUS_INVALID_ARGUMENT);
    scrapgpu_tensor_desc huge_in[2] = {f44, f44};
    huge_in[0].nb[1] = std::numeric_limits<uint64_t>::max() / 2;
    CHECK(supported(SCRAPGPU_OP_ADD, huge_in, 2, &f44, 1, nullptr, 0) == SCRAPGPU_STATUS_UNSUPPORTED);

    std::printf("PASS  op support matrix\n");
}

// ---------------------------------------------------------------------------------------------
// execute-time failure statuses
// ---------------------------------------------------------------------------------------------

void test_failure_statuses() {
    TB a = tmake(tdesc(SCRAPGPU_DTYPE_F32, 4, 4));
    TB b = tmake(tdesc(SCRAPGPU_DTYPE_F32, 4, 4));
    TB d = tmake(tdesc(SCRAPGPU_DTYPE_F32, 4, 4));
    tfill(a, std::vector<float>(16, 1.0f));
    tfill(b, std::vector<float>(16, 2.0f));

    scrapgpu_tensor_desc in[2] = {a.desc, b.desc};

    // an output that overlaps an input of a non-elementwise op is refused
    scrapgpu_tensor_desc mm_dst = tdesc(SCRAPGPU_DTYPE_F32, 4, 4); // valid MUL_MAT dst shape...
    mm_dst.buffer = a.buf;                                       // ...but overlaps the src0 storage
    scrapgpu_mul_mat_params mmp{sizeof(mmp), 0};
    scrapgpu_tensor_desc mm_in[2] = {a.desc, b.desc};            // a [4,4], b [4,4] as [K,N]x[K,M]
    CHECK_ST(exec_op(SCRAPGPU_OP_MUL_MAT, mm_in, 2, &mm_dst, 1, &mmp, sizeof(mmp)),
             SCRAPGPU_STATUS_INVALID_ARGUMENT);

    // SOFT_MAX in the identical view is elementwise-safe and allowed
    scrapgpu_soft_max_params sm{sizeof(sm), 1.0f, nullptr};
    scrapgpu_tensor_desc self = a.desc;
    CHECK_OK(exec_op(SCRAPGPU_OP_SOFT_MAX, &self, 1, &self, 1, &sm, sizeof(sm)));
    std::vector<float> sm_got = tread(a);
    for (float v : sm_got) { // every input was 1.0 -> uniform softmax
        CHECK(std::fabs(v - 0.25f) < 1e-6f);
    }

    // a tensor that reaches past its buffer is refused
    scrapgpu_tensor_desc past = d.desc;
    past.offset = tspan(d.desc); // starts at the end: any element is out of bounds
    CHECK_ST(exec_op(SCRAPGPU_OP_ADD, in, 2, &past, 1, nullptr, 0), SCRAPGPU_STATUS_INVALID_ARGUMENT);

    // missing params / wrong counts fail in the descriptor validation
    scrapgpu_tensor_desc uin = a.desc;
    scrapgpu_tensor_desc uout = d.desc;
    CHECK_ST(exec_op(SCRAPGPU_OP_SCALE, &uin, 1, &uout, 1, nullptr, 0), SCRAPGPU_STATUS_INVALID_ARGUMENT);
    CHECK_ST(exec_op(SCRAPGPU_OP_ADD, &uin, 1, &uout, 1, nullptr, 0), SCRAPGPU_STATUS_INVALID_ARGUMENT);

    // GET_ROWS index values are data: an out-of-range row index is INVALID at execute time
    scrapgpu_get_rows_params gr{sizeof(gr), 0};
    TB rows = tmake(tdesc(SCRAPGPU_DTYPE_F32, 4, 3));     // 3 rows of 4
    TB ids = tmake(tdesc(SCRAPGPU_DTYPE_I32, 2));
    TB rows_out = tmake(tdesc(SCRAPGPU_DTYPE_F32, 4, 2));
    tfill(rows, {0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23});
    tfill_i32(ids, {2, 0});
    scrapgpu_tensor_desc gr_in[2] = {rows.desc, ids.desc};
    scrapgpu_tensor_desc gr_out = rows_out.desc;
    CHECK_OK(exec_op(SCRAPGPU_OP_GET_ROWS, gr_in, 2, &gr_out, 1, &gr, sizeof(gr)));
    std::vector<float> got = tread(rows_out);
    CHECK((got == std::vector<float>{20, 21, 22, 23, 0, 1, 2, 3}));

    tfill_i32(ids, {1, 7}); // row 7 does not exist
    CHECK_ST(exec_op(SCRAPGPU_OP_GET_ROWS, gr_in, 2, &gr_out, 1, &gr, sizeof(gr)),
             SCRAPGPU_STATUS_INVALID_ARGUMENT);
    std::printf("PASS  failure statuses\n");
}

// ---------------------------------------------------------------------------------------------
// plans: deep copies, buffer lifetime, execution
// ---------------------------------------------------------------------------------------------

void test_plan() {
    TB a = tmake(tdesc(SCRAPGPU_DTYPE_F32, 8));
    TB b = tmake(tdesc(SCRAPGPU_DTYPE_F32, 8));
    TB d = tmake(tdesc(SCRAPGPU_DTYPE_F32, 8));
    TB e = tmake(tdesc(SCRAPGPU_DTYPE_F32, 8));
    tfill(a, {1, 2, 3, 4, 5, 6, 7, 8});
    tfill(b, {10, 20, 30, 40, 50, 60, 70, 80});

    // node 0: d = a + b; node 1: e = d * scale (the params are copied at build time)
    scrapgpu_tensor_desc n0_in[2] = {a.desc, b.desc};
    scrapgpu_tensor_desc n0_out = d.desc;
    scrapgpu_scale_params sp{sizeof(sp), 2.0f, 0.0f};
    scrapgpu_tensor_desc n1_in = d.desc;
    scrapgpu_tensor_desc n1_out = e.desc;

    scrapgpu_plan_node nodes[2] = {};
    nodes[0].struct_size = sizeof(nodes[0]);
    nodes[0].op = make_desc(SCRAPGPU_OP_ADD, n0_in, 2, &n0_out, 1, nullptr, 0);
    nodes[1].struct_size = sizeof(nodes[1]);
    nodes[1].op = make_desc(SCRAPGPU_OP_SCALE, &n1_in, 1, &n1_out, 1, &sp, sizeof(sp));

    scrapgpu_plan_desc pd{};
    pd.struct_size = sizeof(pd);
    pd.nodes = nodes;
    pd.node_count = 2;

    scrapgpu_plan_t * plan = nullptr;
    CHECK_OK(scrapgpu_plan_build(g_dev, &pd, &plan));
    CHECK(plan != nullptr);
    sp.scale = 1.0e9f; // must not affect the built plan: params were deep-copied

    // a buffer referenced by a live plan cannot be freed
    CHECK_ST(scrapgpu_buffer_free(a.buf), SCRAPGPU_STATUS_INVALID_ARGUMENT);
    CHECK_ST(scrapgpu_buffer_free(d.buf), SCRAPGPU_STATUS_INVALID_ARGUMENT);

    CHECK_OK(scrapgpu_plan_execute(g_stream, plan));
    std::vector<float> got_d = tread(d);
    std::vector<float> got_e = tread(e);
    for (size_t i = 0; i < 8; ++i) {
        CHECK(got_d[i] == (float) (i + 1) + (float) ((i + 1) * 10));
        CHECK(got_e[i] == got_d[i] * 2.0f); // scale = 2, not 1e9
    }

    CHECK_OK(scrapgpu_plan_free(plan));
    CHECK_OK(scrapgpu_plan_free(nullptr));
    // buffers are owned by the caller and survive the plan
    CHECK_OK(scrapgpu_buffer_free(a.buf));
    CHECK_OK(scrapgpu_buffer_free(d.buf));
    g_buffers.erase(std::remove(g_buffers.begin(), g_buffers.end(), a.buf), g_buffers.end());
    g_buffers.erase(std::remove(g_buffers.begin(), g_buffers.end(), d.buf), g_buffers.end());

    // SOFT_MAX with a mask in a plan: the params->mask desc pointer is rebound to the plan's copy
    TB x = tmake(tdesc(SCRAPGPU_DTYPE_F32, 4, 2));
    TB mask = tmake(tdesc(SCRAPGPU_DTYPE_F32, 4, 2));
    TB sm_out = tmake(tdesc(SCRAPGPU_DTYPE_F32, 4, 2));
    tfill(x, {1, 0, 0, 0, 0, 0, 0, 0});
    tfill(mask, {0, -1e30f, -1e30f, -1e30f, 0, 0, 0, 0});
    scrapgpu_soft_max_params smp{sizeof(smp), 1.0f, &mask.desc};
    scrapgpu_tensor_desc sm_in[2] = {x.desc, mask.desc};
    scrapgpu_tensor_desc sm_dst = sm_out.desc;
    scrapgpu_plan_node sn{};
    sn.struct_size = sizeof(sn);
    sn.op = make_desc(SCRAPGPU_OP_SOFT_MAX, sm_in, 2, &sm_dst, 1, &smp, sizeof(smp));
    scrapgpu_plan_desc spd{};
    spd.struct_size = sizeof(spd);
    spd.nodes = &sn;
    spd.node_count = 1;
    scrapgpu_plan_t * splan = nullptr;
    CHECK_OK(scrapgpu_plan_build(g_dev, &spd, &splan));
    std::memset(&smp, 0xAB, sizeof(smp)); // scribble over the caller's params: the plan is independent
    CHECK_OK(scrapgpu_plan_execute(g_stream, splan));
    got_e = tread(sm_out);
    // row 0: mask kills all but element 0 -> [1,0,0,0]; row 1: uniform -> [0.25]*4
    CHECK(got_e[0] == 1.0f && got_e[1] == 0.0f && got_e[2] == 0.0f && got_e[3] == 0.0f);
    for (int i = 4; i < 8; ++i) {
        CHECK(std::fabs(got_e[(size_t) i] - 0.25f) < 1e-6f);
    }
    CHECK_OK(scrapgpu_plan_free(splan));

    // a plan with an unsupported node fails at build time
    scrapgpu_tensor_desc bad_in = a.desc;
    scrapgpu_tensor_desc bad_out = e.desc;
    scrapgpu_unary_params bad_unary{sizeof(bad_unary), (scrapgpu_unary_kind) 77};
    scrapgpu_plan_node bad{};
    bad.struct_size = sizeof(bad);
    bad.op = make_desc(SCRAPGPU_OP_UNARY, &bad_in, 1, &bad_out, 1, &bad_unary, sizeof(bad_unary));
    scrapgpu_plan_desc bpd{};
    bpd.struct_size = sizeof(bpd);
    bpd.nodes = &bad;
    bpd.node_count = 1;
    scrapgpu_plan_t * bad_plan = nullptr;
    CHECK_ST(scrapgpu_plan_build(g_dev, &bpd, &bad_plan), SCRAPGPU_STATUS_UNSUPPORTED);
    CHECK(bad_plan == nullptr);

    // stream/plan pairing is checked by the core
    CHECK_ST(scrapgpu_plan_execute(g_stream, reinterpret_cast<scrapgpu_plan_t *>(&spd)),
             SCRAPGPU_STATUS_INVALID_ARGUMENT);

    std::printf("PASS  plans: deep copy, buffer lifetime, masked softmax, failures\n");
}

} // namespace

int main() {
    test_register();
    test_buffers();
    test_foreign_objects();
    test_tensor_transfers();
    test_elementwise();
    test_mul_mat_f32();
    test_mul_mat_q4();
    test_supported_matrix();
    test_failure_statuses();
    test_plan();

    CHECK_OK(scrapgpu_stream_free(g_stream));
    for (scrapgpu_buffer_t * b : g_buffers) {
        CHECK_OK(scrapgpu_buffer_free(b));
    }
    std::printf("PASS  all CPU backend tests\n");
    return 0;
}

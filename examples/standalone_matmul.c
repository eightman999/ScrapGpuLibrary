/* A standalone C program that runs a Q4_0 matmul on the GPU through the ScrapGpu C API only:
 * no ggml, no llama.cpp, no OpenCL types in user code.
 *
 *   weights (F32, random) --scrapgpu_ref_quantize--> Q4_0 bytes --upload as R32--> GPU
 *   x (F32) --upload--> GPU --MUL_MAT--> y --download--> compare with scrapgpu_ref_execute
 */
#include "scrapgpu/api.h"
#include "scrapgpu/buffer.h"
#include "scrapgpu/device.h"
#include "scrapgpu/opencl11.h"
#include "scrapgpu/op.h"
#include "scrapgpu/reference.h"
#include "scrapgpu/status.h"
#include "scrapgpu/stream.h"
#include "scrapgpu/tensor.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define K 256  /* a multiple of the Q4_0 block size (32) */
#define M 512
#define N 7    /* a small batch of vectors */

static void fail(const char * what, scrapgpu_status st) {
    fprintf(stderr, "%s: %s (%s)\n", what, scrapgpu_status_string(st), scrapgpu_last_error());
    exit(1);
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + 1e-9 * (double) ts.tv_nsec;
}

/* xorshift32: no <random> in C99 */
static uint32_t rng_state = 0x9e3779b9u;
static float frand(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return ((float) (rng_state >> 8) / 16777216.0f) * 2.0f - 1.0f;
}

static scrapgpu_tensor_desc tensor(scrapgpu_dtype dt, scrapgpu_buffer_t * buf, int64_t n0, int64_t n1, scrapgpu_layout layout) {
    scrapgpu_tensor_desc t;
    memset(&t, 0, sizeof(t));
    t.struct_size = sizeof(t);
    t.dtype = dt;
    t.ne[0] = n0;
    t.ne[1] = n1;
    t.ne[2] = t.ne[3] = 1;
    t.nb[0] = scrapgpu_dtype_type_size(dt);
    t.nb[1] = scrapgpu_tensor_row_size(dt, n0);
    t.nb[2] = t.nb[1] * (uint64_t) n1;
    t.nb[3] = t.nb[2];
    t.buffer = buf;
    t.offset = 0;
    t.layout = layout;
    return t;
}

int main(void) {
    scrapgpu_status st;

    st = scrapgpu_opencl11_register();
    if (st != SCRAPGPU_STATUS_OK) {
        fail("scrapgpu_opencl11_register", st);
    }
    if (scrapgpu_get_device_count() == 0) {
        printf("no OpenCL device found, nothing to run\n");
        return 0;
    }
    scrapgpu_device_t * dev = NULL;
    st = scrapgpu_get_device(0, &dev);
    if (st != SCRAPGPU_STATUS_OK) {
        fail("scrapgpu_get_device", st);
    }
    {
        scrapgpu_device_info info;
        memset(&info, 0, sizeof(info));
        st = scrapgpu_device_info_init(&info, sizeof(info));
        if (st == SCRAPGPU_STATUS_OK) {
            st = scrapgpu_device_get_info(dev, &info);
        }
        if (st == SCRAPGPU_STATUS_OK) {
            printf("device: %s (%s)\n", info.name, info.tuning_profile_name);
        }
    }

    scrapgpu_stream_t * stream = NULL;
    st = scrapgpu_stream_create(dev, &stream);
    if (st != SCRAPGPU_STATUS_OK) {
        fail("scrapgpu_stream_create", st);
    }

    /* host data: F32 weights -> Q4_0 natural bytes */
    float * w_f = (float *) malloc((size_t) K * M * sizeof(float));
    float * x = (float *) malloc((size_t) K * N * sizeof(float));
    float * y = (float *) malloc((size_t) M * N * sizeof(float));
    float * y_ref = (float *) malloc((size_t) M * N * sizeof(float));
    const size_t w_bytes = (size_t) (K / 32) * M * 18; /* Q4_0: 18 bytes per 32 weights */
    void * w_q = malloc(w_bytes);
    if (!w_f || !x || !y || !y_ref || !w_q) {
        fprintf(stderr, "out of host memory\n");
        return 1;
    }
    for (int64_t i = 0; i < (int64_t) K * M; ++i) {
        w_f[i] = frand();
    }
    for (int64_t i = 0; i < (int64_t) K * N; ++i) {
        x[i] = frand();
    }
    st = scrapgpu_ref_quantize(SCRAPGPU_DTYPE_Q4_0, w_f, w_q, (int64_t) K * M);
    if (st != SCRAPGPU_STATUS_OK) {
        fail("scrapgpu_ref_quantize", st);
    }

    scrapgpu_buffer_t * bw = NULL, *bx = NULL, *by = NULL;
    if ((st = scrapgpu_buffer_alloc(dev, w_bytes, &bw)) != SCRAPGPU_STATUS_OK) fail("buffer_alloc w", st);
    if ((st = scrapgpu_buffer_alloc(dev, (size_t) K * N * 4, &bx)) != SCRAPGPU_STATUS_OK) fail("buffer_alloc x", st);
    if ((st = scrapgpu_buffer_alloc(dev, (size_t) M * N * 4, &by)) != SCRAPGPU_STATUS_OK) fail("buffer_alloc y", st);

    /* R32 layout: the upload packs the natural Q4_0 bytes into the kernel layout */
    scrapgpu_tensor_desc w = tensor(SCRAPGPU_DTYPE_Q4_0, bw, K, M, SCRAPGPU_LAYOUT_R32);
    scrapgpu_tensor_desc xt = tensor(SCRAPGPU_DTYPE_F32, bx, K, N, SCRAPGPU_LAYOUT_NATURAL);
    scrapgpu_tensor_desc yt = tensor(SCRAPGPU_DTYPE_F32, by, M, N, SCRAPGPU_LAYOUT_NATURAL);

    if ((st = scrapgpu_tensor_upload(stream, &w, w_q, w_bytes)) != SCRAPGPU_STATUS_OK) fail("upload w", st);
    if ((st = scrapgpu_tensor_upload(stream, &xt, x, (size_t) K * N * 4)) != SCRAPGPU_STATUS_OK) fail("upload x", st);

    const scrapgpu_tensor_desc ins[2] = {w, xt};
    scrapgpu_mul_mat_params mm;
    memset(&mm, 0, sizeof(mm));
    mm.struct_size = sizeof(mm);
    scrapgpu_op_desc op;
    memset(&op, 0, sizeof(op));
    op.struct_size = sizeof(op);
    op.op = SCRAPGPU_OP_MUL_MAT;
    op.inputs = ins;
    op.input_count = 2;
    op.outputs = &yt;
    op.output_count = 1;
    op.params = &mm;
    op.params_size = sizeof(mm);

    const char * reason = NULL;
    st = scrapgpu_op_supported(dev, &op, &reason);
    if (st != SCRAPGPU_STATUS_OK) {
        fprintf(stderr, "MUL_MAT is not supported: %s\n", reason ? reason : "?");
        return 1;
    }

    /* the first call builds the OpenCL program (seconds on Mesa): run it once before timing */
    if ((st = scrapgpu_op_execute(stream, &op)) != SCRAPGPU_STATUS_OK) fail("op_execute", st);
    if ((st = scrapgpu_stream_sync(stream)) != SCRAPGPU_STATUS_OK) fail("stream_sync", st);

    const int reps = 10;
    const double t0 = now_s();
    for (int r = 0; r < reps; ++r) {
        if ((st = scrapgpu_op_execute(stream, &op)) != SCRAPGPU_STATUS_OK) fail("op_execute", st);
    }
    if ((st = scrapgpu_stream_sync(stream)) != SCRAPGPU_STATUS_OK) fail("stream_sync", st);
    const double ms = (now_s() - t0) * 1e3 / reps;

    if ((st = scrapgpu_tensor_download(stream, &yt, y, (size_t) M * N * 4)) != SCRAPGPU_STATUS_OK) fail("download y", st);

    /* reference: the same op on the host, weights in the natural Q4_0 layout */
    {
        scrapgpu_tensor_desc wn = w, xn = xt, yn = yt;
        wn.layout = SCRAPGPU_LAYOUT_NATURAL;
        wn.buffer = NULL;
        xn.buffer = NULL;
        yn.buffer = NULL;
        const scrapgpu_tensor_desc rins[2] = {wn, xn};
        scrapgpu_op_desc rop = op;
        rop.inputs = rins;
        rop.outputs = &yn;
        void * rdata_in[2] = {w_q, x};
        void * rdata_out[1] = {y_ref};
        st = scrapgpu_ref_execute(&rop, rdata_in, rdata_out);
        if (st != SCRAPGPU_STATUS_OK) {
            fail("scrapgpu_ref_execute", st);
        }
    }

    double num = 0.0, den = 0.0;
    for (int64_t i = 0; i < (int64_t) M * N; ++i) {
        const double d = (double) y[i] - y_ref[i];
        num += d * d;
        den += (double) y_ref[i] * y_ref[i];
    }
    printf("MUL_MAT Q4_0(R32) [%d x %d] * [%d x %d]: NMSE %.3e, %.3f ms per call (after a warm-up)\n", K, M, K, N,
           den > 0.0 ? num / den : num, ms);

    scrapgpu_buffer_free(bw);
    scrapgpu_buffer_free(bx);
    scrapgpu_buffer_free(by);
    scrapgpu_stream_free(stream);
    free(w_f);
    free(x);
    free(y);
    free(y_ref);
    free(w_q);
    return 0;
}

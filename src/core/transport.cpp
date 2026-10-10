// The multi-GPU transport of the public C API (docs/API_CONTRACT.md, "デバイスをまたぐ転送"):
// scrapgpu_tensor_copy moves a contiguous tensor between any two devices. This is generic core code
// on top of the backend vtable only: a device-side buffer_copy when both tensors live on the same
// device, otherwise a host staging buffer - a blocking read on the source stream, then a write on
// the destination stream (OpenCL 1.1 has no cross-context events).

#include "scrapgpu/backend.h"
#include "scrapgpu/status.h"
#include "scrapgpu/tensor.h"

#include "tensor/tensor.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace {

scrapgpu_status copy_invalid(const char * message) {
    scrapgpu_set_last_error(message);
    return SCRAPGPU_STATUS_INVALID_ARGUMENT;
}

scrapgpu_status copy_unsupported(const char * message) {
    scrapgpu_set_last_error(message);
    return SCRAPGPU_STATUS_UNSUPPORTED;
}

// the backend an object belongs to (the first member of every backend object is a
// scrapgpu_object_header), nullptr when it does not carry the header
const scrapgpu_backend_vtable * backend_of(const void * object) {
    return object == nullptr ? nullptr : static_cast<const scrapgpu_object_header *>(object)->backend;
}

// the caller-visible fields of a tensor desc (a smaller struct_size keeps the defaults of the tail:
// buffer = NULL, offset = 0, layout = NATURAL)
scrapgpu_tensor_desc load_desc(const scrapgpu_tensor_desc & in, bool * ok) {
    *ok = in.struct_size >= sizeof(size_t);
    scrapgpu_tensor_desc out = {};
    if (*ok) {
        std::memcpy(&out, &in, std::min(in.struct_size, sizeof(out)));
        out.struct_size = sizeof(out);
    }
    return out;
}

} // namespace

extern "C" {

scrapgpu_status scrapgpu_tensor_copy(scrapgpu_stream_t * src_stream, const scrapgpu_tensor_desc * src,
                                     scrapgpu_stream_t * dst_stream, const scrapgpu_tensor_desc * dst) {
    if (src_stream == nullptr || dst_stream == nullptr || src == nullptr || dst == nullptr) {
        return copy_invalid("src_stream, dst_stream, src or dst is null");
    }
    bool ok = false;
    const scrapgpu_tensor_desc s = load_desc(*src, &ok);
    if (!ok) {
        return copy_invalid("src tensor desc is too small");
    }
    const scrapgpu_tensor_desc d = load_desc(*dst, &ok);
    if (!ok) {
        return copy_invalid("dst tensor desc is too small");
    }

    const sgt::TensorDesc sgt_src = sgt::desc_of(s);
    if (sgt::dtype_type_size(s.dtype) == 0) {
        return copy_unsupported("unknown tensor dtype");
    }
    if (s.dtype != d.dtype) {
        return copy_unsupported("src and dst dtypes differ");
    }
    for (int i = 0; i < sgt::kMaxDims; ++i) {
        if (s.ne[i] != d.ne[i]) {
            return copy_unsupported("src and dst shapes differ");
        }
        if (s.ne[i] < 0) {
            return copy_invalid("negative tensor dim");
        }
    }
    if (s.layout == SCRAPGPU_LAYOUT_R32 || d.layout == SCRAPGPU_LAYOUT_R32) {
        return copy_unsupported("R32 tensors are not allowed: their device bytes are in the repacked layout; "
                                "move them through the host with tensor_download / tensor_upload");
    }
    if (!scrapgpu_tensor_is_contiguous(&s) || !scrapgpu_tensor_is_contiguous(&d)) {
        return copy_unsupported("tensor is not contiguous; use buffer_read / buffer_write for raw strided transfers");
    }
    if (s.buffer == nullptr || d.buffer == nullptr) {
        return copy_invalid("src or dst tensor desc has no buffer");
    }

    const uint64_t size = sgt::nbytes(sgt_src);
    if (size == 0) {
        return SCRAPGPU_STATUS_OK; // an empty tensor: nothing to move, the ordering holds trivially
    }
    if (size > (uint64_t) std::numeric_limits<size_t>::max()) {
        return copy_unsupported("the tensor does not fit a host staging buffer");
    }
    const size_t bytes = (size_t) size;

    const scrapgpu_backend_vtable * src_backend = backend_of(src_stream);
    const scrapgpu_backend_vtable * dst_backend = backend_of(dst_stream);
    if (src_backend == nullptr || dst_backend == nullptr) {
        return copy_invalid("stream does not belong to a registered backend");
    }

    // same device: one device-side copy command. It is enqueued on dst_stream, so a different
    // src_stream of the same device is synchronized first; with the same (in-order) stream the
    // queue order is the guarantee. An overlapping copy inside one buffer falls through to the
    // host path (the device copy would be undefined there).
    scrapgpu_device_t * src_dev = src_backend->stream_device != nullptr ? src_backend->stream_device(src_stream) : nullptr;
    scrapgpu_device_t * dst_dev = dst_backend->stream_device != nullptr ? dst_backend->stream_device(dst_stream) : nullptr;
    const bool overlapping = s.buffer == d.buffer &&
        (s.offset < d.offset ? d.offset - s.offset < size : s.offset - d.offset < size);
    if (src_dev != nullptr && src_dev == dst_dev && !overlapping && dst_backend->buffer_copy != nullptr &&
        (src_stream == dst_stream || src_backend->stream_sync != nullptr)) {
        if (src_stream != dst_stream) {
            const scrapgpu_status st = src_backend->stream_sync(src_stream);
            if (st != SCRAPGPU_STATUS_OK) {
                return st;
            }
        }
        return dst_backend->buffer_copy(dst_stream, d.buffer, d.offset, s.buffer, s.offset, bytes, 0);
    }

    // different devices or backends (and the corner cases above): through the host. The blocking
    // read waits for the work enqueued on src_stream before it; the write is enqueued on dst_stream
    // and therefore precedes the work enqueued later on it. The write is blocking as well, so the
    // staging buffer can be released when the call returns.
    if (src_backend->buffer_read == nullptr || dst_backend->buffer_write == nullptr) {
        return copy_unsupported("the src backend cannot read / the dst backend cannot write buffers");
    }
    uint8_t * host = (uint8_t *) std::malloc(bytes);
    if (host == nullptr) {
        scrapgpu_set_last_error("out of host memory for the staging buffer");
        return SCRAPGPU_STATUS_OUT_OF_MEMORY;
    }
    scrapgpu_status st = src_backend->buffer_read(src_stream, s.buffer, s.offset, host, bytes, 1);
    if (st == SCRAPGPU_STATUS_OK) {
        st = dst_backend->buffer_write(dst_stream, d.buffer, d.offset, host, bytes, 1);
    }
    std::free(host);
    return st;
}

}

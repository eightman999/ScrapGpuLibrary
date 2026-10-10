#include "scrapgpu/compat.h"
#include "scrapgpu/backend.h"
#include "scrapgpu/reference.h"
#include "tensor/tensor.h"

#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

namespace {
scrapgpu_status fail(scrapgpu_status status, const char * reason) {
    scrapgpu_set_last_error(reason);
    return status;
}

bool same_shape(const scrapgpu_tensor_desc & a, const scrapgpu_tensor_desc & b) {
    for (int i = 0; i < 4; ++i) if (a.ne[i] != b.ne[i]) return false;
    return true;
}

// Deliberately conservative: this whitelist defines the supported portable
// fallback contract. Validate before invoking a reference kernel or doing I/O.
scrapgpu_status fallback_supported(const scrapgpu_op_desc & op) {
    const auto & a = op.inputs[0];
    const auto & d = op.outputs[0];
    const bool copy = op.op == SCRAPGPU_OP_CPY || op.op == SCRAPGPU_OP_DUP || op.op == SCRAPGPU_OP_CONT;
    if (op.output_count != 1 || (!copy && d.dtype != SCRAPGPU_DTYPE_F32))
        return fail(SCRAPGPU_STATUS_UNSUPPORTED, "CPU fallback requires one F32 output");
    switch (op.op) {
        case SCRAPGPU_OP_ADD:
        case SCRAPGPU_OP_SUB:
        case SCRAPGPU_OP_MUL:
        case SCRAPGPU_OP_DIV: {
            if (!same_shape(a, d)) break;
            const auto & b = op.inputs[1];
            for (int i = 0; i < 4; ++i) if (a.ne[i] % b.ne[i])
                return fail(SCRAPGPU_STATUS_INVALID_ARGUMENT, "CPU fallback broadcast mismatch");
            return SCRAPGPU_STATUS_OK;
        }
        case SCRAPGPU_OP_MUL_MAT: {
            const auto & b = op.inputs[1];
            if (a.ne[0] == b.ne[0] && d.ne[0] == a.ne[1] && d.ne[1] == b.ne[1] &&
                d.ne[2] == b.ne[2] && d.ne[3] == b.ne[3] &&
                b.ne[2] % a.ne[2] == 0 && b.ne[3] % a.ne[3] == 0)
                return SCRAPGPU_STATUS_OK;
            break;
        }
        case SCRAPGPU_OP_SCALE:
        case SCRAPGPU_OP_CLAMP:
        case SCRAPGPU_OP_SQR:
        case SCRAPGPU_OP_SQRT:
        case SCRAPGPU_OP_RMS_NORM:
        case SCRAPGPU_OP_UNARY:
            if (a.dtype == SCRAPGPU_DTYPE_F32 && same_shape(a, d)) return SCRAPGPU_STATUS_OK;
            break;
        case SCRAPGPU_OP_SOFT_MAX:
            if (a.dtype != SCRAPGPU_DTYPE_F32 || !same_shape(a, d)) break;
            if (op.input_count == 2) {
                const auto & mask = op.inputs[1];
                if ((mask.dtype != SCRAPGPU_DTYPE_F32 && mask.dtype != SCRAPGPU_DTYPE_F16) ||
                    mask.ne[0] != a.ne[0] || mask.ne[1] < a.ne[1] ||
                    a.ne[2] % mask.ne[2] || a.ne[3] % mask.ne[3]) break;
            }
            return SCRAPGPU_STATUS_OK;
        case SCRAPGPU_OP_CPY:
        case SCRAPGPU_OP_DUP:
        case SCRAPGPU_OP_CONT:
            if ((a.dtype == SCRAPGPU_DTYPE_F32 || a.dtype == SCRAPGPU_DTYPE_F16 || a.dtype == SCRAPGPU_DTYPE_BF16) &&
                sgt::nelements(sgt::desc_of(a)) == sgt::nelements(sgt::desc_of(d)))
                return SCRAPGPU_STATUS_OK;
            break;
        default:
            return fail(SCRAPGPU_STATUS_UNSUPPORTED, "op has no compatible CPU fallback");
    }
    return fail(SCRAPGPU_STATUS_UNSUPPORTED, "shape or dtype has no compatible CPU fallback");
}

scrapgpu_status execute(scrapgpu_device_t * device, scrapgpu_stream_t * stream,
                        const scrapgpu_op_desc * desc, const scrapgpu_compat_options * options,
                        scrapgpu_execution_report & report) {
    if (!device || !stream || !desc)
        return fail(SCRAPGPU_STATUS_INVALID_ARGUMENT, "compatible execution needs device, stream and op");
    const auto * backend = reinterpret_cast<const scrapgpu_object_header *>(stream)->backend;
    if (!backend || backend != reinterpret_cast<const scrapgpu_object_header *>(device)->backend ||
        !backend->stream_device || backend->stream_device(stream) != device)
        return fail(SCRAPGPU_STATUS_INVALID_ARGUMENT, "stream does not belong to the selected device");
    if (options && options->struct_size < sizeof(*options))
        return fail(SCRAPGPU_STATUS_INVALID_ARGUMENT, "compat options are too small");
    // Validate sizes before the generic op validator reads tensor dtype fields,
    // and before copying entire descriptors for the staging path.
    if (desc->struct_size < sizeof(*desc) || desc->input_count > 32 || desc->output_count > 32 ||
        (desc->input_count && !desc->inputs) || (desc->output_count && !desc->outputs))
        return fail(SCRAPGPU_STATUS_INVALID_ARGUMENT, "compatible op descriptor is incomplete");
    for (size_t i = 0; i < desc->input_count; ++i)
        if (desc->inputs[i].struct_size < sizeof(scrapgpu_tensor_desc))
            return fail(SCRAPGPU_STATUS_INVALID_ARGUMENT, "compatible input descriptor is too small");
    for (size_t i = 0; i < desc->output_count; ++i)
        if (desc->outputs[i].struct_size < sizeof(scrapgpu_tensor_desc))
            return fail(SCRAPGPU_STATUS_INVALID_ARGUMENT, "compatible output descriptor is too small");
    scrapgpu_status status = scrapgpu_op_desc_validate(desc);
    if (status != SCRAPGPU_STATUS_OK) return status;
    status = scrapgpu_op_supported(device, desc, nullptr);
    if (status == SCRAPGPU_STATUS_OK) {
        report.path = SCRAPGPU_EXECUTION_NATIVE;
        return scrapgpu_op_execute(stream, desc);
    }
    if (status != SCRAPGPU_STATUS_UNSUPPORTED || !options || !options->allow_cpu_fallback) return status;
    if (desc->input_count == 0 || desc->output_count == 0)
        return fail(SCRAPGPU_STATUS_UNSUPPORTED, "metadata-only ops have no CPU fallback");

    std::vector<scrapgpu_tensor_desc> inputs(desc->inputs, desc->inputs + desc->input_count);
    std::vector<scrapgpu_tensor_desc> outputs(desc->outputs, desc->outputs + desc->output_count);
    std::vector<size_t> sizes;
    uint64_t total = 0;
    // Inspect the complete request before allocation, transfers or output writes.
    for (auto * tensors : {&inputs, &outputs}) {
        for (auto & t : *tensors) {
            if (t.struct_size < sizeof(t) || !t.buffer)
                return fail(SCRAPGPU_STATUS_INVALID_ARGUMENT, "fallback tensor descriptor is incomplete");
            for (int i = 0; i < 4; ++i) if (t.ne[i] <= 0)
                return fail(SCRAPGPU_STATUS_INVALID_ARGUMENT, "fallback tensor dimensions must be positive");
            const size_t block = scrapgpu_dtype_block_size(t.dtype);
            if (!block || !scrapgpu_dtype_type_size(t.dtype) || t.ne[0] % block)
                return fail(SCRAPGPU_STATUS_UNSUPPORTED, "invalid fallback dtype or quantization block");
            if (t.layout != SCRAPGPU_LAYOUT_NATURAL && t.layout != SCRAPGPU_LAYOUT_R32)
                return fail(SCRAPGPU_STATUS_UNSUPPORTED, "unknown fallback tensor layout");
            if (t.layout == SCRAPGPU_LAYOUT_R32 && !scrapgpu_tensor_r32_eligible(&t))
                return fail(SCRAPGPU_STATUS_UNSUPPORTED, "fallback R32 tensor is not eligible");
            if (!scrapgpu_tensor_is_contiguous(&t))
                return fail(SCRAPGPU_STATUS_UNSUPPORTED, "CPU fallback requires contiguous tensors");
            const uint64_t size = sgt::nbytes(sgt::desc_of(t));
            if (!size || size > SIZE_MAX || size > options->max_staging_bytes ||
                total > options->max_staging_bytes - size || t.offset > UINT64_MAX - size)
                return fail(SCRAPGPU_STATUS_OUT_OF_MEMORY, "CPU fallback exceeds staging budget or address space");
            sizes.push_back(static_cast<size_t>(size));
            total += size;
        }
    }
    status = fallback_supported(*desc);
    if (status != SCRAPGPU_STATUS_OK) return status;

    std::vector<std::vector<unsigned char>> staging(sizes.size());
    std::vector<void *> in_data(inputs.size()), out_data(outputs.size());
    for (size_t i = 0; i < sizes.size(); ++i) staging[i].resize(sizes[i]);
    report.staging_bytes = total;
    size_t index = 0;
    for (auto * tensors : {&inputs, &outputs}) {
        auto & ptrs = tensors == &inputs ? in_data : out_data;
        for (size_t i = 0; i < tensors->size(); ++i, ++index) {
            // Download outputs too: this preserves destinations of future
            // partial-write reference ops if the whitelist is extended.
            auto & t = (*tensors)[i];
            status = scrapgpu_tensor_download(stream, &t, staging[index].data(), sizes[index]);
            if (status != SCRAPGPU_STATUS_OK) return status;
            report.downloaded_bytes += sizes[index];
            ptrs[i] = staging[index].data();
            t.offset = 0;
            t.layout = SCRAPGPU_LAYOUT_NATURAL;
            t.buffer = nullptr;
        }
    }
    scrapgpu_op_desc host = *desc;
    host.inputs = inputs.data();
    host.outputs = outputs.data();
    scrapgpu_soft_max_params softmax = {};
    if (host.op == SCRAPGPU_OP_SOFT_MAX) {
        softmax = *static_cast<const scrapgpu_soft_max_params *>(host.params);
        softmax.mask = inputs.size() > 1 ? &inputs[1] : nullptr;
        host.params = &softmax;
        host.params_size = sizeof(softmax);
    }
    report.path = SCRAPGPU_EXECUTION_CPU_FALLBACK;
    status = scrapgpu_ref_execute(&host, in_data.data(), out_data.data());
    if (status != SCRAPGPU_STATUS_OK) return status;
    for (size_t i = 0; i < outputs.size(); ++i) {
        status = scrapgpu_tensor_upload(stream, &desc->outputs[i], out_data[i], sizes[inputs.size() + i]);
        if (status != SCRAPGPU_STATUS_OK) return status;
        report.uploaded_bytes += sizes[inputs.size() + i];
    }
    scrapgpu_clear_last_error();
    return SCRAPGPU_STATUS_OK;
}
}

extern "C" scrapgpu_status scrapgpu_op_execute_compatible(
    scrapgpu_device_t * device, scrapgpu_stream_t * stream, const scrapgpu_op_desc * desc,
    const scrapgpu_compat_options * options, scrapgpu_execution_report * report) {
    if (report && report->struct_size < sizeof(*report))
        return fail(SCRAPGPU_STATUS_INVALID_ARGUMENT, "execution report is too small");
    scrapgpu_execution_report result = {sizeof(result), SCRAPGPU_EXECUTION_NONE, 0, 0, 0};
    scrapgpu_status status;
    try { status = execute(device, stream, desc, options, result); }
    catch (const std::bad_alloc &) { status = fail(SCRAPGPU_STATUS_OUT_OF_MEMORY, "CPU fallback allocation failed"); }
    catch (const std::length_error &) { status = fail(SCRAPGPU_STATUS_OUT_OF_MEMORY, "CPU fallback allocation is too large"); }
    if (report) {
        const size_t caller_size = report->struct_size;
        *report = result;
        report->struct_size = caller_size;
    }
    return status;
}

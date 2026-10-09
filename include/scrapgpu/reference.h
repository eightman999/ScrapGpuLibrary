#ifndef SCRAPGPU_REFERENCE_H
#define SCRAPGPU_REFERENCE_H

#include "scrapgpu/op.h"

#ifdef __cplusplus
extern "C" {
#endif

// CPU reference implementations of the ops (docs/API_CONTRACT.md chapter 10).
//
// Conventions of scrapgpu_ref_execute:
//   - input_data[i] / output_data[i] are host pointers to the first byte of the
//     data of desc->inputs[i] / desc->outputs[i]; desc->offset is added, the
//     strides desc->nb[] are in bytes and are honoured for every tensor.
//   - Quantized tensors must use SCRAPGPU_LAYOUT_NATURAL in ggml's block format
//     (the same bytes as GGUF). SCRAPGPU_LAYOUT_R32 is rejected with
//     SCRAPGPU_STATUS_UNSUPPORTED.
//   - The descriptor is validated with scrapgpu_op_desc_validate() first; the
//     reference then checks the shapes and strides it needs and returns
//     SCRAPGPU_STATUS_INVALID_ARGUMENT / SCRAPGPU_STATUS_UNSUPPORTED instead of
//     crashing on combinations it does not implement.
//   - Optional extra tensors travel as additional inputs:
//       SOFT_MAX: [src0] or [src0, mask (F32 or F16)]
//       ROPE:     [src0, positions (I32)] or [src0, positions, freq_factors (F32)]
//     (the desc pointers inside the params structs carry only metadata.)
//   - Compute is done in float (accumulation in float or double), the results
//     are written in the output dtype (F16 stores round to nearest even).
//
// Quantization helpers:
//   - scrapgpu_ref_dequantize / scrapgpu_ref_quantize work on rows of natural
//     blocks; n must be a positive multiple of the block size.
//   - Q4_0 / Q8_0 quantize follows ggml's reference implementation
//     (quantize_row_q4_0_ref / quantize_row_q8_0_ref).
//   - Q4_K / Q6_K quantize uses a simple method in the correct block format:
//     per sub-block min/max scale search without the weighted refinement of
//     ggml, scales and quants rounded to the nearest integer. Dequantization is
//     exact (the same arithmetic as ggml's dequantize_row_q4_K / q6_K).
SCRAPGPU_API scrapgpu_status scrapgpu_ref_execute(const scrapgpu_op_desc * desc, void * const * input_data, void * const * output_data);

SCRAPGPU_API scrapgpu_status scrapgpu_ref_dequantize(scrapgpu_dtype dtype, const void * src, float * dst, int64_t n);

SCRAPGPU_API scrapgpu_status scrapgpu_ref_quantize(scrapgpu_dtype dtype, const float * src, void * dst, int64_t n);

#ifdef __cplusplus
}
#endif

#endif

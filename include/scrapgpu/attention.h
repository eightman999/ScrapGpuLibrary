#ifndef SCRAPGPU_ATTENTION_H
#define SCRAPGPU_ATTENTION_H

#include "scrapgpu/stream.h"
#include "scrapgpu/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

// High-level scaled dot product attention:
//   query  = [D,  Q, Hq,  B]
//   key    = [D,  K, Hkv, B]
//   value  = [Dv, K, Hkv, B]
//   output = [Dv, Q, Hq,  B]
//
// All tensors must be F32, natural-layout and contiguous. Grouped query attention is supported when
// Hq is divisible by Hkv; query head h uses key/value head h / (Hq / Hkv).
//
// The optional additive mask is also F32 and contiguous:
//   mask = [K, Q, 1 or Hq, 1 or B]
//
// When causal is non-zero, key position k is visible to query position q only when:
//   k <= query_position_offset + q
//
// Temporary device workspace is bounded by max_workspace_bytes. The implementation tiles the query
// dimension and rejects the call before allocation when the budget cannot hold the transposed V
// slice plus at least one query tile.
typedef struct scrapgpu_attention_params {
    size_t struct_size;
    const scrapgpu_tensor_desc * query;
    const scrapgpu_tensor_desc * key;
    const scrapgpu_tensor_desc * value;
    const scrapgpu_tensor_desc * mask;   // optional, may be NULL
    const scrapgpu_tensor_desc * output;
    uint64_t query_position_offset;
    uint64_t max_workspace_bytes;
    int causal;
    // Explicit finite score multiplier; normally 1/sqrt(D). Zero and negative
    // values are valid. Only -INFINITY in an additive mask means fully masked.
    float scale;
} scrapgpu_attention_params;

SCRAPGPU_API scrapgpu_status scrapgpu_scaled_dot_product_attention(scrapgpu_device_t * device,
                                                                   scrapgpu_stream_t * stream,
                                                                   const scrapgpu_attention_params * params);

#ifdef __cplusplus
}
#endif

#endif

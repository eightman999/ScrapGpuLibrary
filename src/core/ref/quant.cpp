// Natural-layout quantized blocks (same bytes as GGUF / ggml's ggml-quants.c):
//   Q4_0: 18 bytes = f16 d, uint8 qs[16]      value[j]    = (qs[j] & 0xF) - 8) * d
//                                              value[j+16] = (qs[j] >> 4)  - 8) * d
//   Q8_0: 34 bytes = f16 d, int8 qs[32]       value[j] = qs[j] * d
//   Q4_K: 144 bytes = f16 d, f16 dmin, uint8 scales[12], uint8 qs[128]
//   Q6_K: 210 bytes = uint8 ql[128], uint8 qh[64], int8 scales[16], f16 d

#include "internal.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace scrapgpu {
namespace ref {

namespace {

// ggml's get_scale_min_k4: 6-bit scales and mins of the 8 sub-blocks
void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = static_cast<uint8_t>((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *m = static_cast<uint8_t>((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

void dequantize_row_q4_0(const uint8_t * row, int64_t n, float * dst) {
    const int64_t nb = n / 32;
    for (int64_t b = 0; b < nb; ++b) {
        const uint8_t * blk = row + b * 18;
        const float d = half_to_float(load_u16(blk));
        for (int j = 0; j < 16; ++j) {
            const int v = blk[2 + j];
            dst[b * 32 + j] = static_cast<float>((v & 0x0F) - 8) * d;
            dst[b * 32 + j + 16] = static_cast<float>((v >> 4) - 8) * d;
        }
    }
}

void dequantize_row_q8_0(const uint8_t * row, int64_t n, float * dst) {
    const int64_t nb = n / 32;
    for (int64_t b = 0; b < nb; ++b) {
        const uint8_t * blk = row + b * 34;
        const float d = half_to_float(load_u16(blk));
        for (int j = 0; j < 32; ++j) {
            dst[b * 32 + j] = static_cast<float>(static_cast<int8_t>(blk[2 + j])) * d;
        }
    }
}

void dequantize_row_q4_k(const uint8_t * row, int64_t n, float * dst) {
    const int64_t nb = n / 256;
    for (int64_t b = 0; b < nb; ++b) {
        const uint8_t * blk = row + b * 144;
        const uint8_t * q = blk + 16;
        const float d = half_to_float(load_u16(blk));
        const float dmin = half_to_float(load_u16(blk + 2));
        int is = 0;
        uint8_t sc = 0;
        uint8_t m = 0;
        float * y = dst + b * 256;
        for (int j = 0; j < 256; j += 64) {
            get_scale_min_k4(is + 0, blk + 4, &sc, &m);
            const float d1 = d * sc;
            const float m1 = dmin * m;
            get_scale_min_k4(is + 1, blk + 4, &sc, &m);
            const float d2 = d * sc;
            const float m2 = dmin * m;
            for (int l = 0; l < 32; ++l) {
                y[j + l] = d1 * static_cast<float>(q[l] & 0x0F) - m1;
                y[j + l + 32] = d2 * static_cast<float>(q[l] >> 4) - m2;
            }
            q += 32;
            is += 2;
        }
    }
}

void dequantize_row_q6_k(const uint8_t * row, int64_t n, float * dst) {
    const int64_t nb = n / 256;
    for (int64_t b = 0; b < nb; ++b) {
        const uint8_t * blk = row + b * 210;
        const float d = half_to_float(load_u16(blk + 208));
        const uint8_t * ql = blk;
        const uint8_t * qh = blk + 128;
        const int8_t * sc = reinterpret_cast<const int8_t *>(blk + 192);
        float * y = dst + b * 256;
        for (int j = 0; j < 256; j += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int q1 = ((ql[l + 0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = ((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[j + l + 0] = d * static_cast<float>(sc[is + 0]) * static_cast<float>(q1);
                y[j + l + 32] = d * static_cast<float>(sc[is + 2]) * static_cast<float>(q2);
                y[j + l + 64] = d * static_cast<float>(sc[is + 4]) * static_cast<float>(q3);
                y[j + l + 96] = d * static_cast<float>(sc[is + 6]) * static_cast<float>(q4);
            }
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

// ggml's quantize_row_q4_0_ref
void quantize_row_q4_0(const float * x, uint8_t * dst, int64_t n) {
    const int64_t nb = n / 32;
    for (int64_t b = 0; b < nb; ++b) {
        const float * xb = x + b * 32;
        float amax = 0.0f;
        float max = 0.0f;
        for (int j = 0; j < 32; ++j) {
            const float v = xb[j];
            if (amax < std::fabs(v)) {
                amax = std::fabs(v);
                max = v;
            }
        }
        const float d = max / -8.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        uint8_t * blk = dst + b * 18;
        store_u16(blk, float_to_half(d));
        for (int j = 0; j < 16; ++j) {
            const float x0 = xb[0 + j] * id;
            const float x1 = xb[16 + j] * id;
            const uint8_t xi0 = std::min<uint8_t>(15, static_cast<uint8_t>(static_cast<int8_t>(x0 + 8.5f)));
            const uint8_t xi1 = std::min<uint8_t>(15, static_cast<uint8_t>(static_cast<int8_t>(x1 + 8.5f)));
            blk[2 + j] = static_cast<uint8_t>(xi0 | (xi1 << 4));
        }
    }
}

// ggml's quantize_row_q8_0_ref
void quantize_row_q8_0(const float * x, uint8_t * dst, int64_t n) {
    const int64_t nb = n / 32;
    for (int64_t b = 0; b < nb; ++b) {
        const float * xb = x + b * 32;
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) {
            amax = std::max(amax, std::fabs(xb[j]));
        }
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        uint8_t * blk = dst + b * 34;
        store_u16(blk, float_to_half(d));
        for (int j = 0; j < 32; ++j) {
            blk[2 + j] = static_cast<uint8_t>(static_cast<int8_t>(std::lroundf(xb[j] * id)));
        }
    }
}

// Ported from ggml-quants.c of ggml / llama.cpp (MIT License, Copyright (c) 2023-2026 The ggml authors), see THIRD_PARTY_NOTICES.md

constexpr float kGroupMaxEps = 1e-15f;

int ggml_nearest_int(float fval) {
    float val = fval + 12582912.f;
    int i;
    std::memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

float make_qx_quants(int n, int nmax, const float * x, int8_t * L, int rmse_type, const float * qw) {
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        const float ax = std::fabs(x[i]);
        if (ax > amax) {
            amax = ax;
            max = x[i];
        }
    }
    if (amax < kGroupMaxEps) {
        for (int i = 0; i < n; ++i) {
            L[i] = 0;
        }
        return 0.f;
    }
    float iscale = -static_cast<float>(nmax) / max;
    if (rmse_type == 0) {
        for (int i = 0; i < n; ++i) {
            int l = ggml_nearest_int(iscale * x[i]);
            l = std::max(-nmax, std::min(nmax - 1, l));
            L[i] = static_cast<int8_t>(nmax + l);
        }
        return 1 / iscale;
    }
    bool return_early = false;
    if (rmse_type < 0) {
        rmse_type = -rmse_type;
        return_early = true;
    }
    float sumlx = 0;
    float suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = ggml_nearest_int(iscale * x[i]);
        l = std::max(-nmax, std::min(nmax - 1, l));
        L[i] = static_cast<int8_t>(l + nmax);
        const float w = qw != nullptr ? qw[i]
                                      : rmse_type == 1 ? x[i] * x[i]
                                                       : rmse_type == 2 ? 1.f : rmse_type == 3 ? std::fabs(x[i]) : std::sqrt(std::fabs(x[i]));
        sumlx += w * x[i] * l;
        suml2 += w * static_cast<float>(l) * static_cast<float>(l);
    }
    float scale = suml2 ? sumlx / suml2 : 0.0f;
    if (return_early) {
        return suml2 > 0 ? 0.5f * (scale + 1 / iscale) : 1 / iscale;
    }
    float best = scale * sumlx;
    for (int is = -9; is <= 9; ++is) {
        if (is == 0) {
            continue;
        }
        iscale = -(static_cast<float>(nmax) + 0.1f * static_cast<float>(is)) / max;
        sumlx = suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = ggml_nearest_int(iscale * x[i]);
            l = std::max(-nmax, std::min(nmax - 1, l));
            const float w = qw != nullptr ? qw[i]
                                          : rmse_type == 1 ? x[i] * x[i]
                                                           : rmse_type == 2 ? 1.f : rmse_type == 3 ? std::fabs(x[i]) : std::sqrt(std::fabs(x[i]));
            sumlx += w * x[i] * l;
            suml2 += w * static_cast<float>(l) * static_cast<float>(l);
        }
        if (suml2 > 0 && sumlx * sumlx > best * suml2) {
            for (int i = 0; i < n; ++i) {
                int l = ggml_nearest_int(iscale * x[i]);
                l = std::max(-nmax, std::min(nmax - 1, l));
                L[i] = static_cast<int8_t>(nmax + l);
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    return scale;
}

float make_qkx2_quants(int n, int nmax, const float * x, const float * weights, uint8_t * L, float * the_min, uint8_t * Laux,
                       float rmin, float rdelta, int nstep, bool use_mad) {
    float min = x[0];
    float max = x[0];
    float sum_w = weights[0];
    float sum_x = sum_w * x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] < min) {
            min = x[i];
        }
        if (x[i] > max) {
            max = x[i];
        }
        const float w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if (min > 0) {
        min = 0;
    }
    if (max == min) {
        for (int i = 0; i < n; ++i) {
            L[i] = 0;
        }
        *the_min = -min;
        return 0.f;
    }
    float iscale = static_cast<float>(nmax) / (max - min);
    float scale = 1 / iscale;
    float best_error = 0;
    for (int i = 0; i < n; ++i) {
        int l = ggml_nearest_int(iscale * (x[i] - min));
        l = std::max(0, std::min(nmax, l));
        L[i] = static_cast<uint8_t>(l);
        float diff = scale * static_cast<float>(l) + min - x[i];
        diff = use_mad ? std::fabs(diff) : diff * diff;
        best_error += weights[i] * diff;
    }
    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }
    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta * static_cast<float>(is) + static_cast<float>(nmax)) / (max - min);
        float sum_l = 0;
        float sum_l2 = 0;
        float sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            int l = ggml_nearest_int(iscale * (x[i] - min));
            l = std::max(0, std::min(nmax, l));
            Laux[i] = static_cast<uint8_t>(l);
            const float w = weights[i];
            sum_l += w * static_cast<float>(l);
            sum_l2 += w * static_cast<float>(l) * static_cast<float>(l);
            sum_xl += w * static_cast<float>(l) * x[i];
        }
        const float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
            float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            float cur_error = 0;
            for (int i = 0; i < n; ++i) {
                float diff = this_scale * static_cast<float>(Laux[i]) + this_min - x[i];
                diff = use_mad ? std::fabs(diff) : diff * diff;
                cur_error += weights[i] * diff;
            }
            if (cur_error < best_error) {
                for (int i = 0; i < n; ++i) {
                    L[i] = Laux[i];
                }
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}

void quantize_row_q4_k(const float * x, uint8_t * dst, int64_t n) {
    const int64_t nb = n / 256;
    uint8_t L[256];
    uint8_t Laux[32];
    float weights[32];
    float mins[8];
    float scales[8];

    for (int64_t i = 0; i < nb; ++i) {
        uint8_t * blk = dst + i * 144;
        float max_scale = 0;
        float max_min = 0;
        for (int j = 0; j < 8; ++j) {
            float sum_x2 = 0;
            for (int l = 0; l < 32; ++l) {
                sum_x2 += x[32 * j + l] * x[32 * j + l];
            }
            const float av_x = std::sqrt(sum_x2 / 32);
            for (int l = 0; l < 32; ++l) {
                weights[l] = av_x + std::fabs(x[32 * j + l]);
            }
            scales[j] = make_qkx2_quants(32, 15, x + 32 * j, weights, L + 32 * j, &mins[j], Laux, -1.f, 0.1f, 20, false);
            if (scales[j] > max_scale) {
                max_scale = scales[j];
            }
            if (mins[j] > max_min) {
                max_min = mins[j];
            }
        }

        const float inv_scale = max_scale > 0 ? 63.f / max_scale : 0.f;
        const float inv_min = max_min > 0 ? 63.f / max_min : 0.f;
        for (int j = 0; j < 8; ++j) {
            uint8_t ls = static_cast<uint8_t>(ggml_nearest_int(inv_scale * scales[j]));
            uint8_t lm = static_cast<uint8_t>(ggml_nearest_int(inv_min * mins[j]));
            ls = static_cast<uint8_t>(std::min(63, static_cast<int>(ls)));
            lm = static_cast<uint8_t>(std::min(63, static_cast<int>(lm)));
            if (j < 4) {
                blk[4 + j] = ls;
                blk[4 + j + 4] = lm;
            } else {
                blk[4 + j + 4] = static_cast<uint8_t>((ls & 0xF) | ((lm & 0xF) << 4));
                blk[4 + j - 4] = static_cast<uint8_t>(blk[4 + j - 4] | ((ls >> 4) << 6));
                blk[4 + j] = static_cast<uint8_t>(blk[4 + j] | ((lm >> 4) << 6));
            }
        }
        store_u16(blk, float_to_half(max_scale / 63.f));
        store_u16(blk + 2, float_to_half(max_min / 63.f));

        for (int j = 0; j < 8; ++j) {
            uint8_t sc = 0;
            uint8_t m = 0;
            get_scale_min_k4(j, blk + 4, &sc, &m);
            const float d = half_to_float(load_u16(blk)) * sc;
            if (d == 0.f) {
                continue;
            }
            const float dm = half_to_float(load_u16(blk + 2)) * m;
            for (int ii = 0; ii < 32; ++ii) {
                int l = ggml_nearest_int((x[32 * j + ii] + dm) / d);
                l = std::max(0, std::min(15, l));
                L[32 * j + ii] = static_cast<uint8_t>(l);
            }
        }

        uint8_t * q = blk + 16;
        for (int j = 0; j < 256; j += 64) {
            for (int l = 0; l < 32; ++l) {
                q[l] = static_cast<uint8_t>(L[j + l] | (L[j + l + 32] << 4));
            }
            q += 32;
        }

        x += 256;
    }
}

void quantize_row_q6_k(const float * x, uint8_t * dst, int64_t n) {
    const int64_t nb = n / 256;
    int8_t L[256];
    float scales[16];

    for (int64_t i = 0; i < nb; ++i) {
        uint8_t * blk = dst + i * 210;
        float max_scale = 0;
        float max_abs_scale = 0;

        for (int ib = 0; ib < 16; ++ib) {
            const float scale = make_qx_quants(16, 32, x + 16 * ib, L + 16 * ib, 1, nullptr);
            scales[ib] = scale;
            const float abs_scale = std::fabs(scale);
            if (abs_scale > max_abs_scale) {
                max_abs_scale = abs_scale;
                max_scale = scale;
            }
        }

        if (max_abs_scale < kGroupMaxEps) {
            std::memset(blk, 0, 210);
            store_u16(blk + 208, float_to_half(0.f));
            x += 256;
            continue;
        }

        const float iscale = -128.f / max_scale;
        store_u16(blk + 208, float_to_half(1 / iscale));
        int8_t * sc = reinterpret_cast<int8_t *>(blk + 192);
        for (int ib = 0; ib < 16; ++ib) {
            sc[ib] = static_cast<int8_t>(std::min(127, ggml_nearest_int(iscale * scales[ib])));
        }

        for (int j = 0; j < 16; ++j) {
            const float d = half_to_float(load_u16(blk + 208)) * sc[j];
            if (d == 0.f) {
                continue;
            }
            for (int ii = 0; ii < 16; ++ii) {
                int l = ggml_nearest_int(x[16 * j + ii] / d);
                l = std::max(-32, std::min(31, l));
                L[16 * j + ii] = static_cast<int8_t>(l + 32);
            }
        }

        uint8_t * ql = blk;
        uint8_t * qh = blk + 128;
        for (int j = 0; j < 256; j += 128) {
            for (int l = 0; l < 32; ++l) {
                const uint8_t q1 = static_cast<uint8_t>(L[j + l + 0] & 0xF);
                const uint8_t q2 = static_cast<uint8_t>(L[j + l + 32] & 0xF);
                const uint8_t q3 = static_cast<uint8_t>(L[j + l + 64] & 0xF);
                const uint8_t q4 = static_cast<uint8_t>(L[j + l + 96] & 0xF);
                ql[l + 0] = static_cast<uint8_t>(q1 | (q3 << 4));
                ql[l + 32] = static_cast<uint8_t>(q2 | (q4 << 4));
                qh[l] = static_cast<uint8_t>((L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) | ((L[j + l + 64] >> 4) << 4) |
                                             ((L[j + l + 96] >> 4) << 6));
            }
            ql += 64;
            qh += 32;
        }

        x += 256;
    }
}

} // namespace

scrapgpu_status dequantize_row(scrapgpu_dtype dtype, const uint8_t * row, int64_t n, float * dst) {
    if (row == nullptr || dst == nullptr || n <= 0) {
        return ref_invalid("dequantize needs a source, a destination and a positive element count");
    }
    const size_t block = scrapgpu_dtype_block_size(dtype);
    if (block == 0 || !dtype_is_quantized(dtype)) {
        return ref_unsupported("dequantize expects a quantized dtype");
    }
    if (n % static_cast<int64_t>(block) != 0) {
        return ref_invalid("dequantize element count must be a multiple of the block size");
    }
    switch (dtype) {
        case SCRAPGPU_DTYPE_Q4_0:
            dequantize_row_q4_0(row, n, dst);
            return SCRAPGPU_STATUS_OK;
        case SCRAPGPU_DTYPE_Q8_0:
            dequantize_row_q8_0(row, n, dst);
            return SCRAPGPU_STATUS_OK;
        case SCRAPGPU_DTYPE_Q4_K:
            dequantize_row_q4_k(row, n, dst);
            return SCRAPGPU_STATUS_OK;
        case SCRAPGPU_DTYPE_Q6_K:
            dequantize_row_q6_k(row, n, dst);
            return SCRAPGPU_STATUS_OK;
        default:
            return ref_unsupported("dequantize dtype is not implemented");
    }
}

scrapgpu_status quantize_row(scrapgpu_dtype dtype, const float * src, uint8_t * dst, int64_t n) {
    if (src == nullptr || dst == nullptr || n <= 0) {
        return ref_invalid("quantize needs a source, a destination and a positive element count");
    }
    const size_t block = scrapgpu_dtype_block_size(dtype);
    if (block == 0 || !dtype_is_quantized(dtype)) {
        return ref_unsupported("quantize expects a quantized dtype");
    }
    if (n % static_cast<int64_t>(block) != 0) {
        return ref_invalid("quantize element count must be a multiple of the block size");
    }
    switch (dtype) {
        case SCRAPGPU_DTYPE_Q4_0:
            quantize_row_q4_0(src, dst, n);
            return SCRAPGPU_STATUS_OK;
        case SCRAPGPU_DTYPE_Q8_0:
            quantize_row_q8_0(src, dst, n);
            return SCRAPGPU_STATUS_OK;
        case SCRAPGPU_DTYPE_Q4_K:
            quantize_row_q4_k(src, dst, n);
            return SCRAPGPU_STATUS_OK;
        case SCRAPGPU_DTYPE_Q6_K:
            quantize_row_q6_k(src, dst, n);
            return SCRAPGPU_STATUS_OK;
        default:
            return ref_unsupported("quantize dtype is not implemented");
    }
}

} // namespace ref
} // namespace scrapgpu

extern "C" {

scrapgpu_status scrapgpu_ref_dequantize(scrapgpu_dtype dtype, const void * src, float * dst, int64_t n) {
    return scrapgpu::ref::dequantize_row(dtype, static_cast<const uint8_t *>(src), n, dst);
}

scrapgpu_status scrapgpu_ref_quantize(scrapgpu_dtype dtype, const float * src, void * dst, int64_t n) {
    return scrapgpu::ref::quantize_row(dtype, src, static_cast<uint8_t *>(dst), n);
}

} // extern "C"

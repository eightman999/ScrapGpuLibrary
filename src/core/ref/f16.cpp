#include "internal.h"

#include <cstring>

namespace scrapgpu {
namespace ref {

float half_to_float(uint16_t h) {
    const uint32_t sign = (static_cast<uint32_t>(h) >> 15) & 1u;
    const uint32_t exp = (static_cast<uint32_t>(h) >> 10) & 0x1Fu;
    const uint32_t man = static_cast<uint32_t>(h) & 0x3FFu;
    uint32_t bits = 0;
    if (exp == 0) {
        if (man == 0) {
            bits = sign << 31;
        } else {
            // subnormal half -> normalized float
            uint32_t m = man;
            uint32_t e = 0;
            while ((m & 0x0400u) == 0u) {
                m <<= 1;
                ++e;
            }
            bits = (sign << 31) | ((113u - e) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = (sign << 31) | 0x7F800000u | (man << 13);
    } else {
        bits = (sign << 31) | ((exp - 15u + 127u) << 23) | (man << 13);
    }
    float f = 0.0f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

uint16_t float_to_half(float f) {
    uint32_t x = 0;
    std::memcpy(&x, &f, sizeof(x));
    const uint16_t sign = static_cast<uint16_t>((x >> 16) & 0x8000u);
    const uint32_t a = x & 0x7FFFFFFFu;
    const uint32_t exp = a >> 23;

    if (exp == 0xFFu) {
        // Inf / NaN
        return static_cast<uint16_t>(sign | 0x7C00u | ((a & 0x7FFFFFu) != 0u ? 0x0200u : 0u));
    }
    if (exp > 142u) {
        // |v| >= 2^16: rounds to Inf (after rounding the largest half 65504 + the
        // halfway 65520 case below, everything at or above 2^16 overflows)
        return static_cast<uint16_t>(sign | 0x7C00u);
    }
    if (exp >= 113u) {
        // normal half: round the mantissa to 10 bits, ties to even
        const uint32_t mant = a & 0x7FFFFFu;
        const uint32_t gr = mant & 0x1FFFu;
        uint32_t m = mant >> 13;
        if (gr > 0x1000u || (gr == 0x1000u && (m & 1u) != 0u)) {
            ++m;
        }
        uint32_t he = exp - 112u;
        if (m == 0x400u) {
            // carry into the exponent
            m = 0;
            ++he;
        }
        if (he >= 0x1Fu) {
            return static_cast<uint16_t>(sign | 0x7C00u);
        }
        return static_cast<uint16_t>(sign | (he << 10) | m);
    }
    if (exp < 103u) {
        // rounds to zero or the smallest subnormal half (2^-24); the halfway
        // 2^-25 rounds to zero (ties to even)
        const uint32_t one = (exp == 102u && (a & 0x7FFFFFu) != 0u) ? 1u : 0u;
        return static_cast<uint16_t>(sign | one);
    }
    // subnormal half: keep 1..10 bits of the significand
    const uint32_t sig = (a & 0x7FFFFFu) | 0x800000u;
    const uint32_t drop = 126u - exp;
    const uint32_t rem = sig & ((1u << drop) - 1u);
    const uint32_t half = 1u << (drop - 1u);
    uint32_t m = sig >> drop;
    if (rem > half || (rem == half && (m & 1u) != 0u)) {
        ++m;
    }
    if ((m & 0x400u) != 0u) {
        // rounded up to the smallest normal half
        return static_cast<uint16_t>(sign | 0x0400u);
    }
    return static_cast<uint16_t>(sign | m);
}

} // namespace ref
} // namespace scrapgpu

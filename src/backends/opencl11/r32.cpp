#include "r32.h"

#include "device_state.h"

#include <cstdio>
#include <vector>

#define CL11_RP_CHECK(expr)                                                                   \
    do {                                                                                      \
        const cl_int _err = (expr);                                                           \
        if (_err != CL_SUCCESS) {                                                             \
            fprintf(stderr, "%s: %s failed: %s (%d)\n", __func__, #expr, cl11_err_str(_err), (int) _err); \
            return false;                                                                     \
        }                                                                                     \
    } while (0)

// copy `size` bytes at byte `offset` of the (natural layout of the) tensor from / to the device. The conversion is done
// per chunk of whole groups of rows of a slice; a chunk that is only partly written is read, patched and written back.
bool cl11_rp_transfer(cl11_device * c, cl_mem mem, const sgt::TensorDesc & t, uint8_t * host, size_t offset, size_t size, bool to_dev) {
    const sgt::r32_geom G = sgt::r32_geometry(t);
    if (G.rowb == 0 || G.M <= 0) {
        return false; // a partial-block row has no R32 geometry (r32_eligible rejects it first)
    }
    const size_t slice_nat = (size_t) G.M * G.rowb;
    const size_t end = offset + size;
    const size_t base = (size_t) t.offset;
    const int64_t chunk_groups = std::max<int64_t>(1, (int64_t) ((size_t) 24 * 1024 * 1024 / (32 * G.rowb)));

    std::vector<uint8_t> abuf[3], nbuf;
    for (size_t s = offset / slice_nat; s <= (end - 1) / slice_nat; ++s) {
        const size_t slice_lo = s * slice_nat;
        const size_t lo = std::max(offset, slice_lo) - slice_lo;
        const size_t hi = std::min(end, slice_lo + slice_nat) - slice_lo;
        const int64_t g_first = (int64_t) (lo / G.rowb) / 32;
        const int64_t g_last  = (int64_t) ((hi - 1) / G.rowb) / 32;
        const size_t q_slice = base + slice_lo;

        for (int64_t gc = g_first; gc <= g_last; gc += chunk_groups) {
            const int64_t rA = gc * 32;
            const int64_t rB = std::min<int64_t>(G.M, std::min(g_last + 1, gc + chunk_groups) * 32);
            const int64_t nrows = rB - rA;
            const size_t nat_lo = (size_t) rA * G.rowb, nat_hi = (size_t) rB * G.rowb;
            const size_t ov_lo = std::max(lo, nat_lo), ov_hi = std::min(hi, nat_hi);
            const bool full = ov_lo == nat_lo && ov_hi == nat_hi;
            uint8_t * hp = full ? host + (slice_lo + nat_lo - offset) : nullptr; // the chunk lies inside of the host range

            uint8_t * ap[3] = { nullptr, nullptr, nullptr };
            size_t   an[3] = { 0, 0, 0 }, ao[3] = { 0, 0, 0 };
            for (int k = 0; k < G.na; ++k) {
                an[k] = (size_t) nrows * G.nb * G.area[k].len;
                ao[k] = q_slice + (size_t) G.M * G.nb * sgt::r32_area_before(G, k) + (size_t) rA * G.nb * G.area[k].len;
                abuf[k].resize(an[k]);
                ap[k] = abuf[k].data();
            }
            const uint8_t * const * cap = (const uint8_t * const *) ap;
            if (to_dev) {
                if (full) {
                    cl11_rp_pack(G, hp, nrows, ap);
                } else {
                    nbuf.resize((size_t) nrows * G.rowb);
                    for (int k = 0; k < G.na; ++k) {
                        CL11_RP_CHECK(clEnqueueReadBuffer(c->queue, mem, CL_TRUE, ao[k], an[k], ap[k], 0, nullptr, nullptr));
                    }
                    cl11_rp_unpack(G, cap, nrows, nbuf.data());
                    memcpy(nbuf.data() + (ov_lo - nat_lo), host + (slice_lo + ov_lo - offset), ov_hi - ov_lo);
                    cl11_rp_pack(G, nbuf.data(), nrows, ap);
                }
                for (int k = 0; k < G.na; ++k) {
                    CL11_RP_CHECK(clEnqueueWriteBuffer(c->queue, mem, CL_TRUE, ao[k], an[k], ap[k], 0, nullptr, nullptr));
                }
            } else {
                for (int k = 0; k < G.na; ++k) {
                    CL11_RP_CHECK(clEnqueueReadBuffer(c->queue, mem, CL_TRUE, ao[k], an[k], ap[k], 0, nullptr, nullptr));
                }
                if (full) {
                    cl11_rp_unpack(G, cap, nrows, hp);
                } else {
                    nbuf.resize((size_t) nrows * G.rowb);
                    cl11_rp_unpack(G, cap, nrows, nbuf.data());
                    memcpy(host + (slice_lo + ov_lo - offset), nbuf.data() + (ov_lo - nat_lo), ov_hi - ov_lo);
                }
            }
        }
    }
    return true;
}

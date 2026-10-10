// Host work used by CL11 tensor/broadcast launch descriptors. Build the same
// source against baseline and candidate headers; the checksum must agree.
#include "tensor/tensor.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

int main() {
    volatile uint32_t divisors[] = {1, 1, 1024, 3072, 128, 28, 64, 256, 14336, 32, 63, 4096};
    constexpr int iterations = 2000000;
    std::vector<double> samples;
    uint64_t checksum = 0;
    for (int repeat = 0; repeat < 7; ++repeat) {
        uint64_t sum = 0;
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i) {
            uint32_t m, l;
            sgt::fastdiv_init(divisors[i % 12], m, l);
            sum += static_cast<uint64_t>(m) + l;
        }
        const auto stop = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::nano>(stop - start).count() / iterations);
        if (repeat && sum != checksum) return 1;
        checksum = sum;
    }
    std::sort(samples.begin(), samples.end());
    std::printf("iterations,median_ns_per_descriptor,checksum\n%d,%.6f,%llu\n",
                iterations, samples[3], static_cast<unsigned long long>(checksum));
}

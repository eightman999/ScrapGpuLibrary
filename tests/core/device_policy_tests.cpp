#include "device.h"
#include <cstdio>
#include <cstdlib>
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"%d: %s\n",__LINE__,#x); std::exit(1); } } while (0)
int main() {
    cl11_device_info info;
    info.name = "AMD Radeon RX 6400 (radeonsi, navi24, LLVM 19.1.7)";
    info.platform_name = "Clover";
    CHECK(cl11_device_rejection_reason(info) != nullptr);
    info.platform_name = "rusticl";
    CHECK(cl11_device_rejection_reason(info) == nullptr);
    info.platform_name = "Clover"; info.name = "AMD Radeon Pro WX 2100 (polaris12)";
    CHECK(cl11_device_rejection_reason(info) == nullptr);
    info.platform_name = "NVIDIA CUDA"; info.name = "GeForce GT 430";
    CHECK(cl11_device_rejection_reason(info) == nullptr);
    info.platform_name = "Clover"; info.name = "unknown GPU";
    CHECK(cl11_device_rejection_reason(info) == nullptr);
    // the conflict check opens nothing without NVIDIA GPUs of two generations: the probe must not fail whatever the host has
    CHECK(cl11_probe_device_infos([](const cl11_device_info &, bool) {}));
    std::puts("device policy tests passed");
}

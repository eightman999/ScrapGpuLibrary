#include "scrapgpu/api.h"
#include "scrapgpu/status.h"
#include "scrapgpu/device.h"
#include "scrapgpu/buffer.h"
#include "scrapgpu/tensor.h"
#include "scrapgpu/op.h"
#include "scrapgpu/backend.h"
#include "scrapgpu/remote.h"
#include "scrapgpu/cpu.h"
#include "scrapgpu/attention.h"
#include "scrapgpu/compat.h"

int main() {
    return scrapgpu_get_abi_version() == 0u ? 1 : 0;
}

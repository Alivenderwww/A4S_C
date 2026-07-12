#include "aec_runtime.h"

#include <stdio.h>

int main(void) {
    int count = 0;
    aecDeviceInfoData info;
    if (aecDeviceCount(&count) != AEC_SUCCESS || count != 1 ||
        aecDeviceInfo(0, &info) != AEC_SUCCESS) {
        fprintf(stderr, "device query failed: %s\n",
                aecGetErrorName(aecPeekAtLastError()));
        return 1;
    }
    printf("%s: Runtime ABI %u, AEC ISA %u/profile %u, memory %llu bytes\n",
           info.name, info.abi_version, info.isa_version, info.isa_profile,
           (unsigned long long)info.memory_bytes);
    return 0;
}

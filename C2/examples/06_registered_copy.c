#include "aec_runtime.h"

#include <stdio.h>
#include <stdlib.h>

int main(void) {
    const size_t bytes = 65536;
    void *host = aligned_alloc(64, bytes);
    aecDevicePtr device = 0;
    if (host == NULL || aecAlloc(&device, bytes) != AEC_SUCCESS) return 2;
    if (aecHostRegister(host, bytes) != AEC_SUCCESS ||
        aecCopyH2D(device, host, bytes) != AEC_SUCCESS) return 1;
    aecRuntimeStats stats;
    if (aecGetRuntimeStats(&stats) != AEC_SUCCESS) return 1;
    printf("zero-copy commands: %llu\n",
           (unsigned long long)stats.zero_copy_commands);
    aecHostUnregister(host); aecFree(device); free(host);
    return 0;
}

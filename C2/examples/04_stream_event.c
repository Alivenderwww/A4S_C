#include "aec_runtime.h"

#include <stdio.h>

int main(void) {
    aecStream_t stream = NULL;
    aecEvent_t start = NULL, end = NULL;
    aecDevicePtr device = 0;
    unsigned char host[4096] = {0};
    if (aecAlloc(&device, sizeof(host)) != AEC_SUCCESS ||
        aecStreamCreate(&stream) != AEC_SUCCESS ||
        aecEventCreate(&start) != AEC_SUCCESS ||
        aecEventCreate(&end) != AEC_SUCCESS) return 2;
    aecEventRecord(start, stream);
    aecCopyAsync(device, host, sizeof(host), AEC_COPY_HOST_TO_DEVICE, stream);
    aecEventRecord(end, stream);
    if (aecEventSynchronize(end) != AEC_SUCCESS) return 1;
    uint64_t cycles = 0;
    if (aecEventElapsedCycles(start, end, &cycles) != AEC_SUCCESS) return 1;
    printf("copy interval: %llu virtual cycles\n", (unsigned long long)cycles);
    aecEventDestroy(start); aecEventDestroy(end);
    aecStreamDestroy(stream); aecFree(device);
    return 0;
}

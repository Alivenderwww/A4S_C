#include "aec_runtime.h"

#include <stdio.h>

int main(void) {
    const float a[5] = {1.0f, -2.0f, 3.5f, 4.0f, 0.25f};
    const float b[5] = {0.5f, 2.0f, -1.5f, 8.0f, -0.25f};
    float c[5] = {0};
    aecDevicePtr da = 0, db = 0, dc = 0;
    if (aecAlloc(&da, sizeof(a)) != AEC_SUCCESS ||
        aecAlloc(&db, sizeof(b)) != AEC_SUCCESS ||
        aecAlloc(&dc, sizeof(c)) != AEC_SUCCESS) {
        fprintf(stderr, "implement allocation first (%s)\n",
                aecGetErrorName(aecPeekAtLastError()));
        return 2;
    }
    aecCopyH2D(da, a, sizeof(a));
    aecCopyH2D(db, b, sizeof(b));
    const aecVectorAddArgs args = {da, db, dc, 5};
    aecError_t status = aecLaunch(
        AEC_KERNEL_VECTOR_ADD_F32, (aecDim3){1, 1, 1}, (aecDim3){32, 1, 1},
        &args, sizeof(args), NULL);
    if (status == AEC_SUCCESS) status = aecCopyD2H(c, dc, sizeof(c));
    for (int i = 0; status == AEC_SUCCESS && i < 5; ++i) {
        printf("c[%d] = %.3f\n", i, c[i]);
    }
    aecFree(da); aecFree(db); aecFree(dc);
    return status == AEC_SUCCESS ? 0 : 1;
}

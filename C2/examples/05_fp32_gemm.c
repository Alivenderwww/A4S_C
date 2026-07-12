#include "aec_runtime.h"

#include <stdio.h>

int main(void) {
    const float a[6] = {1, -2, 0.5f, 3, 4, -1};
    const float b[9] = {2, 1, -1, 0.5f, 3, 2, -2, 1.5f, 0.25f};
    float c[6] = {0};
    aecDevicePtr da = 0, db = 0, dc = 0;
    if (aecAlloc(&da, sizeof(a)) || aecAlloc(&db, sizeof(b)) ||
        aecAlloc(&dc, sizeof(c))) return 2;
    if (aecCopyH2D(da, a, sizeof(a)) || aecCopyH2D(db, b, sizeof(b)) ||
        aecMatmulF32(da, db, dc, 2, 3, 3, NULL) ||
        aecCopyD2H(c, dc, sizeof(c))) return 1;
    for (int i = 0; i < 6; ++i) printf("%.6f%c", c[i], i == 5 ? '\n' : ' ');
    aecFree(da); aecFree(db); aecFree(dc);
    return 0;
}

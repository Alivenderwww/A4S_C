#ifndef AEC_RUNTIME_H
#define AEC_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AEC_RUNTIME_ABI_VERSION 2u

typedef uint64_t aecDevicePtr;
typedef struct aecStreamOpaque *aecStream_t;
typedef struct aecEventOpaque *aecEvent_t;

typedef enum aecError {
    AEC_SUCCESS = 0,
    AEC_ERROR_INVALID_ARGUMENT = 1,
    AEC_ERROR_OUT_OF_MEMORY = 2,
    AEC_ERROR_INVALID_HANDLE = 3,
    AEC_ERROR_INVALID_ADDRESS = 4,
    AEC_ERROR_NOT_READY = 5,
    AEC_ERROR_NOT_SUPPORTED = 6,
    AEC_ERROR_DEVICE = 7,
    AEC_ERROR_INTERNAL = 8,
    AEC_ERROR_ISA_TRAP = 9
} aecError_t;

typedef enum aecCopyDirection {
    AEC_COPY_HOST_TO_DEVICE = 1,
    AEC_COPY_DEVICE_TO_HOST = 2
} aecCopyDirection;

typedef enum aecDataType {
    AEC_DTYPE_FP4_E2M1 = 1,
    AEC_DTYPE_FP8_E4M3 = 2,
    AEC_DTYPE_FP8_E5M2 = 3,
    AEC_DTYPE_FP16 = 4,
    AEC_DTYPE_BF16 = 5,
    AEC_DTYPE_FP32 = 6,
    AEC_DTYPE_FP64 = 7,
    AEC_DTYPE_INT4 = 8,
    AEC_DTYPE_INT8 = 9,
    AEC_DTYPE_INT32 = 10
} aecDataType;

typedef enum aecFp8Format {
    AEC_FP8_E4M3 = 1,
    AEC_FP8_E5M2 = 2
} aecFp8Format;

typedef enum aecKernelId {
    AEC_KERNEL_VECTOR_ADD_F32 = 1,
    AEC_KERNEL_GEMM_NAIVE = 10,
    AEC_KERNEL_GEMM_TILED = 11,
    AEC_KERNEL_GEMM_VECTORIZED = 12,
    AEC_KERNEL_AXPY_F32 = 20,
    AEC_KERNEL_DOT_F32 = 21,
    AEC_KERNEL_NRM2_F32 = 22
} aecKernelId;

typedef struct aecDim3 {
    uint32_t x;
    uint32_t y;
    uint32_t z;
} aecDim3;

typedef struct aecDeviceInfoData {
    uint32_t abi_version;
    char name[64];
    uint64_t memory_bytes;
    uint32_t dma_channels;
    uint32_t max_threads_per_block;
    uint32_t isa_version;
    uint32_t isa_profile;
    uint32_t max_parameter_bytes;
    uint32_t reserved;
} aecDeviceInfoData;

typedef struct aecRuntimeStats {
    uint32_t abi_version;
    uint32_t reserved;
    uint64_t submitted_commands;
    uint64_t dma_commands;
    uint64_t kernel_commands;
    uint64_t zero_copy_commands;
    uint64_t channel_commands[2];
    uint64_t total_virtual_cycles;
    uint64_t last_virtual_cycles;
    uint64_t isa_launches;
    uint64_t instructions_retired;
    uint64_t isa_traps;
    uint64_t last_kernel_handle;
    uint64_t last_trace_digest;
} aecRuntimeStats;

typedef struct aecVectorAddArgs {
    aecDevicePtr a;
    aecDevicePtr b;
    aecDevicePtr c;
    uint64_t count;
} aecVectorAddArgs;

typedef struct aecGemmArgs {
    aecDevicePtr a;
    aecDevicePtr b;
    aecDevicePtr c;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t dtype;
    uint32_t reserved;
} aecGemmArgs;

typedef struct aecAxpyArgs {
    aecDevicePtr x;
    aecDevicePtr y;
    uint64_t count;
    float alpha;
    uint32_t reserved;
} aecAxpyArgs;

typedef struct aecDotArgs {
    aecDevicePtr x;
    aecDevicePtr y;
    aecDevicePtr result;
    uint64_t count;
} aecDotArgs;

typedef struct aecNrm2Args {
    aecDevicePtr x;
    aecDevicePtr result;
    uint64_t count;
} aecNrm2Args;

aecError_t aecDeviceCount(int *count);
aecError_t aecDeviceInfo(int device, aecDeviceInfoData *info);
aecError_t aecGetLastError(void);
aecError_t aecPeekAtLastError(void);
const char *aecGetErrorName(aecError_t error);

aecError_t aecAlloc(aecDevicePtr *out_ptr, size_t bytes);
aecError_t aecFree(aecDevicePtr ptr);
aecError_t aecCopyH2D(aecDevicePtr dst, const void *src, size_t bytes);
aecError_t aecCopyD2H(void *dst, aecDevicePtr src, size_t bytes);
aecError_t aecCopyAsync(aecDevicePtr device_ptr, void *host_ptr, size_t bytes,
                        aecCopyDirection direction, aecStream_t stream);

aecError_t aecStreamCreate(aecStream_t *stream);
aecError_t aecStreamDestroy(aecStream_t stream);
aecError_t aecStreamSync(aecStream_t stream);
aecError_t aecEventCreate(aecEvent_t *event);
aecError_t aecEventDestroy(aecEvent_t event);
aecError_t aecEventRecord(aecEvent_t event, aecStream_t stream);
aecError_t aecEventSynchronize(aecEvent_t event);
aecError_t aecEventQuery(aecEvent_t event);
aecError_t aecEventElapsedCycles(aecEvent_t start, aecEvent_t end,
                                 uint64_t *cycles);

aecError_t aecHostRegister(void *ptr, size_t bytes);
aecError_t aecHostUnregister(void *ptr);
aecError_t aecGetRuntimeStats(aecRuntimeStats *stats);
aecError_t aecResetRuntimeStats(void);

aecError_t aecLaunch(aecKernelId kernel, aecDim3 grid, aecDim3 block,
                     const void *args, size_t args_size, aecStream_t stream);

aecError_t aecMatmulF4(aecDevicePtr a, aecDevicePtr b, aecDevicePtr c,
                       uint32_t m, uint32_t n, uint32_t k,
                       aecStream_t stream);
aecError_t aecMatmulF8(aecDevicePtr a, aecDevicePtr b, aecDevicePtr c,
                       uint32_t m, uint32_t n, uint32_t k,
                       aecFp8Format format, aecStream_t stream);
aecError_t aecMatmulF16(aecDevicePtr a, aecDevicePtr b, aecDevicePtr c,
                        uint32_t m, uint32_t n, uint32_t k,
                        aecStream_t stream);
aecError_t aecMatmulBF16(aecDevicePtr a, aecDevicePtr b, aecDevicePtr c,
                         uint32_t m, uint32_t n, uint32_t k,
                         aecStream_t stream);
aecError_t aecMatmulF32(aecDevicePtr a, aecDevicePtr b, aecDevicePtr c,
                        uint32_t m, uint32_t n, uint32_t k,
                        aecStream_t stream);
aecError_t aecMatmulF64(aecDevicePtr a, aecDevicePtr b, aecDevicePtr c,
                        uint32_t m, uint32_t n, uint32_t k,
                        aecStream_t stream);
aecError_t aecMatmulI4(aecDevicePtr a, aecDevicePtr b, aecDevicePtr c,
                       uint32_t m, uint32_t n, uint32_t k,
                       aecStream_t stream);
aecError_t aecMatmulI8(aecDevicePtr a, aecDevicePtr b, aecDevicePtr c,
                       uint32_t m, uint32_t n, uint32_t k,
                       aecStream_t stream);
aecError_t aecMatmulI32(aecDevicePtr a, aecDevicePtr b, aecDevicePtr c,
                        uint32_t m, uint32_t n, uint32_t k,
                        aecStream_t stream);

aecError_t aecAxpy(aecDevicePtr x, aecDevicePtr y, uint64_t count, float alpha,
                   aecStream_t stream);
aecError_t aecDot(aecDevicePtr x, aecDevicePtr y, aecDevicePtr result,
                  uint64_t count, aecStream_t stream);
aecError_t aecNrm2(aecDevicePtr x, aecDevicePtr result, uint64_t count,
                   aecStream_t stream);

#ifdef __cplusplus
}
#endif

#endif

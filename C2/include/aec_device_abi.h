#ifndef AEC_DEVICE_ABI_H
#define AEC_DEVICE_ABI_H

#include <stddef.h>
#include <stdint.h>

#include "aec_isa.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__) || defined(__clang__)
#define AEC_DEVICE_API __attribute__((visibility("default")))
#else
#define AEC_DEVICE_API
#endif

#define AEC_DEVICE_ABI_VERSION 2u
#define AEC_DEVICE_MEMORY_BYTES (64ull * 1024ull * 1024ull)
#define AEC_DEVICE_MAX_PARAM_BYTES 64u

typedef uint64_t aecDevicePtr;
typedef uint64_t aecDeviceKernelHandle;

typedef enum aecDeviceStatus {
    AEC_DEVICE_SUCCESS = 0,
    AEC_DEVICE_INVALID_ARGUMENT = 1,
    AEC_DEVICE_OUT_OF_MEMORY = 2,
    AEC_DEVICE_INVALID_ADDRESS = 3,
    AEC_DEVICE_UNSUPPORTED = 4,
    AEC_DEVICE_INJECTED_FAULT = 5,
    AEC_DEVICE_ISA_TRAP = 6,
    AEC_DEVICE_INTERNAL = 7
} aecDeviceStatus;

typedef enum aecDeviceOpcode {
    AEC_DEVICE_OP_H2D = 1,
    AEC_DEVICE_OP_D2H = 2,
    AEC_DEVICE_OP_ISA_LAUNCH = 3,
    AEC_DEVICE_OP_BARRIER = 4
} aecDeviceOpcode;

typedef enum aecDeviceCommandFlags {
    AEC_DEVICE_FLAG_NONE = 0,
    AEC_DEVICE_FLAG_REGISTERED = 1u << 0,
    AEC_DEVICE_FLAG_ZERO_COPY = 1u << 1
} aecDeviceCommandFlags;

typedef enum aecDeviceFault {
    AEC_DEVICE_FAULT_NONE = 0,
    AEC_DEVICE_FAULT_NEXT_DMA = 1,
    AEC_DEVICE_FAULT_NEXT_KERNEL = 2,
    AEC_DEVICE_FAULT_NEXT_COMMAND = 3
} aecDeviceFault;

typedef struct aecDeviceDim3 {
    uint32_t x;
    uint32_t y;
    uint32_t z;
} aecDeviceDim3;

typedef struct aecDeviceKernelInfo {
    uint32_t abi_version;
    uint32_t isa_version;
    aecDeviceKernelHandle handle;
    uint32_t image_id;
    uint32_t entry_pc;
    uint32_t parameter_bytes;
    uint32_t image_flags;
    uint64_t instruction_hash;
} aecDeviceKernelInfo;

typedef struct aecDeviceCommand {
    uint32_t abi_version;
    uint16_t opcode;
    uint16_t flags;
    uint64_t sequence;
    uint64_t stream_id;

    aecDevicePtr src;
    aecDevicePtr dst;
    uint64_t host_address;
    uint64_t bytes;
    uint32_t chunk_bytes;
    uint16_t queue_depth;
    uint8_t channel;
    uint8_t reserved0;

    aecDeviceKernelHandle kernel_handle;
    uint32_t isa_version;
    uint32_t entry_pc;
    aecDeviceDim3 grid;
    aecDeviceDim3 block;
    uint32_t parameter_bytes;
    uint32_t dynamic_shared_bytes;
    uint8_t parameters[AEC_DEVICE_MAX_PARAM_BYTES];
} aecDeviceCommand;

typedef struct aecDeviceCompletion {
    uint32_t abi_version;
    uint32_t status;
    uint64_t sequence;
    uint64_t virtual_cycles;
    uint64_t bytes_completed;
    uint64_t instructions_retired;
    uint64_t trace_digest;
    uint32_t fault_code;
    uint32_t trap_pc;
} aecDeviceCompletion;

typedef struct aecDeviceCaps {
    uint32_t abi_version;
    uint32_t device_count;
    uint64_t memory_bytes;
    uint32_t dma_channels;
    uint32_t max_queue_depth;
    uint32_t address_alignment;
    uint32_t isa_version;
    uint32_t isa_profile;
    uint32_t max_parameter_bytes;
    uint32_t max_threads_per_block;
} aecDeviceCaps;

typedef struct aecDeviceStats {
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
} aecDeviceStats;

AEC_DEVICE_API aecDeviceStatus aecDeviceGetCaps(aecDeviceCaps *caps);
AEC_DEVICE_API aecDeviceStatus aecDeviceReset(void);
AEC_DEVICE_API aecDeviceStatus aecDeviceAlloc(uint64_t bytes, uint64_t alignment,
                                              aecDevicePtr *out_ptr);
AEC_DEVICE_API aecDeviceStatus aecDeviceFree(aecDevicePtr ptr);
AEC_DEVICE_API aecDeviceStatus aecDeviceResolveKernel(
    uint32_t semantic_kernel_id, uint32_t dtype, uint32_t variant,
    aecDeviceKernelInfo *info);
/* Author-owned, read-only policy oracle. It interprets the selected image with
 * synthetic non-overlapping spans and does not submit a command or mutate stats. */
AEC_DEVICE_API aecDeviceStatus aecDeviceEvaluateKernel(
    uint32_t semantic_kernel_id, uint32_t dtype, uint32_t variant,
    uint32_t m, uint32_t n, uint32_t k, uint32_t alignment,
    uint64_t workspace_bytes, aecDeviceCompletion *completion);
AEC_DEVICE_API aecDeviceStatus aecDeviceSubmit(
    const aecDeviceCommand *command, aecDeviceCompletion *completion);
AEC_DEVICE_API aecDeviceStatus aecDeviceGetStats(aecDeviceStats *stats);
AEC_DEVICE_API aecDeviceStatus aecDeviceResetStats(void);
AEC_DEVICE_API aecDeviceStatus aecDeviceInjectFault(aecDeviceFault fault);

#ifdef __cplusplus
}
#endif

#endif

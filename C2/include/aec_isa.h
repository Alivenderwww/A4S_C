#ifndef AEC_ISA_H
#define AEC_ISA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Version 2 uses the canonical Track-B/B3 encoding from simple-gpgpu main. */
#define AEC_ISA_VERSION 2u
#define AEC_ISA_PROFILE_C2 1u
#define AEC_ISA_INSTRUCTION_BYTES 16u
#define AEC_ISA_REGISTER_COUNT 256u
#define AEC_ISA_PREDICATE_COUNT 8u
#define AEC_ISA_PREDICATE_NONE 15u
#define AEC_ISA_WARP_SIZE 32u
#define AEC_ISA_WARPS_PER_SM 8u
#define AEC_ISA_WARPS_PER_GROUP 4u
#define AEC_ISA_IMAGE_MAGIC 0x49434541u /* "AECI" as little-endian bytes. */
#define AEC_ISA_IMAGE_VERSION 2u

#define AEC_KERNEL_PARAM_VECTOR_ADD_BYTES 32u
#define AEC_KERNEL_PARAM_GEMM_BYTES 40u
#define AEC_KERNEL_PARAM_AXPY_BYTES 28u
#define AEC_KERNEL_PARAM_DOT_BYTES 32u
#define AEC_KERNEL_PARAM_NRM2_BYTES 24u

/* Canonical B3.1 opcode allocation.  C2 must not renumber these values. */
typedef enum aecIsaOpcode {
    AEC_ISA_OP_ADD = 0x0001,
    AEC_ISA_OP_SUB = 0x0002,
    AEC_ISA_OP_MUL = 0x0003,
    AEC_ISA_OP_MAD = 0x0004,
    AEC_ISA_OP_FMA = 0x0005,
    AEC_ISA_OP_DIV = 0x0006,
    AEC_ISA_OP_NEG = 0x0007,
    AEC_ISA_OP_ABS = 0x0008,
    AEC_ISA_OP_MIN = 0x0009,
    AEC_ISA_OP_MAX = 0x000a,

    AEC_ISA_OP_AND = 0x0010,
    AEC_ISA_OP_OR = 0x0011,
    AEC_ISA_OP_XOR = 0x0012,
    AEC_ISA_OP_NOT = 0x0013,
    AEC_ISA_OP_SHL = 0x0014,
    AEC_ISA_OP_SHR = 0x0015,
    AEC_ISA_OP_BFX = 0x0016,
    AEC_ISA_OP_BINS = 0x0017,
    AEC_ISA_OP_POPC = 0x0018,
    AEC_ISA_OP_FLO = 0x0019,

    AEC_ISA_OP_CMP = 0x0020,
    AEC_ISA_OP_CMPP = 0x0021,
    AEC_ISA_OP_SEL = 0x0022,
    AEC_ISA_OP_PICK = 0x0023,

    AEC_ISA_OP_LD = 0x0030,
    AEC_ISA_OP_ST = 0x0031,
    AEC_ISA_OP_LDC = 0x0032,
    AEC_ISA_OP_ATOM = 0x0033,

    AEC_ISA_OP_BR = 0x0040,
    AEC_ISA_OP_BRX = 0x0041,
    AEC_ISA_OP_JMP = 0x0042,
    AEC_ISA_OP_CALL = 0x0043,
    AEC_ISA_OP_RET = 0x0044,
    AEC_ISA_OP_HALT = 0x0045,
    AEC_ISA_OP_SSYNC = 0x0046,
    AEC_ISA_OP_SYNC_CT = 0x0047,
    AEC_ISA_OP_SYNC_WG = 0x0048,
    AEC_ISA_OP_MBAR = 0x0049,

    AEC_ISA_OP_LOADI = 0x0050,
    AEC_ISA_OP_CPY = 0x0051,
    AEC_ISA_OP_LOADI64 = 0x0052,
    AEC_ISA_OP_CVTFF = 0x0053,
    AEC_ISA_OP_CVTFI = 0x0054,
    AEC_ISA_OP_CVTIF = 0x0055,
    AEC_ISA_OP_CVTII = 0x0056,
    AEC_ISA_OP_SHUF = 0x0057,
    AEC_ISA_OP_VOTE = 0x0058,
    AEC_ISA_OP_MTCH = 0x0059,

    AEC_ISA_OP_TMUL = 0x0060,
    AEC_ISA_OP_TMUL_S = 0x0061,
    AEC_ISA_OP_TLDA = 0x0062,
    AEC_ISA_OP_TSTA = 0x0063,
    AEC_ISA_OP_TMOV = 0x0064,
    AEC_ISA_OP_TDUP = 0x0065,

    AEC_ISA_OP_RCP = 0x0070,
    AEC_ISA_OP_RSQ = 0x0071,
    AEC_ISA_OP_SIN = 0x0072,
    AEC_ISA_OP_COS = 0x0073,
    AEC_ISA_OP_EXP = 0x0074,
    AEC_ISA_OP_LOG = 0x0075,
    AEC_ISA_OP_SQRT = 0x0076,

    AEC_ISA_OP_RDTSC = 0x0080,
    AEC_ISA_OP_RDPMC = 0x0081
} aecIsaOpcode;

/* Canonical B Pred/Ctrl[6:3] type selectors. */
typedef enum aecIsaType {
    AEC_ISA_TYPE_FP32 = 0,
    AEC_ISA_TYPE_FP64 = 1,
    AEC_ISA_TYPE_FP16 = 2,
    AEC_ISA_TYPE_BF16 = 3,
    AEC_ISA_TYPE_FP8_E4M3 = 4,
    AEC_ISA_TYPE_FP8_E5M2 = 5,
    AEC_ISA_TYPE_FP4_E2M1 = 6,
    AEC_ISA_TYPE_S32 = 7,
    AEC_ISA_TYPE_U32 = 8,
    AEC_ISA_TYPE_S8 = 9,
    AEC_ISA_TYPE_U8 = 10,
    AEC_ISA_TYPE_S4 = 11,
    AEC_ISA_TYPE_U4 = 12,
    AEC_ISA_TYPE_B32 = 13,
    AEC_ISA_TYPE_B64 = 14,
    AEC_ISA_TYPE_NONE = 15,
    AEC_ISA_TYPE_INT4 = AEC_ISA_TYPE_S4,
    AEC_ISA_TYPE_INT8 = AEC_ISA_TYPE_S8,
    AEC_ISA_TYPE_INT32 = AEC_ISA_TYPE_S32,
    AEC_ISA_TYPE_U64 = AEC_ISA_TYPE_B64
} aecIsaType;

typedef enum aecIsaControl {
    AEC_ISA_CTRL_NONE = 0
} aecIsaControl;

typedef enum aecIsaMemorySpace {
    AEC_ISA_SPACE_GMEM = 0,
    AEC_ISA_SPACE_SMEM = 1,
    AEC_ISA_SPACE_CMEM = 2,
    AEC_ISA_SPACE_LMEM = 3,
    /* B defines pmem semantics but has not frozen its selector.  C2 reserves 4. */
    AEC_ISA_SPACE_PMEM = 4
} aecIsaMemorySpace;

typedef enum aecIsaCompareOperation {
    AEC_ISA_COMPARE_EQ = 0,
    AEC_ISA_COMPARE_NE = 1,
    AEC_ISA_COMPARE_LT = 2,
    AEC_ISA_COMPARE_LE = 3,
    AEC_ISA_COMPARE_GT = 4,
    AEC_ISA_COMPARE_GE = 5
} aecIsaCompareOperation;

typedef enum aecIsaSpecialRegister {
    AEC_ISA_SPECIAL_TID_X = 0x0100,
    AEC_ISA_SPECIAL_NTID_X = 0x0101,
    AEC_ISA_SPECIAL_CTAID_X = 0x0102,
    AEC_ISA_SPECIAL_NCTAID_X = 0x0103,
    AEC_ISA_SPECIAL_LANEID = 0x0104,
    AEC_ISA_SPECIAL_WARPID = 0x0105,
    AEC_ISA_SPECIAL_TID_Y = 0x0110,
    AEC_ISA_SPECIAL_NTID_Y = 0x0111,
    AEC_ISA_SPECIAL_CTAID_Y = 0x0112,
    AEC_ISA_SPECIAL_NCTAID_Y = 0x0113,
    AEC_ISA_SPECIAL_TID_Z = 0x0120,
    AEC_ISA_SPECIAL_NTID_Z = 0x0121,
    AEC_ISA_SPECIAL_CTAID_Z = 0x0122,
    AEC_ISA_SPECIAL_NCTAID_Z = 0x0123
} aecIsaSpecialRegister;

typedef enum aecKernelVariant {
    AEC_KERNEL_VARIANT_DEFAULT = 0,
    AEC_KERNEL_VARIANT_NAIVE = 1,
    AEC_KERNEL_VARIANT_TILED = 2,
    AEC_KERNEL_VARIANT_VECTORIZED = 3
} aecKernelVariant;

typedef enum aecIsaImageFlags {
    AEC_ISA_IMAGE_FLAG_NONE = 0,
    AEC_ISA_IMAGE_FLAG_SPMD = 1u << 0,
    AEC_ISA_IMAGE_FLAG_SINGLE_INVOCATION = 1u << 1,
    AEC_ISA_IMAGE_FLAG_BUILTIN = 1u << 2
} aecIsaImageFlags;

#define AEC_ISA_PRED_ENABLE 0x8000u
#define AEC_ISA_TYPE_SHIFT 3u
#define AEC_ISA_FAMILY_SHIFT 8u
#define AEC_ISA_SPACE_SHIFT 11u
#define AEC_ISA_TYPE_MASK 0x000fu
#define AEC_ISA_FAMILY_MASK 0x0007u
#define AEC_ISA_SPACE_MASK 0x0007u

/*
 * Original Track-B bit positions, serialized as four little-endian u32 words:
 *   bits 127:112 opcode
 *   bits 111:96  Pred/Ctrl (pred-enable[15], family/extended[14:8],
 *                           type[6:3], predicate[2:0])
 *   bits 95:80   destination register (high eight bits reserved)
 *   bits 79:64   source 1 register (high eight bits reserved)
 *   bits 63:32   source2 register or instruction-specific field
 *   bits 31:0    imm32 or source3 register
 *
 * These assignments match simple-gpgpu main's B3 CModel contract.
 */
typedef struct aecIsaInstruction {
    uint32_t word0;
    uint32_t word1;
    uint32_t word2;
    uint32_t word3;
} aecIsaInstruction;

typedef struct aecIsaImageHeader {
    uint32_t magic;
    uint32_t isa_version;
    uint32_t image_version;
    uint32_t header_bytes;
    uint32_t image_id;
    uint32_t semantic_kernel_id;
    uint32_t dtype;
    uint32_t variant;
    uint32_t instruction_count;
    uint32_t entry_pc;
    uint32_t parameter_bytes;
    uint32_t flags;
    uint64_t instruction_hash;
    uint64_t reserved;
} aecIsaImageHeader;

static inline int aecIsaUsesImmediate(uint16_t opcode, uint32_t memory_space) {
    return opcode == AEC_ISA_OP_LOADI || opcode == AEC_ISA_OP_LOADI64 ||
           opcode == AEC_ISA_OP_BR || opcode == AEC_ISA_OP_BRX ||
           opcode == AEC_ISA_OP_CALL || opcode == AEC_ISA_OP_SSYNC ||
           opcode == AEC_ISA_OP_RDPMC ||
           (opcode == AEC_ISA_OP_LD && memory_space == AEC_ISA_SPACE_PMEM);
}

static inline uint8_t aecIsaTypeForDtype(uint32_t dtype) {
    switch (dtype) {
    case 1: return AEC_ISA_TYPE_FP4_E2M1;
    case 2: return AEC_ISA_TYPE_FP8_E4M3;
    case 3: return AEC_ISA_TYPE_FP8_E5M2;
    case 4: return AEC_ISA_TYPE_FP16;
    case 5: return AEC_ISA_TYPE_BF16;
    case 6: return AEC_ISA_TYPE_FP32;
    case 7: return AEC_ISA_TYPE_FP64;
    case 8: return AEC_ISA_TYPE_S4;
    case 9: return AEC_ISA_TYPE_S8;
    case 10: return AEC_ISA_TYPE_S32;
    default: return AEC_ISA_TYPE_NONE;
    }
}

static inline uint8_t aecIsaTensorModeForType(uint8_t type) {
    switch (type) {
    case AEC_ISA_TYPE_FP32: return 0;
    case AEC_ISA_TYPE_FP16: return 1;
    case AEC_ISA_TYPE_BF16: return 2;
    case AEC_ISA_TYPE_S8: return 3;
    case AEC_ISA_TYPE_S4: return 4;
    case AEC_ISA_TYPE_FP8_E4M3: return 5;
    case AEC_ISA_TYPE_FP4_E2M1: return 6;
    case AEC_ISA_TYPE_FP64:
    case AEC_ISA_TYPE_S32:
    case AEC_ISA_TYPE_FP8_E5M2:
        return 7;
    default: return 0xffu;
    }
}

static inline uint8_t aecIsaTensorExtendedModeForType(uint8_t type) {
    if (type == AEC_ISA_TYPE_FP64) return 0;
    if (type == AEC_ISA_TYPE_S32) return 1;
    /* C2 extension while B reserves selector 2. */
    if (type == AEC_ISA_TYPE_FP8_E5M2) return 2;
    return 0;
}

static inline aecIsaInstruction aecIsaEncode(
    uint16_t opcode, uint8_t type, uint8_t predicate, uint8_t control,
    uint16_t dst, uint16_t src1, uint16_t src2, uint16_t src3,
    uint32_t immediate, uint32_t modifier) {
    aecIsaInstruction instruction;
    uint16_t pred_ctrl = 0;
    if (type != AEC_ISA_TYPE_NONE) {
        pred_ctrl |= (uint16_t)((type & AEC_ISA_TYPE_MASK) << AEC_ISA_TYPE_SHIFT);
    }
    (void)control;
    if (opcode == AEC_ISA_OP_BRX) {
        /* B BRX always names its predicate in bits [2:0]. */
        pred_ctrl |= (uint16_t)(predicate & 0x7u);
    } else if (predicate != AEC_ISA_PREDICATE_NONE) {
        pred_ctrl |= AEC_ISA_PRED_ENABLE | (uint16_t)(predicate & 0x7u);
    }
    if (opcode == AEC_ISA_OP_CMP || opcode == AEC_ISA_OP_CMPP) {
        pred_ctrl |= (uint16_t)((modifier & AEC_ISA_FAMILY_MASK) <<
                               AEC_ISA_FAMILY_SHIFT);
    } else if (opcode == AEC_ISA_OP_LD || opcode == AEC_ISA_OP_ST) {
        pred_ctrl |= (uint16_t)((modifier & AEC_ISA_SPACE_MASK) <<
                               AEC_ISA_SPACE_SHIFT);
    } else if (opcode == AEC_ISA_OP_TMUL || opcode == AEC_ISA_OP_TMUL_S) {
        const uint8_t mode = aecIsaTensorModeForType(type);
        pred_ctrl |= (uint16_t)((mode & AEC_ISA_FAMILY_MASK) <<
                               AEC_ISA_FAMILY_SHIFT);
        if (mode == 7) {
            pred_ctrl |= (uint16_t)(aecIsaTensorExtendedModeForType(type) << 11);
        }
    } else if (opcode == AEC_ISA_OP_TLDA || opcode == AEC_ISA_OP_TSTA) {
        pred_ctrl |= (uint16_t)((modifier & AEC_ISA_FAMILY_MASK) <<
                               AEC_ISA_FAMILY_SHIFT);
    } else if (opcode == AEC_ISA_OP_MBAR) {
        pred_ctrl |= (uint16_t)((modifier & 0x3u) << AEC_ISA_FAMILY_SHIFT);
    }
    instruction.word0 = aecIsaUsesImmediate(opcode, modifier)
                            ? immediate
                            : (uint32_t)src3;
    instruction.word1 = (uint32_t)src2;
    instruction.word2 = ((uint32_t)dst << 16) | (uint32_t)src1;
    instruction.word3 = ((uint32_t)opcode << 16) | pred_ctrl;
    return instruction;
}

static inline uint16_t aecIsaGetOpcode(const aecIsaInstruction *instruction) {
    return (uint16_t)(instruction->word3 >> 16);
}

static inline uint8_t aecIsaGetType(const aecIsaInstruction *instruction) {
    return (uint8_t)((instruction->word3 >> AEC_ISA_TYPE_SHIFT) &
                     AEC_ISA_TYPE_MASK);
}

static inline uint8_t aecIsaGetPredicate(const aecIsaInstruction *instruction) {
    if (aecIsaGetOpcode(instruction) == AEC_ISA_OP_BRX) {
        return (uint8_t)(instruction->word3 & 0x7u);
    }
    if ((instruction->word3 & AEC_ISA_PRED_ENABLE) == 0) {
        return AEC_ISA_PREDICATE_NONE;
    }
    return (uint8_t)(instruction->word3 & 0x7u);
}

static inline uint8_t aecIsaGetControl(const aecIsaInstruction *instruction) {
    (void)instruction;
    return AEC_ISA_CTRL_NONE;
}

static inline uint16_t aecIsaGetDst(const aecIsaInstruction *instruction) {
    return (uint16_t)(instruction->word2 >> 16);
}

static inline uint16_t aecIsaGetSrc1(const aecIsaInstruction *instruction) {
    return (uint16_t)(instruction->word2 & 0xffffu);
}

static inline uint16_t aecIsaGetSrc2(const aecIsaInstruction *instruction) {
    return (uint16_t)(instruction->word1 & 0xffffu);
}

static inline uint16_t aecIsaGetSrc3(const aecIsaInstruction *instruction) {
    return (uint16_t)(instruction->word0 & 0xffffu);
}

static inline uint32_t aecIsaGetImmediate(const aecIsaInstruction *instruction) {
    return instruction->word0;
}

static inline uint8_t aecIsaGetMemorySpace(const aecIsaInstruction *instruction) {
    return (uint8_t)((instruction->word3 >> AEC_ISA_SPACE_SHIFT) &
                     AEC_ISA_SPACE_MASK);
}

static inline uint8_t aecIsaGetFamilyOperation(const aecIsaInstruction *instruction) {
    return (uint8_t)((instruction->word3 >> AEC_ISA_FAMILY_SHIFT) &
                     AEC_ISA_FAMILY_MASK);
}

static inline uint8_t aecIsaGetTensorExtendedMode(
    const aecIsaInstruction *instruction) {
    return (uint8_t)((instruction->word3 >> 11) & 0x0fu);
}

#if defined(__cplusplus)
static_assert(sizeof(aecIsaInstruction) == AEC_ISA_INSTRUCTION_BYTES,
              "AEC instructions must be exactly 128 bits");
static_assert(sizeof(aecIsaImageHeader) == 64,
              "AEC image headers must be exactly 64 bytes");
#else
_Static_assert(sizeof(aecIsaInstruction) == AEC_ISA_INSTRUCTION_BYTES,
               "AEC instructions must be exactly 128 bits");
_Static_assert(sizeof(aecIsaImageHeader) == 64,
               "AEC image headers must be exactly 64 bytes");
#endif

#ifdef __cplusplus
}
#endif

#endif

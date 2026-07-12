#include "aec_isa.h"

#include <inttypes.h>
#include <stdio.h>

int main(void) {
    const aecIsaInstruction add = aecIsaEncode(
        AEC_ISA_OP_ADD, AEC_ISA_TYPE_FP32, 3,
        AEC_ISA_CTRL_NONE, 1, 2, 3, 4, 0, 0);
    printf("little-endian words: %08" PRIx32 " %08" PRIx32
           " %08" PRIx32 " %08" PRIx32 "\n",
           add.word0, add.word1, add.word2, add.word3);
    if (add.word0 != 0x00000004u || add.word1 != 0x00000003u ||
        add.word2 != 0x00010002u || add.word3 != 0x00018003u) {
        return 1;
    }
    return 0;
}

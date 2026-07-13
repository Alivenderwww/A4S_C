"""aec_decode.py — .aecbin parser + 128-bit AEC instruction decoder.

Inverse of C1/src/isa/encoder.cpp. Opcode and type numbering follow the AEC
Precise ISA in Track-B spec.md appendix A (§A.1 opcodes, §4 types).

Word layout (little-endian words w0..w3):
    word3 = (opcode<<16) | pred_ctrl
    word2 = (dst<<16)    | src1
    word1 = src2
    word0 = imm32   (if the op carries an immediate)  else  src3
pred_ctrl:
    [2:0]   predicate index (BRX branch pred, or guard pred when [15]=1)
    [6:3]   type selector
    [10:8]  family subop  (CMP/CMPP compare op, TMUL mode, TLDA/TSTA layout, MBAR scope)
    [13:11] memory space   (LD/ST)
    [14]    pred_neg
    [15]    pred_en

Run `py -3.13 aec_decode.py --selftest` to verify the decoder against the 8
golden vectors baked into encoder.cpp.
"""
import struct
import sys

# --- opcode numbering (Track-B §A.1, mirror of C1/include/aec/isa.h) -------
OPCODES = {
    0x0001: "ADD", 0x0002: "SUB", 0x0003: "MUL", 0x0004: "MAD", 0x0005: "FMA",
    0x0006: "DIV", 0x0007: "NEG", 0x0008: "ABS", 0x0009: "MIN", 0x000a: "MAX",
    0x0010: "AND", 0x0011: "OR", 0x0012: "XOR", 0x0013: "NOT", 0x0014: "SHL",
    0x0015: "SHR", 0x0016: "BFX", 0x0017: "BINS", 0x0018: "POPC", 0x0019: "FLO",
    0x0020: "CMP", 0x0021: "CMPP", 0x0022: "SEL", 0x0023: "PICK",
    0x0030: "LD", 0x0031: "ST", 0x0032: "LDC", 0x0033: "ATOM",
    0x0040: "BR", 0x0041: "BRX", 0x0042: "JMP", 0x0043: "CALL", 0x0044: "RET",
    0x0045: "HALT", 0x0046: "SSYNC", 0x0047: "SYNC_CT", 0x0048: "SYNC_WG",
    0x0049: "MBAR",
    0x0050: "CVTFF", 0x0051: "CVTFI", 0x0052: "CVTIF", 0x0053: "CVTII",
    0x0054: "CPY", 0x0055: "LOADI", 0x0056: "LOADI64", 0x0057: "SHUF",
    0x0058: "VOTE", 0x0059: "MTCH",
    0x0060: "TMUL", 0x0061: "TMUL_S", 0x0062: "TLDA", 0x0063: "TSTA",
    0x0064: "TMOV", 0x0065: "TDUP",
    0x0070: "RCP", 0x0071: "RSQ", 0x0072: "SIN", 0x0073: "COS", 0x0074: "EXP",
    0x0075: "LOG", 0x0076: "SQRT", 0x0080: "RDTSC", 0x0081: "RDPMC",
}
# type selector -> name (Track-B §4)
TYPES = {0x0: "b32", 0x1: "b64", 0x2: "u32", 0x3: "s32", 0x4: "u8", 0x5: "s8",
         0x8: "f32", 0x9: "f64", 0xa: "f16", 0xb: "bf16", 0xf: "none"}
SPACES = {0: "gmem", 1: "smem", 2: "cmem", 3: "lmem", 4: "pmem"}
CMPS = {0: "eq", 1: "ne", 2: "lt", 3: "le", 4: "gt", 5: "ge"}
SPECIALS = {0x0100: "tid.x", 0x0101: "ntid.x", 0x0102: "ctaid.x",
            0x0103: "nctaid.x", 0x0104: "laneid", 0x0105: "warpid",
            0x0110: "tid.y", 0x0111: "ntid.y", 0x0112: "ctaid.y",
            0x0113: "nctaid.y", 0x0120: "tid.z", 0x0121: "ntid.z",
            0x0122: "ctaid.z", 0x0123: "nctaid.z"}

_IMM_OPS = {"LOADI", "LOADI64", "BR", "BRX", "CALL", "SSYNC", "RDPMC"}


def uses_immediate(op, space):
    return op in _IMM_OPS or (op == "LD" and space == 4)  # LD.pmem


class Instr:
    __slots__ = ("op", "type", "type_code", "src_type", "src_type_code",
                 "pred_en", "pred", "pred_neg", "subop", "space", "dst",
                 "src1", "src2", "src3", "imm", "special", "words")

    def __init__(self, words):
        w0, w1, w2, w3 = words
        self.words = words
        opcode = (w3 >> 16) & 0xffff
        pc = w3 & 0xffff
        self.op = OPCODES.get(opcode, "OP_%04x" % opcode)
        self.type_code = (pc >> 3) & 0xf
        self.type = TYPES.get(self.type_code, "?")
        # CVT* carries the SOURCE type in bits [13:10] (dest type in [6:3]).
        self.src_type_code = (pc >> 10) & 0xf
        self.src_type = TYPES.get(self.src_type_code, "?")
        self.pred_en = (pc >> 15) & 1
        self.pred_neg = (pc >> 14) & 1
        self.pred = pc & 0x7
        self.subop = (pc >> 8) & 0x7
        self.space = (pc >> 11) & 0x7
        self.dst = (w2 >> 16) & 0xffff
        self.src1 = w2 & 0xffff
        self.src2 = w1 & 0xffff
        if uses_immediate(self.op, self.space):
            self.imm = w0
            self.src3 = 0
        else:
            self.imm = None
            self.src3 = w0 & 0xffff
        self.special = SPECIALS.get(self.src1) if self.src1 >= 0x100 else None

    def rd(self):
        return self.dst & 0xff

    def rs1(self):
        return self.src1 & 0xff

    def rs2(self):
        return self.src2 & 0xff

    def rs3(self):
        return self.src3 & 0xff

    def __str__(self):
        t = "" if self.type == "none" else "." + self.type
        g = ""
        if self.op == "BRX":
            g = " P%d" % self.pred
        elif self.pred_en:
            g = " @%sP%d" % ("!" if self.pred_neg else "", self.pred)
        extra = ""
        if self.op in ("CMP", "CMPP"):
            t = "." + CMPS.get(self.subop, "?") + t
        if self.op in ("LD", "ST"):
            t = "." + SPACES.get(self.space, "?") + t
        if self.imm is not None:
            extra = " imm=0x%x" % self.imm
        sp = " %%%s" % self.special if self.special else ""
        return "%s%s%s d=%d s1=%d s2=%d s3=%d%s%s" % (
            self.op, t, g, self.rd(), self.rs1(), self.rs2(), self.rs3(), sp, extra)


# --- .aecbin container -----------------------------------------------------
class Image:
    def __init__(self):
        self.entry_pc = 0
        self.instr_count = 0
        self.param_bytes = 0
        self.code = []       # list[Instr]
        self.data = b""
        self.relocs = []     # list[(instrIndex, kind, addend)]
        self.symbols = []    # list[(name, value, kind)]


def load_aecbin(path):
    with open(path, "rb") as f:
        b = f.read()
    if len(b) < 32:
        raise ValueError("file smaller than header")
    magic, version, hdr_bytes, sec_count, entry, icount, pbytes, flags = \
        struct.unpack_from("<8I", b, 0)
    if magic != 0x31434541:
        raise ValueError("bad magic 0x%08x (not .aecbin)" % magic)
    img = Image()
    img.entry_pc, img.instr_count, img.param_bytes = entry, icount, pbytes
    table = hdr_bytes or 32
    for s in range(sec_count):
        typ, off, size, ent = struct.unpack_from("<4I", b, table + s * 16)
        if typ == 1:  # CODE
            for i in range(size // 16):
                words = struct.unpack_from("<4I", b, off + i * 16)
                img.code.append(Instr(words))
        elif typ == 2:  # DATA
            img.data = b[off:off + size]
        elif typ == 3 and size >= 4:  # RELOC
            (n,) = struct.unpack_from("<I", b, off)
            for i in range(n):
                ii, kind, add, _ = struct.unpack_from("<4I", b, off + 4 + i * 16)
                img.relocs.append((ii, kind, add))
        elif typ == 4 and size >= 4:  # SYMBOL
            (n,) = struct.unpack_from("<I", b, off)
            p = off + 4
            for i in range(n):
                (nl,) = struct.unpack_from("<I", b, p); p += 4
                name = b[p:p + nl].decode("latin1"); p += nl
                val, kind = struct.unpack_from("<2I", b, p); p += 8
                img.symbols.append((name, val, kind))
    return img


# --- decoder self-test against encoder.cpp's 8 golden vectors --------------
def _selftest():
    cases = [
        ("LOADI.none R10,0x100", (256, 0, 655360, 5570680),
         dict(op="LOADI", type="none", dst=10, imm=0x100)),
        ("CPY.u32 R1,%laneid", (0, 0, 65796, 5505040),
         dict(op="CPY", type="u32", dst=1, special="laneid")),
        ("MUL.u32 R3,R1,R2", (0, 2, 196609, 196624),
         dict(op="MUL", type="u32", dst=3, src1=1, src2=2)),
        ("ADD.u32 R10,R10,R3", (0, 3, 655370, 65552),
         dict(op="ADD", type="u32", dst=10, src1=10, src2=3)),
        ("CVTIF.f32.u32 R4,R1", (0, 0, 262145, 5376064),
         dict(op="CVTIF", type="f32", src_type="u32", dst=4, src1=1)),
        ("CVTFF.f16.f32 R8,R4", (0, 0, 524292, 5251152),
         dict(op="CVTFF", type="f16", src_type="f32", dst=8, src1=4)),
        ("ST.gmem.u32 [R10],R8", (0, 8, 10, 3211280),
         dict(op="ST", type="u32", src1=10, src2=8, space=0)),
        ("HALT", (0, 0, 0, 4522104),
         dict(op="HALT", type="none")),
    ]
    ok = True
    for name, words, exp in cases:
        ins = Instr(words)
        for k, v in exp.items():
            got = getattr(ins, "rd")() if k == "dst" else \
                  getattr(ins, "rs1")() if k == "src1" else \
                  getattr(ins, "rs2")() if k == "src2" else \
                  getattr(ins, "rs3")() if k == "src3" else \
                  getattr(ins, k)
            if got != v:
                ok = False
                print("  MISMATCH %s: %s expected %r got %r" % (name, k, v, got))
    print("[decode selftest] %s" % ("all 8 golden vectors decode correctly"
                                    if ok else "FAILED"))
    return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if _selftest() else 1)
    for path in sys.argv[1:]:
        img = load_aecbin(path)
        print("entry=%d instr=%d param_bytes=%d" %
              (img.entry_pc, img.instr_count, img.param_bytes))
        for i, ins in enumerate(img.code):
            print("%4d: %s" % (i, ins))

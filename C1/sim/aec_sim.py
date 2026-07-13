"""aec_sim.py — AEC functional simulator (warp-lockstep, numpy lane-vectorized).

A local correctness ORACLE for C1, since the official golden/cycle model is not
released (it is a Track-B deliverable). Executes a compiled `.aecbin` over a
grid of threads and returns the resulting GMEM so a harness can compare against
an independent numpy reference.

Faithful to the AEC model where it matters for these kernels:
  * 32-lane warps, lockstep PC, per-launch active mask (partial last warp).
  * 256 u32 GPR + 8 predicates per lane; GPR/pred init 0.
  * `BRX` requires a UNIFORM condition across active lanes — a divergent branch
    (e.g. an un-predicated `if (tid>=n) return;` on a partial block) raises an
    ExecError, matching Track-B §A.2 (non-uniform BRX is an execution error).
    This is what catches the "must if-convert the bounds guard" bug
    (see C1_实现流程分析.md §1.3).

Deliberate simplifications (documented so results are not over-trusted):
  * No cross-thread interaction is modelled beyond shared GMEM (these public
    kernels use no smem/atomics/barriers). smem is per-CTA zeroed; lmem per-lane.
  * FP is IEEE via numpy; FMA is single-rounded (float64 intermediate), MAD is
    two-rounded — so a wrong `mad.f32→MAD` mapping shows up as a mismatch.
  * Integer ADD/SUB/MUL wrap at 32 bits for any integer/bit type. LD/ST widths
    follow the Track-B §4.1 legal-type matrix (an illegal 64-bit store is an
    ExecError); `strict=True` additionally flags questionable arithmetic types.

This oracle validates that the compiler emits the INTENDED computation and legal
control flow; it is not a bit-exact stand-in for the hidden golden model.
"""
import numpy as np

from aec_decode import load_aecbin

W = 32  # warp size

# Per-instruction result latency (cycles) for the scoreboard cycle model.
# GMEM/LMEM loads use the 32-cycle external memory service (Track-B §7); on-chip
# is fast; TMUL/SFU are multi-cycle. This is OUR proxy model (the official cycle
# model is unreleased), used to measure scheduling/latency-hiding — not golden.
_LAT = {"TMUL": 16, "TMUL_S": 16, "TLDA": 6, "TSTA": 6, "DIV": 12,
        "RCP": 12, "RSQ": 12, "SQRT": 12, "SIN": 12, "COS": 12, "EXP": 12,
        "LOG": 12, "BR": 2, "BRX": 2, "JMP": 2}
_NOWRITE = frozenset(("ST", "BR", "BRX", "JMP", "RET", "HALT", "SYNC_CT",
                      "SYNC_WG", "SSYNC", "MBAR", "ATOM"))


class ExecError(Exception):
    pass


def _u32(a):
    return a.astype(np.uint32, copy=False)


class Sim:
    def __init__(self, image, gmem, param_block=b"", smem_bytes=0, strict=False,
                 max_cycles=2_000_000):
        self.code = image.code
        self.entry = image.entry_pc
        self.gmem = gmem                      # np.uint8 array (shared)
        self.pmem = np.frombuffer(param_block, np.uint8) if param_block else np.zeros(0, np.uint8)
        self.smem_bytes = smem_bytes
        self.strict = strict
        self.max_cycles = max_cycles
        self.total_cycles = 0
        self.stall_cycles = 0    # cycles a warp waits on operand (memory) latency
        self.warps = 0

    # ---- memory helpers (per-lane gather/scatter) ------------------------
    def _load(self, mem, addr, width, mask):
        out = np.zeros(W, np.uint32)
        lanes = np.nonzero(mask)[0]
        if lanes.size == 0:
            return out
        base = addr[lanes].astype(np.int64)
        if (base.max() + width) > mem.size or base.min() < 0:
            raise ExecError("OOB load: addr max=%d width=%d space size=%d"
                            % (int(base.max()), width, mem.size))
        for k in range(width):
            out[lanes] |= mem[base + k].astype(np.uint32) << (8 * k)
        return out

    def _store(self, mem, addr, width, data, mask):
        lanes = np.nonzero(mask)[0]
        if lanes.size == 0:
            return
        base = addr[lanes].astype(np.int64)
        if (base.max() + width) > mem.size or base.min() < 0:
            raise ExecError("OOB store: addr max=%d width=%d space size=%d"
                            % (int(base.max()), width, mem.size))
        for k in range(width):
            mem[base + k] = ((data[lanes] >> (8 * k)) & 0xff).astype(np.uint8)

    # ---- launch ----------------------------------------------------------
    def run(self, grid, block):
        gx, gy, gz = (list(grid) + [1, 1])[:3]
        bx, by, bz = (list(block) + [1, 1])[:3]
        block_threads = bx * by * bz
        if not (1 <= block_threads <= 256):
            raise ExecError("blockDim product %d not in 1..256" % block_threads)
        nwarps = (block_threads + W - 1) // W
        for cz in range(gz):
            for cy in range(gy):
                for cx in range(gx):
                    smem = np.zeros(self.smem_bytes, np.uint8)
                    for wid in range(nwarps):
                        self._run_warp(wid, (cx, cy, cz), (bx, by, bz),
                                       (gx, gy, gz), block_threads, smem)
        return self.gmem

    def _run_warp(self, wid, ctaid, ntid, nctaid, block_threads, smem):
        self.warps += 1
        bx, by, bz = ntid
        # linear thread id per lane, then decompose to (x,y,z)
        t = wid * W + np.arange(W)
        active = t < block_threads
        tx = (t % bx).astype(np.uint32)
        ty = ((t // bx) % by).astype(np.uint32)
        tz = (t // (bx * by)).astype(np.uint32)
        laneid = np.arange(W, dtype=np.uint32)
        special = {
            "tid.x": tx, "tid.y": ty, "tid.z": tz,
            "ntid.x": np.uint32(bx), "ntid.y": np.uint32(by), "ntid.z": np.uint32(bz),
            "ctaid.x": np.uint32(ctaid[0]), "ctaid.y": np.uint32(ctaid[1]), "ctaid.z": np.uint32(ctaid[2]),
            "nctaid.x": np.uint32(nctaid[0]), "nctaid.y": np.uint32(nctaid[1]), "nctaid.z": np.uint32(nctaid[2]),
            "laneid": laneid, "warpid": np.uint32(wid),
        }
        R = np.zeros((W, 256), np.uint32)
        P = np.zeros((W, 8), bool)
        lmem = np.zeros(4096 * W, np.uint8)  # per-thread; lane l at [l*4096:...]
        pc = self.entry
        n = len(self.code)
        steps = 0
        ready = [0] * 256           # scoreboard: cycle each register becomes ready.
        pready = [0] * 8
        clock = 0
        issued = 0                  # instructions already issued in the current cycle.
        while 0 <= pc < n:
            ins = self.code[pc]
            steps += 1
            if steps > self.max_cycles:
                raise ExecError("timeout in warp (possible infinite loop)")
            op = ins.op

            # --- scoreboard cycle model (latency + dual-issue, per warp) ---
            lat = (32 if ins.space in (0, 3) else 2) if op == "LD" else _LAT.get(op, 1)
            t = 0
            if op not in ("LOADI", "LOADI64"):
                for r in (ins.rs1(), ins.rs2(), ins.rs3()):
                    if ready[r] > t:
                        t = ready[r]
            if ins.pred_en or op == "BRX":
                if pready[ins.pred] > t:
                    t = pready[ins.pred]
            if t > clock or issued >= 2:            # stall for operand or 2-wide issue
                if t > clock + 1:                   # operand-wait (memory latency) stall
                    self.stall_cycles += t - (clock + 1)
                clock = t if t > clock + 1 else clock + 1
                issued = 0
            issued += 1
            if op == "CMPP":
                pready[ins.dst & 7] = clock + lat
            elif op not in _NOWRITE:
                ready[ins.rd()] = clock + lat

            # guard mask
            if ins.pred_en:
                pv = P[:, ins.pred]
                if ins.pred_neg:
                    pv = ~pv
                em = active & pv
            else:
                em = active

            # ---- control flow ----
            if op == "RET" or op == "HALT":
                break
            if op == "BR" or op == "JMP":
                pc = ins.imm if ins.imm is not None else pc + 1
                continue
            if op == "BRX":
                cond = P[:, ins.pred][active]
                if cond.size and cond.any() and not cond.all():
                    raise ExecError(
                        "non-uniform BRX at pc=%d (divergent branch — the bounds "
                        "guard must be if-converted to predication, see §1.3)" % pc)
                pc = (ins.imm if ins.imm is not None else pc + 1) if (cond.size and cond.all()) else pc + 1
                continue

            self._exec(ins, R, P, em, special, lmem, smem)
            pc += 1
        self.total_cycles += int(clock)

    # ---- data ops --------------------------------------------------------
    def _exec(self, ins, R, P, em, special, lmem, smem):
        op, ty = ins.op, ins.type
        d, s1, s2, s3 = ins.rd(), ins.rs1(), ins.rs2(), ins.rs3()

        def wr_u32(reg, val):
            R[:, reg] = np.where(em, _u32(val), R[:, reg])

        def wr_f32(reg, val):
            R[:, reg] = np.where(em, val.astype(np.float32).view(np.uint32), R[:, reg])

        def a_u32(r): return R[:, r]
        def a_s32(r): return R[:, r].view(np.int32)
        def a_f32(r): return R[:, r].view(np.float32)
        def a_f16(r): return (R[:, r] & 0xffff).astype(np.uint16).view(np.float16)

        def a_f64(r):   # 64-bit double lives in the register pair {R[r+1], R[r]}.
            return (R[:, r].astype(np.uint64) |
                    (R[:, (r + 1) & 0xff].astype(np.uint64) << 32)).view(np.float64)

        def wr_f64(reg, val):
            bits = val.astype(np.float64).view(np.uint64)
            R[:, reg] = np.where(em, (bits & 0xffffffff).astype(np.uint32), R[:, reg])
            hi = (reg + 1) & 0xff
            R[:, hi] = np.where(em, (bits >> np.uint64(32)).astype(np.uint32), R[:, hi])

        is_f = ty in ("f32", "f16", "bf16", "f64")

        if op == "LOADI":
            wr_u32(d, np.full(W, ins.imm, np.uint32)); return
        if op == "CPY":
            if ins.special is not None:
                val = special.get(ins.special, np.zeros(W, np.uint32))
                wr_u32(d, np.broadcast_to(val, (W,))); return
            wr_u32(d, R[:, s1])  # register copy (any 32-bit type)
            return
        if op in ("ADD", "SUB", "MUL", "MAD", "MIN", "MAX", "NEG", "ABS"):
            if ty == "f64":                       # 64-bit double (register pair).
                x = a_f64(s1)
                y = a_f64(s2) if op not in ("NEG", "ABS") else None
                if op == "ADD": r = x + y
                elif op == "SUB": r = x - y
                elif op == "MUL": r = x * y
                elif op == "MIN": r = np.minimum(x, y)
                elif op == "MAX": r = np.maximum(x, y)
                elif op == "NEG": r = -x
                elif op == "ABS": r = np.abs(x)
                elif op == "MAD": r = x * y + a_f64(s3)
                wr_f64(d, r); return
            if is_f:
                x = a_f32(s1) if ty in ("f32",) else a_f32(s1)  # f16/bf16 promoted below
                if ty == "f16":
                    x = a_f16(s1).astype(np.float32)
                    y = a_f16(s2).astype(np.float32) if op not in ("NEG", "ABS") else None
                else:
                    y = a_f32(s2) if op not in ("NEG", "ABS") else None
                if op == "ADD": r = x + y
                elif op == "SUB": r = x - y
                elif op == "MUL": r = x * y
                elif op == "MIN": r = np.minimum(x, y)
                elif op == "MAX": r = np.maximum(x, y)
                elif op == "NEG": r = -x
                elif op == "ABS": r = np.abs(x)
                elif op == "MAD":  # round(round(a*b)+c)
                    c = a_f16(s3).astype(np.float32) if ty == "f16" else a_f32(s3)
                    r = (x.astype(np.float32) * y.astype(np.float32)).astype(np.float32) + c
                if ty == "f16":
                    R[:, d] = np.where(em, r.astype(np.float16).view(np.uint16).astype(np.uint32), R[:, d])
                else:
                    wr_f32(d, r)
                return
            # integer / bit types: 32-bit wraparound
            if self.strict and ty in ("b32", "b64", "none"):
                raise ExecError("%s.%s is not a legal ISA type" % (op, ty))
            signed = ty == "s32"
            x = a_s32(s1) if signed else a_u32(s1)
            y = a_s32(s2) if signed else a_u32(s2)
            if op == "ADD": r = a_u32(s1).astype(np.int64) + a_u32(s2).astype(np.int64)
            elif op == "SUB": r = a_u32(s1).astype(np.int64) - a_u32(s2).astype(np.int64)
            elif op == "MUL": r = a_u32(s1).astype(np.int64) * a_u32(s2).astype(np.int64)
            elif op == "MAD": r = a_u32(s1).astype(np.int64) * a_u32(s2).astype(np.int64) + a_u32(s3).astype(np.int64)
            elif op == "MIN": r = np.minimum(x, y)
            elif op == "MAX": r = np.maximum(x, y)
            elif op == "NEG": r = -a_s32(s1).astype(np.int64)
            elif op == "ABS": r = np.abs(a_s32(s1).astype(np.int64))
            wr_u32(d, (r & 0xffffffff).astype(np.uint32)); return
        if op == "FMA":  # single-rounded a*b+c via float64
            x, y, c = a_f32(s1).astype(np.float64), a_f32(s2).astype(np.float64), a_f32(s3).astype(np.float64)
            wr_f32(d, (x * y + c)); return
        if op in ("AND", "OR", "XOR", "SHL", "SHR", "NOT"):
            a = a_u32(s1)
            if op == "AND": r = a & a_u32(s2)
            elif op == "OR": r = a | a_u32(s2)
            elif op == "XOR": r = a ^ a_u32(s2)
            elif op == "NOT": r = ~a
            elif op == "SHL": r = a << (a_u32(s2) & 31)
            elif op == "SHR":
                r = (a_s32(s1) >> (a_u32(s2) & 31)).view(np.uint32) if ty == "s32" \
                    else (a >> (a_u32(s2) & 31))
            wr_u32(d, _u32(r)); return
        if op in ("CMP", "CMPP"):
            from aec_decode import CMPS
            cop = CMPS.get(ins.subop, "eq")
            if ty == "f32":
                x, y = a_f32(s1), a_f32(s2)
            elif ty == "s32":
                x, y = a_s32(s1), a_s32(s2)
            else:
                x, y = a_u32(s1), a_u32(s2)
            r = {"eq": x == y, "ne": x != y, "lt": x < y, "le": x <= y,
                 "gt": x > y, "ge": x >= y}[cop]
            if op == "CMPP":
                P[:, ins.dst & 0x7] = np.where(em, r, P[:, ins.dst & 0x7]); return
            wr_u32(d, r.astype(np.uint32)); return
        if op == "SEL":  # Rd = Pn ? Rs1 : Rs2
            pv = P[:, ins.pred]
            wr_u32(d, np.where(pv, R[:, s1], R[:, s2])); return
        if op in ("CVTFF", "CVTIF", "CVTFI"):
            # FAITHFUL: honor BOTH the encoded dst type ([6:3]) and src type
            # ([13:10]). If the compiler forgets to encode the source type it
            # reads back as f32/u32 (code 0/8) and the conversion is wrong here
            # too — which is correct oracle behavior (it flags the bug rather
            # than silently guessing f16). See C1_实现流程分析.md §1.4/§5.
            st = ins.src_type

            def read_float(reg, tn):
                if tn == "f16": return a_f16(reg).astype(np.float32)
                if tn == "bf16":
                    return ((R[:, reg] & 0xffff) << 16).view(np.float32)
                return a_f32(reg)  # f32 (f64 pair unsupported here)

            if op == "CVTFF":                        # float -> float
                src = read_float(s1, st)
                if ty == "f16":
                    R[:, d] = np.where(em, src.astype(np.float16).view(np.uint16).astype(np.uint32), R[:, d])
                elif ty == "bf16":
                    R[:, d] = np.where(em, (src.view(np.uint32) >> 16) & 0xffff, R[:, d])
                else:
                    wr_f32(d, src)
                return
            if op == "CVTIF":                        # int -> float
                src = a_s32(s1) if st in ("s32", "s8") else a_u32(s1)
                wr_f32(d, src.astype(np.float32)); return
            if op == "CVTFI":                        # float -> int (trunc+clamp)
                v = np.trunc(read_float(s1, st)).astype(np.int64)
                wr_u32(d, (v & 0xffffffff).astype(np.uint32)); return
        if op == "LD":
            if ty not in ("b32", "b64", "u32", "s32", "f32"):
                raise ExecError("illegal LD type '.%s' (Track-B §4.1)" % ty)
            width = 8 if ty == "b64" else 4
            addr = R[:, s1]          # byte address/offset in a register (all spaces)
            mem = {0: self.gmem, 1: smem, 3: lmem, 4: self.pmem}.get(ins.space, self.gmem)
            lo = self._load(mem, addr, 4, em)
            wr_u32(d, lo)
            if width == 8:
                hi = self._load(mem, (addr.astype(np.int64) + 4).astype(np.uint32), 4, em)
                R[:, (d + 1) & 0xff] = np.where(em, hi, R[:, (d + 1) & 0xff])
            return
        if op == "ST":
            if ty not in ("b32", "u32", "s32", "f32"):
                raise ExecError("illegal ST type '.%s' (Track-B §4.1: ST is 32-bit)" % ty)
            addr = R[:, s1]
            mem = {0: self.gmem, 1: smem, 3: lmem}.get(ins.space, self.gmem)
            self._store(mem, addr, 4, R[:, s2], em)
            return
        # anything else: flag (helps find missing lowering / illegal ops)
        raise ExecError("unimplemented/illegal op '%s.%s' at runtime" % (op, ty))


def simulate(aecbin_path, grid, block, param_block=b"", gmem_size=0, gmem_init=None,
             smem_bytes=0, strict=False):
    """Convenience: load, run, return (gmem np.uint8, cycles, warps)."""
    img = load_aecbin(aecbin_path)
    if gmem_init is not None:
        gmem = np.frombuffer(bytes(gmem_init), np.uint8).copy()
    else:
        gmem = np.zeros(gmem_size, np.uint8)
    sim = Sim(img, gmem, param_block, smem_bytes, strict)
    sim.run(grid, block)
    return gmem, sim.total_cycles, sim.warps

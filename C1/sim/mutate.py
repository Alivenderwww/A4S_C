"""mutate.py — PTX mutations matching the competition's robustness variants
(scoring.md): register renaming, basic-block reorder, dead-code insertion,
register-pressure increase, loop-count change, and GEMM dtype change.

Two kinds:
  * SEMANTIC-PRESERVING (rename / reorder / deadcode / pressure): the I/O
    behaviour is unchanged, so bench.py checks the variant against the SAME
    numpy reference. Register-pressure is only meaningful at -O0 (else DCE
    removes it).
  * SEMANTIC-CHANGING (set_loop_count / gemm_to_dtype): the result changes
    predictably, so bench.py rebuilds a matching reference (loop_count / dtype).

Register classes in the public kernels: %p (pred), %r (b32), %rd (b64),
%f (f32), %h (f16). Special registers (%tid.x/%ctaid.x/...) start with
%t/%c/%n/%l and are never touched.
"""
import re

_REG = re.compile(r"%(rd|r|f|h|p)(\d+)\b")   # 'rd' before 'r' so it wins.


def _lcg(seed):
    s = (seed * 2654435761 + 12345) & 0x7fffffff
    while True:
        s = (s * 1103515245 + 12345) & 0x7fffffff
        yield s


def _shuffle(lst, seed):
    g = _lcg(seed)
    for i in range(len(lst) - 1, 0, -1):
        j = next(g) % (i + 1)
        lst[i], lst[j] = lst[j], lst[i]
    return lst


# --- 1. register renaming (semantic-preserving) ---------------------------
def rename_registers(ptx, seed=0):
    used = {}
    for cls, idx in _REG.findall(ptx):
        used.setdefault(cls, set()).add(int(idx))
    perm = {}
    for cls, idxs in used.items():
        order = _shuffle(sorted(idxs), seed + hash(cls) % 97)
        perm[cls] = {src: dst for src, dst in zip(sorted(idxs), order)}

    def sub(m):
        cls, idx = m.group(1), int(m.group(2))
        return "%%%s%d" % (cls, perm.get(cls, {}).get(idx, idx))

    return _REG.sub(sub, ptx)


# --- 2. dead-code insertion (semantic-preserving) -------------------------
def insert_dead_code(ptx, seed=0, count=4):
    lines = ptx.splitlines()
    decl_re = re.compile(r"(\.reg\s+\.f32\s+%f<)(\d+)(>)")
    base = None
    for i, ln in enumerate(lines):
        m = decl_re.search(ln)
        if m:
            base = int(m.group(2))
            lines[i] = decl_re.sub(lambda mm: mm.group(1) + str(base + count) + mm.group(3), ln)
            break
    if base is None:
        return ptx
    last_decl = max(i for i, l in enumerate(lines) if ".reg" in l)
    dead = ["    add.f32 %%f%d, %%f0, %%f0;   // dead (seed=%d)" % (base + k, seed)
            for k in range(count)]
    lines[last_decl + 1:last_decl + 1] = dead
    return "\n".join(lines) + "\n"


# --- 3. basic-block reorder (semantic-preserving) -------------------------
_LABEL = re.compile(r"^\s*(\w+):\s*$")


def _is_terminated(ins):
    for l in reversed(ins):
        s = l.strip()
        if not s or s.startswith("//"):
            continue
        if re.match(r"^(ret|exit)\s*;", s):
            return True
        if re.match(r"^bra\s+\w+\s*;", s):   # unconditional bra (no leading @)
            return True
        return False
    return False


def reorder_blocks(ptx, seed=0):
    """Make every fall-through explicit (append `bra next`), then shuffle all
    non-entry blocks. Tests CFG construction independence from block order."""
    lines = ptx.splitlines()
    try:
        open_i = next(i for i, l in enumerate(lines) if l.strip() == "{")
        close_i = max(i for i, l in enumerate(lines) if l.strip() == "}")
    except (StopIteration, ValueError):
        return ptx
    if close_i <= open_i:
        return ptx
    head, tail = lines[:open_i + 1], lines[close_i:]
    body = lines[open_i + 1:close_i]

    preamble, code, in_code = [], [], False
    for l in body:
        s = l.strip()
        if not in_code and (s.startswith(".reg") or s.startswith(".local") or
                            s.startswith(".shared") or s == ""):
            preamble.append(l)
        else:
            in_code = True
            code.append(l)

    blocks, cur_label, cur = [], None, []
    for l in code:
        m = _LABEL.match(l)
        if m:
            if cur or cur_label is not None:
                blocks.append((cur_label, cur))
            cur_label, cur = m.group(1), []
        else:
            cur.append(l)
    if cur or cur_label is not None:
        blocks.append((cur_label, cur))
    if len(blocks) < 3:
        return ptx

    labels = [blocks[0][0]] + [(blocks[i][0] or "BBR%d" % i) for i in range(1, len(blocks))]
    fixed = []
    for i, (_, ins) in enumerate(blocks):
        ins = list(ins)
        if not _is_terminated(ins):
            ins.append("    bra %s;" % labels[i + 1] if i + 1 < len(blocks) else "    ret;")
        fixed.append((labels[i], ins))

    order = [0] + _shuffle(list(range(1, len(fixed))), seed)
    out = list(head) + preamble
    for bi in order:
        lab, ins = fixed[bi]
        if lab is not None:
            out.append("%s:" % lab)
        out.extend(ins)
    out += tail
    return "\n".join(out) + "\n"


# --- 4. register-pressure increase (semantic-preserving; test at -O0) ------
def increase_register_pressure(ptx, seed=0, n=100):
    """Inject n independent live temps (all = %tid.x) plus a dead reduction that
    keeps them simultaneously live -> stresses the register allocator. The
    reduction's result is unused, so behaviour is unchanged (and -O2 DCE removes
    it; use -O0 to actually exercise allocation)."""
    lines = ptx.splitlines()
    last_decl = max((i for i, l in enumerate(lines) if ".reg" in l), default=None)
    if last_decl is None:
        return ptx
    decls = ["    .reg .b32 %%rp<%d>;" % n, "    .reg .b32 %rq<2>;"]
    inj = ["    mov.u32 %%rp%d, %%tid.x;" % i for i in range(n)]
    inj.append("    add.u32 %rq0, %rp0, %rp1;")
    inj += ["    add.u32 %%rq0, %%rq0, %%rp%d;" % i for i in range(2, n)]
    inj.append("    // ^ dead pressure chain (seed=%d)" % seed)
    lines[last_decl + 1:last_decl + 1] = decls + inj
    return "\n".join(lines) + "\n"


# --- 5. loop-count change (semantic-CHANGING; ref rebuilt with new count) ---
def set_loop_count(ptx, new_count):
    """Rewrite the literal trip count in a `setp.lt.u32 %pX, %rY, N;` loop
    condition (poly=32, reuse=16). GEMM's loop bound is a register, so untouched."""
    return re.sub(r"(setp\.lt\.u32\s+%p\d+,\s*%r\d+,\s*)\d+(\s*;)",
                  r"\g<1>%d\g<2>" % new_count, ptx)


# --- 6. GEMM dtype change (semantic-CHANGING; ref rebuilt in that dtype) ----
def gemm_to_bf16(ptx):
    """f16 -> bf16: same 2-byte storage, only the convert changes."""
    return ptx.replace("cvt.f32.f16", "cvt.f32.bf16")


def gemm_to_f32(ptx):
    """f16 -> f32: 4-byte elements (stride 2->4), direct f32 loads, drop cvt."""
    lines, cvtmap = [], {}
    for l in ptx.splitlines():
        m = re.match(r"\s*cvt\.f32\.f16\s+(%f\d+),\s*(%r\d+)\s*;", l)
        if m:
            cvtmap[m.group(2)] = m.group(1)      # remember dst<-src, drop cvt.
            continue
        lines.append(l)
    out = []
    for l in lines:
        m = re.match(r"(\s*)ld\.global\.u16\s+(%r\d+),\s*(\[[^\]]+\])\s*;", l)
        if m and m.group(2) in cvtmap:
            out.append("%sld.global.f32 %s, %s;" % (m.group(1), cvtmap[m.group(2)], m.group(3)))
        else:
            out.append(re.sub(r"(mul\.wide\.u32\s+%rd\d+,\s*%r\d+,\s*)2(\s*;)", r"\g<1>4\g<2>", l))
    return "\n".join(out) + "\n"


PRESERVING = {
    "rename": rename_registers,
    "deadcode": insert_dead_code,
    "reorder": reorder_blocks,
    "pressure": increase_register_pressure,   # bench runs this at -O0.
}

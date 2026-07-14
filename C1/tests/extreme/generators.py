"""Semantic-preserving PTX mutations for deterministic contract testing.

Provides deterministic mutation functions that preserve kernel semantics for
PTX 9.3 on the current AEC ISA (no FP16/BF16/TMUL/multi-kernel constructs).
The composition function ``mutate_contract_source(base_ptx, seed)`` applies
all mutations in a fixed order with derived sub-seeds.
"""

from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

# ── deterministic class salt (replaces built-in hash() for cross-process stability) ──

_CLASS_SALT: Dict[str, int] = {
    "rd": 1,
    "r": 2,
    "f": 3,
    "p": 4,
}

# ── inline test PTX constants ──────────────────────────────────────────

BASE_PTX: str = """.version 9.3
.target sm_90
.address_size 64

.visible .entry test_kernel(
    .param .u64 param_a,
    .param .u64 param_b,
    .param .u64 param_c,
    .param .u32 param_n
)
{
    .reg .pred %p<2>;
    .reg .u32  %r<8>;
    .reg .u64  %rd<10>;
    .reg .f32  %f<4>;

    ld.param.u64 %rd1, [param_a];
    ld.param.u64 %rd2, [param_b];
    ld.param.u64 %rd3, [param_c];
    ld.param.u32 %r1,  [param_n];

    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;

    mad.lo.u32 %r5, %r3, %r4, %r2;

    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra DONE;

    mul.wide.u32 %rd4, %r5, 4;

    add.u64 %rd5, %rd1, %rd4;
    add.u64 %rd6, %rd2, %rd4;
    add.u64 %rd7, %rd3, %rd4;

    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd6];

    add.f32 %f3, %f1, %f2;

    st.global.f32 [%rd7], %f3;

DONE:
    ret;
}
"""

LOOP_PTX: str = """.version 9.3
.target sm_90
.address_size 64

.visible .entry loop_kernel(
    .param .u64 param_a,
    .param .u32 param_n
)
{
    .reg .pred %p<3>;
    .reg .u32  %r<10>;
    .reg .u64  %rd<10>;
    .reg .f32  %f<4>;

    ld.param.u64 %rd1, [param_a];
    ld.param.u32 %r1,  [param_n];

    mov.u32 %r2, 0;

LOOP:
    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra DONE;

    mul.wide.u32 %rd2, %r2, 4;
    add.u64 %rd3, %rd1, %rd2;
    ld.global.f32 %f1, [%rd3];
    add.f32 %f2, %f1, %f2;
    st.global.f32 [%rd3], %f2;

    add.u32 %r2, %r2, 1;
    bra LOOP;

DONE:
    ret;
}
"""

# ── deterministic randomness helpers ───────────────────────────────────


def _lcg(seed: int):
    """LCG-based deterministic pseudo-random generator (same as C1/sim/mutate.py)."""
    s = (seed * 2654435761 + 12345) & 0x7FFFFFFF
    while True:
        s = (s * 1103515245 + 12345) & 0x7FFFFFFF
        yield s


def _shuffle(lst: list, seed: int) -> list:
    """Deterministic Fisher-Yates shuffle."""
    g = _lcg(seed)
    for i in range(len(lst) - 1, 0, -1):
        j = next(g) % (i + 1)
        lst[i], lst[j] = lst[j], lst[i]
    return lst


# ── 1. register renaming (never touch special registers) ───────────────

# Matches general registers only: %rdN, %rN, %fN, %pN
# The "rd" alternative must come before "r" so %rd5 matches as class 'rd', not 'r'.
_REG_USE = re.compile(r"%(rd|r|f|p)(\d+)\b")


def rename_registers(ptx: str, seed: int = 0) -> str:
    """Deterministically rename general registers within each class.

    Special registers (``%tid.x``, ``%ctaid.x``, ``%ntid.x``, ``%laneid``,
    etc.) are never matched by ``_REG_USE`` and thus preserved.
    The permutation is a bijection on the used register indices, so
    ``.reg`` declarations remain correct.
    """
    used: Dict[str, Set[int]] = {}
    for m in _REG_USE.finditer(ptx):
        cls = m.group(1)
        idx = int(m.group(2))
        used.setdefault(cls, set()).add(idx)

    perm: Dict[str, Dict[int, int]] = {}
    for cls, idxs in used.items():
        order = _shuffle(sorted(idxs), seed + _CLASS_SALT[cls] % 97)
        perm[cls] = {src: dst for src, dst in zip(sorted(idxs), order)}

    def _sub(m: re.Match) -> str:
        cls = m.group(1)
        idx = int(m.group(2))
        mapped = perm.get(cls, {}).get(idx, idx)
        return f"%{cls}{mapped}"

    return _REG_USE.sub(_sub, ptx)


# ── 2. sparse register numbering with .reg decl updates ────────────────


def sparse_register_numbering(ptx: str, seed: int = 0) -> str:
    """Sparsify register numbers within each class and update ``.reg`` declarations.

    Each used register index *i* is mapped to ``i * gap + 1`` where *gap*
    is 2–4, creating gaps in the numbering.  The declaration ``%r<N>``
    is updated to cover the new maximum, and every register use is renamed.
    Unused indices (e.g. ``%r0``) are left untouched.
    """
    lines = ptx.splitlines(True)

    for cls in ("rd", "r", "f", "p"):
        cls_re = re.compile(rf"%({cls})(\d+)\b")

        # Collect used indices from NON-decl lines
        used: Set[int] = set()
        for ln in lines:
            if ".reg" in ln:
                continue
            for m in cls_re.finditer(ln):
                used.add(int(m.group(2)))
        if not used:
            continue

        gap = 2 + (seed + _CLASS_SALT[cls]) % 3  # gap ∈ {2, 3, 4}
        mapping: Dict[int, int] = {i: i * gap + 1 for i in used}
        new_max = max(mapping.values())

        # --- update .reg declarations ---
        decl_re = re.compile(rf"(\.reg\s+\.\w+\s+%{cls}<)(\d+)(>)")
        new_lines: List[str] = []

        for ln in lines:
            if ".reg" in ln:
                def _upd(m: re.Match, cls=cls, new_max=new_max) -> str:
                    return m.group(1) + str(max(int(m.group(2)), new_max + 1)) + m.group(3)
                ln = decl_re.sub(_upd, ln)
            else:
                def _ren(m: re.Match, cls=cls, mapping=mapping) -> str:
                    idx = int(m.group(2))
                    return f"%{cls}{mapping.get(idx, idx)}"
                ln = cls_re.sub(_ren, ln)
            new_lines.append(ln)

        lines = new_lines

    return "".join(lines)


# ── 3. block reordering with explicit fallthrough ──────────────────────

_LABEL_RE = re.compile(r"^\s*(\w+):\s*$")


def _is_terminated(ins: List[str]) -> bool:
    """Return True if *ins* end with a terminator (ret, exit, or unconditional bra)."""
    for l in reversed(ins):
        s = l.strip()
        if not s or s.startswith("//"):
            continue
        if re.match(r"^(ret|exit)\s*;", s):
            return True
        if re.match(r"^bra\s+\w+\s*;", s):
            return True
        return False
    return False


def reorder_blocks(ptx: str, seed: int = 0) -> str:
    """Reorder all basic blocks after the entry block and add explicit fallthrough branches.

    Every block that ends without a terminator gets an explicit ``bra next_label;``
    so the shuffle is semantically neutral.  Requires at least three blocks to
    do anything; returns the PTX unchanged otherwise.
    """
    lines = ptx.splitlines(True)

    # Find function body delimiters
    try:
        open_i = next(i for i, l in enumerate(lines) if l.strip() == "{")
        close_i = max(i for i, l in enumerate(lines) if l.strip() == "}")
    except (StopIteration, ValueError):
        return ptx
    if close_i <= open_i:
        return ptx

    head = lines[:open_i + 1]
    tail = lines[close_i:]
    body = lines[open_i + 1:close_i]

    # Separate preamble (declarations) from code
    preamble: List[str] = []
    code: List[str] = []
    in_code = False
    for l in body:
        s = l.strip()
        if not in_code and (s.startswith(".reg") or s.startswith(".local")
                            or s.startswith(".shared") or s == ""):
            preamble.append(l)
        else:
            in_code = True
            code.append(l)

    # Split code into basic blocks by label
    blocks: List[Tuple[str | None, List[str]]] = []
    cur_label: str | None = None
    cur: List[str] = []
    for l in code:
        m = _LABEL_RE.match(l)
        if m:
            if cur or cur_label is not None:
                blocks.append((cur_label, cur))
            cur_label = m.group(1)
            cur = []
        else:
            cur.append(l)
    if cur or cur_label is not None:
        blocks.append((cur_label, cur))

    if len(blocks) < 3:
        return ptx  # not enough blocks to meaningfully reorder

    # Build label names (synthesize for unnamed entry block)
    labels: List[str | None] = [blocks[0][0]]
    labels += [(blocks[i][0] or f"BBR{i}") for i in range(1, len(blocks))]

    # Add explicit fallthrough branches
    fixed: List[Tuple[str | None, List[str]]] = []
    for i, (lab, ins) in enumerate(blocks):
        ins = list(ins)
        if not _is_terminated(ins):
            if i + 1 < len(blocks):
                # Ensure fallthrough label is not None
                ft_label = labels[i + 1] if labels[i + 1] is not None else f"BBR{i+1}"
                ins.append(f"    bra {ft_label};\n")
            else:
                ins.append("    ret;\n")
        fixed.append((lab, ins))

    # Shuffle non-entry blocks
    order = [0] + _shuffle(list(range(1, len(fixed))), seed)

    out = list(head) + preamble
    for bi in order:
        lab, ins = fixed[bi]
        if lab is not None:
            out.append(f"{lab}:\n")
        out.extend(ins)
    out += tail
    return "".join(out)


# ── 4. dead FP32/integer computation insertion ─────────────────────────


def insert_dead_code(ptx: str, seed: int = 0, count: int = 2) -> str:
    """Insert *count* dead ``add.f32`` instructions that use existing values.

    The existing ``.reg .f32 %f<N>`` declaration is expanded so the new
    temporary registers are valid.  Results are unused so program semantics
    are preserved.
    """
    lines = ptx.splitlines(True)

    f32_decl_re = re.compile(r"(\.reg\s+\.f32\s+%f<)(\d+)(>)")
    last_decl_idx = -1
    orig_f_count = 0

    for i, ln in enumerate(lines):
        m = f32_decl_re.search(ln)
        if m:
            orig_f_count = int(m.group(2))
            lines[i] = f32_decl_re.sub(
                lambda mm: mm.group(1) + str(orig_f_count + count) + mm.group(3),
                ln,
            )
        if ".reg" in ln:
            last_decl_idx = i

    if last_decl_idx < 0 or orig_f_count < 2:
        return ptx  # cannot insert meaningful dead code

    dead: List[str] = []
    for k in range(count):
        lhs = orig_f_count + k
        rhs1 = k % orig_f_count
        rhs2 = (k + 1) % orig_f_count
        dead.append(f"    add.f32 %f{lhs}, %f{rhs1}, %f{rhs2};  // dead (seed={seed})\n")

    # Insert after the last .reg declaration (before the first code line)
    lines[last_decl_idx + 1:last_decl_idx + 1] = dead

    return "".join(lines)


# ── 5. comments / whitespace variation ─────────────────────────────────


def vary_comments_whitespace(ptx: str, seed: int = 0) -> str:
    """Deterministically add trailing comments to some instruction lines."""
    g = _lcg(seed)
    lines = ptx.splitlines(True)
    out: List[str] = []

    for ln in lines:
        s = ln.strip()
        # Only annotate instruction lines (skip .directives, labels, braces, comments)
        if (s and not s.startswith(".") and not s.startswith("//")
                and "{" not in s and "}" not in s and ":" not in s
                and s.endswith(";")):
            choice = next(g) % 3
            if choice == 0:
                ln = ln.rstrip("\n") + f"  // variant (seed={seed})\n"
            elif choice == 1:
                ln = ln.rstrip("\n") + "  // mv\n"
        out.append(ln)

    return "".join(out)


# ── 6. equivalent address forms ────────────────────────────────────────


def equivalent_address_forms(ptx: str, seed: int = 0) -> str:
    """Replace ``mul.wide.u32`` + ``add.u64`` address computation with an
    equivalent restricted-subset multiply-and-double sequence when the
    multiplier is a power of two greater than one.

    Example transformation::

        mul.wide.u32 %rd4, %r5, 4;
        add.u64 %rd5, %rd1, %rd4;

    becomes::

        mul.wide.u32 %rd4, %r5, 2;
        add.u64 %rd4, %rd4, %rd4;
        add.u64 %rd5, %rd1, %rd4;

    The same registers are reused so no new declarations are needed.
    """
    _ = seed  # deterministic but unused for this specific transform
    lines = ptx.splitlines(True)

    mul_re = re.compile(
        r"(\s*)mul\.wide\.u32\s+(%rd\d+),\s*(%r\d+),\s*(\d+)\s*;"
    )

    for i, ln in enumerate(lines):
        m = mul_re.match(ln)
        if not m:
            continue
        indent = m.group(1)
        rd_tmp = m.group(2)
        r_src = m.group(3)
        mult_str = m.group(4)
        mult = int(mult_str)

        # Halve the multiplier and double the resulting legal .u64 pair.
        if mult <= 1 or (mult & (mult - 1)) != 0:
            continue

        # Check nearby lines for add.u64 that uses the mul result
        found = False
        for j in range(i + 1, min(i + 5, len(lines))):
            add_m = re.match(
                r"\s*add\.u64\s+(%rd\d+),\s*(%rd\d+),\s*"
                + re.escape(rd_tmp)
                + r"\s*;",
                lines[j],
            )
            if add_m:
                # Found a matching pair — replace the mul line
                lines[i] = (
                    f"{indent}mul.wide.u32 {rd_tmp}, {r_src}, {mult // 2};\n"
                    f"{indent}add.u64 {rd_tmp}, {rd_tmp}, {rd_tmp};\n"
                )
                found = True
                break

        if found:
            # Update remaining add.u64 lines that use the same rd_tmp
            for j in range(i + 1, min(i + 5, len(lines))):
                if lines[j].strip().startswith("add.u64") and rd_tmp in lines[j]:
                    # The add.u64 is already correct — just skip
                    pass
            # If we replaced the mul, continue scanning past the affected area
            # (the line after the pair keeps its original add.u64)
            continue

    return "".join(lines)


# ════════════════════════════════════════════════════════════════════════
# Composition
# ════════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════
# Cross-block pressure generators (Task 5)
# ════════════════════════════════════════════════════════════════════════


def _expected_gpr_sum(count: int) -> int:
    """Independent reference: sum(1..count) = count*(count+1)//2."""
    return count * (count + 1) // 2


def generate_gpr_pressure(count: int) -> str:
    """Generate PTX 9.3 with *count* 32-bit GPR live across a CFG boundary.

    Each ``%r<i+1>`` gets value ``i+1`` in ENTRY.  After ``bra REDUCE;`` every
    value is summed into ``%r0`` and stored to GMEM, producing an
    independently checkable integer result.

    Uses only the legal register families: ``%r<N>`` for values/accumulator and
    ``%rd<N>`` for the output pointer.  Accumulator and output pointer are
    initialized/loaded inside REDUCE so they are not part of the cross-block
    live set.
    """
    body = []
    body.append(".version 9.3")
    body.append(".target sm_90")
    body.append(".address_size 64")
    body.append("")
    body.append(".visible .entry gpr_pressure_%d(" % count)
    body.append("    .param .u64 param_out")
    body.append(")")
    body.append("{")
    body.append("    .reg .u32 %%r<%d>;" % (count + 1))
    body.append("    .reg .u64 %rd<2>;")
    body.append("")
    for i in range(count):
        body.append("    mov.u32 %%r%d, %d;" % (i + 1, i + 1))
    body.append("    bra REDUCE;")
    body.append("")
    body.append("REDUCE:")
    body.append("    ld.param.u64 %rd1, [param_out];")
    body.append("    mov.u32 %r0, 0;")
    for i in range(count):
        body.append("    add.u32 %%r0, %%r0, %%r%d;" % (i + 1))
    body.append("    st.global.u32 [%rd1], %r0;")
    body.append("    ret;")
    body.append("}")
    body.append("")
    return "\n".join(body)


def _expected_pair_sum(count: int) -> int:
    """Read *count* u32 values (each 1) from GMEM input slots and sum them.
    Returns *count* (each loaded value is 1 at offset ``256+4*i``).
    """
    return count


def generate_pair_pressure(count: int) -> str:
    """Generate PTX 9.3 with *count* 64-bit pair registers live across a CFG boundary.

    Each ``%bd<i>`` holds GMEM offset ``256 + 4*i`` in ENTRY.  After
    ``bra REDUCE;`` REDUCE loads a u32 from each offset via ``ld.global.u32``,
    accumulates with ``add.u32``, and stores the sum (equal to *count*) to
    offset 0 of the output.

    Uses only legal register families: ``%bd<N>`` for pressure values,
    ``%rd<N>`` for the base pointer and address temporaries, and ``%r<N>``
    for the accumulator and load destination.  Base pointer and accumulator
    are initialized inside REDUCE so they are not part of the cross-block
    live set.
    """
    body = []
    body.append(".version 9.3")
    body.append(".target sm_90")
    body.append(".address_size 64")
    body.append("")
    body.append(".visible .entry pair_pressure_%d(" % count)
    body.append("    .param .u64 param_out")
    body.append(")")
    body.append("{")
    body.append("    .reg .b64 %%bd<%d>;" % count)
    body.append("    .reg .u64 %rd<3>;")
    body.append("    .reg .u32 %r<2>;")
    body.append("")
    for i in range(count):
        body.append("    mov.u64 %%bd%d, %d;" % (i, 256 + 4 * i))
    body.append("    bra REDUCE;")
    body.append("")
    body.append("REDUCE:")
    body.append("    ld.param.u64 %rd1, [param_out];")
    body.append("    mov.u32 %r0, 0;")
    for i in range(count):
        body.append("    add.u64 %%rd2, %%rd1, %%bd%d;" % i)
        body.append("    ld.global.u32 %r1, [%rd2];")
        body.append("    add.u32 %r0, %r0, %r1;")
    body.append("    st.global.u32 [%rd1], %r0;")
    body.append("    ret;")
    body.append("}")
    body.append("")
    return "\n".join(body)


def _expected_pred_sum(count: int) -> int:
    """Independent reference: each true predicate contributes 1."""
    return count


def generate_predicate_pressure(count: int) -> str:
    """Generate PTX 9.3 with *count* predicates live across a CFG boundary.

    Each ``%p<i>`` is set to true in ENTRY.  After ``bra REDUCE;`` every
    predicate guards an ``add.u32`` that contributes 1 to the accumulator.

    Uses only legal register families: ``%p<N>`` for predicates, ``%r<N>``
    for the constant and accumulator, and ``%rd<N>`` for the output pointer.
    Accumulator and output pointer are initialized/loaded inside REDUCE so
    they are not part of the cross-block live set.
    """
    body = []
    body.append(".version 9.3")
    body.append(".target sm_90")
    body.append(".address_size 64")
    body.append("")
    body.append(".visible .entry pred_pressure_%d(" % count)
    body.append("    .param .u64 param_out")
    body.append(")")
    body.append("{")
    body.append("    .reg .pred %%p<%d>;" % count)
    body.append("    .reg .u64 %rd<2>;")
    body.append("    .reg .u32 %r<2>;")
    body.append("")
    body.append("    mov.u32 %r1, 1;")
    body.append("")
    for i in range(count):
        body.append("    setp.eq.u32 %%p%d, %%r1, 1;" % i)
    body.append("    bra REDUCE;")
    body.append("")
    body.append("REDUCE:")
    body.append("    ld.param.u64 %rd1, [param_out];")
    body.append("    mov.u32 %r0, 0;")
    for i in range(count):
        body.append("    @%%p%d add.u32 %%r0, %%r0, 1;" % i)
    body.append("    st.global.u32 [%rd1], %r0;")
    body.append("    ret;")
    body.append("}")
    body.append("")
    return "\n".join(body)


# ════════════════════════════════════════════════════════════════════════
# Composition
# ════════════════════════════════════════════════════════════════════════


def mutate_contract_source(base_ptx: str, seed: int) -> str:
    """Apply all contract-safe semantic-preserving mutations in fixed order.

    Mutations applied:
    1. Register renaming (preserves ``%tid.x`` / ``%ctaid.x`` etc.)
    2. Sparse register numbering with updated ``.reg`` declarations
    3. Block reordering with explicit fallthrough branches
    4. Dead FP32 computation insertion
    5. Comments / whitespace variation
    6. Equivalent address forms

    Args:
        base_ptx: Input PTX 9.3 source.
        seed: Deterministic seed (expected values: 0, 1, 7, 17).

    Returns:
        Mutated PTX.  Guaranteed to never contain FP16/BF16/TMUL/multi-kernel.
    """
    ptx = rename_registers(base_ptx, seed)
    ptx = sparse_register_numbering(ptx, seed + 1)
    ptx = reorder_blocks(ptx, seed + 2)
    ptx = insert_dead_code(ptx, seed + 3, count=2)
    ptx = vary_comments_whitespace(ptx, seed + 4)
    ptx = equivalent_address_forms(ptx, seed + 5)
    return ptx

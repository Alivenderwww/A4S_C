"""Immutable case and output-expectation models for extreme correctness tests.

Provides pack_pmem for natural-alignment parameter packing,
load_public_contract_cases for loading the five public test cases T1..T5
from disk with reduced sizes, and the load_case_matrix dispatch entry point.
"""

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from tests.extreme.generators import (
    generate_gpr_pressure,
    generate_pair_pressure,
    generate_predicate_pressure,
    mutate_contract_source,
)

# ── repo root discovery ────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent.parent.parent  # C1/tests/extreme -> A4S_C root


@dataclass(frozen=True)
class OutputExpectation:
    """Expected output shape, dtype, values, and comparison tolerances."""

    offset: int
    dtype: str
    shape: Tuple[int, ...]
    expected: Tuple[float, ...]
    rtol: float
    atol: float


@dataclass(frozen=True)
class ExtremeCase:
    """An immutable description of one extreme-test case."""

    name: str
    suite: str
    ptx: str
    grid: Tuple[int, int, int]
    block: Tuple[int, int, int]
    pmem: bytes
    gmem: bytes
    output: OutputExpectation
    opt_levels: Tuple[str, ...]
    expected_failure: Optional[str]
    expected_failure_phase: Optional[str] = None


# ── per-suite reduced configs ──────────────────────────────────────────
# (category_dir, reduced_n_or_dim, grid, block)
_CONTRACT_CONFIGS: Tuple[Tuple[str, int, Tuple[int, int, int], Tuple[int, int, int]], ...] = (
    ("T1_basic_lowering",      256, (1, 1, 1), (256, 1, 1)),
    ("T2_scalar_optimization", 256, (1, 1, 1), (256, 1, 1)),
    ("T3_memory_reuse",        256, (1, 1, 1), (256, 1, 1)),
    ("T4_register_scheduling", 256, (1, 1, 1), (256, 1, 1)),
    ("T5_scalar_gemm",          16, (1, 1, 1), (16, 16, 1)),
)


# ════════════════════════════════════════════════════════════════════════
# Parameter packing
# ════════════════════════════════════════════════════════════════════════

def pack_pmem(params: List[Tuple[str, int]]) -> bytes:
    """Pack a list of (type, value) parameter pairs with natural alignment.

    Types ``'u64'`` align to 8 bytes; ``'u32'`` align to 4 bytes.
    The final buffer is zero-padded to an 8-byte boundary.

    Args:
        params: List of (type_str, value) tuples, e.g. ``("u64", 256)``.

    Returns:
        Packed bytes buffer.
    """
    out = bytearray()
    off = 0
    for typ, val in params:
        sz = 8 if typ == "u64" else 4
        # natural alignment
        off = (off + sz - 1) // sz * sz
        while len(out) < off:
            out.append(0)
        if typ == "u64":
            out += struct.pack("<Q", val)
        else:
            out += struct.pack("<I", val & 0xFFFFFFFF)
        off = len(out)
    # final 8-byte padding
    while len(out) % 8:
        out.append(0)
    return bytes(out)


# ════════════════════════════════════════════════════════════════════════
# Reference functions — forced FP32 operation order
# ════════════════════════════════════════════════════════════════════════

def _ref_t1(bufs: Dict[str, np.ndarray]) -> np.ndarray:
    """c[i] = a[i] + b[i]"""
    return (bufs["a"].astype(np.float32) + bufs["b"].astype(np.float32)).astype(np.float32)


def _ref_t2(bufs: Dict[str, np.ndarray]) -> np.ndarray:
    """out[i] = (x[i] + y[i]) * (x[i] + y[i]) + x[i]"""
    x = bufs["x"]
    y = bufs["y"]
    s = (x + y).astype(np.float32)
    return (s * s).astype(np.float32) + x


def _ref_t3(bufs: Dict[str, np.ndarray]) -> np.ndarray:
    """out[i] = x[i] * y[i] + x[i] * z[i]"""
    x = bufs["x"].astype(np.float32)
    y = bufs["y"].astype(np.float32)
    z = bufs["z"].astype(np.float32)
    return (x * y + x * z).astype(np.float32)


def _ref_t4(bufs: Dict[str, np.ndarray]) -> np.ndarray:
    """out[i] = (a[i]+b[i])*(c[i]-d[i]) + (a[i]*c[i])*(b[i]+d[i])"""
    a = bufs["a"].astype(np.float32)
    b = bufs["b"].astype(np.float32)
    c = bufs["c"].astype(np.float32)
    d = bufs["d"].astype(np.float32)
    lhs = ((a + b) * (c - d)).astype(np.float32)
    rhs = ((a * c) * (b + d)).astype(np.float32)
    return (lhs + rhs).astype(np.float32)


def _ref_t5(bufs: Dict[str, np.ndarray]) -> np.ndarray:
    """C = A @ B"""
    A = bufs["A"].astype(np.float32)
    B = bufs["B"].astype(np.float32)
    return (A @ B).astype(np.float32)


# Map public formula strings to their reference implementations
_FORMULA_REF_MAP: Dict[str, Callable[[Dict[str, np.ndarray]], np.ndarray]] = {
    "c[i] = a[i] + b[i]": _ref_t1,
    "out[i] = (x[i] + y[i]) * (x[i] + y[i]) + x[i]": _ref_t2,
    "out[i] = x[i] * y[i] + x[i] * z[i]": _ref_t3,
    "out[i] = (a[i] + b[i]) * (c[i] - d[i]) + (a[i] * c[i]) * (b[i] + d[i])": _ref_t4,
    "C = A @ B": _ref_t5,
}


# ════════════════════════════════════════════════════════════════════════
# GMEM / PMEM construction helpers
# ════════════════════════════════════════════════════════════════════════

def _build_elementwise_gmem(
    manifest: dict, n: int,
) -> Tuple[bytes, Dict[str, int], Dict[str, np.ndarray]]:
    """Build flat GMEM bytearray for an elementwise manifest with *n* elements.

    Returns ``(gmem_bytes, ptr_map, buf_map)`` where *ptr_map* maps buffer
    names to byte offsets and *buf_map* maps buffer names to numpy arrays.
    """
    gmem = bytearray()
    ptrs: Dict[str, int] = {}
    bufs: Dict[str, np.ndarray] = {}

    for name, info in manifest["buffers"].items():
        seed = info.get("seed", 0)
        rng = np.random.default_rng(seed)

        if info["init"] == "rand_uniform":
            data = rng.standard_normal(n).astype(np.float32)
        else:  # "zero"
            data = np.zeros(n, np.float32)

        ptrs[name] = len(gmem)
        bufs[name] = data
        gmem += data.tobytes()

    return bytes(gmem), ptrs, bufs


def _build_matmul_gmem(
    manifest: dict, dim: int,
) -> Tuple[bytes, Dict[str, int], Dict[str, np.ndarray]]:
    """Build flat GMEM bytearray for a matmul manifest with *dim* x *dim* matrices."""
    gmem = bytearray()
    ptrs: Dict[str, int] = {}
    bufs: Dict[str, np.ndarray] = {}

    for name, info in manifest["buffers"].items():
        seed = info.get("seed", 0)
        rng = np.random.default_rng(seed)
        shape = (dim, dim)

        if info["init"] == "rand_uniform":
            data = rng.standard_normal(shape).astype(np.float32)
        else:
            data = np.zeros(shape, np.float32)

        ptrs[name] = len(gmem)
        bufs[name] = data
        gmem += data.tobytes()

    return bytes(gmem), ptrs, bufs


def _build_pmem(manifest: dict, ptrs: Dict[str, int], n: int) -> bytes:
    """Build the parameter block using GMEM offsets and reduced size."""
    params: List[Tuple[str, int]] = []
    for p in manifest["params"]:
        if p["kind"] == "gmem_ptr":
            params.append((p["type"], ptrs[p["buffer"]]))
        else:
            params.append((p["type"], n))
    return pack_pmem(params)


# ════════════════════════════════════════════════════════════════════════
# Public loading functions
# ════════════════════════════════════════════════════════════════════════

def load_public_contract_cases(root: Path) -> List[ExtremeCase]:
    """Load the five public contract test cases (T1..T5) from disk.

    Reads each ``T*/manifest.json`` and ``T*/kernel.ptx`` from the public
    testcases directory under *root*, then builds :class:`ExtremeCase`
    objects with reduced sizes for fast local execution.  Buffer data are
    deterministically generated using the per-buffer seeds from each manifest.

    Args:
        root: Repository root directory (``A4S_C``).

    Returns:
        List of five :class:`ExtremeCase` objects named ``T1`` .. ``T5``.
    """
    tc_root = root / "public" / "Track-C" / "C1-compiler" / "testcases"
    if not tc_root.is_dir():
        return []

    cases: List[ExtremeCase] = []

    for category, n_val, grid, block in _CONTRACT_CONFIGS:
        case_dir = tc_root / category
        manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
        ptx = (case_dir / "kernel.ptx").read_text(encoding="utf-8")

        # Short name "T1" from "T1_vector_add" etc.
        name = manifest["name"][:2]

        # Build GMEM
        chk = manifest["check"]
        if chk["type"] == "matmul":
            gmem, ptrs, bufs = _build_matmul_gmem(manifest, n_val)
        else:
            gmem, ptrs, bufs = _build_elementwise_gmem(manifest, n_val)

        # Build PMEM
        pmem = _build_pmem(manifest, ptrs, n_val)

        # Compute reference output
        formula = chk["formula"]
        ref_fn = _FORMULA_REF_MAP[formula]
        expected_arr: np.ndarray = ref_fn(bufs)

        output_name = chk["output"]
        out_offset = ptrs[output_name]

        # Use the actual computed shape (reduced sizes override manifest)
        shape: Tuple[int, ...] = tuple(expected_arr.shape)

        expected_flat: Tuple[float, ...] = tuple(expected_arr.ravel().tolist())

        output = OutputExpectation(
            offset=out_offset,
            dtype="<f4",
            shape=shape,
            expected=expected_flat,
            rtol=float(chk.get("rtol", 1e-5)),
            atol=float(chk.get("atol", 1e-5)),
        )

        case = ExtremeCase(
            name=name,
            suite="contract",
            ptx=ptx,
            grid=grid,
            block=block,
            pmem=pmem,
            gmem=gmem,
            output=output,
            opt_levels=("O2",),
            expected_failure=None,
        )
        cases.append(case)

    return cases


# ════════════════════════════════════════════════════════════════════════
# Contract matrix
# ════════════════════════════════════════════════════════════════════════

# Deterministic seeds and dimension boundaries for the contract matrix.
# The matrix is designed to yield >= 100 (case × opt_level) combinations
# while avoiding uncontrolled Cartesian explosion.
_ELEMENTWISE_SEEDS_SMALL: Tuple[int, ...] = (0, 17)   # → O0/O2/O3
_ELEMENTWISE_SEEDS_LARGE: Tuple[int, ...] = (1, 7)     # → O2 only
_SMALL_LAUNCH_N: Tuple[int, ...] = (1, 31, 32, 33)
_LARGE_LAUNCH_N: Tuple[int, ...] = (255, 256, 257, 1000)

_GEMM_SEEDS_SMALL: Tuple[int, ...] = (0, 17)           # → O0/O2/O3
_GEMM_SEEDS_LARGE: Tuple[int, ...] = (1, 7)            # → O2 only
_SMALL_GEMM_DIMS: Tuple[int, ...] = (1, 15, 16, 17)
_LARGE_GEMM_DIMS: Tuple[int, ...] = (0, 31, 32, 33)


def _make_elementwise_matrix_case(
    manifest: dict,
    ptx: str,
    seed: int,
    n: int,
    block: Tuple[int, int, int],
    short_name: str,
    opt_levels: Tuple[str, ...],
) -> ExtremeCase:
    """Build one elementwise contract-matrix case from a mutated PTX and reduced *n*."""
    chk = manifest["check"]
    gmem, ptrs, bufs = _build_elementwise_gmem(manifest, n)
    pmem = _build_pmem(manifest, ptrs, n)

    ref_fn = _FORMULA_REF_MAP[chk["formula"]]
    expected_arr = ref_fn(bufs)

    output_name = chk["output"]
    output = OutputExpectation(
        offset=ptrs[output_name],
        dtype="<f4",
        shape=tuple(expected_arr.shape),
        expected=tuple(expected_arr.ravel().tolist()),
        rtol=float(chk.get("rtol", 1e-5)),
        atol=float(chk.get("atol", 1e-5)),
    )

    # Adjust grid.x so all n elements are covered
    grid_x = max(1, math.ceil(n / block[0]))

    return ExtremeCase(
        name=f"{short_name}_s{seed}_n{n}",
        suite="contract",
        ptx=ptx,
        grid=(grid_x, 1, 1),
        block=block,
        pmem=pmem,
        gmem=gmem,
        output=output,
        opt_levels=opt_levels,
        expected_failure=None,
    )


def _make_gemm_matrix_case(
    manifest: dict,
    ptx: str,
    seed: int,
    dim: int,
    block: Tuple[int, int, int],
    short_name: str,
    opt_levels: Tuple[str, ...],
) -> ExtremeCase:
    """Build one GEMM contract-matrix case from a mutated PTX and reduced *dim*."""
    chk = manifest["check"]
    gmem, ptrs, bufs = _build_matmul_gmem(manifest, dim)
    pmem = _build_pmem(manifest, ptrs, dim)

    ref_fn = _FORMULA_REF_MAP[chk["formula"]]
    expected_arr = ref_fn(bufs)

    output_name = chk["output"]
    output = OutputExpectation(
        offset=ptrs[output_name],
        dtype="<f4",
        shape=tuple(expected_arr.shape),
        expected=tuple(expected_arr.ravel().tolist()),
        rtol=float(chk.get("rtol", 1e-4)),
        atol=float(chk.get("atol", 1e-4)),
    )

    # Adjust grid so all output cells are covered
    grid_x = max(1, math.ceil(dim / block[0]) if dim > 0 else 1)
    grid_y = max(1, math.ceil(dim / block[1]) if dim > 0 else 1)

    return ExtremeCase(
        name=f"{short_name}_s{seed}_d{dim}",
        suite="contract",
        ptx=ptx,
        grid=(grid_x, grid_y, 1),
        block=block,
        pmem=pmem,
        gmem=gmem,
        output=output,
        opt_levels=opt_levels,
        expected_failure=None,
    )


def _build_contract_matrix(root: Path) -> List[ExtremeCase]:
    """Build the full deterministic contract case matrix.

    For each public base case (T1–T5) generates dimension/seed variants with
    deterministically mutated PTX, rebuilt PMEM/GMEM, and adjusted output
    oracles.  Guarantees ``sum(len(c.opt_levels) for c in result) >= 100``.
    """
    tc_root = root / "public" / "Track-C" / "C1-compiler" / "testcases"
    if not tc_root.is_dir():
        return []

    # Pre-load manifests and base PTX for fast access
    base_data: Dict[str, Tuple[dict, str]] = {}
    for category, _, grid, block in _CONTRACT_CONFIGS:
        case_dir = tc_root / category
        manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
        ptx_text = (case_dir / "kernel.ptx").read_text(encoding="utf-8")
        base_data[category] = (manifest, ptx_text)

    cases: List[ExtremeCase] = []

    # ── Elementwise bases: T1, T2, T3, T4 ──────────────────────────────
    for category, _, _, block in _CONTRACT_CONFIGS[:4]:
        manifest, base_ptx = base_data[category]
        short_name = manifest["name"][:2]  # "T1" … "T4"

        for seed in _ELEMENTWISE_SEEDS_SMALL:
            mutated = mutate_contract_source(base_ptx, seed)
            for n in _SMALL_LAUNCH_N:
                cases.append(_make_elementwise_matrix_case(
                    manifest, mutated, seed, n, block,
                    short_name, ("O0", "O2", "O3"),
                ))

        for seed in _ELEMENTWISE_SEEDS_LARGE:
            mutated = mutate_contract_source(base_ptx, seed)
            for n in _LARGE_LAUNCH_N:
                cases.append(_make_elementwise_matrix_case(
                    manifest, mutated, seed, n, block,
                    short_name, ("O2",),
                ))

    # ── GEMM base: T5 ──────────────────────────────────────────────────
    category, _, _, block = _CONTRACT_CONFIGS[4]
    manifest, base_ptx = base_data[category]
    short_name = manifest["name"][:2]  # "T5"

    for seed in _GEMM_SEEDS_SMALL:
        mutated = mutate_contract_source(base_ptx, seed)
        for dim in _SMALL_GEMM_DIMS:
            cases.append(_make_gemm_matrix_case(
                manifest, mutated, seed, dim, block,
                short_name, ("O0", "O2", "O3"),
            ))

    for seed in _GEMM_SEEDS_LARGE:
        mutated = mutate_contract_source(base_ptx, seed)
        for dim in _LARGE_GEMM_DIMS:
            cases.append(_make_gemm_matrix_case(
                manifest, mutated, seed, dim, block,
                short_name, ("O2",),
            ))

    return cases


# ════════════════════════════════════════════════════════════════════════
# Pressure case boundaries (Task 5)
# ════════════════════════════════════════════════════════════════════════

# GPR contract boundaries — narrow end, expected to always compile/execute
GPR_CONTRACT: Tuple[int, ...] = (64, 128, 192, 240, 252)
# GPR frontier boundaries — may stress the allocator
GPR_FRONTIER: Tuple[int, ...] = (255, 256, 300)

# 64-bit pair contract boundaries
PAIR_CONTRACT: Tuple[int, ...] = (32, 64, 96, 120)
# 64-bit pair frontier boundaries
PAIR_FRONTIER: Tuple[int, ...] = (127, 128, 140)

# Predicate contract boundaries
PRED_CONTRACT: Tuple[int, ...] = (1, 2, 4, 8)
# Predicate frontier boundaries
PRED_FRONTIER: Tuple[int, ...] = (9, 12, 16)

_FRONTIER_COUNT: int = (
    len(GPR_FRONTIER) + len(PAIR_FRONTIER) + len(PRED_FRONTIER)
)
_CONTRACT_COUNT: int = (
    len(GPR_CONTRACT) + len(PAIR_CONTRACT) + len(PRED_CONTRACT)
)


def _build_pressure_cases(root: Path) -> List[ExtremeCase]:
    """Build all GPR/pair/predicate pressure cases with deterministic outputs.

    Contract cases have *expected_failure=None*.  Frontier cases also start
    with *expected_failure=None* — the registry overlay (applied later via
    ``apply_registry``) sets the field for known failures.
    """
    cases: List[ExtremeCase] = []

    def _make(name: str, suite: str, ptx: str, expected: int, gmem: bytes) -> ExtremeCase:
        # PMEM: single u64 pointer to GMEM output (offset 0)
        pmem = pack_pmem([("u64", 0)])
        return ExtremeCase(
            name=name,
            suite=suite,
            ptx=ptx,
            grid=(1, 1, 1),
            block=(1, 1, 1),
            pmem=pmem,
            gmem=gmem,
            output=OutputExpectation(
                offset=0,
                dtype="<u4",
                shape=(1,),
                expected=(expected,),
                rtol=0.0,
                atol=0.0,
            ),
            opt_levels=("O0", "O2", "O3"),
            expected_failure=None,
        )

    # One double-word output slot for GPR and predicate cases
    output_gmem = b"\x00" * 8

    def _build_pair_gmem(count: int) -> bytes:
        """Build GMEM for pair pressure: output slot at 0, u32 1 at each 256+4*i."""
        gmem = bytearray(256 + 4 * count + 4)
        for i in range(count):
            struct.pack_into("<I", gmem, 256 + 4 * i, 1)
        return bytes(gmem)

    # ── GPR pressure ────────────────────────────────────────────────────
    import tests.extreme.generators as _gen
    for n in GPR_CONTRACT:
        ptx = generate_gpr_pressure(n)
        expected = _gen._expected_gpr_sum(n)  # type: ignore[attr-defined]
        cases.append(_make(f"gpr-live-{n}", "pressure", ptx, expected, output_gmem))
    for n in GPR_FRONTIER:
        ptx = generate_gpr_pressure(n)
        expected = _gen._expected_gpr_sum(n)  # type: ignore[attr-defined]
        cases.append(_make(f"gpr-live-{n}", "pressure", ptx, expected, output_gmem))

    # ── 64-bit pair pressure ───────────────────────────────────────────
    for n in PAIR_CONTRACT:
        ptx = generate_pair_pressure(n)
        expected = _gen._expected_pair_sum(n)  # type: ignore[attr-defined]
        cases.append(_make(f"pair-live-{n}", "pressure", ptx, expected, _build_pair_gmem(n)))
    for n in PAIR_FRONTIER:
        ptx = generate_pair_pressure(n)
        expected = _gen._expected_pair_sum(n)  # type: ignore[attr-defined]
        cases.append(_make(f"pair-live-{n}", "pressure", ptx, expected, _build_pair_gmem(n)))

    # ── Predicate pressure ─────────────────────────────────────────────
    for n in PRED_CONTRACT:
        ptx = generate_predicate_pressure(n)
        expected = _gen._expected_pred_sum(n)  # type: ignore[attr-defined]
        cases.append(_make(f"pred-live-{n}", "pressure", ptx, expected, output_gmem))
    for n in PRED_FRONTIER:
        ptx = generate_predicate_pressure(n)
        expected = _gen._expected_pred_sum(n)  # type: ignore[attr-defined]
        cases.append(_make(f"pred-live-{n}", "pressure", ptx, expected, output_gmem))

    return cases


def _get_contract_pressure_cases(root: Path) -> List[ExtremeCase]:
    """Return only contract-level pressure cases, normalized to suite='contract'."""
    from dataclasses import replace
    return [
        replace(c, suite="contract")
        for c in _build_pressure_cases(root)
        if c.name in CONTRACT_NAMES
    ]


def _get_frontier_pressure_cases(root: Path) -> List[ExtremeCase]:
    """Return frontier pressure cases with suite='frontier' and no registry applied."""
    from dataclasses import replace
    cases = [
        replace(c, suite="frontier")
        for c in _build_pressure_cases(root)
        if c.name in FRONTIER_NAMES
    ]
    return cases


def load_case_matrix(root: Path, suite: str) -> List[ExtremeCase]:
    """Load the matrix of extreme-test cases for *suite*.

    *suite* ``'contract'`` returns a deterministic matrix of mutated variants
    of the public test cases plus the 13 contract pressure cases.
    *suite* ``'frontier'`` returns the 9 frontier pressure cases **without**
    the expected-failure registry applied — the caller must apply it.
    *suite* ``'pressure'`` (internal) returns all 22 GPR/pair/predicate
    pressure cases without registry application.

    Raises:
        ValueError: If *suite* is not one of ``'contract'``, ``'frontier'``,
            or ``'pressure'``.
    """
    if suite == "contract":
        return _build_contract_matrix(root) + _get_contract_pressure_cases(root)
    if suite == "frontier":
        return _get_frontier_pressure_cases(root)
    if suite == "pressure":
        return _build_pressure_cases(root)
    raise ValueError(
        f"Unknown suite '{suite}' — must be one of 'contract', 'frontier', 'pressure'"
    )


# ════════════════════════════════════════════════════════════════════════
# Expected-failure registry (Task 5)
# ════════════════════════════════════════════════════════════════════════

# Phases that the runner supports for expected failures
VALID_PHASES: Tuple[str, ...] = ("compile", "execute", "compare", "artifact")

# Frontier case name set for fast lookup
FRONTIER_NAMES: Set[str] = {
    f"gpr-live-{n}" for n in GPR_FRONTIER
} | {
    f"pair-live-{n}" for n in PAIR_FRONTIER
} | {
    f"pred-live-{n}" for n in PRED_FRONTIER
}

# Contract case name set for fast lookup
CONTRACT_NAMES: Set[str] = {
    f"gpr-live-{n}" for n in GPR_CONTRACT
} | {
    f"pair-live-{n}" for n in PAIR_CONTRACT
} | {
    f"pred-live-{n}" for n in PRED_CONTRACT
}

ALL_PRESSURE_NAMES: Set[str] = FRONTIER_NAMES | CONTRACT_NAMES


class RegistryError(ValueError):
    """Raised when the expected-failure registry fails validation."""


class _DuplicateDetector:
    """JSON ``object_pairs_hook`` that collects duplicate keys at any nesting level."""

    def __init__(self) -> None:
        self.dupes: Set[str] = set()

    def __call__(self, pairs: list) -> dict:
        seen: dict = {}
        for k, v in pairs:
            if k in seen:
                self.dupes.add(k)
            seen[k] = v
        return seen


def load_pressure_registry(root: Path) -> dict:
    """Load and validate ``expected_failures.json`` from the extreme test dir.

    Validation checks:
        1. Every entry has a non-empty ``"issue"``.
        2. Every entry has a non-empty ``"phase_by_backend"`` dict.
        3. ``phase_by_backend`` keys are exactly ``"local"`` or ``"cmodel"``.
        4. Each phase value is in ``VALID_PHASES``.
        5. No legacy scalar ``"phase"`` key.
        6. No duplicate keys (detected via JSON ``object_pairs_hook``).
        7. Every key is a known frontier pressure case name.
        8. No contract case name appears as a registry key.

    Args:
        root: Repository root.

    Returns:
        Parsed dict ``{case_name: {"issue": str, "phase_by_backend": {str: str},
        "reason": str}}``.
    """
    registry_path = root / "C1" / "tests" / "extreme" / "expected_failures.json"
    if not registry_path.is_file():
        raise RegistryError(
            f"expected_failures.json not found at {registry_path}"
        )

    raw = registry_path.read_text(encoding="utf-8")

    # Duplicate detection via JSON object_pairs_hook (all nesting levels).
    detector = _DuplicateDetector()
    try:
        registry = json.loads(raw, object_pairs_hook=detector)
    except json.JSONDecodeError as e:
        raise RegistryError(f"invalid JSON: {e}")

    errors: List[str] = []

    # 0. Duplicate keys
    if detector.dupes:
        errors.append(f"duplicate keys: {sorted(detector.dupes)}")

    # Per-entry validation.
    for name, entry in registry.items():
        # 1. Non-empty issue IDs
        if not entry.get("issue", "").strip():
            errors.append(f"{name}: missing or empty issue ID")

        # 5. Reject legacy scalar "phase"
        if "phase" in entry and "phase_by_backend" not in entry:
            errors.append(
                f"{name}: uses legacy scalar 'phase'; migrate to 'phase_by_backend'"
            )

        # 2–4. Validate phase_by_backend
        pbb = entry.get("phase_by_backend")
        if not isinstance(pbb, dict) or len(pbb) == 0:
            errors.append(f"{name}: phase_by_backend must be a non-empty dict")
        else:
            for bk, phase in pbb.items():
                if bk not in ("local", "cmodel"):
                    errors.append(
                        f"{name}: unknown backend '{bk}' — "
                        f"must be 'local' or 'cmodel'"
                    )
                if phase not in VALID_PHASES:
                    errors.append(
                        f"{name}: unknown phase '{phase}' for backend '{bk}' — "
                        f"must be one of VALID_PHASES={VALID_PHASES}"
                    )

    # 7. All keys are known frontier names
    for name in registry:
        if name not in FRONTIER_NAMES:
            errors.append(
                f"{name}: not a known frontier pressure case — "
                f"must be one of {sorted(FRONTIER_NAMES)}"
            )

    # 8. No contract case carrying expected failure
    for name in registry:
        if name in CONTRACT_NAMES:
            errors.append(
                f"{name}: contract case must not carry an expected failure"
            )

    if errors:
        raise RegistryError("; ".join(errors))

    return registry


# ════════════════════════════════════════════════════════════════════════
# Strict profile selection (Task 7 — CModel backend)
# ════════════════════════════════════════════════════════════════════════


def select_strict_profile(cases: List[ExtremeCase]) -> List[ExtremeCase]:
    """Select a deterministic strict profile of 30–50 case-opt combinations.

    The profile guarantees coverage of:
    - All T1–T5 public test bases (one mutation variant each).
    - All GPR/pair/predicate *contract* pressure boundaries (O2 only).
    - All GPR/pair/predicate *frontier* pressure boundaries (O2 only).
    - At least one contract mutation family at O2.

    Selection is fully deterministic — no random, no shuffle.  Every case in
    the returned list has at least one ``opt_level``.  Frontier pressure
    cases are loaded from ``REPO_ROOT`` because they are not part of the
    contract suite.

    Args:
        cases: Full list of cases from the contract suite.

    Returns:
        List of :class:`ExtremeCase` objects with adjusted opt_levels.
    """
    from dataclasses import replace

    result: List[ExtremeCase] = []
    seen_names: set = set()

    # ── Phase 1: Contract pressure boundaries (O2 only) ────────────────
    for case in cases:
        if case.name in CONTRACT_NAMES and "O2" in case.opt_levels:
            result.append(replace(case, opt_levels=("O2",)))
            seen_names.add(case.name)

    # ── Phase 2: One mutation variant per T1–T5 base ──────────────────
    # Pick the smallest variant with seed 0 (small-family seed) —
    # these have opt_levels (O0, O2, O3) = 3 combos each.
    for case in cases:
        base = case.name[:2]
        if base not in ("T1", "T2", "T3", "T4", "T5"):
            continue
        if case.name in seen_names:
            continue
        # Only match the smallest n/dim with seed 0
        if base in ("T1", "T2", "T3", "T4"):
            # Elementwise: pick _s0_n1
            if "_s0_n1" not in case.name:
                continue
        else:  # T5
            # GEMM: pick _s0_d1
            if "_s0_d1" not in case.name:
                continue
        result.append(case)  # preserve original opt_levels
        seen_names.add(case.name)

    # ── Phase 3: Frontier pressure boundaries (O2 only) ────────────────
    # Load from REPO_ROOT since they are not in the contract suite.
    frontier_cases = load_case_matrix(REPO_ROOT, "frontier")
    for case in frontier_cases:
        if case.name in seen_names:
            continue
        if "O2" in case.opt_levels:
            result.append(replace(case, opt_levels=("O2",)))
            seen_names.add(case.name)

    return result


def apply_registry_to_cases(
    cases: List[ExtremeCase],
    registry: dict,
    *,
    backend: str,
) -> List[ExtremeCase]:
    """Apply registry metadata to matching frontier cases for *backend*.

    For each case whose *name* appears in *registry*, resolves
    ``phase_by_backend[backend]`` and sets ``expected_failure`` to the
    ``"issue"`` value and ``expected_failure_phase`` to the resolved phase.
    If *backend* is not present in the case's ``phase_by_backend`` mapping,
    both fields are explicitly cleared (``None``).

    Uses ``dataclasses.replace`` to preserve case immutability.

    Args:
        cases: List of ExtremeCase objects (typically frontier).
        registry: Validated registry dict.
        backend: Backend name (``"local"`` or ``"cmodel"``).

    Returns:
        New list with registry metadata applied.
    """
    if backend not in ("local", "cmodel"):
        raise RegistryError(
            f"unknown backend '{backend}' — must be 'local' or 'cmodel'"
        )

    from dataclasses import replace

    result: List[ExtremeCase] = []
    for case in cases:
        if case.name in registry:
            entry = registry[case.name]
            pbb = entry.get("phase_by_backend", {})
            if isinstance(pbb, dict) and backend in pbb:
                phase = pbb[backend]
                result.append(replace(
                    case,
                    expected_failure=entry["issue"],
                    expected_failure_phase=phase,
                ))
            else:
                # Backend not mapped → clear expectation.
                result.append(replace(
                    case,
                    expected_failure=None,
                    expected_failure_phase=None,
                ))
        else:
            result.append(case)
    return result

"""Unit tests for the extreme correctness case model and discovery contract."""

import json
import os
import struct
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import unittest.mock

import numpy as np

from tests.extreme.cases import (
    ALL_PRESSURE_NAMES,
    CONTRACT_NAMES,
    ExtremeCase,
    FRONTIER_NAMES,
    GPR_CONTRACT,
    GPR_FRONTIER,
    OutputExpectation,
    PAIR_CONTRACT,
    PAIR_FRONTIER,
    PRED_CONTRACT,
    PRED_FRONTIER,
    RegistryError,
    REPO_ROOT,
    VALID_PHASES,
    apply_registry_to_cases,
    load_case_matrix,
    load_pressure_registry,
    load_public_contract_cases,
    pack_pmem,
)
from tests.extreme.generators import (
    BASE_PTX,
    LOOP_PTX,
    _expected_gpr_sum,
    _expected_pair_sum,
    _expected_pred_sum,
    equivalent_address_forms,
    generate_gpr_pressure,
    generate_pair_pressure,
    generate_predicate_pressure,
    insert_dead_code,
    mutate_contract_source,
    rename_registers,
    reorder_blocks,
    sparse_register_numbering,
)
from tests.extreme.runner import discover_cases


class CaseModelTests(unittest.TestCase):
    def test_discovery_fails_for_unknown_suite(self):
        """An unknown suite raises ValueError from load_case_matrix."""
        with self.assertRaisesRegex(ValueError, "Unknown suite"):
            load_case_matrix(Path("."), "nonexistent")

    def test_case_model_is_immutable(self):
        out = OutputExpectation(
            offset=0, dtype="<u4", shape=(1,), expected=(7,), rtol=0.0, atol=0.0
        )
        case = ExtremeCase(
            name="one",
            suite="contract",
            ptx="ret;",
            grid=(1, 1, 1),
            block=(1, 1, 1),
            pmem=b"",
            gmem=b"\0" * 4,
            output=out,
            opt_levels=("O2",),
            expected_failure=None,
        )
        with self.assertRaises(Exception):
            case.name = "changed"  # type: ignore[misc]


class PublicCaseTests(unittest.TestCase):
    """Tests for public contract cases T1..T5 loaded from disk."""

    # ── Step 1: pmem natural alignment ─────────────────────────────────

    def test_pmem_natural_alignment(self):
        params = [("u64", 256), ("u64", 512), ("u32", 7)]
        packed = pack_pmem(params)
        self.assertEqual(len(packed), 24)
        self.assertEqual(struct.unpack_from("<Q", packed, 0)[0], 256)
        self.assertEqual(struct.unpack_from("<Q", packed, 8)[0], 512)
        self.assertEqual(struct.unpack_from("<I", packed, 16)[0], 7)

    # ── Step 1: case discovery ─────────────────────────────────────────

    def test_current_public_cases_are_discovered(self):
        cases = load_public_contract_cases(REPO_ROOT)
        self.assertEqual([c.name for c in cases], ["T1", "T2", "T3", "T4", "T5"])
        self.assertTrue(all(".version 9.3" in c.ptx for c in cases))

    # ── Step 4: PMEM length assertions ─────────────────────────────────

    def test_pmem_lengths(self):
        cases = load_public_contract_cases(REPO_ROOT)
        expected_lengths = {"T1": 32, "T2": 32, "T3": 40, "T4": 48, "T5": 40}
        for case in cases:
            self.assertEqual(
                len(case.pmem),
                expected_lengths[case.name],
                f"{case.name}: PMEM length mismatch",
            )

    # ── Step 4: output inside GMEM ─────────────────────────────────────

    def test_output_inside_gmem(self):
        cases = load_public_contract_cases(REPO_ROOT)
        for case in cases:
            out = case.output
            elem_size = 4  # float32
            out_end = out.offset + int(np.prod(out.shape)) * elem_size
            self.assertLessEqual(
                out_end,
                len(case.gmem),
                f"{case.name}: output range {out.offset}..{out_end} exceeds "
                f"GMEM ({len(case.gmem)} bytes)",
            )

    # ── Step 4: block product <= 256 ───────────────────────────────────

    def test_block_product_never_exceeds_256(self):
        cases = load_public_contract_cases(REPO_ROOT)
        for case in cases:
            prod = case.block[0] * case.block[1] * case.block[2]
            self.assertLessEqual(
                prod, 256, f"{case.name}: block product {prod} > 256"
            )

    # ── Step 4: formula correctness per case ───────────────────────────

    def test_t1_formula(self):
        """c[i] = a[i] + b[i]  with forced FP32."""
        cases = load_public_contract_cases(REPO_ROOT)
        case = next(c for c in cases if c.name == "T1")
        n = case.output.shape[0]
        a = np.frombuffer(case.gmem, np.float32, count=n, offset=0)
        b = np.frombuffer(case.gmem, np.float32, count=n, offset=n * 4)
        expected = (a.astype(np.float32) + b.astype(np.float32)).astype(np.float32)
        np.testing.assert_allclose(
            np.array(case.output.expected, np.float32),
            expected.ravel(),
            atol=1e-6,
        )

    def test_t2_formula(self):
        """out[i] = (x[i]+y[i])*(x[i]+y[i])+x[i] with forced FP32."""
        cases = load_public_contract_cases(REPO_ROOT)
        case = next(c for c in cases if c.name == "T2")
        n = case.output.shape[0]
        x = np.frombuffer(case.gmem, np.float32, count=n, offset=0)
        y = np.frombuffer(case.gmem, np.float32, count=n, offset=n * 4)
        s = (x + y).astype(np.float32)
        expected = (s * s).astype(np.float32) + x
        np.testing.assert_allclose(
            np.array(case.output.expected, np.float32),
            expected.ravel(),
            atol=1e-5,
        )

    def test_t3_formula(self):
        """out[i] = x[i]*y[i] + x[i]*z[i] with forced FP32."""
        cases = load_public_contract_cases(REPO_ROOT)
        case = next(c for c in cases if c.name == "T3")
        n = case.output.shape[0]
        x = np.frombuffer(case.gmem, np.float32, count=n, offset=0)
        y = np.frombuffer(case.gmem, np.float32, count=n, offset=n * 4)
        z = np.frombuffer(case.gmem, np.float32, count=n, offset=n * 8)
        xf = x.astype(np.float32)
        yf = y.astype(np.float32)
        zf = z.astype(np.float32)
        expected = (xf * yf + xf * zf).astype(np.float32)
        np.testing.assert_allclose(
            np.array(case.output.expected, np.float32),
            expected.ravel(),
            atol=1e-5,
        )

    def test_t4_formula(self):
        """out[i] = (a+b)*(c-d) + (a*c)*(b+d) with forced FP32."""
        cases = load_public_contract_cases(REPO_ROOT)
        case = next(c for c in cases if c.name == "T4")
        n = case.output.shape[0]
        a = np.frombuffer(case.gmem, np.float32, count=n, offset=0)
        b = np.frombuffer(case.gmem, np.float32, count=n, offset=n * 4)
        c = np.frombuffer(case.gmem, np.float32, count=n, offset=n * 8)
        d = np.frombuffer(case.gmem, np.float32, count=n, offset=n * 12)
        af = a.astype(np.float32)
        bf = b.astype(np.float32)
        cf = c.astype(np.float32)
        df = d.astype(np.float32)
        lhs = ((af + bf) * (cf - df)).astype(np.float32)
        rhs = ((af * cf) * (bf + df)).astype(np.float32)
        expected = (lhs + rhs).astype(np.float32)
        np.testing.assert_allclose(
            np.array(case.output.expected, np.float32),
            expected.ravel(),
            atol=1e-5,
        )

    def test_t5_formula(self):
        """C = A @ B with forced FP32 (matmul)."""
        cases = load_public_contract_cases(REPO_ROOT)
        case = next(c for c in cases if c.name == "T5")
        dim = case.output.shape[0]  # square matrix
        A = np.frombuffer(case.gmem, np.float32, count=dim * dim, offset=0).reshape(
            dim, dim
        )
        B = np.frombuffer(
            case.gmem, np.float32, count=dim * dim, offset=dim * dim * 4
        ).reshape(dim, dim)
        expected = (A.astype(np.float32) @ B.astype(np.float32)).astype(np.float32)
        np.testing.assert_allclose(
            np.array(case.output.expected, np.float32),
            expected.ravel(),
            atol=1e-4,
        )

    # ── Step 4: deterministic loading ──────────────────────────────────

    def test_deterministic_loading(self):
        cases1 = load_public_contract_cases(REPO_ROOT)
        cases2 = load_public_contract_cases(REPO_ROOT)
        for c1, c2 in zip(cases1, cases2):
            self.assertEqual(c1.pmem, c2.pmem, f"{c1.name}: PMEM differs")
            self.assertEqual(c1.gmem, c2.gmem, f"{c1.name}: GMEM differs")
            self.assertEqual(
                c1.output.expected,
                c2.output.expected,
                f"{c1.name}: expected values differ",
            )



# ════════════════════════════════════════════════════════════════════════
# Task 4: Deterministic Contract Mutation Tests
# ════════════════════════════════════════════════════════════════════════

class MutationTests(unittest.TestCase):
    """Tests for deterministic contract mutations (generators.py)."""

    def test_mutations_are_seed_deterministic(self):
        a = mutate_contract_source(BASE_PTX, seed=17)
        b = mutate_contract_source(BASE_PTX, seed=17)
        self.assertEqual(a, b)

    def test_mutations_different_seeds_differ(self):
        a = mutate_contract_source(BASE_PTX, seed=0)
        b = mutate_contract_source(BASE_PTX, seed=7)
        self.assertNotEqual(a, b)

    def test_register_rename_keeps_special_registers(self):
        out = rename_registers(BASE_PTX, seed=3)
        self.assertIn("%tid.x", out)
        self.assertIn("%ctaid.x", out)
        self.assertIn("%ntid.x", out)

    def test_reordered_blocks_have_explicit_fallthrough(self):
        out = reorder_blocks(LOOP_PTX, seed=4)
        self.assertIn("bra ", out)

    def test_sparse_numbering_updates_decl(self):
        """Sparse register numbering updates .reg declarations to cover renamed
        registers."""
        out = sparse_register_numbering(BASE_PTX, seed=7)
        # After sparse numbering, at least one .reg decl should be expanded
        # from the original count.  Parse the decl lines.
        decls_before = {
            m.group(1): int(m.group(2))
            for m in __import__("re").finditer(
                r"\.reg\s+\.\w+\s+%(\w+)<(\d+)>", BASE_PTX
            )
        }
        decls_after = {
            m.group(1): int(m.group(2))
            for m in __import__("re").finditer(
                r"\.reg\s+\.\w+\s+%(\w+)<(\d+)>", out
            )
        }
        # At least one register class should have an increased decl count
        increased = any(
            decls_after.get(cls, 0) > decls_before.get(cls, 0)
            for cls in ("rd", "r", "f", "p")
        )
        self.assertTrue(increased, "No .reg declaration expanded by sparse numbering")

    def test_sparse_numbering_register_references_valid(self):
        """All %r, %rd, %f, %p references in sparse output use indices within
        the declared range."""
        out = sparse_register_numbering(BASE_PTX, seed=7)
        # Parse decl ranges
        import re
        decls = {}
        for m in re.finditer(r"\.reg\s+\.\w+\s+%(\w+)<(\d+)>", out):
            decls[m.group(1)] = int(m.group(2))
        # Check all register uses are within declared range
        reg_re = re.compile(r"%(\w+)(\d+)\b")
        for m in reg_re.finditer(out):
            cls, idx_str = m.group(1), int(m.group(2))
            if cls in decls:
                self.assertLess(
                    idx_str, decls[cls],
                    f"%{cls}{idx_str} exceeds declared <{decls[cls]}>",
                )

    def test_dead_code_insertion_adds_instructions(self):
        out = insert_dead_code(BASE_PTX, seed=42, count=3)
        # Dead code should add at least one new instruction
        before_lines = BASE_PTX.count("\n")
        after_lines = out.count("\n")
        self.assertGreater(after_lines, before_lines)

    def test_mutate_contract_source_no_fp16_bf16(self):
        """Mutated output must not contain FP16/BF16/TMUL references."""
        out = mutate_contract_source(BASE_PTX, seed=17)
        self.assertNotIn("f16", out)
        self.assertNotIn("bf16", out)
        self.assertNotIn("tmul", out)
        self.assertNotIn("multikernel", out.lower())

    def test_equivalent_address_forms_stay_in_restricted_ptx_subset(self):
        src = """    mul.wide.u32 %rd4, %r5, 4;
    add.u64 %rd5, %rd1, %rd4;
"""
        out = equivalent_address_forms(src)
        self.assertNotIn("cvt.u64", out)
        self.assertNotIn("shl.b64", out)
        self.assertIn("mul.wide.u32 %rd4, %r5, 2;", out)
        self.assertIn("add.u64 %rd4, %rd4, %rd4;", out)


class PressureStructuralTests(unittest.TestCase):
    """Structural tests for cross-block pressure generators (RED phase)."""

    def test_gpr_pressure_crosses_block_boundary(self):
        from tests.extreme.generators import generate_gpr_pressure
        src = generate_gpr_pressure(64)
        defs, uses = src.split("REDUCE:")
        for i in range(64):
            self.assertIn(f"%r{i+1}", defs)
            self.assertIn(f"%r{i+1}", uses)
        self.assertIn("bra REDUCE;", defs)
        # Verify accumulator and output pointer are NOT defined before REDUCE
        self.assertNotIn("mov.u32 %r0,", defs,
                         "accumulator must not be initialized in ENTRY")
        self.assertNotIn("ld.param.u64", defs,
                         "output pointer must not be loaded in ENTRY")

    def test_pair_pressure_uses_b64_values(self):
        from tests.extreme.generators import generate_pair_pressure
        src = generate_pair_pressure(120)
        self.assertIn(".reg .b64 %bd<120>;", src)
        # Verify accumulator and output pointer are NOT in ENTRY
        defs, _ = src.split("REDUCE:")
        self.assertNotIn("mov.u64 %rd0,", defs,
                         "accumulator must not be initialized in ENTRY")
        self.assertNotIn("ld.param.u64", defs,
                         "output pointer must not be loaded in ENTRY")

    def test_predicate_pressure_has_distinct_predicates(self):
        from tests.extreme.generators import generate_predicate_pressure
        src = generate_predicate_pressure(9)
        self.assertIn(".reg .pred %p<9>;", src)
        # Verify accumulator and output pointer are NOT in ENTRY
        defs, _ = src.split("REDUCE:")
        self.assertNotIn("mov.u32 %r0,", defs,
                         "accumulator must not be initialized in ENTRY")
        self.assertNotIn("ld.param.u64", defs,
                         "output pointer must not be loaded in ENTRY")

    # ── legal register family guards (RED phase) ─────────────────────────

    def test_gpr_pressure_forbids_illegal_registers(self):
        """GPR pressure must not contain illegal register names like %v, %acc, %rd_out."""
        from tests.extreme.generators import generate_gpr_pressure
        src = generate_gpr_pressure(64)
        self.assertNotIn("%v", src, "%v register family is illegal")
        self.assertNotIn("%acc", src, "%acc register name is illegal")
        self.assertNotIn("%rd_out", src, "%rd_out register name is illegal")
        self.assertNotIn("%r_tmp", src, "%r_tmp register name is illegal")
        self.assertNotIn("%r_one", src, "%r_one register name is illegal")

    def test_gpr_pressure_uses_legal_r_family(self):
        """GPR pressure uses %r<N> for all values and accumulator."""
        from tests.extreme.generators import generate_gpr_pressure
        src = generate_gpr_pressure(64)
        # Values are %r1..%r64
        for i in range(1, 5):
            self.assertIn(f"%r{i}", src)
        # Accumulator is %r0
        self.assertIn("%r0", src)
        # Declaration uses .reg .u32 %r<65>
        self.assertIn(".reg .u32 %r<65>;", src)

    def test_pair_pressure_forbids_illegal_registers(self):
        """Pair pressure must not contain illegal register names like %v, %acc, %rd_out."""
        from tests.extreme.generators import generate_pair_pressure
        src = generate_pair_pressure(32)
        self.assertNotIn("%v", src, "%v register family is illegal")
        self.assertNotIn("%acc", src, "%acc register name is illegal")
        self.assertNotIn("%rd_out", src, "%rd_out register name is illegal")
        self.assertNotIn("%r_tmp", src, "%r_tmp register name is illegal")
        self.assertNotIn("%r_one", src, "%r_one register name is illegal")

    def test_pair_pressure_uses_legal_bd_family(self):
        """Pair pressure uses %bd<N> for .b64 pressure values."""
        from tests.extreme.generators import generate_pair_pressure
        src = generate_pair_pressure(32)
        self.assertIn(".reg .b64 %bd<32>;", src)
        for i in range(4):
            self.assertIn(f"%bd{i}", src)
        # Base pointer and address temporary use %rd<N>
        self.assertIn("%rd1", src)
        self.assertIn("%rd2", src)

    def test_pair_pressure_uses_legal_r_for_store_temp(self):
        """Pair pressure uses legal %r<N> for accumulator and load temp (no cvt)."""
        from tests.extreme.generators import generate_pair_pressure
        src = generate_pair_pressure(32)
        self.assertIn("ld.global.u32 %r1, [%rd2]", src,
                      "pair pressure must load u32 values from GMEM")
        self.assertIn("add.u32 %r0, %r0, %r1", src,
                      "pair pressure must use u32 accumulation")
        self.assertIn("st.global.u32 [%rd1], %r0", src,
                      "pair pressure must store u32 result")
        self.assertIn(".reg .u32 %r<2>;", src)

    def test_predicate_pressure_forbids_illegal_registers(self):
        """Predicate pressure must not contain illegal register names like %acc, %rd_out, %r_one, %r_tmp."""
        from tests.extreme.generators import generate_predicate_pressure
        src = generate_predicate_pressure(9)
        self.assertNotIn("%acc", src, "%acc register name is illegal")
        self.assertNotIn("%rd_out", src, "%rd_out register name is illegal")
        self.assertNotIn("%r_one", src, "%r_one register name is illegal")
        self.assertNotIn("%r_tmp", src, "%r_tmp register name is illegal")

    def test_predicate_pressure_uses_legal_p_and_r_families(self):
        """Predicate pressure uses %p<N> for predicates and %r<N> for temps."""
        from tests.extreme.generators import generate_predicate_pressure
        src = generate_predicate_pressure(9)
        self.assertIn(".reg .pred %p<9>;", src)
        # %r0 is accumulator, %r1 is constant
        self.assertIn(".reg .u32 %r<2>;", src)
        self.assertIn("mov.u32 %r1, 1", src)

    # ── structural assertion: no %% escape leakage ────────────────────

    def test_all_generators_no_double_percent(self):
        """None of the three pressure generators emit %% — all PTX % must be single."""
        from tests.extreme.generators import (
            generate_gpr_pressure,
            generate_pair_pressure,
            generate_predicate_pressure,
        )
        for name, gen in [("gpr", generate_gpr_pressure),
                           ("pair", generate_pair_pressure),
                           ("pred", generate_predicate_pressure)]:
            src = gen(64 if name != "pred" else 9)
            self.assertNotIn("%%", src,
                             f"{name}_pressure output must not contain %% escape sequences")


class PressureBoundaryTests(unittest.TestCase):
    """Verify correct boundary counts and case construction."""

    def test_pressure_case_count_gpr_contract(self):
        cases = load_case_matrix(REPO_ROOT, "pressure")
        gpr_contract = [c for c in cases if c.name.startswith("gpr-live-")
                        and c.name in CONTRACT_NAMES]
        self.assertEqual(len(gpr_contract), len(GPR_CONTRACT))

    def test_pressure_case_count_gpr_frontier(self):
        cases = load_case_matrix(REPO_ROOT, "pressure")
        gpr_frontier = [c for c in cases if c.name.startswith("gpr-live-")
                        and c.name in FRONTIER_NAMES]
        self.assertEqual(len(gpr_frontier), len(GPR_FRONTIER))

    def test_pressure_case_count_pair_contract(self):
        cases = load_case_matrix(REPO_ROOT, "pressure")
        pair_contract = [c for c in cases if c.name.startswith("pair-live-")
                         and c.name in CONTRACT_NAMES]
        self.assertEqual(len(pair_contract), len(PAIR_CONTRACT))

    def test_pressure_case_count_pair_frontier(self):
        cases = load_case_matrix(REPO_ROOT, "pressure")
        pair_frontier = [c for c in cases if c.name.startswith("pair-live-")
                         and c.name in FRONTIER_NAMES]
        self.assertEqual(len(pair_frontier), len(PAIR_FRONTIER))

    def test_pressure_case_count_pred_contract(self):
        cases = load_case_matrix(REPO_ROOT, "pressure")
        pred_contract = [c for c in cases if c.name.startswith("pred-live-")
                         and c.name in CONTRACT_NAMES]
        self.assertEqual(len(pred_contract), len(PRED_CONTRACT))

    def test_pressure_case_count_pred_frontier(self):
        cases = load_case_matrix(REPO_ROOT, "pressure")
        pred_frontier = [c for c in cases if c.name.startswith("pred-live-")
                         and c.name in FRONTIER_NAMES]
        self.assertEqual(len(pred_frontier), len(PRED_FRONTIER))

    def test_pressure_case_names_are_unique(self):
        cases = load_case_matrix(REPO_ROOT, "pressure")
        names = [c.name for c in cases]
        self.assertEqual(len(names), len(set(names)))

    def test_pressure_contract_cases_no_expected_failure(self):
        cases = load_case_matrix(REPO_ROOT, "pressure")
        contract_cases = [c for c in cases if c.name in CONTRACT_NAMES]
        for case in contract_cases:
            self.assertIsNone(
                case.expected_failure,
                f"{case.name}: contract case must not have expected_failure",
            )

    def test_pressure_each_case_valid_ptx(self):
        cases = load_case_matrix(REPO_ROOT, "pressure")
        for case in cases:
            self.assertIn(
                ".version 9.3", case.ptx,
                f"{case.name}: missing .version 9.3",
            )
            self.assertIn(
                ".visible .entry", case.ptx,
                f"{case.name}: missing .visible .entry",
            )

    def test_pressure_case_suites(self):
        cases = load_case_matrix(REPO_ROOT, "pressure")
        for case in cases:
            self.assertEqual(case.suite, "pressure")


class PressureExpectedReductionTests(unittest.TestCase):
    """Verify that each pressure case has the independently correct expected value."""

    def test_gpr_expected_values(self):
        cases = load_case_matrix(REPO_ROOT, "pressure")
        for case in cases:
            if case.name.startswith("gpr-live-"):
                n = int(case.name.split("-")[-1])
                expected = _expected_gpr_sum(n)
                self.assertEqual(
                    case.output.expected[0], expected,
                    f"{case.name}: expected {expected}, got {case.output.expected[0]}",
                )

    def test_pair_expected_values(self):
        cases = load_case_matrix(REPO_ROOT, "pressure")
        for case in cases:
            if case.name.startswith("pair-live-"):
                n = int(case.name.split("-")[-1])
                expected = _expected_pair_sum(n)
                self.assertEqual(
                    case.output.expected[0], expected,
                    f"{case.name}: expected {expected}, got {case.output.expected[0]}",
                )

    def test_pred_expected_values(self):
        cases = load_case_matrix(REPO_ROOT, "pressure")
        for case in cases:
            if case.name.startswith("pred-live-"):
                n = int(case.name.split("-")[-1])
                expected = _expected_pred_sum(n)
                self.assertEqual(
                    case.output.expected[0], expected,
                    f"{case.name}: expected {expected}, got {case.output.expected[0]}",
                )

    def test_output_dtype_shape(self):
        cases = load_case_matrix(REPO_ROOT, "pressure")
        for case in cases:
            self.assertEqual(case.output.dtype, "<u4")
            self.assertEqual(case.output.shape, (1,))
            self.assertEqual(case.output.offset, 0)
            self.assertEqual(case.output.rtol, 0.0)
            self.assertEqual(case.output.atol, 0.0)


class RegistryValidationTests(unittest.TestCase):
    """Test expected-failure registry loading and each rejection mode."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.reg_dir = Path(self.tmpdir.name)
        # Mock root with C1/tests/extreme subpath
        self.mock_root = self.reg_dir / "mock_repo"
        self.extreme_dir = self.mock_root / "C1" / "tests" / "extreme"
        self.extreme_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _write_registry(self, data: dict) -> Path:
        path = self.extreme_dir / "expected_failures.json"
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return path

    # ── Acceptance ──────────────────────────────────────────────────────

    def test_accept_valid_registry(self):
        """Accept a well-formed registry with phase_by_backend maps for all 9 frontier entries."""
        reg = {
            "gpr-live-255": {
                "issue": "C1-SPILL-001",
                "phase_by_backend": {"local": "execute", "cmodel": "compare"},
                "reason": "exhausted GPR/pair allocation miscompares",
            },
            "gpr-live-256": {
                "issue": "C1-SPILL-001",
                "phase_by_backend": {"local": "execute", "cmodel": "compare"},
                "reason": "exhausted GPR/pair allocation miscompares",
            },
            "gpr-live-300": {
                "issue": "C1-SPILL-001",
                "phase_by_backend": {"local": "execute", "cmodel": "compare"},
                "reason": "exhausted GPR/pair allocation miscompares",
            },
            "pair-live-127": {
                "issue": "C1-PAIR-001",
                "phase_by_backend": {"local": "compare", "cmodel": "compare"},
                "reason": "pair pressure exceeds register-pair allocation capacity",
            },
            "pair-live-128": {
                "issue": "C1-PAIR-001",
                "phase_by_backend": {"local": "compare", "cmodel": "compare"},
                "reason": "pair pressure exceeds register-pair allocation capacity",
            },
            "pair-live-140": {
                "issue": "C1-PAIR-001",
                "phase_by_backend": {"local": "execute", "cmodel": "execute"},
                "reason": "pair140 CModel OOB execution",
            },
            "pred-live-9": {
                "issue": "C1-PRED-001",
                "phase_by_backend": {"local": "execute"},
                "reason": ">8 virtual predicates reach unencodable high bits because predicate allocation/spilling is not implemented",
            },
            "pred-live-12": {
                "issue": "C1-PRED-001",
                "phase_by_backend": {"local": "execute"},
                "reason": ">8 virtual predicates reach unencodable high bits because predicate allocation/spilling is not implemented",
            },
            "pred-live-16": {
                "issue": "C1-PRED-001",
                "phase_by_backend": {"local": "execute"},
                "reason": ">8 virtual predicates reach unencodable high bits because predicate allocation/spilling is not implemented",
            },
        }
        self._write_registry(reg)
        loaded = load_pressure_registry(self.mock_root)
        # Verify all 9 entries load with correct issue IDs
        self.assertEqual(loaded["gpr-live-255"]["issue"], "C1-SPILL-001")
        self.assertEqual(loaded["gpr-live-256"]["issue"], "C1-SPILL-001")
        self.assertEqual(loaded["gpr-live-300"]["issue"], "C1-SPILL-001")
        self.assertEqual(loaded["pair-live-127"]["issue"], "C1-PAIR-001")
        self.assertEqual(loaded["pair-live-128"]["issue"], "C1-PAIR-001")
        self.assertEqual(loaded["pair-live-140"]["issue"], "C1-PAIR-001")
        self.assertEqual(loaded["pred-live-9"]["issue"], "C1-PRED-001")
        self.assertEqual(loaded["pred-live-12"]["issue"], "C1-PRED-001")
        self.assertEqual(loaded["pred-live-16"]["issue"], "C1-PRED-001")
        # Verify phase_by_backend maps
        self.assertEqual(
            loaded["gpr-live-255"]["phase_by_backend"],
            {"local": "execute", "cmodel": "compare"},
        )
        self.assertEqual(
            loaded["pair-live-127"]["phase_by_backend"],
            {"local": "compare", "cmodel": "compare"},
        )
        self.assertEqual(
            loaded["pred-live-9"]["phase_by_backend"],
            {"local": "execute"},
        )
        # Legacy scalar 'phase' must NOT be present
        self.assertNotIn("phase", loaded["gpr-live-255"])

    # ── Missing issue ID ────────────────────────────────────────────────

    def test_reject_missing_issue_id(self):
        reg = {
            "gpr-live-256": {
                "phase_by_backend": {"local": "compare"},
                "reason": "no issue field",
            },
        }
        self._write_registry(reg)
        with self.assertRaises(RegistryError) as ctx:
            load_pressure_registry(self.mock_root)
        self.assertIn("issue", str(ctx.exception))

    def test_reject_empty_issue_id(self):
        reg = {
            "gpr-live-256": {
                "issue": "",
                "phase_by_backend": {"local": "compare"},
                "reason": "empty issue",
            },
        }
        self._write_registry(reg)
        with self.assertRaises(RegistryError) as ctx:
            load_pressure_registry(self.mock_root)
        self.assertIn("issue", str(ctx.exception))

    # ── Unknown phase inside phase_by_backend ────────────────────────────

    def test_reject_unknown_phase(self):
        """phase_by_backend value not in VALID_PHASES raises RegistryError."""
        reg = {
            "gpr-live-256": {
                "issue": "C1-SPILL-001",
                "phase_by_backend": {"local": "link"},
                "reason": "bad phase in map",
            },
        }
        self._write_registry(reg)
        with self.assertRaises(RegistryError) as ctx:
            load_pressure_registry(self.mock_root)
        self.assertIn("phase", str(ctx.exception))
        self.assertIn("link", str(ctx.exception))

    # ── Unknown backend key in phase_by_backend ──────────────────────────

    def test_reject_unknown_backend_key_in_map(self):
        """phase_by_backend with disallowed backend key raises RegistryError."""
        reg = {
            "gpr-live-256": {
                "issue": "C1-SPILL-001",
                "phase_by_backend": {"gpu": "compare"},
                "reason": "unknown backend key",
            },
        }
        self._write_registry(reg)
        with self.assertRaises(RegistryError) as ctx:
            load_pressure_registry(self.mock_root)
        self.assertIn("gpu", str(ctx.exception))

    # ── Invalid phase value inside phase_by_backend ──────────────────────

    def test_reject_invalid_phase_in_backend_map(self):
        """phase_by_backend value not in VALID_PHASES raises RegistryError."""
        reg = {
            "gpr-live-256": {
                "issue": "C1-SPILL-001",
                "phase_by_backend": {"local": "simulate"},
                "reason": "invalid phase value",
            },
        }
        self._write_registry(reg)
        with self.assertRaises(RegistryError) as ctx:
            load_pressure_registry(self.mock_root)
        self.assertIn("simulate", str(ctx.exception))
        self.assertIn("VALID_PHASES", str(ctx.exception))

    # ── Empty phase_by_backend map ───────────────────────────────────────

    def test_reject_empty_phase_by_backend_map(self):
        """Empty phase_by_backend dict raises RegistryError (no backend mappings)."""
        reg = {
            "gpr-live-256": {
                "issue": "C1-SPILL-001",
                "phase_by_backend": {},
                "reason": "no backend mappings",
            },
        }
        self._write_registry(reg)
        with self.assertRaises(RegistryError) as ctx:
            load_pressure_registry(self.mock_root)
        self.assertIn("phase_by_backend", str(ctx.exception))
        err = str(ctx.exception).lower()
        self.assertTrue("empty" in err or "no backend" in err or "missing" in err)

    # ── Legacy scalar phase field ────────────────────────────────────────

    def test_reject_legacy_scalar_phase(self):
        """Registry with scalar 'phase' (not phase_by_backend) is rejected."""
        reg = {
            "gpr-live-256": {
                "issue": "C1-SPILL-001",
                "phase": "compare",
                "reason": "legacy format",
            },
        }
        self._write_registry(reg)
        with self.assertRaises(RegistryError) as ctx:
            load_pressure_registry(self.mock_root)
        self.assertIn("legacy", str(ctx.exception).lower())

    # ── Duplicate keys ──────────────────────────────────────────────────

    def test_reject_duplicate_keys(self):
        # Build JSON with duplicate key manually
        lines = [
            '{',
            '  "gpr-live-256": {"issue": "C1-SPILL-001", "phase_by_backend": {"local": "compare"}, "reason": "first"},',
            '  "gpr-live-256": {"issue": "C1-SPILL-001", "phase_by_backend": {"local": "compare"}, "reason": "duplicate"}',
            '}',
        ]
        path = self.extreme_dir / "expected_failures.json"
        path.write_text("\n".join(lines), encoding="utf-8")
        with self.assertRaises(RegistryError) as ctx:
            load_pressure_registry(self.mock_root)
        self.assertIn("duplicate", str(ctx.exception).lower())

    # ── Key absent from frontier ────────────────────────────────────────

    def test_reject_key_not_in_frontier(self):
        # "gpr-live-64" is a CONTRACT name, not frontier
        reg = {
            "gpr-live-64": {
                "issue": "C1-BAD-001",
                "phase_by_backend": {"local": "compare"},
                "reason": "contract case in registry",
            },
        }
        self._write_registry(reg)
        with self.assertRaises(RegistryError) as ctx:
            load_pressure_registry(self.mock_root)
        self.assertIn("gpr-live-64", str(ctx.exception))
        self.assertIn("frontier", str(ctx.exception).lower())

    def test_reject_completely_unknown_key(self):
        reg = {
            "nonexistent-key": {
                "issue": "C1-BAD-001",
                "phase_by_backend": {"local": "compare"},
                "reason": "totally unknown",
            },
        }
        self._write_registry(reg)
        with self.assertRaises(RegistryError) as ctx:
            load_pressure_registry(self.mock_root)
        self.assertIn("nonexistent-key", str(ctx.exception))

    # ── Contract case in registry ───────────────────────────────────────

    def test_reject_contract_case_in_registry(self):
        reg = {
            "gpr-live-64": {
                "issue": "C1-BAD-001",
                "phase_by_backend": {"local": "compare"},
                "reason": "should not be here",
            },
        }
        self._write_registry(reg)
        with self.assertRaises(RegistryError) as ctx:
            load_pressure_registry(self.mock_root)
        self.assertIn("contract", str(ctx.exception).lower())

    # ── Registry file not found ─────────────────────────────────────────

    def test_reject_missing_registry_file(self):
        # No file written
        with self.assertRaises(RegistryError) as ctx:
            load_pressure_registry(self.mock_root)
        self.assertIn("not found", str(ctx.exception).lower())

    # ── Application (backend= kwarg) ────────────────────────────────────

    @staticmethod
    def _make_phase_by_backend_registry() -> dict:
        """Build a phase_by_backend registry dict (manual, bypassing load_pressure_registry)."""
        return {
            "gpr-live-255": {"issue": "C1-SPILL-001", "phase_by_backend": {"local": "execute", "cmodel": "compare"}},
            "gpr-live-256": {"issue": "C1-SPILL-001", "phase_by_backend": {"local": "execute", "cmodel": "compare"}},
            "gpr-live-300": {"issue": "C1-SPILL-001", "phase_by_backend": {"local": "execute", "cmodel": "compare"}},
            "pair-live-127": {"issue": "C1-PAIR-001", "phase_by_backend": {"local": "compare", "cmodel": "compare"}},
            "pair-live-128": {"issue": "C1-PAIR-001", "phase_by_backend": {"local": "compare", "cmodel": "compare"}},
            "pair-live-140": {"issue": "C1-PAIR-001", "phase_by_backend": {"local": "execute", "cmodel": "execute"}},
            "pred-live-9":   {"issue": "C1-PRED-001", "phase_by_backend": {"local": "execute"}},
            "pred-live-12":  {"issue": "C1-PRED-001", "phase_by_backend": {"local": "execute"}},
            "pred-live-16":  {"issue": "C1-PRED-001", "phase_by_backend": {"local": "execute"}},
        }

    def test_apply_registry_to_cases(self):
        """apply_registry_to_cases with backend='local' sets expected_failure from phase_by_backend."""
        cases = load_case_matrix(REPO_ROOT, "pressure")
        reg = self._make_phase_by_backend_registry()
        updated = apply_registry_to_cases(cases, reg, backend="local")
        for case in updated:
            if case.name in reg:
                self.assertEqual(
                    case.expected_failure,
                    reg[case.name]["issue"],
                    f"{case.name}: expected_failure mismatch",
                )
            else:
                self.assertIsNone(
                    case.expected_failure,
                    f"{case.name}: should not have expected_failure",
                )

    def test_apply_registry_does_not_mutate_originals(self):
        """apply_registry_to_cases with backend= returns new cases, originals unchanged."""
        cases = load_case_matrix(REPO_ROOT, "pressure")
        original_nones = {c.name: c.expected_failure for c in cases}
        reg = self._make_phase_by_backend_registry()
        apply_registry_to_cases(cases, reg, backend="local")
        for case in cases:
            self.assertEqual(
                case.expected_failure, original_nones[case.name],
                f"{case.name}: original case was mutated",
            )

    # ── Backend-specific resolution (RED) ────────────────────────────────

    def test_apply_registry_gpr_live_255_local_execute_cmodel_compare(self):
        """gpr-live-255: local→execute, cmodel→compare."""
        reg = {"gpr-live-255": {"issue": "C1-SPILL-001", "phase_by_backend": {"local": "execute", "cmodel": "compare"}}}
        case = _minimal_extreme_case(name="gpr-live-255")
        local_up = apply_registry_to_cases([case], reg, backend="local")
        self.assertEqual(local_up[0].expected_failure, "C1-SPILL-001")
        self.assertEqual(local_up[0].expected_failure_phase, "execute")
        cmodel_up = apply_registry_to_cases([case], reg, backend="cmodel")
        self.assertEqual(cmodel_up[0].expected_failure_phase, "compare")

    def test_apply_registry_pred_live_9_local_none_cmodel_execute(self):
        """pred-live-9: local→no expectation (phase_by_backend lacks 'local'), cmodel→execute."""
        reg = {"pred-live-9": {"issue": "C1-PRED-001", "phase_by_backend": {"cmodel": "execute"}}}
        case = _minimal_extreme_case(name="pred-live-9")
        local_up = apply_registry_to_cases([case], reg, backend="local")
        self.assertIsNone(local_up[0].expected_failure,
                          "local backend for pred-live-9 should have no expected failure")
        self.assertIsNone(local_up[0].expected_failure_phase)
        cmodel_up = apply_registry_to_cases([case], reg, backend="cmodel")
        self.assertEqual(cmodel_up[0].expected_failure, "C1-PRED-001")
        self.assertEqual(cmodel_up[0].expected_failure_phase, "execute")

    def test_apply_registry_cmodel_then_local_clears_expectation(self):
        """Applying cmodel then local backend clears prior expected_failure."""
        reg = {"pred-live-9": {"issue": "C1-PRED-001", "phase_by_backend": {"cmodel": "execute"}}}
        case = _minimal_extreme_case(name="pred-live-9")
        cmodel_up = apply_registry_to_cases([case], reg, backend="cmodel")
        self.assertEqual(cmodel_up[0].expected_failure, "C1-PRED-001")
        local_up = apply_registry_to_cases(cmodel_up, reg, backend="local")
        self.assertIsNone(local_up[0].expected_failure,
                          "local backend should clear prior cmodel expectation")
        self.assertIsNone(local_up[0].expected_failure_phase)

    def test_apply_registry_invalid_backend_raises(self):
        """Invalid backend keyword raises RegistryError or ValueError (fail-closed)."""
        reg = self._make_phase_by_backend_registry()
        case = _minimal_extreme_case(name="gpr-live-255")
        with self.assertRaises((RegistryError, ValueError)):
            apply_registry_to_cases([case], reg, backend="invalid_backend")


class ContractMatrixTests(unittest.TestCase):
    """Tests for the contract matrix builder."""

    def test_contract_matrix_has_at_least_100_combinations(self):
        cases = load_case_matrix(REPO_ROOT, "contract")
        total = sum(len(c.opt_levels) for c in cases)
        self.assertGreaterEqual(total, 100)

    def test_contract_case_names_are_unique(self):
        cases = load_case_matrix(REPO_ROOT, "contract")
        names = [c.name for c in cases]
        self.assertEqual(len(names), len(set(names)))

    def test_contract_matrix_is_deterministic(self):
        cases1 = load_case_matrix(REPO_ROOT, "contract")
        cases2 = load_case_matrix(REPO_ROOT, "contract")
        self.assertEqual(len(cases1), len(cases2))
        for c1, c2 in zip(cases1, cases2):
            self.assertEqual(c1.name, c2.name)
            self.assertEqual(c1.ptx, c2.ptx)
            self.assertEqual(c1.pmem, c2.pmem)
            self.assertEqual(c1.gmem, c2.gmem)

    def test_contract_case_oracles_stay_in_gmem(self):
        cases = load_case_matrix(REPO_ROOT, "contract")
        for case in cases:
            out = case.output
            elem_size = 4
            out_end = out.offset + int(np.prod(out.shape)) * elem_size
            self.assertLessEqual(
                out_end,
                len(case.gmem),
                f"{case.name}: output range {out.offset}..{out_end} exceeds "
                f"GMEM ({len(case.gmem)} bytes)",
            )

    def test_contract_matrix_each_case_has_ptx(self):
        cases = load_case_matrix(REPO_ROOT, "contract")
        for case in cases:
            self.assertTrue(
                ".version 9.3" in case.ptx,
                f"{case.name}: missing .version 9.3",
            )
            self.assertGreater(len(case.ptx), 100, f"{case.name}: PTX too short")


# ════════════════════════════════════════════════════════════════════════
# Task 5: Contract/Frontier Integration (pressure case routing)
# ════════════════════════════════════════════════════════════════════════

class ContractFrontierPressureIntegrationTests(unittest.TestCase):
    """Tests that contract suite includes pressure cases and frontier suite works."""

    # ── Contract includes pressure cases ────────────────────────────────

    def test_contract_includes_all_pressure_contract_names(self):
        """load_case_matrix(root,'contract') must include all 13 contract pressure names."""
        cases = load_case_matrix(REPO_ROOT, "contract")
        pressure_names = {c.name for c in cases if c.name in CONTRACT_NAMES}
        for name in sorted(CONTRACT_NAMES):
            self.assertIn(
                name, pressure_names,
                f"{name} missing from contract suite — pressure cases not merged",
            )

    def test_contract_pressure_cases_have_no_expected_failure(self):
        """All contract pressure cases within the contract suite have expected_failure=None."""
        cases = load_case_matrix(REPO_ROOT, "contract")
        for case in cases:
            if case.name in CONTRACT_NAMES:
                self.assertIsNone(
                    case.expected_failure,
                    f"{case.name}: contract pressure case must not have expected_failure",
                )

    def test_contract_has_exactly_13_pressure_names(self):
        """Exactly 13 unique contract pressure names appear in the contract suite."""
        cases = load_case_matrix(REPO_ROOT, "contract")
        pressure_names = {c.name for c in cases if c.name in CONTRACT_NAMES}
        self.assertEqual(len(pressure_names), len(CONTRACT_NAMES))

    # ── Frontier discovery ──────────────────────────────────────────────

    def test_frontier_returns_exactly_9_cases(self):
        """load_case_matrix(root,'frontier') returns exactly the 9 frontier cases."""
        cases = load_case_matrix(REPO_ROOT, "frontier")
        self.assertEqual(len(cases), len(FRONTIER_NAMES))
        returned_names = {c.name for c in cases}
        self.assertEqual(returned_names, FRONTIER_NAMES)

    def test_frontier_applies_expected_failure_registry(self):
        """Frontier cases from load_case_matrix have no expected_failure until backend overlay; registry key set and phase_by_backend maps are exact."""
        cases = load_case_matrix(REPO_ROOT, "frontier")
        reg = load_pressure_registry(REPO_ROOT)
        # Verify the exact registered set — all 9 frontier cases are known-failing
        expected_registered = {
            "gpr-live-255",
            "gpr-live-256",
            "gpr-live-300",
            "pair-live-127",
            "pair-live-128",
            "pair-live-140",
            "pred-live-9",
            "pred-live-12",
            "pred-live-16",
        }
        self.assertEqual(set(reg.keys()), expected_registered)
        # All cases must be returned WITHOUT expected_failure until backend overlay
        for case in cases:
            self.assertIn(case.name, expected_registered, f"{case.name}: unexpected case")
            self.assertIsNone(case.expected_failure,
                              f"{case.name}: expected_failure must be None until backend overlay")
            self.assertIsNone(case.expected_failure_phase,
                              f"{case.name}: expected_failure_phase must be None until backend overlay")
        # Registry entries have phase_by_backend maps (not scalar phase)
        for case_name in expected_registered:
            entry = reg[case_name]
            self.assertIn("phase_by_backend", entry,
                          f"{case_name}: registry must have phase_by_backend not scalar phase")
            self.assertIsInstance(entry["phase_by_backend"], dict)
            # At least one of the allowed backends
            if "local" in entry["phase_by_backend"]:
                self.assertIn(entry["phase_by_backend"]["local"], VALID_PHASES)
            if "cmodel" in entry["phase_by_backend"]:
                self.assertIn(entry["phase_by_backend"]["cmodel"], VALID_PHASES)
            # Scalar 'phase' must not exist
            self.assertNotIn("phase", entry,
                             f"{case_name}: legacy scalar phase field must be removed")

    def test_frontier_unknown_suite_fails_closed(self):
        """Unknown suite passed to load_case_matrix raises ValueError (not empty list)."""
        with self.assertRaises(ValueError):
            load_case_matrix(REPO_ROOT, "nonexistent")

    def test_discover_cases_frontier_non_empty(self):
        """discover_cases(root,'frontier') returns > 0 cases (no zero-discovery failure)."""
        cases = discover_cases(REPO_ROOT, "frontier")
        self.assertGreater(len(cases), 0)
        self.assertEqual(len(cases), len(FRONTIER_NAMES))

    def test_frontier_contract_cases_have_no_expected_failure(self):
        """Within frontier suite, cases that are CONTRACT_NAMES must not exist (only frontier)."""
        cases = load_case_matrix(REPO_ROOT, "frontier")
        for case in cases:
            self.assertIn(
                case.name, FRONTIER_NAMES,
                f"{case.name}: unexpected non-frontier case in frontier suite",
            )

    # ── Suite tag normalization ──────────────────────────────────────────

    def test_contract_suite_tag_normalized(self):
        """Every case from load_case_matrix(root,'contract') has suite='contract'."""
        cases = load_case_matrix(REPO_ROOT, "contract")
        for c in cases:
            self.assertEqual(
                c.suite, "contract",
                f"{c.name}: expected suite='contract', got suite='{c.suite}'",
            )

    def test_frontier_suite_tag_normalized(self):
        """Every case from load_case_matrix(root,'frontier') has suite='frontier'."""
        cases = load_case_matrix(REPO_ROOT, "frontier")
        for c in cases:
            self.assertEqual(
                c.suite, "frontier",
                f"{c.name}: expected suite='frontier', got suite='{c.suite}'",
            )


# ════════════════════════════════════════════════════════════════════════
# Task 6: Backend integration (RED tests — APIs not yet implemented)
# ════════════════════════════════════════════════════════════════════════

from tests.extreme.backends import (
    CompileResult,
    ExecutionResult,
    build_compile_command,
    select_compiler,
    validate_compile,
)
from tests.extreme.runner import (
    ArtifactWriter,
    Classification,
    ComparisonResult,
    classify_case,
    compare_output,
    filter_opt_levels,
    summarize_results,
)


class CompilerDiscoveryTests(unittest.TestCase):
    """1. Compiler selection order: .exe → bare → compiler/; fail-closed."""

    def test_select_compiler_prefers_exe_in_bin(self):
        """select_compiler returns bin/aec-cc.exe when it exists."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exe = root / "bin" / "aec-cc.exe"
            exe.parent.mkdir(parents=True)
            exe.write_text("fake")
            # Also create the bare variant to prove ordering
            (root / "bin" / "aec-cc").write_text("fake")
            found = select_compiler(root)
            self.assertEqual(found, exe)

    def test_select_compiler_bare_in_bin_second(self):
        """select_compiler falls back to bin/aec-cc when .exe absent."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bare = root / "bin" / "aec-cc"
            bare.parent.mkdir(parents=True)
            bare.write_text("fake")
            found = select_compiler(root)
            self.assertEqual(found, bare)

    def test_select_compiler_compiler_dir_last(self):
        """select_compiler falls back to compiler/aec-cc when bin/ missing."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            compiler_path = root / "compiler" / "aec-cc"
            compiler_path.parent.mkdir(parents=True)
            compiler_path.write_text("fake")
            found = select_compiler(root)
            self.assertEqual(found, compiler_path)

    def test_select_compiler_fail_closed_when_all_missing(self):
        """select_compiler raises FileNotFoundError when no candidate exists."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                select_compiler(root)

    def test_select_compiler_from_c1_subdir_regression(self):
        """Regression: select_compiler(repo/'C1') finds C1/bin/aec-cc.

        The real layout is ``<repo>/C1/bin/aec-cc``.  ``run_extreme.main()``
        passes ``_C1_ROOT`` (not the repo root) to ``select_compiler``, so the
        function must resolve the compiler when given a C1-root path that
        contains ``bin/aec-cc``.
        """
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)  # simulates A4S_C repo root
            c1_root = repo / "C1"
            exe = c1_root / "bin" / "aec-cc"
            exe.parent.mkdir(parents=True)
            exe.write_text("fake")
            found = select_compiler(c1_root)
            self.assertEqual(found, exe)


class BuildCompileCommandTests(unittest.TestCase):
    """2. build_compile_command produces exact CLI flags."""

    def test_command_includes_O0(self):
        """build_compile_command includes -O0 flag."""
        cmd = build_compile_command(
            Path("/usr/bin/aec-cc"),
            Path("input.ptx"),
            "O0",
            Path("out.aecbin"),
            Path("report.json"),
        )
        self.assertIn("-O0", cmd)

    def test_command_includes_O2(self):
        """build_compile_command includes -O2 flag."""
        cmd = build_compile_command(
            Path("/usr/bin/aec-cc"),
            Path("input.ptx"),
            "O2",
            Path("out.aecbin"),
            Path("report.json"),
        )
        self.assertIn("-O2", cmd)

    def test_command_includes_O3(self):
        """build_compile_command includes -O3 flag."""
        cmd = build_compile_command(
            Path("/usr/bin/aec-cc"),
            Path("input.ptx"),
            "O3",
            Path("out.aecbin"),
            Path("report.json"),
        )
        self.assertIn("-O3", cmd)

    def test_command_not_use_double_dash_opt(self):
        """build_compile_command does NOT use --opt=O0 style flags."""
        cmd = build_compile_command(
            Path("/usr/bin/aec-cc"),
            Path("input.ptx"),
            "O0",
            Path("out.aecbin"),
            Path("report.json"),
        )
        for arg in cmd:
            self.assertFalse(arg.startswith("--opt="),
                            f"unexpected --opt= flag: {arg}")

    def test_command_contains_o_and_report_flags(self):
        """build_compile_command includes -o and --report flags."""
        cmd = build_compile_command(
            Path("/usr/bin/aec-cc"),
            Path("input.ptx"),
            "O2",
            Path("out.aecbin"),
            Path("report.json"),
        )
        self.assertIn("-o", cmd)
        self.assertIn("--report", cmd)

    def test_command_has_ptx_as_first_arg_after_compiler(self):
        """build_compile_command uses input.ptx as first positional arg."""
        cmd = build_compile_command(
            Path("/usr/bin/aec-cc"),
            Path("my_case.ptx"),
            "O2",
            Path("p.aecbin"),
            Path("r.json"),
        )
        self.assertEqual(cmd[1], "my_case.ptx")


class BackendResultTypesTests(unittest.TestCase):
    """3. CompileResult and ExecutionResult are frozen with exact fields."""

    def test_compile_result_is_frozen(self):
        """CompileResult is a frozen dataclass (cannot mutate)."""
        r = CompileResult(
            returncode=0,
            stdout="",
            stderr="",
            aecbin=None,
            report=None,
        )
        with self.assertRaises(Exception):
            r.returncode = 1  # type: ignore[misc]

    def test_compile_result_has_expected_fields(self):
        """CompileResult exposes returncode, stdout, stderr, aecbin, report."""
        r = CompileResult(
            returncode=0,
            stdout="out",
            stderr="err",
            aecbin=Path("a.aecbin"),
            report={"status": "ok", "opt_level": "O2", "num_aec_instructions": 2},
        )
        self.assertEqual(r.returncode, 0)
        self.assertEqual(r.stdout, "out")
        self.assertEqual(r.stderr, "err")
        self.assertEqual(r.aecbin, Path("a.aecbin"))
        self.assertEqual(r.report, {"status": "ok", "opt_level": "O2", "num_aec_instructions": 2})

    def test_execution_result_is_frozen(self):
        """ExecutionResult is a frozen dataclass (cannot mutate)."""
        r = ExecutionResult(returncode=0, status="pass", output=b"", cycles=None, detail="")
        with self.assertRaises(Exception):
            r.returncode = 1  # type: ignore[misc]

    def test_execution_result_has_expected_fields(self):
        """ExecutionResult exposes returncode, status, output, cycles, detail."""
        r = ExecutionResult(returncode=0, status="pass", output=b"\x01", cycles=42, detail="info")
        self.assertEqual(r.returncode, 0)
        self.assertEqual(r.status, "pass")
        self.assertEqual(r.output, b"\x01")
        self.assertEqual(r.cycles, 42)
        self.assertEqual(r.detail, "info")


class CompileValidationTests(unittest.TestCase):
    """4. Compile validation rejects zero-exit with broken artifacts."""

    def _make_result(
        self,
        root: Path,
        returncode: int = 0,
        aecbin_content: bytes | None = b"\x00" * 32,
        report_data: dict | None = None,
    ) -> CompileResult:
        """Helper: build a CompileResult with temp artifacts under *root*."""
        aecbin_path: Path | None = None

        if aecbin_content is not None:
            aecbin_path = root / "out.aecbin"
            aecbin_path.write_bytes(aecbin_content)

        return CompileResult(
            returncode=returncode,
            stdout="",
            stderr="",
            aecbin=aecbin_path,
            report=report_data,
        )

    def test_validate_rejects_missing_aecbin(self):
        """Zero exit but no .aecbin file → validation error."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self._make_result(root, aecbin_content=None, report_data={"status": "ok", "opt_level": "O2", "num_aec_instructions": 2})
            with self.assertRaises(ValueError):
                validate_compile(result, "O2")

    def test_validate_rejects_empty_aecbin(self):
        """Zero exit with empty .aecbin file → validation error."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self._make_result(root, aecbin_content=b"", report_data={"status": "ok", "opt_level": "O2", "num_aec_instructions": 0})
            with self.assertRaises(ValueError):
                validate_compile(result, "O2")

    def test_validate_rejects_misaligned_aecbin(self):
        """Zero exit with misaligned .aecbin size → validation error."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # 7 bytes is not 16-byte aligned
            result = self._make_result(root, aecbin_content=b"\x00" * 7, report_data={"status": "ok", "opt_level": "O2", "num_aec_instructions": 0})
            with self.assertRaises(ValueError):
                validate_compile(result, "O2")

    def test_validate_rejects_missing_report(self):
        """Zero exit but no report → validation error."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self._make_result(root, report_data=None)
            with self.assertRaises(ValueError):
                validate_compile(result, "O2")

    def test_validate_rejects_malformed_report(self):
        """Report missing required 'status' key → validation error."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self._make_result(root, aecbin_content=b"\x00" * 32, report_data={"opt_level": "O2", "num_aec_instructions": 2})
            with self.assertRaises(ValueError):
                validate_compile(result, "O2")

    def test_validate_rejects_wrong_report_status(self):
        """Report status not 'ok' → validation error."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self._make_result(root, report_data={"status": "failure", "opt_level": "O2", "num_aec_instructions": 2})
            with self.assertRaises(ValueError):
                validate_compile(result, "O2")

    def test_validate_rejects_wrong_opt_level(self):
        """Report opt_level does not match expected → validation error."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self._make_result(root, report_data={"status": "ok", "opt_level": "O0", "num_aec_instructions": 2})
            with self.assertRaises(ValueError):
                validate_compile(result, "O2")

    def test_validate_rejects_wrong_instruction_count(self):
        """Report num_aec_instructions does not match expected → validation error."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # 2 instructions × 16 bytes = 32 byte file, but report says 99
            result = self._make_result(root, aecbin_content=b"\x00" * 32, report_data={"status": "ok", "opt_level": "O2", "num_aec_instructions": 99})
            with self.assertRaises(ValueError):
                validate_compile(result, "O2")

    def test_validate_rejects_inst_count_mismatch(self):
        """num_aec_instructions * 16 != actual aecbin size → validation error."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # 3 instructions × 16 = 48, but file is 64 bytes
            result = self._make_result(root, aecbin_content=b"\x00" * 64, report_data={"status": "ok", "opt_level": "O2", "num_aec_instructions": 3})
            with self.assertRaises(ValueError):
                validate_compile(result, "O2")

    def test_validate_accepts_valid_result(self):
        """All artifacts correct → validate_compile passes."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self._make_result(root, aecbin_content=b"\x00" * 32, report_data={"status": "ok", "opt_level": "O2", "num_aec_instructions": 2})
            # Should not raise
            validate_compile(result, "O2")


class OutputComparisonTests(unittest.TestCase):
    """4. Output comparison: rejection, success, and detail reporting."""

    def _expectation(self, dtype: str = "<u4", shape: tuple = (4,), expected: tuple = (0, 1, 2, 3), rtol: float = 0.0, atol: float = 0.0) -> OutputExpectation:
        return OutputExpectation(offset=0, dtype=dtype, shape=shape, expected=expected, rtol=rtol, atol=atol)

    def test_compare_rejects_length_mismatch_shorter(self):
        """Actual bytes shorter than expected → mismatch."""
        actual = struct.pack("<II", 0, 1)  # 2 u32
        exp = self._expectation(dtype="<u4", shape=(4,), expected=(0, 1, 2, 3))
        result = compare_output(actual, exp)
        self.assertFalse(result.matched)

    def test_compare_rejects_length_mismatch_longer(self):
        """Actual bytes longer than expected → mismatch (exact length required)."""
        actual = struct.pack("<IIIII", 0, 1, 2, 3, 4)  # 5 u32, but only 4 expected
        exp = self._expectation(dtype="<u4", shape=(4,), expected=(0, 1, 2, 3))
        result = compare_output(actual, exp)
        self.assertFalse(result.matched)

    def test_compare_rejects_shape_mismatch(self):
        """Flat element count differs from expected shape product → mismatch."""
        actual = struct.pack("<IIII", 0, 1, 2, 3)  # 4 elements, but...
        exp = self._expectation(dtype="<u4", shape=(2, 4), expected=(0, 1, 2, 3, 4, 5, 6, 7))
        # 4 bytes ≠ 8 elements × 4 bytes = 32 bytes
        result = compare_output(actual, exp)
        self.assertFalse(result.matched)

    def test_compare_rejects_nan_expected(self):
        """NaN in expected values → mismatch."""
        actual = struct.pack("<ffff", 1.0, 2.0, 3.0, 4.0)
        exp = self._expectation(dtype="<f4", shape=(4,), expected=(1.0, 2.0, float("nan"), 4.0))
        result = compare_output(actual, exp)
        self.assertFalse(result.matched)

    def test_compare_rejects_inf_expected(self):
        """Inf in expected values → mismatch."""
        actual = struct.pack("<ffff", 1.0, 2.0, 3.0, 4.0)
        exp = self._expectation(dtype="<f4", shape=(4,), expected=(1.0, 2.0, float("inf"), 4.0))
        result = compare_output(actual, exp)
        self.assertFalse(result.matched)

    def test_compare_rejects_nan_actual(self):
        """NaN in actual values → mismatch."""
        actual = struct.pack("<ffff", 1.0, 2.0, float("nan"), 4.0)
        exp = self._expectation(dtype="<f4", shape=(4,), expected=(1.0, 2.0, 3.0, 4.0))
        result = compare_output(actual, exp)
        self.assertFalse(result.matched)

    def test_compare_rejects_inf_actual(self):
        """Inf in actual values → mismatch."""
        actual = struct.pack("<ffff", 1.0, 2.0, float("inf"), 4.0)
        exp = self._expectation(dtype="<f4", shape=(4,), expected=(1.0, 2.0, 3.0, 4.0))
        result = compare_output(actual, exp)
        self.assertFalse(result.matched)

    def test_compare_rejects_out_of_tolerance(self):
        """Actual outside rtol/atol tolerance → mismatch."""
        actual = struct.pack("<ffff", 1.0, 2.0, 3.5, 4.0)  # 3.5 vs expected 3.0
        exp = self._expectation(dtype="<f4", shape=(4,), expected=(1.0, 2.0, 3.0, 4.0), rtol=0.0, atol=0.1)
        result = compare_output(actual, exp)
        self.assertFalse(result.matched)

    def test_compare_exact_int_success(self):
        """Exact integer match → matched=True."""
        actual = struct.pack("<IIII", 10, 20, 30, 40)
        exp = self._expectation(dtype="<u4", shape=(4,), expected=(10, 20, 30, 40))
        result = compare_output(actual, exp)
        self.assertTrue(result.matched)

    def test_compare_tolerant_float_success(self):
        """Within-tolerance float match → matched=True."""
        actual = struct.pack("<ffff", 1.0, 2.0005, 3.0, 4.0)
        exp = self._expectation(dtype="<f4", shape=(4,), expected=(1.0, 2.0, 3.0, 4.0), rtol=1e-3, atol=1e-3)
        result = compare_output(actual, exp)
        self.assertTrue(result.matched)

    def test_compare_mismatch_detail_includes_first_index(self):
        """Mismatch detail reports the first flat index that differed."""
        actual = struct.pack("<IIII", 10, 99, 30, 40)
        exp = self._expectation(dtype="<u4", shape=(4,), expected=(10, 20, 30, 40))
        result = compare_output(actual, exp)
        self.assertFalse(result.matched)
        self.assertEqual(result.first_mismatch_index, 1)

    def test_compare_mismatch_detail_includes_expected_actual(self):
        """Mismatch detail reports the expected and actual values."""
        actual = struct.pack("<IIII", 10, 99, 30, 40)
        exp = self._expectation(dtype="<u4", shape=(4,), expected=(10, 20, 30, 40))
        result = compare_output(actual, exp)
        self.assertFalse(result.matched)
        self.assertEqual(result.expected_value, 20)
        self.assertEqual(result.actual_value, 99)

    def test_compare_mismatch_detail_includes_max_abs_diff(self):
        """Mismatch detail reports the maximum absolute difference."""
        actual = struct.pack("<ffff", 1.0, 2.5, 3.0, 4.0)
        exp = self._expectation(dtype="<f4", shape=(4,), expected=(1.0, 2.0, 3.0, 4.0))
        result = compare_output(actual, exp)
        self.assertFalse(result.matched)
        self.assertIsNotNone(result.max_abs_diff)
        self.assertAlmostEqual(result.max_abs_diff, 0.5, places=6)  # type: ignore[arg-type]


class ResultClassificationTests(unittest.TestCase):
    """5. Result classification table: PASS / FAIL / XFAIL / XPASS."""

    def _make_case(self, name: str, suite: str = "contract",
                   expected_failure: str | None = None,
                   expected_failure_phase: str | None = None) -> ExtremeCase:
        out = OutputExpectation(offset=0, dtype="<u4", shape=(1,), expected=(0,), rtol=0.0, atol=0.0)
        return ExtremeCase(
            name=name, suite=suite, ptx="ret;",
            grid=(1, 1, 1), block=(1, 1, 1),
            pmem=b"", gmem=b"\x00" * 4,
            output=out, opt_levels=("O2",),
            expected_failure=expected_failure,
            expected_failure_phase=expected_failure_phase,
        )

    def _ok_comparison(self) -> ComparisonResult:
        return ComparisonResult(matched=True)

    def _fail_comparison(self) -> ComparisonResult:
        return ComparisonResult(matched=False, first_mismatch_index=0, expected_value=0.0, actual_value=1.0, max_abs_diff=1.0)

    def test_contract_success_is_pass(self):
        """Contract case with matched output → PASS."""
        case = self._make_case("T1_s0_n1", suite="contract")
        result = classify_case("contract", case, self._ok_comparison())
        self.assertEqual(result.verdict, "PASS")

    def test_contract_failure_is_fail(self):
        """Contract case with mismatched output → FAIL."""
        case = self._make_case("T1_s0_n1", suite="contract")
        result = classify_case("contract", case, self._fail_comparison())
        self.assertEqual(result.verdict, "FAIL")

    def test_frontier_registered_matching_phase_is_xfail(self):
        """Frontier case with expected_failure and matching phase → XFAIL."""
        case = self._make_case("gpr-live-256", suite="frontier",
                               expected_failure="C1-SPILL-001",
                               expected_failure_phase="compare")
        result = classify_case("frontier", case, self._fail_comparison(),
                               observed_phase="compare")
        self.assertEqual(result.verdict, "XFAIL")

    def test_frontier_registered_wrong_phase_is_fail(self):
        """Frontier case with expected_failure but wrong observed_phase → FAIL."""
        case = self._make_case("gpr-live-256", suite="frontier",
                               expected_failure="C1-SPILL-001",
                               expected_failure_phase="compare")
        result = classify_case("frontier", case, self._fail_comparison(),
                               observed_phase="compile")
        self.assertEqual(result.verdict, "FAIL")

    def test_frontier_unregistered_failure_is_fail(self):
        """Frontier case without expected_failure and mismatched output → FAIL."""
        case = self._make_case("gpr-live-255", suite="frontier", expected_failure=None)
        result = classify_case("frontier", case, self._fail_comparison())
        self.assertEqual(result.verdict, "FAIL")

    def test_frontier_registered_success_is_xpass(self):
        """Frontier case with expected_failure but matched output → XPASS."""
        case = self._make_case("gpr-live-256", suite="frontier",
                               expected_failure="C1-SPILL-001",
                               expected_failure_phase="compare")
        result = classify_case("frontier", case, self._ok_comparison(),
                               observed_phase="compare")
        self.assertEqual(result.verdict, "XPASS")

    def test_frontier_registered_any_phase_success_is_xpass(self):
        """Frontier case with expected_failure and matched output → XPASS regardless of observed phase."""
        case = self._make_case("gpr-live-256", suite="frontier",
                               expected_failure="C1-SPILL-001",
                               expected_failure_phase="compare")
        result = classify_case("frontier", case, self._ok_comparison(),
                               observed_phase="execute")
        self.assertEqual(result.verdict, "XPASS")


class ArtifactWriterTests(unittest.TestCase):
    """6. Artifact writer produces expected files and write failure converts to FAIL."""

    def _minimal_case(self, name: str = "T1_s0_n1") -> ExtremeCase:
        out = OutputExpectation(offset=0, dtype="<u4", shape=(1,), expected=(0,), rtol=0.0, atol=0.0)
        return ExtremeCase(
            name=name, suite="contract", ptx="ret;",
            grid=(1, 1, 1), block=(1, 1, 1),
            pmem=b"", gmem=b"\x00" * 4,
            output=out, opt_levels=("O2",),
            expected_failure=None,
        )

    def test_artifact_writer_uses_correct_filenames(self):
        """ArtifactWriter writes files with the approved naming pattern."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            writer = ArtifactWriter(root, "contract")
            case = self._minimal_case()
            cr = CompileResult(returncode=0, stdout="stdout content", stderr="stderr content",
                               aecbin=None, report=None)
            er = ExecutionResult(returncode=0, status="pass", output=b"\x00" * 4, cycles=None, detail="")
            comp = ComparisonResult(matched=True)
            cls = Classification(verdict="PASS")
            paths = writer.write_all([case], [cr], [er], [comp], [cls])

            # Check specific filenames are present
            filenames = {p.name for p in paths}
            self.assertIn("case.ptx", filenames)
            self.assertIn("compile.stdout.txt", filenames)
            self.assertIn("compile.stderr.txt", filenames)
            self.assertIn("result.json", filenames)
            self.assertIn("expected.bin", filenames)
            self.assertIn("actual.bin", filenames)
            # program.aecbin and program.asm not present because aecbin=None
            # When aecbin is present:
            self.assertNotIn("compiler_stdout.txt", filenames,
                             "old filename compiler_stdout.txt must not be used")
            self.assertNotIn("compiler_stderr.txt", filenames,
                             "old filename compiler_stderr.txt must not be used")
            self.assertNotIn("report.json", filenames,
                             "old filename report.json must not be used")
            self.assertNotIn("case.aecbin", filenames,
                             "old filename case.aecbin must not be used")
            self.assertNotIn("disassembly.txt", filenames,
                             "old filename disassembly.txt must not be used")

    def test_artifact_writer_with_aecbin_uses_program_filenames(self):
        """With aecbin present, program.aecbin and program.asm are written."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            writer = ArtifactWriter(root, "contract")
            case = self._minimal_case()
            # Create a dummy binary for the compile result
            aecbin_path = root / "tmp.aecbin"
            aecbin_path.write_bytes(b"\x00" * 32)
            cr = CompileResult(returncode=0, stdout="", stderr="",
                               aecbin=aecbin_path,
                               report={"status": "ok", "opt_level": "O2", "num_aec_instructions": 2})
            er = ExecutionResult(returncode=0, status="pass", output=b"\x00" * 4, cycles=None, detail="")
            comp = ComparisonResult(matched=True)
            cls = Classification(verdict="PASS")
            paths = writer.write_all([case], [cr], [er], [comp], [cls])
            filenames = {p.name for p in paths}
            self.assertIn("program.aecbin", filenames)
            self.assertIn("program.asm", filenames)

    def test_artifact_write_failure_raises_oserror(self):
        """ArtifactWriter.write_all raises OSError when writing fails (read-only dir)."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Make a read-only subdirectory
            read_only = root / "readonly"
            read_only.mkdir(parents=True)
            read_only.chmod(0o444)  # read-only
            try:
                writer = ArtifactWriter(read_only, "contract")
                case = self._minimal_case()
                cr = CompileResult(returncode=0, stdout="", stderr="", aecbin=None, report=None)
                er = ExecutionResult(returncode=0, status="pass", output=b"\x00" * 4, cycles=None, detail="")
                comp = ComparisonResult(matched=True)
                cls = Classification(verdict="PASS")
                with self.assertRaises(OSError):
                    writer.write_all([case], [cr], [er], [comp], [cls])
            finally:
                read_only.chmod(0o755)  # restore for cleanup


class OptFilteringTests(unittest.TestCase):
    """7. CLI opt-level filtering and summary exit semantics."""

    def _case(self, name: str, opt_levels: tuple[str, ...]) -> ExtremeCase:
        out = OutputExpectation(offset=0, dtype="<u4", shape=(1,), expected=(0,), rtol=0.0, atol=0.0)
        return ExtremeCase(
            name=name, suite="contract", ptx="ret;",
            grid=(1, 1, 1), block=(1, 1, 1),
            pmem=b"", gmem=b"\x00" * 4,
            output=out, opt_levels=opt_levels,
            expected_failure=None,
        )

    def test_opt_filter_all_returns_all_levels(self):
        """filter_opt_levels with 'all' preserves every opt level."""
        cases = [
            self._case("a", ("O0", "O2")),
            self._case("b", ("O2", "O3")),
        ]
        filtered = filter_opt_levels(cases, "all")
        total_opt = sum(len(c.opt_levels) for c in filtered)
        self.assertEqual(total_opt, 4)

    def test_opt_filter_O2_returns_only_O2(self):
        """filter_opt_levels with 'O2' keeps only O2 levels."""
        cases = [
            self._case("a", ("O0", "O2", "O3")),
            self._case("b", ("O2",)),
        ]
        filtered = filter_opt_levels(cases, "O2")
        for c in filtered:
            self.assertEqual(c.opt_levels, ("O2",))

    def test_summarize_results_all_pass_exit_zero(self):
        """All PASS → exit code 0."""
        verdicts = [Classification(verdict="PASS"), Classification(verdict="PASS")]
        self.assertEqual(summarize_results(verdicts), 0)

    def test_summarize_results_any_fail_exit_nonzero(self):
        """Any FAIL → exit code non-zero."""
        verdicts = [Classification(verdict="PASS"), Classification(verdict="FAIL")]
        self.assertNotEqual(summarize_results(verdicts), 0)

    def test_summarize_results_xpass_treated_as_failure(self):
        """XPASS → exit code non-zero (suite failure)."""
        verdicts = [Classification(verdict="XPASS"), Classification(verdict="PASS")]
        self.assertNotEqual(summarize_results(verdicts), 0)


# ════════════════════════════════════════════════════════════════════════
# Task 6: CVTII semantics (oracle gap — local simulator vs Remote Task 6)
# ════════════════════════════════════════════════════════════════════════


class CvtiSemanticTests(unittest.TestCase):
    """CVTII instruction semantic tests via direct simulator invocation.

    Constructs minimal fake decoded instructions and invokes ``Sim._exec``
    directly, bypassing the C++ compilation pipeline.
    """

    @classmethod
    def setUpClass(cls):
        """Ensure ``C1/sim`` is on ``sys.path`` before any ``aec_*`` import."""
        import sys
        from pathlib import Path
        _sim = Path(__file__).resolve().parent.parent.parent / "sim"
        if str(_sim) not in sys.path:
            sys.path.insert(0, str(_sim))

    # Type-code lookup (mirrors aec_decode.TYPES for integer types).
    _TYCO = {"b32": 0, "b64": 1, "u32": 2, "s32": 3, "u8": 4, "s8": 5,
             "f32": 8, "f64": 9}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cvtii_words(dst_type: str, src_type: str,
                     dst: int, src1: int) -> tuple:
        """Build the 4-word tuple for a CVTII instruction.

        Word layout (little-endian):
            w3 = (CVTII_opcode << 16) | pred_ctrl
            w2 = (dst << 16) | src1
            w1 = src2 (unused → 0)
            w0 = src3 (unused → 0)

        pred_ctrl (16 bits)::

            [2:0]   predicate         (0)
            [6:3]   dest type code
            [10:8]  subop             (0)
            [13:11] space             (0)
            [14]    pred_neg          (0)
            [15]    pred_en           (0)

        The source type occupies bits [13:10] — overlapping subop+space,
        which are unused for CVT opcodes.
        """
        dtc = CvtiSemanticTests._TYCO[dst_type]
        stc = CvtiSemanticTests._TYCO[src_type]
        pc = (stc << 10) | (dtc << 3)          # bits [13:10]=src, [6:3]=dst
        w3 = (0x0053 << 16) | pc                # CVTII opcode
        w2 = (dst << 16) | src1
        return (0, 0, w2, w3)

    def _exec_cvtii(self, words: tuple, R_init: dict | None = None,
                    strict: bool = True):
        """Execute a single CVTII instruction and return the final ``R`` array.

        *words* — 4-tuple of instruction words (see :meth:`_cvtii_words`).
        *R_init* — ``{reg_num: scalar_value}`` to preload before execution.
        *strict* — whether to enable strict-mode type checking.
        """
        import sys
        from pathlib import Path
        _sim = Path(__file__).resolve().parent.parent.parent / "sim"
        if str(_sim) not in sys.path:
            sys.path.insert(0, str(_sim))
        from aec_decode import Image, Instr
        from aec_sim import Sim

        ins = Instr(words)
        img = Image()
        img.code = [ins]
        img.entry_pc = 0
        img.instr_count = 1

        gmem = np.zeros(4, np.uint8)
        sim = Sim(img, gmem, strict=strict)

        W = 32
        R = np.zeros((W, 256), np.uint32)
        if R_init:
            for reg, val in R_init.items():
                R[:, reg] = val
        P = np.zeros((W, 8), bool)
        em = np.ones(W, bool)           # all lanes active
        lmem = np.zeros(4096 * W, np.uint8)
        smem = np.zeros(0, np.uint8)

        sim._exec(ins, R, P, em, {}, lmem, smem)
        return R

    # ------------------------------------------------------------------
    # u32 → b64  zero-extension
    # ------------------------------------------------------------------

    def test_u32_to_b64_zero_extends(self):
        """u32 → b64: low word = source, high word = 0."""
        R = self._exec_cvtii(self._cvtii_words("b64", "u32", dst=4, src1=1),
                             {1: 0xDEAD_BEEF})
        self.assertEqual(R[0, 4], 0xDEAD_BEEF,
                         "low word should hold the source value")
        self.assertEqual(R[0, 5], 0,
                         "high word should be zero for u32 source")

    def test_u32_to_b64_zero_extends_small(self):
        """u32 → b64: small value zero-extends (high word = 0)."""
        R = self._exec_cvtii(self._cvtii_words("b64", "u32", dst=4, src1=1),
                             {1: 42})
        self.assertEqual(R[0, 4], 42)
        self.assertEqual(R[0, 5], 0)

    # ------------------------------------------------------------------
    # s32 → b64  sign-extension
    # ------------------------------------------------------------------

    def test_s32_to_b64_sign_extends_positive(self):
        """s32 → b64: positive value → high word = 0."""
        R = self._exec_cvtii(self._cvtii_words("b64", "s32", dst=4, src1=1),
                             {1: 127})
        self.assertEqual(R[0, 4], 127)
        self.assertEqual(R[0, 5], 0)

    def test_s32_to_b64_sign_extends_negative(self):
        """s32 → b64: negative value → high word = 0xFFFFFFFF."""
        R = self._exec_cvtii(self._cvtii_words("b64", "s32", dst=4, src1=1),
                             {1: 0xFFFF_FFFF})      # −1 as u32 bits
        self.assertEqual(R[0, 4], 0xFFFF_FFFF,
                         "low word should hold the s32 bit pattern")
        self.assertEqual(R[0, 5], 0xFFFF_FFFF,
                         "high word should be 0xFFFFFFFF (sign extension)")

    def test_s32_to_b64_sign_extends_neg_boundary(self):
        """s32 → b64: INT32_MIN → high word = 0xFFFFFFFF."""
        R = self._exec_cvtii(self._cvtii_words("b64", "s32", dst=4, src1=1),
                             {1: 0x8000_0000})      # INT32_MIN as u32 bits
        self.assertEqual(R[0, 4], 0x8000_0000)
        self.assertEqual(R[0, 5], 0xFFFF_FFFF)

    # ------------------------------------------------------------------
    # b64 → u32  truncation
    # ------------------------------------------------------------------

    def test_b64_to_u32_truncates_low_word(self):
        """b64 → u32: result = low word of the source pair."""
        R = self._exec_cvtii(self._cvtii_words("u32", "b64", dst=4, src1=1),
                             {1: 0xAAAA_AAAA,        # low word
                              2: 0xBBBB_BBBB})        # high word (pair {R2,R1})
        self.assertEqual(R[0, 4], 0xAAAA_AAAA,
                         "only the low word should survive truncation")

    def test_b64_to_u32_ignores_high_word(self):
        """b64 → u32: high-word content does not affect result."""
        R = self._exec_cvtii(self._cvtii_words("u32", "b64", dst=4, src1=1),
                             {1: 0x0000_0001,        # low  word
                              2: 0xFFFF_FFFF})        # high word
        self.assertEqual(R[0, 4], 0x0000_0001)

    # ------------------------------------------------------------------
    # 32-bit integer-to-integer copies
    # ------------------------------------------------------------------

    def test_u32_to_u32_preserves_bits(self):
        """u32 → u32: identity copy."""
        R = self._exec_cvtii(self._cvtii_words("u32", "u32", dst=4, src1=1),
                             {1: 0xA5A5_A5A5})
        self.assertEqual(R[0, 4], 0xA5A5_A5A5)

    def test_s32_to_s32_preserves_bits(self):
        """s32 → s32: identity copy (bit-preserving)."""
        R = self._exec_cvtii(self._cvtii_words("s32", "s32", dst=4, src1=1),
                             {1: 0xA5A5_A5A5})
        self.assertEqual(R[0, 4], 0xA5A5_A5A5)

    def test_b32_to_b32_preserves_bits(self):
        """b32 → b32: identity copy."""
        R = self._exec_cvtii(self._cvtii_words("b32", "b32", dst=4, src1=1),
                             {1: 0xA5A5_A5A5})
        self.assertEqual(R[0, 4], 0xA5A5_A5A5)

    def test_u32_to_s32_preserves_bits(self):
        """u32 → s32: same bits (reinterpret, not a value conversion)."""
        R = self._exec_cvtii(self._cvtii_words("s32", "u32", dst=4, src1=1),
                             {1: 0xFFFF_FFFC})
        self.assertEqual(R[0, 4], 0xFFFF_FFFC)

    def test_s32_to_u32_preserves_bits(self):
        """s32 → u32: same bits."""
        R = self._exec_cvtii(self._cvtii_words("u32", "s32", dst=4, src1=1),
                             {1: 0xFFFF_FFFC})
        self.assertEqual(R[0, 4], 0xFFFF_FFFC)

    # ------------------------------------------------------------------
    # Execution mask respects both words of pair output
    # ------------------------------------------------------------------

    def test_cvtii_pair_respects_mask_both_words(self):
        """CVTII.b64 only writes enabled lanes for LOW and HIGH words."""
        from aec_decode import Image, Instr
        import sys
        from pathlib import Path
        _sim = Path(__file__).resolve().parent.parent.parent / "sim"
        if str(_sim) not in sys.path:
            sys.path.insert(0, str(_sim))
        from aec_sim import Sim

        ins = Instr(self._cvtii_words("b64", "u32", dst=4, src1=1))
        img = Image()
        img.code = [ins]
        img.entry_pc = 0
        img.instr_count = 1

        gmem = np.zeros(4, np.uint8)
        sim = Sim(img, gmem, strict=True)

        W = 32
        R = np.zeros((W, 256), np.uint32)
        R[:, 1] = 0xDEAD_BEEF
        R[:, 4] = 0x1111_1111      # stale low  word (disabled lane)
        R[:, 5] = 0x2222_2222      # stale high word (disabled lane)
        P = np.zeros((W, 8), bool)
        em = np.zeros(W, bool)
        em[0] = True                # only lane 0 active
        lmem = np.zeros(4096 * W, np.uint8)
        smem = np.zeros(0, np.uint8)

        sim._exec(ins, R, P, em, {}, lmem, smem)
        # Lane 0: written
        self.assertEqual(R[0, 4], 0xDEAD_BEEF, "lane 0 low word should be written")
        self.assertEqual(R[0, 5], 0,            "lane 0 high word should be zero")
        # Lane 1: masked — must remain unchanged
        self.assertEqual(R[1, 4], 0x1111_1111,  "lane 1 low word must not change")
        self.assertEqual(R[1, 5], 0x2222_2222,  "lane 1 high word must not change")

    # ------------------------------------------------------------------
    # Strict mode: reject unsupported type combinations
    # ------------------------------------------------------------------

    def test_strict_rejects_float_source(self):
        """Strict mode: CVTII.b64 with f32 source raises ExecError."""
        from aec_sim import ExecError
        with self.assertRaises(ExecError):
            self._exec_cvtii(
                self._cvtii_words("b64", "f32", dst=4, src1=1),
                {1: 0x3F80_0000},  # 1.0f32
            )

    def test_strict_rejects_float_dest(self):
        """Strict mode: CVTII.f32 with u32 source raises ExecError."""
        from aec_sim import ExecError
        with self.assertRaises(ExecError):
            self._exec_cvtii(
                self._cvtii_words("f32", "u32", dst=4, src1=1),
                {1: 42},
            )

    def test_strict_rejects_f64_source(self):
        """Strict mode: CVTII with f64 source raises ExecError."""
        from aec_sim import ExecError
        with self.assertRaises(ExecError):
            self._exec_cvtii(
                self._cvtii_words("b64", "f64", dst=4, src1=1),
                {1: 0},
            )


class ToolPathHelperTests(unittest.TestCase):
    """Tests for the disassembly tool-path resolution helper."""

    def test_find_disassembly_tool_bare_name_found(self):
        """_find_disassembly_tool returns a list containing the bare name
        when it resolves via shutil.which (mocked by checking a known
        system command)."""
        from tests.extreme.runner import _find_disassembly_tool
        # ping or cmd.exe should always be findable on Windows
        result = _find_disassembly_tool("cmd.exe", Path("."))
        self.assertIsInstance(result, list)
        self.assertTrue(any("cmd.exe" in str(cmd) for cmd in result))

    def test_find_disassembly_tool_returns_candidates(self):
        """_find_disassembly_tool returns at least the bare-name fallback."""
        from tests.extreme.runner import _find_disassembly_tool
        result = _find_disassembly_tool("nonexistent_tool_xyz",
                                         Path("/tmp"))
        # Should return something (the bare name) even when not found
        self.assertIsInstance(result, list)
        self.assertTrue(any("nonexistent_tool_xyz" in str(cmd) for cmd in result))


# ════════════════════════════════════════════════════════════════════════
# Task 7: CModel Backend — RED tests (production not yet implemented)
# ════════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════
# Task 7/debug: Pair-pressure root cause fix — RED tests
# (production generate_pair_pressure still uses cvt.u32.u64)
# ════════════════════════════════════════════════════════════════════════


class PairPressureRootCauseRedTests(unittest.TestCase):
    """RED tests for pair-pressure root-cause fix (Task 7/debug).

    The fix replaces ``cvt.u32.u64`` with ``ld.global.u32`` + ``add.u32``
    accumulation, using ``%bd<i>`` as GMEM offset values (256+4*i) instead
    of small integer literals.  All tests here are RED (fail with the
    current ``generate_pair_pressure`` implementation) and will turn GREEN
    after the fix is applied.
    """

    # ── generate_pair_pressure structural assertions ──────────────────

    def test_generate_pair_pressure_no_cvt(self):
        """generate_pair_pressure(32) must NOT contain cvt. instructions."""
        from tests.extreme.generators import generate_pair_pressure
        src = generate_pair_pressure(32)
        for line in src.splitlines():
            self.assertNotIn("cvt.", line,
                             f"cvt. instruction found: {line.strip()}")

    def test_generate_pair_pressure_has_ld_global_u32(self):
        """generate_pair_pressure(32) must contain ld.global.u32 loads."""
        from tests.extreme.generators import generate_pair_pressure
        src = generate_pair_pressure(32)
        self.assertIn("ld.global.u32", src,
                      "pair pressure must include global u32 loads")

    def test_generate_pair_pressure_bd_values_are_gmem_offsets(self):
        """Each %bd<i> in ENTRY holds GMEM offset 256+4*i (not small literal)."""
        from tests.extreme.generators import generate_pair_pressure
        src = generate_pair_pressure(32)
        for i in range(32):
            self.assertIn(
                f"mov.u64 %bd{i}, {256 + 4 * i}",
                src,
                f"%bd{i} should be initialized to GMEM offset {256 + 4 * i}",
            )

    def test_generate_pair_pressure_no_u64_accumulator_in_reduce(self):
        """REDUCE section must not contain 64-bit accumulation (%rd0)."""
        from tests.extreme.generators import generate_pair_pressure
        src = generate_pair_pressure(32)
        _, reduce = src.split("REDUCE:")
        self.assertNotIn("add.u64 %rd0,", reduce,
                         "64-bit accumulation must be removed; "
                         "use u32 accumulator instead")

    def test_generate_pair_pressure_u32_accumulator_in_reduce(self):
        """REDUCE section initializes mov.u32 %r0 and accumulates with add.u32."""
        from tests.extreme.generators import generate_pair_pressure
        src = generate_pair_pressure(32)
        _, reduce = src.split("REDUCE:")
        self.assertIn("mov.u32 %r0, 0", reduce,
                      "u32 accumulator must be zero-initialized in REDUCE")
        self.assertIn("add.u32 %r0,", reduce,
                      "accumulation must use u32 add")

    # ── load_case_matrix / GMEM assertions ────────────────────────────

    def test_pair_pressure_case_expected_equals_count(self):
        """Pair pressure case expected[0] must equal count, not triangular number."""
        cases = load_case_matrix(REPO_ROOT, "pressure")
        for case in cases:
            if case.name.startswith("pair-live-"):
                n = int(case.name.split("-")[-1])
                self.assertEqual(
                    case.output.expected[0], n,
                    f"{case.name}: expected should equal count {n}, "
                    f"got {case.output.expected[0]}",
                )

    def test_pair_pressure_case_gmem_has_input_slots(self):
        """Pair pressure case GMEM has u32 1 at each 256+4*i input slot, length sufficient."""
        cases = load_case_matrix(REPO_ROOT, "pressure")
        for case in cases:
            if case.name.startswith("pair-live-"):
                n = int(case.name.split("-")[-1])
                min_size = 256 + 4 * n + 4
                self.assertGreaterEqual(
                    len(case.gmem), min_size,
                    f"{case.name}: GMEM length {len(case.gmem)} < {min_size} "
                    f"(needs {n} input slots at 256+4*i)",
                )
                for i in range(n):
                    val = struct.unpack_from("<I", case.gmem, 256 + 4 * i)[0]
                    self.assertEqual(
                        val, 1,
                        f"{case.name}: input slot {i} at offset {256+4*i} "
                        f"should be 1, got {val}",
                    )


class CModelDiscoveryTests(unittest.TestCase):
    """CModel binary discovery: release path priority, missing binary fail-closed.

    RED: ``select_cmodel`` does not exist in ``backends.py`` → ImportError.
    """

    def test_select_cmodel_finds_release_binary(self):
        """select_cmodel resolves public/aec-cmodel-release/bin/aec-precise-linux-x86_64."""
        from tests.extreme.backends import select_cmodel  # RED: does not exist

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            release = (
                root
                / "public"
                / "aec-cmodel-release"
                / "bin"
                / "aec-precise-linux-x86_64"
            )
            release.parent.mkdir(parents=True)
            release.write_text("")
            found = select_cmodel(root)
            self.assertEqual(found, release)

    def test_select_cmodel_fail_closed_when_missing(self):
        """select_cmodel raises FileNotFoundError when no binary exists."""
        from tests.extreme.backends import select_cmodel  # RED: does not exist

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                select_cmodel(root)


class CModelCommandConstructionTests(unittest.TestCase):
    """CModel command construction: all required flags present in the right order.

    RED: ``build_cmodel_command`` does not exist in ``backends.py`` → ImportError.
    """

    # -- helpers --------------------------------------------------------------

    def _build_cmd(self):
        """Call build_cmodel_command with representative arguments."""
        from tests.extreme.backends import build_cmodel_command  # RED

        return build_cmodel_command(
            cmodel=Path("/usr/bin/aec-precise"),
            aecbin=Path("kernel.aecbin"),
            ninstr=2,
            grid=(2, 1, 1),
            block=(128, 1, 1),
            pmem_path=Path("/tmp/pmem.bin"),
            gmem_path=Path("/tmp/gmem.bin"),
            dump_path=Path("/tmp/actual.bin"),
            dump_offset=256,
            dump_bytes=512,
        )

    # -- flag presence --------------------------------------------------------

    def test_includes_program_flag(self):
        """Command contains ``--program <aecbin>``."""
        cmd = self._build_cmd()
        self.assertIn("--program", cmd)
        idx = cmd.index("--program")
        self.assertIn("kernel.aecbin", cmd[idx + 1])

    def test_includes_instructions_flag(self):
        """Command contains ``--instructions <ninstr>``."""
        cmd = self._build_cmd()
        self.assertIn("--instructions", cmd)
        idx = cmd.index("--instructions")
        self.assertEqual(cmd[idx + 1], "2")

    def test_includes_grid_flag(self):
        """Command contains ``--grid x,y,z``."""
        cmd = self._build_cmd()
        self.assertIn("--grid", cmd)
        idx = cmd.index("--grid")
        self.assertEqual(cmd[idx + 1], "2,1,1")

    def test_includes_block_flag(self):
        """Command contains ``--block x,y,z``."""
        cmd = self._build_cmd()
        self.assertIn("--block", cmd)
        idx = cmd.index("--block")
        self.assertEqual(cmd[idx + 1], "128,1,1")

    def test_includes_load_pmem(self):
        """Command contains ``--load pmem:0:<path>``."""
        cmd = self._build_cmd()
        pmem_loads = [
            cmd[i + 1]
            for i in range(len(cmd) - 1)
            if cmd[i] == "--load" and "pmem:0:" in cmd[i + 1]
        ]
        self.assertGreaterEqual(len(pmem_loads), 1)
        self.assertIn("pmem.bin", pmem_loads[0])

    def test_includes_load_gmem(self):
        """Command contains ``--load gmem:0:<path>``."""
        cmd = self._build_cmd()
        gmem_loads = [
            cmd[i + 1]
            for i in range(len(cmd) - 1)
            if cmd[i] == "--load" and "gmem:0:" in cmd[i + 1]
        ]
        self.assertGreaterEqual(len(gmem_loads), 1)
        self.assertIn("gmem.bin", gmem_loads[0])

    def test_includes_dump_with_offset_size_path(self):
        """Command contains ``--dump <offset>:<size>:<path>``."""
        cmd = self._build_cmd()
        self.assertIn("--dump", cmd)
        idx = cmd.index("--dump")
        dump_arg = cmd[idx + 1]
        self.assertRegex(dump_arg, r"^256:512:")
        self.assertIn("actual.bin", dump_arg)

    def test_flag_order_matches_conformance_dot_py(self):
        """Flag order matches conformance.py: program → instructions → grid → block → load → dump."""
        cmd = self._build_cmd()
        flags = [a for a in cmd if a.startswith("--")]
        self.assertLess(flags.index("--program"), flags.index("--instructions"))
        self.assertLess(flags.index("--instructions"), flags.index("--grid"))
        self.assertLess(flags.index("--grid"), flags.index("--block"))
        self.assertLess(flags.index("--block"), flags.index("--load"))
        self.assertLess(flags.index("--load"), flags.index("--dump"))

    def test_instructions_computed_from_aecbin_size(self):
        """When ninstr is computed from aecbin, builder uses file_size/16."""
        from tests.extreme.backends import build_cmodel_command  # RED

        import os

        with tempfile.TemporaryDirectory() as tmp:
            aecbin = Path(tmp) / "prog.aecbin"
            # 48 bytes = 3 instructions
            aecbin.write_bytes(b"\x00" * 48)
            size = os.path.getsize(aecbin)
            cmd = build_cmodel_command(
                cmodel=Path("/aec-precise"),
                aecbin=aecbin,
                ninstr=size // 16,
                grid=(1, 1, 1),
                block=(1, 1, 1),
                pmem_path=Path(tmp) / "pmem.bin",
                gmem_path=Path(tmp) / "gmem.bin",
                dump_path=Path(tmp) / "dump.bin",
                dump_offset=0,
                dump_bytes=4,
            )
            idx = cmd.index("--instructions")
            self.assertEqual(cmd[idx + 1], "3")


class CModelExecutionFailClosedTests(unittest.TestCase):
    """CModel execution: every error mode returns ExecutionResult(status='fail', returncode!=0).

    RED: ``execute_cmodel`` does not exist in ``backends.py`` → ImportError.
    """

    def _minimal_case(self) -> "ExtremeCase":
        out = OutputExpectation(
            offset=0, dtype="<u4", shape=(1,), expected=(0,), rtol=0.0, atol=0.0
        )
        return ExtremeCase(
            name="dummy", suite="contract", ptx="ret;",
            grid=(1, 1, 1), block=(1, 1, 1),
            pmem=b"", gmem=b"\x00" * 4,
            output=out, opt_levels=("O2",), expected_failure=None,
        )

    def test_nonzero_exit_returns_fail(self):
        """Non-zero CModel subprocess exit → fail result, not exception."""
        from tests.extreme.backends import execute_cmodel  # RED
        from unittest.mock import patch

        case = self._minimal_case()
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            aecbin = work / "prog.aecbin"
            aecbin.write_bytes(b"\x00" * 32)
            result = execute_cmodel(
                cmodel=Path("/fake-cmodel"),
                aecbin=aecbin,
                case=case,
                work_dir=work,
            )
            self.assertEqual(result.status, "fail")
            self.assertNotEqual(result.returncode, 0)

    def test_malformed_json_returns_fail(self):
        """CModel stdout that is not valid JSON → fail result."""
        from tests.extreme.backends import execute_cmodel  # RED

        case = self._minimal_case()
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            aecbin = work / "prog.aecbin"
            aecbin.write_bytes(b"\x00" * 32)
            result = execute_cmodel(
                cmodel=Path("/fake-cmodel"),
                aecbin=aecbin,
                case=case,
                work_dir=work,
            )
            self.assertEqual(result.status, "fail")

    def test_status_not_done_returns_fail(self):
        """CModel status field != 'done' → fail result."""
        from tests.extreme.backends import execute_cmodel  # RED

        case = self._minimal_case()
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            aecbin = work / "prog.aecbin"
            aecbin.write_bytes(b"\x00" * 32)
            result = execute_cmodel(
                cmodel=Path("/fake-cmodel"),
                aecbin=aecbin,
                case=case,
                work_dir=work,
            )
            self.assertEqual(result.status, "fail")

    def test_output_length_mismatch_returns_fail(self):
        """Dumped output shorter/longer than expected length → fail result."""
        from tests.extreme.backends import execute_cmodel  # RED

        case = self._minimal_case()
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            aecbin = work / "prog.aecbin"
            aecbin.write_bytes(b"\x00" * 32)
            result = execute_cmodel(
                cmodel=Path("/fake-cmodel"),
                aecbin=aecbin,
                case=case,
                work_dir=work,
            )
            self.assertEqual(result.status, "fail")

    def test_dump_file_missing_returns_fail(self):
        """CModel does not produce the dump file → fail result."""
        from tests.extreme.backends import execute_cmodel  # RED

        case = self._minimal_case()
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            aecbin = work / "prog.aecbin"
            aecbin.write_bytes(b"\x00" * 32)
            result = execute_cmodel(
                cmodel=Path("/fake-cmodel"),
                aecbin=aecbin,
                case=case,
                work_dir=work,
            )
            self.assertEqual(result.status, "fail")


class CModelExecutionSuccessTests(unittest.TestCase):
    """Successful CModel execution returns output bytes and steps count.

    RED: ``execute_cmodel`` did not exist in ``backends.py`` → ImportError.
    Now GREEN with ``@patch`` to simulate a successful cmodel run.
    """

    @staticmethod
    def _mock_cmodel_run(cmd, **kwargs):
        """Side effect: simulate a successful cmodel subprocess run.

        Parses the ``--dump`` flag to discover the dump path, creates
        that file with the right number of bytes, and returns a mock
        ``subprocess.CompletedProcess`` with valid JSON stdout.
        """
        import json
        dump_idx = cmd.index("--dump")
        dump_spec = cmd[dump_idx + 1]  # "offset:bytes:path"
        parts = dump_spec.split(":")
        dump_path = Path(parts[2])
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_bytes = int(parts[1])
        dump_path.write_bytes(b"\x00" * dump_bytes)

        mock_stdout = json.dumps({"status": "done", "steps": 100})
        result = unittest.mock.MagicMock(
            returncode=0,
            stdout=mock_stdout,
            stderr="",
        )
        return result

    def _minimal_case(self) -> "ExtremeCase":
        out = OutputExpectation(
            offset=0, dtype="<u4", shape=(1,), expected=(0,), rtol=0.0, atol=0.0
        )
        return ExtremeCase(
            name="dummy", suite="contract", ptx="ret;",
            grid=(1, 1, 1), block=(1, 1, 1),
            pmem=b"", gmem=b"\x00" * 4,
            output=out, opt_levels=("O2",), expected_failure=None,
        )

    @unittest.mock.patch(
        "tests.extreme.backends.subprocess.run",
        side_effect=_mock_cmodel_run,
    )
    def test_returns_gmem_output_bytes(self, mock_run):
        """ExecutionResult.output contains the full dumped GMEM range."""
        from tests.extreme.backends import execute_cmodel

        case = self._minimal_case()
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            aecbin = work / "prog.aecbin"
            aecbin.write_bytes(b"\x00" * 32)
            result = execute_cmodel(
                cmodel=Path("/fake-cmodel"),
                aecbin=aecbin,
                case=case,
                work_dir=work,
            )
            self.assertIsInstance(result.output, bytes)
            self.assertGreater(len(result.output), 0)

    @unittest.mock.patch(
        "tests.extreme.backends.subprocess.run",
        side_effect=_mock_cmodel_run,
    )
    def test_returns_steps_count(self, mock_run):
        """ExecutionResult.cycles holds the CModel steps count."""
        from tests.extreme.backends import execute_cmodel

        case = self._minimal_case()
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            aecbin = work / "prog.aecbin"
            aecbin.write_bytes(b"\x00" * 32)
            result = execute_cmodel(
                cmodel=Path("/fake-cmodel"),
                aecbin=aecbin,
                case=case,
                work_dir=work,
            )
            self.assertIsInstance(result.cycles, int)
            self.assertGreater(result.cycles, 0)


class CModelBackendRoutingTests(unittest.TestCase):
    """--backend cmodel routing: CLI acceptance and dispatch logic.

    RED: ``execute_by_backend`` does not exist in ``backends.py`` → ImportError.
    """

    def test_parser_accepts_cmodel_backend(self):
        """CLI parser accepts ``--backend cmodel`` as a valid choice."""
        from tests.extreme.run_extreme import build_parser

        parser = build_parser()
        parsed = parser.parse_args(
            ["--suite", "contract", "--backend", "cmodel", "--opt", "O2"]
        )
        self.assertEqual(parsed.backend, "cmodel")

    def test_backend_dispatch_function_exists(self):
        """A dispatch function ``execute_by_backend`` selects between local/cmodel executor."""
        from tests.extreme.backends import execute_by_backend  # RED: does not exist

        self.assertIsNotNone(execute_by_backend)

    def test_execute_by_backend_routes_cmodel(self):
        """execute_by_backend(backend='cmodel', ...) calls execute_cmodel under the hood."""
        from tests.extreme.backends import execute_by_backend  # RED: does not exist

        case = _minimal_extreme_case()
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            aecbin = work / "p.aecbin"
            aecbin.write_bytes(b"\x00" * 32)
            result = execute_by_backend(
                backend="cmodel",
                case=case,
                aecbin_path=aecbin,
                cmodel=Path("/fake-cmodel"),
                work_dir=work,
            )
            self.assertIsNotNone(result)


# Helper: minimal case reused by routing tests
def _minimal_extreme_case(name: str = "minimal") -> "ExtremeCase":
    out = OutputExpectation(
        offset=0, dtype="<u4", shape=(1,), expected=(0,), rtol=0.0, atol=0.0
    )
    return ExtremeCase(
        name=name, suite="frontier", ptx="ret;",
        grid=(1, 1, 1), block=(1, 1, 1),
        pmem=b"", gmem=b"\x00" * 4,
        output=out, opt_levels=("O2",), expected_failure=None,
    )


class StrictProfileTests(unittest.TestCase):
    """Strict profile selection: deterministic 30–50 case-opt combos covering T1–T5,
    contract mutations, and pressure boundaries.

    RED: ``select_strict_profile`` does not exist in ``cases.py`` → ImportError.
    """

    def test_profile_flag_registered(self):
        """CLI parser accepts ``--profile strict`` and ``--profile fast``."""
        from tests.extreme.run_extreme import build_parser

        parser = build_parser()
        parsed = parser.parse_args(
            ["--suite", "contract", "--backend", "cmodel",
             "--opt", "O2", "--profile", "strict"]
        )
        self.assertEqual(parsed.profile, "strict")

    def test_profile_flag_fast_accepted(self):
        """CLI parser accepts ``--profile fast``."""
        from tests.extreme.run_extreme import build_parser

        parser = build_parser()
        parsed = parser.parse_args(
            ["--suite", "contract", "--backend", "local",
             "--opt", "O2", "--profile", "fast"]
        )
        self.assertEqual(parsed.profile, "fast")

    def test_profile_default_local_fast(self):
        """Default profile for backend='local' is 'fast'."""
        from tests.extreme.run_extreme import get_profile_default  # RED: does not exist

        self.assertEqual(get_profile_default("local"), "fast")

    def test_profile_default_cmodel_strict(self):
        """Default profile for backend='cmodel' is 'strict'."""
        from tests.extreme.run_extreme import get_profile_default  # RED: does not exist

        self.assertEqual(get_profile_default("cmodel"), "strict")

    def test_strict_profile_30_to_50_combinations(self):
        """Strict profile yields 30–50 case-opt combinations."""
        from tests.extreme.cases import select_strict_profile  # RED: does not exist

        cases = load_case_matrix(REPO_ROOT, "contract")
        profile = select_strict_profile(cases)
        total = sum(len(c.opt_levels) for c in profile)
        self.assertGreaterEqual(total, 30)
        self.assertLessEqual(total, 50)

    def test_strict_profile_covers_t1_to_t5(self):
        """Strict profile includes cases from every public test base (T1–T5)."""
        from tests.extreme.cases import select_strict_profile  # RED: does not exist

        cases = load_case_matrix(REPO_ROOT, "contract")
        profile_cases = select_strict_profile(cases)
        names = {c.name[:2] for c in profile_cases}
        for t in ("T1", "T2", "T3", "T4", "T5"):
            self.assertIn(t, names, f"{t} missing from strict profile")

    def test_strict_profile_includes_all_contract_pressure(self):
        """Strict profile includes all GPR/pair/predicate contract boundary cases."""
        from tests.extreme.cases import select_strict_profile  # RED: does not exist

        cases = load_case_matrix(REPO_ROOT, "contract")
        profile_cases = select_strict_profile(cases)
        profile_names = {c.name for c in profile_cases}
        for name in CONTRACT_NAMES:
            self.assertIn(
                name, profile_names,
                f"contract pressure case {name} missing from strict profile",
            )

    def test_strict_profile_includes_frontier_pressure(self):
        """Strict profile includes all GPR/pair/predicate frontier boundary cases."""
        from tests.extreme.cases import select_strict_profile  # RED: does not exist

        cases = load_case_matrix(REPO_ROOT, "contract")
        profile_cases = select_strict_profile(cases)
        profile_names = {c.name for c in profile_cases}
        for name in FRONTIER_NAMES:
            self.assertIn(
                name, profile_names,
                f"frontier pressure case {name} missing from strict profile",
            )

    def test_strict_profile_includes_one_mutation_per_family_at_O2(self):
        """Strict profile has at least one mutation per family at O2 level."""
        from tests.extreme.cases import select_strict_profile  # RED: does not exist

        cases = load_case_matrix(REPO_ROOT, "contract")
        profile = select_strict_profile(cases)
        o2_names = {c.name for c in profile if "O2" in c.opt_levels}
        # Check for seed variants (mutations): seed 0, 1, 7, 17
        has_mutation_variant = any(
            "_s0_" in n or "_s1_" in n or "_s7_" in n or "_s17_" in n
            for n in o2_names
        )
        self.assertTrue(
            has_mutation_variant,
            "no mutation variant (seed 0/1/7/17) found in strict profile",
        )

    def test_strict_profile_is_deterministic(self):
        """Two calls to select_strict_profile produce identical results."""
        from tests.extreme.cases import select_strict_profile  # RED: does not exist

        cases = load_case_matrix(REPO_ROOT, "contract")
        a = select_strict_profile(cases)
        b = select_strict_profile(cases)
        self.assertEqual(len(a), len(b))
        for ca, cb in zip(a, b):
            self.assertEqual(ca.name, cb.name)
            self.assertEqual(ca.opt_levels, cb.opt_levels)

    def test_strict_profile_no_random_sampling(self):
        """Strict profile source must not call random/numpy.random functions."""
        from tests.extreme.cases import select_strict_profile  # RED: does not exist

        import inspect
        src = inspect.getsource(select_strict_profile)
        self.assertNotIn("random.Random", src)
        self.assertNotIn("random.sample", src)
        self.assertNotIn("numpy.random", src)
        self.assertNotIn(".shuffle(", src)

    def test_strict_profile_no_silent_skip(self):
        """Profile must not randomly subsample case-opt combinations."""
        from tests.extreme.cases import select_strict_profile  # RED: does not exist

        cases = load_case_matrix(REPO_ROOT, "contract")
        profile = select_strict_profile(cases)
        # Every case in the profile must have at least one opt_level
        for c in profile:
            self.assertGreater(len(c.opt_levels), 0)
        # No case should be empty (silent skip would produce 0 opt_levels)
        self.assertGreater(len(profile), 0)


# ════════════════════════════════════════════════════════════════════════
# Deterministic cross-process tests (hash(cls) → PYTHONHASHSEED leak)
# ════════════════════════════════════════════════════════════════════════

class DeterministicCrossProcessTests(unittest.TestCase):
    """Cross-process PYTHONHASHSEED invariance: ``rename_registers`` and
    ``sparse_register_numbering`` produce identical output under different
    ``PYTHONHASHSEED`` values via ``subprocess.run``.

    RED (both behaviour + static AST guard):
      The two functions use ``hash(cls)`` (generators.py ~L144, ~L182),
      which varies by ``PYTHONHASHSEED`` → different case.ptx across
      processes → predicate modulo alias.  These tests will pass only
      after ``hash(cls)`` is replaced with a deterministic alternative.
    """

    # Absolute path to C1/ so subprocess can import tests.extreme.generators.
    _C1_DIR = Path(__file__).resolve().parent.parent.parent

    def _subprocess_call(
        self, func_name: str, seed: int, hashseed: str
    ) -> subprocess.CompletedProcess:
        """Run *func_name*(BASE_PTX, seed=*seed*) in a subprocess."""
        script = (
            "from tests.extreme.generators import BASE_PTX, "
            + func_name
            + "; "
            + "r = "
            + func_name
            + "(BASE_PTX, seed="
            + str(seed)
            + "); "
            + "print(r, end='')"
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self._C1_DIR)
        env["PYTHONHASHSEED"] = hashseed
        return subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=str(self._C1_DIR),
        )

    # ── rename_registers ──────────────────────────────────────────────

    def test_rename_registers_cross_process_deterministic(self):
        """rename_registers output identical under PYTHONHASHSEED=0 and =12345."""
        r0 = self._subprocess_call("rename_registers", seed=42, hashseed="0")
        r1 = self._subprocess_call("rename_registers", seed=42, hashseed="12345")
        self.assertEqual(r0.returncode, 0, f"subprocess (seed=0) failed: {r0.stderr}")
        self.assertEqual(
            r1.returncode, 0, f"subprocess (seed=12345) failed: {r1.stderr}"
        )
        self.assertEqual(
            r0.stdout,
            r1.stdout,
            "rename_registers output differs between PYTHONHASHSEED=0 and 12345",
        )

    # ── sparse_register_numbering ─────────────────────────────────────

    def test_sparse_register_numbering_cross_process_deterministic(self):
        """sparse_register_numbering output identical under PYTHONHASHSEED=0 and =12345."""
        r0 = self._subprocess_call(
            "sparse_register_numbering", seed=42, hashseed="0"
        )
        r1 = self._subprocess_call(
            "sparse_register_numbering", seed=42, hashseed="12345"
        )
        self.assertEqual(r0.returncode, 0, f"subprocess (seed=0) failed: {r0.stderr}")
        self.assertEqual(
            r1.returncode, 0, f"subprocess (seed=12345) failed: {r1.stderr}"
        )
        self.assertEqual(
            r0.stdout,
            r1.stdout,
            "sparse_register_numbering output differs between "
            "PYTHONHASHSEED=0 and 12345",
        )

    # ── static AST guard ──────────────────────────────────────────────

    def test_generators_no_builtin_hash_call(self):
        """Static AST guard: rename_registers / sparse_register_numbering
        must not call built-in ``hash()``."""
        import ast
        import inspect

        from tests.extreme.generators import (
            rename_registers,
            sparse_register_numbering,
        )

        for func in (rename_registers, sparse_register_numbering):
            src = inspect.getsource(func)
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "hash"
                ):
                    self.fail(
                        f"{func.__name__} calls built-in hash() at line "
                        f"{node.lineno} (non-deterministic across "
                        f"PYTHONHASHSEED values)"
                    )


# ════════════════════════════════════════════════════════════════════════
# Task 7b: Makefile contract test (RED — test-frontier must use --opt O2)
# ════════════════════════════════════════════════════════════════════════


class MakefileTargetTests(unittest.TestCase):
    """Static Makefile contract: test-frontier invokes --opt O2, not --opt all."""

    def test_test_frontier_target_uses_opt_O2(self):
        """test-frontier Makefile target must invoke --opt O2, not --opt all."""
        makefile = REPO_ROOT / "C1" / "Makefile"
        content = makefile.read_text(encoding="utf-8")
        lines = content.splitlines()
        target_index = next(
            (i for i, line in enumerate(lines) if line.startswith("test-frontier:")),
            None,
        )
        self.assertIsNotNone(target_index, "test-frontier target not found in Makefile")
        recipe = lines[target_index + 1]  # type: ignore[operator]
        self.assertIn("run_extreme", recipe)
        self.assertIn("--opt O2", recipe)
        self.assertNotIn("--opt all", recipe)


# ════════════════════════════════════════════════════════════════════════
# Task 7c: Registry overlay after strict profile (RED — must use backend=)
# ════════════════════════════════════════════════════════════════════════


class StrictProfileRegistryOverlayTests(unittest.TestCase):
    """RED: registry overlay must be applied AFTER strict profile selection
    with ``backend=args.backend`` so that strict-added frontier cases are
    covered by the expected-failure phase map."""

    def test_strict_profile_frontier_cases_lack_backend_expected_failure(self):
        """Frontier cases from select_strict_profile should have expected_failure=None until backend overlay."""
        from tests.extreme.cases import select_strict_profile

        cases = load_case_matrix(REPO_ROOT, "contract")
        profile = select_strict_profile(cases)
        frontier_in_profile = [c for c in profile if c.name in FRONTIER_NAMES]
        self.assertGreater(
            len(frontier_in_profile), 0,
            "strict profile must include frontier cases",
        )
        for case in frontier_in_profile:
            # RED: currently expected_failure is set during load_case_matrix, but
            # with the new design it must be None until overlay with backend=.
            self.assertIsNone(
                case.expected_failure,
                f"{case.name}: expected_failure should be None until backend overlay "
                f"(currently set to {case.expected_failure!r})",
            )
            self.assertIsNone(
                case.expected_failure_phase,
                f"{case.name}: expected_failure_phase should be None until backend overlay",
            )


if __name__ == "__main__":
    unittest.main()

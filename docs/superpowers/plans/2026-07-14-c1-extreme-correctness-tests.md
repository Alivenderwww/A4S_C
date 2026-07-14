# C1 Extreme Correctness Tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic mutation and true live-pressure tests with fail-closed contract/frontier gates, fast local simulation, and full-output remote CModel verification.

**Architecture:** A new `C1/tests/extreme` package owns immutable case descriptions, PTX generators, backend adapters, result classification, and failure artifacts. The local and CModel backends consume the same cases; contract failures always fail, while only explicitly registered frontier defects become XFAIL. Existing public tests are repaired first so zero discovery and obsolete container checks cannot report success.

**Tech Stack:** Python 3.12/3.13 standard library, NumPy, existing `C1/sim/aec_sim.py`, official `aec-precise` on the remote Linux server, Bash/Make, C++11 compiler binaries.

**Version-control constraint:** Do not commit, amend, or push unless the user explicitly requests it. Use `jj` for repository status/diff checks.

---

## File Map

- Create `C1/tests/__init__.py`: make test helpers importable.
- Create `C1/tests/extreme/__init__.py`: extreme-suite package marker.
- Create `C1/tests/extreme/cases.py`: immutable case/output models, current manifest loaders, PMEM/GMEM construction, reference formulas, deterministic matrices.
- Create `C1/tests/extreme/generators.py`: semantic-preserving PTX mutations and cross-block GPR/pair/predicate pressure kernels.
- Create `C1/tests/extreme/backends.py`: cross-platform compiler selection, local simulator adapter, remote CModel adapter.
- Create `C1/tests/extreme/runner.py`: execution, comparison, result classification, XFAIL policy, artifact capture, summaries.
- Create `C1/tests/extreme/expected_failures.json`: narrow frontier defect registry.
- Create `C1/tests/extreme/run_extreme.py`: CLI for suite/backend/optimization/seed/artifact selection.
- Create `C1/tests/extreme/test_extreme_unit.py`: framework and generator unit tests.
- Modify `C1/tests/run_public.sh`: current `T*/kernel.ptx` discovery, raw-stream validation, zero-case failure.
- Modify `C1/Makefile`: local contract/frontier commands.
- Modify `C1/README.md`: current test commands and local/remote evidence distinction.

---

### Task 1: Repair the Existing Public Test Gate

**Files:**
- Modify: `C1/tests/run_public.sh:1-65`
- Test: `C1/tests/run_public.sh`

- [ ] **Step 1: Run the existing gate and capture the false success**

Run:

```powershell
rtk make test
```

Working directory: `C1`

Expected current evidence:

```text
(no testcases found ...)
summary: 1 passed, 0 failed
```

The command currently exits zero despite discovering no PTX cases.

- [ ] **Step 2: Replace obsolete discovery and magic validation**

Implement the loop around current testcase directories:

```bash
mapfile -t cases < <(find "$TESTS" -mindepth 2 -maxdepth 2 \
  -type f -name kernel.ptx -print | sort)

if [ "${#cases[@]}" -eq 0 ]; then
  echo "FAIL: no public kernel.ptx files found at $TESTS" >&2
  exit 1
fi

for f in "${cases[@]}"; do
  name="$(basename "$(dirname "$f")")"
  out="$WORK/$name.aecbin"
  report="$WORK/$name.json"

  if "$CC" "$f" -O2 -o "$out" --report "$report" >/dev/null 2>&1; then
    note_pass "$name compile -O2"
  else
    note_fail "$name compile -O2"
    continue
  fi

  size="$(wc -c < "$out")"
  if [ "$size" -gt 0 ] && [ $((size % 16)) -eq 0 ]; then
    note_pass "$name raw-aecbin"
  else
    note_fail "$name raw-aecbin(size=$size)"
  fi

  if "$OD" "$out" > "$WORK/$name.asm" 2>/dev/null && [ -s "$WORK/$name.asm" ]; then
    note_pass "$name objdump"
  else
    note_fail "$name objdump"
  fi

  for lvl in O0 O3; do
    if "$CC" "$f" -$lvl -o "$WORK/$name.$lvl.aecbin" >/dev/null 2>&1; then
      note_pass "$name -$lvl"
    else
      note_fail "$name -$lvl"
    fi
  done
done
```

Remove the `AEC1` header check. The current contract is a non-empty raw stream whose size is divisible by 16.

- [ ] **Step 3: Run the repaired gate**

Run:

```powershell
rtk make test
```

Expected: five `T*/kernel.ptx` cases are discovered; encoder plus four checks per case produce `21 passed, 0 failed` and exit zero.

- [ ] **Step 4: Verify zero discovery fails**

Run the script with a temporary invalid testcase root exposed through a test-only environment override added as:

```bash
TESTS="${AEC_PUBLIC_TESTS:-$ROOT/../public/Track-C/C1-compiler/testcases}"
```

Command:

```powershell
$env:AEC_PUBLIC_TESTS='Z:\missing-c1-tests'; bash tests/run_public.sh; $code=$LASTEXITCODE; Remove-Item Env:AEC_PUBLIC_TESTS; exit $code
```

Expected: non-zero exit with `no public kernel.ptx files found`.

---

### Task 2: Define the Case Model and Fail-Closed CLI Skeleton

**Files:**
- Create: `C1/tests/__init__.py`
- Create: `C1/tests/extreme/__init__.py`
- Create: `C1/tests/extreme/cases.py`
- Create: `C1/tests/extreme/runner.py`
- Create: `C1/tests/extreme/run_extreme.py`
- Create: `C1/tests/extreme/test_extreme_unit.py`

- [ ] **Step 1: Write failing unit tests for empty discovery and immutable models**

Add:

```python
from pathlib import Path
import tempfile
import unittest

from tests.extreme.cases import ExtremeCase, OutputExpectation
from tests.extreme.runner import discover_cases


class CaseModelTests(unittest.TestCase):
    def test_empty_discovery_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "zero extreme cases"):
                discover_cases(Path(tmp), suite="contract")

    def test_case_model_is_immutable(self):
        out = OutputExpectation(offset=0, dtype="<u4", shape=(1,), expected=(7,), rtol=0.0, atol=0.0)
        case = ExtremeCase(
            name="one", suite="contract", ptx="ret;", grid=(1, 1, 1),
            block=(1, 1, 1), pmem=b"", gmem=b"\0" * 4,
            output=out, opt_levels=("O2",), expected_failure=None,
        )
        with self.assertRaises(Exception):
            case.name = "changed"
```

- [ ] **Step 2: Run tests and verify they fail because the modules do not exist**

Run:

```powershell
py -3.13 -m unittest tests.extreme.test_extreme_unit -v
```

Working directory: `C1`

Expected: import failure for `tests.extreme.cases`.

- [ ] **Step 3: Implement the immutable models and discovery contract**

Use frozen dataclasses:

```python
@dataclass(frozen=True)
class OutputExpectation:
    offset: int
    dtype: str
    shape: Tuple[int, ...]
    expected: Tuple[float, ...]
    rtol: float
    atol: float


@dataclass(frozen=True)
class ExtremeCase:
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
```

`discover_cases()` must reject an empty result:

```python
def discover_cases(root: Path, suite: str) -> List[ExtremeCase]:
    cases = load_case_matrix(root, suite)
    if not cases:
        raise RuntimeError("zero extreme cases discovered for suite %s" % suite)
    return cases
```

The initial `load_case_matrix()` may return an empty list; the next tasks populate it.

- [ ] **Step 4: Add CLI validation**

`run_extreme.py` accepts only:

```text
--suite contract|frontier
--backend local|cmodel
--opt O0|O2|O3|all
--seed INTEGER
--artifacts PATH
--list
```

It resolves the repository root from `__file__`, calls `discover_cases()`, and returns non-zero on every exception after printing a concise error to stderr.

- [ ] **Step 5: Run unit tests**

Run:

```powershell
py -3.13 -m unittest tests.extreme.test_extreme_unit -v
```

Expected: both model/discovery tests pass.

---

### Task 3: Load Current Public Cases and Build Exact Oracles

**Files:**
- Modify: `C1/tests/extreme/cases.py`
- Modify: `C1/tests/extreme/test_extreme_unit.py`
- Read/reuse: `C1/sim/run_manifest.py:37-114`
- Test data: `public/Track-C/C1-compiler/testcases/T*/{kernel.ptx,manifest.json}`

- [ ] **Step 1: Write failing tests for PMEM packing and current layout**

Add tests asserting:

```python
def test_pmem_natural_alignment(self):
    params = [("u64", 256), ("u64", 512), ("u32", 7)]
    packed = pack_pmem(params)
    self.assertEqual(len(packed), 24)
    self.assertEqual(struct.unpack_from("<Q", packed, 0)[0], 256)
    self.assertEqual(struct.unpack_from("<Q", packed, 8)[0], 512)
    self.assertEqual(struct.unpack_from("<I", packed, 16)[0], 7)

def test_current_public_cases_are_discovered(self):
    cases = load_public_contract_cases(REPO_ROOT)
    self.assertEqual([c.name for c in cases], ["T1", "T2", "T3", "T4", "T5"])
    self.assertTrue(all(".version 9.3" in c.ptx for c in cases))
```

- [ ] **Step 2: Run tests and verify missing functions fail**

Run:

```powershell
py -3.13 -m unittest tests.extreme.test_extreme_unit -v
```

Expected: `pack_pmem` and `load_public_contract_cases` are missing.

- [ ] **Step 3: Implement manifest loading and buffer construction**

Requirements:

- Read `T*/manifest.json` and sibling `kernel.ptx` directly.
- Lay buffers consecutively in one flat `bytearray`.
- Use manifest seeds with `numpy.random.default_rng(seed)`.
- Pack pointers as GMEM byte offsets.
- Pack `u64` with `<Q`, `u32` with `<I`, natural alignment, final 8-byte padding.
- Reduce public sizes for local execution while preserving formulas and boundary semantics; record overrides in the case name.

Reference functions must force FP32 operation order:

```python
def ref_t2(x, y):
    s = (x + y).astype(np.float32)
    return (s * s).astype(np.float32) + x

def ref_t4(a, b, c, d):
    lhs = ((a + b).astype(np.float32) * (c - d).astype(np.float32)).astype(np.float32)
    rhs = ((a * c).astype(np.float32) * (b + d).astype(np.float32)).astype(np.float32)
    return (lhs + rhs).astype(np.float32)
```

Use manifest tolerances: T1 `1e-6`, T2–T4 `1e-5`, T5 `1e-4`.

- [ ] **Step 4: Add shape and PMEM assertions for all five cases**

Assert exact parameter block lengths:

```text
T1=32, T2=32, T3=40, T4=48, T5=40 bytes
```

Assert output ranges stay inside GMEM and block products never exceed 256.

- [ ] **Step 5: Run unit tests**

Expected: current paths, packing, formulas, and case counts pass without compiling C++.

---

### Task 4: Add Deterministic Contract Mutations

**Files:**
- Create: `C1/tests/extreme/generators.py`
- Modify: `C1/tests/extreme/cases.py`
- Modify: `C1/tests/extreme/test_extreme_unit.py`
- Read/reuse: `C1/sim/mutate.py`

- [ ] **Step 1: Write failing determinism and semantic-shape tests**

Cover:

```python
def test_mutations_are_seed_deterministic(self):
    a = mutate_contract_source(BASE_PTX, seed=17)
    b = mutate_contract_source(BASE_PTX, seed=17)
    self.assertEqual(a, b)

def test_register_rename_keeps_special_registers(self):
    out = rename_registers(BASE_PTX, seed=3)
    self.assertIn("%tid.x", out)
    self.assertIn("%ctaid.x", out)

def test_reordered_blocks_have_explicit_fallthrough(self):
    out = reorder_blocks(LOOP_PTX, seed=4)
    self.assertIn("bra ", out)
```

- [ ] **Step 2: Run tests and confirm missing generator APIs**

- [ ] **Step 3: Port only current-contract-safe mutations**

Implement deterministic functions for:

- register renaming;
- sparse register numbering while updating `.reg` declarations;
- block reordering with explicit fallthrough branches;
- dead FP32/integer computation insertion;
- comments and whitespace variation;
- equivalent address forms (`mul.wide + add.u64` versus `mad.lo`-based 32-bit offset construction where the spec permits it).

Do not port old FP16/BF16 GEMM transforms.

- [ ] **Step 4: Build the contract matrix**

Generate fixed combinations rather than a Cartesian explosion:

```text
seeds: 0, 1, 7, 17
launch N: 1,31,32,33,255,256,257,1000
GEMM dimensions: 0/1, 15/16/17, 31/32/33 representative triples
opt levels: O0/O2/O3 on pass-sensitive cases; O2 on remaining cases
```

The resulting local contract matrix must contain at least 100 case/optimization combinations and remain deterministic.

- [ ] **Step 5: Add a count test**

```python
def test_contract_matrix_has_at_least_100_combinations(self):
    cases = load_case_matrix(REPO_ROOT, "contract")
    total = sum(len(c.opt_levels) for c in cases)
    self.assertGreaterEqual(total, 100)
```

- [ ] **Step 6: Run unit tests**

Expected: deterministic mutation tests and matrix-count test pass.

---

### Task 5: Generate True Cross-Block Pressure Cases

**Files:**
- Modify: `C1/tests/extreme/generators.py`
- Modify: `C1/tests/extreme/cases.py`
- Create: `C1/tests/extreme/expected_failures.json`
- Modify: `C1/tests/extreme/test_extreme_unit.py`

- [ ] **Step 1: Write failing structural tests for forced liveness**

Add:

```python
def test_gpr_pressure_crosses_block_boundary(self):
    src = generate_gpr_pressure(64)
    defs, uses = src.split("REDUCE:")
    for i in range(64):
        self.assertIn("%v%d" % i, defs)
        self.assertIn("%v%d" % i, uses)
    self.assertIn("bra REDUCE;", defs)

def test_pair_pressure_uses_b64_values(self):
    src = generate_pair_pressure(120)
    self.assertIn(".reg .b64 %v<120>;", src)

def test_predicate_pressure_has_distinct_predicates(self):
    src = generate_predicate_pressure(9)
    self.assertIn(".reg .pred %p<9>;", src)
```

- [ ] **Step 2: Run tests and confirm generator functions are missing**

- [ ] **Step 3: Implement pressure generators**

Each generator must emit valid PTX 9.3 with:

```ptx
ENTRY:
    // all values defined here
    bra REDUCE;
REDUCE:
    // every value consumed here
    st.global.u32 [%rd_out], %acc;
    ret;
```

Use a CFG boundary because same-block definitions can be sunk next to uses by the scheduler.

Generate boundaries:

- Narrow GPR contract: 64, 128, 192, 240, 252.
- Narrow GPR frontier: 255, 256, 300.
- 64-bit pair contract: 32, 64, 96, 120.
- 64-bit pair frontier: 127, 128, 140.
- Predicate contract: 1, 2, 4, 8.
- Predicate frontier: 9, 12, 16.

Expected results are integer reductions computable independently in Python.

- [ ] **Step 4: Add the initial expected-failure registry**

Create:

```json
{
  "gpr-live-256": {
    "issue": "C1-SPILL-001",
    "phase": "compare",
    "reason": "allocator clamps exhausted intervals instead of inserting LMEM spill/reload"
  },
  "gpr-live-300": {
    "issue": "C1-SPILL-001",
    "phase": "compare",
    "reason": "allocator clamps exhausted intervals instead of inserting LMEM spill/reload"
  },
  "pred-live-9": {
    "issue": "C1-PRED-001",
    "phase": "compare",
    "reason": "predicate ids are masked to three bits and alias after P7"
  }
}
```

Only add pair cases after observing their exact stable failure phase; an unregistered failure remains a hard FAIL.

- [ ] **Step 5: Add registry validation tests**

Reject missing issue IDs, unknown phases, duplicate registry keys, registry keys absent from frontier cases, and contract cases carrying expected failures.

- [ ] **Step 6: Run unit tests**

Expected: structure, boundary count, independent references, and registry validation pass.

---

### Task 6: Implement Local Execution, Full Comparison, and Artifacts

**Files:**
- Create: `C1/tests/extreme/backends.py`
- Modify: `C1/tests/extreme/runner.py`
- Modify: `C1/tests/extreme/run_extreme.py`
- Modify: `C1/tests/extreme/test_extreme_unit.py`
- Reuse: `C1/sim/aec_sim.py`, `C1/sim/aec_decode.py`

- [ ] **Step 1: Write failing tests for compiler selection and result classification**

Test selection order:

```text
bin/aec-cc.exe -> bin/aec-cc -> compiler/aec-cc
```

Test classification table:

```text
contract success -> PASS
contract failure -> FAIL
frontier registered matching failure -> XFAIL
frontier unregistered failure -> FAIL
frontier registered success -> XPASS and non-zero suite result
```

- [ ] **Step 2: Implement compiler and local simulator adapters**

Define:

```python
@dataclass(frozen=True)
class CompileResult:
    returncode: int
    stdout: str
    stderr: str
    aecbin: Optional[Path]
    report: Optional[dict]

@dataclass(frozen=True)
class ExecutionResult:
    returncode: int
    status: str
    output: bytes
    cycles: Optional[int]
    detail: str
```

The compiler adapter always requests `--report`. A zero exit without a non-empty, 16-byte-aligned `.aecbin` or valid report is a compile failure.

The local backend calls:

```python
gmem, cycles, warps = simulate(
    str(aecbin), case.grid, case.block,
    param_block=case.pmem,
    gmem_init=case.gmem,
    strict=True,
)
```

It extracts the complete expected output range.

- [ ] **Step 3: Implement fail-closed comparison**

Comparison rejects:

- output length mismatch;
- shape mismatch;
- NaN/Inf in expected or actual floating output;
- any value outside `np.allclose` with the case's own tolerances;
- exceptions and timeouts.

Record first differing flat index, expected value, actual value, and maximum absolute difference.

- [ ] **Step 4: Implement artifact capture**

Write one directory per `case/opt/backend` containing:

```text
case.ptx
compile.stdout.txt
compile.stderr.txt
compile_report.json
program.aecbin
program.asm
result.json
expected.bin
actual.bin
```

No artifact write failure may convert a test failure into success.

- [ ] **Step 5: Run the local contract suite**

Run:

```powershell
py -3.13 tests/extreme/run_extreme.py --suite contract --backend local --opt all
```

Expected: at least 100 combinations discovered; summary contains no FAIL/XFAIL/XPASS; exit zero; normal runtime target below two minutes.

- [ ] **Step 6: Run the local frontier suite**

Run:

```powershell
py -3.13 tests/extreme/run_extreme.py --suite frontier --backend local --opt O2
```

Expected: registered current defects are XFAIL; unknown failures or XPASS make the command non-zero. Inspect every first-run failure before adding any registry entry.

---

### Task 7: Add the Official CModel Backend and Remote Strict Profile

**Files:**
- Modify: `C1/tests/extreme/backends.py`
- Modify: `C1/tests/extreme/cases.py`
- Modify: `C1/tests/extreme/run_extreme.py`
- Modify: `C1/tests/extreme/test_extreme_unit.py`
- Remote tool: `scripts/remote_exec.py`

- [ ] **Step 1: Write command-construction tests**

Assert the CModel command includes:

```text
--program <aecbin>
--instructions <file_size/16>
--grid x,y,z
--block x,y,z
--load pmem:0:<pmem.bin>
--load gmem:0:<gmem.bin>
--dump <offset>:<full-output-bytes>:<actual.bin>
```

Reject missing CModel binary before executing any case.

- [ ] **Step 2: Implement `CModelBackend`**

Resolve the model from:

```text
<repo>/public/aec-cmodel-release/bin/aec-precise-linux-x86_64
```

For every case, write complete PMEM and GMEM images, execute with a finite timeout and `--max-steps`, parse stdout JSON, require `status == "done"`, and compare the complete dumped output.

- [ ] **Step 3: Add a strict remote profile selector**

The CModel profile contains all architectural boundaries plus representative mutation seeds, initially 30–50 case/optimization combinations:

- all GPR/pair/predicate boundary points;
- all partial-block boundaries;
- all GEMM K boundaries;
- one mutation per family at O2;
- pass-sensitive subset at O0 and O3.

Add `--profile fast|strict`; local defaults to `fast`, CModel defaults to `strict`.

- [ ] **Step 4: Prepare an isolated remote snapshot**

Use `scripts/remote_exec.py` to verify a new target does not exist before upload. Do not overwrite `/home/mig02/A4S/c1test` or another existing checkout. Record:

- source revision/change ID;
- SHA-256 of `cases.py`, `generators.py`, `backends.py`, `runner.py`;
- Python/GCC/CModel paths.

- [ ] **Step 5: Run remote unit and contract tests**

Commands, using the isolated path selected in Step 4:

```powershell
py -3.13 scripts/remote_exec.py "cd <isolated>/C1 && make build && python3 -m unittest tests.extreme.test_extreme_unit -v"
py -3.13 scripts/remote_exec.py "cd <isolated>/C1 && python3 tests/extreme/run_extreme.py --suite contract --backend cmodel --profile strict --opt all"
```

Expected: unit tests pass; every selected contract case compares the full output buffer and exits zero.

- [ ] **Step 6: Run the remote frontier suite**

```powershell
py -3.13 scripts/remote_exec.py "cd <isolated>/C1 && python3 tests/extreme/run_extreme.py --suite frontier --backend cmodel --profile strict --opt O2"
```

Expected: only registry-matched current defects are XFAIL. Do not add an expected failure merely to make the command green; first inspect the PTX, report, CModel status, and output mismatch.

---

### Task 8: Integrate Make Targets and Documentation

**Files:**
- Modify: `C1/Makefile:32-82`
- Modify: `C1/README.md:65-117`
- Test: all local and remote commands from the approved design

- [ ] **Step 1: Add Make targets**

Add phony targets:

```make
.PHONY: test test-public test-extreme test-frontier

test: test-public test-extreme

test-public: build
	bash tests/run_public.sh

test-extreme: build
	python3 tests/extreme/run_extreme.py --suite contract --backend local --profile fast --opt all

test-frontier: build
	python3 tests/extreme/run_extreme.py --suite frontier --backend local --profile fast --opt O2
```

On Windows, document use of `py -3.13` if `python3` is unavailable; do not make the runner silently select a different interpreter.

- [ ] **Step 2: Update README commands and evidence labels**

Document:

```text
make test          = public compile/raw/objdump gate + local contract suite
make test-frontier = registered known-limit reproducers
remote CModel      = Linux server via scripts/remote_exec.py, no local WSL required
```

State that local simulator PASS is not official CModel evidence and that XFAIL is a tracked compiler defect, not a skip.

- [ ] **Step 3: Run the complete local verification**

Run:

```powershell
rtk make selftest
rtk make test-public
py -3.13 -m unittest tests.extreme.test_extreme_unit -v
py -3.13 tests/extreme/run_extreme.py --suite contract --backend local --profile fast --opt all
py -3.13 tests/extreme/run_extreme.py --suite frontier --backend local --profile fast --opt O2
```

Expected:

- encoder 8/8;
- public test discovery count is five and 21 checks pass;
- unit tests pass;
- contract has zero non-PASS results;
- frontier has only registered XFAIL results and no unknown FAIL/XPASS.

- [ ] **Step 4: Run the complete remote verification**

Run through `scripts/remote_exec.py` in the isolated server directory:

```text
GCC 13.3 build
public gate
unit tests
strict CModel contract
strict CModel frontier
```

Capture stdout, stderr, exit codes, source hashes, PASS/FAIL/XFAIL/XPASS counts, and elapsed time.

- [ ] **Step 5: Inspect the final diff and repository status**

Run:

```powershell
jj --no-pager diff --git
jj --no-pager st
```

Confirm:

- no C3 files changed;
- no generated `.aecbin`, reports, inputs, dumps, or remote keys are tracked;
- only the planned C1 test files, Makefile, README, spec, and plan changed;
- no commit or push occurred.

---

## Plan Self-Review

- Spec coverage: contract/frontier layering, deterministic seeds, true cross-block liveness, local/remote backends, full-output comparison, artifact capture, zero-case failure, and no local WSL requirement are all assigned to tasks.
- Scope: compiler fixes remain separate; this plan adds evidence and explicit XFAILs only.
- Type/API consistency: `ExtremeCase`, `OutputExpectation`, `CompileResult`, and `ExecutionResult` are defined before backend/runner use.
- Verification: every module has a failing-test-first step and an exact local or remote command.
- Version control: implementation ends with `jj` inspection and does not commit without explicit user approval.

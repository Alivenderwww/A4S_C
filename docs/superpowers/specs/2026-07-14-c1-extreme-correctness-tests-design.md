# C1 Extreme Correctness Tests Design

**Date:** 2026-07-14

**Status:** Approved design; implementation not started

## Goal

Add deterministic mutation and pressure tests that expose C1 compiler correctness failures near and beyond architectural limits. Run a fast local suite on every change and a stricter full-output suite against the official CModel on the remote server.

This work sits between submission hardening and the current-path full-manifest grader. It improves confidence and defect discovery; it does not claim hidden-grader coverage or an official score.

## Non-goals

- Implementing spill/reload or predicate allocation.
- Fixing compiler defects found by frontier tests in the same change.
- Open-ended PTX grammar fuzzing.
- Testing obsolete Agent, TMUL, multi-precision GEMM, or multi-kernel requirements.
- Treating the self-built simulator as equivalent to the official CModel.

## Approach

Use deterministic PTX generators plus a small set of handwritten adversarial kernels. Every generated case has a fixed seed, explicit expected behavior, and enough metadata to reproduce it independently.

The same case model runs through two backends:

1. **Local simulator backend** for fast development feedback.
2. **Official CModel backend** for authoritative AEC execution semantics on the remote Linux server.

Random grammar fuzzing is deferred because it would require a substantially larger independent PTX semantic oracle and would make generator defects harder to distinguish from compiler defects.

## Suite Policy

### Contract suite

Contract cases are within the current public C1 specification and must all pass. Any compile error, execution error, timeout, missing resource, malformed output, empty output, non-finite result, or value mismatch fails the command.

Contract coverage includes:

- Compilation at `-O0`, `-O2`, and `-O3` for pass-sensitive cases.
- Register renaming, sparse register numbering, comments/whitespace variation, and basic-block reordering.
- Dead-code and unrelated-computation insertion.
- Partial blocks and launch boundaries around 1, 31, 32, 33, 255, 256, 257, and 1000 elements.
- Scalar GEMM boundaries around 0/1, 15/16/17, and 31/32/33 for `M`, `N`, and `K`, including non-divisible shapes.
- Positive and negated predicates, diamond control flow, loop back-edges, and divergent bounds guards.
- Up to eight simultaneously live predicates.
- High but legal 32-bit GPR and 64-bit register-pair pressure.
- Equivalent address-computation forms.
- Repeated loads, load/store alias invalidation, and safe load reuse.
- Fail-closed behavior for unsupported PTX instructions.

### Frontier suite

Frontier cases deliberately exercise known implementation limits. They may fail only when matched by an explicit expected-failure record.

Initial frontier categories:

- More than 255 simultaneously live 32-bit values.
- More 64-bit register pairs than the physical register file can hold.
- More than eight simultaneously live predicates.
- Stable reproducers for spill clamping or predicate aliasing.

Each expected failure records:

- A stable issue identifier.
- The case family and parameter boundary.
- The expected failure phase or output mismatch.
- The reason it is outside the current implementation capability.

The runner reports `PASS`, `FAIL`, `XFAIL`, and `XPASS`. Unknown failures always fail the command. An XPASS is visible and must be reviewed; after the underlying defect is fixed, the case moves into the contract suite.

No case may be silently skipped.

## Pressure Construction

Source-level register declarations do not prove physical pressure because the pre-register-allocation scheduler can move each definition close to its use. Existing `pressure.py` demonstrates this: 300 source values become only six allocated physical registers.

New pressure kernels force a real live set across a control-flow edge:

1. Basic block A defines every pressure value.
2. A terminator transfers control to basic block B.
3. Basic block B consumes every value.
4. Liveness therefore requires all values to be live-out of A and live-in to B.

Separate generators cover:

- Narrow 32-bit GPR values.
- 64-bit values requiring consecutive register pairs.
- Predicate values consumed after the block boundary.

Boundary scans include values immediately below, at, and above the architectural capacity. Tests assert the compiler report and executed result, not only the number of declared registers.

## Case Model

Each case contains:

- Stable name and suite (`contract` or `frontier`).
- Generator family and fixed seed.
- PTX source.
- Grid and block dimensions.
- PMEM values and GMEM buffers.
- Expected output buffers and comparison rules.
- Applicable optimization levels.
- Backend eligibility.
- Optional expected-failure identifier.

Semantic-preserving mutations reuse the base case oracle. Semantic-changing mutations must construct a new independent expected result.

## Backends

### Local simulator backend

- Runs approximately 100–200 deterministic case/optimization combinations.
- Targets a normal execution time below two minutes.
- Compares complete output buffers.
- Is the default developer and `make test` path.
- Clearly labels results as self-built simulator evidence.

### Official CModel backend

- Runs on Linux through the repository's remote execution workflow.
- Linux-only build and CModel commands run directly on the remote server through `scripts/remote_exec.py`; the suite does not require or configure local WSL.
- Runs all architectural boundaries and representative mutation combinations, initially about 30–50 cases.
- Runs all contract cases at `-O2`; runs a pass-sensitive subset at `-O0` and `-O3`.
- Compares complete output buffers with the testcase's specified tolerance.
- Records CModel status, step count, and process exit code.
- Fails if the CModel or testcase resources are unavailable.

The runner must not overwrite an existing remote checkout. Remote execution uses an isolated audit directory whose source identity is recorded before testing.

## Failure Artifacts

For every unexpected failure, preserve:

- Generated PTX.
- Seed and generator parameters.
- Compiler command, stdout, stderr, and exit code.
- Compile report.
- Generated `.aecbin` and disassembly when available.
- Backend, optimization level, and execution status.
- Expected and actual output metadata.
- First differing index and a bounded value excerpt.

Artifacts are written under a gitignored build directory and use stable case names.

## File Boundaries

Create a focused package:

```text
C1/tests/extreme/
├── __init__.py
├── generators.py
├── cases.py
├── backends.py
├── runner.py
├── expected_failures.json
└── run_extreme.py
```

Responsibilities:

- `generators.py`: deterministic PTX mutations and adversarial pressure kernels.
- `cases.py`: contract/frontier matrices and independent expected results.
- `backends.py`: compiler invocation plus local-simulator and CModel execution adapters.
- `runner.py`: case execution, fail-closed comparison, artifact capture, and summary accounting.
- `expected_failures.json`: narrowly scoped frontier failure registry.
- `run_extreme.py`: CLI selecting suite, backend, optimization level, seed, and artifact directory.

Existing mutation helpers may be reused only after their semantics are verified against current `T*/kernel.ptx` testcases. The new suite must not inherit the obsolete root-level `PTX-0*.ptx` paths or old FP16 test assumptions.

## Commands

Target command contract:

```bash
make test
make test-frontier
python3 tests/extreme/run_extreme.py --suite contract --backend local
python3 tests/extreme/run_extreme.py --suite frontier --backend local
python3 tests/extreme/run_extreme.py --suite contract --backend cmodel
python3 tests/extreme/run_extreme.py --suite frontier --backend cmodel
```

`make test` runs the current public-case checks and local contract suite. It must fail when zero public or extreme cases are discovered.

`make test-frontier` permits only registered `XFAIL` results and fails on unknown failures, malformed expected-failure entries, or zero discovered cases.

## Acceptance Criteria

1. Local contract execution discovers at least 100 deterministic case/optimization combinations and exits zero only when all pass.
2. Frontier execution contains real cross-block pressure cases for GPRs, register pairs, and predicates.
3. Known limitations appear as explicit `XFAIL`; no missing resource or exception becomes a skip.
4. Every unexpected failure emits a standalone reproducer and complete diagnostic artifacts.
5. A remote CModel run executes complete buffers for all selected contract boundaries and produces a machine-readable summary.
6. The current public testcase directory layout is used directly.
7. Running with an empty or missing testcase directory returns non-zero.
8. Test output clearly separates local simulator evidence from official CModel evidence.

## Follow-up

Failures found by this suite become separate compiler-fix tasks. Fixing real spill/reload and predicate allocation is intentionally deferred until the new frontier evidence demonstrates the precise failing boundaries.

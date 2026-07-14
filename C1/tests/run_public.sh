#!/usr/bin/env bash
# run_public.sh - Compile + disassemble every public PTX test and sanity-check.
#
# Runs from the C1 root (or anywhere; paths are resolved relative to this
# script). For each of the five public PTX inputs it:
#   1. compiles with aec-cc -O2 --report and validates the JSON report,
#   2. checks raw output is non-empty and byte-size divisible by 16,
#   3. disassembles it with aec-objdump and checks the output is non-empty,
#   4. also re-compiles at -O0 and -O3 to ensure every level is legal.
# It first runs the encoder golden self-test. Exits non-zero if anything fails.
#
# Environment:
#   AEC_PUBLIC_TESTS   Override the testcase root (for zero-discovery testing).
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CC="$ROOT/bin/aec-cc"
OD="$ROOT/bin/aec-objdump"
TESTS="${AEC_PUBLIC_TESTS:-$ROOT/../public/Track-C/C1-compiler/testcases}"
WORK="$(mktemp -d)" || { echo "FATAL: mktemp -d failed" >&2; exit 1; }
trap 'rm -rf "$WORK"' EXIT

pass=0
fail=0

note_pass() { echo "  PASS: $1"; pass=$((pass + 1)); }
note_fail() { echo "  FAIL: $1"; fail=$((fail + 1)); }

# ---- helpers ----

# validate_report <report.json> <binary.aecbin>
#   Parses the compiler JSON report and cross-checks against the binary.
#   Returns 0 if the report is valid, 1 otherwise.
validate_report() {
  local report="$1" binary="$2"
  [ -f "$report" ] || { echo "  (missing report: $report)"; return 1; }
  [ -f "$binary" ] || { echo "  (missing binary: $binary)"; return 1; }

  python3 -c "
import json, os, sys
try:
    with open(sys.argv[1]) as f:
        d = json.load(f)
except json.JSONDecodeError as e:
    sys.exit(f'report not valid JSON: {e}')
except Exception as e:
    sys.exit(f'read error: {e}')
if d.get('status') != 'ok':
    sys.exit(f'report status={d.get(\"status\")!r} != \"ok\"')
if d.get('opt_level') != 'O2':
    sys.exit(f'report opt_level={d.get(\"opt_level\")!r} != \"O2\"')
n = d.get('num_aec_instructions')
if not isinstance(n, int) or n < 0:
    sys.exit(f'report num_aec_instructions={n!r} is not a non-negative integer')
sz = os.path.getsize(sys.argv[2])
if n * 16 != sz:
    sys.exit(f'num_aec_instructions*16 = {n*16} != binary size {sz}')
" "$report" "$binary" 2>&1 || return 1
  return 0
}

# ---- encoder self-test ----

echo "== encoder golden self-test =="
if "$CC" --selftest; then note_pass "encoder selftest"; else note_fail "encoder selftest"; fi

# ---- public PTX tests ----

echo "== public PTX tests =="

# Discover sorted T*/kernel.ptx directories (parent must start with T).
cases=()
for f in "$TESTS"/T*/kernel.ptx; do
  [ -f "$f" ] && cases+=("$f")
done

if [ "${#cases[@]}" -eq 0 ]; then
  echo "FAIL: no public kernel.ptx files found at $TESTS" >&2
  exit 1
fi

for f in "${cases[@]}"; do
  name="$(basename "$(dirname "$f")")"
  out="$WORK/$name.aecbin"
  report="$WORK/$name.json"
  log="$WORK/$name.compile.log"

  # ---- -O2 compile + report validation ----
  if "$CC" "$f" -O2 -o "$out" --report "$report" >"$log" 2>&1; then
    if validate_report "$report" "$out"; then
      note_pass "$name compile -O2"
    else
      note_fail "$name compile -O2"
      cat "$log" 2>/dev/null
    fi
  else
    note_fail "$name compile -O2"
    cat "$log" 2>/dev/null
    continue
  fi

  # ---- raw binary sanity ----
  size="$(wc -c < "$out")"
  if [ "$size" -gt 0 ] && [ $((size % 16)) -eq 0 ]; then
    note_pass "$name raw-aecbin"
  else
    note_fail "$name raw-aecbin(size=$size)"
  fi

  # ---- objdump ----
  obj_log="$WORK/$name.objdump.log"
  if "$OD" "$out" > "$WORK/$name.asm" 2>"$obj_log" && [ -s "$WORK/$name.asm" ]; then
    note_pass "$name objdump"
  else
    note_fail "$name objdump"
    cat "$obj_log" 2>/dev/null
  fi

  # ---- -O0, -O3 compiles ----
  for lvl in O0 O3; do
    lvl_log="$WORK/$name.$lvl.compile.log"
    if "$CC" "$f" -$lvl -o "$WORK/$name.$lvl.aecbin" >"$lvl_log" 2>&1; then
      note_pass "$name -$lvl"
    else
      note_fail "$name -$lvl"
      cat "$lvl_log" 2>/dev/null
    fi
  done
done

echo "== summary: $pass passed, $fail failed =="
if [ "$fail" -gt 0 ]; then
  exit 1
fi

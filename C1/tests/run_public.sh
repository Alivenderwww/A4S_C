#!/usr/bin/env bash
# run_public.sh - Compile + disassemble every public PTX test and sanity-check.
#
# Runs from the C1 root (or anywhere; paths are resolved relative to this
# script). For each of the five public PTX inputs it:
#   1. compiles with aec-cc -O2,
#   2. checks the .aecbin exists and starts with the 'AEC1' magic,
#   3. disassembles it with aec-objdump and checks the output is non-empty,
#   4. also re-compiles at -O0 and -O3 to ensure every level is legal.
# It first runs the encoder golden self-test. Exits non-zero if anything fails.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CC="$ROOT/bin/aec-cc"
OD="$ROOT/bin/aec-objdump"
TESTS="$ROOT/../public/Track-C/C1-compiler/testcases"
WORK="$(mktemp -d 2>/dev/null || echo /tmp/aec_test)"
mkdir -p "$WORK"

pass=0
fail=0

note_pass() { echo "  PASS: $1"; pass=$((pass + 1)); }
note_fail() { echo "  FAIL: $1"; fail=$((fail + 1)); }

echo "== encoder golden self-test =="
if "$CC" --selftest; then note_pass "encoder selftest"; else note_fail "encoder selftest"; fi

echo "== public PTX tests =="
for f in "$TESTS"/PTX-0*.ptx; do
  [ -e "$f" ] || { echo "  (no testcases found at $TESTS)"; break; }
  name="$(basename "$f")"
  out="$WORK/${name%.ptx}.aecbin"

  if "$CC" "$f" -O2 -o "$out" >/dev/null 2>&1; then
    :
  else
    note_fail "$name compile -O2"; continue
  fi

  # Magic check: first 4 bytes must be 'A','E','C','1'.
  magic="$(head -c 4 "$out" 2>/dev/null)"
  if [ "$magic" = "AEC1" ]; then note_pass "$name magic"; else note_fail "$name magic"; fi

  # Disassemble.
  if "$OD" "$out" > "$WORK/${name%.ptx}.asm" 2>/dev/null && \
     [ -s "$WORK/${name%.ptx}.asm" ]; then
    note_pass "$name objdump"
  else
    note_fail "$name objdump"
  fi

  # -O0 and -O3 legality.
  for lvl in O0 O3; do
    if "$CC" "$f" -$lvl -o "$WORK/${name%.ptx}.$lvl.aecbin" >/dev/null 2>&1; then
      note_pass "$name -$lvl"
    else
      note_fail "$name -$lvl"
    fi
  done
done

echo "== summary: $pass passed, $fail failed =="
[ "$fail" -eq 0 ]

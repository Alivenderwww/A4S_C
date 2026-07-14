#!/usr/bin/env bash
# probe_failures.sh -- failure-mode probes for run_public.sh report validation
# and temp-dir cleanup logic.
set -u

command -v python3 >/dev/null 2>&1 || { echo "FATAL: probe_failures requires python3" >&2; exit 1; }

pass=0; fail=0
note_pass() { echo "  PASS: $1"; pass=$((pass + 1)); }
note_fail() { echo "  FAIL: $1"; fail=$((fail + 1)); }

# ---- copy of validate_report (must match run_public.sh) ----
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

# ---- helpers ----
make_binary() {
  python3 -c "import sys; sys.stdout.buffer.write(b'\\x00' * $1)" > "$2"
}

WORK="$(mktemp -d)" || { echo "FATAL: mktemp -d failed" >&2; exit 1; }

# ---- probe: cleanup trap removes temp dir ----
echo "=== probe: cleanup trap ==="
subdir="$(mktemp -d)"
if [ ! -d "$subdir" ]; then
  note_fail "setup: mktemp for subshell"
else
  (
    trap 'rm -rf "$subdir"' EXIT
    touch "$subdir/guard"
    [ -f "$subdir/guard" ] || exit 99
  )
  if [ -d "$subdir" ]; then
    note_fail "cleanup trap did not remove temp dir"
  else
    note_pass "cleanup trap removes temp dir"
  fi
fi

# ---- probe: validate_report missing report file ----
echo "=== probe: validate_report missing report ==="
make_binary 32 "$WORK/dummy.bin"
if validate_report "$WORK/nosuch.json" "$WORK/dummy.bin"; then
  note_fail "missing report should fail"
else
  note_pass "missing report rejected"
fi

# ---- probe: validate_report missing binary ----
echo "=== probe: validate_report missing binary ==="
cat > "$WORK/good.json" <<-EOF
{"status":"ok","opt_level":"O2","num_aec_instructions":2}
EOF
if validate_report "$WORK/good.json" "$WORK/nosuch.bin"; then
  note_fail "missing binary should fail"
else
  note_pass "missing binary rejected"
fi

# ---- probe: validate_report malformed JSON ----
echo "=== probe: validate_report malformed JSON ==="
echo "not json" > "$WORK/bad.json"
if validate_report "$WORK/bad.json" "$WORK/dummy.bin"; then
  note_fail "malformed JSON should fail"
else
  note_pass "malformed JSON rejected"
fi

# ---- probe: validate_report status != ok ----
echo "=== probe: validate_report wrong status ==="
cat > "$WORK/err.json" <<-EOF
{"status":"error","opt_level":"O2","num_aec_instructions":2}
EOF
if validate_report "$WORK/err.json" "$WORK/dummy.bin"; then
  note_fail "wrong status should fail"
else
  note_pass "wrong status rejected"
fi

# ---- probe: validate_report wrong opt_level ----
echo "=== probe: validate_report wrong opt_level ==="
cat > "$WORK/notO2.json" <<-EOF
{"status":"ok","opt_level":"O1","num_aec_instructions":2}
EOF
if validate_report "$WORK/notO2.json" "$WORK/dummy.bin"; then
  note_fail "wrong opt_level should fail"
else
  note_pass "wrong opt_level rejected"
fi

# ---- probe: validate_report missing num_aec_instructions ----
echo "=== probe: validate_report missing num_aec_instructions ==="
cat > "$WORK/nocnt.json" <<-EOF
{"status":"ok","opt_level":"O2"}
EOF
if validate_report "$WORK/nocnt.json" "$WORK/dummy.bin"; then
  note_fail "missing num_aec_instructions should fail"
else
  note_pass "missing num_aec_instructions rejected"
fi

# ---- probe: validate_report non-integer num_aec_instructions ----
echo "=== probe: validate_report non-integer num_aec_instructions ==="
cat > "$WORK/strcnt.json" <<-EOF
{"status":"ok","opt_level":"O2","num_aec_instructions":"lots"}
EOF
if validate_report "$WORK/strcnt.json" "$WORK/dummy.bin"; then
  note_fail "non-integer num_aec_instructions should fail"
else
  note_pass "non-integer num_aec_instructions rejected"
fi

# ---- probe: validate_report size mismatch ----
echo "=== probe: validate_report size mismatch ==="
# num_aec_instructions=3 -> expected size = 48
cat > "$WORK/sizemismatch.json" <<-EOF
{"status":"ok","opt_level":"O2","num_aec_instructions":3}
EOF
make_binary 64 "$WORK/mismatch.bin"   # 64 != 48
if validate_report "$WORK/sizemismatch.json" "$WORK/mismatch.bin"; then
  note_fail "size mismatch should fail"
else
  note_pass "size mismatch rejected"
fi

# ---- probe: validate_report valid report passes ----
echo "=== probe: validate_report valid report ==="
cat > "$WORK/valid.json" <<-EOF
{"status":"ok","opt_level":"O2","num_aec_instructions":3}
EOF
make_binary 48 "$WORK/valid.bin"
if validate_report "$WORK/valid.json" "$WORK/valid.bin"; then
  note_pass "valid report accepted"
else
  note_fail "valid report should pass"
fi

echo "== summary: $pass passed, $fail failed =="
[ "$fail" -eq 0 ] || exit 1

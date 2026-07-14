#!/usr/bin/env python3
"""Tests for C2 packaging: libaec_device.so removed from submission zip.

Verifies:
  - pack_trackc.stage_c2 does NOT copy lib/libaec_device.so into the staging
    directory (only libaec.so and agents/ are copied).
  - verify_trackc.layer_a hard-FAILs when C2/lib/libaec_device.so is found
    inside the submission zip.
"""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
for p in (str(SCRIPTS), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import trackc_common as tc          # noqa: E402
import verify_trackc as vt          # noqa: E402
import pack_trackc as pt            # noqa: E402

# Reusable helpers from the earlier permission test module
import test_trackc_permissions as tp  # noqa: E402
_ROOT = tp._ROOT
_add_file = tp._add_file
_add_dir = tp._add_dir
_write_zip = tp._write_zip


# ---------------------------------------------------------------------------
# Tests: stage_c2
# ---------------------------------------------------------------------------

class TestStageC2NoDeviceLib(unittest.TestCase):
    """stage_c2 must skip lib/libaec_device.so entirely."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.c2_src = self.tmpdir / "C2"
        self.c2_src.mkdir()
        # libaec.so (ELF magic)
        (self.c2_src / "libaec.so").write_bytes(b"\x7fELF" + b"\0" * 100)
        # lib/libaec_device.so
        (self.c2_src / "lib").mkdir()
        (self.c2_src / "lib" / "libaec_device.so").write_bytes(
            b"\x7fELF" + b"\0" * 100
        )
        # agents/
        (self.c2_src / "agents").mkdir()
        (self.c2_src / "agents" / "dma_agent.py").write_text("# empty\n")
        (self.c2_src / "agents" / "kernel_agent.py").write_text("# empty\n")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_device_lib_not_copied(self):
        """stage_c2 output must contain libaec.so + agents but NOT lib/*."""
        dst = self.tmpdir / "out"
        dst.mkdir()
        pt.stage_c2(self.c2_src, dst)

        self.assertTrue(
            (dst / "libaec.so").exists(),
            "libaec.so must be present",
        )
        self.assertFalse(
            (dst / "lib" / "libaec_device.so").exists(),
            "lib/libaec_device.so must NOT be copied",
        )
        self.assertTrue(
            (dst / "agents" / "dma_agent.py").exists(),
            "agents/dma_agent.py must be present",
        )

    def test_no_lib_dir_created_at_all(self):
        """stage_c2 must not create a lib/ directory in the output."""
        dst = self.tmpdir / "out2"
        dst.mkdir()
        pt.stage_c2(self.c2_src, dst)
        self.assertFalse(
            (dst / "lib").exists(),
            "lib/ directory must not exist in output",
        )


# ---------------------------------------------------------------------------
# Helpers: build a minimal zip that includes the forbidden device lib
# ---------------------------------------------------------------------------

def _build_zip_with_forbidden_device_lib(create_system_dev: int = 3) -> io.BytesIO:
    """Return a minimal TrackC zip that also carries C2/lib/libaec_device.so.

    All other entries match what build_minimal_zip produces so that the
    zip passes every non-C2 check in layer_a.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # --- directory tree (same as build_minimal_zip) ---
        for d in [
            _ROOT,
            f"{_ROOT}/C1",
            f"{_ROOT}/C1/compiler",
            f"{_ROOT}/C1/compiler/src",
            f"{_ROOT}/C1/compiler/src/src",
            f"{_ROOT}/C2",
            f"{_ROOT}/C2/agents",
            f"{_ROOT}/C2/lib",          # ← the device-lib parent dir
            f"{_ROOT}/C3",
            f"{_ROOT}/C3/src",
            f"{_ROOT}/C3/src/scheduler",
            f"{_ROOT}/C3/src/runtime",
            f"{_ROOT}/C3/src/tools",
        ]:
            _add_dir(zf, d)

        # --- C1 ---
        aec_body = b'#!/bin/sh\nexec "$(dirname -- "$0")/src"/aec-cc "$@"\n'
        _add_file(zf, f"{_ROOT}/C1/compiler/aec-cc", aec_body, mode=0o100755)
        _add_file(zf, f"{_ROOT}/C1/compiler/src/Makefile",
                  b"all:\n\tgcc -o aec-cc main.c\n")

        # --- C2 ---
        _add_file(zf, f"{_ROOT}/C2/libaec.so",
                  b"\x7fELF" + b"\0" * 100, mode=0o100755)
        # ★ forbidden entry
        _add_file(zf, f"{_ROOT}/C2/lib/libaec_device.so",
                  b"\x7fELF" + b"\0" * 100, mode=0o100755,
                  zcs=create_system_dev)

        # --- C3 ---
        _add_file(zf, f"{_ROOT}/C3/readme.md",
                  b"# C3\n\n--onnx model.onnx --output dag.json\ninfer_worker.py\n")

    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# Tests: layer_a  —  reject C2/lib/libaec_device.so
# ---------------------------------------------------------------------------

class TestVerifyLayerADeviceLibForbidden(unittest.TestCase):
    """layer_a must hard-FAIL when the submission includes the device lib."""

    def test_device_lib_present_causes_hard_fail(self):
        """A zip with C2/lib/libaec_device.so must be rejected."""
        zip_data = _build_zip_with_forbidden_device_lib()
        tmppath = _write_zip(zip_data, f"{_ROOT}.zip")
        try:
            rep = vt.Reporter()
            vt.layer_a(tmppath, rep)

            # Find the device-lib check
            dev_items = [c for c in rep.items
                         if "libaec_device" in c.name or "设备库" in c.name]
            self.assertGreater(
                len(dev_items), 0,
                "Expected a check for C2/lib/libaec_device.so in layer_a output",
            )
            for ci in dev_items:
                self.assertEqual(
                    ci.severity, "FAIL",
                    f"Expected FAIL for forbidden device lib, got "
                    f"{ci.severity}: {ci.name}",
                )
                self.assertFalse(ci.ok)

            # Overall hard fail
            self.assertTrue(rep.hard_fail,
                            "Zip with device lib must produce hard FAIL")
        finally:
            tmppath.unlink(missing_ok=True)
            tmppath.parent.rmdir()

    def test_no_device_lib_passes(self):
        """A clean zip without device lib must pass C2 checks."""
        zip_data = tp.build_minimal_zip(create_system=3)
        tmppath = _write_zip(zip_data, f"{_ROOT}.zip")
        try:
            rep = vt.Reporter()
            vt.layer_a(tmppath, rep)

            dev_items = [c for c in rep.items
                         if "libaec_device" in c.name or "设备库" in c.name]
            if dev_items:
                for ci in dev_items:
                    self.assertEqual(
                        ci.severity, "PASS",
                        f"Expected PASS when device lib absent, got "
                        f"{ci.severity}: {ci.name}",
                    )

            # No FAIL for the device lib check
            dev_fails = [
                c for c in rep.items
                if c.severity == "FAIL"
                and ("libaec_device" in c.name or "设备库" in c.name)
            ]
            self.assertEqual(
                len(dev_fails), 0,
                "There should be no FAIL for device lib when absent",
            )
        finally:
            tmppath.unlink(missing_ok=True)
            tmppath.parent.rmdir()


if __name__ == "__main__":
    unittest.main()

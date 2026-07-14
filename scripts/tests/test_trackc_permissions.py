#!/usr/bin/env python3
"""Tests for C1 ZIP executable permission handling.

Verifies:
  - add_file_entry / add_dir_entry set create_system=3 (Unix origin) so unzip
    on Linux correctly restores +x bits.
  - verify_trackc.layer_a hard-FAILs when aec-cc has create_system != 3.
  - verify_trackc.layer_a passes when aec-cc has create_system == 3 and +x bits.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import trackc_common as tc  # noqa: E402
import verify_trackc as vt  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers for constructing minimal valid zips
# ---------------------------------------------------------------------------

_ROOT = "TrackC-00000000a-00000001b-00000002c"


def _add_file(zf: zipfile.ZipFile, arcname: str, content: bytes = b"x",
              mode: int = 0o100644, zcs: int = 3) -> None:
    """Add a file entry with explicit create_system and mode bits."""
    info = zipfile.ZipInfo(arcname)
    info.create_system = zcs
    info.external_attr = mode << 16
    info.file_size = len(content)
    zf.writestr(info, content)


def _add_dir(zf: zipfile.ZipFile, arcname: str, mode: int = 0o40755,
             zcs: int = 3) -> None:
    """Add a directory entry with trailing slash and DOS dir flag."""
    name = arcname if arcname.endswith("/") else arcname + "/"
    info = zipfile.ZipInfo(name)
    info.create_system = zcs
    info.external_attr = (mode << 16) | 0x10  # MS-DOS directory attribute
    info.file_size = 0
    zf.writestr(info, "")


def build_minimal_zip(create_system: int, aec_mode: int = 0o100755) -> io.BytesIO:
    """Build a minimal TrackC zip that passes all non-permission layer_a checks.

    Args:
        create_system:  value for C1/compiler/aec-cc ZipInfo.create_system.
        aec_mode:       Unix mode (with file-type bits) for the aec-cc entry.

    Returns:  BytesIO of the full valid zip.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # --- directory tree ---
        for d in [
            _ROOT,
            f"{_ROOT}/C1",
            f"{_ROOT}/C1/compiler",
            f"{_ROOT}/C1/compiler/src",
            f"{_ROOT}/C1/compiler/src/src",
            f"{_ROOT}/C2",
            f"{_ROOT}/C2/agents",
            f"{_ROOT}/C2/lib",
            f"{_ROOT}/C3",
            f"{_ROOT}/C3/src",
            f"{_ROOT}/C3/src/scheduler",
            f"{_ROOT}/C3/src/runtime",
            f"{_ROOT}/C3/src/tools",
        ]:
            _add_dir(zf, d)

        # --- C1/compiler/aec-cc (the entry under test) ---
        aec_body = b'#!/bin/sh\nexec "$(dirname -- "$0")/src"/aec-cc "$@"\n'
        _add_file(zf, f"{_ROOT}/C1/compiler/aec-cc", aec_body,
                  mode=aec_mode, zcs=create_system)

        # --- C1/compiler/src/Makefile ---
        _add_file(zf, f"{_ROOT}/C1/compiler/src/Makefile",
                  b"all:\n\tgcc -o aec-cc main.c\n")

        # --- C2/libaec.so (needs ELF magic for header check) ---
        _add_file(zf, f"{_ROOT}/C2/libaec.so",
                  b"\x7fELF" + b"\0" * 100, mode=0o100755)

        # --- C3/readme.md (needs --onnx / --output / infer_worker.py) ---
        _add_file(zf, f"{_ROOT}/C3/readme.md",
                  b"# C3\n\n--onnx model.onnx --output dag.json\ninfer_worker.py\n")

    buf.seek(0)
    return buf


def _write_zip(zip_data: io.BytesIO, name: str) -> Path:
    """Write zip BytesIO to a temporary file with the given name."""
    tmpdir = Path(tempfile.mkdtemp())
    p = tmpdir / name
    p.write_bytes(zip_data.read())
    return p


# ---------------------------------------------------------------------------
# Tests: trackc_common.add_file_entry / _zinfo_for
# ---------------------------------------------------------------------------

class TestAddFileEntryCreateSystem(unittest.TestCase):
    """add_file_entry must produce ZipInfo with create_system=3 (Unix)."""

    def _add_and_getinfo(self, arcname: str,
                         content: bytes = b"test\n") -> zipfile.ZipInfo:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            tmp = Path(tempfile.mkdtemp()) / "tmpfile"
            try:
                tmp.write_bytes(content)
                tc.add_file_entry(zf, tmp, arcname)
            finally:
                tmp.unlink(missing_ok=True)
        buf.seek(0)
        with zipfile.ZipFile(buf) as zf:
            return zf.getinfo(arcname)

    def _dir_getinfo(self, arcname: str) -> zipfile.ZipInfo:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            tc.add_dir_entry(zf, arcname)
        buf.seek(0)
        with zipfile.ZipFile(buf) as zf:
            # add_dir_entry stores with trailing /
            key = arcname if arcname.endswith("/") else arcname + "/"
            return zf.getinfo(key)

    # ---- create_system==3 checks ----

    def test_aec_cc_create_system_is_3(self):
        """aec-cc must get create_system=3 (Unix)."""
        zi = self._add_and_getinfo("TrackC-x/C1/compiler/aec-cc")
        self.assertEqual(zi.create_system, 3,
                         f"Expected 3, got {zi.create_system}")

    def test_aec_cc_has_executable_bits(self):
        """aec-cc must have executable permission bits."""
        zi = self._add_and_getinfo("TrackC-x/C1/compiler/aec-cc")
        mode = tc.zip_mode(zi)
        self.assertTrue(mode & 0o111,
                        f"Expected +x, got mode={oct(mode)}")

    def test_so_create_system_is_3(self):
        zi = self._add_and_getinfo("TrackC-x/C2/libaec.so")
        self.assertEqual(zi.create_system, 3)

    def test_py_create_system_is_3(self):
        zi = self._add_and_getinfo("TrackC-x/C3/tools/infer_worker.py")
        self.assertEqual(zi.create_system, 3)

    def test_dir_create_system_is_3(self):
        """Directory entries must also carry create_system=3."""
        zi = self._dir_getinfo("TrackC-x/C1/")
        self.assertEqual(zi.create_system, 3)


# ---------------------------------------------------------------------------
# Tests: verify_trackc.layer_a  —  create_system enforcement
# ---------------------------------------------------------------------------

class TestVerifyLayerACreateSystem(unittest.TestCase):
    """layer_a must hard-FAIL when aec-cc has create_system != 3."""

    def test_windows_origin_fails(self):
        """aec-cc with create_system=0 (FAT) must produce hard FAIL."""
        zip_data = build_minimal_zip(create_system=0, aec_mode=0o100755)
        tmppath = _write_zip(zip_data, f"{_ROOT}.zip")
        try:
            rep = vt.Reporter()
            vt.layer_a(tmppath, rep)

            # Locate the create_system check by name
            origin_items = [c for c in rep.items
                            if "create_system" in c.name or "归因" in c.name]
            self.assertGreater(len(origin_items), 0,
                               "Expected a create_system check in layer_a output")
            for ci in origin_items:
                self.assertEqual(ci.severity, "FAIL",
                                 f"Expected FAIL for windows-origin aec-cc, got "
                                 f"{ci.severity}: {ci.name}")
                self.assertFalse(ci.ok)

            # The zip must be overall rejected
            self.assertTrue(rep.hard_fail,
                            "Windows-origin zip should produce hard FAIL")
        finally:
            tmppath.unlink(missing_ok=True)
            tmppath.parent.rmdir()

    def test_unix_origin_permission_checks_pass(self):
        """aec-cc with create_system=3 (Unix) and correct +x must pass."""
        zip_data = build_minimal_zip(create_system=3, aec_mode=0o100755)
        tmppath = _write_zip(zip_data, f"{_ROOT}.zip")
        try:
            rep = vt.Reporter()
            vt.layer_a(tmppath, rep)

            # Origin check should PASS
            origin_items = [c for c in rep.items
                            if "create_system" in c.name or "归因" in c.name]
            if origin_items:
                for ci in origin_items:
                    self.assertEqual(
                        ci.severity, "PASS",
                        f"Expected PASS for Unix-origin aec-cc, got "
                        f"{ci.severity}: {ci.name}")

            # +x check for aec-cc should also PASS
            exec_items = [c for c in rep.items
                          if c.name == "aec-cc 可执行位 (+x)"]
            self.assertGreater(len(exec_items), 0,
                               "Expected aec-cc +x check in output")
            for ci in exec_items:
                self.assertEqual(
                    ci.severity, "PASS",
                    f"Expected PASS for +x check, got {ci.severity}: {ci.name}")

            # No hard failure from permission-related checks
            permission_fails = [
                c for c in rep.items
                if c.severity == "FAIL"
                and ("可执行" in c.name or "create_system" in c.name or "归因" in c.name)
            ]
            self.assertEqual(len(permission_fails), 0,
                             f"Permission checks unexpectedly failed: "
                             + ", ".join(f"{c.name}" for c in permission_fails))
        finally:
            tmppath.unlink(missing_ok=True)
            tmppath.parent.rmdir()

    def test_windows_origin_without_plus_x_also_fails(self):
        """Even with create_system=0 and no +x, origin check fails."""
        zip_data = build_minimal_zip(create_system=0, aec_mode=0o100644)
        tmppath = _write_zip(zip_data, f"{_ROOT}.zip")
        try:
            rep = vt.Reporter()
            vt.layer_a(tmppath, rep)

            origin_items = [c for c in rep.items
                            if "create_system" in c.name or "归因" in c.name]
            for ci in origin_items:
                self.assertEqual(ci.severity, "FAIL",
                                 f"Expected FAIL for create_system=0")

            # +x check for aec-cc should also be FAIL (mode lacks +x)
            exec_items = [c for c in rep.items
                          if c.name == "aec-cc 可执行位 (+x)"]
            self.assertGreater(len(exec_items), 0,
                               "Expected aec-cc +x check in output")
            for ci in exec_items:
                self.assertEqual(ci.severity, "FAIL",
                                 f"Expected FAIL for missing +x")

            self.assertTrue(rep.hard_fail)
        finally:
            tmppath.unlink(missing_ok=True)
            tmppath.parent.rmdir()


if __name__ == "__main__":
    unittest.main()

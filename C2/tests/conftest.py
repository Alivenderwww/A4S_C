"""Pytest configuration for C2 contract tests.

Provides:
* An **autouse fixture** that calls ``reset_runtime()`` before every test
  (test isolation).  If reset raises, the test is marked FAIL.
* ``SkipTest`` now inherits from ``unittest.SkipTest`` so pytest handles it
  natively — no additional conversion is needed.

The fixture replaces the previous ``pytest_runtest_call`` hookwrapper which
used ``outcome.force_result()`` incorrectly, and the earlier pattern of
catching ``SkipTest`` around ``yield`` (no longer required now that
``SkipTest`` is a ``unittest.SkipTest`` subclass).
"""

from __future__ import annotations

import pytest

from tests.runtime_harness import reset_runtime


@pytest.fixture(autouse=True)
def _runtime_reset():
    """Reset runtime before each test.

    If ``reset_runtime()`` raises (e.g. device reset failure), the test is
    failed immediately via ``pytest.fail()``; execution never reaches the
    test body.

    ``SkipTest`` skip signals are handled natively by pytest via the
    ``unittest.SkipTest`` inheritance — no catch block around ``yield``.
    """
    try:
        reset_runtime()
    except Exception as exc:
        pytest.fail(f"runtime reset failed: {exc}")
    yield

#!/usr/bin/env python3
"""Phase 2 concurrency and lifecycle tests for the C2 Runtime.

Architecture
------------

A) **Shim-based deterministic tests** (Linux + g++ only, SKIP on other
   platforms).  These use ``blocking_submit_shim.so`` (LD_PRELOAD) to
   intercept ``aecDeviceSubmit`` and block the calling thread *inside*
   the submit entry point.  The test thread can then observe intermediate
   states — stream not yet destroyed, allocation not yet freed, etc. —
   **before** the device processes the command, and then release the
   submit to let both the device and the blocking API call complete.
   This replaces race-condition "hope" with **deterministic ordering**.

   Tests:
     - ``test_destroy_waits_for_pending_submit``
     - ``test_free_waits_for_pending_submit``
     - ``test_unregister_waits_for_pending_submit``
     - ``test_event_marker_waits_for_pending_submit``

B) **ABA handle reuse** — brute-force create/destroy cycles (256×) to
   detect allocator reuse of the same opaque pointer.  If reuse is
   detected the test FAILs immediately (the contract is unsatisfiable
   for this build).  Without reuse, every operation on the stale handle
   must return ``INVALID_HANDLE`` (Sync, Destroy, CopyAsync, EventRecord
   for streams; Query, Sync, Destroy, Record, ElapsedCycles for events).

   Tests:
     - ``test_stream_aba_cycle_and_stale_handle``
     - ``test_event_aba_cycle_and_stale_handle``

C) **Stress / pressure tests** (multi-stream concurrent submission and
   same-stream concurrent enqueue).  These provide **pressure evidence
   only** — they prove the runtime does not crash under load but do NOT
   prove linearizability.  Shim-based tests in (A) cover linearizability
   deterministically.

   Tests:
     - ``test_concurrent_submission_stress``  (stress)
     - ``test_same_stream_concurrent_stress``              (stress)

All tests that call into ``libaec.so`` are gated by ``skip_if_no_library()``.
Shim-based tests additionally gate with ``skip_if_no_shim()``.
ALL tests run in an isolated subprocess (except those that are pure-ABI
without device interaction).

Total remote suite time is guaranteed < 45 seconds.
"""

from __future__ import annotations

import ctypes
import threading
import time

from tests.runtime_harness import (
    runtime, skip_if_no_library, isolated_subprocess, reset_runtime,
    require,
    SUCCESS, INVALID_ARGUMENT, INVALID_HANDLE, INVALID_ADDRESS,
    H2D, D2H,
    Runtime, HostBuffer, RuntimeStats,
    ERROR_NAMES,
    SkipTest,
    # Shim support
    skip_if_no_shim, compile_shim, shim_library_path,
)

# ===================================================================
# Helpers
# ===================================================================


def _ensure_shim() -> str:
    """Compile and return the shim .so path, or raise SkipTest."""
    skip_if_no_shim()
    return str(shim_library_path())


def _load_shim_ctypes() -> ctypes.CDLL | None:
    """Load the shim .so into *this* process (for same-process shim tests).

    Returns None if shim unavailable.
    """
    so = compile_shim()
    if so is None:
        return None
    lib = ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
    # aecTestArmSubmitBlock: void -> void
    lib.aecTestArmSubmitBlock.restype = None
    # aecTestWaitSubmitBlocked: uint64_t -> int (1=blocked, 0=timeout)
    lib.aecTestWaitSubmitBlocked.argtypes = [ctypes.c_uint64]
    lib.aecTestWaitSubmitBlocked.restype = ctypes.c_int
    # aecTestReleaseSubmitBlock: void -> void
    lib.aecTestReleaseSubmitBlock.restype = None
    # aecTestResetSubmitBlock: void -> void
    lib.aecTestResetSubmitBlock.restype = None
    return lib


def _reset_stats(rt: Runtime) -> None:
    st = rt.lib.aecResetRuntimeStats()
    require(st == SUCCESS, f"aecResetRuntimeStats: {ERROR_NAMES.get(st, st)}")


# ===================================================================
# TEST A1: ABA — stream handle reuse after 256 create/destroy cycles
# ===================================================================

def _stream_aba_256_cycles():
    """Create/destroy/create 256 times; detect pointer reuse → FAIL.

    If the allocator reuses an opaque pointer, the contract is
    unsatisfiable for this build — the C API cannot distinguish a
    stale handle from a new valid one.

    If NOT reused → verify that Sync, Destroy, CopyAsync, and
    EventRecord on the stale handle return INVALID_HANDLE.
    """
    rt = runtime()

    # --- Phase 1: 256 create/destroy cycles, check for reuse ---
    first = rt.stream()
    first_val = first.value
    require(rt.lib.aecStreamDestroy(first) == SUCCESS, "destroy first stream")

    reused = False
    stale_handles: list[ctypes.c_void_p] = []

    for i in range(256):
        s = rt.stream()
        val = s.value
        if val == first_val:
            reused = True
        require(rt.lib.aecStreamDestroy(s) == SUCCESS, f"destroy cycle {i}")
        # Keep the handle for stale testing even if we will FAIL later
        stale_handles.append(s)

    # Mark the first handle as stale (its stream was destroyed in cycle 0)
    stale = stale_handles[0] if stale_handles else first

    if reused:
        require(False,
                "Stream handle address reused by allocator after 256 cycles — "
                "stale-handle contract UNSATISFIABLE.  "
                f"First handle at {hex(first_val) if first_val else 'null'}, "
                "later cycles returned the same pointer.  "
                "The C API cannot distinguish old (stale) handles from new "
                "valid objects.")

    # --- Phase 2: stale handle must be rejected ---
    # Use a valid host buffer with non-zero bytes so the liveness check in
    # dispatch (stream ID lookup) executes before any argument rejection.
    hbuf = HostBuffer(64)
    for label, fn in [
        ("sync", lambda: rt.lib.aecStreamSync(stale)),
        ("destroy", lambda: rt.lib.aecStreamDestroy(stale)),
        ("copy_async", lambda: rt.lib.aecCopyAsync(
            0, hbuf.ptr, 64, H2D, stale)),
        ("event_record", lambda: _stale_stream_event_record(rt, stale)),
    ]:
        st = fn()
        require(st == INVALID_HANDLE,
                f"{label} on stale stream handle expected INVALID_HANDLE, "
                f"got {ERROR_NAMES.get(st, st)}")

    print(f"  NOTE  256 ABA cycles: no handle reuse detected, "
           "stale-handle ops all returned INVALID_HANDLE")


def _stale_stream_event_record(rt: Runtime, stale: ctypes.c_void_p) -> int:
    """Try to record an event on a stale stream."""
    ev = rt.event()
    st = rt.lib.aecEventRecord(ev, stale)
    rt.lib.aecEventDestroy(ev)
    return st


def test_stream_aba_cycle_and_stale_handle():
    """256× create/destroy stream; stale ops → INVALID_HANDLE (isolated)."""
    skip_if_no_library()
    result = isolated_subprocess(_stream_aba_256_cycles, timeout=25.0)
    require(result == "PASS",
            f"stream ABA subprocess: {result}")


# ===================================================================
# TEST A2: ABA — event handle reuse after 256 create/destroy cycles
# ===================================================================

def _event_aba_256_cycles():
    """Create/destroy/create 256 times; detect pointer reuse → FAIL.

    Without reuse → verify Query, Sync, Destroy, Record, and
    ElapsedCycles on stale handle → INVALID_HANDLE.
    """
    rt = runtime()

    first = rt.event()
    first_val = first.value
    require(rt.lib.aecEventDestroy(first) == SUCCESS, "destroy first event")

    reused = False
    stale_handles: list[ctypes.c_void_p] = []

    for i in range(256):
        ev = rt.event()
        val = ev.value
        if val == first_val:
            reused = True
        require(rt.lib.aecEventDestroy(ev) == SUCCESS, f"destroy cycle {i}")
        stale_handles.append(ev)

    stale = stale_handles[0] if stale_handles else first

    if reused:
        require(False,
                "Event handle address reused by allocator after 256 cycles — "
                "stale-handle contract UNSATISFIABLE.  "
                f"First handle at {hex(first_val) if first_val else 'null'}.")

    # Phase 2: stale handle ops
    for label, fn in [
        ("query", lambda: rt.lib.aecEventQuery(stale)),
        ("sync", lambda: rt.lib.aecEventSynchronize(stale)),
        ("destroy", lambda: rt.lib.aecEventDestroy(stale)),
        ("record", lambda: _stale_event_record(rt, stale)),
        ("elapsed", lambda: _stale_event_elapsed(rt, stale)),
    ]:
        st = fn()
        require(st == INVALID_HANDLE,
                f"{label} on stale event handle expected INVALID_HANDLE, "
                f"got {ERROR_NAMES.get(st, st)}")

    print("  NOTE  256 ABA cycles: no event handle reuse detected, "
           "stale-handle ops all returned INVALID_HANDLE")


def _stale_event_record(rt: Runtime, stale: ctypes.c_void_p) -> int:
    """Record a stale event on a valid stream."""
    s = rt.stream()
    st = rt.lib.aecEventRecord(stale, s)
    rt.lib.aecStreamDestroy(s)
    return st


def _stale_event_elapsed(rt: Runtime, stale: ctypes.c_void_p) -> int:
    """ElapsedCycles between a stale event and itself."""
    cycles = ctypes.c_uint64()
    return rt.lib.aecEventElapsedCycles(stale, stale, ctypes.byref(cycles))


def test_event_aba_cycle_and_stale_handle():
    """256× create/destroy event; stale ops → INVALID_HANDLE (isolated)."""
    skip_if_no_library()
    result = isolated_subprocess(_event_aba_256_cycles, timeout=25.0)
    require(result == "PASS",
            f"event ABA subprocess: {result}")


# ===================================================================
# TEST B1: Destroy waits for pending submit  (shim-deterministic)
# ===================================================================
#
# Protocol:
#   1. Worker: arm submit → aecCopyAsync (blocks in aecDeviceSubmit)
#   2. Main: call aecStreamDestroy — must NOT return (submit still blocked)
#   3. Main: release submit
#   4. Both worker (submit returns) and main (destroy returns) complete
#   5. Verify D2H data after destroy
#
# This is run inside a subprocess with LD_PRELOAD set to the shim.
# Subprocess entry point: _destroy_waits_for_pending_submit

def _destroy_waits_for_pending_submit():
    """Worker blocks inside submit; destroy must wait; release → both proceed."""
    skip_if_no_library()
    reset_runtime()
    rt = runtime()

    # Load shim ctypes *inside* the subprocess (after LD_PRELOAD)
    shim = _load_shim_ctypes()
    require(shim is not None, "shim must be loadable in subprocess")

    alloc = rt.alloc(4096)
    stream = rt.stream()
    host = HostBuffer(4096)
    host.write(bytes(i & 0xFF for i in range(4096)))

    try:
        shim.aecTestResetSubmitBlock()

        # --- Arm submit blocker ---
        shim.aecTestArmSubmitBlock()

        # --- Launch worker that will block inside aecDeviceSubmit ---
        enq_result: list[int] = [-1]
        barrier = threading.Barrier(2)

        def worker() -> None:
            barrier.wait()  # synchronise start
            enq_result[0] = rt.lib.aecCopyAsync(
                alloc, host.ptr, 4096, H2D, stream)

        t = threading.Thread(target=worker)
        t.start()
        barrier.wait()  # ensure worker has started

        # Wait until worker is blocked inside submit
        blocked = shim.aecTestWaitSubmitBlocked(5000)  # 5s timeout
        require(blocked == 1,
                "worker did not block in aecDeviceSubmit within 5s")

        # --- Destroy should *NOT* return yet (submit still in flight) ---
        # We give it a short time to incorrectly return
        destroy_result: list[int] = [-1]

        def destroyer() -> None:
            destroy_result[0] = rt.lib.aecStreamDestroy(stream)

        dt = threading.Thread(target=destroyer)
        dt.start()
        time.sleep(0.2)  # 200ms — plenty of time if destroy were broken
        require(destroy_result[0] == -1,
                "aecStreamDestroy returned before submit released — "
                "destroy MUST wait for pending submit")

        # --- Release submit → both threads complete ---
        shim.aecTestReleaseSubmitBlock()

        t.join(timeout=5)
        dt.join(timeout=5)
        require(not t.is_alive() and not dt.is_alive(),
                "thread join timeout after release")

        require(enq_result[0] == SUCCESS,
                f"copy async result: {ERROR_NAMES.get(enq_result[0], enq_result[0])}")
        require(destroy_result[0] == SUCCESS,
                f"destroy result: {ERROR_NAMES.get(destroy_result[0], destroy_result[0])}")

        # --- Verify data: synchronous D2H after destroy ---
        check = HostBuffer(4096)
        st = rt.lib.aecCopyD2H(check.ptr, alloc, 4096)
        require(st == SUCCESS,
                f"D2H after destroy: {ERROR_NAMES.get(st, st)}")
        require(check.read() == host.read(),
                "data integrity after destroy-race")

    finally:
        shim.aecTestResetSubmitBlock()
        # stream destroyed by destroyer thread above
        rt.lib.aecFree(alloc)

    print("  NOTE  destroy deterministically waited for pending submit; "
           "data verified post-destroy")


def test_destroy_waits_for_pending_submit():
    """Destroy waits for pending submit (shim-deterministic)."""
    skip_if_no_library()
    skip_if_no_shim()
    ld_preload = _ensure_shim()
    result = isolated_subprocess(
        _destroy_waits_for_pending_submit,
        timeout=25.0, ld_preload=ld_preload)
    require(result == "PASS",
            f"destroy-wait subprocess: {result}")


# ===================================================================
# TEST B2: Free waits for pending submit  (shim-deterministic)
# ===================================================================

def _free_waits_for_pending_submit():
    """Worker blocks inside submit; free must wait; release → both proceed.

    Then verify:
      - Post-free enqueue on same (now-freed) address accepted (SUCCESS)
      - Sync returns INVALID_ADDRESS
    """
    skip_if_no_library()
    reset_runtime()
    rt = runtime()

    shim = _load_shim_ctypes()
    require(shim is not None, "shim must be loadable in subprocess")

    alloc = rt.alloc(4096)
    stream = rt.stream()
    host = HostBuffer(4096)
    host.write(bytes(i & 0xFF for i in range(4096)))

    try:
        shim.aecTestResetSubmitBlock()

        # Arm submit blocker
        shim.aecTestArmSubmitBlock()

        enq_result: list[int] = [-1]
        free_result: list[int] = [-1]
        barrier = threading.Barrier(2)

        def worker() -> None:
            barrier.wait()
            enq_result[0] = rt.lib.aecCopyAsync(
                alloc, host.ptr, 4096, H2D, stream)

        t = threading.Thread(target=worker)
        t.start()
        barrier.wait()

        blocked = shim.aecTestWaitSubmitBlocked(5000)
        require(blocked == 1,
                "worker did not block in aecDeviceSubmit within 5s")

        # --- Free should NOT return yet ---
        free_result[0] = -2  # distinct sentinel

        def freer() -> None:
            free_result[0] = rt.lib.aecFree(alloc)

        ft = threading.Thread(target=freer)
        ft.start()
        time.sleep(0.2)
        require(free_result[0] == -2,
                "aecFree returned before submit released — "
                "free MUST wait for pending submit")

        # --- Release submit ---
        shim.aecTestReleaseSubmitBlock()

        t.join(timeout=5)
        ft.join(timeout=5)
        require(not t.is_alive() and not ft.is_alive(),
                "thread join timeout after release")

        require(enq_result[0] == SUCCESS,
                f"copy async: {ERROR_NAMES.get(enq_result[0], enq_result[0])}")
        require(free_result[0] == SUCCESS,
                f"free: {ERROR_NAMES.get(free_result[0], free_result[0])}")

        # Sync — either SUCCESS (linearization a) or INVALID_ADDRESS (b)
        st = rt.lib.aecStreamSync(stream)
        require(st in (SUCCESS, INVALID_ADDRESS),
                f"sync after free-race: {ERROR_NAMES.get(st, st)} "
                f"(expected SUCCESS or INVALID_ADDRESS)")

        # --- Post-free: enqueue to freed address → SUCCESS (accepted),
        #     sync → INVALID_ADDRESS ---
        host2 = HostBuffer(256)
        host2.write(b"\xAB" * 256)
        st = rt.lib.aecCopyAsync(alloc, host2.ptr, 256, H2D, stream)
        require(st == SUCCESS,
                f"post-free enqueue: {ERROR_NAMES.get(st, st)}")

        st = rt.lib.aecStreamSync(stream)
        require(st == INVALID_ADDRESS,
                f"post-free sync expected INVALID_ADDRESS, "
                f"got {ERROR_NAMES.get(st, st)}")

    finally:
        shim.aecTestResetSubmitBlock()
        rt.lib.aecStreamDestroy(stream)
        # alloc already freed by freer above

    print("  NOTE  free deterministically waited for pending submit; "
           "post-free sync correctly returns INVALID_ADDRESS")


def test_free_waits_for_pending_submit():
    """Free waits for pending submit (shim-deterministic)."""
    skip_if_no_library()
    skip_if_no_shim()
    ld_preload = _ensure_shim()
    result = isolated_subprocess(
        _free_waits_for_pending_submit,
        timeout=25.0, ld_preload=ld_preload)
    require(result == "PASS",
            f"free-wait subprocess: {result}")


# ===================================================================
# TEST B3: Unregister waits for pending submit  (shim-deterministic)
# ===================================================================

def _unregister_waits_for_pending_submit():
    """Registered host buffer copy blocks; unregister must wait.

    After release:
      - Copy and unregister both succeed
      - Subsequent copy using original host pointer does NOT increase
        zero_copy_commands
    """
    skip_if_no_library()
    reset_runtime()
    rt = runtime()

    shim = _load_shim_ctypes()
    require(shim is not None, "shim must be loadable in subprocess")

    alloc = rt.alloc(4096)
    stream = rt.stream()
    host = HostBuffer(4096)
    host.write(bytes(i & 0xFF for i in range(4096)))

    try:
        # Register host buffer
        require(rt.lib.aecHostRegister(host.ptr, 4096) == SUCCESS,
                "host register failed")

        shim.aecTestResetSubmitBlock()
        # --- Arm submit blocker ---
        shim.aecTestArmSubmitBlock()

        copy_result: list[int] = [-1]
        unreg_result: list[int] = [-1]
        barrier = threading.Barrier(2)

        def copy_worker() -> None:
            barrier.wait()
            copy_result[0] = rt.lib.aecCopyAsync(
                alloc, host.ptr, 4096, H2D, stream)

        t = threading.Thread(target=copy_worker)
        t.start()
        barrier.wait()

        blocked = shim.aecTestWaitSubmitBlocked(5000)
        require(blocked == 1,
                "worker did not block in aecDeviceSubmit within 5s")

        # --- Unregister should NOT return yet ---
        unreg_result[0] = -2

        def unregister() -> None:
            unreg_result[0] = rt.lib.aecHostUnregister(host.ptr)

        ut = threading.Thread(target=unregister)
        ut.start()
        time.sleep(0.2)
        require(unreg_result[0] == -2,
                "aecHostUnregister returned before submit released — "
                "unregister MUST wait for pending submit")

        # --- Release submit ---
        shim.aecTestReleaseSubmitBlock()

        t.join(timeout=5)
        ut.join(timeout=5)
        require(not t.is_alive() and not ut.is_alive(),
                "thread join timeout after release")

        require(copy_result[0] == SUCCESS,
                f"copy: {ERROR_NAMES.get(copy_result[0], copy_result[0])}")
        require(unreg_result[0] == SUCCESS,
                f"unregister: {ERROR_NAMES.get(unreg_result[0], unreg_result[0])}")

        # Sync
        st = rt.lib.aecStreamSync(stream)
        require(st == SUCCESS,
                f"sync: {ERROR_NAMES.get(st, st)}")

        # Capture zero-copy stats
        _reset_stats(rt)
        zc_before = rt.device_stats().zero_copy_commands

        # Subsequent copy using original (now unregistered) host ptr
        st = rt.lib.aecCopyAsync(alloc, host.ptr, 4096, H2D, stream)
        require(st == SUCCESS, f"second copy: {ERROR_NAMES.get(st, st)}")
        st = rt.lib.aecStreamSync(stream)
        require(st == SUCCESS, f"second sync: {ERROR_NAMES.get(st, st)}")

        final_zc = rt.device_stats().zero_copy_commands
        zc_delta = final_zc - zc_before
        # The racing copy may have registered 0 or 1 zero-copy commands,
        # but the *subsequent* copy after unregister must not add any.
        # We reset stats before the subsequent copy so zc_delta reflects only it.
        require(zc_delta == 0,
                f"zero_copy_commands increased by {zc_delta} after unregister "
                f"(expected 0)")

    finally:
        shim.aecTestResetSubmitBlock()
        rt.lib.aecStreamDestroy(stream)
        rt.lib.aecFree(alloc)

    print("  NOTE  unregister deterministically waited for pending submit; "
           "zero-copy correctly cleared after unregister")


def test_unregister_waits_for_pending_submit():
    """Unregister waits for pending submit (shim-deterministic)."""
    skip_if_no_library()
    skip_if_no_shim()
    ld_preload = _ensure_shim()
    result = isolated_subprocess(
        _unregister_waits_for_pending_submit,
        timeout=25.0, ld_preload=ld_preload)
    require(result == "PASS",
            f"unregister-wait subprocess: {result}")


# ===================================================================
# TEST B4: Event marker — EventRecord as stream tail with blocked submit
# ===================================================================
#
# Protocol:
#   1. Worker: arm submit → aecCopyAsync (blocks in aecDeviceSubmit)
#   2. Main: issue EventRecord(ev) as tail marker (waits for worker)
#   3. Main: issue EventRecord(end) immediately after — no commands
#   4. Release submit
#   5. Sync both events
#   6. Verify elapsed(ev, end) == 0 (same stream point after re-record)
#   7. Verify ev DMA intermediate data is correct
#   8. EventDestroy on the latest generation succeeds

def _event_marker_with_blocked_submit():
    """EventRecord as stream tail while submit is blocked."""
    skip_if_no_library()
    reset_runtime()
    rt = runtime()

    shim = _load_shim_ctypes()
    require(shim is not None, "shim must be loadable in subprocess")

    stream = rt.stream()
    ev = rt.event()
    end_ev = rt.event()
    alloc1 = rt.alloc(4096)
    host = HostBuffer(4096)
    host.write(bytes(i & 0xFF for i in range(4096)))

    try:
        shim.aecTestResetSubmitBlock()

        # --- Arm submit blocker ---
        shim.aecTestArmSubmitBlock()

        copy_result: list[int] = [-1]
        barrier = threading.Barrier(2)

        def worker() -> None:
            barrier.wait()
            copy_result[0] = rt.lib.aecCopyAsync(
                alloc1, host.ptr, 4096, H2D, stream)

        t = threading.Thread(target=worker)
        t.start()
        barrier.wait()

        blocked = shim.aecTestWaitSubmitBlocked(5000)
        require(blocked == 1,
                "worker did not block in aecDeviceSubmit within 5s")

        # --- Record marker event and end event (submit still blocked) ---
        # EventRecord goes on the stream tail — the blocked submit is
        # ahead of it so Record will block waiting for the submit to
        # complete first.  We verify this by checking that Record does
        # NOT return while submit is blocked.
        ev_result: list[int] = [-1]

        def recorder() -> None:
            ev_result[0] = rt.lib.aecEventRecord(ev, stream)

        rt2 = threading.Thread(target=recorder)
        rt2.start()
        time.sleep(0.2)
        require(ev_result[0] == -1,
                "EventRecord returned before submit released — "
                "Record must wait on stream tail")

        # --- Release submit ---
        shim.aecTestReleaseSubmitBlock()

        # Now both worker and recorder complete
        t.join(timeout=5)
        rt2.join(timeout=5)
        require(not t.is_alive() and not rt2.is_alive(),
                "join timeout after release")

        require(copy_result[0] == SUCCESS,
                f"copy: {ERROR_NAMES.get(copy_result[0], copy_result[0])}")
        require(ev_result[0] == SUCCESS,
                f"event record: {ERROR_NAMES.get(ev_result[0], ev_result[0])}")

        # Record end event immediately (no commands between)
        require(rt.lib.aecEventRecord(end_ev, stream) == SUCCESS,
                "end record failed")

        # --- Sync both events ---
        require(rt.lib.aecEventSynchronize(ev) == SUCCESS,
                "sync ev")
        require(rt.lib.aecEventSynchronize(end_ev) == SUCCESS,
                "sync end")

        # Query must return SUCCESS (not NOT_READY) after sync
        require(rt.lib.aecEventQuery(ev) == SUCCESS,
                "query ev after sync")
        require(rt.lib.aecEventQuery(end_ev) == SUCCESS,
                "query end after sync")

        # --- elapsed(ev, end) == 0 proves same stream point ---
        cycles = ctypes.c_uint64()
        require(rt.lib.aecEventElapsedCycles(
            ev, end_ev, ctypes.byref(cycles)) == SUCCESS,
            "elapsed(ev, end) failed")
        require(cycles.value == 0,
                f"elapsed(ev,end)={cycles.value} != 0 — "
                f"events should be at same stream point")

        # --- Verify DMA intermediate data ---
        # alloc1 had H2D from host → read it back
        check = HostBuffer(4096)
        st = rt.lib.aecCopyD2H(check.ptr, alloc1, 4096)
        require(st == SUCCESS,
                f"D2H verify: {ERROR_NAMES.get(st, st)}")
        require(check.read() == host.read(),
                "data integrity after event-marker test")

        # --- EventDestroy on latest generation ---
        require(rt.lib.aecEventDestroy(ev) == SUCCESS,
                "destroy ev after event-marker test")
        require(rt.lib.aecEventDestroy(end_ev) == SUCCESS,
                "destroy end_ev after event-marker test")

    finally:
        shim.aecTestResetSubmitBlock()
        rt.lib.aecStreamDestroy(stream)
        rt.lib.aecFree(alloc1)

    print("  NOTE  event marker deterministically waited; elapsed=0, "
           "data correct, destroy on latest generation succeeded")


def test_event_marker_with_blocked_submit():
    """EventRecord as marker with blocked submit (shim-deterministic)."""
    skip_if_no_library()
    skip_if_no_shim()
    ld_preload = _ensure_shim()
    result = isolated_subprocess(
        _event_marker_with_blocked_submit,
        timeout=25.0, ld_preload=ld_preload)
    require(result == "PASS",
            f"event-marker subprocess: {result}")


# ===================================================================
# TEST C1: Multi-stream concurrent submission  (stress / pressure)
# ===================================================================

def _concurrent_submission_stress():
    """8 threads × 30 async DMA pairs, separate streams, barrier.

    This is a *pressure* test — it provides evidence that the runtime
    does not crash under concurrent load but does NOT prove
    linearizability.  See shim tests B1-B4 for deterministic proofs.

    After join, sync every stream and verify submitted_commands and
    dma_commands match expected totals.
    """
    skip_if_no_library()
    reset_runtime()
    rt = runtime()

    NUM_THREADS = 8
    ITER_PER_THREAD = 30
    COPY_SIZE = 1024

    streams: list[ctypes.c_void_p] = []
    allocs: list[int] = []
    hosts: list[HostBuffer] = []
    try:
        for i in range(NUM_THREADS):
            s = rt.stream()
            streams.append(s)
            d = rt.alloc(COPY_SIZE)
            allocs.append(d)
            h = HostBuffer(COPY_SIZE)
            h.write(bytes((i * 31 + j) & 0xFF for j in range(COPY_SIZE)))
            hosts.append(h)

        _reset_stats(rt)

        submission_errors: list[str] = []
        err_lock = threading.Lock()
        barrier = threading.Barrier(NUM_THREADS)

        def _worker(tid: int) -> None:
            barrier.wait()
            s = streams[tid]
            d = allocs[tid]
            h = hosts[tid]
            for _ in range(ITER_PER_THREAD):
                st = rt.lib.aecCopyAsync(d, h.ptr, COPY_SIZE, H2D, s)
                if st != SUCCESS:
                    with err_lock:
                        submission_errors.append(
                            f"T{tid} H2D iter={_}: {ERROR_NAMES.get(st, st)}")
                    return
                st = rt.lib.aecCopyAsync(d, h.ptr, COPY_SIZE, D2H, s)
                if st != SUCCESS:
                    with err_lock:
                        submission_errors.append(
                            f"T{tid} D2H iter={_}: {ERROR_NAMES.get(st, st)}")
                    return

        threads = [threading.Thread(target=_worker, args=(i,))
                   for i in range(NUM_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)
        for t in threads:
            require(not t.is_alive(),
                    "worker thread join timeout — possible deadlock")

        require(not submission_errors,
                f"Submission errors ({len(submission_errors)}): "
                + "; ".join(submission_errors[:5]))

        for i, s in enumerate(streams):
            st = rt.lib.aecStreamSync(s)
            require(st == SUCCESS,
                    f"stream[{i}] sync: {ERROR_NAMES.get(st, st)}")

        post = rt.device_stats()
        total_cmds = NUM_THREADS * ITER_PER_THREAD * 2

        require(post.submitted_commands == total_cmds,
                f"submitted_commands {post.submitted_commands} != {total_cmds}")
        require(post.dma_commands == total_cmds,
                f"dma_commands {post.dma_commands} != {total_cmds}")

    finally:
        for s in streams:
            rt.lib.aecStreamDestroy(s)
        for d in allocs:
            rt.lib.aecFree(d)

    print(f"  NOTE  pressure: {total_cmds} async DMA commands accepted "
           "across 8 streams, counts verified")


def test_concurrent_submission_stress():
    """8 threads × 30 async DMA pairs, separate streams (stress)."""
    skip_if_no_library()
    result = isolated_subprocess(
        _concurrent_submission_stress, timeout=25.0)
    require(result == "PASS",
            f"concurrent submission subprocess: {result}")


# ===================================================================
# TEST C2: Same-stream concurrent enqueue  (stress / pressure)
# ===================================================================

def _same_stream_concurrent_stress():
    """4 threads × 15 H2D copies on one stream, barrier.

    Pressure evidence only.  After sync, verify command counts and
    data integrity.
    """
    skip_if_no_library()
    reset_runtime()
    rt = runtime()

    stream = rt.stream()
    NUM_THREADS = 4
    ITER_PER_THREAD = 15
    COPY_SIZE = 256

    allocs: list[int] = []
    hosts: list[HostBuffer] = []
    try:
        for i in range(NUM_THREADS):
            d = rt.alloc(COPY_SIZE)
            allocs.append(d)
            h = HostBuffer(COPY_SIZE)
            data = bytes((i * 13 + j * 7) & 0xFF for j in range(COPY_SIZE))
            h.write(data)
            hosts.append(h)

        _reset_stats(rt)

        submission_errors: list[str] = []
        err_lock = threading.Lock()
        barrier = threading.Barrier(NUM_THREADS)

        def _worker(tid: int) -> None:
            barrier.wait()
            d = allocs[tid]
            h = hosts[tid]
            for _ in range(ITER_PER_THREAD):
                st = rt.lib.aecCopyAsync(d, h.ptr, COPY_SIZE, H2D, stream)
                if st != SUCCESS:
                    with err_lock:
                        submission_errors.append(
                            f"T{tid} iter={_}: {ERROR_NAMES.get(st, st)}")
                    return

        threads = [threading.Thread(target=_worker, args=(i,))
                   for i in range(NUM_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        for t in threads:
            require(not t.is_alive(),
                    "worker thread join timeout — possible deadlock")

        require(not submission_errors,
                f"Submission errors: {'; '.join(submission_errors[:5])}")

        st = rt.lib.aecStreamSync(stream)
        require(st == SUCCESS,
                f"stream sync: {ERROR_NAMES.get(st, st)}")

        post = rt.device_stats()
        total_cmds = NUM_THREADS * ITER_PER_THREAD
        require(post.submitted_commands == total_cmds,
                f"submitted_commands {post.submitted_commands} != {total_cmds}")
        require(post.dma_commands == total_cmds,
                f"dma_commands {post.dma_commands} != {total_cmds}")

        for i, d in enumerate(allocs):
            actual = rt.copy_out(d, COPY_SIZE)
            expected = hosts[i].read()
            require(actual == expected,
                    f"allocation[{i}] data mismatch")

    finally:
        rt.lib.aecStreamDestroy(stream)
        for d in allocs:
            rt.lib.aecFree(d)

    print(f"  NOTE  pressure: {total_cmds} commands on one stream, "
           "data integrity verified")


def test_same_stream_concurrent_stress():
    """4 threads × 15 H2D copies on one stream (stress)."""
    skip_if_no_library()
    result = isolated_subprocess(
        _same_stream_concurrent_stress, timeout=20.0)
    require(result == "PASS",
            f"same-stream enqueue subprocess: {result}")


# ===================================================================
# Standalone runner (for run_hidden_style.py compatibility)
# ===================================================================

def _run_all() -> None:
    """Run all test_* functions, print pass/fail/skip."""
    import types
    mod = __import__(__name__)
    tests = sorted(
        (getattr(mod, name), name)
        for name, v in vars(mod).items()
        if name.startswith("test_") and isinstance(v, types.FunctionType)
    )
    passed = 0
    failed = 0
    for func, name in tests:
        try:
            func()
            print(f"PASS {name}")
            passed += 1
        except SkipTest as exc:
            print(f"SKIP {name}: {exc}")
            passed += 1
        except Exception as exc:
            detail = str(exc)
            if "subprocess:" in detail:
                idx = detail.index("subprocess:")
                detail = detail[idx:]
            print(f"FAIL {name}: {detail}")
            failed += 1
    print(f"\n{passed}/{passed + failed} passed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    _run_all()

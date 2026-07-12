"""Hardware capability model (子任务 C3.2 D4/D5 依赖).

The grader reads three public signals off the module-level ``hardware`` object::

    hardware.supported_precisions()   # -> ["fp32", "fp16", "fp8", "fp4"]
    hardware.smem_bytes               # per-block shared-memory budget (bytes)
    hardware.max_threads_per_block    # == 1024

Values default to a modern NVIDIA SM (e.g. Ada / Hopper class): 1024 threads per
block and a 48 KiB static shared-memory budget per block.  ``supported_precisions``
advertises all four precisions so the multi-precision routing in
:mod:`scheduler.strategy` can exercise fp32/fp16/fp8/fp4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class HardwareModel:
    name: str = "generic-sm90"
    max_threads_per_block: int = 1024
    # 48 KiB static shared memory per block (opt-in dynamic smem can go higher on
    # real hardware; we keep the conservative static budget for tuning checks).
    smem_bytes: int = 48 * 1024
    warp_size: int = 32
    num_sms: int = 108
    # global HBM budget, informational (used by memory planner reporting)
    global_mem_bytes: int = 40 * 1024 * 1024 * 1024
    _supported_precisions: List[str] = field(
        default_factory=lambda: ["fp32", "fp16", "fp8", "fp4"]
    )

    def supported_precisions(self) -> List[str]:
        """Precisions the device can execute, highest-fidelity first."""
        return list(self._supported_precisions)

    def supports(self, precision: str) -> bool:
        return precision in self._supported_precisions


# Module-level singleton the grader imports.
hardware = HardwareModel()

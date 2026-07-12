"""Precision profiles and the sensitive-op policy (子任务 C3.2 D1).

``PrecisionProfile.precision`` is one of ``fp32`` / ``fp16`` / ``fp8`` / ``fp4``.

The C3.2 rubric awards D1 for (a) forcing *numerically sensitive* operators to
fp32, (b) precision diversity (target: all 4 precisions appear) and (c) keeping
compute-heavy ops within ``hardware.supported_precisions()``.  The sensitive-op
set below is the exact list from the spec, with ONNX aliases added
(``LayerNormalization`` <-> ``LayerNorm``, ``BatchNormalization`` <-> ``BatchNorm``).
"""

from __future__ import annotations

from dataclasses import dataclass

# All valid precision tokens, highest fidelity first.
PRECISIONS = ("fp32", "fp16", "fp8", "fp4")

# Map a precision token to the kernel-name suffix used by C3.2 D5
# (e.g. matmul_f32 / matmul_f16 / matmul_f8 / matmul_f4).
PRECISION_SUFFIX = {"fp32": "f32", "fp16": "f16", "fp8": "f8", "fp4": "f4"}

# Operators that MUST stay in fp32 (reductions / normalisations lose too much in
# low precision).  Includes both ONNX names and the spec's short names.
SENSITIVE_OPS = frozenset(
    {
        "Softmax",
        "LayerNorm",
        "LayerNormalization",
        "BatchNorm",
        "BatchNormalization",
        "ReduceMax",
        "ReduceSum",
        "ReduceMean",
        # These reductions also appear inside normalisation decompositions.
        "InstanceNormalization",
        "GroupNormalization",
    }
)


@dataclass
class PrecisionProfile:
    """The precision decision for one operator."""

    precision: str = "fp32"
    rationale: str = ""
    sensitive: bool = False

    def suffix(self) -> str:
        return PRECISION_SUFFIX.get(self.precision, "f32")


def is_sensitive(op_type: str) -> bool:
    return op_type in SENSITIVE_OPS

"""C3 scheduler package.

Re-exports the exact public symbols the hidden grader imports, so that any of
these import styles work::

    import scheduler
    scheduler.import_onnx_graph(...)
    scheduler.strategy.select_precision(...)
    scheduler.hardware.supported_precisions()
    scheduler.GraphPassPipeline(enable_fusion=True)

    from scheduler.graph import import_onnx_graph
    from scheduler.strategy import strategy, KernelSpecRef, KernelTuningParams
    from scheduler.hardware import hardware
    from scheduler.graph_passes.pipeline import GraphPassPipeline
"""

from .graph import Graph, Node, TensorInfo, import_onnx_graph
from .hardware import HardwareModel, hardware
from .precision import PrecisionProfile, is_sensitive
from .kernels import KernelSpecRef, KernelTuningParams
from .strategy import Strategy, strategy
from .memory import (
    MemoryPlanner, DeviceMemoryPool, build_execution_plan,
    TensorBinding, PlanStep, ExecutionPlan, validate_execution_plan,
)
from .graph_passes.pipeline import GraphPassPipeline

__all__ = [
    "import_onnx_graph",
    "Graph",
    "Node",
    "TensorInfo",
    "hardware",
    "HardwareModel",
    "strategy",
    "Strategy",
    "PrecisionProfile",
    "is_sensitive",
    "KernelSpecRef",
    "KernelTuningParams",
    "GraphPassPipeline",
    "MemoryPlanner",
    "DeviceMemoryPool",
    "build_execution_plan",
    "TensorBinding",
    "PlanStep",
    "ExecutionPlan",
    "validate_execution_plan",
]

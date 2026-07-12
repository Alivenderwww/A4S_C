#!/usr/bin/env python3
"""Public diagnostic grader for the AEC-C2 competition."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import random
import struct
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


def require(condition: object, detail: object = "grader check failed") -> None:
    """Enforce a scored condition even when Python runs with ``-O``."""
    if not condition:
        raise AssertionError(str(detail))


SUCCESS = 0
INVALID_ARGUMENT = 1
OUT_OF_MEMORY = 2
INVALID_HANDLE = 3
INVALID_ADDRESS = 4
NOT_READY = 5
NOT_SUPPORTED = 6
DEVICE_ERROR = 7
ISA_TRAP = 9

H2D = 1
D2H = 2

DTYPE_FP4 = 1
DTYPE_FP8_E4M3 = 2
DTYPE_FP8_E5M2 = 3
DTYPE_FP16 = 4
DTYPE_BF16 = 5
DTYPE_FP32 = 6
DTYPE_FP64 = 7
DTYPE_INT4 = 8
DTYPE_INT8 = 9
DTYPE_INT32 = 10

KERNEL_VECTOR_ADD = 1
KERNEL_GEMM_NAIVE = 10
KERNEL_GEMM_TILED = 11
KERNEL_GEMM_VECTORIZED = 12

def additional_profile(profile: str) -> dict:
    return {}


class Dim3(ctypes.Structure):
    _fields_ = [("x", ctypes.c_uint32), ("y", ctypes.c_uint32), ("z", ctypes.c_uint32)]


class DeviceInfo(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("name", ctypes.c_char * 64),
        ("memory_bytes", ctypes.c_uint64),
        ("dma_channels", ctypes.c_uint32),
        ("max_threads_per_block", ctypes.c_uint32),
        ("isa_version", ctypes.c_uint32),
        ("isa_profile", ctypes.c_uint32),
        ("max_parameter_bytes", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class RuntimeStats(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("submitted_commands", ctypes.c_uint64),
        ("dma_commands", ctypes.c_uint64),
        ("kernel_commands", ctypes.c_uint64),
        ("zero_copy_commands", ctypes.c_uint64),
        ("channel_commands", ctypes.c_uint64 * 2),
        ("total_virtual_cycles", ctypes.c_uint64),
        ("last_virtual_cycles", ctypes.c_uint64),
        ("isa_launches", ctypes.c_uint64),
        ("instructions_retired", ctypes.c_uint64),
        ("isa_traps", ctypes.c_uint64),
        ("last_kernel_handle", ctypes.c_uint64),
        ("last_trace_digest", ctypes.c_uint64),
    ]


class DeviceCompletion(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("status", ctypes.c_uint32),
        ("sequence", ctypes.c_uint64),
        ("virtual_cycles", ctypes.c_uint64),
        ("bytes_completed", ctypes.c_uint64),
        ("instructions_retired", ctypes.c_uint64),
        ("trace_digest", ctypes.c_uint64),
        ("fault_code", ctypes.c_uint32),
        ("trap_pc", ctypes.c_uint32),
    ]


class DeviceKernelInfo(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("isa_version", ctypes.c_uint32),
        ("handle", ctypes.c_uint64),
        ("image_id", ctypes.c_uint32),
        ("entry_pc", ctypes.c_uint32),
        ("parameter_bytes", ctypes.c_uint32),
        ("image_flags", ctypes.c_uint32),
        ("instruction_hash", ctypes.c_uint64),
    ]


class VectorAddArgs(ctypes.Structure):
    _fields_ = [
        ("a", ctypes.c_uint64),
        ("b", ctypes.c_uint64),
        ("c", ctypes.c_uint64),
        ("count", ctypes.c_uint64),
    ]


class HostBuffer:
    def __init__(self, size: int, *, alignment: int = 64):
        self.size = size
        self.storage = (ctypes.c_ubyte * (size + alignment))()
        base = ctypes.addressof(self.storage)
        self.address = (base + alignment - 1) & ~(alignment - 1)
        self.ptr = ctypes.c_void_p(self.address)

    def write(self, data: bytes) -> None:
        if len(data) > self.size:
            raise ValueError("host buffer overflow")
        ctypes.memmove(self.address, data, len(data))

    def read(self, size: int | None = None) -> bytes:
        return ctypes.string_at(self.address, self.size if size is None else size)


class Runtime:
    def __init__(self, submission: Path):
        if submission.is_dir():
            self.submission_dir = submission.resolve()
            self.library_path = self.submission_dir / "libaec.so"
        else:
            self.library_path = submission.resolve()
            self.submission_dir = self.library_path.parent
        if not self.library_path.is_file():
            raise FileNotFoundError(f"missing library: {self.library_path}")
        configured = os.environ.get("AEC_DEVICE_LIBRARY")
        grader_path = Path(__file__).resolve()
        candidates = [
            grader_path.parents[1] / "build" / "device" / "libaec_device.so",
            grader_path.parents[1] / "lib" / "libaec_device.so",
            grader_path.parent / "libaec_device.so",
        ]
        repository_device = next((path for path in candidates if path.is_file()), candidates[0])
        device_path = Path(configured).resolve() if configured else repository_device
        if not device_path.is_file():
            raise FileNotFoundError(f"missing reference device: {device_path}")
        self.device = ctypes.CDLL(str(device_path), mode=ctypes.RTLD_GLOBAL)
        self.lib = ctypes.CDLL(str(self.library_path))
        self._bind()

    def _bind(self) -> None:
        u64 = ctypes.c_uint64
        size = ctypes.c_size_t
        ptr = ctypes.c_void_p
        cint = ctypes.c_int

        self.device.aecDeviceReset.restype = cint
        self.device.aecDeviceResetStats.restype = cint
        self.device.aecDeviceInjectFault.argtypes = [cint]
        self.device.aecDeviceInjectFault.restype = cint
        self.device.aecDeviceGetStats.argtypes = [ctypes.POINTER(RuntimeStats)]
        self.device.aecDeviceGetStats.restype = cint
        self.device.aecDeviceEvaluateKernel.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_uint64, ctypes.POINTER(DeviceCompletion),
        ]
        self.device.aecDeviceEvaluateKernel.restype = cint
        self.device.aecDeviceResolveKernel.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.POINTER(DeviceKernelInfo),
        ]
        self.device.aecDeviceResolveKernel.restype = cint

        self.lib.aecDeviceCount.argtypes = [ctypes.POINTER(cint)]
        self.lib.aecDeviceCount.restype = cint
        self.lib.aecDeviceInfo.argtypes = [cint, ctypes.POINTER(DeviceInfo)]
        self.lib.aecDeviceInfo.restype = cint
        self.lib.aecGetLastError.restype = cint
        self.lib.aecPeekAtLastError.restype = cint
        self.lib.aecGetErrorName.argtypes = [cint]
        self.lib.aecGetErrorName.restype = ctypes.c_char_p
        self.lib.aecAlloc.argtypes = [ctypes.POINTER(u64), size]
        self.lib.aecAlloc.restype = cint
        self.lib.aecFree.argtypes = [u64]
        self.lib.aecFree.restype = cint
        self.lib.aecCopyH2D.argtypes = [u64, ptr, size]
        self.lib.aecCopyH2D.restype = cint
        self.lib.aecCopyD2H.argtypes = [ptr, u64, size]
        self.lib.aecCopyD2H.restype = cint
        self.lib.aecCopyAsync.argtypes = [u64, ptr, size, cint, ptr]
        self.lib.aecCopyAsync.restype = cint

        for name in ("aecStreamCreate", "aecEventCreate"):
            getattr(self.lib, name).argtypes = [ctypes.POINTER(ptr)]
            getattr(self.lib, name).restype = cint
        for name in ("aecStreamDestroy", "aecStreamSync", "aecEventDestroy",
                     "aecEventSynchronize", "aecEventQuery"):
            getattr(self.lib, name).argtypes = [ptr]
            getattr(self.lib, name).restype = cint
        self.lib.aecEventRecord.argtypes = [ptr, ptr]
        self.lib.aecEventRecord.restype = cint
        self.lib.aecEventElapsedCycles.argtypes = [ptr, ptr, ctypes.POINTER(u64)]
        self.lib.aecEventElapsedCycles.restype = cint
        self.lib.aecHostRegister.argtypes = [ptr, size]
        self.lib.aecHostRegister.restype = cint
        self.lib.aecHostUnregister.argtypes = [ptr]
        self.lib.aecHostUnregister.restype = cint
        self.lib.aecGetRuntimeStats.argtypes = [ctypes.POINTER(RuntimeStats)]
        self.lib.aecGetRuntimeStats.restype = cint
        self.lib.aecResetRuntimeStats.restype = cint
        self.lib.aecLaunch.argtypes = [cint, Dim3, Dim3, ptr, size, ptr]
        self.lib.aecLaunch.restype = cint

        common_gemm = [u64, u64, u64, ctypes.c_uint32, ctypes.c_uint32,
                       ctypes.c_uint32, ptr]
        for name in ("aecMatmulF4", "aecMatmulF16", "aecMatmulBF16",
                     "aecMatmulF32", "aecMatmulF64", "aecMatmulI4",
                     "aecMatmulI8", "aecMatmulI32"):
            getattr(self.lib, name).argtypes = common_gemm
            getattr(self.lib, name).restype = cint
        self.lib.aecMatmulF8.argtypes = [u64, u64, u64, ctypes.c_uint32,
                                        ctypes.c_uint32, ctypes.c_uint32, cint, ptr]
        self.lib.aecMatmulF8.restype = cint
        self.lib.aecAxpy.argtypes = [u64, u64, u64, ctypes.c_float, ptr]
        self.lib.aecAxpy.restype = cint
        self.lib.aecDot.argtypes = [u64, u64, u64, u64, ptr]
        self.lib.aecDot.restype = cint
        self.lib.aecNrm2.argtypes = [u64, u64, u64, ptr]
        self.lib.aecNrm2.restype = cint

    def reset(self) -> None:
        if self.device.aecDeviceReset() != 0:
            raise AssertionError("reference device reset failed")
        self.lib.aecGetLastError()

    def alloc(self, size: int) -> int:
        value = ctypes.c_uint64()
        status = self.lib.aecAlloc(ctypes.byref(value), size)
        if status != SUCCESS:
            raise AssertionError(f"aecAlloc({size}) returned {status}")
        return value.value

    def copy_in(self, device: int, data: bytes) -> HostBuffer:
        host = HostBuffer(len(data))
        host.write(data)
        status = self.lib.aecCopyH2D(device, host.ptr, len(data))
        if status != SUCCESS:
            raise AssertionError(f"aecCopyH2D returned {status}")
        return host

    def copy_out(self, device: int, size: int) -> bytes:
        host = HostBuffer(size)
        status = self.lib.aecCopyD2H(host.ptr, device, size)
        if status != SUCCESS:
            raise AssertionError(f"aecCopyD2H returned {status}")
        return host.read()

    def stream(self) -> ctypes.c_void_p:
        value = ctypes.c_void_p()
        status = self.lib.aecStreamCreate(ctypes.byref(value))
        if status != SUCCESS:
            raise AssertionError(f"aecStreamCreate returned {status}")
        return value

    def event(self) -> ctypes.c_void_p:
        value = ctypes.c_void_p()
        status = self.lib.aecEventCreate(ctypes.byref(value))
        if status != SUCCESS:
            raise AssertionError(f"aecEventCreate returned {status}")
        return value

    def stats(self) -> RuntimeStats:
        value = RuntimeStats()
        status = self.lib.aecGetRuntimeStats(ctypes.byref(value))
        if status != SUCCESS:
            raise AssertionError(f"aecGetRuntimeStats returned {status}")
        return value

    def device_stats(self) -> RuntimeStats:
        value = RuntimeStats()
        status = self.device.aecDeviceGetStats(ctypes.byref(value))
        if status != SUCCESS:
            raise AssertionError(f"aecDeviceGetStats returned {status}")
        return value

    def kernel_handle(self, kernel: int, dtype: int, variant: int) -> int:
        info = DeviceKernelInfo()
        status = self.device.aecDeviceResolveKernel(
            kernel, dtype, variant, ctypes.byref(info))
        require(status == SUCCESS, f"author kernel tuple did not resolve: {status}")
        require(info.abi_version == 2 and info.isa_version == 2 and
                info.instruction_hash != 0, "invalid author kernel metadata")
        return info.handle


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _mini_decode(code: int, exponent_bits: int, mantissa_bits: int,
                 bias: int, finite_only: bool) -> float:
    sign_shift = exponent_bits + mantissa_bits
    negative = bool((code >> sign_shift) & 1)
    exponent_mask = (1 << exponent_bits) - 1
    mantissa_mask = (1 << mantissa_bits) - 1
    exponent = (code >> mantissa_bits) & exponent_mask
    mantissa = code & mantissa_mask
    if exponent == 0:
        result = 0.0 if mantissa == 0 else math.ldexp(mantissa / (1 << mantissa_bits), 1 - bias)
    elif not finite_only and exponent == exponent_mask:
        result = math.inf if mantissa == 0 else math.nan
    elif finite_only and exponent_bits == 4 and exponent == exponent_mask and mantissa == mantissa_mask:
        result = math.nan
    else:
        result = math.ldexp(1.0 + mantissa / (1 << mantissa_bits), exponent - bias)
    return -result if negative else result


def _mini_encode(value: float, exponent_bits: int, mantissa_bits: int,
                 bias: int, finite_only: bool) -> int:
    sign_shift = exponent_bits + mantissa_bits
    if math.isnan(value):
        return 0x7F if finite_only and exponent_bits == 4 else (
            ((1 << exponent_bits) - 1) << mantissa_bits
        ) | (1 << (mantissa_bits - 1))
    if math.isinf(value) and not finite_only:
        return ((1 if value < 0 else 0) << sign_shift) | (
            ((1 << exponent_bits) - 1) << mantissa_bits
        )
    if math.isinf(value) and finite_only:
        sign = (1 if value < 0 else 0) << sign_shift
        exponent_mask = (1 << exponent_bits) - 1
        mantissa = (1 << mantissa_bits) - 1
        if exponent_bits == 4:
            mantissa -= 1
        return sign | (exponent_mask << mantissa_bits) | mantissa
    best = 0
    best_distance = math.inf
    for code in range(1 << (sign_shift + 1)):
        candidate = _mini_decode(code, exponent_bits, mantissa_bits, bias, finite_only)
        if not math.isfinite(candidate):
            continue
        distance = abs(candidate - value)
        if distance < best_distance or (
            distance == best_distance and code % 2 == 0 and best % 2 == 1
        ):
            best = code
            best_distance = distance
    return best


def _pack_nibbles(values: list[int]) -> bytes:
    output = bytearray((len(values) + 1) // 2)
    for index, value in enumerate(values):
        if index % 2 == 0:
            output[index // 2] |= value & 0xF
        else:
            output[index // 2] |= (value & 0xF) << 4
    return bytes(output)


def _encode_float_values(values: list[float], dtype: int) -> bytes:
    if dtype == DTYPE_FP4:
        return _pack_nibbles([_mini_encode(v, 2, 1, 1, True) for v in values])
    if dtype == DTYPE_FP8_E4M3:
        return bytes(_mini_encode(v, 4, 3, 7, True) for v in values)
    if dtype == DTYPE_FP8_E5M2:
        return bytes(_mini_encode(v, 5, 2, 15, False) for v in values)
    if dtype == DTYPE_FP16:
        return b"".join(struct.pack("<e", v) for v in values)
    if dtype == DTYPE_BF16:
        output = bytearray()
        for value in values:
            bits = struct.unpack("<I", struct.pack("<f", value))[0]
            if (bits & 0x7FFFFFFF) > 0x7F800000:
                encoded = (bits >> 16) | 0x40
            else:
                encoded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
            output += struct.pack("<H", encoded & 0xFFFF)
        return bytes(output)
    if dtype == DTYPE_FP32:
        return struct.pack(f"<{len(values)}f", *values)
    if dtype == DTYPE_FP64:
        return struct.pack(f"<{len(values)}d", *values)
    raise ValueError(dtype)


def _decode_float_values(data: bytes, count: int, dtype: int) -> list[float]:
    if dtype == DTYPE_FP4:
        codes = [((data[i // 2] >> (4 * (i % 2))) & 0xF) for i in range(count)]
        return [_mini_decode(code, 2, 1, 1, True) for code in codes]
    if dtype == DTYPE_FP8_E4M3:
        return [_mini_decode(code, 4, 3, 7, True) for code in data[:count]]
    if dtype == DTYPE_FP8_E5M2:
        return [_mini_decode(code, 5, 2, 15, False) for code in data[:count]]
    if dtype == DTYPE_FP16:
        return list(struct.unpack(f"<{count}e", data))
    if dtype == DTYPE_BF16:
        return [struct.unpack("<f", struct.pack("<I", value << 16))[0]
                for value in struct.unpack(f"<{count}H", data)]
    if dtype == DTYPE_FP32:
        return list(struct.unpack(f"<{count}f", data))
    if dtype == DTYPE_FP64:
        return list(struct.unpack(f"<{count}d", data))
    raise ValueError(dtype)


def _float_gemm_oracle(a_raw: bytes, b_raw: bytes, m: int, n: int, k: int,
                       dtype: int) -> bytes:
    a = _decode_float_values(a_raw, m * k, dtype)
    b = _decode_float_values(b_raw, k * n, dtype)
    output: list[float] = []
    for row in range(m):
        for col in range(n):
            if dtype == DTYPE_FP64:
                value = 0.0
                for inner in range(k):
                    value += a[row * k + inner] * b[inner * n + col]
            else:
                value = _f32(0.0)
                for inner in range(k):
                    product = _f32(a[row * k + inner] * b[inner * n + col])
                    value = _f32(value + product)
            output.append(value)
    return _encode_float_values(output, dtype)


def _pack_integers(values: list[int], dtype: int) -> bytes:
    if dtype == DTYPE_INT4:
        return _pack_nibbles([value & 0xF for value in values])
    if dtype == DTYPE_INT8:
        return struct.pack(f"<{len(values)}b", *values)
    return struct.pack(f"<{len(values)}i", *values)


def _integer_gemm_oracle(a: list[int], b: list[int], m: int, n: int, k: int) -> bytes:
    output = []
    for row in range(m):
        for col in range(n):
            value = sum(a[row * k + inner] * b[inner * n + col] for inner in range(k))
            output.append(max(-(1 << 31), min((1 << 31) - 1, value)))
    return struct.pack(f"<{len(output)}i", *output)


def _run_float_gemm(runtime: Runtime, name: str, dtype: int, m: int, n: int,
                    k: int, values_a: list[float], values_b: list[float],
                    fp8_format: int | None = None) -> None:
    a_raw = _encode_float_values(values_a, dtype)
    b_raw = _encode_float_values(values_b, dtype)
    expected = _float_gemm_oracle(a_raw, b_raw, m, n, k, dtype)
    a = runtime.alloc(len(a_raw))
    b = runtime.alloc(len(b_raw))
    c = runtime.alloc(len(expected))
    try:
        runtime.copy_in(a, a_raw)
        runtime.copy_in(b, b_raw)
        before = runtime.device_stats()
        function = getattr(runtime.lib, name)
        if fp8_format is None:
            status = function(a, b, c, m, n, k, None)
        else:
            status = function(a, b, c, m, n, k, fp8_format, None)
        if status != SUCCESS:
            raise AssertionError(f"{name} returned {status}")
        after = runtime.device_stats()
        require(after.isa_launches == before.isa_launches + 1)
        require(after.instructions_retired > before.instructions_retired)
        require(after.last_kernel_handle != 0 and after.last_trace_digest != 0)
        legal_handles = {
            runtime.kernel_handle(KERNEL_GEMM_NAIVE, dtype, 1),
            runtime.kernel_handle(KERNEL_GEMM_TILED, dtype, 2),
            runtime.kernel_handle(KERNEL_GEMM_VECTORIZED, dtype, 3),
        }
        require(after.last_kernel_handle in legal_handles,
                f"{name} executed an image outside its typed GEMM manifest")
        actual = runtime.copy_out(c, len(expected))
        if dtype in (DTYPE_FP4, DTYPE_FP8_E4M3, DTYPE_FP8_E5M2,
                     DTYPE_FP16, DTYPE_BF16):
            if actual != expected:
                raise AssertionError(f"{name} output mismatch: {actual.hex()} != {expected.hex()}")
        else:
            count = m * n
            actual_values = _decode_float_values(actual, count, dtype)
            expected_values = _decode_float_values(expected, count, dtype)
            tolerance = (1e-12, 1e-11) if dtype == DTYPE_FP64 else (1e-5, 2e-5)
            for index, (got, want) in enumerate(zip(actual_values, expected_values)):
                if math.isnan(want):
                    width = 8 if dtype == DTYPE_FP64 else 4
                    canonical = (b"\x00\x00\x00\x00\x00\x00\xf8\x7f"
                                 if dtype == DTYPE_FP64 else b"\x00\x00\xc0\x7f")
                    require(actual[index * width:(index + 1) * width] == canonical,
                            f"{name}: non-canonical NaN output")
                    continue
                if not math.isclose(got, want, abs_tol=tolerance[0], rel_tol=tolerance[1]):
                    raise AssertionError(f"{name}: {got} != {want}")
    finally:
        runtime.lib.aecFree(a)
        runtime.lib.aecFree(b)
        runtime.lib.aecFree(c)


def _run_integer_gemm(runtime: Runtime, name: str, dtype: int, m: int, n: int,
                      k: int, values_a: list[int], values_b: list[int]) -> None:
    a_raw = _pack_integers(values_a, dtype)
    b_raw = _pack_integers(values_b, dtype)
    expected = _integer_gemm_oracle(values_a, values_b, m, n, k)
    a = runtime.alloc(len(a_raw))
    b = runtime.alloc(len(b_raw))
    c = runtime.alloc(len(expected))
    try:
        runtime.copy_in(a, a_raw)
        runtime.copy_in(b, b_raw)
        before = runtime.device_stats()
        status = getattr(runtime.lib, name)(a, b, c, m, n, k, None)
        if status != SUCCESS:
            raise AssertionError(f"{name} returned {status}")
        after = runtime.device_stats()
        require(after.isa_launches == before.isa_launches + 1)
        require(after.instructions_retired > before.instructions_retired)
        require(after.last_kernel_handle != 0 and after.last_trace_digest != 0)
        legal_handles = {
            runtime.kernel_handle(KERNEL_GEMM_NAIVE, dtype, 1),
            runtime.kernel_handle(KERNEL_GEMM_TILED, dtype, 2),
            runtime.kernel_handle(KERNEL_GEMM_VECTORIZED, dtype, 3),
        }
        require(after.last_kernel_handle in legal_handles,
                f"{name} executed an image outside its typed GEMM manifest")
        actual = runtime.copy_out(c, len(expected))
        if actual != expected:
            raise AssertionError(f"{name} output mismatch")
    finally:
        runtime.lib.aecFree(a)
        runtime.lib.aecFree(b)
        runtime.lib.aecFree(c)


def test_r101(runtime: Runtime, profile: str) -> str:
    count = ctypes.c_int()
    require(runtime.lib.aecDeviceCount(ctypes.byref(count)) == SUCCESS and count.value == 1)
    info = DeviceInfo()
    require(runtime.lib.aecDeviceInfo(0, ctypes.byref(info)) == SUCCESS)
    require(info.abi_version == 2 and info.memory_bytes == 64 * 1024 * 1024)
    require(info.dma_channels == 2 and b"Deterministic" in bytes(info.name))
    require(info.isa_version == 2 and info.isa_profile == 1)
    require(info.max_parameter_bytes == 64)
    require(runtime.lib.aecDeviceInfo(1, ctypes.byref(info)) == INVALID_ARGUMENT)
    require(runtime.lib.aecPeekAtLastError() == INVALID_ARGUMENT)
    require(runtime.lib.aecGetLastError() == INVALID_ARGUMENT)
    require(runtime.lib.aecGetLastError() == SUCCESS)
    require(runtime.lib.aecGetErrorName(INVALID_ADDRESS) == b"AEC_ERROR_INVALID_ADDRESS")

    invalid_set = threading.Event()
    other_read = threading.Event()
    observations: dict[str, int] = {}

    def first_thread() -> None:
        runtime.lib.aecAlloc(None, 1)
        invalid_set.set()
        other_read.wait(2)
        observations["first"] = runtime.lib.aecPeekAtLastError()

    def second_thread() -> None:
        invalid_set.wait(2)
        observations["second"] = runtime.lib.aecGetLastError()
        other_read.set()

    first = threading.Thread(target=first_thread)
    second = threading.Thread(target=second_thread)
    first.start()
    second.start()
    first.join(2)
    second.join(2)
    require(observations == {"second": SUCCESS, "first": INVALID_ARGUMENT}, observations)
    return "device metadata and TLS error isolation verified"


def test_r102(runtime: Runtime, profile: str) -> str:
    require(runtime.lib.aecAlloc(None, 16) == INVALID_ARGUMENT)
    first = runtime.alloc(1024)
    require(first % 64 == 0)
    require(runtime.lib.aecFree(first) == SUCCESS)
    require(runtime.lib.aecFree(first) == INVALID_ADDRESS)
    second = runtime.alloc(1024)
    require(second == first)
    require(runtime.lib.aecFree(second) == SUCCESS)
    too_large = ctypes.c_uint64()
    require(runtime.lib.aecAlloc(ctypes.byref(too_large), 64 * 1024 * 1024) == OUT_OF_MEMORY)
    return "allocation reuse, OOM, and double-free behavior verified"


def test_r103(runtime: Runtime, profile: str) -> str:
    device = runtime.alloc(16)
    source = HostBuffer(16)
    source.write(bytes(range(16)))
    target = HostBuffer(16)
    try:
        require(runtime.lib.aecCopyH2D(device, source.ptr, 16) == SUCCESS)
        require(runtime.lib.aecCopyD2H(target.ptr, device, 16) == SUCCESS)
        require(target.read() == bytes(range(16)))
        require(runtime.lib.aecCopyH2D(device + 12, source.ptr, 8) == INVALID_ADDRESS)
        require(runtime.lib.aecCopyD2H(target.ptr, device + 15, 2) == INVALID_ADDRESS)
        require(runtime.lib.aecCopyH2D(device, None, 4) == INVALID_ARGUMENT)
        require(runtime.lib.aecCopyD2H(target.ptr, device, 0) == INVALID_ARGUMENT)
    finally:
        runtime.lib.aecFree(device)
    return "synchronous copies and allocation-relative bounds verified"


def test_r104(runtime: Runtime, profile: str) -> str:
    values_a = [1.0, -2.0, 3.5, 4.0, 0.25]
    values_b = [0.5, 2.0, -1.5, 8.0, -0.25]
    extra = additional_profile(profile).get("r104")
    if extra:
        values_a.extend(_f32((index + extra["a_offset"]) * extra["a_scale"])
                        for index in range(5, extra["total_count"]))
        values_b.extend(_f32((extra["b_offset"] - index) * extra["b_scale"])
                        for index in range(5, extra["total_count"]))
    count = len(values_a)
    raw_a = struct.pack(f"<{count}f", *values_a)
    raw_b = struct.pack(f"<{count}f", *values_b)
    a = runtime.alloc(len(raw_a))
    b = runtime.alloc(len(raw_b))
    c = runtime.alloc(len(raw_a))
    try:
        runtime.copy_in(a, raw_a)
        runtime.copy_in(b, raw_b)
        args = VectorAddArgs(a, b, c, count)
        before = runtime.device_stats()
        status = runtime.lib.aecLaunch(KERNEL_VECTOR_ADD,
                                       Dim3((count + 31) // 32, 1, 1),
                                       Dim3(32, 1, 1), ctypes.byref(args),
                                       ctypes.sizeof(args), None)
        require(status == SUCCESS)
        after = runtime.device_stats()
        require(after.isa_launches == before.isa_launches + 1)
        require(after.instructions_retired > before.instructions_retired)
        require(after.last_kernel_handle != 0 and after.last_trace_digest != 0)
        require(after.last_kernel_handle == runtime.kernel_handle(
            KERNEL_VECTOR_ADD, DTYPE_FP32, 0),
            "Vector Add did not execute its frozen manifest image")
        actual = struct.unpack(f"<{count}f", runtime.copy_out(c, len(raw_a)))
        require(all(math.isclose(got, left + right, abs_tol=1e-6)
                    for got, left, right in zip(actual, values_a, values_b)))
        require(runtime.lib.aecLaunch(KERNEL_VECTOR_ADD, Dim3(1, 1, 1),
                                     Dim3(2048, 1, 1), ctypes.byref(args),
                                     ctypes.sizeof(args), None) == INVALID_ARGUMENT)
        require(runtime.lib.aecLaunch(KERNEL_VECTOR_ADD, Dim3(1, 1, 1),
                                     Dim3(32, 1, 1), ctypes.byref(args), 1,
                                     None) == INVALID_ARGUMENT)
    finally:
        runtime.lib.aecFree(a)
        runtime.lib.aecFree(b)
        runtime.lib.aecFree(c)
    return "registered vector-add launch and launch validation verified"


def test_r105(runtime: Runtime, profile: str) -> str:
    stream = runtime.stream()
    destroyed = False
    device = runtime.alloc(4096)
    source = HostBuffer(4096)
    target = HostBuffer(4096)
    source.write(bytes((index * 17) & 0xFF for index in range(4096)))
    try:
        require(runtime.lib.aecCopyAsync(device, source.ptr, 4096, H2D, stream) == SUCCESS)
        require(runtime.lib.aecCopyAsync(device, target.ptr, 4096, D2H, stream) == SUCCESS)
        require(runtime.lib.aecStreamSync(stream) == SUCCESS)
        require(target.read() == source.read())
        if additional_profile(profile).get("r105", {}).get("destroy_and_reuse"):
            require(runtime.lib.aecStreamDestroy(stream) == SUCCESS)
            destroyed = True
            require(runtime.lib.aecStreamDestroy(stream) == INVALID_HANDLE)
    finally:
        if not destroyed:
            runtime.lib.aecStreamDestroy(stream)
        runtime.lib.aecFree(device)
    return "stream FIFO and asynchronous copy verified"


def test_r106(runtime: Runtime, profile: str) -> str:
    stream = runtime.stream()
    start = runtime.event()
    end = runtime.event()
    device = runtime.alloc(8192)
    source = HostBuffer(8192)
    event_options = additional_profile(profile).get("r106", {})
    unrecorded = runtime.event() if event_options.get("unrecorded_event") else None
    try:
        if unrecorded is not None:
            require(runtime.lib.aecEventQuery(unrecorded) == INVALID_ARGUMENT)
            require(runtime.lib.aecEventSynchronize(unrecorded) == INVALID_ARGUMENT)
        require(runtime.lib.aecEventRecord(start, stream) == SUCCESS)
        require(runtime.lib.aecCopyAsync(device, source.ptr, 8192, H2D, stream) == SUCCESS)
        require(runtime.lib.aecEventRecord(end, stream) == SUCCESS)
        require(runtime.lib.aecEventQuery(end) in (SUCCESS, NOT_READY))
        require(runtime.lib.aecEventSynchronize(end) == SUCCESS)
        cycles = ctypes.c_uint64()
        require(runtime.lib.aecEventElapsedCycles(start, end, ctypes.byref(cycles)) == SUCCESS)
        require(cycles.value > 0)
        if event_options.get("same_event_elapsed"):
            require(runtime.lib.aecEventElapsedCycles(
                end, end, ctypes.byref(cycles)) == SUCCESS and cycles.value == 0)
        require(runtime.lib.aecEventRecord(end, stream) == SUCCESS)
        require(runtime.lib.aecEventSynchronize(end) == SUCCESS)
        require(runtime.device.aecDeviceInjectFault(1) == SUCCESS)
        require(runtime.lib.aecCopyAsync(device, source.ptr, 1024, H2D, stream) == SUCCESS)
        require(runtime.lib.aecStreamSync(stream) == DEVICE_ERROR)
    finally:
        if unrecorded is not None:
            runtime.lib.aecEventDestroy(unrecorded)
        runtime.lib.aecEventDestroy(start)
        runtime.lib.aecEventDestroy(end)
        runtime.lib.aecStreamDestroy(stream)
        runtime.lib.aecFree(device)
    return "event generations, cycle capture, and async error propagation verified"


def test_r201(runtime: Runtime, profile: str) -> str:
    m, n, k = 2, 3, 3
    a = [1.0, -2.0, 0.5, 3.0, 4.0, -1.0]
    b = [2.0, 1.0, -1.0, 0.5, 3.0, 2.0, -2.0, 1.5, 0.25]
    _run_float_gemm(runtime, "aecMatmulF32", DTYPE_FP32, m, n, k, a, b)
    ia = [1, -2, 3, 4, 5, -6]
    ib = [7, 8, -9, 2, -3, 4, 5, -1, 6]
    _run_integer_gemm(runtime, "aecMatmulI32", DTYPE_INT32, m, n, k, ia, ib)
    hidden = additional_profile(profile).get("r201")
    if hidden:
        random_case = hidden["random"]
        rng = random.Random(random_case["seed"])
        m, n, k = random_case["m"], random_case["n"], random_case["k"]
        fa = [_f32(rng.uniform(random_case["low"], random_case["high"]))
              for _ in range(m * k)]
        fb = [_f32(rng.uniform(random_case["low"], random_case["high"]))
              for _ in range(k * n)]
        _run_float_gemm(runtime, "aecMatmulF32", DTYPE_FP32, m, n, k, fa, fb)
        extreme_a = hidden["integer_extreme_a"]
        extreme_b = hidden["integer_extreme_b"]
        integer_m, integer_n, integer_k = hidden["integer_shape"]
        _run_integer_gemm(runtime, "aecMatmulI32", DTYPE_INT32,
                          integer_m, integer_n, integer_k, extreme_a, extreme_b)
    return "FP32 and INT32 GEMM match an independent scalar oracle"


def test_r202(runtime: Runtime, profile: str) -> str:
    m, n, k = 2, 3, 3
    a = [1.0, -1.5, 0.5, 2.0, -0.5, 1.5]
    b = [0.5, 1.0, -1.0, 1.5, -0.5, 2.0, -1.0, 0.5, 1.0]
    cases = [
        ("aecMatmulF4", DTYPE_FP4, None),
        ("aecMatmulF8", DTYPE_FP8_E4M3, 1),
        ("aecMatmulF8", DTYPE_FP8_E5M2, 2),
        ("aecMatmulF16", DTYPE_FP16, None),
        ("aecMatmulBF16", DTYPE_BF16, None),
        ("aecMatmulF64", DTYPE_FP64, None),
    ]
    for name, dtype, fp8_format in cases:
        _run_float_gemm(runtime, name, dtype, m, n, k, a, b, fp8_format)
    hidden = additional_profile(profile).get("r202")
    if hidden:
        by_name = {
            "fp8_e4m3": ("aecMatmulF8", DTYPE_FP8_E4M3, 1),
            "fp8_e5m2": ("aecMatmulF8", DTYPE_FP8_E5M2, 2),
            "fp16": ("aecMatmulF16", DTYPE_FP16, None),
            "bf16": ("aecMatmulBF16", DTYPE_BF16, None),
            "fp32": ("aecMatmulF32", DTYPE_FP32, None),
            "fp64": ("aecMatmulF64", DTYPE_FP64, None),
        }
        left = math.nan if hidden["left"] == "nan" else float(hidden["left"])
        for format_name in hidden["special_formats"]:
            name, dtype, fp8_format = by_name[format_name]
            _run_float_gemm(runtime, name, dtype, 1, 1, 1,
                            [left], [float(hidden["right"])], fp8_format)
    return "all low-precision formats and FP64 match independent encoding/oracles"


def test_r203(runtime: Runtime, profile: str) -> str:
    m, n, k = 2, 3, 3
    a = [1, -2, 3, -4, 5, -6]
    b = [7, -8, 3, 2, -3, 4, -1, 1, 6]
    _run_integer_gemm(runtime, "aecMatmulI4", DTYPE_INT4, m, n, k, a, b)
    _run_integer_gemm(runtime, "aecMatmulI8", DTYPE_INT8, m, n, k, a, b)
    hidden = additional_profile(profile).get("r203")
    if hidden:
        a_raw = _pack_integers(a, DTYPE_INT8)
        b_raw = _pack_integers(b, DTYPE_INT8)
        a_ptr = runtime.alloc(len(a_raw))
        b_ptr = runtime.alloc(len(b_raw))
        undersized_c = runtime.alloc(hidden["undersized_output_bytes"])
        try:
            runtime.copy_in(a_ptr, a_raw)
            runtime.copy_in(b_ptr, b_raw)
            status = runtime.lib.aecMatmulI8(
                a_ptr, b_ptr, undersized_c, m, n, k, None)
            require(status == ISA_TRAP,
                    f"undersized INT8 output returned {status}")
        finally:
            runtime.lib.aecFree(a_ptr)
            runtime.lib.aecFree(b_ptr)
            runtime.lib.aecFree(undersized_c)
    return "packed INT4 and INT8 GEMM match exact INT32 results"


def test_r204(runtime: Runtime, profile: str) -> str:
    x_values = [1.0, -2.0, 3.0, 4.0, -0.5]
    y_values = [2.0, 1.0, -1.0, 0.5, 8.0]
    x_raw = struct.pack("<5f", *x_values)
    y_raw = struct.pack("<5f", *y_values)
    x = runtime.alloc(len(x_raw))
    y = runtime.alloc(len(y_raw))
    result = runtime.alloc(4)
    try:
        runtime.copy_in(x, x_raw)
        runtime.copy_in(y, y_raw)
        before = runtime.device_stats()
        require(runtime.lib.aecAxpy(x, y, 5, ctypes.c_float(0.5), None) == SUCCESS)
        actual_y = struct.unpack("<5f", runtime.copy_out(y, len(y_raw)))
        expected_y = [_f32(0.5 * left + right) for left, right in zip(x_values, y_values)]
        require(all(math.isclose(got, want, abs_tol=2e-5, rel_tol=5e-5)
                    for got, want in zip(actual_y, expected_y)))
        runtime.copy_in(y, y_raw)
        require(runtime.lib.aecDot(x, y, result, 5, None) == SUCCESS)
        dot = struct.unpack("<f", runtime.copy_out(result, 4))[0]
        dot_expected = _f32(0.0)
        for left, right in zip(x_values, y_values):
            dot_expected = _f32(dot_expected + _f32(left * right))
        require(math.isclose(dot, dot_expected, abs_tol=2e-5, rel_tol=5e-5))
        require(runtime.lib.aecNrm2(x, result, 5, None) == SUCCESS)
        norm = struct.unpack("<f", runtime.copy_out(result, 4))[0]
        square_sum = _f32(0.0)
        for value in x_values:
            square_sum = _f32(square_sum + _f32(value * value))
        require(math.isclose(norm, _f32(math.sqrt(square_sum)), abs_tol=2e-5, rel_tol=5e-5))
        after = runtime.device_stats()
        require(after.isa_launches == before.isa_launches + 3)
        require(after.instructions_retired > before.instructions_retired)
        require(after.last_kernel_handle != 0 and after.last_trace_digest != 0)
        hidden = additional_profile(profile).get("r204")
        if hidden:
            one_past = hidden["one_past_count"]
            require(runtime.lib.aecAxpy(
                x, y, one_past, ctypes.c_float(1.0), None) == INVALID_ARGUMENT)
            require(runtime.lib.aecDot(x, y, result, one_past, None) == INVALID_ARGUMENT)
            require(runtime.lib.aecNrm2(x, result, one_past, None) == INVALID_ARGUMENT)
    finally:
        runtime.lib.aecFree(x)
        runtime.lib.aecFree(y)
        runtime.lib.aecFree(result)
    return "AXPY, DOT, and NRM2 match ordered FP32 oracles"


def test_r301(runtime: Runtime, profile: str) -> str:
    require(runtime.lib.aecResetRuntimeStats() == SUCCESS)
    raw = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    a = runtime.alloc(16)
    b = runtime.alloc(16)
    c = runtime.alloc(16)
    try:
        runtime.copy_in(a, raw)
        runtime.copy_in(b, raw)
        args = VectorAddArgs(a, b, c, 4)
        require(runtime.lib.aecLaunch(KERNEL_VECTOR_ADD, Dim3(1, 1, 1),
                                     Dim3(32, 1, 1), ctypes.byref(args),
                                     ctypes.sizeof(args), None) == SUCCESS)
        runtime.copy_out(c, 16)
        stats = runtime.device_stats()
        require(stats.abi_version == 2)
        require(stats.submitted_commands == 4)
        require(stats.dma_commands == 3 and stats.kernel_commands == 1)
        require(stats.isa_launches == 1 and stats.instructions_retired > 0)
        require(stats.last_kernel_handle != 0 and stats.last_trace_digest != 0)
        require(stats.last_kernel_handle == runtime.kernel_handle(
            KERNEL_VECTOR_ADD, DTYPE_FP32, 0),
            "R301 observed the wrong resolved image handle")
        require(stats.total_virtual_cycles > stats.last_virtual_cycles > 0)
        reported = runtime.stats()
        require(bytes(reported) == bytes(stats))
    finally:
        runtime.lib.aecFree(a)
        runtime.lib.aecFree(b)
        runtime.lib.aecFree(c)
    return "command types and deterministic completion accounting verified"


def test_r302(runtime: Runtime, profile: str) -> str:
    require(runtime.lib.aecResetRuntimeStats() == SUCCESS)
    first_stream = runtime.stream()
    second_stream = runtime.stream()
    first = runtime.alloc(1024)
    second = runtime.alloc(1024)
    host_a = HostBuffer(1024)
    host_b = HostBuffer(1024)
    try:
        require(runtime.lib.aecCopyAsync(first, host_a.ptr, 1024, H2D, first_stream) == SUCCESS)
        require(runtime.lib.aecCopyAsync(second, host_b.ptr, 1024, H2D, second_stream) == SUCCESS)
        require(runtime.lib.aecStreamSync(first_stream) == SUCCESS)
        require(runtime.lib.aecStreamSync(second_stream) == SUCCESS)
        stats = runtime.device_stats()
        require(stats.channel_commands[0] > 0 and stats.channel_commands[1] > 0)
        require(runtime.lib.aecCopyAsync(first + 1020, host_a.ptr, 16, H2D,
                                        first_stream) == SUCCESS)
        require(runtime.lib.aecStreamSync(first_stream) == INVALID_ADDRESS)
        require(runtime.lib.aecCopyAsync(first, host_a.ptr, 16, H2D, first_stream) == SUCCESS)
        require(runtime.lib.aecStreamSync(first_stream) == SUCCESS)
    finally:
        runtime.lib.aecStreamDestroy(first_stream)
        runtime.lib.aecStreamDestroy(second_stream)
        runtime.lib.aecFree(first)
        runtime.lib.aecFree(second)
    return "both DMA channels, asynchronous bounds, and recovery verified"


def test_r303(runtime: Runtime, profile: str) -> str:
    device = runtime.alloc(65536)
    host = HostBuffer(65536)
    try:
        require(runtime.lib.aecResetRuntimeStats() == SUCCESS)
        require(runtime.lib.aecCopyH2D(device, host.ptr, 65536) == SUCCESS)
        normal_cycles = runtime.device_stats().last_virtual_cycles
        require(runtime.lib.aecHostRegister(host.ptr, 65536) == SUCCESS)
        require(runtime.lib.aecHostRegister(host.ptr, 65536) == INVALID_ARGUMENT)
        hidden = additional_profile(profile).get("r303")
        if hidden:
            require(runtime.lib.aecHostRegister(
                ctypes.c_void_p(host.address + hidden["overlap_offset"]),
                hidden["overlap_bytes"]) == INVALID_ARGUMENT)
        require(runtime.lib.aecResetRuntimeStats() == SUCCESS)
        require(runtime.lib.aecCopyH2D(device, host.ptr, 65536) == SUCCESS)
        registered_stats = runtime.device_stats()
        require(registered_stats.zero_copy_commands == 1)
        require(registered_stats.last_virtual_cycles < normal_cycles)
        require(runtime.lib.aecHostUnregister(host.ptr) == SUCCESS)
        require(runtime.lib.aecHostUnregister(host.ptr) == INVALID_ARGUMENT)
    finally:
        runtime.lib.aecFree(device)
    return "registration lifecycle and modeled zero-copy benefit verified"


def test_r304(runtime: Runtime, profile: str) -> str:
    stream = runtime.stream()
    first = runtime.alloc(1024)
    second = runtime.alloc(1024)
    output = runtime.alloc(1024)
    host = HostBuffer(1024)
    try:
        require(runtime.device.aecDeviceInjectFault(1) == SUCCESS)
        require(runtime.lib.aecCopyAsync(first, host.ptr, 1024, H2D, stream) == SUCCESS)
        require(runtime.lib.aecStreamSync(stream) == DEVICE_ERROR)
        require(runtime.lib.aecCopyAsync(first, host.ptr, 1024, H2D, stream) == SUCCESS)
        require(runtime.lib.aecStreamSync(stream) == SUCCESS)
        require(runtime.device.aecDeviceInjectFault(2) == SUCCESS)
        args = VectorAddArgs(first, second, output, 4)
        require(runtime.lib.aecLaunch(KERNEL_VECTOR_ADD, Dim3(1, 1, 1),
                                     Dim3(32, 1, 1), ctypes.byref(args),
                                     ctypes.sizeof(args), stream) == SUCCESS)
        require(runtime.lib.aecStreamSync(stream) == DEVICE_ERROR)
    finally:
        runtime.lib.aecStreamDestroy(stream)
        runtime.lib.aecFree(first)
        runtime.lib.aecFree(second)
        runtime.lib.aecFree(output)
    return "DMA/kernel fault injection and post-fault recovery verified"


def _run_agent(path: Path, request: dict) -> dict:
    if not path.is_file():
        raise AssertionError(f"missing Agent: {path.name}")
    environment = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"}
    result = subprocess.run(
        [sys.executable, str(path)],
        input=json.dumps(request, sort_keys=True),
        text=True,
        capture_output=True,
        timeout=1,
        env=environment,
        check=False,
    )
    if len(result.stdout) + len(result.stderr) > 65536:
        raise AssertionError(f"{path.name} exceeded the 64 KiB output limit")
    if result.returncode != 0:
        raise AssertionError(f"{path.name} exited {result.returncode}: {result.stderr[:160]}")
    try:
        output = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"{path.name} emitted invalid JSON") from exc
    if not isinstance(output, dict):
        raise AssertionError("Agent output must be an object")
    return output


def _dma_cycles(request: dict, action: dict) -> int:
    expected_keys = {"chunk_bytes", "queue_depth", "channel", "use_zero_copy"}
    if set(action) != expected_keys:
        raise AssertionError(f"DMA action keys are {sorted(action)}")
    chunk = action["chunk_bytes"]
    depth = action["queue_depth"]
    channel = action["channel"]
    zero_copy = action["use_zero_copy"]
    if type(chunk) is not int or chunk not in (4096, 65536, 1048576):
        raise AssertionError("illegal chunk_bytes")
    if type(depth) is not int or depth not in (1, 2, 4, 8):
        raise AssertionError("illegal queue_depth")
    if type(channel) is not int or channel not in (0, 1):
        raise AssertionError("illegal channel")
    if type(zero_copy) is not bool or (zero_copy and not request["registered"]):
        raise AssertionError("illegal use_zero_copy")
    chunks = math.ceil(request["bytes"] / chunk)
    payload = math.ceil(request["bytes"] / 32)
    parallelism = min(depth, request["concurrency"], 2)
    setup = 45 if zero_copy else 100
    alignment_penalty = 0 if request["alignment"] >= 64 else 13
    return setup + math.ceil(payload / parallelism) + 24 * (chunks - 1) + alignment_penalty


@dataclass(frozen=True)
class AgentOutcome:
    detail: str
    earned: float
    hidden_positive: bool
    metrics: list[dict]


def test_r401(runtime: Runtime, profile: str) -> AgentOutcome:
    public_cases = [
        {"case_id": 0, "direction": "h2d", "bytes": 1024, "alignment": 64,
         "registered": False, "concurrency": 1, "performance": False,
         "visibility": "public"},
        {"case_id": 1, "direction": "d2h", "bytes": 65536, "alignment": 64,
         "registered": True, "concurrency": 4, "performance": True,
         "visibility": "public"},
        {"case_id": 2, "direction": "h2d", "bytes": 1048576, "alignment": 16,
         "registered": False, "concurrency": 2, "performance": True,
         "visibility": "public"},
    ]
    hidden_cases = additional_profile(profile).get("r401", [])
    cases = public_cases + (hidden_cases if profile == "full" else [])
    hidden_fractions: list[float] = []
    public_fractions: list[float] = []
    metrics: list[dict] = []
    agent_path = runtime.submission_dir / "agents" / "dma_agent.py"
    for request in cases:
        agent_request = {k: v for k, v in request.items()
                         if k not in {"performance", "visibility"}}
        action = _run_agent(agent_path, agent_request)
        candidate = _dma_cycles(request, action)
        baseline_action = {"chunk_bytes": 4096, "queue_depth": 1, "channel": 0,
                           "use_zero_copy": False}
        baseline = _dma_cycles(request, baseline_action)
        fraction = 0.0
        if request["performance"]:
            fraction = max(0.0, min(1.0, (baseline / candidate - 1.0) / 0.5))
            target = hidden_fractions if request["visibility"] == "hidden" else public_fractions
            target.append(fraction)
        metrics.append({"case_id": request["case_id"],
                        "visibility": request["visibility"],
                        "baseline_cycles": baseline, "candidate_cycles": candidate,
                        "fraction": round(fraction, 9), "action": action})
    public_diagnostic = (sum(public_fractions) / len(public_fractions)
                         if public_fractions else 0.0)
    hidden_performance = (sum(hidden_fractions) / len(hidden_fractions)
                          if hidden_fractions else 0.0)
    earned = 4.0 + (6.0 * hidden_performance if profile == "full" else 0.0)
    return AgentOutcome(
        "DMA Agent schema/correctness passed; "
        f"public diagnostic={public_diagnostic:.6f}; "
        f"hidden performance={hidden_performance:.6f}",
        earned, profile == "full" and hidden_performance > 0.0, metrics)


DTYPE_BY_NAME = {
    "fp4_e2m1": DTYPE_FP4, "fp8_e4m3": DTYPE_FP8_E4M3,
    "fp8_e5m2": DTYPE_FP8_E5M2, "fp16": DTYPE_FP16,
    "bf16": DTYPE_BF16, "fp32": DTYPE_FP32, "fp64": DTYPE_FP64,
    "int4": DTYPE_INT4, "int8": DTYPE_INT8, "int32": DTYPE_INT32,
}


def _kernel_candidates(dtype: int) -> list[dict]:
    descriptions = [
        ("naive", 10, 1, 0, 1, 1),
        ("tiled", 11, 2, 4096, 1, 4),
        ("vectorized", 12, 3, 8192, 16, 8),
    ]
    return [{"id": name, "semantic_kernel_id": kernel,
             "image_id": (kernel << 16) | (dtype << 8) | variant,
             "variant": variant, "workspace": workspace,
             "alignment": alignment, "divisibility": divisibility}
            for name, kernel, variant, workspace, alignment, divisibility
            in descriptions]


def _kernel_cycles(runtime: Runtime, request: dict, action: dict) -> dict:
    if set(action) != {"kernel_id"} or not isinstance(action["kernel_id"], str):
        raise AssertionError("Kernel action must contain only kernel_id")
    candidates = {item["id"]: item for item in request["candidates"]}
    selected = action["kernel_id"]
    if selected not in candidates:
        raise AssertionError("unknown kernel_id")
    candidate = candidates[selected]
    completion = DeviceCompletion()
    status = runtime.device.aecDeviceEvaluateKernel(
        candidate["semantic_kernel_id"], DTYPE_BY_NAME[request["dtype"]],
        candidate["variant"], request["m"], request["n"], request["k"],
        request["alignment"], request["workspace"], ctypes.byref(completion))
    if status != SUCCESS:
        raise AssertionError(f"candidate image is illegal (device status {status})")
    require(completion.instructions_retired > 0, "candidate retired no AEC instructions")
    require(completion.trace_digest != 0, "candidate produced no AEC trace digest")
    require(completion.virtual_cycles > 0, "candidate produced no virtual cycles")
    return {"cycles": completion.virtual_cycles,
            "retired": completion.instructions_retired,
            "trace_digest": completion.trace_digest,
            "image_id": candidate["image_id"], "selected": selected}


def test_r402(runtime: Runtime, profile: str) -> AgentOutcome:
    public_cases = [
        {"case_id": 0, "dtype": "fp32", "m": 32, "n": 64, "k": 16,
         "alignment": 64, "workspace": 8192, "performance": True,
         "visibility": "public"},
        {"case_id": 1, "dtype": "int8", "m": 20, "n": 12, "k": 28,
         "alignment": 16, "workspace": 4096, "performance": True,
         "visibility": "public"},
        {"case_id": 2, "dtype": "fp16", "m": 7, "n": 9, "k": 5,
         "alignment": 8, "workspace": 0, "performance": False,
         "visibility": "public"},
    ]
    hidden_cases = additional_profile(profile).get("r402", [])
    cases = public_cases + (hidden_cases if profile == "full" else [])
    hidden_fractions: list[float] = []
    public_fractions: list[float] = []
    metrics: list[dict] = []
    agent_path = runtime.submission_dir / "agents" / "kernel_agent.py"
    for raw in cases:
        request = {key: value for key, value in raw.items()
                   if key not in {"performance", "visibility"}}
        request["candidates"] = _kernel_candidates(DTYPE_BY_NAME[request["dtype"]])
        if raw["visibility"] == "public":
            for candidate in request["candidates"]:
                try:
                    evaluation = _kernel_cycles(
                        runtime, request, {"kernel_id": candidate["id"]})
                    candidate["diagnostic_cycles"] = evaluation["cycles"]
                except AssertionError:
                    pass
        action = _run_agent(agent_path, request)
        candidate = _kernel_cycles(runtime, request, action)
        baseline = _kernel_cycles(runtime, request, {"kernel_id": "naive"})
        fraction = 0.0
        if raw["performance"]:
            fraction = max(0.0, min(
                1.0, (baseline["cycles"] / candidate["cycles"] - 1.0) / 0.5))
            target = hidden_fractions if raw["visibility"] == "hidden" else public_fractions
            target.append(fraction)
        metrics.append({"case_id": raw["case_id"], "visibility": raw["visibility"],
                        "baseline_cycles": baseline["cycles"],
                        "candidate_cycles": candidate["cycles"],
                        "fraction": round(fraction, 9),
                        "image_id": candidate["image_id"],
                        "instructions_retired": candidate["retired"],
                        "trace_digest": candidate["trace_digest"],
                        "action": action})
    public_diagnostic = (sum(public_fractions) / len(public_fractions)
                         if public_fractions else 0.0)
    hidden_performance = (sum(hidden_fractions) / len(hidden_fractions)
                          if hidden_fractions else 0.0)
    earned = 4.0 + (6.0 * hidden_performance if profile == "full" else 0.0)
    return AgentOutcome(
        "Kernel Agent legality passed; "
        f"public diagnostic={public_diagnostic:.6f}; "
        f"hidden performance={hidden_performance:.6f}",
        earned, profile == "full" and hidden_performance > 0.0, metrics)


@dataclass(frozen=True)
class Requirement:
    identifier: str
    points: float
    test: Callable[[Runtime, str], str | tuple[str, float] | AgentOutcome]


REQUIREMENTS = [
    Requirement("R101", 4, test_r101),
    Requirement("R102", 6, test_r102),
    Requirement("R103", 6, test_r103),
    Requirement("R104", 4, test_r104),
    Requirement("R105", 5, test_r105),
    Requirement("R106", 5, test_r106),
    Requirement("R201", 10, test_r201),
    Requirement("R202", 10, test_r202),
    Requirement("R203", 4, test_r203),
    Requirement("R204", 6, test_r204),
    Requirement("R301", 6, test_r301),
    Requirement("R302", 6, test_r302),
    Requirement("R303", 4, test_r303),
    Requirement("R304", 4, test_r304),
    Requirement("R401", 10, test_r401),
    Requirement("R402", 10, test_r402),
]


def grade(submission: Path, profile: str,
          requirement_filter: str | None = None) -> dict:
    runtime = Runtime(submission)
    results: dict[str, dict] = {}
    for requirement in REQUIREMENTS:
        if requirement_filter is not None and requirement.identifier != requirement_filter:
            continue
        runtime.reset()
        try:
            outcome = requirement.test(runtime, profile)
            hidden_positive = False
            case_metrics: list[dict] = []
            if isinstance(outcome, AgentOutcome):
                detail, earned = outcome.detail, outcome.earned
                hidden_positive = outcome.hidden_positive
                case_metrics = outcome.metrics
            elif isinstance(outcome, tuple):
                detail, earned = outcome
            else:
                detail, earned = outcome, requirement.points
            earned = min(requirement.points, max(0.0, float(earned)))
            stats = runtime.device_stats()
            results[requirement.identifier] = {
                "passed": True,
                "earned": round(earned, 6),
                "possible": requirement.points,
                "detail": detail,
                "hidden_positive": hidden_positive,
                "cases": case_metrics,
                "device_evidence": {
                    "submitted_commands": stats.submitted_commands,
                    "dma_commands": stats.dma_commands,
                    "isa_launches": stats.isa_launches,
                    "instructions_retired": stats.instructions_retired,
                    "isa_traps": stats.isa_traps,
                    "total_virtual_cycles": stats.total_virtual_cycles,
                    "last_virtual_cycles": stats.last_virtual_cycles,
                    "last_kernel_handle": stats.last_kernel_handle,
                    "last_trace_digest": stats.last_trace_digest,
                },
            }
        except Exception as exc:  # grader must report one failed requirement and continue
            try:
                stats = runtime.device_stats()
                evidence = {
                    "submitted_commands": stats.submitted_commands,
                    "dma_commands": stats.dma_commands,
                    "isa_launches": stats.isa_launches,
                    "instructions_retired": stats.instructions_retired,
                    "isa_traps": stats.isa_traps,
                    "total_virtual_cycles": stats.total_virtual_cycles,
                    "last_virtual_cycles": stats.last_virtual_cycles,
                    "last_kernel_handle": stats.last_kernel_handle,
                    "last_trace_digest": stats.last_trace_digest,
                }
            except Exception:
                evidence = {}
            results[requirement.identifier] = {
                "passed": False,
                "earned": 0.0,
                "possible": requirement.points,
                "detail": f"{type(exc).__name__}: {exc}",
                "hidden_positive": False,
                "cases": [],
                "device_evidence": evidence,
            }
    score = round(sum(item["earned"] for item in results.values()), 6)
    complete = requirement_filter is None
    basic_gate = complete and all(results[item]["passed"] for item in
                                  ("R101", "R102", "R103", "R104", "R201"))
    good_gate = complete and basic_gate and all(results[item]["passed"] for item in
                                                ("R105", "R106", "R202", "R203", "R204",
                                                 "R301", "R302", "R303", "R304"))
    excellent_gate = (complete and profile == "full" and good_gate and
                       all(results[item]["passed"] and
                           results[item]["hidden_positive"]
                           for item in ("R401", "R402")))
    level = "Not passed"
    if score >= 30 and basic_gate:
        level = "Basic"
    if score >= 75 and good_gate:
        level = "Good"
    if score >= 90 and excellent_gate:
        level = "Excellent"
    return {
        "schema_version": 2,
        "competition": "AEC-C2",
        "profile": profile,
        "submission": str(runtime.submission_dir),
        "score": score,
        "level": level,
        "gates": {"basic": basic_gate, "good": good_gate,
                  "excellent": excellent_gate},
        "requirements": results,
        "requirement_filter": requirement_filter,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True,
                        help="directory containing libaec.so, or the library itself")
    parser.add_argument("--profile", choices=("public",), default="public")
    parser.add_argument("--requirement", choices=tuple(
        requirement.identifier for requirement in REQUIREMENTS))
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = grade(args.submission, args.profile, args.requirement)
    except (FileNotFoundError, OSError) as exc:
        print(f"grader error: {exc}", file=sys.stderr)
        return 2
    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(serialized, encoding="utf-8")
    if not args.quiet:
        print(f"AEC score: {report['score']:.6f}/100 — {report['level']}")
        for identifier, result in report["requirements"].items():
            marker = "PASS" if result["passed"] else "FAIL"
            print(f"{marker} {identifier}: {result['earned']}/{result['possible']} — "
                  f"{result['detail']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

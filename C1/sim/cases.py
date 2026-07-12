"""cases.py — parametric launch configs + numpy references for the 5 public PTX.

Each builder returns a case dict {ptx, grid, block, param, gmem, out, ref, note}.
Every builder takes size parameters (defaulting to the small, blockDim-dividing
values used by dogfood.py) so bench.py can sweep sizes — including N values that
are NOT a multiple of blockDim, which exercises the divergent bounds-guard path
(see C1_实现流程分析.md §1.3).

The param block is packed to match aec-cc's natural, tight, declaration-order
layout. Pointers are GMEM byte offsets into a single flat GMEM image.
"""
import struct
import numpy as np

F32 = np.dtype("<f4")
F16 = np.dtype("<f2")


def _f32(x):
    return np.asarray(x, dtype=np.float32)


def _place(gmem, off, arr_bytes):
    gmem[off:off + len(arr_bytes)] = arr_bytes


def vector_add(n=256, block=64, seed=1):
    rng = np.random.default_rng(seed)
    a = _f32(rng.standard_normal(n)); b = _f32(rng.standard_normal(n))
    a_off, b_off, c_off = 0, n * 4, 2 * n * 4
    gmem = bytearray(3 * n * 4)
    _place(gmem, a_off, a.tobytes()); _place(gmem, b_off, b.tobytes())
    grid = (n + block - 1) // block
    return dict(ptx="PTX-01_vector_add.ptx", grid=(grid, 1, 1),
                block=(block, 1, 1), param=struct.pack("<QQQI", a_off, b_off, c_off, n),
                gmem=gmem, out=(c_off, n, F32), ref=a + b)


def invariant_poly(n=256, block=256, loop_count=32, seed=2):
    rng = np.random.default_rng(seed)
    x = _f32(rng.standard_normal(n)); a = np.float32(1.5); b = np.float32(-0.25)
    x_off, y_off = 0, n * 4
    gmem = bytearray(2 * n * 4)
    _place(gmem, x_off, x.tobytes())
    acc = np.zeros(n, np.float32)
    for _ in range(loop_count):               # loop count baked into the PTX.
        f5 = np.float32(a + b)
        f8 = (x * f5).astype(np.float32) + f5
        acc = (acc + f8).astype(np.float32)
    grid = (n + block - 1) // block
    return dict(ptx="PTX-02_invariant_poly.ptx", grid=(grid, 1, 1),
                block=(block, 1, 1),
                param=struct.pack("<QQIff", x_off, y_off, n, float(a), float(b)),
                gmem=gmem, out=(y_off, n, F32), ref=acc)


def repeated_reuse(grid=4, block=32, loop_count=16, seed=3):
    # PTX-03 quirks: the K-loop count is HARDCODED to 16 (not a param), and each
    # thread reads x[col] with col = ctaid.x*32 + (tid.x & 31) while writing
    # y[idx] with idx = ctaid.x*blockDim + tid.x. col == idx only when block==32.
    K = 16                                     # w array size; loop reads w[0:loop_count].
    n = grid * block                          # threads / y elements.
    xcols = grid * 32                          # distinct x columns read.
    rng = np.random.default_rng(seed)
    x = _f32(rng.standard_normal(xcols)); w = _f32(rng.standard_normal(K))
    x_off, w_off, y_off = 0, xcols * 4, xcols * 4 + K * 4
    gmem = bytearray(xcols * 4 + K * 4 + n * 4)
    _place(gmem, x_off, x.tobytes()); _place(gmem, w_off, w.tobytes())
    idx = np.arange(n)
    col = (idx // block) * 32 + (idx % block % 32)
    xc = x[col]
    acc = np.zeros(n, np.float32)
    for k in range(loop_count):
        acc = np.float32(xc * np.float32(w[k]) + acc)
    return dict(ptx="PTX-03_repeated_reuse.ptx", grid=(grid, 1, 1),
                block=(block, 1, 1),
                param=struct.pack("<QQQI", x_off, w_off, y_off, n),
                gmem=gmem, out=(y_off, n, F32), ref=acc)


def reg_schedule(n=128, block=64, seed=4):
    rng = np.random.default_rng(seed)
    a, b, c, d = (_f32(rng.standard_normal(n)) for _ in range(4))
    offs = [i * n * 4 for i in range(5)]
    gmem = bytearray(5 * n * 4)
    for arr, off in zip((a, b, c, d), offs):
        _place(gmem, off, arr.tobytes())
    f5 = a + b; f6 = c + d; f7 = a - b; f8 = c - d
    ref = ((f5 * f6) + (f7 * f8)) + ((a * c) + (b * d))
    grid = (n + block - 1) // block
    return dict(ptx="PTX-04_reg_schedule.ptx", grid=(grid, 1, 1),
                block=(block, 1, 1),
                param=struct.pack("<QQQQQI", offs[0], offs[1], offs[2], offs[3], offs[4], n),
                gmem=gmem, out=(offs[4], n, F32), ref=ref.astype(np.float32))


def _to_dtype(mat, dtype):
    """Return (bytes, values-as-f32) for storing `mat` as dtype in GMEM."""
    f = mat.astype(np.float32)
    if dtype == "f16":
        h = f.astype(np.float16)
        return h.tobytes(), h.astype(np.float32)
    if dtype == "bf16":                        # bf16 bits = top 16 of f32.
        bits = (f.view(np.uint32) >> 16).astype(np.uint16)
        return bits.tobytes(), (bits.astype(np.uint32) << 16).view(np.float32)
    if dtype == "f32":
        return f.tobytes(), f
    raise ValueError(dtype)


def gemm_f16(M=16, N=16, K=16, dtype="f16", seed=5):
    esize = 4 if dtype == "f32" else 2
    rng = np.random.default_rng(seed)
    A_bytes, A_v = _to_dtype(rng.standard_normal((M, K)), dtype)
    B_bytes, B_v = _to_dtype(rng.standard_normal((K, N)), dtype)
    a_off = 0; b_off = M * K * esize; c_off = b_off + K * N * esize
    gmem = bytearray(c_off + M * N * 4)
    _place(gmem, a_off, A_bytes); _place(gmem, b_off, B_bytes)
    ref = (A_v @ B_v).reshape(-1)              # C is always f32.
    grid = ((N + 15) // 16, (M + 15) // 16, 1)
    return dict(ptx="PTX-05_gemm_f16.ptx", grid=grid, block=(256, 1, 1),
                param=struct.pack("<QQQIII", a_off, b_off, c_off, M, N, K),
                gmem=gmem, out=(c_off, M * N, F32), ref=ref, dtype=dtype)


ALL = {
    "vadd": vector_add, "poly": invariant_poly, "reuse": repeated_reuse,
    "reg": reg_schedule, "gemm": gemm_f16,
}

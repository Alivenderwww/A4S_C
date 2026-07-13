#!/usr/bin/env python3
"""服务器环境探测脚本 - 了解评测服务器的软硬件环境。
运行: python3 server_probe.py
"""
import sys, subprocess, shutil

print("=" * 60)
print("1. Python 环境")
print("=" * 60)
print("Python:", sys.version)
print("可执行:", sys.executable)
print()

print("=" * 60)
print("2. GPU 与 CUDA 驱动")
print("=" * 60)
try:
    r = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
    # 只打印关键的几行
    for line in r.stdout.split("\n"):
        if any(k in line for k in ["Driver Version", "CUDA Version", "MiB", "GPU-"]):
            print(line)
except Exception as e:
    print("nvidia-smi 不可用:", e)
print()

print("=" * 60)
print("3. 关键 Python 库是否已安装 + 版本")
print("=" * 60)
for lib in ["cupy", "numpy", "onnx", "onnxruntime", "torch", "scipy"]:
    try:
        mod = __import__(lib)
        ver = getattr(mod, "__version__", "未知")
        print(f"  [已装] {lib:14} {ver}")
    except ImportError:
        print(f"  [缺失] {lib}")
    except Exception as e:
        print(f"  [异常] {lib}: {e}")
print()

print("=" * 60)
print("4. CuPy GPU 实际可用性(关键!)")
print("=" * 60)
try:
    import cupy as cp
    print("cupy 导入成功")
    props = cp.cuda.runtime.getDeviceProperties(0)
    print("  GPU名称:", props.get("name", b"?").decode() if isinstance(props.get("name"), bytes) else props.get("name"))
    free, total = cp.cuda.runtime.memGetInfo()
    print(f"  显存: {free//1024//1024} MB 可用 / {total//1024//1024} MB 总计")
    rt_ver = cp.cuda.runtime.runtimeGetVersion()
    print(f"  CUDA runtime 版本: {rt_ver//1000}.{(rt_ver%1000)//10}")
    # 实际跑一次 GPU 计算
    a = cp.arange(6, dtype=cp.float32).reshape(2, 3)
    result = (a @ a.T).sum().item()
    print(f"  GPU 计算测试 (matmul): 成功, 结果={result}")
    # 测试 erf (transformer GELU 用)
    try:
        from cupyx.scipy.special import erf
        print(f"  cupyx.scipy.special.erf 可用: erf(1.0)={float(erf(cp.array([1.0],dtype=cp.float32))[0]):.6f}")
    except Exception as e:
        print(f"  cupyx erf 不可用: {e}")
    print("  >>> CuPy 完全可用 <<<")
except Exception as e:
    print(f"  CuPy 不可用: {type(e).__name__}: {e}")
print()

print("=" * 60)
print("5. onnxruntime 的 GPU provider")
print("=" * 60)
try:
    import onnxruntime as ort
    print("  可用 providers:", ort.get_available_providers())
except Exception as e:
    print("  onnxruntime 检测失败:", e)
print()

print("=" * 60)
print("6. 当前目录与磁盘空间")
print("=" * 60)
import os
print("  当前目录:", os.getcwd())
try:
    r = subprocess.run(["df", "-h", "."], capture_output=True, text=True, timeout=5)
    print(r.stdout.strip())
except Exception:
    pass
print()

print("=" * 60)
print("探测完成。请把以上全部输出复制给我。")
print("=" * 60)

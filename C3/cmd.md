# C3 赛题操作命令手册

> **运行环境**：服务器 `mig02@39.107.68.147 -p 1102`，工作目录 `~/A4S/c3`
>
> 所有命令均在 `~/A4S/c3` 下执行。Python 3.12 + CuPy 14.1.1 + CUDA 12.8 + H200 GPU。

---

## 0. 环境准备（一次性）

```bash
cd ~/A4S/c3
pip install -r requirements.txt
```

依赖：`numpy`、`onnx`、`onnxruntime`（CPU 回退用）、`cupy`（C3.5 GPU 后端，spec 要求）。

---

## 1. C3.1 计算图解析与导出（10 分，自动检查）

### 报名提交的命令模板

```
python tools/export_dag.py --onnx {onnx} --output {output}
```

### 逐模型运行

```bash
cd ~/A4S/c3

# MLP
python tools/export_dag.py --onnx mlp_v1.onnx --output mlp_dag.json

# ResNet
python tools/export_dag.py --onnx resnet_v1.onnx --output resnet_dag.json

# Transformer
python tools/export_dag.py --onnx transformer_v1.onnx --output transformer_dag.json

# BigFormer (模型在 /workspace)
python tools/export_dag.py --onnx /workspace/C3/testcases/models/bigformer_v1.onnx --output bigformer_dag.json
```

### 检查退出码 + 结构

```bash
# 每条命令后检查退出码（必须 = 0）
echo $?

# 检查 JSON 结构
python -c "import json; d=json.load(open('mlp_dag.json')); print('format_version:', d['format_version'], '| nodes:', len(d['nodes']), '| edges:', len(d['edges']))"
```

---

## 2. C3.2 算子分解与内核选择（15 分，微基准）

### 本地自评分（D1–D5）

```bash
cd ~/A4S/c3

# 官方评测模型（mlp + resnet）
python benchmarks/c32_c33/bench_c32_c33.py --models mnist_mlp cifar_resnet18 --output-dir results

# 含 transformer（完整三模型）
python benchmarks/c32_c33/bench_c32_c33.py --models mnist_mlp cifar_resnet18 transformer --output-dir results
```

输出含每个模型的 `bench_<model>.json`（分解 + tuning + 中间张量明细）和 `scores.json`（最终分）。

---

## 3. C3.3 算子融合与图优化（15 分，微基准）

### 通过 bench 获取 F1–F4

```bash
cd ~/A4S/c3

# bench 已含 C3.3（与 C3.2 同一脚本）
python benchmarks/c32_c33/bench_c32_c33.py --models mnist_mlp cifar_resnet18 --output-dir results
```

### 独立严格评委测试（逐条核验 spec 触发条件）

```bash
cd ~/A4S/c3
python tests/selftest_c33.py
```

### 数值对齐自检（F4 硬指标）

```bash
cd ~/A4S/c3
python -c "
import sys; sys.path.insert(0,'.')
import numpy as np
from scheduler.graph import import_onnx_graph
from scheduler.graph_passes.pipeline import GraphPassPipeline
from runtime.mock_runtime import MockRuntime
for name in ['mlp_v1','resnet_v1','transformer_v1']:
    g = import_onnx_graph(name+'.onnx')
    pipe = GraphPassPipeline(enable_fusion=True); opt = pipe.run(g)
    rng = np.random.default_rng(0)
    if 'transformer' in name:
        x = rng.integers(0,13,size=(2,18),dtype=np.int64); feed={'input_ids':x}
    elif 'resnet' in name:
        x = rng.standard_normal((2,3,32,32)).astype(np.float32); feed={'input':x}
    else:
        x = rng.standard_normal((2,1,28,28)).astype(np.float32); feed={'input':x}
    o1 = MockRuntime(g).run(feed); o2 = MockRuntime(opt).run(feed)
    key = list(o1.keys())[0]
    md = float(np.max(np.abs(o1[key]-o2[key])))
    print(f'{name}: max_diff={md:.2e} {\"PASS\" if md<=1e-3 else \"FAIL\"}')
"
```

---

## 4. C3.4 内存规划与调度（10 分，Code Review）

### 输出 A–E 可追溯证据

```bash
cd ~/A4S/c3

# ResNet
python -c "
from scheduler.memory import build_execution_plan
from scheduler.graph import import_onnx_graph
import json
p = build_execution_plan(import_onnx_graph('resnet_v1.onnx'), batch=256)
print(json.dumps(p.summary['c3d_evidence'], indent=2))
"
```

### C3.4 门禁自测

```bash
cd ~/A4S/c3
python tests/selftest_c34.py
```

---

## 5. C3.5 典型模型部署（50 分，端到端测试）

### 报名提交的 worker 启动命令

```
python tools/infer_worker.py
```

> 不带任务参数。评测机启动后通过 stdin 下发 JSON 任务，worker 经 stdout 返回结果。
> 详见 `C35_WORKER_PROTOCOL.md`。

### 逐模型推理（一次性 CLI，兼容旧调用）

```bash
cd ~/A4S/c3

# MLP
python tools/infer.py --onnx mlp_v1.onnx --input testdata_mlp/input --output out_mlp --batch-size 256 --backend cupy

# ResNet
python tools/infer.py --onnx resnet_v1.onnx --input testdata_resnet/input --output out_resnet --batch-size 256 --backend cupy

# Transformer
python tools/infer.py --onnx transformer_v1.onnx --input testdata_tf/input --output out_transformer --batch-size 256 --backend cupy

# BigFormer (模型和测试数据在 /workspace)
python tools/infer.py --onnx /workspace/C3/testcases/models/bigformer_v1.onnx --input /workspace/C3/testcases/testdata/c35/bigformer_v1/input --output out_bigformer --batch-size 4 --backend cupy
```

### Worker 协议单模型测试

```bash
cd ~/A4S/c3

# MLP
echo '{"onnx":"mlp_v1.onnx","input":"testdata_mlp/input","output":"wout_mlp","batch_size":256}
{"cmd":"exit"}' | python tools/infer_worker.py

# ResNet
echo '{"onnx":"resnet_v1.onnx","input":"testdata_resnet/input","output":"wout_resnet","batch_size":256}
{"cmd":"exit"}' | python tools/infer_worker.py

# Transformer
echo '{"onnx":"transformer_v1.onnx","input":"testdata_tf/input","output":"wout_transformer","batch_size":256}
{"cmd":"exit"}' | python tools/infer_worker.py

# BigFormer
echo '{"onnx":"/workspace/C3/testcases/models/bigformer_v1.onnx","input":"/workspace/C3/testcases/testdata/c35/bigformer_v1/input","output":"wout_bigformer","batch_size":256}
{"cmd":"exit"}' | python tools/infer_worker.py
```

### 官方 selfcheck（四模型完整协议测试 + 精度校验）

```bash
cd ~/A4S/c3

# 四模型全部（spec 口径：2 warmup + 5 timed）
python /workspace/C3/testcases/selfcheck_worker.py \
    --worker "python tools/infer_worker.py" \
    --models mlp_v1 resnet_v1 transformer_v1 bigformer_v1 \
    --data-root /workspace/C3/testcases/testdata/c35 \
    --models-root /workspace/C3/testcases/models \
    --warmup 2 --timed 5 --batch-size 256 --check-precision

# 单独测某个模型（如只测 ResNet + BigFormer 的性能）
python /workspace/C3/testcases/selfcheck_worker.py \
    --worker "python tools/infer_worker.py" \
    --models resnet_v1 bigformer_v1 \
    --data-root /workspace/C3/testcases/testdata/c35 \
    --models-root /workspace/C3/testcases/models \
    --warmup 2 --timed 5 --batch-size 256 --check-precision
```

### 精度核对（手动与 golden 比对）

```bash
cd ~/A4S/c3

# MLP
python -c "import numpy as np; o=np.load('wout_mlp/logits.npy'); g=np.load('testdata_mlp/golden/logits.npy'); l=np.load('testdata_mlp/labels.npy'); print('allclose(1e-3):', np.allclose(o,g,rtol=1e-3,atol=1e-3), '| max_diff:', np.abs(o-g).max(), '| top1:', (o.argmax(-1)==l).mean(), '(gate 0.98)')"

# ResNet
python -c "import numpy as np; o=np.load('wout_resnet/logits.npy'); g=np.load('testdata_resnet/golden/logits.npy'); l=np.load('testdata_resnet/labels.npy'); print('allclose(1e-3):', np.allclose(o,g,rtol=1e-3,atol=1e-3), '| max_diff:', np.abs(o-g).max(), '| top1:', (o.argmax(-1)==l).mean(), '(gate 0.85)')"

# Transformer
python -c "import numpy as np; o=np.load('wout_transformer/logits.npy'); g=np.load('testdata_tf/golden/logits.npy'); print('allclose(1e-3):', np.allclose(o,g,rtol=1e-3,atol=1e-3), '| max_diff:', np.abs(o-g).max())"

# BigFormer
python -c "import numpy as np; o=np.load('wout_bigformer/logits.npy'); g=np.load('/workspace/C3/testcases/testdata/c35/bigformer_v1/golden/logits.npy'); print('allclose(1e-3):', np.allclose(o,g,rtol=1e-3,atol=1e-3), '| max_diff:', np.abs(o-g).max())"
```

---

## 6. C3.5 性能测量（运行时间 + 峰值显存）

```bash
cd ~/A4S/c3

# ResNet + BigFormer 性能（spec 只对这两个模型测性能）
python -c "
import subprocess, threading, time, json
import cupy as cp

def gpu_used_mib():
    f, t = cp.cuda.runtime.memGetInfo()
    return (t - f) // 1024 // 1024

MODELS = [
    ('ResNet', 'resnet_v1.onnx', 'testdata_resnet/input', 256),
    ('BigFormer', '/workspace/C3/testcases/models/bigformer_v1.onnx',
     '/workspace/C3/testcases/testdata/c35/bigformer_v1/input', 256),
]
for name, onnx, inp, bs in MODELS:
    proc = subprocess.Popen(['python', 'tools/infer_worker.py'],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    proc.stdout.readline()  # READY
    baseline = gpu_used_mib()
    task = json.dumps({'onnx': onnx, 'input': inp, 'output': '/tmp/perf_'+name, 'batch_size': bs})
    peak = [baseline]; stop = [False]
    def sampler():
        while not stop[0]:
            m = gpu_used_mib()
            if m > peak[0]: peak[0] = m
            time.sleep(0.02)
    th = threading.Thread(target=sampler, daemon=True); th.start()
    t0 = time.time()
    proc.stdin.write(task + '\n'); proc.stdin.flush()
    result = proc.stdout.readline()
    dt = time.time() - t0
    stop[0] = True; th.join(timeout=1)
    proc.stdin.write(json.dumps({'cmd':'exit'}) + '\n'); proc.stdin.flush()
    proc.wait()
    print('%s: time=%.2fs peak_mem=%dMiB (%.2fGB)' % (name, dt, peak[0]-baseline, (peak[0]-baseline)/1024))
"
```

---

## 7. 全套自测（一键运行）

```bash
cd ~/A4S/c3

# C3.1 + C3.5 端到端自测（DAG + 精度 + 准确率门槛）
python tests/selftest_c31_c35.py

# C3.2 + C3.3 微基准自评分
python benchmarks/c32_c33/bench_c32_c33.py --models mnist_mlp cifar_resnet18 transformer --output-dir results

# C3.3 独立严格评委
python tests/selftest_c33.py

# C3.4 内存规划门禁
python tests/selftest_c34.py

# C3.5 官方 worker 协议四模型完整测试
python /workspace/C3/testcases/selfcheck_worker.py \
    --worker "python tools/infer_worker.py" \
    --models mlp_v1 resnet_v1 transformer_v1 bigformer_v1 \
    --data-root /workspace/C3/testcases/testdata/c35 \
    --models-root /workspace/C3/testcases/models \
    --warmup 2 --timed 5 --batch-size 256 --check-precision
```

---

## 附：提交的两个命令模板（报名时填写）

| 子任务 | 命令模板 |
|--------|----------|
| **C3.1** | `python tools/export_dag.py --onnx {onnx} --output {output}` |
| **C3.5** | `python tools/infer_worker.py` |

- C3.1：占位符 `{onnx}` / `{output}` 由评测机填入。
- C3.5：不带任务参数，常驻 worker 进程，评测机经 stdin/stdout JSON 通信。

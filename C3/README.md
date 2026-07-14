# C3 算子调度与模型部署

## 命令模板（报名提交）

> 评测时以 `C3/` 为工作目录执行以下命令。

### C3.1 计算图解析与导出

```
python src/tools/export_dag.py --onnx {onnx} --output {output}
```

- `{onnx}`：ONNX 模型文件路径（评测机填入）
- `{output}`：输出 DAG JSON 文件路径（评测机填入）
- 程序以退出码 0 结束；stdout 不参与评测，评测仅读取 `--output` 指定的文件

### C3.5 模型推理

```
python src/tools/infer_worker.py
```

- 不带任务参数，常驻 worker 进程
- 评测机通过 stdin 下发 JSON 任务：`{"onnx":"...","input":"...","output":"...","batch_size":256}`
- worker 完成后经 stdout 回复：`{"status":"ok","samples":N}`
- 评测机发送 `{"cmd":"exit"}` 后 worker 干净退出（退出码 0）
- stdout 仅输出 `READY` 与结果行；所有日志走 stderr
- 完整协议见 `C35_WORKER_PROTOCOL.md`

## 环境依赖

```
numpy>=1.24
onnx>=1.15
onnxruntime>=1.17
cupy>=13.0
```

## 目录结构

```
C3/
├── README.md                     本文档
├── requirements.txt              依赖
└── src/                          框架源码
    ├── scheduler/                调度器核心库（C3.1/C3.2/C3.3/C3.4）
    │   ├── graph.py              ONNX 解析与 DAG 导出（C3.1）
    │   ├── strategy.py           算子分解与内核选择（C3.2）
    │   ├── precision.py          精度路由（C3.2 D1）
    │   ├── kernels.py            KernelSpecRef / KernelTuningParams（C3.2）
    │   ├── hardware.py           硬件能力模型（C3.2 D4/D5）
    │   ├── memory.py             内存规划与调度（C3.4 A–E）
    │   └── graph_passes/         算子融合（C3.3）
    │       ├── pipeline.py       GraphPassPipeline 入口
    │       ├── fusion.py         融合 pattern 实现
    │       └── shape_infer.py    形状推理
    ├── runtime/                  推理执行器（C3.3/C3.5）
    │   ├── cupy_runtime.py       GPU 图执行器 + weight streaming
    │   ├── ops_cupy.py           18 算子 CuPy GPU 实现
    │   ├── ops_numpy.py          18 算子 numpy 参考实现
    │   └── mock_runtime.py       C3.3 数值对齐用
    ├── tools/
    │   ├── export_dag.py         C3.1 CLI
    │   ├── infer_worker.py       C3.5 持久化 Worker
    │   └── infer.py              C3.5 一次性 CLI（兼容）
    ├── benchmarks/c32_c33/
    │   └── bench_c32_c33.py      C3.2/C3.3 自评分
    └── tests/                    自测脚本
```

## 模块导入

评分程序以 `C3/` 为工作目录，`scheduler`/`runtime` 等包位于 `C3/src/` 下。
各脚本内部通过 `sys.path.insert(0, src/)` 自动解析，`from scheduler import ...` 可直接使用。

## 自测

```bash
cd C3

# C3.1 DAG 导出
python src/tools/export_dag.py --onnx mlp_v1.onnx --output dag.json

# C3.2 + C3.3 自评分
python src/benchmarks/c32_c33/bench_c32_c33.py --models mnist_mlp cifar_resnet18 --output-dir results

# C3.5 worker 协议测试
echo '{"onnx":"resnet_v1.onnx","input":"testdata_resnet/input","output":"out","batch_size":256}
{"cmd":"exit"}' | python src/tools/infer_worker.py
```

## ResNet 运行时间优化计划

### 现状（实测）
- ResNet worker 计时中位数 **6.19s**（batch=256，40 个 batch）
- 单 batch 155ms = 纯 GPU 计算 154ms（H2D/D2H 仅 1ms）
- 20 个 Conv 占 151.5ms（98%），其中 layer1 的 4 个 Conv 各 14ms（共 56ms，占 37%）
- Q&A 确认"允许 CuPy 自定义 Kernel"

### 第一步：POC 验证 RawKernel direct conv 是否更快
在服务器上跑一个 POC：对 layer1 配置（n=256, ic=64, oc=64, 32×32, 3×3）对比：
- 当前 im2col + cuBLAS matmul（~14ms）
- CuPy RawKernel direct conv（每个线程算一个输出元素）

**关键不确定性**：direct conv 没用 Tensor Core，可能比 cuBLAS matmul 慢。POC 先验证，再决定是否投入。

### 决策树
- 若 RawKernel 更快 → 写完整的 direct conv kernel 替换 op_Conv，目标 ResNet 6s → 3s
- 若 RawKernel 更慢 → 放弃这个方向，ResNet 6s 已是 CuPy 手写算子的合理上限，转向其他优化（如 im2col 的 shared memory 优化）

### 改动范围
- 只改 `runtime/ops_cupy.py` 的 `op_Conv`（加 RawKernel 路径）
- 数值一致性：与 golden 比对，max_diff ≤ 1e-3
- 不影响 C3.1/C3.2/C3.3/C3.4

先跑 POC，根据结果决定是否继续。
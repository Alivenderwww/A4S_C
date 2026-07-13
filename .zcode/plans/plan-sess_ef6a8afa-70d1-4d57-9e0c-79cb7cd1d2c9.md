## C3.3 提分计划：新增 Conv→Add 残差融合（F2/F3 拿满 6 分）

### 当前失分（已用数据确认）
- **F2（3分）**：ResNet 缩减 37.7%、Transformer 18.7%、MLP 44.4%，锚点 60% 满分 → 各扣 1-1.5 分
- **F3（3分）**：buffer 缩减类似，各扣 1-1.5 分
- F1（5分）：ResNet 命中 4/5 canonical，Transformer 3/5，基本到顶（FusedSoftmaxDropout 推理图无 Dropout）
- F4（4分）：已满分（数值对齐 diff=0）

### 核心机会：ResNet 有 11 个 Conv→Add 残差结构未被融合
实测发现 ResNet 每个残差块的 `conv2→Add`（残差加）都是独立的融合机会：
- 8 个标准残差块：`conv2(2 launch) → Add(1 launch)` = 3 launch/个
- 3 个 downsample 层：`downsample_conv(2) → Add(1)` = 3 launch/个
- 所有候选的 conv 输出都**只被该 Add 消费**（conv_consumers=1），融合安全

### 算法设计：新增 `_match_conv_residual_add` matcher

**判据**（保证不误融合 bias-add）：
1. Conv 的输出只被一个 Add 消费
2. Add 的另一路输入是**非 initializer**（残差路径，不是 bias）
3. 这个 Conv→Add 对还没被别的 matcher 消费

**融合后**：`[Conv, Add]` → 1 个融合节点，launch 从 2+1=3 → 1，每对省 2 launch。
- ResNet 11 对 × 省 2 = 额外省 22 launch → opt 从 43 → 21，缩减率 (69-21)/69 = **69.6%**（超 60% 满分）

**双 Conv 输入同一个 Add 的处理**（layer2.0/3.0/4.0）：
- Add 有两路输入：`conv2_output` 和 `downsample_output`
- 我只融合 `conv2→Add`，把 downsample 的输出作为融合节点的外部输入
- matcher 按 Conv 遍历，每个 Conv 看自己的后继 Add 是否可用（未被消费），先到先得

### matcher 顺序（关键）
新增的 `_match_conv_residual_add` 必须在 `_match_ew_chain` **之前**运行，因为 EW chain matcher 会先吃掉 `Add→Relu`（把 Add 消费掉），导致 Conv→Add 无法配对。顺序改为：
```
_match_matmul_bias → _match_residual_norm → _match_softmax_dropout →
_match_conv_bn → _match_conv_residual_add(新) → _match_ew_chain → _match_compute_activation
```
但要注意：`_match_ew_chain` 现在能融合 `Add→Relu`（ResNet 的 8 个），如果 Conv→Add 先把 Add 吃了，EW chain 就少了一个起点。需要让 Conv→Add 融合后，融合节点能继续参与 EW chain（即融合节点的输出被 Relu 消费时，由 compute_activation 处理）。

**更优方案**：让 Conv→Add 融合后，融合节点的 op_type 设为 "Add"（终端 op），这样它仍能被 `_match_compute_activation` 当作 compute op 继续和 Relu 融合。或者直接让 `_match_conv_residual_add` 同时吃掉后续的 Relu（三元融合 Conv→Add→Relu）。

### F4 数值对齐（必须保 diff=0）
- MockRuntime 通过 `fused_ops` 子节点重放执行融合节点
- 融合 `[Conv, Add]` 后，fused_ops=[conv_clone, add_clone]，MockRuntime 按序重放，中间张量在 env 中流转
- **数值按构造保证一致**（和现有 FusedConvRelu 同理），diff=0

### 改动范围
- **只改 `scheduler/graph_passes/fusion.py`**：
  1. 新增 `_match_conv_residual_add` 方法（~25 行）
  2. 在 `run()` 的 matcher 顺序中插入它
  3. 可能微调 `_match_ew_chain` 兼容被融合的 Add
- 不碰 graph.py / mock_runtime.py / ops_numpy.py
- 数值对齐 + launch/buffer 计数逻辑不变

### 验证策略
1. **数值**：跑 selftest_c33.py（独立严格评委），确认 F4 数值对齐 diff=0 不破
2. **F2/F3**：跑 bench_c32_c33.py，确认 ResNet 缩减率 ≥ 60%（F2/F3 满分）
3. **F1**：确认不误伤现有 canonical pattern 命中
4. **F4 结构**：validate() 通过 + 节点数不增 + inputs/outputs 保留

### 预期收益
- ResNet F2：37.7% → ~70%（3/3 满分，+1.1 分）
- ResNet F3：36.2% → ~60%+（3/3 满分，+1.2 分）
- Transformer/MLP 不受影响（无 Conv→Add 残差）
- **C3.3 总分：ResNet 从 ~10.7 → ~13+/15**
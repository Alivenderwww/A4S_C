# 官方 CModel 验证 (`verify.py`)

用官方 AEC golden model(`public/aec-cmodel-release/bin/aec-precise-*`)对 C1
编译器输出做**端到端验证**:正确性(dump 输出 vs 参考)+ 官方性能指标
(`steps` = warp-level 动态执行指令数,即 B 类评分指标)。

## 用法(WSL,linux-x86_64 CModel)

先在 Windows 端 build 好 `compiler/aec-cc.exe`(`cd C1 && make build submit`),
然后在 **WSL** 里跑(脚本通过 interop 调用 `aec-cc.exe`,`aec-precise` 是原生
Linux 程序):

```bash
python3 C1/sim/cmodel/verify.py
```

脚本对 5 个 public case 自动:

1. 按官方固定 GMEM/PMEM 布局与 seed(`PUBLIC_AEC_PRECISE_COMMANDS.md`)生成
   `pmem.bin` + `input_<buffer>.bin`(写到 WSL `/tmp/c1_inputs/`);
2. 用 `aec-cc` 编译 kernel.ptx(`-O0` 和 `-O2`)到 `sim/build/cmodel/`(gitignore);
3. 跑 `aec-precise`,读 stdout JSON 的 `status` + `steps`,`--dump` 输出 buffer;
4. 抽查输出 vs 参考(elementwise 公式 / matmul,numpy-free)。

输出每个 case 的:`status`、`O0/O2 steps`、`O0/O2`(动态指令数加速比)、正确性。

路径全部相对本文件解析,任意 clone 位置可用。评测机为原生 Linux(无 `.exe`)时
脚本自动改用 `compiler/aec-cc`。

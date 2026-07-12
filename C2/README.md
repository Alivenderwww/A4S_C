# AEC-C2 参赛资料

## 任务

实现 AEC 虚拟 GPGPU 的 Host Runtime：

```text
libaec.so
```

冲击 Excellent 时，再提交：

```text
agents/dma_agent.py
agents/kernel_agent.py
```

主办方提供公共头文件、固定 Kernel image、受控虚拟设备、起始代码、示例和公开测试。
参赛者可自行组织 Runtime 源码和构建方式，但不能修改或替换公共头文件、设备库、Kernel
image 或 grader 来改变评分契约。

## 快速开始

```bash
make -j2
make examples
./bin/01_device_query
./bin/02_isa_encoding
python3 grader/public_grade.py --submission . --profile public
```

单项或全部公开测试：

```bash
python3 cases/test_r101.py --submission .
python3 cases/test_r201.py --submission .
make public-cases
```

公开测试用于开发和排错。最终评分使用同一套公共 ABI、ISA、数值规则和 requirement，
另加入隐藏边界、并发、故障和性能输入。公开测试全部通过不等于满分。

## 最终评分

- 总分 100：Runtime 30、计算库 30、虚拟 Driver 20、Agents 20。
- R101–R106、R201–R204、R301–R304 按 requirement 计分；该项的公共和隐藏检查全部通过才得分。
- R401/R402 各含 4 分正确性和 6 分隐藏虚拟周期性能。
- Basic、Good、Excellent 除总分外还有强制 gate，见 `docs/05_Agent与评分标准.md`。
- 正式评分只使用提交截止时保存的同一份 artifact；性能只看虚拟周期，不看 wall-clock。

## 阅读顺序

1. `docs/01_赛题说明.md`
2. `include/aec_runtime.h`
3. `docs/02_Runtime与设备规范.md`
4. `docs/03_AEC_ISA规范.md`、`docs/04_数值规范.md`
5. `docs/05_Agent与评分标准.md`
6. `docs/06_提交与公开测试指南.md`

`RELEASE_MANIFEST.json` 记录本资料包的 ABI/ISA 版本和文件哈希。若资料包内文件与赛事官网
勘误冲突，以官网最新正式勘误为准；没有正式勘误时，不自行推断或修改公共契约，应向主办方
提交问题。

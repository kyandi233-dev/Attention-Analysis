# Tests

根 `tests/` 用于验证仓库级源码、行为分析、历史 NIR 逻辑和 portable runtime 边界；它不是 `runtime/nir-formal/` 自包含运行包内部测试的替代品。

## 当前职责

| 测试 | 覆盖范围 |
|---|---|
| `test_protocol_and_config.py` | 协议与配置语义 |
| `test_behavior*.py` | 行为分析、阶段处理与报告 |
| `test_io.py` | dropped 行、AVI 位置映射与 block 解析 |
| `test_nir.py` | NIR 几何、ROI、椭圆与缺失语义 |
| `test_benchmark.py` | 历史 benchmark / 评估逻辑 |
| `test_formal_nir.py` | 正式 NIR 项目级接口与约束 |
| `test_portable_nir_gpu_package.py` | portable NIR GPU runtime 的仓库级结构检查 |
| `test_review.py` | 人工复核、抽样与保存/恢复语义 |

其他 `test_*.py` 继续按文件名对应具体模块。

## 与 runtime tests 的边界

```text
tests/
└── 仓库级回归：src / configs / behavior / NIR / portable-package contract

runtime/nir-formal/tests/
└── 自包含正式运行包内部测试
```

当前 `runtime/nir-formal/tests/` 只保留 `test_phase_windows.py`，用于验证正式 phase-window 逻辑。它随 runtime 一起迁移，目的是在不依赖根仓库测试套件的情况下对运行包做最小自检。

因此两套测试有意并存：根 `tests/` 负责项目开发与回归，runtime tests 负责正式运行包的独立可移植性。

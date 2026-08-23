# Tests

根 `tests/` 用于验证仓库级源码、当前 Behavior、共享 IO/NIR 几何和仍保留的历史兼容逻辑；它不是 `runtime/nir-formal/` 自包含运行包内部测试的替代品。

## 当前职责

| 测试 | 覆盖范围 |
|---|---|
| `test_behavior_formal_bb.py` | 当前 FocusWave v3.1.3 BB Behavior 逻辑 |
| `test_behavior*.py` | 早期通用 Behavior / 报告兼容逻辑；不作为 current 正式口径 |
| `test_io.py` | dropped 行、AVI 位置映射与 block 解析 |
| `test_nir.py` | NIR 几何、ROI、椭圆与缺失语义 |
| `test_benchmark.py` | 历史 benchmark / 评估逻辑 |
| `test_formal_nir.py` | 正式 NIR 项目级接口与约束 |
| `test_portable_nir_gpu_package.py` | portable/formal NIR runtime 的仓库级结构检查 |
| `test_review.py` | 人工复核、抽样与保存/恢复语义 |

旧 Behavior/NIR 测试之所以暂时仍存在，是因为对应历史兼容源码尚未获得删除授权；它们不得被解释为当前分析入口。后续若明确删除对应历史源码，应同步删除或迁移这些测试，不保留孤立回归。

## 与 runtime tests 的边界

```text
tests/
└── 仓库级回归：src / configs / current Behavior / shared NIR / retained compatibility

runtime/nir-formal/tests/
└── 自包含正式运行包内部测试
```

`runtime/nir-formal/tests/` 随正式 runtime 一起迁移，用来在不依赖根仓库测试套件的情况下验证正式运行包自身关键逻辑。根 tests 与 runtime tests 有意分离：前者服务仓库开发和兼容回归，后者服务正式运行包可移植性。

# Tests

根 `tests/` 用于验证仓库级源码、当前 Behavior、共享 IO/NIR 几何和仍保留的历史兼容逻辑；它不是 `runtime/nir-formal/` 自包含运行包内部测试的替代品。

## 当前职责

| 测试 | 覆盖范围 |
|---|---|
| `test_behavior_formal_bb.py` | 当前 FocusWave v3.1.3 BB Behavior 逻辑 |
| `test_behavior*.py` | 早期通用 Behavior / 报告兼容逻辑；不作为 current 正式口径 |
| `test_current_data_roots.py` | 当前 Behavior/NIR 四候选数据根与重复被试保护 |
| `test_io.py` | dropped 行、AVI 位置映射与 block 解析 |
| `test_nir.py` | NIR 几何、ROI、椭圆与缺失语义 |
| `test_benchmark.py` | 历史 benchmark / 评估逻辑 |
| `test_formal_nir.py` | 正式 NIR 项目级接口与约束 |
| `test_portable_nir_gpu_package.py` | 当前 `runtime/nir-formal/` 的仓库级结构与配置检查 |
| `test_review.py` | 人工复核、抽样与保存/恢复语义 |

旧 Behavior/NIR 测试之所以暂时仍存在，是因为对应历史兼容源码尚未获得删除授权；它们不得被解释为当前分析入口。后续若明确删除对应历史源码，应同步删除或迁移这些测试，不保留孤立回归。

## Current baseline suite

NVIDIA `1.0.0` 基线和后续 AMD/DirectML 分支的可移植仓库级验收以 `.github/workflows/ci.yml` 中的 current baseline suite 为准：

```text
tests/test_behavior_formal_bb.py
tests/test_current_data_roots.py
tests/test_formal_nir.py
tests/test_io.py
tests/test_nir.py
tests/test_portable_nir_gpu_package.py
runtime/nir-formal/tests/
```

这些测试使用临时数据或仓库内冻结资产，适合在干净机器/CI 中运行。根目录直接执行 `pytest -q` 还会包含保留的历史兼容与历史集成测试；其中部分测试依赖旧预实验数据、历史 artifacts 或特定复现环境，因此**不能把全量根 `pytest -q` 的结果等同于 current baseline 是否有效**。

## 与 runtime tests 的边界

```text
tests/
└── 仓库级回归：src / configs / current Behavior / shared NIR / retained compatibility

runtime/nir-formal/tests/
└── 自包含正式运行包内部测试
```

`runtime/nir-formal/tests/` 随正式 runtime 一起迁移，用来在不依赖根仓库历史测试套件的情况下验证正式运行包自身关键逻辑。根 tests 与 runtime tests 有意分离：前者服务仓库开发和兼容回归，后者服务正式运行包可移植性。

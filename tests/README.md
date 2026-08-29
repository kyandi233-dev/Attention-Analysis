# Tests

根 `tests/` 同时承担当前正式 Behavior/NIR/RGB 合同、仓库级共享逻辑和受控历史兼容回归。它不是 `runtime/nir-formal/tests/` 自包含运行包测试的替代品，也不能把历史资产的 skip 解释成对应功能已经通过验证。

## 当前 CI 三层验收

`.github/workflows/ci.yml` 当前包含三类互补检查：

1. `formal-adapter-contract`：编译正式分析模块，并运行 Behavior、共享身份、NIR staged、RGB lightweight 等针对性合同测试；
2. `current-baseline`：保留当前仓库与 `runtime/nir-formal/` 的基线回归；
3. `portable-full-suite`：在干净 Ubuntu/Python 3.11 环境执行根目录 `python -m pytest -q`，随后再执行 `runtime/nir-formal/tests`。

因此现在**干净环境的根 `pytest -q` 本身应当完成成功**。如果它红了，不能再默认解释成“历史测试本来就会失败”，必须先逐项分类。

## 当前正式重点测试

| 测试 | 覆盖范围 |
|---|---|
| `test_formal_identity_questionnaire.py` / `test_formal_identity_invariants.py` | `participant_key` 来源、`participant_group_id` canonical 分组、legacy alias、禁止 session fallback |
| `test_formal_cross_modal_contracts.py` | 同一 synthetic cohort 在 Behavior/NIR/RGB 三条路线上的身份分组一致性、OAR QC 保留、path-registry 版本合同 |
| `test_behavior_science_v3_contract.py` 等 | Go/No-Go 分母、RT、多尺度、probe 锚点、Q1/Q2、cluster/bootstrap/CV 合同 |
| `test_nir_*` 正式合同组 | staged pupil-only materialization、身份 parity、时间/视觉 gate、候选与 failure tables |
| `test_nir_validation_authority.py` | 包级默认 NIR validation 必须指向 pupil-only；旧 PIR runner 只能显式 legacy 导入 |
| `test_nir_authoritative_manifest_semantics.py` | 正式 NIR manifest 必须把 PIR/iris geometry refusal 与 ocular-aperture QC preservation 分开，旧 `pir_oar_*` 混合键不能继续出现在权威输出 |
| `test_rgb_formal_lightweight.py` | Motion/Pose/Blink 轻量路线、双眼一致性、时间断点、identity gate；某模态/组件不可用不能伪装成 0 或成功 |
| `test_portable_nir_gpu_package.py` | `runtime/nir-formal/` 仓库级结构与配置检查 |

本轮新增的 identity invariant、NIR validation authority 和 NIR manifest semantics tests 同时进入 targeted `formal-adapter-contract`；即使 full suite 尚未跑完，也能更快定位当前正式合同倒退。

## 模态缺失在测试中的正确解释

participant/session identity 与 modality availability 是两层合同。当前测试和未来新增测试应遵循：

```text
known governed session
  + RGB/mmWave/NIR source missing
  != participant missing
```

应测试的是：

- 缺模态有明确 `source_missing/not_estimable`；
- 其他模态 session 不被删除；
- 不自动填 0；
- 不自动重跑昂贵 producer；
- paired multimodal subset 只能显式构造。

不应为了制造 44/44 coverage 而给缺失模态造 synthetic success 行。

## 两类显式历史 skip

### `requires_local_raw_data`

部分早期/预实验测试继续复用历史 `configs/preexperiment.yaml` 和原采集机器数据。若配置中的 legacy raw root 在当前机器不存在，测试会显式 `SKIP`，而不是报一组误导性的路径失败。

这类 skip 的含义是：

- 历史测试源码仍保留用于 provenance；
- 当前干净环境没有验证对应本地原始数据路径；
- **不是测试通过，也不是当前正式分析依赖这些机器路径。**

### `legacy_optional_backend`

部分历史测试仍引用当前仓库已不再包含的旧 ROI/adapter 脚本，例如旧 `roi_*`、`nir_sequence_detect.py` 或个别 `nir_detect_batch.py` 接口。根据项目“未授权不得删除历史资产”的规则，这些测试暂时保留，但会显式 `SKIP` 并标记为 provenance-only。

同样，这表示“历史目标已经不属于当前仓库 active route”，而不是对应 backend 已通过当前验收。若以后获得删除/归档授权，应同时处理实现记录和这些孤立测试。

## 与 runtime tests 的边界

```text
tests/
└── 仓库级：正式分析合同 + shared logic + 显式隔离的历史兼容回归

runtime/nir-formal/tests/
└── 自包含正式 NIR 生产运行包内部测试
```

当前正式入口以 `ANALYSIS_SETUP_FIRST.md`、`configs/README.md` 与 `docs/060-formal-analysis/009-正式管线修复后完整复审_20260830.md` 为准；历史测试文件名或历史 package API 不得反向定义当前生产路线。

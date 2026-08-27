# 配置

`configs/` 保存当前行为分析配置，以及用户明确要求继续保留、可直接重跑的历史分析配置。正式 NIR GPU 分析使用独立 frozen runtime 配置：`runtime/nir-formal/config.yaml`。

| 文件 | 用途 | 当前定位 |
|---|---|---|
| `behavior_formal.yaml` | FocusWave v3.1.3、最终正式 BB 行为分析 | **当前行为配置** |
| `sart_bbb_v3_0.yaml` | 2026-08-16、sub-011~030、BBB SART 分析 | **历史可执行配置，不是当前口径** |
| `preexperiment.yaml` | 预实验 v2 路径、窗口、审批门等 | 历史兼容配置；不作为 current CLI 默认入口 |
| `formal.yaml` | 08-16 阶段 NIR ROI / PuReST 候选链 | 历史兼容配置；不作为 current NIR 配置 |
| `nir_pypupilext_native_benchmark.yaml` | production evidence → source-pixel crop → 七算法 benchmark | **当前测量学验证配置；不运行 YOLO/RITnet，不产生 accuracy 结论** |
| `../runtime/nir-formal/config.yaml` | YOLO26n + RITnet 正式 NIR | **当前正式 NIR 配置** |

## 正式原始数据根

正式原始数据位于两个逻辑目录 `正式实验` 与 `Data`，但两块外接存储设备在 Windows 下的盘符可能根据连接顺序在 `E:` / `F:` 之间变化。因此 current Behavior 与 NIR 配置统一声明四个候选根：

```text
E:/正式实验
F:/正式实验
E:/Data
F:/Data
```

运行时忽略不存在的候选路径，并在所有有效根中发现被试。若同一被试在多个有效根中出现重复正式数据，current reader 应拒绝静默选择并明确报错。

当前 BB reader 同时接受 `sub-XXX_` 与 `sub-XXX` 被试目录；正式 BB 数据只纳入编号 ≥31 且同时具有 `Block1`、`Block2` 行为文件的完整被试。

旧 BBB 的配置、程序和结果均与当前 BB 分离：配置为 `sart_bbb_v3_0.yaml`，脚本为 `scripts/sart_bbb_v3_0_analysis.py`，实现包为 `src/attention_pipeline/behavior_bbb_v3_0/`。历史报告和图保存在 `docs/030-behavior/history/BBB-v3.0/`。这样以后需要重跑 BBB 时不必重新写代码，同时不会把三 block 统计逻辑混入当前 BB。

当前行为配置不从 BBB 直接继承每 block 的 trial / No-Go / probe 固定数；这些值依据最终 v3.1.3 任务源文件和正式数据审计后再冻结。

# 配置

`configs/` 保存仓库级实验、行为分析和历史诊断配置。**当前正式 NIR GPU 分析不使用本目录中的 NIR 配置**；已经用于全量正式分析的配置只认：

```text
runtime/nir-formal/config.yaml
```

| 文件 | 用途 | 当前定位 |
|---|---|---|
| `preexperiment.yaml` | 预实验 v2 路径、被试、窗口、审批门与早期 NIR 门槛 | 历史预实验配置 |
| `formal.yaml` | 08-16 阶段 NIR ROI / PuReST 候选链参数 | 历史兼容配置，不是当前正式 NIR 配置 |
| `sart_formal.yaml` | 2026-08-16 的 v3.0 BBB / sub-011~030 SART 分析配置 | **历史行为分析配置；不是最终 v3.1.3 BB 配置** |
| `../runtime/nir-formal/config.yaml` | FocusWave v3.1.3 + YOLO26n + RITnet 正式 NIR 运行配置，并冻结最终正式阶段结构 | **当前正式运行配置** |

## 当前行为版本边界

最终正式实验已在 runtime 中明确冻结：

```text
FocusWave v3.1.3
min_subject_number: 31
expected_formal_blocks: 2
block1 + block2
```

因此当前没有一个已经完成审计的“最终 BB 行为分析配置”可以直接等同于旧 `sart_formal.yaml`。后续应先以最终数据核实每 block 试次数、No-Go 数、探针位置、实际被试集合与异常被试，再建立新的 current behavior config；不能只把旧配置中的 `BBB` 文本改成 `BB`。

## 历史配置原则

配置文件保留研究阶段差异，不为了统一格式追溯改写历史参数。历史脚本被删除后，旧配置仍可作为当时研究设计和工作记录的结构化 provenance；当前运行入口始终以对应模块 README 与 runtime 为准。

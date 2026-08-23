# 配置

`configs/` 保存仓库级实验与分析配置。正式 NIR GPU 运行包为了可迁移和自包含，使用 `runtime/nir-formal/config.yaml`，不依赖本目录中的历史 NIR 配置。

| 文件 | 用途 | 当前定位 |
|---|---|---|
| `preexperiment.yaml` | 预实验 v2 路径、被试、窗口、审批门与早期 NIR 门槛 | 历史预实验配置 |
| `formal.yaml` | 早期正式 NIR / ROI / PuReST 候选参数 | 历史配置，不是当前正式 GPU runtime 配置 |
| `sart_formal.yaml` | 正式 SART 行为分析配置 | 当前行为分析配置 |
| `../runtime/nir-formal/config.yaml` | FocusWave v3.1.3 + YOLO26n + RITnet 正式 NIR 运行配置 | 当前正式 NIR 配置 |

配置文件保留研究阶段差异；不要为了统一格式把历史配置改写成当前参数。

# 配置

`configs/` 保存仓库级实验、行为分析和历史诊断配置。**当前正式 NIR GPU 分析不使用本目录中的 NIR 配置**；已经用于全量正式分析的配置只认：

```text
runtime/nir-formal/config.yaml
```

| 文件 | 用途 | 当前定位 |
|---|---|---|
| `preexperiment.yaml` | 预实验 v2 路径、被试、窗口、审批门与早期 NIR 门槛 | 历史预实验配置 |
| `formal.yaml` | 08-16 阶段 NIR ROI / PuReST 候选链参数 | **历史兼容配置**；保留旧脚本/工作记录路径，不是当前正式 NIR 配置 |
| `sart_formal.yaml` | 正式 SART 行为分析配置 | 当前行为分析配置 |
| `../runtime/nir-formal/config.yaml` | FocusWave v3.1.3 + YOLO26n + RITnet 正式 NIR 运行配置 | **当前正式 NIR 配置** |

## 为什么仍保留 `formal.yaml`

`formal.yaml` 的文件名确实容易与当前正式 runtime 混淆，但仓库中的历史脚本、命令和工作记录已经大量引用 `configs/formal.yaml`。为保持研究过程可复现，本轮不强行重命名它，而是在文件头加入醒目的历史配置警告。

判断当前 NIR 配置时，不看文件名中的 `formal`，而看运行入口：

```text
当前正式 NIR  → runtime/nir-formal/config.yaml
历史 ROI 诊断 → configs/formal.yaml
正式 SART     → configs/sart_formal.yaml
```

配置文件保留研究阶段差异；不要为了统一格式把历史参数改写成当前参数。

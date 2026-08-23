# 配置

`configs/` 保存当前行为分析配置和历史兼容配置。正式 NIR GPU 分析使用独立 frozen runtime 配置：

```text
runtime/nir-formal/config.yaml
```

| 文件 | 用途 | 当前定位 |
|---|---|---|
| `behavior_formal.yaml` | FocusWave v3.1.3、正式 BB 行为分析 | **当前行为配置** |
| `preexperiment.yaml` | 预实验 v2 路径、窗口、审批门等 | 历史配置 |
| `formal.yaml` | 08-16 阶段 NIR ROI / PuReST 候选链 | 历史兼容配置 |
| `../runtime/nir-formal/config.yaml` | YOLO26n + RITnet 正式 NIR | **当前正式 NIR 配置** |

旧 `sart_formal.yaml`（v3.0 BBB、sub-011~030）已经退出 active configs；其副本与旧分析 bundle 位于：

```text
docs/030-behavior/history-bbb-v3.0/
```

完整可执行旧状态另外冻结在：

```text
history/behavior-bbb-v3.0
```

当前行为配置不写死最终被试总数，也不从 BBB 直接继承每 block 的 trial / No-Go / probe 数；这些由正式 v3.1.3 数据首先校验。

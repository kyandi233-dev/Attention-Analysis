# 配置

`configs/` 只保存当前仍可能直接运行或仍作为兼容入口存在的配置。正式 NIR GPU 分析使用独立 frozen runtime 配置：`runtime/nir-formal/config.yaml`。

| 文件 | 用途 | 当前定位 |
|---|---|---|
| `behavior_formal.yaml` | FocusWave v3.1.3、最终正式 BB 行为分析 | **当前行为配置** |
| `preexperiment.yaml` | 预实验 v2 路径、窗口、审批门等 | 历史兼容配置 |
| `formal.yaml` | 08-16 阶段 NIR ROI / PuReST 候选链 | 历史兼容配置 |
| `../runtime/nir-formal/config.yaml` | YOLO26n + RITnet 正式 NIR | **当前正式 NIR 配置** |

旧 `sart_formal.yaml`（v3.0 BBB、sub-011~030）已经退出 active configs。冻结副本位于：

```text
docs/030-behavior/history/BBB-v3.0/sart_formal_v3.0.yaml
```

完整旧 BBB 可执行代码由 `history/behavior-bbb-v3.0` 分支冻结。

当前行为配置不从 BBB 直接继承每 block 的 trial / No-Go / probe 固定数；这些值应依据最终 v3.1.3 任务脚本或正式数据重新核验后再冻结。

# 00｜配置目录与映射

## 当前状态（2026-08-23）

`configs/` 中同时保留预实验、早期 NIR 候选阶段和正式 SART 行为分析的配置。它们属于不同研究阶段，不能再把某一个 YAML 描述成整个项目的“唯一运行配置”。

| 文件 | 当前角色 | 注意事项 |
|---|---|---|
| `preexperiment.yaml` | 预实验 v2 历史/兼容配置 | 保留早期审批门、预实验被试、窗口与 NIR benchmark/sequence 设置；不是当前全量正式 NIR 配置 |
| `formal.yaml` | **早期正式 NIR ROI 候选阶段历史兼容配置** | 文件头已明确标注 historical compatibility；其中 candidate/blocked/production 未冻结状态只代表当时 full-face ROI 阶段 |
| `sart_formal.yaml` | 正式 BBB SART 行为分析配置 | 对应 `src/attention_pipeline/behavior_formal/`；包含正式行为数据路径、被试范围、协议、统计 seed 与分析阶段 |

YOLO26n + tracking + RITnet 的 portable route 配置不在本目录，而被冻结在：

```text
runtime/nir-yolo-tracking-ritnet-v1/config.yaml
```

该 runtime 配置是 2026-08-22 package 创建时的跨机准入默认值，并不自动等于后来全量正式运行的最终冻结参数。当前 Git 分支尚未找到正式 full-run 的最终 `run_manifest.json`，因此暂不创造新的 `configs/nir_final.yaml`。

## 配置使用原则

1. 读取配置前先确认它属于哪个研究阶段和入口，不根据文件名中的 `formal` 自动判断“这是当前最终配置”。
2. `configs/formal.yaml` 继续保留旧脚本兼容路径，不重命名或删除，除非后续完成引用迁移并单独批准。
3. `sart_formal.yaml` 与正式行为分析模块对应；NIR 和行为分析不强行共用一份 YAML。
4. portable runtime v1 保持冻结，当前状态说明写在 `runtime/README.md`，不直接修改 package 内配置以避免破坏 checksum。
5. 找到真实 full-run manifest 后，再决定是否需要建立一个新的最终 NIR 配置摘要，而不是根据默认值重建。

---

## 历史说明

> 2026-08-13 15:35（Asia/Shanghai）｜当时 `preexperiment.yaml` 被定义为预实验 v2 的唯一运行配置。

这条说明保留用于解释 08-13 阶段的仓库状态；随着正式 NIR、SART 和 portable runtime 后续加入，它已不再描述整个仓库当前配置结构。

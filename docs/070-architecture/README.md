# Architecture｜架构说明

本目录用于说明 Attention-Analysis 的整体系统结构，回答“各模块如何组织和连接”。

主要包括：

- NIR 正式分析 pipeline 的整体流程
- 输入数据、时间轴、模型、运行时和输出之间的数据流
- `datasets/`、`models/`、`runtime/`、`src/`、`configs/`、`scripts/` 的职责边界
- 正式分析与诊断/历史验证流程的关系
- 可复现运行结构与跨设备部署关系

具体算法原理放在 `060-methods/`；方案选择理由放在 `080-decisions/`；历史执行过程继续保留在 `工作记录/`。

# GitHub 仓库改名与 NVIDIA 分支、Tag 整理工作记录

> 2026-08-24 02:16（Asia/Shanghai）｜仓库已改名为 `Attention-Analysis`，当前默认维护分支收敛为 `nvidia-cuda`，退出路线由三个 annotated tags 保存。

## 总结

本次将 GitHub 仓库从 `attention-pipeline-v2` 改名为 `Attention-Analysis`，把已经完成正式全量分析的 NVIDIA/CUDA 代码线从通用名 `main` 收敛为 `nvidia-cuda`，并删除已被当前分支或 tag 完整替代的远程、远端跟踪和本地旧分支。未来 AMD 路线采用用户确认的名称 `amd-DirectML`，但本次不创建没有实现内容的空分支。

三个历史入口为：`v0.8-tracking`、`v0.9-nir-formal`、`behavior-bbb-v3.0`。tag 标记完整仓库快照，不是给单个文件分别编号；计划中的 `v1.0`、`v1.1` 仍需在对应完整版本真正冻结后创建，本次不提前占位。

## 目录

- [计划](#计划)
- [执行决策过程](#执行决策过程)
- [校验](#校验)
- [最终决策结果概览](#最终决策结果概览)

## 计划

背景：GitHub 默认分支仍指向退出的 tracking 路线，远端存在多个临时历史分支；项目文档已使用 `Attention-Analysis`，但仓库名与本地 remote 仍是旧名称。

目标：使用 tag 保存历史阶段，使当前维护线、默认分支、仓库名称和当前入口文档一致，同时为未来 AMD/DirectML 路线保留清晰命名空间。

步骤：

1. 同步远端引用并核验当前分支、提交和未提交文件。
2. 在用户明确授权后，将本地旧工作树恢复到远端当前版本，保留 `venv-labelimg`。
3. 创建并推送三个 annotated tags，再删除被 tags 或当前分支替代的历史远程分支。
4. 将 GitHub 仓库改名为 `Attention-Analysis`，同步本地 `origin`。
5. 创建并推送 `nvidia-cuda`，设为 GitHub default，删除远端 `main` 和旧 tracking 分支。
6. 核验本地旧分支提交均可由当前分支或 tag 到达后，再删除本地旧分支引用。
7. 更新当前入口文档；日期型历史快照保持原文，不追溯替换其中的旧分支或目录名称。

校验：远端 HEAD、远端分支列表、本地 tracking、tag 目标提交、Git 状态、remote URL、当前文档引用，以及旧本地分支提交的可达性。

## 执行决策过程

> 用户：本地的文件除了虚拟环境全都能删，都是老版本了。

回答与落实：先执行 `fetch` 确认远端 `main` 存在，再把本地工作树恢复到远端当前状态；保留 `.git` 和 `venv-labelimg`，删除旧 checkout 的未提交改动、缓存、旧模型和旧输出。Git 跟踪内容仍可从提交历史恢复；被清理的本地未跟踪旧文件没有单独备份。

> 用户：为什么有这么多分支，为什么不是打 tag？

回答与落实：开发分支不再承担永久历史归档。创建 annotated tags 后，删除 `history/tracking-era-2026-08`、`history/behavior-bbb-v3.0`、`codex/nir-formal-gpu-v3`、`codex/v2-YOLO+Tracking+RInet` 等退出分支。tracking 历史由 tag 恢复，不再由常驻 branch 表示。

> 用户：`main` 可以换个名字吗？后续会建立 AMD 卡分支。

回答与落实：分支改名不改变文件或提交历史，但需要同步 default、tracking 与当前操作说明。用户最终确认当前分支名为 `nvidia-cuda`，未来 AMD 分支名为 `amd-DirectML`；本次只创建已有真实实现的 `nvidia-cuda`。

> 用户：为什么只剩 3 个 tag？

回答与落实：本次开始前远端没有现成 tags；此前看到的是 branch。引用对话中 `v1.0` 和 `v1.1` 是未来版本规划，不是已经创建的 tag。本次只创建有明确历史提交和语义的三个 tag，不提前制造空版本。

## 校验

| 检查项 | 结果 |
|---|---|
| GitHub 仓库 | `kyandi233-dev/Attention-Analysis` |
| 本地 remote | `https://github.com/kyandi233-dev/Attention-Analysis.git` |
| GitHub default | `nvidia-cuda` |
| 当前远端维护分支 | `nvidia-cuda` |
| 当前本地分支 | `nvidia-cuda`，tracking `origin/nvidia-cuda` |
| `v0.8-tracking` | `825238ce6a7fab721b48fbf80039ab23bbc4d671` |
| `v0.9-nir-formal` | `5c8847ba6cb8bbf8aa1f502a239ab025d5a87f37` |
| `behavior-bbb-v3.0` | `07667b63dc1ca6116210464e90ba46c847fc6a00` |
| 未来 AMD 名称 | `amd-DirectML`，尚未创建 |
| 算法/参数/正式结果 | 未修改 |

## 最终决策结果概览

| 决策项 | 结果 | 依据 | 文件位置 |
|---|---|---|---|
| 项目名称 | `Attention-Analysis` | 项目文档与 GitHub 仓库统一 | 根 `README.md`、GitHub repository |
| NVIDIA 当前维护线 | `nvidia-cuda`，GitHub default | 当前正式 runtime 已在 NVIDIA/CUDA 路线上完成全量运行 | `AGENTS.md`、`runtime/nir-formal/README.md` |
| AMD 后续维护线 | 预留 `amd-DirectML`，本次不创建 | 避免空分支被误解为已有实现 | 根 `README.md`、`AGENTS.md` |
| tracking 历史 | tag `v0.8-tracking` | 退出开发线应冻结为历史快照 | Git tag |
| 正式 NIR 完成阶段 | tag `v0.9-nir-formal` | 保存逐帧 YOLO26n + RITnet 正式阶段 | Git tag |
| 历史 BBB | tag `behavior-bbb-v3.0`，同时保留可执行实现 | 用户要求未来可直接重跑 | Git tag 与 `src/attention_pipeline/behavior_bbb_v3_0/` |
| `v1.0` / `v1.1` | 本次不创建 | 尚未到对应完整版本冻结点 | 后续版本工作记录 |

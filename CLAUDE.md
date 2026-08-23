# CLAUDE.md｜attention-pipeline-v2 入口

> 2026-08-23｜Claude 的仓库内稳定入口。具体仓库工作规则统一以 `AGENTS.md` 为准，本文件不重复维护完整项目状态。

## 开始任务前

优先读取仓库内相对路径：

1. `AGENTS.md`
2. `README.md`
3. `000-项目总览与架构.md`
4. `docs/00-目录与映射.md`
5. 与当前任务直接相关的目录索引
6. 需要追溯历史时读取 `docs/工作记录/00-目录与映射.md` 和对应日期记录

不要把某台电脑上的绝对 `D:/...` 路径作为仓库启动前提。仓库应能够整体迁移到另一台机器后继续阅读和维护。

## 当前需要避免的误判

- 正式 NIR 全量分析已经完成，不要把项目重新描述成“等待 ROI 准入”或“准备正式全量”。
- `configs/formal.yaml` 是历史 full-face ROI 候选阶段的兼容配置，不代表当前项目状态。
- `src/attention_pipeline/cli.py` 是较早统一 CLI，不是当前已核验的 YOLO26n + tracking + RITnet 完整实现入口。
- 已核验的 portable 完整实现位于 `runtime/nir-yolo-tracking-ritnet-v1/run_pipeline.py`。
- portable package 创建时的默认 tracker / redetect / ROI expansion 参数不能在缺少最终 full-run manifest 时自动写成全量最终冻结参数。

## 外部项目记忆

如果当前设备另有厚璨杯统一项目记忆或其他仓库外上下文，可以作为补充材料读取；但它们不是本仓库可移植性的硬依赖。仓库内事实应能够通过 `AGENTS.md`、README、总览、docs 和工作记录独立追溯。

## 编辑约束

删除、以重命名为名删除旧路径、删除字节级重复副本等操作仍遵循 `AGENTS.md`：必须先取得用户明确同意。当前已知待确认项包括根目录 `venv-labelimg/` 和工作记录中的精确重复副本；在获得批准前保持原文件存在。

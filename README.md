# Attention-Analysis

> GitHub 仓库当前仍使用历史名称 `attention-pipeline-v2`；项目文档统一使用 `Attention-Analysis`。

## 2026-08-23 当前项目状态

- NIR 眼框 YOLO26n 已完成 100 epochs 训练；训练产物保留在 `training/nir-eye-yolo/runs/yolo26n_eye_100epoch/`，`best.pt` 已复制进入正式 runtime。
- 正式 NIR pipeline 已完成全量运行，不再处于“准备正式分析 / 等待全量推理”的阶段。
- 当前正式 runtime 为 `runtime/nir-formal/`；正式模式采用逐帧 YOLO，tracking 仅保留用于诊断和历史复现。
- 当前正式配置以 FocusWave v3.1.3 时间语义为准，分析 `baseline / instructions / practice / block1 / block2`；RITnet 使用 batch 推理，当前配置为 `batch_size=16`、`fp32`。
- 当前工作重点已经转为：仓库结构整理、正式资产归档、历史记录保留与可复现性维护。
- 结构整理以 `main` 为当前工作主线；原 `codex/v2-YOLO+Tracking+RInet` 与 `codex/nir-formal-gpu-v3` 分支暂时保留，不删除。

当前入口：

- [项目总览与架构](项目总览与架构.md)
- [文档目录](docs/README.md)
- [NIR 正式分析当前入口与资产映射](docs/010-nir/08-23-01-NIR正式分析当前入口与资产映射.md)
- [正式 NIR runtime](runtime/nir-formal/README.md)
- [仓库工作规则](AGENTS.md)

> 下方 2026-08-21 及更早内容属于当时的阶段性记录。为保留研究过程，不删除、不改写其历史语境；判断当前状态时以上述 2026-08-23 状态为准。

## 2026-08-21 历史状态

> 2026-08-21 17:30（Asia/Shanghai）｜北京已完成一百多名正式被试的数据采集；预计 2026-08-23 晚回珠海后启动正式分析。

- NIR 眼框数据集 v1 与 YOLO26n 100 epochs 训练已完成；冻结 test 为 7 名被试、85 张图、169 个眼框，mAP50=0.9913、mAP50-95=0.6589。`best.pt` 已进入静态 test 入围，但尚未通过正式视频生产准入。
- 正式视频处理路线暂定为：`OpenCV 读取视频 → YOLO 周期性重新检测 → tracking 更新中间帧 ROI → RITnet 处理眼睛 ROI → 时序质量控制与指标提取`。
- RX 6750 GRE 电脑当前不使用 PyTorch GPU 加速；训练完成后，计划视条件将权重和推理环境转移至 NVIDIA 电脑，处理一百多个约 25 分钟视频。
- tracking 算法、重新检测间隔和 RITnet 参数仍需在珠海用短视频冒烟测试确定；在此之前不运行全量正式提取。
- 当时已生成 `runtime/nir-yolo-tracking-ritnet-v1.zip`，用于 GPU 电脑的 YOLO26n + CSRT/KCF + RITnet 真实短视频联调；该历史压缩包已在 2026-08-23 仓库整理时删除，历史事实保留在本段。

> 08-16（Asia/Shanghai）｜NIR高严重度修复、528眼轴角复核和历史连续序列复核已完成；正式三种人脸ROI在双眼特写上均未通过身份硬门，生产冻结按计划停止。

跨仓库项目状态与研究决策以厚璨杯统一项目记忆为准；本 README 维护当前仓库入口和运行状态。

脚本入口仍见 `scripts/00-目录与映射.md`；历史工作记录入口仍见 `docs/工作记录/00-目录与映射.md`。这两个目录暂按历史引用兼容规则保留原入口文件名。

## 历史 NIR 结论（08-16 阶段）

- 椭圆轴角已修复并测试；历史阶段4/4b重新运行后，六算法和18项调优仍全部未达准入门槛，最佳约17.2%。
- 新PuReST适配层历史复跑：可见覆盖0.5230、恢复192 ms，连续性仍优于PuRe；这不是正式准确率。
- 正式sub-011 Block1是双眼/鼻梁特写，不是完整人脸。60时点中MediaPipe、YuNet、当时YOLO-face正确双眼ROI均为0。
- 该阶段因ROI硬门失败，尚未冻结正式`minPx/maxPx`、门控、模型、runtime或`scripts/run_nir_pipeline.py`；此结论后来被 YOLO26n 眼框 + RITnet 正式 runtime 路线取代。
- RGB、跨模态与专注评分接口继续关闭。

详细记录：

- [阶段4与5修复复核](docs/工作记录/08-16-04-NIR阶段4与5修复复核工作记录.md)
- [正式ROI入围检查](docs/工作记录/08-16-05-NIR正式ROI入围检查工作记录.md)

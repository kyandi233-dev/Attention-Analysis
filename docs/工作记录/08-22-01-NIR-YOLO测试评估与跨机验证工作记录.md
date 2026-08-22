# 08-22-01｜NIR YOLO测试评估与跨机验证工作记录

> 2026-08-22（Asia/Shanghai）｜归档 YOLO26n 眼框模型的本机 test 评价与其他电脑短视频、tracking、RITnet 联调计划；当前只评价 `best.pt`，不启动正式视频全量提取。

## 目录

- [总结](#总结)
- [计划](#计划)
- [执行决策过程](#执行决策过程)
- [最终决策结果概览](#最终决策结果概览)

## 总结

本记录把当前 YOLO26n 眼框模型分为两个地点、三个阶段：本电脑完成冻结 test 集总体指标和逐被试统计；其他电脑完成短视频 YOLO/tracking 基准、ROI 扩展与 RITnet 联调、全量运行前准入。YOLO 只负责眼睛 ROI 定位，不能替代瞳孔测量验证。

## 计划

### 一、计划归档

- 本记录保留完整计划；执行结果只追加，不删除计划内容。
- 同步更新工作记录索引和 NIR 目录入口。
- 当前边界：只评价 `yolotrain/runs/yolo26n_eye_100epoch/weights/best.pt`；不训练对照模型、不修改 test 标签、不启动正式视频全量提取。

### 二、本电脑：test 总体与逐被试评价

新增 `scripts/evaluate_yolo_eye_test.py`，使用 `D:/Code/python/python.exe` 和现有 Ultralytics 环境。先在 val 选择 IoU 0.50 下 F1 最大的运行阈值（并列取较高阈值），再冻结阈值评价 test；AP 使用 Ultralytics 原生置信度扫描指标。

输入核验必须记录模型类别、SHA256、环境、被试级 split、图片—标签配对和 `sub-055` 的单框样本。test 预期为 7 名被试、85 张图片、169 个标注框。

test 输出总体和逐被试 precision、recall、F1、mAP50、mAP50-95、TP/FP/FN、IoU、中心误差、双眼成功率和失败样本表。匹配为按置信度排序的一对一匹配，不强制每张图产生两个预测框。

完成 val 冒烟、匹配逻辑测试和 test 完整评价后停止本机工作，不接 tracking、RITnet 或正式视频。

### 三、其他电脑：短视频与 tracking

迁移权重、哈希、评价报告、split、环境和脚本后，先做模型加载、单帧和视频读取检查。从 5 名不同被试的 10%、50%、90% 视频位置各取 20 秒，比较 YOLO 每帧、CSRT/KCF 和 5/10/15/30 帧重检测间隔。tracking 丢失、越界、身份交换、预测框数量异常或中心单帧跳变超过上一框宽度 50% 时立即重检测。

每段每 30 帧人工复核一帧。候选需达到人工可用率 ≥95%、严重裁断率 ≤2%；通过者中选 FPS 最高方案，全部不通过则保留 YOLO 每帧基线。

### 四、其他电脑：ROI 扩展与 RITnet

比较横向扩展 20%/30%/40% 与纵向扩展 30%/45%/60% 的 9 组组合，裁到边界后生成 320×160 单眼 ROI，再接现有 RITnet 640×400 输入封装。先满足 ROI 人工可用率 ≥95%、严重裁断率 ≤2%，再在通过者中选面积最小且 RITnet 有效率最高者。

`yolo_missing`、`single_eye`、`extra_boxes`、`tracker_lost`、`roi_clipped`、`pupil_invisible`、`ritnet_missing`、`ritnet_rejected`、`observed`、`interpolated` 必须分轨记录；准确率评价阶段不插值。

### 五、全量前准入

先运行一段完整约 25 分钟视频，再运行 3 名被试小批量试点，检查时间戳、眼别、失败状态、速度、磁盘、断点恢复和汇总字段。全部通过后才冻结模型哈希、阈值、imgsz、tracker、重检测间隔、ROI 扩展、RITnet 权重和质量门。全量一百多名被试仍需单独审批。

### 本机执行结果（2026-08-22）

本机使用 `D:/Code/python/python.exe`（Python 3.13、Ultralytics 8.4.120、Torch 2.12.1+cpu、CUDA unavailable）加载 `best.pt`。模型类别核验为 `{0: eye}`，模型文件 SHA256 与 `run_manifest.json` 一并保存。test 被试没有泄漏，输入为 7 名被试、85 张图片、169 个标注框；`sub-055_frame_046581` 的 1 个标注框按原始标签保留。

val 仅用于运行阈值选择，IoU 0.50 下 F1 最优阈值为 0.40。冻结该阈值后，test 一对一匹配结果为：TP=166、FP=8、FN=3，precision=0.9540、recall=0.9822、F1=0.9679；全部已标注眼框均找到的图片比例为 0.9647，真值双眼图片上的双眼成功率为 0.9643。匹配框 IoU 均值 0.8354、中位数 0.8441、P10 0.7437，中心误差相对真值框对角线中位数为 0.0251。

Ultralytics 原生 test 指标为 precision=0.9754、recall=0.9645、mAP50=0.9913、mAP50-95=0.6589。逐被试 AP 仅作描述：`sub-012` mAP50-95=0.6949，`sub-021`=0.5977，`sub-032`=0.6370，`sub-041`=0.7461，`sub-051`=0.6918，`sub-055`=0.6103，`sub-9504`=0.6953。静态 test 结果支持进入跨机短视频验证，但不构成视频生产准入。

失败索引共 8 张：`sub-021` 4 张、`sub-9504` 3 张、`sub-055` 1 张；具体逐图 TP/FP/FN 保存在 `artifacts/yolo-eye-evaluation/yolo26n_eye_100epoch/failure_index.csv`。全部评价产物包括 `overall_metrics.json/csv`、`per_image_predictions.csv`、`per_image_summary.csv`、`per_subject_metrics.csv`、`native_test/`、`native_subject/`、`per_subject_metrics.png` 和 `run_manifest.json`。

匹配逻辑测试 4 项通过，脚本通过 `py_compile`。完整仓库 `pytest -q` 未通过：失败主要来自当前机器不可用的 `E:/预实验`、`E:/正式实验` 外部数据和 `C:/Users/goven/AppData/Local/Temp/pytest-of-goven` 权限错误，另有既存正式协议断言不匹配；未见新增 YOLO 匹配测试失败。原始图片、标签、训练权重和训练 runs 未被修改；Ultralytics 运行产生的 `yolotrain/labels/test.cache` 已在完成后删除。

## 执行决策过程

> 用户：当前权重能否使用；希望把 YOLO 权重带到已配置 GPU 版 YOLO/RITnet 的电脑，并提供包含 tracking 的打包文件。正式数据分别位于 `F:/正式实验` 和 `E:/Data`，视频结构为 `sub-xxx_/nir/sub-xxx_nir.avi`。

落实：当前 `best.pt` 可进入 GPU 静态图和短视频试跑，但静态 test 不能代替视频生产准入。新增独立包 `runtime/nir-yolo-tracking-ritnet-v1/` 及同名 ZIP，内含 YOLO26n 权重、RITnet 权重/网络定义、CSRT/KCF、周期重检测、ROI 扩展、RITnet 推理、逐帧/逐眼状态、可视化、双根目录发现和 SHA256 清单。

> 用户后续要求：直接写代码，不需要创造视频验证。

落实：停止视频合成和运行；临时冒烟 AVI 与输出已删除，不作为交付证据。最终只保留代码、权重包、静态环境检查、`py_compile` 和 9 项针对性单元测试。另一台电脑仍须按 README 用真实正式短片完成视频验证。

> 用户：当前权重能否使用；希望把 YOLO 权重带到已配置 GPU 版 YOLO/RITnet 的电脑，并提供包含 tracking 的打包文件。正式数据分别位于 `F:/正式实验` 和 `E:/Data`，视频结构为 `sub-xxx_/nir/sub-xxx_nir.avi`。

落实：当前 `best.pt` 可进入 GPU 静态图和短视频试跑，但静态 test 不能代替视频生产准入。新增独立包 `runtime/nir-yolo-tracking-ritnet-v1/` 及同名 ZIP，内含 YOLO26n 权重、RITnet 权重/网络定义、CSRT/KCF、周期重检测、ROI 扩展、RITnet 推理、逐帧/逐眼状态、可视化、双根目录发现和 SHA256 清单。

> 用户后续要求：直接写代码，不需要创造视频验证。

落实：停止视频合成和运行；临时冒烟 AVI 与输出已删除，不作为交付证据。最终只保留代码、权重包、静态环境检查、`py_compile` 和 9 项针对性单元测试。另一台电脑仍须按 README 用真实正式短片完成视频验证。

> 用户：本电脑完成 1、2 部分，3–5 部分在其他电脑完成；当前只写计划并归档。

落实：本记录先归档计划；本机只运行静态 test 评价与逐被试统计。其他电脑部分保留为后续交接计划，不在本机执行。

> 数据审计：test 共 7 名被试、85 张图片、169 个眼框；`sub-055_frame_046581` 只有一个标注框。

落实：统计按真实标注数进行；“双眼成功率”只在真值确有两个眼框的图片上统计，不能把单框样本补成双眼。

## 最终决策结果概览

| 决策项 | 结果 | 依据 | 文件位置 |
|---|---|---|---|
| 本轮模型 | 只评价 yolo26n `best.pt` | 用户确认 | `yolotrain/runs/yolo26n_eye_100epoch/weights/best.pt` |
| 本机评价 | val 选阈值后完成 test 总体与逐被试评价 | 防止 test 调参 | `scripts/evaluate_yolo_eye_test.py` |
| 本机 test 结果 | TP=166、FP=8、FN=3；mAP50=0.9913、mAP50-95=0.6589 | 85 张图、169 个眼框、7 名独立 test 被试 | `artifacts/yolo-eye-evaluation/yolo26n_eye_100epoch/` |
| GPU 试跑包 | YOLO26n + CSRT/KCF + RITnet；支持 F盘/E盘两类正式路径 | 用户跨机分析需求 | `runtime/nir-yolo-tracking-ritnet-v1.zip` |
| 包验证 | 代码编译、环境检查、9 项针对性测试通过；未以视频运行作为最终验收 | 用户要求只写代码 | 包内 README、`tests/test_portable_nir_gpu_package.py` |
| GPU 试跑包 | YOLO26n + CSRT/KCF + RITnet；支持 F盘/E盘两类正式路径 | 用户跨机分析需求 | `runtime/nir-yolo-tracking-ritnet-v1.zip` |
| 包验证 | 代码编译、环境检查、9 项针对性测试通过；未以视频运行作为最终验收 | 用户要求只写代码 | 包内 README、`tests/test_portable_nir_gpu_package.py` |
| test 标签 | 只读，不修订 | 保持冻结测试集 | `yolotrain/labels/test/` |
| 跨机路线 | YOLO 每帧基线、tracking、ROI 扩展、RITnet、全量前试点 | 正式管线停止门 | 本记录计划第三至第五节 |
| 生产准入 | 当前不冻结 | 静态指标不能替代视频和下游验证 | 本记录计划第五节 |

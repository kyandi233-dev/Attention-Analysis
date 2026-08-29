# NIR 正式队列状态

更新时间：2026-08-30（本地只读质量审计）

## 当前状态

- 状态：COMPLETE（本批次已结束，当前无 NIR 分析进程）
- 正式队列：单 GPU、单 parent-child 链；未启动第二条 GPU 队列
- 正式环境：`D:\CondaEnvs\nir-nvidia\python.exe`
- Git 分支：`nvidia-cuda`
- Git commit：`39bd896`
- launcher commit：`015bff5`（Conda 固定解释器、CUDA DLL 自动注入、CUDA EP fail-closed）
- 输出根：`D:\Project\厚粲杯\11_数据\01_Attention-Analysis_nvidia-cuda_formal_NIR\ritnet-fullclass-final`
- 运行边界：未重跑 YOLO；未修改 RITnet、Topology、QC、metrics 或 schema

## 初步质量审计

- 完成 subject：65 个
- `completion.json`：65/65 为 `complete`
- `summary.json`：65/65 的 source eye rows 与输出 eye rows 一致；source frame rows 与输出 frame rows 一致
- QC 图：65/65 均为 80 张；无预算跳过
- pixel evidence：65/65 均为 16 条；无预算跳过
- 结果文件：眼动和帧覆盖压缩 CSV 均存在且可读取；抽查列结构正常
- 图片抽查：baseline、block1 的双眼 ROI/瞳孔叠加位置正常；闭眼样本被正确标记为 `not_detected`
- QC 中的 `temporal_jump`、`pupil_fragmented`、`single_eye` 等为图像质量事件记录，不等同于队列失败；下游分析仍需按 QC 规则筛选
- 结论边界：本记录是 producer/output 初步质量审计，不替代下游行为对齐、统计分析或最终科学解释

## 已完成 subject

| Subject | Status | Eye rows | Frame rows | QC | Pixel |
|---|---|---:|---:|---:|---:|
| sub-056 | complete | 73207 | 38903 | 80 | 16 |
| sub-057 | complete | 76298 | 38704 | 80 | 16 |
| sub-058 | complete | 81594 | 41058 | 80 | 16 |
| sub-059 | complete | 77725 | 39411 | 80 | 16 |
| sub-062 | complete | 78419 | 39482 | 80 | 16 |
| sub-064 | complete | 75145 | 38542 | 80 | 16 |
| sub-065 | complete | 82106 | 41527 | 80 | 16 |
| sub-067 | complete | 82529 | 41759 | 80 | 16 |
| sub-068 | complete | 76304 | 38964 | 80 | 16 |
| sub-070 | complete | 88718 | 45138 | 80 | 16 |
| sub-071 | complete | 82999 | 42537 | 80 | 16 |
| sub-072 | complete | 85061 | 43343 | 80 | 16 |
| sub-073 | complete | 79106 | 41099 | 80 | 16 |
| sub-074 | complete | 80061 | 40321 | 80 | 16 |
| sub-075 | complete | 79350 | 40200 | 80 | 16 |
| sub-076 | complete | 75140 | 38244 | 80 | 16 |
| sub-077 | complete | 82805 | 42124 | 80 | 16 |
| sub-078 | complete | 81340 | 41809 | 80 | 16 |
| sub-081 | complete | 76155 | 38801 | 80 | 16 |
| sub-082 | complete | 78140 | 39888 | 80 | 16 |
| sub-083 | complete | 61490 | 39875 | 80 | 16 |
| sub-084 | complete | 54800 | 41643 | 80 | 16 |
| sub-085 | complete | 79383 | 39914 | 80 | 16 |
| sub-093 | complete | 85636 | 43119 | 80 | 16 |
| sub-094 | complete | 84142 | 43186 | 80 | 16 |
| sub-095 | complete | 81364 | 41248 | 80 | 16 |
| sub-096 | complete | 84841 | 43164 | 80 | 16 |
| sub-098 | complete | 79152 | 40011 | 80 | 16 |
| sub-100 | complete | 68180 | 39205 | 80 | 16 |
| sub-104 | complete | 82245 | 41639 | 80 | 16 |
| sub-106 | complete | 76585 | 38749 | 80 | 16 |
| sub-107 | complete | 83588 | 42570 | 80 | 16 |
| sub-108 | complete | 78345 | 39876 | 80 | 16 |
| sub-109 | complete | 76968 | 39035 | 80 | 16 |
| sub-110 | complete | 77577 | 39463 | 80 | 16 |
| sub-114 | complete | 79448 | 40151 | 80 | 16 |
| sub-116 | complete | 78896 | 41088 | 80 | 16 |
| sub-117 | complete | 78525 | 40571 | 80 | 16 |
| sub-118 | complete | 74480 | 39463 | 80 | 16 |
| sub-119 | complete | 79896 | 41118 | 80 | 16 |
| sub-122 | complete | 78660 | 41309 | 80 | 16 |
| sub-123 | complete | 81992 | 41523 | 80 | 16 |
| sub-124 | complete | 76675 | 39261 | 80 | 16 |
| sub-125 | complete | 79656 | 40372 | 80 | 16 |
| sub-126 | complete | 74913 | 38242 | 80 | 16 |
| sub-127 | complete | 78129 | 40077 | 80 | 16 |
| sub-128 | complete | 77362 | 39294 | 80 | 16 |
| sub-129 | complete | 78091 | 40402 | 80 | 16 |
| sub-130 | complete | 84060 | 42599 | 80 | 16 |
| sub-131 | complete | 76324 | 38681 | 80 | 16 |
| sub-133 | complete | 75495 | 38788 | 80 | 16 |
| sub-134 | complete | 63139 | 40833 | 80 | 16 |
| sub-139 | complete | 80098 | 40742 | 80 | 16 |
| sub-143 | complete | 77361 | 38960 | 80 | 16 |
| sub-145 | complete | 76300 | 39321 | 80 | 16 |
| sub-147 | complete | 76179 | 39233 | 80 | 16 |
| sub-148 | complete | 75016 | 38583 | 80 | 16 |
| sub-154 | complete | 77035 | 39862 | 80 | 16 |
| sub-158 | complete | 76137 | 38655 | 80 | 16 |
| sub-160 | complete | 75286 | 38285 | 80 | 16 |
| sub-162 | complete | 77284 | 39223 | 80 | 16 |
| sub-166 | complete | 76665 | 38910 | 80 | 16 |
| sub-170 | complete | 78367 | 39315 | 80 | 16 |
| sub-175 | complete | 75478 | 38475 | 80 | 16 |
| sub-178 | complete | 84810 | 43080 | 80 | 16 |

## 队列故障记录

此前批处理退出记录保留在本地 `_runtime_logs`：

- `conda_formal_resume_20260829_111726.err.log`：PowerShell 在重定向环境中设置控制台标题时出现 Win32 `0xE9`
- `conda_formal_queue_20260829_094228.err.log`：旧工作树清洁门控和 GBK 解码错误
- `ritnet_fullclass_batch_summary.json`：历史失败项返回 `3221225794`（`0xC0000142`，Windows 进程/DLL 初始化层），不是已完成结果的 completion 失败

这些历史错误不改变本次 65 个正式输出的 completion/QC 审计结果。

# AMD 与 NVIDIA 双设备 NIR 加速及方法评估计划

> 2026-08-24（Asia/Shanghai）｜本记录归档用户确认的执行计划。当前先记录背景、目标、步骤、风险与校验；执行结果和最终决策在任务完成后追加，不追改既有工作记录。

## 背景与当前事实

用户要求在 2026-08-26 00:00 前尽量完成 114 名正式被试的 NIR 分析，并明确排除 `sub-9504`。当前 AMD/DirectML 管线已完成 600 帧短测，但正式完整输出为 0 人；用户同时要求修复正式完成判定、寻找兼顾瞳孔、虹膜与眼睑闭合度的更快方法，并把优化版本同步到 NVIDIA 分支，以便多设备分开运行后安全合并。

当前只挂载 `E:/正式实验`。排除 `sub-9504` 后，实测可发现 44 名有效被试，五个正式 phase 共 1,805,423 帧；114 人总量暂按这 44 人的均值外推为约 4,677,687 帧，待另一块数据盘挂载后重新核验。AMD 最新有效 smoke 为 600 帧、44.656 秒、13.436 FPS；历史 NVIDIA RTX 4060 Laptop 同类完整管线约 16.48 FPS。旧方法两机合计约 29.9 FPS，仍低于带运行余量的截止需求。

性能分解显示 AMD smoke 中 RITnet 约占 wall time 68.48%，YOLO 约占 21.15%，视频解码约占 9.25%。因此不能只凭候选模型参数量或论文 FPS 断言提速；必须在同一真实 ROI 和同一端到端片段上实测。

## 总体目标

1. 先修复 AMD 与 NVIDIA 两端共同存在的完成判定漏洞，避免读帧失败、部分输出或 smoke 被误认为正式完成。
2. 显式排除 `sub-9504`，保持 FocusWave v3.1.3 五个 phase、YOLO `conf=0.40`、ROI 320×160、FP32、缺失/QC 语义和既有 CSV 主 schema。
3. 将当前 compact RITnet/YOLO 优化能力同步为 NVIDIA 可选高速 profile，同时保留 NVIDIA 原 `.pt/.pkl` CUDA 复现入口。
4. 在 Git 仓库外短测 RITnet 完整四分类后处理、3DeepVOG、Worldcoin open-iris、EllSeg、PupilEXT/pupil-detectors 和 Pistol 外部对照。
5. AMD 与 NVIDIA 允许使用各自最快、但经过跨后端等价验证的 FP32 实现；按被试分片并安全合并，禁止同一被试中途混用后端。
6. 输出 pupil、iris、连续 closure、`open/partial/closed/unknown`、blink events、有效覆盖率和 PERCLOS80；unknown 不插值、不当作 closed。

## 阶段一：公共完整性修复与版本发布

### AMD `0.1.1`

- 新增后端无关 `formal_completion.py`，原子维护 `completion.json`。
- marker 状态为 `running / complete / failed / smoke_complete`；batch 只接受通过身份、计数和产物校验的 `complete`。
- 正式成功必须无 `--max-frames`、无读帧失败，实际 `(phase, segment, frame_idx)` 集合与 phase windows 完全一致，decoded/processed/expected 计数一致，CSV、summary、manifest 和 phase windows 均可解析。
- `cap.read()` 失败保留诊断输出，写 `failed` 并退出码 3；异常退出码 1；smoke 永不触发正式 skip。
- batch 在跳过前和子进程返回 0 后使用同一验证器，不再以 `summary.json` 存在作为完成依据。
- `batch.subjects.exclude` 增加 `sub-9504`；当前 E 盘 dry-run 必须得到 44 人。
- package 升为 `0.1.1`，测试后提交、推送并创建 `amd-v0.1.1`；已有 `amd-v0.1` 不移动。

### NVIDIA `1.0.1`

- 远端 `nvidia-cuda` 权威基线为 `e63675ad15c17db6ea2ac7a3bb1c1ac6fc106e06`。本地 remote-tracking ref 当前陈旧，实施时先 fetch，并从远端精确提交建立独立干净 worktree。
- 在 `e63675a` 补建 annotated tag `nvidia-v1.0.0`，再手工移植公共 marker、读帧失败退出、batch 复核、`sub-9504` 排除和共同测试。
- 不整文件覆盖 CUDA 初始化；继续保留 NVIDIA `.pt/.pkl`、Ultralytics/PyTorch CUDA、CUDA 环境检查和历史复现入口。
- package 升为 `1.0.1`，提交、推送并创建 `nvidia-v1.0.1`。

本阶段不加入帧级 checkpoint/resume。历史工作记录中“最后一批 3 个真实 ROI + 13 个补位”的解释不追改；在本记录执行部分新增纠偏：实际 smoke 的 13 个补位分布在多个未满批，最后一批为 10 个真实 ROI + 6 个补位，正式输出行数本身正确。

## 阶段二：同步 NVIDIA 高速 profile

- 将 AMD 已冻结的 YOLO ONNX 与 compact RITnet ONNX 作为 NVIDIA 新增的 ORT CUDA profile，不删除原 PyTorch 模型。
- 在同一短片比较 `pytorch-cuda-fp32-b16` 与 `ort-cuda-fp32-b16`；仅当 TensorRT FP32 已安装且无需额外部署时才增加对照。
- 本轮禁止 FP16、AMP 和 TF32。PyTorch 关闭 TF32；ORT CUDA 设置 `use_tf32=0`。
- “最快”只在通过跨后端 parity 的 profile 中选择。若 native PyTorch CUDA 不等价，优先回退到与 AMD 同一 ONNX 的 ORT CUDA。
- 当前 RITnet 四分类 label map 增加 iris 椭圆、可见眼区和 closure/unknown 后处理，作为不增加第二次神经网络推理的候选 0。

## 阶段三：Git 外候选短测

试验统一写入 `D:/AttentionModels/nir-candidate-bench-20260824`。AMD 使用现有 `D:/CondaEnvs/nir-amd` 和 `D:/CondaPkgs`；需要 PyTorch/MONAI 时建立独立临时环境，不污染正式 runtime。

### 固定数据

- 性能段：`sub-031` 正式 `block1` 前 600 帧（20 秒）。
- 热稳定段：胜者追加 1,800 帧（60 秒）。
- 正式质量集：`sub-031、042、053、064、074`，从 timestamp 解析的 block1/block2 共抽 120 帧。
- 压力集：旧 `nir-ellipse-review-20260813` 的 132 张图；其人工标签文件为空，只作图像来源，重新标注 `open/partial/closed/unknown` 和 pupil/iris 是否可测。
- 至少覆盖 30 个半闭、闭眼或 unknown 样本，以及眼镜、反光、模糊、极端视线和 ROI 截断。

### 候选顺序

1. 当前 RITnet 完整四分类后处理。
2. [3DeepVOG v2](https://github.com/DSGZ-MotionLab/3DeepVOG)：同一官方 checkpoint 导出 raw-logit FP32 ONNX。
3. [Worldcoin open-iris](https://github.com/worldcoin/open-iris/blob/main/SEMSEG_MODEL_CARD.md)：使用官方 ONNX，重点检查闭眼时是否产生幻觉式完整掩膜。
4. [EllSeg](https://github.com/RSKothari/EllSeg)：只作 pupil/iris 几何参考。
5. [PupilEXT](https://github.com/openPupil/Open-PupilEXT) 与 [pupil-detectors](https://github.com/pupil-labs/pupil-detectors)：只作 pupil-only 速度/QC 基线。
6. Pistol：完成来源、SHA、签名与 Defender 检查后，仅在 Git 外跑 20 秒 CPU 黑盒对照，不提交或重分发。

### 准入标准

- 每个 GPU profile 热身 20 次、计时 100 次，记录预处理、GPU 执行/回传、后处理、batch median/p95 和端到端 FPS。
- pupil 人工可用率至少 90%，iris 至少 85%，closed F1 至少 0.90，无眼/错误 ROI 被误判 closed 不超过 5%，pupil 可用率不得比当前 RITnet 低超过 2 个百分点。
- 多机速度门为：`所有合格 worker 的 FPS 之和 >= 剩余帧数 / 截止前剩余秒数 * 1.10`。
- 第一位同时通过质量、parity 和速度门的候选立即进入正式集成，不等待所有次要黑盒对照完成。

## 阶段四：跨后端等价验证

用户选择“各自最快后端”，但两端必须冻结同一源 checkpoint、FP32、batch=16、尾批补位/丢弃规则、resize/归一化、左右眼规则和共享 CPU 后处理。

在 256–500 个分层真实 ROI 上校验：

- 输入 tensor 逐元素一致；
- raw logits 目标 `atol=1e-5, rtol=1e-4`；
- mask Dice 至少 0.999；
- pupil/iris 中心和直径差 p95 不超过 0.5 px；
- closure 差 p95 不超过 0.5 个百分点；
- valid、missing、open、closed、unknown 决策一致；
- blink 事件数一致，起止最多相差 1 帧；
- 同一 ROI 位于不同 batch slot 和尾批补位组合时结果不变；
- 同一后端重复运行 5 次稳定。

YOLO 必须保证检测数量、左右眼选择一致，匹配框坐标 p95 差不超过 0.5 px。阈值附近若存在微小数值漂移，使用预先冻结的统一 deadband 并标记 unknown，不能按后端分别调阈值。

ORT 两端同时设置创建期 `session.disable_cpu_ep_fallback=1`、只注册目标 EP，并调用运行期 `disable_fallback()`；不支持的节点必须使 session 创建失败。PyTorch CUDA 启动时断言模型与 tensor 均在 CUDA，异常直接失败。

## 阶段五：双设备 campaign 与安全合并

- 两台机器分别导出只读 `inventory-<worker>.json`，汇总后必须恰好得到 114 名有效被试，且不含 `sub-9504`。
- 同一 subject 在多处出现时不静默选择：相同指纹记录 replica 并明确 canonical location；不同指纹直接中止。
- `source_id` 由 subject、视频大小/帧数/首尾块指纹、timestamps/timeline/practice SHA 和 phase windows SHA 构成，不包含 E/F 盘符。
- `method_id` 由模型/checkpoint SHA、预后处理版本、科研配置、schema 和已通过 parity 的实现版本构成；`run_id = hash(method_id + source_id + phase_windows)`；重试只改变 `attempt_id`。
- 生成不可变 `campaign_manifest.json`，记录截止时间、114 人 roster、模型/config/环境哈希、worker GPU/profile/FPS、每人 expected frames 和 assignment generation。
- 按 worker 实测 FPS 做 weighted LPT 分片，并按数据卷和 subject number 分层；一个 subject 的五个 phase 全部由同一 worker 处理。
- `run_formal_batch.py` 增加 `--campaign` 与 `--worker-id`。每个 GPU 同时只运行一个 worker，并写入独立本地根；AMD/NVIDIA 不共享 subject 目录或 batch summary。
- 每个 attempt 先写 staging。只有产物、frame key、schema、哈希和 run identity 全部通过后，才原子写 `completion.json: complete`。
- 失败被试整人重跑，不拼接半段 CSV；重派生成新的 assignment generation，旧失败 attempt 保留诊断但不进入 merge。
- 汇总机复制到 `incoming/<worker>/<attempt>`，验证每文件哈希，确保 114 个预期 run_id 各有且仅有一个被接受的 complete attempt，再生成 `merge_manifest.json` 和派生总表。

## 正式输出接口

现有 `eyes.csv` 列和顺序保留，并追加：

- `segmentation_model/status/backend/device/batch_size`；
- iris center、axes、angle、mask area；
- `pupil_visibility_ratio`；
- `closure_fraction`；
- `eye_state=open|partial|closed|unknown`；
- `quality_status`。

新增 `blinks.csv`，记录 subject、eye、phase、起止 frame/time、duration 和 `blink/prolonged_closure`；新增 phase 级有效观测覆盖率、blink count/rate、持续时间分布和 PERCLOS80。PERCLOS 以每眼 baseline 的有效睁眼水平归一化，只以有效观测时间为分母，unknown 不插值、不计 closed。

阈值在 `sub-031、042、053` 上选择，在 `sub-064、074` 上冻结验证。Zotero 文献用于规范 pupil、blink rate、PERCLOS 和覆盖率的分别报告，但不把单一眼部指标解释成确定疲劳状态。

## 分支与版本

- 完整性修复：AMD `0.1.1` / `amd-v0.1.1`；NVIDIA `1.0.1` / `nvidia-v1.0.1`。
- 若扩展 RITnet 胜出：AMD `amd-DirectML` 升为 `0.2.0`；NVIDIA `nvidia-cuda` 升为 `1.1.0`。
- 若 3DeepVOG 胜出：建立 `amd-3deepvog` 与 `nvidia-3deepvog` 测试分支；NVIDIA 验证后合并到 `nvidia-cuda`，使用方法名明确的 tags。
- Worldcoin 胜出时同理使用 `amd-openiris` / `nvidia-openiris`。

## 风险与停止条件

1. 当前完整 114 人帧数仍含 70 人外推；第二盘挂载后必须重新冻结真实 inventory 和 ETA。
2. 3DeepVOG 论文速度来自 RTX 4090，不能外推到本机 DirectML；只有本机实测有效。
3. 各自最快后端可能产生数值差异；未通过 parity 的实现不能共享 method_id 或混入同一正式结果。
4. 第二盘未及时挂载、跨机数据根重复或冲突会阻止冻结完整 campaign。
5. 如果所有合格 worker 的总吞吐低于动态截止门，则明确报告现有硬件无法按全帧五 phase 口径如期完成，不通过降采样、删 phase 或错误状态映射伪装完成。

## 校验清单

- 公共单元测试、fake VideoCapture 读失败、无效 marker、summary-only、sub-9504 排除。
- AMD 与 NVIDIA `check-env`、no-CPU-fallback、模型和依赖哈希。
- 600 帧端到端 parity 与速度；胜者 1,800 帧热稳定。
- batch slot、尾批补位、重复运行稳定性。
- campaign inventory、跨盘 duplicate、weighted assignment、失败重派和 merge validator。
- 全量 114 人 completion、frame key、schema、主键、method_id 和分层 QC。

## 执行状态

- 已完成：计划确认与本记录归档；AMD `0.1.1` 完成性修复的实现与本机验证。
- 进行中：AMD `0.1.1` 提交、推送与 tag 发布。
- 未完成：NVIDIA 同步、候选短测、跨后端 parity、campaign 分片、正式全量运行与合并。

## AMD 0.1.1 执行与决策过程

1. 新增 `formal_completion.py`，以同目录临时文件、`fsync` 和 `os.replace` 原子发布 `completion.json`。完成校验不信任子进程返回码或 `summary.json` 单文件，会核对运行身份、phase windows、CSV 帧主键集合、计数、manifest 和必需产物。
2. 任何 `--max-frames` 或非完整 phase 运行均只能得到 `smoke_complete`，包括帧数上限大于实际阶段帧数的边界情况。读帧失败写 `failed` 并返回 3；产物不完整返回 4。
3. batch 在跳过旧运行前、以及子进程返回 0 之后都运行同一严格校验器；无合法 marker 时即使子进程返回 0 也会记为失败。`sub-9504` 已写入默认排除表，CLI 显式选择也不绕过排除。
4. 定向回归 18 项全部通过，包含 DirectML 不可用时拒绝 CPU fallback、RITnet 尾批补位与丢弃补位输出、YOLO `conf=0.40`、严格完成 marker 和排除被试。全仓测试的其余失败来自当前分支已不存在的历史脚本和未挂载的旧预实验数据，与本次修复无关。
5. 使用 `D:/CondaEnvs/nir-amd` 真实运行 `sub-031` 300 帧 smoke。phase windows 从时间戳解析出 baseline 起点为视频第 2145 帧，300 帧用时 21.553 秒，端到端 13.919 FPS；处理 300/300 帧、无读帧失败，marker 为 `smoke_complete`，smoke 校验通过而正式完成校验按预期拒绝。输出位于仓库外 `D:/AttentionModels/nir-amd-validation-20260824/amd-directml/`。

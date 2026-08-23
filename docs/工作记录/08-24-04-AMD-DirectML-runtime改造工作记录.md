# AMD/DirectML runtime 改造工作记录

> 2026-08-24（Asia/Shanghai）｜本记录用于跟踪 `amd-DirectML` 分支从 NVIDIA/CUDA 冻结基线到 ONNX Runtime DirectML 运行时的实际改造、校验与发布过程。

## 总结

AMD runtime 代码、模型、配置、测试与当前文档已完成改造，等待 Git 提交、远端推送和 `amd-v0.1` tag 发布。实现使用 ONNX Runtime DirectML 执行 YOLO26n 和 RITnet，强制 RITnet FP32 batch=16，尾批复制最后一个真实 ROI 补位并丢弃补位输出，默认输出进入 `amd-directml` 隔离层。

RX 6750 GRE 上的 `sub-031` block1 轻量端到端验证已完成：20 秒视频、600 帧、1187 只眼；压缩 RITnet 输出后的最终版本用时 44.66 秒、13.44 FPS。其中 581 帧双眼、13 帧单眼、6 帧额外框；RITnet 1079 个 observed、108 个显式 missing。CSV 中 1187 行的 `ritnet_batch_size` 全为 16；1187 对 16 余 3，证明尾批为 3 个真实 ROI + 13 个补位 slot，且 CSV 没有写入补位行。

## 原计划

### 背景

`amd-DirectML` 远端分支已从 `e63675ad15c17db6ea2ac7a3bb1c1ac6fc106e06` 创建，暂无新提交。NVIDIA 基线使用 Ultralytics/PyTorch CUDA 运行 YOLO26n 和 RITnet；本次只在 AMD 分支改为 ONNX Runtime DirectML，不回写 NVIDIA 复现口径。

### 目标

1. 将 YOLO 与 RITnet 的正式推理后端替换为 ONNX Runtime DirectML。
2. 冻结 YOLO ONNX 与 RITnet batch-16 FP32 ONNX 权重，移除本次已明确授权删除的 `.pt` 和 `.pkl` runtime 权重。
3. 强制 RITnet `batch=16` 和 FP32；尾批用真实 ROI 补位到 16，推理后丢弃补位输出。
4. DirectML 不可用时立即失败，不允许 CPU provider 成为静默 fallback。
5. 保持 FocusWave phase 语义、YOLO `conf=0.40`、ROI/FP32 口径和 CSV schema；AMD 输出目录增加 `amd-directml` 隔离层。
6. 将 package version 设为 `0.1.0`，测试后提交、推送并创建 `amd-v0.1` tag。

### 步骤

1. 从远端独立克隆并核验分支起点，再将当前干净工作区切换到跟踪 `origin/amd-DirectML`。
2. 审查 runtime 推理、批处理、输出命名、环境检查、依赖和现有测试。
3. 准备并校验两个 ONNX 模型及 RITnet external data，然后改造后端适配层和正式流程。
4. 增补 DirectML provider、固定 batch/FP32、尾批补位、输出隔离、manifest 和 schema 回归测试。
5. 更新 AMD 分支的 README、INSTALL、配置、依赖、校验和权重哈希，不改写旧日期型工作记录。
6. 执行静态、单元、runtime 与可行的 DirectML 环境/短视频验证，记录未能在本机完成的硬件边界。
7. 复核 diff 与 Git 状态，提交、推送 `amd-DirectML`，创建并推送 annotated tag `amd-v0.1`。

### 风险

- ONNX 导出图的输入/输出布局、YOLO 后处理和 RITnet class axis 如与 PyTorch 基线不一致，可能造成静默数值偏差。
- RITnet external data 文件必须与 `.onnx` 一起存在且哈希固定，否则 session 无法加载。
- ONNX Runtime 可能在 DirectML provider 初始化失败后自动选用 CPU，必须通过 provider 列表、session provider 与禁用 fallback 多层校验阻断。
- 本机若无 AMD/DirectML 可用环境，可完成 provider-failure 、mock 和图约束测试，但真实 GPU 数值/性能验证需在 AMD 主机补做并明确记录。
- 大模型文件、GitHub 推送和 tag 需要确认远端文件上限、认证与无同名 tag 冲突。

### 校验

- 配置/资产：两个 ONNX 路径和 `.onnx.data` 存在，旧 `.pt`/`.pkl` 不存在，SHA256 清单与实际一致。
- 后端：只允许 `DmlExecutionProvider`，provider 不可用或 session 未实际使用 DML 时失败。
- RITnet：固定 NCHW FP32 `16×1×400×640`；尾批补位后只返回真实 ROI 数量。
- YOLO/ROI：固定 FP32 输入、`conf=0.40`、原有 bbox 选择和 ROI 逻辑保持。
- 输出：默认目录含 `amd-directml`，NVIDIA 历史目录不冲突；CSV 列名和 phase 相关字段不变。
- 版本/发布：package version `0.1.0`，工作区干净，远端分支指向新 commit，`amd-v0.1` 指向该发布 commit。

## 执行与决策过程

### 仓库与环境核验

1. 先在当前仓库同级独立 clone `amd-DirectML`，确认远端 HEAD 为用户给定的 `e63675ad15c17db6ea2ac7a3bb1c1ac6fc106e06`，然后才将当前干净工作区切换到该跟踪分支。
2. 初始环境核验误用了默认 `D:\Code\python\python.exe`，并向该环境安装了 `onnx 1.20.1`、`onnxruntime-directml 1.24.4`；用户随后明确指出已建好 `D:\CondaEnvs\nir-amd` 和 `D:\CondaPkgs`。此后所有真实 DirectML 加载、GPU 推理和视频验证都改用 `nir-amd`。默认 Python 中新增的两个包本次未擅自卸载。
3. 通过用户给定的 ChatGPT 会话核验已有结果：`nir-amd` 中 ORT 1.24.4 可见 `DmlExecutionProvider`，YOLO `[1,3,640,640] → [1,300,6]`，RITnet `[16,1,400,640] → [16,4,400,640]`，两者已完成独立 forward。

### 模型 provenance 纠正

`D:\AttentionModels\nir-eye-yolo26n-best.pt` 的 SHA256 为 `38cdbc9d34a022289b0efe34cb3670afe275025292ffd62ebc45611a3f99c227`，与 NVIDIA runtime/训练冻结权重 `004e98bccac26d528e6daa5e5fe56d6fb17502d603dd7f29807d754b1850799d` 不同。最初复制的 ONNX 是从前者导出，在正式视频上只给出约 0.017 置信度。

因此 YOLO ONNX 改为从 canonical 训练权重 `training/nir-eye-yolo/runs/yolo26n_eye_100epoch/weights/best.pt` 重新导出。新 ONNX SHA256 为 `e38cc13b589c186e5a60bd09d826034ae3dcf772d7ddf6fd402f69adb5fdd898`。在 `sub-031` block1 首帧与 PyTorch 冻结模型对照，两个检测框的最大坐标差为 0.533 px，最大 confidence 差为 0.000684。

RITnet 冻结资产为：

- `ritnet-b16-fp32.onnx`：`1933f44f483b350e17249a37b4a2ebe8b5e32f83fc8c1eb1a21c27e96477e621`；
- `ritnet-b16-fp32.onnx.data`：`1be9bc249f18998d0d28a7d759aff385471fa06ae8bfc05061f31f897ba5cef6`。

两个 ONNX 均通过 `onnx.checker` full check。RITnet 哈希在加入压缩输出节点并重新封装 external data 后冻结为上述值；网络参数和四分类 argmax 未变。AMD runtime 中已按用户明确授权删除 `nir-eye-yolo26n-best.pt` 和 `ritnet-best_model.pkl`。

### DirectML provider 决策

ORT 的 `session.disable_cpu_ep_fallback=1` 会使当前两个图因少量默认 CPU EP 节点而初始化失败。因此实现使用如下可核验边界：

- session 创建前必须存在 `DmlExecutionProvider`；
- 创建时 DML 必须排在 provider 列表第一位；
- 创建后再核验实际首选 provider；
- 调用 `session.disable_fallback()` 禁止后续运行时整体回退到 CPU。

这一边界禁止 DirectML 不可用时整个 session 静默变成纯 CPU；不虚假声称图中所有形状/控制节点都是 GPU。

### 代码与输出

- 新增 `directml_runtime.py`，统一 DirectML session 约束、YOLO letterbox、end-to-end `[x1,y1,x2,y2,score,class]` 解析和原图坐标恢复。
- `ritnet_runtime.py` 不再导入 PyTorch，固定 16 张 FP32 ROI 推理，只对真实 slot 执行后处理。
- `run_pipeline.py` 保持 phase、ROI、confidence=0.40 和 CSV row 字段，manifest 改记 DirectML providers 与 RITnet external-data SHA。
- 新增 `--max-frames` 轻量正式链路测试开关；输出名带 `_smokeN`，summary 显式记录 `truncated_for_smoke_test=true`。
- 默认输出和所有用户覆盖输出都经过 `amd-directml` 命名空间；batch dry-run 已确认不会重复追加该层。

### 轻量真实数据验证

1. 首先对 `sub-031` practice 运行 1206 帧，用时 27.55 s，但该窗口全部 `yolo_missing`。继续核查发现原因是错误 YOLO 权重 provenance，不是将“没有人脸”简化为算法成功。
2. 根据 `master_timeline.csv + nir_timestamps.csv` 解析 block1 真实起始帧 10186，重新导出正确 YOLO 后抽帧稳定检出双眼，confidence 约 0.90–0.93。
3. 正式链路只运行 block1 前 600 帧（20 s），产出位于 `D:\AttentionModels\pipeline-smoke-output\amd-directml\sub-031_formal_v3.1.3_yolo_b16_fp32_smoke600`，未写入 Git 仓库。
4. 平均分阶段耗时：decode 7.217 ms/frame，YOLO 16.376 ms/frame，ROI 0.269 ms/frame，RITnet attributed 63.220 ms/frame，overlay 0.059 ms/frame，总归因 87.142 ms/frame。当前主瓶颈是 RITnet 全分辨率 logits 传回与 CPU 后处理，不是 YOLO。

### 眨眼边界与 RITnet 后处理优化

用户补充指出 RITnet 的完整输出包含 background、sclera、iris、pupil 四类，当前代码只消费 pupil 类。复核后新增 `docs/020-nir/021-眨眼检测边界与RITnet派生开合度.md`：明确 `ocular = sclera ∪ iris ∪ pupil` 可以派生候选 aperture/openness，且不需要第二次 RITnet forward；同时明确完全闭眼与分割失败不可仅靠空 mask 区分，正式 blink/PERCLOS 仍需被试×眼别基线、open/closed/unknown 三态、时间戳时序规则和人工真值验证。本次必须保持原 CSV schema，因此不在 AMD `0.1.0` 中静默增加未经验证的 blink 列。

性能压力基准显示，固定 batch=16 的 RITnet 输出 logits 为 62.5 MiB；RX 6750 GRE 上 DirectML forward+回传中位约 143 ms，原先对全部像素执行 CPU dense softmax 约 150 ms。先验证了 CPU 稀疏 softmax可把该后处理降到约 76 ms；随后进一步在 ONNX 图内追加 ArgMax/Cast/Softmax/Gather/Squeeze，只回传 3.9 MiB UINT8 四分类 label map 与 15.6 MiB pupil probability map。一次尝试把 Reduce/Where 也放进图内，在首次 DirectML 执行时失败，且按设计没有 CPU fallback，因此未采用。最终折中图约 113 ms/batch，四分类 label 与原图完全一致，概率图最大绝对差约 `7.3e-12`。

同一 `sub-031` block1 前 600 帧最终复测为 44.66 s / 13.44 FPS；相比初版 52.92 s / 11.34 FPS，耗时缩短 15.6%、吞吐提高约 18.5%，RITnet 平均归因从 63.22 降到 50.97 ms/frame。1187 行除耗时字段外，仅 `pupil_confidence` 存在最大 `2.38e-7` 的 FP32 舍入差，其余科研字段一致。

### 自动校验

- `py_compile`：4 个 runtime 主要 Python 文件通过。
- 回归测试：29 passed，1 skipped；skip 原因为历史 preexperiment dataset 未挂载，与 AMD 改造无关。
- DirectML `check-env`：YOLO 和 RITnet 都以 `DmlExecutionProvider` 为首选 provider。
- batch dry-run：`sub-031`、FP32、batch=16、AMD 隔离输出和实际 E 盘视频路径均正确。

## 最终决策结果

1. AMD package 版本定为 `0.1.0`，不改写 NVIDIA `1.0.0` 分支。
2. YOLO 与 RITnet runtime 均只使用 ONNX Runtime DirectML；运行依赖不再包含 Ultralytics/PyTorch。
3. RITnet 不提供 batch 或 precision 比较开关；非 16 或非 FP32 立即报错。
4. `CPUExecutionProvider` 可作为 ORT 图节点后备 provider 存在，但 DML 必须存在并排第一，整体 session CPU fallback 被禁止。
5. AMD 输出必须经过 `amd-directml` 隔离层；smoke 运行不得与完整正式运行共用名称或 completion 语义。

## 已完成 / 未完成 / 待确认事项

- 已完成：克隆/分支起点、DirectML 后端、模型替换、旧 runtime 权重删除、固定 batch/FP32/尾批、输出隔离、文档、哈希、单元/回归测试、DirectML 硬件检查、batch dry-run 与 20 秒真实数据端到端验证。
- 未完成：Git commit、推送 `amd-DirectML`、创建/推送 `amd-v0.1`。
- 待本次收尾核验：更新 runtime SHA 清单并执行最终回归、Git commit、分支/tag 推送。
- 待后续独立研究：从完整四分类图验证 RITnet-derived openness、blink 与 PERCLOS；在人工时序验证和 schema/version 迁移前，不进入 AMD `0.1.0` 正式 CSV。

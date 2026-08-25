# 042｜面部分析工具与 Benchmark

## 1. 当前问题

RGB Face 分支当前比较 **Py-Feat Detectorv2** 与 **LibreFace 2.0**。目标不是比较“哪个模型给出的数字更大”，而是决定：在本项目正式 RGB 视频、Windows/AMD 环境和有限时间预算下，哪一个候选能够以更好的**输出覆盖、检测稳定性、速度和工程可部署性**完成全量分析。

没有人工 FACS ground truth，因此两个模型之间的一致性只能作为描述性参考，不能直接称为准确率。

## 2. Py-Feat Detectorv2

截至 2026-08 的官方 Py-Feat 主线，推荐新工作使用 `Detectorv2`。它是 multi-task 模型，可覆盖 20 个 AU、7 类 emotion、valence/arousal、gaze、6-DoF head pose、478-point 3D FaceMesh、blendshapes 等。Raw benchmark 必须尽量保留 Detectorv2 原生全部字段，而不是只挑 AU。

### sub-031 第一轮 benchmark 实测

共同输入集 350 张正式实验 JPEG，CPU / batch=8：

- Py-Feat 2.1.1 / Python 3.11.15 / Torch 2.13.0；
- 模型初始化约 51.2 s；
- inference 约 662.1 s（约 11.0 min）；
- 约 0.529 input images/s；
- 350 张输入产生 352 个 face rows；
- 输出 2,182 列；
- raw 原生字段完整保存在 `pyfeat_raw.parquet`。

`352 rows > 350 inputs` 本身不能直接解释为错误；Py-Feat 是 face-row 输出，最可能意味着少量 benchmark 图像检测到多个 face。新增 `face-pyfeat-qc` 必须逐图片映射确认 0/1/multi-face，而不能用总行数近似“检测率”。

运行：

```powershell
python scripts/rgb_analysis.py --stage face-pyfeat-qc --subject sub-031
```

输出：

```text
pyfeat_qc.json
pyfeat_qc_per_image.csv
```

QC 检查：逐输入 face count、各 phase coverage、FaceScore 分布、AU / emotion / valence-arousal / gaze / head-pose / FaceMesh 字段组可用性和缺失率，以及 multi-face 输入清单。第一轮 350 张离散图只能检验安装、覆盖、输出完整性、粗略鲁棒性和速度；**不能检验时间连续性或科学准确率**。候选缩小后再用连续短窗口检查 AU/head/gaze 的时序跳变。

## 3. LibreFace 2.0

LibreFace 主仓库已在 2026 年更新到 LibreFace 2.0。当前 Python inference 默认可返回 AU detection、AU intensity（0–5）、facial expression、gaze yaw/pitch，以及 alignment 阶段的 head pose / facial landmarks。LibreFace Python 官方安装说明仍推荐 Python 3.9，并明确要求系统中存在 CMake。

### Windows / dlib 安装注意

在 Windows 上直接 `pip install libreface` 可能触发 `dlib` 源码编译；如果系统 CMake/Visual C++ toolchain 不完整，会在 `Building wheel for dlib` 阶段失败。为了 benchmark 省时，优先在独立 Python 3.9 Conda 环境中安装 conda-forge 的预编译 CPU dlib，再用 pip 安装 LibreFace：

```powershell
conda activate "D:\CondaEnvs\attention-face-libreface"
conda install -c conda-forge dlib-cpu cmake -y
python -m pip install --upgrade libreface pandas pyarrow
```

随后检查：

```powershell
python -c "import dlib, libreface; print('dlib', dlib.__version__); print('libreface ok')"
```

若 `pip install libreface` 仍试图重新源码构建 dlib，再单独处理版本约束；不要先在主 `attention-rgb` 环境安装 Visual Studio/CMake/dlib 污染现有 Pose/Motion runtime。

LibreFace 2.0 还提供 ONNX derivative 路线，因此在 Windows AMD / DirectML 上具有较低的潜在移植门槛，但正式采用前仍必须验证 preprocessing、face alignment、输出命名和数值一致性。

## 4. 第一轮 benchmark：固定共同输入

第一轮从 `sub-031` 的正式时间窗提取固定、按 phase 分层的共同 JPEG 输入集：

```powershell
python scripts/rgb_analysis.py --stage face-sample --subject sub-031
```

抽样为 baseline 50、instructions 25、practice 35、transition 15、block1 90、interblock transition 45、block2 90，共 **350** 张。两候选必须使用完全相同输入。

输出：

```text
D:\_AttentionData\Beijing-RGB\_test\face-benchmark\sub-031\
├── frames\
├── sub-031_face-benchmark_frames.csv
└── sub-031_face-benchmark_manifest.json
```

## 5. Benchmark 比较什么

第一轮至少比较：

1. 安装/初始化是否稳定；
2. 输入图片级 face detection coverage；
3. 无脸/多人处理是否明确；
4. AU、head pose、gaze、expression/VA、landmarks 等实际字段覆盖；
5. Face confidence / alignment success；
6. 350-frame wall time 与吞吐；
7. 明显动作 spot-check；
8. AMD/DirectML 部署成本。

时间连续性只在候选缩小后用连续短窗口测试；没有人工 FACS ground truth 时，模型间一致性不等于准确率。

## 6. 信息保留规则

Face 是昂贵推理，必须遵循 `044-RGB输出Schema与信息保留原则.md`：第一次候选推理尽量保存原生全部可用字段；multi-face 不在 raw 层直接删掉；QC 先 flag；candidate/version/model/device/batch/source-frame manifest 必须记录。

## 7. 环境策略

不要为了 Face benchmark 破坏已经能工作的 RGB/Pose 环境：Py-Feat 使用独立 Python 3.11 环境；LibreFace 使用独立 Python 3.9 环境；两个候选读取同一批 benchmark 图。当前先比较官方 CPU reference backend，候选胜出但 CPU 不够快时才进入 ONNX + DirectML 验证。

# SETUP｜环境配置与可移植部署说明

## 2026-08-23 当前 NIR 路线

正式 NIR 全量分析已经完成。当前仓库中已核验的完整 YOLO26n + tracking + RITnet portable 实现位于：

```text
runtime/nir-yolo-tracking-ritnet-v1/
```

当前需要在另一台 GPU 电脑复现这条路线时，应优先使用该 package，而不是从下方 08-16 的“4 ROI × 4 pupil 算法候选环境”重新开始选型。

### 当前 portable package 内容

```text
runtime/nir-yolo-tracking-ritnet-v1/
├── README.md
├── config.yaml
├── run_pipeline.py
├── ritnet_runtime.py
├── run_examples.ps1
├── requirements.txt
├── SHA256SUMS.txt
├── models/
│   ├── nir-eye-yolo26n-best.pt
│   └── ritnet-best_model.pkl
└── ritnet/
    └── densenet.py
```

包内 YOLO 权重与 `yolotrain/runs/yolo26n_eye_100epoch/weights/best.pt` 为同一 Git blob；RITnet 权重与 `models/RITnet-master/best_model.pkl` 也为同一 Git blob。

### 当前环境依赖

package 自带 `requirements.txt`：

```text
ultralytics==8.4.120
opencv-contrib-python>=4.10
numpy>=1.26
pandas>=2.2
PyYAML>=6.0
torch
torchvision
```

GPU 机器需要安装与本机 CUDA/驱动匹配的 PyTorch。CSRT/KCF tracker 需要 `opencv-contrib-python`，普通 `opencv-python` 可能没有对应 tracker API。

### 当前运行入口

进入 package 目录后先检查环境：

```powershell
python .\run_pipeline.py check-env
python .\run_pipeline.py discover
```

检查项包括 CUDA、两个模型文件以及 CSRT/KCF 是否可用。运行脚本支持 `--subject` / `--video`、`--tracker none|csrt|kcf`、`--redetect-interval`、`--device`、`--skip-ritnet` 和 `--full-video`。

portable package 在 2026-08-22 创建时默认只跑 60 秒，并要求显式 `--full-video` 才处理整段。这是当时的准入保护，不代表代码只能处理短视频。

### provenance 注意

当前 Git 分支没有保存正式全量运行时最终生成的 `run_manifest.json`，因此 package 内默认配置只能用于复现 08-22 portable version，不能自动视为后来 full-run 的最终冻结参数。若要严格复现已经完成的那次全量分析，应优先从实际输出电脑找到当时的 `run_manifest.json`、`summary.json` 或运行命令。

---

# 历史兼容部署说明｜08-16 多后端/多算法阶段

> 08-16（Asia/Shanghai）｜本仓库当时封装了 4 个 ROI 后端 + 4 个瞳孔算法，目标是**自包含可移植**：把 v2 目录拷到另一台设备，按本文装好依赖即可跑全算法（本机不选最优，新设备跑完再选型）。以下内容保留用于历史路线复现，不再表示 2026-08-23 当前正式 NIR 主线。

## 1. 目录自包含说明

除**数据盘**外，所有代码、模型权重、配置均在 v2 目录内，相对路径自动 resolve 到 v2 根（`configs/formal.yaml` 的 `paths` 用 `models/...` 相对路径，`config.path_value` 解析到 `<v2>/models/`）。

唯一例外：`configs/formal.yaml` 的 `formal_data_root: "E:/正式实验"` 是数据盘绝对路径，**新设备需改成你的数据目录**。其余模型路径无需改动。

## 迁移到新设备（完整步骤）

**核心原则：Python venv / 全局环境不能直接拷贝**（`pyvenv.cfg`、`Scripts/python.exe` 的 shebang 都硬编码原机路径，拷过去会坏），必须在新设备重装。只有 v2 目录和数据盘可拷贝。

### 第 1 步：拷贝文件

| 东西 | 说明 |
|---|---|
| `attention-pipeline-v2/` 整目录 | 已自包含：代码 + 模型权重 + `runtime/`(wheel+依赖清单) + 配置 + 测试 |
| 数据盘 | `E:\正式实验`（+ 需要的预实验数据） |

### 第 2 步：装主环境（Python 3.13）

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r runtime/requirements-main.txt
```

### 第 3 步：装 venv-pupil（Python 3.10）

PyPupilEXT wheel 是 `cp310-win_amd64`，只能用 Python 3.10 + Windows 64 位：

```powershell
python3.10 -m venv venv-pupil
venv-pupil/Scripts/pip install runtime/PyPupilEXT-0.0.1-cp310-cp310-win_amd64.whl
venv-pupil/Scripts/pip install pupil-detectors==2.0.2
venv-pupil/Scripts/pip install numpy pandas opencv-python matplotlib
```

### 第 4 步：改配置（`configs/formal.yaml` 仅 3 处绝对路径）

1. `paths.formal_data_root` → 新数据盘路径
2. `runtimes.main_python` → 新主环境 python.exe
3. `runtimes.pypupilext_python` → 新 venv-pupil/Scripts/python.exe

其余模型路径是相对 `models/...`，`config.path_value` 自动解析到 v2 根，不用改。

### 第 5 步：验证

```powershell
$env:PYTHONPATH='src'
python -m pytest -q
python scripts/run_all_backends.py --subject sub-011 --roi-backends faceparts,mediapipe --pupil-algos PuReST,RITnet
```

### 迁移易错点

- PyPupilEXT 锁 Python 3.10（`cp310`），3.11+ 装不上；venv-pupil 必须用 3.10。
- ultralytics 用 ≥8.3（当时主环境 8.4.120），勿装 faceparts 仓库锁的 8.2.27。
- v2 建议放纯 ASCII 路径（如 `D:/attention-pipeline-v2`），规避 `cv2.dnn` 中文路径问题。
- DeepVOG 单独 venv，避免与 torch 环境混装。

## 2. 两个解释器 + 依赖

| 解释器 | 用途 | 必需依赖 |
|---|---|---|
| 主环境（Python 3.13+） | ROI 定位、RITnet、Iris、统计、测试 | 见下“主环境依赖” |
| venv-pupil（Python 3.10） | PyPupilEXT 的 PuRe/PuReST | 见下“pupil 环境依赖” |

### 主环境依赖

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install ultralytics onnxruntime opencv-python numpy pandas pyyaml
python -m pip install mediapipe
python -m pip install pytest
```

- `torch`：RITnet 推理。
- `ultralytics` ≥8.3：faceparts / YOLO 路线。
- `mediapipe`：`face_landmarker.task` 的 FaceLandmarker（ROI + Iris 关键点）。
- `onnxruntime`：YuNet / YOLO-face 的 ONNX 推理。
- 不需要 `supervision`；历史 faceparts 官方脚本使用它画框，本管线直接读 `result.boxes`。

### pupil 环境依赖（venv-pupil）

PuRe/PuReST 走 `venv-pupil/Scripts/python.exe`，依赖 PyPupilEXT（`pypupilext`）。此环境用于历史 PuRe/PuReST 路线复现；当前 YOLO + tracking + RITnet portable runtime 不要求 PyPupilEXT。

### DeepVOG 附加依赖（可选，历史比较）

DeepVOG 是 Keras/TensorFlow 时代代码，与主环境 torch 共存有冲突风险。建议单独 venv：

```powershell
python -m venv venv-deepvog
venv-deepvog/Scripts/pip install tensorflow==2.x keras==2.x scikit-image scikit-video numpy
```

`scripts/deepvog_pupil.py` 已做延迟导入：未装 keras 时 import 不报错，只有实际调用 `load_model()` 才失败。

## 3. 历史模型权重清单

| 模型 | v2 内路径 | 用途 |
|---|---|---|
| `face_landmarker.task` | `models/face_landmarker.task` | MediaPipe ROI + Iris |
| `yunet_2023mar.onnx` | `models/yunet_2023mar.onnx` | YuNet ROI |
| `yolov8n-face.onnx` | `models/yolov8n-face.onnx` | YOLO-face ROI |
| `yolov8n.pt` | `models/yolo-face-parts-detector-main/weights/yolov8n.pt` | faceparts ROI（nano） |
| `yolov8s.pt` | `models/yolo-face-parts-detector-main/weights/yolov8s.pt` | faceparts ROI（small） |
| `best_model.pkl` | `models/RITnet-master/best_model.pkl` | RITnet 瞳孔分割 |
| `DeepVOG_weights.h5` 等 | `models/DeepVOG-master/deepvog/model/` | DeepVOG 分割 |

faceparts 类别映射必须为 `{0: eye, 1: nose, 2: mouth, 3: eyebrow}`。

## 4. 中文路径注意事项

- `cv2.dnn` 读 ONNX（YuNet / YOLO-face）对非 ASCII 路径存在兼容问题，历史代码使用 `roi_common.ascii_model_path()` 处理。
- `face_landmarker.task` 同理，`FaceLandmarkerSession` 内部有对应处理。
- `.pt` 的 torch/Ultralytics 路径兼容性相对更好，但新设备仍建议把仓库放在纯 ASCII 路径。

## 5. 历史运行示例

```powershell
$env:PYTHONPATH='src'

D:/Code/python/python.exe scripts/roi_faceparts.py --subject sub-011 --conf 0.3 --imgsz 1280

D:/Code/python/python.exe scripts/run_all_backends.py `
  --subject sub-011 --roi-backends faceparts,mediapipe --pupil-algos PuReST,RITnet

venv-deepvog/Scripts/python.exe scripts/deepvog_pupil.py --roi_dir <平铺ROI目录> --out deepvog.csv

D:/Code/python/python.exe scripts/iris_landmark.py --image_dir <整帧图片目录> --out iris.csv

D:/Code/python/python.exe -m pytest -q
```

## 6. 许可证提示

`models/yolo-face-parts-detector-main/` 为 AGPL-3.0。研究/内部评估与未来对外发布的许可义务需分别评估；当前仓库继续把其作为历史比较资产保留。

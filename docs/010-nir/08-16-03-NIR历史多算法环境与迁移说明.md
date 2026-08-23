# SETUP｜环境配置与可移植部署说明

> 08-16（Asia/Shanghai）｜本仓库封装了 4 个 ROI 后端 + 4 个瞳孔算法，目标是**自包含可移植**：把 v2 目录拷到另一台设备，按本文装好依赖即可跑全算法（本机不选最优，新设备跑完再选型）。

## 1. 目录自包含说明

除**数据盘**外，所有代码、模型权重、配置均在 v2 目录内，相对路径自动 resolve 到 v2 根（`configs/formal.yaml` 的 `paths` 用 `models/...` 相对路径，`config.path_value` 解析到 `<v2>/models/`）。

唯一例外：`configs/formal.yaml` 的 `formal_data_root: "E:/正式实验"` 是数据盘绝对路径，**新设备需改成你的数据目录**。其余模型路径无需改动。

## 迁移到新设备（完整步骤）

**核心原则：Python venv / 全局环境不能直接拷贝**（`pyvenv.cfg`、`Scripts/python.exe` 的 shebang 都硬编码原机路径，拷过去会坏），必须在新设备重装。只有 v2 目录和数据盘可拷贝。

### 第 1 步：拷贝文件

| 东西                              | 说明                                                                  |
| --------------------------------- | --------------------------------------------------------------------- |
| `attention-pipeline-v2/` 整目录 | 已自包含：代码 + 模型权重 +`runtime/`(wheel+依赖清单) + 配置 + 测试 |
| 数据盘                            | `E:\正式实验`（+ 需要的预实验数据）                                 |

### 第 2 步：装主环境（Python 3.13）

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r runtime/requirements-main.txt   # 全量冻结；或按下方「主环境依赖」精简清单
```

### 第 3 步：装 venv-pupil（Python 3.10 ⚠️）

PyPupilEXT wheel 是 `cp310-win_amd64`，**只能用 Python 3.10 + Windows 64 位**（3.11+ 装不上）：

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

其余模型路径是相对 `models/...`，`config.path_value` 自动解析到 v2 根，**不用改**。

### 第 5 步：验证

```powershell
$env:PYTHONPATH='src'
python -m pytest -q
python scripts/run_all_backends.py --subject sub-011 --roi-backends faceparts,mediapipe --pupil-algos PuReST,RITnet
```

### 迁移易错点

- **PyPupilEXT 锁 Python 3.10**（`cp310`），3.11+ 装不上——venv-pupil 必须用 3.10。
- **ultralytics 用 ≥8.3**（主环境 8.4.120），勿装 faceparts 仓库锁的 8.2.27（不兼容 Python 3.13）。
- **v2 建议放纯 ASCII 路径**（如 `D:/attention-pipeline-v2`），规避 `cv2.dnn` 中文路径（代码已兜底，但更省心）。
- **DeepVOG 单独 venv**，别与 torch 混装。

## 2. 两个解释器 + 依赖

| 解释器                    | 用途                               | 必需依赖               |
| ------------------------- | ---------------------------------- | ---------------------- |
| 主环境（Python 3.13+）    | ROI 定位、RITnet、Iris、统计、测试 | 见下「主环境依赖」     |
| venv-pupil（Python 3.10） | PyPupilEXT 的 PuRe/PuReST          | 见下「pupil 环境依赖」 |

### 主环境依赖

```powershell
# 主环境（本项目当前为 D:/Code/python/python.exe，Python 3.13）
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # torch（RITnet）
python -m pip install ultralytics onnxruntime opencv-python numpy pandas pyyaml            # faceparts / yolo / 通用
python -m pip install mediapipe                                                             # mediapipe ROI + Iris
python -m pip install pytest                                                                # 测试
```

- `torch`：RITnet 推理（`best_model.pkl` 是 torch state_dict）。CPU 版即可。
- `ultralytics` ≥8.3（兼容 Python 3.13；**不要**装 faceparts 仓库锁的 8.2.27，那是 Python 3.11 时代）。
- `mediapipe`：`face_landmarker.task` 的 FaceLandmarker（ROI + Iris 关键点）。
- `onnxruntime`：YuNet / YOLO-face 的 ONNX 推理。
- 不需要 `supervision`（faceparts 官方 `run.py` 用它画框，本管线直接读 `result.boxes`）。

### pupil 环境依赖（venv-pupil）

PuRe/PuReST 走 `venv-pupil/Scripts/python.exe`，依赖 PyPupilEXT（`pypupilext`）。此环境需在原机复现（含 pypupilext 及其 C++ 绑定）；本仓库不重打包 wheel，部署时从原 venv 复制或按 PyPupilEXT 官方安装。

### DeepVOG 附加依赖（可选，部署时另装）

DeepVOG 是 **Keras/TensorFlow 1.x 时代**代码（`DeepVOG_model.py` 用 standalone `keras`，权重 `.h5`），与主环境 torch 共存有冲突风险。建议**单独建一个 venv**：

```powershell
python -m venv venv-deepvog
venv-deepvog/Scripts/pip install tensorflow==2.x keras==2.x scikit-image scikit-video numpy
# DeepVOG 原始依赖：keras、tensorflow、skvideo、skimage、urwid
```

`scripts/deepvog_pupil.py` 已做延迟导入：未装 keras 时 `import` 不报错，只有实际调用 `load_model()` 才失败。装好 keras 后即可跑。

## 3. 模型权重清单（均已内置 v2，无需额外下载）

| 模型                      | v2 内路径                                                   | 大小   | 用途                   |
| ------------------------- | ----------------------------------------------------------- | ------ | ---------------------- |
| `face_landmarker.task`  | `models/face_landmarker.task`                             | 3.8MB  | MediaPipe ROI + Iris   |
| `yunet_2023mar.onnx`    | `models/yunet_2023mar.onnx`                               | 233KB  | YuNet ROI              |
| `yolov8n-face.onnx`     | `models/yolov8n-face.onnx`                                | 12MB   | YOLO-face ROI          |
| `yolov8n.pt`            | `models/yolo-face-parts-detector-main/weights/yolov8n.pt` | 6.2MB  | faceparts ROI（nano）  |
| `yolov8s.pt`            | `models/yolo-face-parts-detector-main/weights/yolov8s.pt` | 22.5MB | faceparts ROI（small） |
| `best_model.pkl`        | `models/RITnet-master/best_model.pkl`                     | ~1MB   | RITnet 瞳孔分割        |
| `DeepVOG_weights.h5` 等 | `models/DeepVOG-master/deepvog/model/`                    | ~94MB  | DeepVOG 分割           |

**faceparts 权重下载 URL**（若缺失需重下）：

```
https://github.com/ignaciohrdz/yolo-face-parts-detector/releases/download/v1.0.0/yolov8n.pt
https://github.com/ignaciohrdz/yolo-face-parts-detector/releases/download/v1.0.0/yolov8s.pt
```

类别映射必须为 `{0: eye, 1: nose, 2: mouth, 3: eyebrow}`（`roi_faceparts.py` 加载时校验）。

## 4. 中文路径注意事项

- `cv2.dnn` 读 ONNX（YuNet / YOLO-face）**不支持非 ASCII 路径** → 已用 `roi_common.ascii_model_path()` 自动复制到临时 ASCII 路径。
- `face_landmarker.task` 同理，`FaceLandmarkerSession` 内部已处理中文路径。
- `.pt`（ultralytics torch 加载）对中文路径 OK，但权重统一放 v2 内 ASCII 子路径最稳。
- **新设备建议把 v2 放在纯 ASCII 路径**（如 `D:/attention-pipeline-v2`），可完全规避上述问题。

## 5. 运行示例

```powershell
$env:PYTHONPATH='src'   # 或把 src 加入 PYTHONPATH

# 单 ROI 后端（faceparts 特写）
D:/Code/python/python.exe scripts/roi_faceparts.py --subject sub-011 --conf 0.3 --imgsz 1280

# 全算法统一入口：faceparts+mediapipe ROI × PuReST+RITnet 瞳孔
D:/Code/python/python.exe scripts/run_all_backends.py `
  --subject sub-011 --roi-backends faceparts,mediapipe --pupil-algos PuReST,RITnet

# DeepVOG（单独 venv 装好 keras 后）
venv-deepvog/Scripts/python.exe scripts/deepvog_pupil.py --roi_dir <平铺ROI目录> --out deepvog.csv

# Iris（整帧，仅全脸画面）
D:/Code/python/python.exe scripts/iris_landmark.py --image_dir <整帧图片目录> --out iris.csv

# 全仓回归
D:/Code/python/python.exe -m pytest -q
```

## 6. 许可证提示

`models/yolo-face-parts-detector-main/` 为 **AGPL-3.0**。研究/内部评估无碍；若未来对外发布或提供网络服务，需单独评估 AGPL 的源代码公开义务（当前不阻塞研究）。

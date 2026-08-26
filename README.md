# Attention-Analysis｜RGB NVIDIA / CUDA

当前分支：`rgb-nvidia`  
目标：在 NVIDIA RTX 5070 工作站上完成正式 RGB `Face + Pose + Motion` raw 提取。正式分析范围统一为 **baseline 开始连续到 Block2 结束**；tracking、眼睑、blink/PERCLOS、Pose features、QC 和统计分析全部后移。

> NVIDIA Face 使用 **Py-Feat 2.1.1 Detectorv2 + native PyTorch CUDA**。不要把 AMD DirectML 的 ONNX Runtime 调用方式、RetinaFace batch / multitask batch 两套参数直接复制到本分支。

---

# 1. 第一次使用：创建环境

固定仓库：

```text
D:\Project\厚粲杯\08_算法\01_Attention-Analysis_nvidia-cuda
```

原始数据：

```text
J:\Data
```

正式 RGB 输出：

```text
D:\Project\厚粲杯\11_数据\04_Attention-Analysis_nvidia-cuda_RGB
```

## 1.1 获取并进入 `rgb-nvidia`

如果本地仓库还不存在：

```powershell
cd "D:\Project\厚粲杯\08_算法"
git clone https://github.com/kyandi233-dev/Attention-Analysis.git 01_Attention-Analysis_nvidia-cuda
```

然后：

```powershell
cd "D:\Project\厚粲杯\08_算法\01_Attention-Analysis_nvidia-cuda"
git fetch origin --prune
git switch rgb-nvidia
git pull --ff-only
git status --short --branch
```

## 1.2 创建 RGB core 环境

负责 Motion、Pose、Face 15 Hz 帧清单、audit 和最终 validator：

```powershell
conda create -p "D:\conda_envs\attention-rgb" python=3.11 -y
conda activate "D:\conda_envs\attention-rgb"

cd "D:\Project\厚粲杯\08_算法\01_Attention-Analysis_nvidia-cuda"
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pyarrow pytest mediapipe
```

检查：

```powershell
where.exe python
python -c "import cv2,numpy,pandas,pyarrow,mediapipe; print('RGB core OK'); print('mediapipe=',mediapipe.__version__)"
```

## 1.3 创建 Face CUDA 环境

负责 Py-Feat 2.1.1 native PyTorch/CUDA Face：

```powershell
conda create -p "D:\conda_envs\attention-face-cuda" python=3.11 -y
conda activate "D:\conda_envs\attention-face-cuda"
python -m pip install --upgrade pip
```

安装 NVIDIA CUDA PyTorch。当前项目使用官方 CUDA wheel；若下列 CUDA wheel 将来被官方替换，应以当时官方仍支持 RTX 5070 的 CUDA wheel 为准，不能安装 CPU-only Torch：

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

安装冻结 Face 环境：

```powershell
python -m pip install "py-feat==2.1.1" pandas pyarrow opencv-python pillow

cd "D:\Project\厚粲杯\08_算法\01_Attention-Analysis_nvidia-cuda"
python -m pip install -e . --no-deps
```

检查：

```powershell
nvidia-smi
python -c "import torch,importlib.metadata as m; print('torch=',torch.__version__); print('torch_cuda=',torch.version.cuda); print('cuda=',torch.cuda.is_available()); print('gpu=',torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); print('py-feat=',m.version('py-feat'))"
```

必须满足：

```text
torch.cuda.is_available() == True
py-feat == 2.1.1
GPU == NVIDIA RTX 5070 工作站实际 GPU
```

---

# 2. 每次打开新终端

先同步 Git：

```powershell
cd "D:\Project\厚粲杯\08_算法\01_Attention-Analysis_nvidia-cuda"
git status --short --branch
git fetch origin --prune
git switch rgb-nvidia
git pull --ff-only
git status --short --branch
```

如果 `git status` 有本地修改，先检查 `git diff`，不要直接 `reset --hard`。

需要检查 RGB core 时：

```powershell
conda activate "D:\conda_envs\attention-rgb"
python -c "import cv2,numpy,pandas,pyarrow,mediapipe; print('RGB core OK')"
```

需要检查 CUDA Face 时：

```powershell
conda activate "D:\conda_envs\attention-face-cuda"
python -c "import torch,importlib.metadata as m; print(torch.__version__,torch.version.cuda,torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,m.version('py-feat'))"
```

正式 PowerShell 总控会直接调用两个固定环境的 `python.exe`，所以完成环境检查后，不要求你在 Motion / Pose / Face 阶段之间手动切环境。

---

# 3. 正式运行前检查和测试

## 3.1 Schema 回归测试

在 Face CUDA 环境：

```powershell
conda activate "D:\conda_envs\attention-face-cuda"
cd "D:\Project\厚粲杯\08_算法\01_Attention-Analysis_nvidia-cuda"
python -m pytest tests/test_rgb_formal_schema.py -q
```

这个测试保护历史上已经出现过的 streaming Parquet `double -> null` schema 故障。

## 3.2 RGB audit

在 RGB core 环境：

```powershell
conda activate "D:\conda_envs\attention-rgb"
cd "D:\Project\厚粲杯\08_算法\01_Attention-Analysis_nvidia-cuda"
python scripts/rgb_analysis.py --config configs/rgb_analysis.yaml --stage audit
```

检查：

```text
J:\Data
D:\Project\厚粲杯\11_数据\04_Attention-Analysis_nvidia-cuda_RGB\rgb_inventory.csv
```

## 3.3 NVIDIA 代表被试 Gate

当前代表被试：

```text
sub-130
```

首次 full-span 验收：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_rgb_formal_subject.ps1 `
  -Subject sub-130 `
  -CudaDevice cuda
```

完成后必须生成并通过：

```text
sub-130_motion_raw.parquet
sub-130_pose_landmarks.parquet
sub-130_face_raw.parquet
sub-130_manifest.json
```

最终 manifest：

```text
completion_status = complete
extraction_complete = true
```

并且 NVIDIA Face manifest 必须证明：

```text
execution_backend = pytorch_cuda
device = cuda / cuda:<index>
```

在 sub-130 Gate、CPU↔CUDA representative parity 和 schema test 通过前，不启动整个 cohort。

---

# 4. 正式全量运行

Gate 通过后：

```powershell
cd "D:\Project\厚粲杯\08_算法\01_Attention-Analysis_nvidia-cuda"

powershell -ExecutionPolicy Bypass -File .\scripts\run_rgb_formal_cohort.ps1 `
  -CudaDevice cuda
```

只跑指定被试：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_rgb_formal_cohort.ps1 `
  -Subjects sub-130,sub-131,sub-132 `
  -CudaDevice cuda
```

cohort runner 会自动：

```text
refresh audit / inventory
→ analysis_eligible=True
→ 已完成 subject 自动 skip
→ 某个 raw branch 已完成则 resume/skip
→ 单被试失败记录后继续
→ cohort_status.csv
→ cohort_manifest.json
```

正常运行不要使用 `-Force`。

---

# 5. 可以调整、且不改变科学定义的参数

以下参数只改变计算分块、GPU/CPU 调度或运行对象，不用于改变科学处理条件。

| 参数 | 位置 | 作用 | 建议 |
|---|---|---|---|
| `-FaceBatch` | 单被试 / cohort PowerShell | Py-Feat native Detectorv2 **端到端 CUDA batch** | RTX 5070 实测 `16 → 32 → 64`，选择吞吐最高且显存稳定的值 |
| `native_cuda_prefetch_batches` | `configs/rgb_analysis.yaml` | CPU 解码到 CUDA Face 的预取队列深度 | 默认 2；只有 I/O 饥饿时再调 |
| `-CudaDevice` | PowerShell | 指定 `cuda` / `cuda:0` 等设备 | 单卡通常保持 `cuda` |
| `-Subjects` | cohort runner | 只选择部分被试运行 | 用于 pilot / 分批执行 |

例如测试 native batch 32：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_rgb_formal_subject.ps1 `
  -Subject sub-130 `
  -CudaDevice cuda `
  -FaceBatch 32
```

测试 64 时改为：

```text
-FaceBatch 64
```

### Batch 的含义

NVIDIA native Detectorv2 只有一个端到端 batch：

```text
FaceBatch = 16 / 32 / 64
```

AMD DirectML 则是两个不同 ONNX 阶段：

```text
RetinaFace batch
multitask batch
```

因此 AMD 的 `B32/B64` 不能解释成 NVIDIA 也必须写成两个 `32/64`。两边 scientific contract 可以一致，但计算分块应分别按各自后端寻找最快值。

Batch、prefetch 等参数不应改变最终科学定义；CUDA 浮点计算可能存在末位数值差异，因此正式冻结前仍用 representative parity 检查确认结果在预设容差内一致。

---

# 6. 不允许为了提速随意调整的科学条件

以下属于正式 scientific contract，不作为普通速度参数开放：

```text
Face inference_fps = 15 Hz
Pose inference_fps = 10 Hz
Motion = 原视频 full FPS
Face detection threshold = 0.5
Py-Feat = 2.1.1 Detectorv2
identity_model = None
analysis span = baseline start → Block2 end
all detected faces retained
no-face planned frames retained
raw-first / QC 后移
```

也不要未经 parity 验证就改：

```text
模型权重
输入 resize / normalization
RetinaFace / Detectorv2 内部 NMS 规则
face crop 规则
landmark / AU / emotion / gaze / pose 的模型后处理
```

---

# 7. 当前 NVIDIA 正式数据流

```text
Face 15 Hz timestamp grid
        ↓
Motion full FPS ─┐
Pose 10 Hz      ├── 三条独立 reader 并行
Face 15 Hz CUDA ┘
        ↓
raw-only validator
        ↓
下一个被试
```

Face：

```text
Py-Feat 2.1.1 Detectorv2
→ native PyTorch CUDA
→ face_raw.parquet
```

Pose：

```text
MediaPipe Pose 10 Hz
→ 33 landmarks + world coordinates + visibility/presence
→ pose_landmarks.parquet
```

Motion：

```text
OpenCV full FPS frame difference
→ motion_raw.parquet
```

以下全部后算，不阻挡全量：

```text
Face tracking
primary-face selection
EAR / 眼睑开度 / aperture-iris
blink / PERCLOS
Pose features
QC
trial / block / time-window / probe analysis
统计建模
```

---

# 8. NVIDIA 与 AMD 的结果一致性原则

目标不是让 CUDA 和 DirectML 的底层代码逐行相同，而是保证两边使用同一 scientific contract，并用同一批输入帧做 parity。

AMD 的“0.5 score 前置后再 decode/NMS”属于在同一 RetinaFace scores、同一最终阈值和按分数降序 greedy NMS 前提下的结果等价计算重排；它不应改变最终保留的 `score >= 0.5` 人脸。

但“AMD 手工 ONNX RetinaFace + multitask pipeline”与“NVIDIA native Py-Feat Detectorv2”不能只靠理论声称完全一致。正式接受标准是：

```text
同一批代表帧
→ 同一 detection threshold
→ 同一科学输出集合
→ face coverage / bbox / AU / emotion / gaze / head pose / mesh / blendshape parity
→ 数值差异在预设容差内
```

NVIDIA 已保留 native CPU reference ↔ CUDA Gate；跨 AMD/NVIDIA 时也应使用共享 fixture，而不是拿不同被试做逐帧 parity。

---

# 9. 详细文档

操作细节和工程 provenance 放在 `docs/040-rgb/`，根 README 只保留实际运行入口：

- `docs/040-rgb/046-NVIDIA-CUDA-RGB运行路线.md`
- `docs/040-rgb/RGB_TEMP_BRANCH_ABSORPTION_RESULT_20260826.md`
- `docs/040-rgb/044-RGB输出Schema与信息保留原则.md`
- `docs/040-rgb/042-面部分析工具与Benchmark.md`

临时分支 `codex/rgb-nvidia-formal-pipeline-v1` 当前仍保留，不在这里删除。

# 08-26-02｜RGB Face DirectML Gate 0/1 实现

> 2026-08-26｜分支：`rgb-dev`｜承接 `08-26-01-RGB-Face阶段交接与DirectML计划.md`。本记录只新增当前实现状态，不改写此前 CPU benchmark 历史结果。

## 1. 本轮目标

不重复 Py-Feat / LibreFace 已完成的 CPU benchmark，直接启动 AMD / ONNX Runtime DirectML 双路线迁移。第一轮先完成两个低风险门控：

1. **Gate 0：同版本模型导出**。从项目现有 reference 环境加载当前实际使用的 PyTorch 权重并导出 ONNX，同时记录源权重与 ONNX 文件 SHA256；
2. **Gate 1：DirectML model-core/provider smoke**。在纯 `attention-face-directml` 环境逐模型创建 DML session，检查 provider、profile 中的 CPU fallback，并测试 batch 1 / 8 / 16 / 32 的基础 model-core 吞吐。

Gate 1 只回答“ONNX 图能否在当前 AMD / DirectML 上可靠执行、哪些算子发生 fallback、哪些 batch 可运行”。它**不使用新的 CPU benchmark，不替代 300 帧 parity，不用于冻结 backend**。Gate 0/1 通过后，下一步才把同一批现有 300 帧接回两条完整 pipeline，做真实输入 parity + raw-frame end-to-end。

## 2. 新增脚本

### `scripts/face_export_libreface_onnx.py`

在 `attention-face-libreface` 环境运行，针对当前项目已经使用的 LibreFace Python reference 导出：

- `libreface2_au_joint.onnx`：一次 ResNet18 encoder，同时输出 12 AU intensity probability 与 12 AU detection probability；正式后处理保持 `intensity = probability × 5`、`detection = probability >= 0.5`；
- `libreface2_expression.onnx`：当前 RepVGG expression model，8 类 score；
- `libreface2_gaze_mlp.onnx`：当前 1404-d MediaPipe landmark feature → yaw/pitch MLP。

这一实现刻意以**当前 Python reference 的实际 checkpoint**为来源，而不是在 parity 前默认旧 ONNX/NuGet derivative 与当前 2.0 权重相同。MediaPipe alignment、head pose/landmark 以及 gaze feature extraction 仍属于 CPU 前处理；learned heads 才迁移到 DML。

### `scripts/face_export_pyfeat_onnx.py`

在 `attention-face-pyfeat` 环境运行，针对 Py-Feat 2.1.1 Detectorv2 导出：

- `pyfeat211_retinaface_r34.onnx`：RetinaFace ResNet34 网络；
- `pyfeat211_multitask_scientific_core.onnx`：20 AU、7 emotion、V/A、gaze、6DoF pose、478×3 mesh、52 blendshape。

identity branch 按 053 当前决策暂不进入第一轮 scientific-core benchmark。导出 manifest 固定记录 Py-Feat 2.1.1 的关键几何/后处理合同：RetinaFace → isotropic square-pad 256 crop（expand 1.2、reflection padding）→ 256 全视野 resize 到 224 → ImageNet normalization；不能误改成 center crop。

RetinaFace ONNX 按当前 300 帧共同样本的实际 H×W 导出固定空间尺寸，仅 batch 维动态。这样符合 DirectML 对已知 tensor shape 更友好的运行条件，同时保证两候选仍使用同一批 300 帧输入。

### `scripts/face_directml_probe.py`

在 `attention-face-directml` 环境运行。脚本强制：

- `enable_mem_pattern=False`；
- `execution_mode=ORT_SEQUENTIAL`；
- `ORT_ENABLE_ALL` graph optimization；
- `DmlExecutionProvider` 优先、`CPUExecutionProvider` 仅作为显式 fallback；
- 每个 batch 保存 ORT profile，并统计 profile 中 DML/CPU kernel events；
- 记录 ONNX SHA256、provider/options、I/O shape、finite fraction、batch throughput 与错误。

如果系统没有 `DmlExecutionProvider`，脚本直接报错，不允许把 CPU session 误记为 AMD benchmark。

## 3. Gate 0：导出命令

先拉取 `rgb-dev`：

```powershell
cd "D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-rgb-dev"
git pull --ff-only
```

### LibreFace

使用既有 LibreFace reference 环境和已经缓存的 reference 权重：

```powershell
conda activate "D:\CondaEnvs\attention-face-libreface"
python -m pip install onnx

python scripts/face_export_libreface_onnx.py `
  --weights-dir "D:\_AttentionData\Beijing-RGB\_test\face-continuous\sub-031\libreface_weights" `
  --output-dir "D:\_AttentionData\Beijing-RGB\_test\face-directml\models\libreface"
```

如果当前连续 benchmark 目录的 `libreface_weights` 已存在，不会重跑 CPU benchmark；脚本只加载这些权重并导出模型。

### Py-Feat

```powershell
conda activate "D:\CondaEnvs\attention-face-pyfeat"
python -m pip install onnx

python scripts/face_export_pyfeat_onnx.py `
  --benchmark-dir "D:\_AttentionData\Beijing-RGB\_test\face-continuous\sub-031" `
  --output-dir "D:\_AttentionData\Beijing-RGB\_test\face-directml\models\pyfeat"
```

该脚本读取现有 300 帧 manifest 只为取得固定 RetinaFace H×W，不执行 `Detectorv2.detect()`，因此不会重复已有 300 帧 CPU reference inference。

## 4. Gate 1：AMD DirectML probe

如果环境尚未创建：

```powershell
conda create -p "D:\CondaEnvs\attention-face-directml" python=3.11 -y
conda activate "D:\CondaEnvs\attention-face-directml"
python -m pip install onnx onnxruntime-directml numpy pandas pyarrow opencv-python pillow
python -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
```

必须看到 `DmlExecutionProvider`。不要在该环境同时安装普通 `onnxruntime` 或 `onnxruntime-gpu`。

LibreFace：

```powershell
python scripts/face_directml_probe.py `
  --model "D:\_AttentionData\Beijing-RGB\_test\face-directml\models\libreface\libreface2_au_joint.onnx" `
  --model "D:\_AttentionData\Beijing-RGB\_test\face-directml\models\libreface\libreface2_expression.onnx" `
  --model "D:\_AttentionData\Beijing-RGB\_test\face-directml\models\libreface\libreface2_gaze_mlp.onnx" `
  --batch-sizes 1,8,16,32 `
  --output "D:\_AttentionData\Beijing-RGB\_test\face-directml\sub-031\libreface_gate1.json"
```

Py-Feat：

```powershell
python scripts/face_directml_probe.py `
  --model "D:\_AttentionData\Beijing-RGB\_test\face-directml\models\pyfeat\pyfeat211_retinaface_r34.onnx" `
  --model "D:\_AttentionData\Beijing-RGB\_test\face-directml\models\pyfeat\pyfeat211_multitask_scientific_core.onnx" `
  --batch-sizes 1,8,16,32 `
  --output "D:\_AttentionData\Beijing-RGB\_test\face-directml\sub-031\pyfeat_gate1.json"
```

Gate 1 使用 synthetic tensor，只用于 provider / ONNX compatibility / fallback / batch 可运行性与基础 model-core 工程门控。因此其中的 images/s **不得与已有 CPU 300 帧速度直接比较，也不得写成最终 AMD 端到端速度**。

## 5. 下一步判定

拿到两个 `*_gate1.json` 后，按以下顺序继续：

1. 先看每个 ONNX / batch 是否 `status=ok`；
2. 看 `session_providers` 是否包含并优先使用 `DmlExecutionProvider`；
3. 看 profile 的 `dml_kernel_events` 与 `cpu_kernel_events`，识别是否存在阻断性的 CPU fallback；
4. 根据可运行 batch 冻结真实 300 帧 benchmark 的候选 batch；
5. 实现/运行 **同 300 帧真实输入**：LibreFace alignment→ONNX heads 与 Py-Feat RetinaFace→square-pad crop→multitask；
6. 与现有 CPU parquet 逐字段做 parity，速度与 parity 分开报告；
7. 只有 parity、coverage、visual sanity check 与 raw schema 都通过后，才讨论 Face backend / fps / primary-face 冻结。

本轮不修改 Pose/Motion 决策，不运行 44 被试全量 Face，不实现 body_motion_energy。

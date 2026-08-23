# NIR YOLO + Tracking + RITnet GPU 试跑包

> 2026-08-22（Asia/Shanghai）｜用于另一台 GPU 电脑的短视频准入测试；当前权重已通过静态 test，但 tracking、ROI 扩展和 RITnet 视频质量尚未冻结。

## 这个包能做什么

流程为：

```text
NIR AVI → YOLO26n 眼框 → CSRT/KCF 或逐帧 YOLO
        → 横坐标排序为 frame_left/frame_right
        → 扩展并缩放到 320×160 单眼 ROI
        → RITnet GPU 分割
        → frames.csv + eyes.csv + overlays + summary.json
```

支持两种已知目录：

```text
F:\正式实验\sub-011_\nir\sub-011_nir.avi
E:\Data\sub-056_\nir\sub-056_nir.avi
```

`frame_left/frame_right`仅表示画面横坐标左右，不是解剖学左右眼。默认只跑 60 秒；整段视频必须显式使用 `--full-video`。

## 目录

```text
nir-yolo-tracking-ritnet-v1/
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

## 第一次运行

在已经配置好 GPU 版 PyTorch、Ultralytics 和 RITnet 的环境中进入本目录：

```powershell
python .\run_pipeline.py check-env
python .\run_pipeline.py discover
```

`check-env`应显示：

- `cuda_available: true`；
- `models_exist: true`；
- `tracker_csrt: true`、`tracker_kcf: true`。

如果 tracker 不可用，安装的是普通 `opencv-python` 而不是 contrib 版本；在你现有环境允许的前提下改用 `opencv-contrib-python`。不要在未确认环境前直接卸载已有 OpenCV。

## 建议的运行顺序

先做 20 秒 YOLO-only 冒烟，不加载 RITnet：

```powershell
python .\run_pipeline.py run `
  --subject sub-056 --root "E:\Data" `
  --duration-sec 20 --tracker none --skip-ritnet
```

确认框的位置后，再做 60 秒联调：

```powershell
python .\run_pipeline.py run `
  --subject sub-056 --root "E:\Data" `
  --duration-sec 60 --tracker csrt --redetect-interval 10
```

默认 `--device 0`，YOLO 与 RITnet 都使用第一张 GPU；如需第二张 GPU 使用 `--device 1`，CPU 诊断可使用 `--device cpu`。

也可直接给视频：

```powershell
python .\run_pipeline.py run `
  --video "F:\正式实验\sub-011_\nir\sub-011_nir.avi" `
  --duration-sec 60 --tracker csrt
```

比较 tracking 时依次运行：

```powershell
python .\run_pipeline.py run --subject sub-056 --root "E:\Data" --duration-sec 60 --tracker none
python .\run_pipeline.py run --subject sub-056 --root "E:\Data" --duration-sec 60 --tracker csrt --redetect-interval 5
python .\run_pipeline.py run --subject sub-056 --root "E:\Data" --duration-sec 60 --tracker csrt --redetect-interval 10
python .\run_pipeline.py run --subject sub-056 --root "E:\Data" --duration-sec 60 --tracker kcf --redetect-interval 10
```

## 输出解释

- `frames.csv`：每帧是 YOLO 还是 tracker、原始框数、选中眼数、处理耗时和失败状态；
- `eyes.csv`：每帧×眼睛的框、扩展 ROI、RITnet 瞳孔椭圆、置信度和状态；
- `overlays/`：默认每 30 帧一张眼框叠加图；
- `summary.json`：速度和状态计数；
- `run_manifest.json`：命令、环境、配置和两个权重 SHA256；
- `rois/`：仅在 `--save-rois` 时保存，避免默认产生大量文件。

状态语义：

- `yolo_missing`：YOLO 没有眼框；
- `single_eye`：只有一个框；
- `extra_boxes`：YOLO 返回多于两个框；
- `roi_clipped`：扩展 ROI 碰到画面边界；
- `ritnet_missing`：ROI 存在但 RITnet 没有形成有效瞳孔椭圆；
- `observed`：当前帧有直接观测到的 RITnet 椭圆；
- `roi_only`：使用了 `--skip-ritnet`。

本包不做插值，也不把失败帧补成成功帧。

如果视频同目录存在 `sub-xxx_nir_timestamps.csv`，脚本会按 `frame_idx` 读取第二列 `unix_ms` 并写入两个 CSV；如果不存在，仍保留 `video_time_ms`，`unix_ms` 留空。`redetect_reason` 会区分计划重检测、tracker 失败回退和上一帧缺失后的重检测。

## 当前参数和边界

- YOLO 阈值 0.40 来自 val，不使用 test 调参；
- 默认 `imgsz=640`、NMS IoU=0.70；
- 默认 CSRT，每 10 帧 YOLO 重检测；
- 默认眼框每侧横向扩展 30%、纵向扩展 45%；这些扩展比例仍需在正式视频上比较；
- RITnet 原始 `pupil_confidence` 只输出，不设生产拒绝阈值；
- 左右眼身份、tracking 方案、扩展比例和 RITnet 门控均未冻结。

不要只因为脚本能跑完就直接处理全部被试。先检查 20 秒、60 秒、不同 tracking 配置和 overlay，再做一段完整视频试点。

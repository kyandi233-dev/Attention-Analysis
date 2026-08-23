# NIR 专用 YOLO 远程训练与迁移指南

## 2026-08-21 执行状态更新

> 2026-08-21 17:30（Asia/Shanghai）｜当前训练已在 AMD RX 6750 GRE 电脑上以 CPU 进行 100 epochs；训练结束后再决定是否将权重和推理环境迁移至 NVIDIA 电脑。

回传时至少保留：

- `weights/best.pt`、`weights/last.pt`；
- `args.yaml`、`results.csv`、验证图和混淆矩阵；
- 数据集版本、具体模型变体、训练命令、`imgsz`、`batch`、设备、Python/PyTorch/Ultralytics 版本；
- 按被试划分的 train/val/test 清单和最终验证结果。

正式批量处理一百多个约 25 分钟视频前，必须在 NVIDIA 电脑完成：模型加载测试、单视频读取测试、YOLO-only 基准、YOLO+tracking+RITnet 短片基准，以及失败状态输出检查。不得只凭 `best.pt` 文件存在就直接启动全量分析。

> 2026-08-17 Asia/Shanghai｜本文件说明训练时需要携带什么、如何在新电脑或远程机器上运行，以及训练结果如何带回 v2。

## 1. 训练机需要的内容

训练机不需要正式实验全部原始视频。第一轮只需要：

```text
nir-eye-dataset-v1/
├── images/
├── labels_yolo/
├── manifests/
└── dataset.yaml
```

另外需要：

- YOLO11n 或 YOLOv8n 的基础权重；
- Python；
- Ultralytics；
- PyTorch；
- 与训练脚本匹配的配置文件；
- 训练记录和固定随机种子。

正式视频和 PuReST 环境不属于 YOLO 训练的必需品，可以不迁移。

## 2. 推荐环境

当前 v2 主环境已经有 Ultralytics，可作为参考：

```text
Python 3.13
ultralytics 8.4.120
torch
opencv-python / opencv-contrib-python
```

远程训练优先使用 Python 3.10 或 3.11、官方兼容的 PyTorch 和 Ultralytics。不要把 `venv-pupil` 当作 YOLO 训练环境；它是 PyPupilEXT 专用环境。

如果远程机器有 NVIDIA GPU，安装与 CUDA 驱动匹配的 PyTorch；如果只有 CPU，也能训练 nano 模型，但速度会明显慢，应减少输入尺寸或训练轮数做试运行。

## 3. dataset.yaml

示例：

```yaml
path: D:/datasets/nir-eye-dataset-v1
train: images/train
val: images/val
test: images/test

names:
  0: eye
```

迁移到 Linux 时改成：

```yaml
path: /data/nir-eye-dataset-v1
```

路径不要写死成当前电脑的盘符；提交训练前可以使用相对路径或在训练机上修改一份副本。

## 4. 训练操作

用同一数据集分别运行两组实验。示意命令如下：

```text
yolo detect train model=yolo11n.pt data=dataset.yaml imgsz=1280 epochs=100 batch=2 device=0 project=runs/nir-eye name=yolo11n_v1 seed=20260817
```

YOLOv8 对照：

```text
yolo detect train model=yolov8n.pt data=dataset.yaml imgsz=1280 epochs=100 batch=2 device=0 project=runs/nir-eye name=yolov8n_v1 seed=20260817
```

参数含义：

- `model`：基础模型权重；
- `data`：数据集定义；
- `imgsz`：训练输入尺寸；眼睛较小，先试 1280；
- `epochs`：完整遍历训练集的轮数；
- `batch`：批大小，显存不足就降低；
- `device=0`：第一张 GPU；CPU 可写 `device=cpu`；
- `project/name`：输出位置和实验名称；
- `seed`：固定随机种子，便于复现。

第一轮不建议同时修改大量增强参数。先用默认增强得到基线，再根据 NIR 结果调整。NIR 灰度数据应避免过强的色彩增强；翻转、轻度亮度/对比度、轻度缩放可以保留。

## 5. 输出必须保存什么

训练完成后不要只拷贝 `best.pt`。至少保存：

```text
runs/nir-eye/yolo11n_v1/
├── weights/best.pt
├── weights/last.pt
├── args.yaml
├── results.csv
├── results.png
├── confusion_matrix.png
└── val_batch*.jpg
```

同时保存：

- 数据集版本；
- train/val/test 被试清单；
- Python、PyTorch、Ultralytics 版本；
- GPU/CPU 信息；
- 实际完整训练命令；
- 最佳 epoch 和验证指标；
- 测试集逐被试结果。

## 6. 把模型带回 v2

最终只把确认过的模型复制到：

```text
attention-pipeline-v2/models/roi/
```

建议命名：

```text
nir-eye-yolo11n-v1-best.pt
nir-eye-yolov8n-v1-best.pt
```

同时保存一个模型说明文件，记录：

```text
model_name, base_model, dataset_version, split_version, imgsz, epochs, device, ultralytics_version, selected_reason
```

在正式冻结前，配置文件仍不要直接指向某个模型；先通过测试集和 PuReST 下游评价，确认哪个模型入选。

## 7. 迁移检查清单

在新电脑或远程机器上开始训练前：

1. 能读取一张 NIR 训练图；
2. 能读取同名 YOLO 标签；
3. `dataset.yaml` 的 train/val/test 路径正确；
4. 类别数为 1，类别名为 `eye`；
5. 随机显示几张图片并叠加标签，确认框位置正确；
6. 先跑 1 个 epoch 的冒烟训练；
7. 冒烟训练无误后再开始完整训练；
8. 训练结束后保留完整 runs 目录和环境记录。

## 8. 当前不需要迁移的内容

第一轮 YOLO 训练不需要：

- DeepVOG 权重；
- RITnet 权重；
- PyPupilEXT/venv-pupil；
- 全部正式原始视频；
- CCNet 代码和像素级掩码。

这些属于后续 ROI 通过后的下游瞳孔分析或备选方案。

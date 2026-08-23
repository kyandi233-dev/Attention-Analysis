# yolotrain｜NIR 眼睛 YOLO 训练工作区

## 当前职责（2026-08-23）

`yolotrain/` 是从 `datasets/nir-eye-dataset-v1/` 派生出的 **YOLO 眼睛检测器训练工作区**。它不是另一份独立原始数据集，而是负责保存合并后的训练数据划分、训练 manifest、Ultralytics 配置、训练结果和模型权重。

当前状态：

- 数据源：`datasets/nir-eye-dataset-v1/`。
- `batch1` 与 `batch2` 在这里合并训练，但 `dataset_manifest.csv` 中继续保留 batch 信息。
- train / val / test 按**被试级别**固定划分，避免同一被试的相邻帧跨 split 泄漏。
- `split_subjects.csv` 与源数据集的 `datasets/nir-eye-dataset-v1/manifests/split_subject.csv` 内容一致；当前 Git blob SHA 均为 `38a48242d46ab40997e95664f3b22f593f8622e8`。
- YOLO26n 正式训练已经完成，共 **100 epochs**。
- 正式训练目录：`runs/yolo26n_eye_100epoch/`。
- 当前 GitHub 分支中的最终训练权重：`runs/yolo26n_eye_100epoch/weights/best.pt`。
- 该模型已经进入后续 NIR pipeline，并已完成正式 NIR 全量分析。

## 数据流

```text
datasets/nir-eye-dataset-v1/
        ↓
冻结源图像 + YOLO 标注 + batch provenance
        ↓
yolotrain/
        ↓
合并 batch1 / batch2
按被试固定划分 train / val / test
生成 dataset_manifest.csv
        ↓
YOLO26n 100 epochs
        ↓
runs/yolo26n_eye_100epoch/weights/best.pt
        ↓
正式 NIR pipeline
```

## 目录

```text
yolotrain/
├── README.md
├── dataset.yaml
├── dataset_manifest.csv
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── review/
├── runs/
│   ├── test_yolo26n/              # 早期 2-epoch 测试训练
│   └── yolo26n_eye_100epoch/      # 正式 100-epoch 训练
│       └── weights/
│           ├── best.pt
│           └── last.pt
└── split_subjects.csv
```

其中：

- `images/{train,val,test}`：训练、验证和最终测试图片。
- `labels/{train,val,test}`：与图片同名的 YOLO 标签。
- `dataset.yaml`：Ultralytics YOLO 数据集入口，显式定义 train / val / test。
- `split_subjects.csv`：被试级别的固定划分。
- `dataset_manifest.csv`：每张纳入图片的 batch、被试、split 和标注框数。
- `review/missing_labels.csv`：构建训练集时未纳入的无标签图片复核记录。
- `runs/test_yolo26n/`：早期环境与流程测试，实际为 2 epochs 训练，不能视作最终 test-set 评估。
- `runs/yolo26n_eye_100epoch/`：正式 YOLO26n 训练结果及最终权重。

## 划分原则

1. batch1 和 batch2 合并训练，但在 manifest 中保留环境标识。
2. 同一被试的所有帧只能属于 train、val 或 test 之一。
3. 只纳入已确认的图片—标签配对；无 `.txt` 的图片进入复核清单。
4. test 不参与训练、epoch 选择或超参数调整。

### Test set

固定 test split 包含 **7 个被试 / 85 张图片 / 169 个 eye boxes**。

被试来自：

```text
batch1: sub-012, sub-021, sub-9504
batch2: sub-032, sub-041, sub-051, sub-055
```

## 正式训练配置

正式 run 的 `args.yaml` 记录：

| 参数 | 值 |
| --- | --- |
| model | `yolo26n.pt` |
| epochs | `100` |
| imgsz | `640` |
| batch | `16` |
| device | `cpu` |
| workers | `0` |
| seed | `0` |
| deterministic | `true` |
| run | `runs/yolo26n_eye_100epoch/` |

100 epochs 已完整跑完。`results.csv` 的第 100 epoch validation 指标为：

| 指标 | epoch 100 |
| --- | ---: |
| Precision | 0.99922 |
| Recall | 1.00000 |
| mAP50 | 0.99500 |
| mAP50-95 | 0.71810 |

这里的数值属于训练过程中对 **validation split** 的记录，不应与 held-out test 指标混淆。

## 最终 test 记录

项目运行记录中已记录最终 held-out test 结果：

| 指标 | test |
| --- | ---: |
| mAP50 | ≈ 0.9913 |
| mAP50-95 | ≈ 0.6589 |

需要注意：截至当前 GitHub 分支核验，仓库中尚未找到一个独立提交的、使用 `best.pt` 对 `split=test` 运行后生成的机器可读评估文件。现有 `runs/test_yolo26n/` 实际是早期 2-epoch **train** run，而不是上述最终 test 评估。因此，上述两个 test 数值目前作为项目运行记录保留；若后续找到当时的 test 输出日志或重新导出 machine-readable evaluation artifact，应把它补充到本目录以完善 provenance。

## 历史记录：2026-08-21 建立训练工作区时的说明

> 2026-08-21（Asia/Shanghai）｜合并 batch1/batch2，按被试划分 train/val/test；原始数据保持不变。

当时的训练建议为：

> 首轮使用 `yolo26n.pt`，并保留同一划分下的 `yolo11n.pt` 对照。两者使用相同的图像尺寸、epoch 和随机种子。

这段内容保留用于记录当时的模型选择过程。当前实际结果已经明确：YOLO26n 完成 100 epochs 正式训练并进入后续正式 NIR pipeline，因此上述文字不再表示当前仍处于“首轮训练建议”阶段。

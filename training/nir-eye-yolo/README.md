# NIR Eye YOLO Training

本目录是由 `datasets/nir-eye-v1/` 派生出的 YOLO 训练工作区，包含固定 train/val/test 划分、训练清单、review 和训练结果。它不是原始标注数据目录，也不是正式 runtime。

## 当前状态

- YOLO26n 已完成 100 epochs 训练。
- 当前正式训练结果：`runs/yolo26n_eye_100epoch/`。
- 最佳权重：`runs/yolo26n_eye_100epoch/weights/best.pt`。
- 正式 runtime 使用同一模型的冻结副本：`../../runtime/nir-formal/models/nir-eye-yolo26n-best.pt`。

## 目录

- `images/{train,val,test}`：训练、验证和最终测试图片。
- `labels/{train,val,test}`：与图片同名的 YOLO 标签。
- `dataset.yaml`：Ultralytics 数据集入口，使用相对路径，可跨电脑迁移。
- `split_subjects.csv`：被试级固定划分。
- `dataset_manifest.csv`：每张纳入图片的 batch、被试、split 和标注框数。
- `review/missing_labels.csv`：未纳入训练的无标签图片复核清单。
- `runs/`：训练运行结果、参数、曲线和权重。

## 数据来源

```text
datasets/nir-eye-v1/
        ↓ 合并 batch1/batch2 + 被试级划分
training/nir-eye-yolo/
        ↓ YOLO26n 训练
runs/yolo26n_eye_100epoch/weights/best.pt
        ↓ 冻结副本
runtime/nir-formal/models/nir-eye-yolo26n-best.pt
```

## 划分原则

1. batch1 和 batch2 合并训练，但 manifest 保留环境标识。
2. 同一被试的所有帧只能属于 train、val 或 test 之一。
3. 只纳入已确认的图片—标签配对；无 `.txt` 的图片保留在 review 清单。
4. test 只用于最终评价，不用于挑选 epoch 或调参。

## 重新训练

进入本目录后可直接使用相对 `dataset.yaml`：

```powershell
cd training/nir-eye-yolo
yolo detect train model=yolo26n.pt data=dataset.yaml imgsz=640 epochs=100
```

若需要严格复现实次正式训练，应优先读取既有 `runs/yolo26n_eye_100epoch/args.yaml`，以其中实际参数为准，而不是仅依赖上面的示例命令。

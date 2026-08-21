# yolotrain｜NIR 眼睛 YOLO 训练集

> 2026-08-21（Asia/Shanghai）｜合并 batch1/batch2，按被试划分 train/val/test；原始数据保持不变。

## 目录

- `images/{train,val,test}`：训练、验证和最终测试图片。
- `labels/{train,val,test}`：与图片同名的 YOLO 标签。
- `dataset.yaml`：Ultralytics YOLO26/YOLO11 数据集入口。
- `split_subjects.csv`：被试级别的固定划分。
- `dataset_manifest.csv`：每张纳入图片的 batch、被试、split 和标注框数。
- `review/missing_labels.csv`：暂未纳入训练的无标签图片。

## 划分原则

1. batch1 和 batch2 合并训练，但在 manifest 中保留环境标识。
2. 同一被试的所有帧只能属于 train、val 或 test 之一。
3. 只纳入已确认的图片—标签配对；无 `.txt` 的图片先放入复核清单。
4. test 仅用于最终比较，不用它调参或挑选 epoch。

## 训练建议

首轮使用 `yolo26n.pt`，并保留同一划分下的 `yolo11n.pt` 对照。两者使用相同的图像尺寸、epoch 和随机种子。

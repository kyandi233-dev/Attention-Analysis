# Datasets

`datasets/` 保存训练/标注数据资产，不保存正式实验原始视频，也不作为正式分析输出目录。

## 当前数据资产

| 路径 | 角色 | 说明 |
|---|---|---|
| `nir-eye-dataset-v1/` | NIR 眼框原始标注数据版本 | 保留 batch1 / batch2 原始抽帧、YOLO 标注、manifest 与数据集说明 |
| `../yolotrain/` | YOLO 训练工作区 | 由 v1 数据整理得到的 train / val / test、训练运行结果与 `best.pt`；当前保留原路径，避免破坏训练记录与已有引用 |

## 资产关系

```text
NIR 正式视频
  ↓ 抽帧 / 人工标注
nir-eye-dataset-v1/
  ↓ 合并并按被试划分
../yolotrain/
  ↓ YOLO26n 训练
../yolotrain/runs/yolo26n_eye_100epoch/weights/best.pt
  ↓ 冻结副本进入正式运行包
../runtime/nir-formal/models/nir-eye-yolo26n-best.pt
```

`nir-eye-dataset-v1/` 是数据版本，`yolotrain/` 是训练工作区；两者职责不同，不因为目录整理而合并。

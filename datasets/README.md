# Datasets

`datasets/` 保存训练/标注数据资产，不保存正式实验原始视频，不保存训练运行结果，也不作为正式分析输出目录。

## 当前数据资产

| 路径 | 角色 | 说明 |
|---|---|---|
| `nir-eye-v1/` | NIR 眼框原始标注数据 v1 | 保留 batch1 / batch2 抽帧、YOLO 标注、manifest 与数据集说明 |
| `../training/nir-eye-yolo/` | YOLO 训练工作区 | 由 v1 数据整理得到的 train / val / test、训练运行结果与 `best.pt` |

## 数据—训练—正式模型关系

```text
NIR 正式视频
  ↓ 抽帧 / 人工标注
datasets/nir-eye-v1/
  ↓ 合并并按被试划分
training/nir-eye-yolo/
  ↓ YOLO26n 训练
training/nir-eye-yolo/runs/yolo26n_eye_100epoch/weights/best.pt
  ↓ 冻结副本
runtime/nir-formal/models/nir-eye-yolo26n-best.pt
```

`nir-eye-v1/` 是原始标注数据版本；`training/nir-eye-yolo/` 是由它派生的训练工作区。二者分开是为了区分数据 provenance 与训练运行产物。

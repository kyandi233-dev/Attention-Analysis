# NIR Eye Dataset v1

`datasets/nir-eye-v1/` 是 NIR 眼框检测器使用的原始抽帧与人工标注数据版本。它保存数据 provenance，不保存正式实验全量视频，也不作为训练运行目录。

## 2026-08-23 当前状态

- 数据集 v1 已完成标注并用于 YOLO26n 训练。
- 合并后的固定 train / val / test 与训练结果位于 `training/nir-eye-yolo/`。
- YOLO26n 已完成 100 epochs 训练；正式训练产物为 `training/nir-eye-yolo/runs/yolo26n_eye_100epoch/weights/best.pt`。
- 正式运行副本位于 `runtime/nir-formal/models/nir-eye-yolo26n-best.pt`。
- 正式 NIR 全量分析已经执行；本目录后续主要承担训练数据来源、标注规则与版本溯源职责。

> 下方 2026-08-19 / 08-21 的“准备训练 / 准备正式分析”等文字属于当时阶段记录，保留其历史语境。

## 2026-08-21 历史状态

> 2026-08-21 17:30（Asia/Shanghai）｜数据集 v1 已进入模型训练阶段；正式视频分析计划在 2026-08-23 晚回珠海后启动准备。

- 当时训练电脑为 AMD RX 6750 GRE 12GB 设备，训练使用 CPU，计划完成 100 epochs。
- 当时尚未回传确认最终模型参数，具体模型变体、训练批次、`imgsz`、`batch` 和最终权重名称以训练目录中的 `args.yaml` 与结果文件为准。
- 当时计划将候选权重迁移到 NVIDIA 电脑，用于一百多个约 25 分钟视频的正式推理。
- 数据集 batch1/batch2 的环境划分继续保留，用于数据来源与环境差异溯源。
- 眼框模型不是最终眼动指标模型；后续实际正式路线最终冻结为 YOLO26n + RITnet。

> 2026-08-19（Asia/Shanghai）｜为 NIR 专用 YOLO 眼框检测器第一轮训练抽帧；两批实验环境分开记录。

## 批次定义（用户确认）

| batch | 环境 | 被试 | 人数 | block/人 | 抽帧数 |
| --- | --- | --- | --- | --- | --- |
| batch1 | 环境一：桌子矮一点、被试更容易离开画面 | `sub-011~030` + `sub-9504` | 21 | 3 | 315 |
| batch2 | 环境二：桌子正常 | `sub-031~055` + `sub-061` | 26 | 2 | 260 |
| 合计 | | | 47 | | 575 |

- 边界：**>=31 才算环境二**；`sub-030`（3 block）属环境一。
- `sub-9504` 归 batch1（全脸取景、特殊编号）。

## 抽帧策略

- 每 `(subject, block)` **按时间均匀**抽 5 帧，只在 `block_start/block_stop` 任务窗口内（`master_timeline.csv`）。
- 目标时刻 = block 内等距 5 个区间中点，边距 `margin=10s` 避开 block 起止过渡帧；用 `unix_ms → frame_idx` 映射。
- **排除**：baseline/对准/静息、cover、instructions、practice、block 间休息、sart_stop 后结束段、batch2 Block2 后空档。
- 确定性 seed `20260817`；同 `(subject,block)` 帧号去重（碰撞 +1 帧）。
- 每帧灰度 JPG（1920×1080），命名 `subject_<stem>_frame_<6位帧号>.jpg`。

## 目录结构

```text
nir-eye-v1/
├── images/batch1/          # 315 张原图
├── images/batch2/          # 260 张原图
├── labels_yolo/batch1/     # LabelImg 导出（YOLO txt，与图片同名）
├── labels_yolo/batch2/
├── manifests/
│   ├── frames.csv          # 抽帧清单（image_id,subject,batch,block,frame_idx,unix_ms,image_path,source_video）
│   ├── annotations.csv     # 标注状态
│   ├── split_subject.csv   # 两批 train/val/test 被试划分（不按帧）
│   ├── summary_by_subject.csv
│   └── split_batch{1,2}_{train,val,test}.txt
├── previews/
├── classes.txt             # 单行 `eye`
├── dataset_batch1.yaml
└── dataset_batch2.yaml
```

## 标注方法记录（LabelImg）

当时使用 LabelImg 1.8.6 + Python 3.10 的专用 Windows venv。为解决中文路径和 Qt 问题，环境内增加了自定义启动器；同时对 LabelImg 的滚轮/缩放 `setValue(float)` 问题做了 4 处 `int(...)` 修正。

这些真正需要长期保留的兼容逻辑已经抽取到：

```text
tools/labelimg/
├── README.md
├── requirements.txt
├── launch.py
└── patch_labelimg.py
```

以后重建标注环境按 `tools/labelimg/README.md` 执行，不需要依赖原机器的完整 venv。仓库根 `venv-labelimg/` 目前只作为待清理的历史环境暂存；在获得删除许可前不会直接删除。

历史标注规则：

1. Open Dir = `images/batch1/`（先标 batch1，再标 batch2）。
2. Save Dir = `labels_yolo/batch1/`（与图片同名 `.txt`）。
3. 格式选 YOLO；`classes.txt` 只含 `eye`。
4. 单类 `eye`、不区分左右；框含眼裂/上下眼睑 + 少量眼周；闭眼仍标；无眼不标框并在 `annotations.csv` 记 `no_eye`。
5. 完成后保留 `images/ + labels_yolo/ + manifests/` 作为数据版本证据。

## 训练结果映射

原始两批数据最终合并为固定的被试级 train / val / test，训练工作区位于：

```text
../../training/nir-eye-yolo/
```

训练结果与正式 runtime 的关系：

```text
datasets/nir-eye-v1/
        ↓ 整理/固定 split
training/nir-eye-yolo/
        ↓ YOLO26n 100 epochs
training/nir-eye-yolo/runs/yolo26n_eye_100epoch/weights/best.pt
        ↓ 冻结副本
runtime/nir-formal/models/nir-eye-yolo26n-best.pt
```

数据版本仍记为 v1；只有在标注规范、来源批次、split 或标签定义发生实质变化时才建立新的正式数据版本。

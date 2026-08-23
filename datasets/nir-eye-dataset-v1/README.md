# nir-eye-dataset-v1｜NIR 眼框训练数据集（两批分开）

## 当前状态（2026-08-23）

`nir-eye-dataset-v1` 是 NIR 眼框数据集 v1 的**冻结源数据 / 标注数据版本**，用于保留原始抽帧、两批实验环境、YOLO 标注及数据 provenance。

当前状态：

- 数据集 v1 已完成抽帧与眼框标注，共约 **47 个被试 / 575 张抽帧图片**。
- `batch1` / `batch2` 的环境划分继续保留，用于记录不同采集条件，不因后续合并训练而删除。
- 本数据集已经派生出仓库顶层的 `yolotrain/` 训练工作区；后者将两批数据合并，并按被试进行固定的 train / val / test 划分。
- `datasets/nir-eye-dataset-v1/manifests/split_subject.csv` 与 `yolotrain/split_subjects.csv` 当前内容一致，Git blob SHA 均为 `38a48242d46ab40997e95664f3b22f593f8622e8`，说明训练工作区继承了 dataset v1 的被试级划分，而不是另行随机划分。
- YOLO26n 眼睛检测器已经完成训练；当前 GitHub 分支中最终训练权重位于 `yolotrain/runs/yolo26n_eye_100epoch/weights/best.pt`。
- 该 YOLO26n 模型已经进入后续 NIR pipeline，并已完成正式 NIR 全量分析。

因此，本目录当前的主要职责不是继续充当“待训练工作区”，而是作为：

```text
冻结源数据 + 原始标注 + 环境批次信息 + 数据划分 provenance
```

训练过程、训练产物和最终权重由 `yolotrain/` 负责保存。

## 与 `yolotrain/` 的关系

```text
datasets/nir-eye-dataset-v1/
        ↓
冻结源数据 + 原始标注 + batch provenance
        ↓
yolotrain/
        ↓
合并 batch1 / batch2
按被试划分 train / val / test
生成训练 manifest
YOLO26n 训练
        ↓
yolotrain/runs/yolo26n_eye_100epoch/weights/best.pt
        ↓
后续正式 NIR pipeline
```

`v1` 表示的是**数据集版本**，不是项目版本。后续如增加困难样本，可以建立新的数据集版本或追加版本，但不应覆盖本目录所记录的 v1 provenance。

---

# 历史记录

以下内容保留 2026-08-19 至 2026-08-21 的数据构建、标注和训练规划记录。其计划性表述反映当时的研究阶段，**不代表当前项目仍处于训练准备或正式分析准备阶段**。

## 2026-08-21 训练与正式分析状态（历史）

> 2026-08-21 17:30（Asia/Shanghai）｜数据集 v1 已进入模型训练阶段；正式视频分析计划在 2026-08-23 晚回珠海后启动准备。

- 当前训练电脑为 AMD RX 6750 GRE 12GB 设备，训练使用 CPU，计划完成 100 epochs。
- 本 README 不预设尚未回传确认的具体模型变体、训练批次、`imgsz`、`batch` 或最终权重名称；这些信息以训练目录中的 `args.yaml` 和结果文件为准。
- 训练完成后，候选权重可能迁移到 NVIDIA 电脑，用于一百多个约 25 分钟视频的正式推理准备。
- 数据集 batch1/batch2 的环境划分仍然保留；正式采用一个模型还是分批模型，须结合验证结果及下游 RITnet 连续性结果决定。
- 眼框模型不是最终眼动指标模型。正式路线仍需在短视频中验证：YOLO 重新检测、tracking、RITnet、失败状态记录和连续检测能力。

> 2026-08-19（Asia/Shanghai）｜为 NIR 专用 YOLO 眼框检测器第一轮训练抽帧；两批实验环境分开训练两个模型。

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

## 目录结构（对齐 docs/010-nir/08-17-02）

```text
nir-eye-dataset-v1/
├── images/batch1/          # 315 张原图
├── images/batch2/          # 260 张原图
├── labels_yolo/batch1/     # LabelImg 导出（YOLO txt，与图片同名，标注后回传）
├── labels_yolo/batch2/
├── manifests/
│   ├── frames.csv          # 抽帧清单（image_id,subject,batch,block,frame_idx,unix_ms,image_path,source_video）
│   ├── annotations.csv     # 标注状态占位（按 08-17-02 字段）
│   ├── split_subject.csv   # 两批 train/val/test 被试划分（不按帧）
│   ├── summary_by_subject.csv
│   └── split_batch{1,2}_{train,val,test}.txt   # 各 split 图片绝对路径清单
├── previews/
├── classes.txt             # 单行 `eye`
├── dataset_batch1.yaml
└── dataset_batch2.yaml
```

## 标注（轻薄本 + LabelImg）

LabelImg 专用环境：`venv-labelimg`（Python 3.10，base=`D:\psychopy`）。用干净启动器 `labelimg_launch.py` 启动（它自动把 Qt 插件拷到 ASCII 路径并设置环境变量，规避中文路径的 Qt 崩溃）：

**启动（首选快捷方式）**——**必须用 venv 的解释器，不要用 `C:\Python314` 的 labelImg（没打补丁）、也不要双击 `labelimg.bat`（双击环节有问题）**：

- **双击桌面 `labelImg.lnk`** 或 v2 根目录 `labelImg.lnk`（用 `pythonw.exe`，无控制台窗口）。
- 备用（PowerShell 粘贴）：

```powershell
& "D:\AAAWORK\07-竞赛\厚璨杯\021-analysisplan\attention-pipeline-v2\venv-labelimg\Scripts\python.exe" "D:\AAAWORK\07-竞赛\厚璨杯\021-analysisplan\attention-pipeline-v2\venv-labelimg\Scripts\labelimg_launch.py"
```

打开后点 **Open Dir** 选 `images\batch1`、**Change Save Dir** 选 `labels_yolo\batch1`，顶部格式下拉从 PascalVOC 切到 **YOLO**。已修复：Qt 插件中文路径崩溃（launcher 自动拷贝到 ASCII 路径）+ 滚轮/缩放 `setValue(float)` 崩溃（`labelImg.py` 4 处 `int()`）。

1. Open Dir = `images/batch1/`（先标 batch1，再标 batch2）。
2. Save Dir = `labels_yolo/batch1/`（与图片同名 `.txt`）。
3. 格式选 YOLO；`classes.txt` 只含 `eye`。
4. 规则（08-17-02 §1）：单类 `eye`、不区分左右；框含眼裂/上下眼睑+少量眼周；闭眼仍标；无眼不标框并在 `annotations.csv` 记 `no_eye`。
5. 完成后把 `images/ + labels_yolo/` 回传本机（或整目录拷回）。

## 训练（当时规划，历史）

- 两批各自训练：`yolo11n.pt`（对照 `yolov8n.pt`），`data=dataset_batch{1,2}.yaml`，NIR 灰度复制 3 通道，`imgsz` 先试 1280（CPU 慢可降 640/960）。
- 按 `split_subject.csv` 的被试划分评测；最终按 mAP + PuReST 下游（检出率/连续性）选模型。
- 数据版本 v1；后续困难画面补抽记为 v1.x 追加集。

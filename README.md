# Attention-Analysis｜AMD RGB 工作线

> 当前 branch：`rgb-amd`。这是 Attention-Analysis 的 **AMD RGB 并行开发工作线**，不是独立项目，也不是只允许放 RGB 文件的子仓库。NIR、Behavior、NIR-Behavior、共享配置和共享 docs 可以从 `amd-DirectML` 同步进来；RGB 通过验收后的成熟改动也会再回并综合线。

> 分支关系与同步规则见 [`docs/010-overview/015-并行分支与同步约定.md`](docs/010-overview/015-并行分支与同步约定.md)。日期型 `docs/工作记录/` 保留历史原文，不追溯改写。

## 当前工作目录

```text
D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-rgb-amd
```

每次开始工作：

```powershell
cd "D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-rgb-amd"
git status --short --branch
git branch --show-current
git fetch origin --prune
git pull --ff-only
```

正常应看到：

```text
rgb-amd
```

## 当前项目状态

| 模块 | 当前状态 |
|---|---|
| NIR | 正式 YOLO26n + RITnet runtime 已存在；AMD DirectML full-class 补充分析资产可从综合线同步 |
| Behavior | FocusWave v3.1.3 BB 正式分析已建立 |
| NIR × Behavior | Unix-ms / trial / probe 对齐、coverage/QC/diagnostics 已建立 |
| RGB Face | Py-Feat 2.1.1 scientific core + DirectML backend 已冻结；15 Hz 已冻结 |
| RGB Pose | MediaPipe Pose 10 Hz science/QC/features 已验证 |
| RGB Motion | full-fps global Motion 已验证 |
| RGB formal single-subject | **完整正式时间段总控入口已实现，当前下一步是 sub-031 从头到尾实机验收** |
| RGB cohort | 44 人 batch + completion/resume 尚未实现 |

## 当前 RGB 正式链

AMD Face 工程基线：

```text
original AVI
→ timestamp-driven 15 Hz
→ reader/preprocess prefetch
→ RetinaFace DirectML B8
→ decode/NMS + square-reflect crop
→ pooled multitask DirectML B16
→ full scientific raw outputs
→ continuous tracking / primary face
→ eyelid / openness derived
```

当前单被试自动流程：

```text
face_formal_prepare.py
→ rgb_formal_motion_pose.py
→ face_formal_directml.py
→ face_formal_derive.py
```

总控入口：

```text
scripts/run_rgb_formal_subject.ps1
```

当前工程优先级：

```text
sub-031 单被试正式全程实机验收
→ 修复实际出现的 orchestration / environment / output 问题
→ 44 人 batch + resume
→ body_motion_energy
→ blink / perclos80_proxy 最终科学规则
```

时间戳 gap 继续保留为 QC 信息，但不再单独阻挡首个完整 pipeline 验收。

## 环境

AMD RGB 继续使用彼此隔离的环境：

| 任务 | Conda 环境 |
|---|---|
| RGB audit / Motion / Pose / Face frame preparation / QC | `D:\CondaEnvs\attention-rgb` |
| Face ONNX Runtime DirectML 正式推理 | `D:\CondaEnvs\attention-face-directml` |
| Py-Feat reference / ONNX export | `D:\CondaEnvs\attention-face-pyfeat` |
| LibreFace historical reference/export | `D:\CondaEnvs\attention-face-libreface` |

不要为了方便把这些依赖塞进同一个环境。

## 正式原始数据与输出

AMD 当前数据 discovery roots 保留：

```text
E:/正式实验
F:/正式实验
E:/Data
F:/Data
```

实际运行以本机 discovery 为准；`sub-9504` 不属于正式 cohort。

RGB 输出统一位于仓库外：

```text
D:\_AttentionData\Beijing-RGB
```

测试/benchmark：

```text
D:\_AttentionData\Beijing-RGB\_test
```

正式 subject：

```text
D:\_AttentionData\Beijing-RGB\sub-XXX
```

Git pull / switch / merge 不管理这些正式结果。

## 文档入口

| 我想找 | 入口 |
|---|---|
| 分支关系与同步原则 | [`docs/010-overview/015-并行分支与同步约定.md`](docs/010-overview/015-并行分支与同步约定.md) |
| 项目整体结构 | [`docs/010-overview/`](docs/010-overview/) |
| NIR | [`docs/020-nir/`](docs/020-nir/) |
| Behavior | [`docs/030-behavior/`](docs/030-behavior/) |
| RGB 当前状态 | [`docs/040-rgb/README.md`](docs/040-rgb/README.md) |
| RGB 输出 Schema / 信息保留 | [`docs/040-rgb/044-RGB输出Schema与信息保留原则.md`](docs/040-rgb/044-RGB输出Schema与信息保留原则.md) |
| AMD RGB 环境与命令 | [`docs/040-rgb/045-RGB开发环境与运行指令.md`](docs/040-rgb/045-RGB开发环境与运行指令.md) |
| 技术决策 | [`docs/050-decisions/`](docs/050-decisions/) |
| 历史工作记录 | [`docs/工作记录/`](docs/工作记录/) |
| 当前 scripts 索引 | [`scripts/README.md`](scripts/README.md) |

完整导航见 [`docs/README.md`](docs/README.md)。

## 历史与 provenance

历史工作记录、旧 `rgb-dev` 名称、旧候选 backend、阶段性“待完成”表述按当时语境保留，不代表当前状态。判断现在该怎么运行时，优先看本 README、`docs/README.md`、`docs/040-rgb/README.md`、`docs/040-rgb/045-RGB开发环境与运行指令.md` 和 `scripts/README.md`。

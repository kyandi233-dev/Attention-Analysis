# AMD NIR 数据目录结构

> **本目录只描述 AMD 这台电脑上的 NIR 开发批次（44 场开发 snapshot）。**
> 它**不是 formal cohort**。formal cohort 的口径由 `D:\_AttentionData\_Formal` 与证据库
> `资产导航\`、`NIR分析\` 综合定义（当前为 116 场 / 61 个匿名分析组）。
>
> 依据：`D:\_AttentionData\AGENTS.md:18` ——「44场开发snapshot和116场formal cohort
> 使用独立目录、manifest和状态说明」。
>
> **不要把本机产物称为"正式"**。历史上这样做已在跨电脑合并时造成混淆：
> 本机 44 场与 formal cohort 116 场是**并列关系，不是包含关系**。

正式数据根：

```text
D:\_AttentionData\Beijing-NIR\amd-directml
```

整理后只需要人工识别以下五个目录：

```text
historical-yolo\
final-topology\
validation\
runtime\
archive\
```

这些名字没有编号含义，直接按用途命名。

## historical-yolo

历史正式 YOLO 输出，仅作为 bbox/source provenance 使用。最终 RITnet 管线严格复用这些 `eyes.csv` / `frames.csv`，绝不重新跑 YOLO。

**注意：该目录下同时存在 RITnet 产物，这是当前引擎的正常行为。** 原因是引擎把输出锚定在**源场次目录的父目录**：

```python
# ritnet_fullclass_final_engine.py:521
subject_dir = context.run_dir.parent / output_dirname / context.subject
# ritnet_fullclass_source.py:272
run_dir = Path(run_dir).resolve()      # 解析 junction
```

由于整理脚本把源场次移入 `historical-yolo\`，且数据根的同名入口是 junction，`.resolve()` 会把它们解析回 `historical-yolo\`，因此

> **`historical-yolo\ritnet-fullclass-final\` 才是当前 41 个场次的实际输出位置**，而不是 `final-topology\`。

## final-topology

早期格式的 cohort 输出。**当前只有 `sub-032`、`sub-033` 两场以本目录为唯一权威副本**
（producer `6045a3ff`，`analysis_domain_version=v3-primary-pupil-topology`，`eye_metrics_schema_version=6`）。

- **`sub-035`：旧域副本，必须弃用。** 它的 `analysis_domain_version` 是
  `source-backed-output-mask-v2-pupil-geometry-only`、config `69de683f…`、producer `100b64e1`。
  权威版本在 `historical-yolo\ritnet-fullclass-final\sub-035`（`v3`，config `f849ad35…`）。
  实测两者同场次 106 列中 87 列相同、19 列不同（`pupil_center_y` 最大差 1805 px）。
- **`sub-034`、`sub-036`、`sub-037`：空壳。** 只有空的 `data\` 子目录，没有
  `completion.json` / `manifest.json` / `summary.json`。**不得当作完成包。**

## validation

方法验证证据。当前保留 `pupil-geometry\sub-031`（三算法 Legacy / Topology / EllSeg）。

`sub-031` **属于 44 场队列成员**，且其唯一权威产物就在这里：producer `cb7c8fe6`、
分支 `amd-DirectML-geometry-validation`、**`eye_metrics_schema_version = 7`**。
读取时必须同时记录 schema 与分支，**不得与 schema 6 的包静默拼接**。

## runtime

程序运行状态，不是科学结果：

- `checkpoints/final-topology`：SQLite interruption recovery；
- `logs`：逐被试 batch summary。

中断后恢复正式计算依赖 checkpoint。**该目录只覆盖早期 5 场**（`sub-031`/`032`/`033`/`034`部分/`036`部分）；
当前 41 场的 checkpoint 在 `historical-yolo\.ritnet-fullclass-work\`。

## archive

仅保存失败、中断、旧版、备份、开发/smoke 产物。这里的内容不是当前结果。

- **`archive\ritnet-fullclass-final\` 不是第三份结果。** 它只有
  `sub-031`/`032`/`033`/`034`/`036` 五个子目录，且 **0 个 `completion.json`**，
  是被取代或部分状态的归档。

## 44 场产物实际路由（2026-09-15 现场核验）

三棵树的**目录并集 = 44 场，双向差集均为空**；有完成包的并集同样是 44：

| 场次 | 权威位置 | producer |
|---|---|---|
| **41 场**（`sub-034`…`sub-177`） | `historical-yolo\ritnet-fullclass-final\` | `dd56f7fc`（= 当前 HEAD） |
| `sub-032`、`sub-033` | `final-topology\` | `6045a3ff` |
| `sub-031` | `validation\pupil-geometry\` | `cb7c8fe6`（独立分支，schema 7） |

41 个 `historical-yolo` 包的 `work_identity` 完全一致：

```text
git_commit                  = dd56f7fc9faccee4d06bb356e9d2df91088f38d3
eye_metrics_schema_version  = 6
analysis_domain_version     = source-backed-output-mask-v3-primary-pupil-topology
ritnet_model_sha256         = cfb38794… (ritnet-b16-fp32-uncertainty.onnx)
config_sha256               = f849ad35f5f442e047f4e3fb39aca98f09a67ccbe7b7adf7b9b3991d4727c73f
roi_algorithm_version       = fixed-aspect-1p6-expanded-context-replicate-v2
```

同一路由另见证据库：`资产导航\1.1:74`、`资产导航\1.3:97`、`NIR分析\1.5:20-21`、
`运行记录与证据\08-29-09:43-44`。

## 正式操作入口

整理目录：

```powershell
powershell -ExecutionPolicy Bypass -File ".\organize_amd_data_root.ps1"
```

从 sub-033 开始运行：

```powershell
powershell -ExecutionPolicy Bypass -File ".\run_amd_final_topology.ps1" -MinSubject 33
```

正式 launcher 按被试逐个运行；任一被试返回错误即停止，不继续启动后续被试，避免系统级 OOM 后连续污染后续任务。

> **注意**：`run_amd_final_topology.ps1` 枚举 `historical-yolo\` 找被试，却把 `--output $Root`
> 传下去；由于引擎以 `run_dir.parent` 为锚点，实际写入的是 `historical-yolo\ritnet-fullclass-final\`。
> **若希望产物落到 `final-topology\`，需要修改引擎的路径锚点，而不是只改 launcher 参数。**

# AMD NIR 数据目录最终结构

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

## final-topology

唯一当前正式 cohort 输出：

```text
历史 YOLO bbox
→ 原始 NIR AVI
→ fixed 1.6 ROI
→ 640×400
→ RITnet FP32 b16
→ primary-pupil-topology
→ OpenCV pupil ellipse
→ compact scalar/QC
```

每个被试只有有效 `completion.json` 才算完成。

## validation

方法验证证据，不是正式 cohort。当前保留 pupil geometry 三算法 Legacy / Topology / EllSeg 的 sub-031 验证结果。

## runtime

程序运行状态，不是科学结果：

- `checkpoints/final-topology`：SQLite interruption recovery；
- `logs`：逐被试 batch summary。

中断后恢复正式计算依赖 checkpoint，因此全量完成前不要删除。

## archive

仅保存失败、中断、旧版、备份、开发/smoke 产物。这里的内容不是当前正式结果。

## 兼容旧路径

当前 Python 代码仍使用少量历史路径名。整理脚本会为这些路径建立隐藏 Junction，例如：

```text
ritnet-fullclass-final  -> final-topology
.ritnet-fullclass-work -> runtime\checkpoints\final-topology
_archive               -> archive
```

这些兼容入口只供程序使用，不作为人工数据目录。

如果本地曾经拉取过短暂使用的 `00/10/20/90/99` 数字目录版本，整理脚本会尝试把它们统一迁回这里定义的普通目录名，不把数字前缀作为最终规范。

## 正式操作入口

整理目录：

```powershell
powershell -ExecutionPolicy Bypass -File ".\organize_amd_data_root.ps1"
```

从 sub-033 开始正式运行：

```powershell
powershell -ExecutionPolicy Bypass -File ".\run_amd_final_topology.ps1" -MinSubject 33
```

正式 launcher 按被试逐个运行；任一被试返回错误即停止，不继续启动后续被试，避免系统级 OOM 后连续污染后续任务。

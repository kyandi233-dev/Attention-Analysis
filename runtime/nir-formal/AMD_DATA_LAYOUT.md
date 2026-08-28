# AMD NIR 数据目录最终结构

正式数据根：

```text
D:\_AttentionData\Beijing-NIR\amd-directml
```

整理后只需要人工识别以下五个目录：

```text
00-source-yolo-historical\
10-final-topology\
20-validation\
90-runtime\
99-archive\
```

## 00-source-yolo-historical

历史正式 YOLO 输出，仅作为 bbox/source provenance 使用。最终 RITnet 管线严格复用这些 `eyes.csv` / `frames.csv`，绝不重新跑 YOLO。

## 10-final-topology

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

## 20-validation

方法验证证据，不是正式 cohort。当前保留 pupil geometry 三算法 Legacy / Topology / EllSeg 的 sub-031 验证结果。

## 90-runtime

程序运行状态，不是科学结果：

- `checkpoints/final-topology`：SQLite interruption recovery；
- `logs`：逐被试 batch summary。

中断后恢复正式计算依赖 checkpoint，因此全量完成前不要删除。

## 99-archive

仅保存失败、中断、旧版、备份、开发/smoke 产物。这里的内容不是当前正式结果。

## 为什么资源管理器可能仍看到旧名字

当前 Python 代码仍使用部分历史路径名。整理脚本会为这些路径建立 **Hidden Junction** 作为兼容入口，例如：

```text
ritnet-fullclass-final  -> 10-final-topology
.ritnet-fullclass-work -> 90-runtime\checkpoints\final-topology
_archive               -> 99-archive
```

这些兼容入口默认隐藏，仅供程序使用，不再作为人工数据目录。

## 两个正式操作入口

整理目录：

```powershell
powershell -ExecutionPolicy Bypass -File ".\organize_amd_data_root.ps1"
```

从 sub-033 开始正式运行：

```powershell
powershell -ExecutionPolicy Bypass -File ".\run_amd_final_topology.ps1" -MinSubject 33
```

正式 launcher 按被试逐个运行；任一被试返回错误即停止，不继续启动后续被试，避免系统级 OOM 后连续污染后续任务。

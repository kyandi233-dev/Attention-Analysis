# 08-26-18｜NIR frame-level edge gate 与 segmentation-quality 审计

## 开始记录

### 背景

Issue #17 要求在继续冻结 30 s / 60 s time-on-task coverage 规则前，审计 frozen RITnet full-class 输出中的整 mask edge gate、最大轮廓触边和孤立误分割，并检查 normalization-valid 帧是否仍有明显 segmentation-quality 异常。

### 目标

在 `analysis/multimodal-integration` 分支对 44 人正式 block1/block2 full-class 结果进行只读审计：复用现有 sparse QC labels/overlays，按 frozen 320×160 分析坐标重建 pupil 与 iris_outer 的整 mask / 最大外轮廓 edge 关系，汇总已有质量字段和 frame-to-frame PIR jump，生成 bounded 外部审计副本，并将事实性结果回复 Issue #17。

### 计划与风险

1. 核验当前分支、工作区和 frozen runtime 的类别映射、resize 与 edge 语义。
2. 新增独立审计脚本，不改生产 CSV、runtime 或已有分析输出。
3. 使用 full-class CSV 的 block1/block2 行进行全 cohort 描述；使用已有 sparse QC labels 对 mask-level 最大轮廓关系做有明确分母的审计。
4. 对质量诊断只做描述性汇总，不设置正式阈值、不删除 subject/eye/Block/frame、不进入统计模型。
5. 运行脚本与相关测试，检查输出完整性和扫描 warning/error。

### 校验

- 生产数据仅读；输出写入仓库外 `D:\_AttentionData\Beijing-NIR\analysis\nir-behavior-v2\cohort-44-exploratory\04_frame_quality_audit\`。
- 低 usable subject 的解释保持事实性，不把 edge-touch 自动等同于真实结构截断。
- 若 sparse labels 无法支持全帧最大轮廓比例，将在输出和 Issue 中明确标注采样分母，不以 sparse 结果冒充全量结果。

## 完成记录

### 总结

已在 frozen `analysis/multimodal-integration` 代码和本地 AMD/DirectML 数据上完成 44 人正式 block1/block2 的 frame-level edge 与 segmentation-quality 描述性审计。44/44 subject 读取成功，共 2,919,835 个正式 Block 帧；已有 sparse QC labels 成功重建 1,824 个 Block 行。生产 CSV 的 pupil/iris_outer current edge flag 与 label 重建的 whole-mask edge 在 sampled 行上完全一致。

### 执行与决策过程

- 新增 `scripts/nir_pir_frame_quality_audit.py`，按列名读取 CSV，使用 frozen 类别映射和 320×160 nearest-neighbor resize 重建 mask。
- 全帧结果只使用生产 CSV 中已有 edge、normalization、component、fill、ellipse/contour、center offset、PIR jump 字段；largest contour 与 stray-only 结果只对实际存在的 sparse QC label 行计算。
- bounded examples 复用已有 label/overlay，仅复制到仓库外审计目录；源视频可访问的示例附 Laplacian variance，不设置 blur cutoff。
- 未修改 frozen CSV、runtime、已有 alignment、Behavior 数据或 coverage/concordance 规则；未删除 subject/eye/Block/frame，未运行 formal alignment 或显著性模型。

### 校验结果

- `D:\CondaEnvs\nir-amd\python.exe -m py_compile scripts/nir_pir_frame_quality_audit.py`：通过。
- 44 人审计脚本：通过，`scan_errors=0`、`scan_warnings=0`。
- `tests/test_nir.py tests/test_nir_behavior_alignment.py tests/test_nir_behavior_cohort_qc.py tests/test_formal_nir.py`：23 passed。
- `runtime/nir-formal/tests`：19 passed。
- 全仓库 `pytest -q` 未能收集完成，原因是当前 conda 环境缺少 `scipy`，行为测试导入时报 `ModuleNotFoundError: No module named 'scipy'`；本轮未安装依赖，也未因该环境问题修改代码。

### 最终产物

审计目录为 `D:\_AttentionData\Beijing-NIR\analysis\nir-behavior-v2\cohort-44-exploratory\04_frame_quality_audit\`，包含 cohort/subject×eye×Block edge 汇总、valid/invalid segmentation-quality 汇总、sparse mask audit、四类 bounded examples、README、scan error/warning 和运行 metadata。

### 待确认事项

largest-contour 与 stray-component 的 cohort 比例目前是 sparse QC label 分母上的事实，不应解释为 2,919,835 个全帧的直接估计。是否将 current edge gate 改为 largest-contour gate，或增加独立 segmentation-quality gate，留待后续方法决策；本轮不冻结规则。

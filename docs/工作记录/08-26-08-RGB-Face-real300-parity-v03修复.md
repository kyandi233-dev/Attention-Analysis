# 08-26-08｜RGB Face real-300 parity v0.3 修复

> 2026-08-26｜分支：`rgb-dev`｜承接 `08-26-07-RGB-Face-真实300帧DirectML验证实现.md`。本记录只修复 parity 统计入口，不重跑或修改任何 LibreFace / Py-Feat real-300 推理输出。

## 1. 首次实机问题

`face_real_parity_v02.py` 在首次 real-300 验证中暴露两个验证层问题：

1. Py-Feat parity 计算 Spearman 时，`pandas.Series.corr(method="spearman")` 会调用 SciPy；初始 `attention-face-directml` 环境未安装 SciPy，因此报 `ModuleNotFoundError: scipy`。该问题发生在结果读取和人脸匹配之后，不代表 Py-Feat 推理或数据失败。
2. LibreFace CPU reference 与 DirectML AU parquet 使用不同列名，例如 CPU `au_1_intensity`、DML `AU01`。v0.2 先 merge 再假定所有列都会得到 `__cpu/__dml` suffix，但 pandas 只对重名列加 suffix，因此访问 `au_1_intensity__cpu` 时触发 `KeyError`。该问题同样属于 parity schema adapter，不代表 LibreFace 推理失败。

## 2. v0.3 修复

新增：

```text
scripts/face_real_parity_v03.py
```

修复原则：

- 保留 `face_real_parity_v02.py` 作为已经执行过的历史入口，不覆盖其实现；
- LibreFace 对于同义但不同名的 numeric outputs，分别按原模型输出顺序映射到统一临时列，再按 `benchmark_index` 合并；
- 如果双方存在真正同名 numeric columns，则优先按同名语义直接比较；
- 输出同时记录 `cpu_columns`、`dml_columns`、`ordered_column_map`，避免 schema 映射成为隐式假设；
- Spearman 改为“average rank 后计算 Pearson”，只依赖 pandas + NumPy，不要求 DirectML runtime 为 parity 额外安装 SciPy；
- 不重新运行任何模型，不修改任何现有 parquet / manifest。

当前最新代码 commit：`41b8c79`（`fix(rgb): finalize parity v0.3 CLI output`）。

## 3. 后续固定命令

LibreFace：

```powershell
python scripts/face_real_parity_v03.py `
  --candidate libreface `
  --benchmark-dir "D:\_AttentionData\Beijing-RGB\_test\face-continuous\sub-031" `
  --prep-dir "D:\_AttentionData\Beijing-RGB\_test\face-directml\real300\sub-031\libreface-prep" `
  --dml-dir "D:\_AttentionData\Beijing-RGB\_test\face-directml\real300\sub-031\libreface" `
  --output "D:\_AttentionData\Beijing-RGB\_test\face-directml\real300\sub-031\parity\libreface_parity.json"
```

Py-Feat：

```powershell
python scripts/face_real_parity_v03.py `
  --candidate pyfeat `
  --benchmark-dir "D:\_AttentionData\Beijing-RGB\_test\face-continuous\sub-031" `
  --dml-dir "D:\_AttentionData\Beijing-RGB\_test\face-directml\real300\sub-031\pyfeat" `
  --output "D:\_AttentionData\Beijing-RGB\_test\face-directml\real300\sub-031\parity\pyfeat_parity.json"
```

生成的 JSON `schema_version` 应为：

```text
rgb-face-real300-parity-v0.3
```

## 4. 解释边界

本次两类报错均发生于 CPU-reference parity 的统计适配层。已有 LibreFace / Py-Feat DirectML real-300 输出继续有效，不允许因为 parity 脚本错误而重跑昂贵推理。只有 v0.3 报告真正显示 coverage / bbox / scientific-output drift 后，才进入模型层解释。

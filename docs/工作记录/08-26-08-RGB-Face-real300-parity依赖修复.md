# 08-26-08｜RGB Face real-300 parity 依赖修复

> 2026-08-26｜分支：`rgb-dev`｜承接 `08-26-07-RGB-Face-真实300帧DirectML验证实现.md`。

## 问题

在 `attention-face-directml` 环境运行 `scripts/face_real_parity_v02.py --candidate pyfeat` 时，前面的 CPU/DML parquet 读取、人脸匹配与 FaceScore 比较均已进入执行，但在计算 Spearman 相关时由 pandas 内部调用 `scipy.stats.spearmanr`，环境中未安装 SciPy，报：

```text
ModuleNotFoundError: No module named 'scipy'
```

这不是 DirectML 推理、Py-Feat 输出或 parity 数据错误，也不需要重跑 300 帧推理。

## 修复

在现有 DirectML 环境补装 SciPy：

```powershell
conda activate "D:\CondaEnvs\attention-face-directml"
python -m pip install scipy
python -c "import scipy; print(scipy.__version__)"
```

然后仅重新运行 LibreFace / Py-Feat parity 命令。

## 复现说明

`face_real_parity_v02.py` 使用 `pandas.Series.corr(method="spearman")`；pandas 对 Spearman 的实现依赖 SciPy。因此后续新机器配置 `attention-face-directml` 时，环境依赖中应包含 `scipy`。

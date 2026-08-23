# Runtime

`runtime/` 只保存可直接运行或用于重建运行环境的内容，不保存正式分析结果。

## 当前正式运行包

```text
runtime/nir-formal/
```

`nir-formal/` 是已经用于正式全量 NIR 分析的自包含 GPU 运行包。换电脑时，优先直接复制或拉取这个目录，并按照其中的 `README.md` 和 `requirements.txt` 重新配置环境。

包内主要内容：

- `README.md`：完整运行说明；
- `config.yaml`：FocusWave v3.1.3、YOLO26n、RITnet 与 batch 参数；
- `run_pipeline.py`：单被试/诊断/正式分析入口；
- `run_formal_batch.py`：多被试正式批处理；
- `phase_windows.py`：正式 phase 时间窗；
- `ritnet_runtime.py` 与 `ritnet/`：RITnet 运行逻辑；
- `models/`：冻结的 YOLO26n 与 RITnet 权重；
- `requirements.txt`：当前正式运行包依赖；
- `tests/`、`SHA256SUMS.txt`：运行包校验与测试。

## Legacy

`legacy/` 只保存旧环境快照，不是当前 `nir-formal/` 的依赖：

- `requirements-main.txt`：历史主环境完整 pip 快照；
- `requirements-pupil.txt`：历史 pupil 环境快照；
- `PyPupilEXT-0.0.1-cp310-cp310-win_amd64.whl`：历史 PyPupilEXT wheel。

这些文件保留用于复现旧阶段环境，但新电脑运行当前正式 NIR pipeline 时不需要先安装它们。

## 输出边界

正式实验输出继续写到仓库外的独立分析目录。`runtime/` 的职责是“可运行”，不是长期存放结果。

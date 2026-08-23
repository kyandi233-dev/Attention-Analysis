# NIR package

`src/attention_pipeline/nir/` 保存项目级 NIR 几何、ROI、评估、人工复核与历史连续序列逻辑。它不是当前正式 GPU runtime 的执行入口。

当前已经用于正式全量分析的 NVIDIA/CUDA 路线固定在：

```text
runtime/nir-formal/
```

其中包含冻结的 YOLO26n / RITnet 模型、正式配置、phase-window 逻辑和批处理入口。`src/attention_pipeline/nir/` 中的 `benchmark.py`、`review.py`、`sequence.py`、`sequence_eval.py` 等主要属于正式路线冻结之前的项目级评估/历史兼容实现，暂时保留但不作为 current pipeline 入口；是否最终删除需按仓库删除规则另行授权。

仍被仓库级 current/兼容测试覆盖的共享 NIR 几何与正式接口代码继续保留，避免在未完成依赖验证前破坏回归。

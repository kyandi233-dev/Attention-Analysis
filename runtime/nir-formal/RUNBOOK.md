# NIR Runbook｜当前入口

本文件是 `runtime/nir-formal/` 的稳定导航入口，避免根 README 指向不存在的 runbook。

基础 formal producer 的安装、protocol compatibility gate（协议兼容性门控）、数据发现、DirectML 环境与跨机器交接继续以 [`RUNBOOK_V1.md`](RUNBOOK_V1.md) 为准。

当前 RITnet full-class 不再使用历史 fast/320×160 补充路径。full-class 的唯一正式完整方法、全量 400×640 hard-label store、严格 resume（断点恢复）、provenance（来源追踪）、SHA256 与 `sub-031` 验收顺序，以 [`RITNET_FULLCLASS_EXTENSION.md`](RITNET_FULLCLASS_EXTENSION.md) 为准。

正式 full-class 用户入口只有：

```text
run_ritnet_fullclass_extension.py
run_ritnet_fullclass_batch.py
```

每次正式运行前必须先 `git pull --ff-only`、确认工作区干净、运行 `python -m pytest tests -q` 和 `python run_pipeline.py check-env`。在 `sub-031` 的 DirectML、完整性、恢复、磁盘/吞吐与 QC 验收通过前，不启动 AMD 当前 cohort 全量。

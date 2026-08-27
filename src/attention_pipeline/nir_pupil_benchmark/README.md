# NIR pupil benchmark

本包是 Issue #19 的七传统算法测量学 benchmark 实现。`formal.py`/`formal_cli.py` 负责 production evidence 发现、确定性抽样、原视频 source-pixel crop、执行、agreement、manual QC 和完整性输出；`runner.py`/`adapters.py` 负责七算法的统一调用和实际参数 provenance；`core.py`/`schema.py` 负责坐标、结果语义和字段契约。

用户入口是仓库根的 `scripts/nir_pupil_benchmark.py`，配置是 `configs/nir_pypupilext_native_benchmark.yaml`，方法契约是 `docs/020-nir/030-七算法官方API审计与统一Benchmark设计.md`。这里的 RITnet 仅作 agreement comparator，不是人工真值；`algorithm_returned`、`official_valid`、`geometry_sane`、人工 `credibility` 必须分轨。

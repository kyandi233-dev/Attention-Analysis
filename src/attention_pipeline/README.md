# attention_pipeline

`src/attention_pipeline/` 保存项目自身可复用的 Python 源码。当前正式分析入口与历史复现代码需要明确区分，不能因为同处一个 package 就视为同一科研口径。

## 当前核心

| 路径 | 定位 |
|---|---|
| `behavior_formal/` | FocusWave v3.1.3、sub-031+、BB 两正式 block 的当前 Behavior 实现 |
| `nir_behavior/` | frozen full-class NIR × v3.1.3 BB Behavior 的下游 Unix-ms 对齐、窗口特征和 alignment QC；不修改正式 NIR runtime，也不提前融合左右眼 |
| `behavior_bbb_v3_0/` | 用户明确要求保留的历史 BBB v3.0 可执行复现实现 |
| `nir/` | 项目级 NIR 几何、评估与历史兼容逻辑；正式 YOLO26n + RITnet 运行入口以 `runtime/nir-formal/` 为准 |
| `config.py` / `io.py` / `protocol.py` 等 | 共享配置、IO 与协议逻辑 |

旧通用 `cli.py` 与旧 `behavior/`/NIR benchmark-sequence 代码不再作为 current 官方入口；正式 Behavior 使用 `scripts/sart_formal_analysis.py`，正式 NIR 使用 `runtime/nir-formal/`。NIR × Behavior 下游对齐使用 `scripts/nir_behavior_alignment.py`。历史兼容源码是否最终删除需按仓库删除规则单独授权。

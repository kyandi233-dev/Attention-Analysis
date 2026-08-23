# 00｜测试目录与映射

> 2026-08-23｜保留当前文件名作为兼容入口。本轮不通过“新建 README + 删除旧文件”的方式模拟重命名，因为删除旧路径需要单独确认。

## tests/ 的职责

根 `tests/` 负责仓库级代码回归、协议行为验证、历史 NIR 算法契约以及 portable runtime 的最小逻辑测试。它不处理正式全量数据，也不保存正式分析结果。

测试通过只能说明“当前代码满足这些契约”，不能反过来证明某个历史候选已经成为正式路线，也不能替代真实 full-run 的运行 manifest / 输出 QC。

## 当前测试分组

### 1. 通用协议与 I/O

| 文件 | 覆盖范围 |
|---|---|
| `test_protocol_and_config.py` | 协议、阶段/探针定义、配置解析与相关开关 |
| `test_io.py` | dropped 行、AVI位置映射、block/输入解析 |

### 2. 行为分析

| 文件 | 覆盖范围 |
|---|---|
| `test_behavior.py` | RT、QC、d′分母、窗口边界与行为统计基础逻辑 |
| `test_behavior_phase2.py` | 早期行为 phase2 报告/统计契约 |
| `test_behavior_phase3.py` | 早期行为 phase3 契约 |
| `test_behavior_phase4.py` | 早期行为 phase4 契约 |
| `test_behavior_reporting.py` | 早期 reporting 工具 |

正式 BBB SART 分析主体位于 `src/attention_pipeline/behavior_formal/`。旧 behavior phase 测试继续保留，是为了防止历史兼容层回归，而不是说明旧 phase 体系仍是当前唯一行为入口。

### 3. 历史 NIR review / benchmark / sequence

| 文件 | 覆盖范围 |
|---|---|
| `test_nir.py` | NIR 几何、ROI、仿射、椭圆与缺失语义 |
| `test_formal_nir.py` | 正式数据路径/时间轴等 NIR 基础契约 |
| `test_review.py` | 标注/复核设计、seed复现、字段与保存/恢复接口 |
| `test_benchmark.py` | 历史六算法 benchmark / evaluate 逻辑 |
| `test_pupil_adapter.py` | PuRe/PuReST 等历史 pupil adapter |
| `test_roi_backends.py` | MediaPipe/YuNet/YOLO-face/faceparts 等历史 ROI 后端 |
| `test_sequence.py` | 历史连续序列、状态与插值逻辑 |

这些测试继续存在，因为对应历史资产仍需可复现；它们不能用于判断当前 YOLO26n + tracking + RITnet 是否“尚未准入”。

### 4. YOLO26n 模型评价

| 文件 | 覆盖范围 |
|---|---|
| `test_yolo_eye_evaluation.py` | YOLO test 评价中的 bbox 匹配、指标计算等辅助逻辑 |

真正的冻结 test 运行结果记录在 2026-08-22 工作记录；当前分支已不保存当时完整 `artifacts/yolo-eye-evaluation/...` 机器可读目录。

### 5. Portable YOLO + tracking + RITnet runtime

| 文件 | 覆盖范围 |
|---|---|
| `test_portable_nir_gpu_package.py` | portable package 的 subject 规范化、中心跳变门、ROI clipping/尺寸、bbox 合法性和 tracker bbox 转换 |

该测试直接加载：

```text
runtime/nir-yolo-tracking-ritnet-v1/run_pipeline.py
```

所以当前仓库对 portable runtime 已有明确回归契约。但这些单元测试不读取真实 25 分钟正式视频，也不等于视频级 production acceptance；最终全量运行参数仍应以当时实际 `run_manifest.json` 或等价记录为准。

## `pyproject.toml` entry point 注意

当前 `pyproject.toml` 仍定义：

```text
attention-analysis → attention_pipeline.cli:main
attention-pipeline → attention_pipeline.cli:main
```

该 CLI 主要连接早期 behavior 和 NIR review / benchmark / sequence 体系；它不是已经核验到的 YOLO26n + tracking + RITnet full-run 实现入口。为避免破坏现有 package 接口，本轮不直接修改 entry point。完整 portable 实现仍以 runtime package 为准。

## 当前整理原则

1. 不把正式分析脚本搬进 `tests/`。
2. 不把历史 artifacts 当作测试夹具随意覆盖。
3. 对已经全量运行的正式 NIR 管线，测试用于防止代码回归，不用于重新定义项目是否“已准入”。
4. 历史路线测试继续保留并明确角色，不因为当前路线改变而批量删除。
5. 如后续决定将本文件改名为 `README.md`，必须按用户批准的方式处理旧路径；当前不删除任何文件。

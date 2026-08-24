# NIR Formal Runtime

这是 Attention-Analysis `amd-DirectML` 分支的正式 NIR 运行包。它保持既有 NVIDIA 正式分析的科研口径，但使用 ONNX Runtime DirectML，并在 AMD 正式主链中采用经实机 benchmark 确认的 **YOLO fixed batch=8 + RITnet fixed batch=16**。

## 当前正式流程

```text
FocusWave v3.1.3 phase windows
        ↓
连续读取最多 8 帧
        ↓
YOLO26n 640×640 / FP32 / DirectML / fixed batch=8
        ↓
逐帧恢复 bbox，并从原始分辨率帧裁剪扩展眼 ROI
        ↓
RITnet 640×400 / FP32 / DirectML / fixed batch=16
        ↓
320×160 analysis geometry
        ↓
frames.csv / eyes.csv / summary.json / run_manifest.json /
phase_windows.json / completion.json / overlays
```

当前 AMD package version 为 **`0.2.0`**。正式批处理入口为 `run_formal_batch.py`，其内部调用 `run_formal_batched.py`。旧的 `run_pipeline.py` 仍保留 diagnostic、discover、check-env 以及历史兼容逻辑，没有删除。

正式模型资产：

```text
models/nir-eye-yolo26n-best.onnx       # 原 fixed-b1 reference/diagnostic
models/nir-eye-yolo26n-best-b8.onnx    # AMD 正式 YOLO
models/ritnet-b16-fp32.onnx            # AMD 正式 RITnet
models/ritnet-b16-fp32.onnx.data
```

YOLO benchmark 变体 `b4`、`b16` 可保留用于性能追溯；`b1`、`b10`、`b12`、`b14` 的临时导出文件不是正式运行依赖。

## v0.2.0 性能选择依据

同一 `sub-031`、同一 1800 帧区间的 DirectML 测试显示：YOLO fixed batch=8 为当前硬件组合的吞吐 sweet spot；RITnet 则随 batch 从 8→16 持续提升，在 b16 达到本轮测试最高吞吐，因此正式组合冻结为 **YOLO b8 + RITnet b16**。

完整主要计算链 benchmark（视频 decode → YOLO b8 → ROI crop → RITnet b16 → pupil postprocess）处理 1800 帧耗时约 59.02 s，约 **30.50 FPS**。此前完整正式 sub-031 运行约 20.21 FPS，因此本次优化在该机器/数据段上约提升 **50.9%**。该数字是硬件与数据相关的实测性能，不作为跨设备保证。

## 正式原始数据发现

正式原始数据在逻辑上位于两个目录：`正式实验` 与 `Data`。两块外接存储设备在 Windows 下的盘符可能随连接顺序在 `E:` / `F:` 之间交换，因此 `config.yaml` 同时声明四个候选根：

```text
E:/正式实验
F:/正式实验
E:/Data
F:/Data
```

`run_formal_batch.py` 会忽略不存在的候选根，并在所有有效根中按 `sub-*_/nir/*_nir.avi` 发现被试。若同一被试的视频同时出现在多个有效根，会直接报告 duplicate，不静默选择其中一份。

## 环境安装

新 AMD/DirectML Windows 机器从 [`INSTALL.md`](INSTALL.md) 开始。已配置机器使用 `D:\CondaEnvs\nir-amd`。安装完成后，在本目录执行：

```powershell
python -m pytest tests -q
python run_pipeline.py check-env
python run_pipeline.py discover --formal-only
python run_formal_batch.py --dry-run
```

DirectML 不可用时立即失败，不允许整个 session 静默退回纯 CPU。

## 数据发现与批处理

先检查当前实际挂载的数据：

```powershell
python run_pipeline.py discover --formal-only
python run_formal_batch.py --dry-run
```

确认后运行：

```powershell
python run_formal_batch.py
```

`batch.subjects.include: []` 表示发现所有编号不低于 `formal.min_subject_number` 的完整正式被试；`exclude` 用于显式排除。命令行 `--subjects sub-031,sub-033` 可临时覆盖 include。

## 单被试正式运行

v0.2.0 的单被试正式入口改为 batched runner：

```powershell
python run_formal_batched.py `
  --video "<实际盘符>:\<数据根>\sub-033_\nir\sub-033_nir.avi" `
  --device 0
```

正式多被试仍推荐使用 `run_formal_batch.py`，因为它会统一执行候选根发现、重复被试检查、skip-completed 和输出命名。

## 关键正式参数

当前 `config.yaml` 冻结的主要参数包括：

- YOLO confidence：0.40
- YOLO imgsz：640
- YOLO fixed batch size：8
- tracking：`none`（正式主链每帧均由 YOLO 分析；batch 只是并行组织，不跳帧）
- 标准 analysis geometry：320 × 160
- RITnet 输入：640 × 400
- RITnet fixed batch size：16
- RITnet precision：fp32
- FocusWave release：v3.1.3
- 正式被试编号下限：31
- phases：baseline / instructions / practice / block1 / block2
- baseline：180 s
- 正式 block 数：2

YOLO 尾批和 RITnet 尾批都通过重复最后一个真实样本补齐固定 batch，padding 输出不会写入正式结果。phase 边界不跨 YOLO batch。

## 输出与恢复

默认正式输出根为：

```text
outputs/amd-directml/formal
```

v0.2.0 正式运行目录名明确区分两个 batch：

```text
sub-031_formal_v3.1.3_yolo-b8_ritnet-b16_fp32
```

`completion.json` identity 现在显式包含：

- package version
- FocusWave release / phases
- YOLO batch size
- YOLO model SHA256
- RITnet batch size / precision
- RITnet model SHA256
- video identity

因此旧版 `..._yolo_b16_fp32` 目录不会被误判成 v0.2.0 已完成结果。`skip_completed: true` 只跳过通过完整 identity 与 artifact 校验的 `status=complete`。

正式运行开始时先原子写 `completion.json: running`，全部 CSV/JSON 产物写完并验证后最后写 `complete`。partial/smoke/读帧失败均不会被当成完整结果。

## 眨眼解释边界

RITnet 输出 background、sclera、iris、pupil 四类分割；当前正式后处理仍只使用 pupil 类拟合椭圆。把 sclera、iris、pupil 合成 ocular mask 后，可以派生候选眼裂高度/宽度和被试内 normalized openness，但尚未作为正式 blink/PERCLOS 指标验证。

`ritnet_missing`、`yolo_missing`、瞳孔面积下降或低置信度都不能单独解释为 blink。完整派生逻辑、unknown 门控、时间戳、基线与验证要求见 `docs/020-nir/021-眨眼检测边界与RITnet派生开合度.md`。

## 代码边界

`run_formal_batched.py` 是 AMD v0.2.0 正式单被试执行器；`run_formal_batch.py` 是正式多被试入口。`run_pipeline.py` 保留 diagnostic / discover / check-env 和历史兼容，不删除。

benchmark/export 工具用于版本性能追溯，不是全量正式分析入口。

## 最小验收

```powershell
# runtime/nir-formal
python -m pytest tests -q
python run_pipeline.py check-env
python run_pipeline.py discover --formal-only
python run_formal_batch.py --dry-run
```

在实际 AMD 机器上还应确认 `models/nir-eye-yolo26n-best-b8.onnx` 已存在。缺失时 `run_formal_batch.py` 会直接失败，不会静默退回原 b1 模型。

# 055｜RGB Face 15 Hz 采样频率冻结

**Status: Accepted**

## 决策

正式 RGB Face 推理频率冻结为：

**15 Hz，按真实 RGB `unix_ms` 时间戳驱动采样，而不是按 AVI frame modulo 抽帧。**

该决策只针对已冻结的 Py-Feat 2.1.1 Detectorv2 + DirectML Face backend。Pose 继续 10 Hz；Motion 继续原始全帧顺序计算。

## 为什么不是 10 Hz

本项目 Face 不只分析 AU / expression / pose，还明确保留眼睑与 blink 相关信号：

- Py-Feat / MediaPipe-style `eyeBlinkLeft` / `eyeBlinkRight` blendshape；
- 478-point mesh 派生 EAR；
- 眼睑开度 / 虹膜直径；
- subject-normalized eye openness / closure proxy；
- 后续 blink event / PERCLOS-like 统计。

典型自发眨眼持续时间通常约 100–400 ms，较新的综述/实验也常给出 150–400 ms 范围。10 Hz 约 100 ms/sample，对较短 blink 容易只留下 1–2 个样本，不利于事件识别和粗略 duration；15 Hz 约 66.7 ms/sample，对 150–400 ms blink 通常可提供约 2–6 个样本。

本项目不宣称用 15 Hz 做精细 eyelid kinematics。高帧率 blink 研究指出，即使 30 fps 也不足以准确恢复更精细的眨眼动力学；因此 15 Hz 的目标是 blink event / coarse duration / slower closure statistics，而不是毫秒级闭合轨迹。

## 为什么不是 30 Hz

sub-031 real-300 的 Py-Feat DirectML raw-frame end-to-end 已实测约 17.29 fps。正式 30 Hz 会明显超过当前单流 backend 的实时吞吐，也几乎把离线 Face 计算量翻倍。

15 Hz 在当前 AMD 上接近但低于已验证 end-to-end throughput，能显著改善 10 Hz 的短事件采样，同时保留可接受的 44-subject 全量计算成本。

以约 26 min formal analysis span 粗算：

- 15 Hz ≈ 23,400 sampled frames / subject；
- 44 subjects ≈ 1.03 million Face frames；
- 以 17.29 fps 粗略折算，纯 Face inference 量级约 16–17 h；
- 这是工程规划估算，不替代 representative full-span dry-run 的实际 wall-clock。

## timestamp-driven sampling

正式采样不得写成：

```text
if video_frame_position % 2 == 0
```

而必须写成：

```text
formal unix-ms target grid at 1000/15 ms
→ select/decode source frame according to real timestamps
→ preserve target_unix_ms + actual unix_ms + sample_error_ms
```

原因是部分被试可能存在 capture-index / timestamp gap。即使 nominal source video 为 30 fps，也不能假设整个实验永远严格两帧对应一个 15 Hz 样本。

## 与其他模态的关系

- Face：15 Hz；
- Pose：10 Hz；
- global/body Motion：原始全帧；
- NIR：保持 NIR 正式 pipeline 自己的原始时间分辨率。

跨模态统计统一按 `unix_ms` / trial / probe windows 对齐，不要求各模态强制同一采样率。

## 仍未由本决策冻结的内容

- blink event threshold；
- `closure80_proxy` 是否最终进入主分析；
- PERCLOS-like rolling window 的最终 QC / valid-time denominator；
- primary-face tracking gates；
- Face raw schema 最终文件级实现。

上述内容进入 representative-subject dry-run，见 `056-RGB-Face-Primary与眼睑派生规则.md`。

## 参考依据

- Tereza Soukupová & Jan Čech (2016), Real-Time Eye Blink Detection using Facial Landmarks：EAR 以短时间窗识别 blink。
- Dinges et al. / FHWA-NHTSA PERCLOS 系列：PERCLOS 关注一段时间内慢性眼睑闭合，而非单次 blink 的精细运动学。
- 2026 high-frame-rate blink study：典型 blink 约 100–400 ms，并指出 30 fps 对精细 blink information 仍有限。

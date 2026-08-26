# 057｜RGB Face 第一档工程优化冻结

**Status: Accepted**

## 决策

RGB Face 正式 runner 的工程结构冻结采用以下第一档优化：

- 原始 AVI 直接解码，不再经过 JPEG95 中间文件；
- timestamp-driven 15 Hz 采样保持不变；
- reader/preprocess 使用 bounded prefetch queue 与主推理线程重叠；
- RetinaFace R34 DirectML 保持 batch 8；
- face chips 跨 RetinaFace batches 放入 pending，优先以 multitask DirectML batch 16 运行；
- 最后不足 16 的尾部才 partial flush；
- RetinaFace 阈值、NMS、1.2 square-reflect crop、模型权重、scientific raw 输出字段均不改变。

## 证据

`sub-031` 同一 3600 个 15 Hz dry-run 时点：

### 速度

Legacy JPEG95 reference：

- 212.1708 s；
- 16.9675 fps；
- multitask DML 42.7643 s。

Optimized direct-AVI：

- pipeline wall 123.4865 s；
- 29.1530 fps；
- 含 parquet 写盘 28.6060 fps；
- multitask DML 19.0804 s；
- 226 次 full batch16，仅 1 次 partial batch；
- 450 次 RetinaFace B8 calls。

因此 throughput 提升约 **1.718×**，pipeline wall 降低约 **41.8%**；multitask DML 时间降低约 **55.4%**。

### Parity

reference 输入是 JPEG quality 95 test frames，candidate 直接读取原 AVI，所以不是完全相同的像素输入。结果仍表现为：

- face-count agreement=0.9997222（3600 帧仅 1 帧不同）；
- bbox mean IoU=0.995838，min=0.940041；
- AU20 Pearson≈0.99784；
- emotion7 Pearson≈0.99826；
- V/A Pearson≈0.99843；
- pose6d Pearson≈0.9999969；
- gaze Pearson≈0.99783；
- blendshape Pearson≈0.999734；
- normalized mesh Pearson≈0.9999959；
- original-frame mesh XY Pearson≈0.99999948。

移除 JPEG 有损 round-trip 后产生的小幅数值差异可接受，且正式 runtime 应优先以原 AVI 为科学输入。

## 正式工程基线

```text
original AVI
→ timestamp-driven 15 Hz
→ prefetch decode/preprocess
→ RetinaFace DirectML B8
→ decode/NMS/crop
→ pending face chips
→ multitask DirectML B16
→ full raw scientific outputs
→ parquet
```

## 不属于本决策

本决策不冻结：

- primary-face tracking 阈值；
- blink event threshold；
- `perclos80_proxy` 最终定义；
- RetinaFace detector 降频/跳帧；
- tracking 替代部分 detector calls。

这些仍按 `056-RGB-Face-Primary与眼睑派生规则.md` 与后续 representative dry-run 收口。

详细记录：`docs/工作记录/08-26-12-RGB-Face第一档提速实现.md`。

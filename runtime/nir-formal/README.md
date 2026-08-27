# NIR Formal Runtime

这是 Attention-Analysis `amd-DirectML` 分支的正式 NIR 运行包。基础 formal producer 使用 ONNX Runtime DirectML；在已经完成的 formal 结果之上，RITnet full-class 只保留一条当前正式补全路径：**640×400 全量 hard-label 证据版**。

需要区分两件事：

1. **基础 formal producer**：YOLO 每帧找眼 ROI，再运行 RITnet，生成历史正式 `frames.csv / eyes.csv / summary / manifest / completion / overlays`；
2. **当前 full-class 完整补全**：复用基础 formal 已保存的 `frame_idx + ROI`，不重跑 YOLO，只重新运行冻结 RITnet，并把过去没有保存的完整四分类证据和可重算派生量全部落盘。

full-class 不再存在“fast 版”和“native 版”两套当前生产路径。历史旧 full-class 文件只作为 provenance（来源追踪）保留。

---

## 1. 基础 formal runtime

当前 AMD package version 为 `0.2.0`，基础正式组合为：

```text
YOLO26n: 640×640 / FP32 / DirectML / fixed batch=8 / every frame
RITnet:  640×400 / FP32 / DirectML / fixed batch=16
tracking: none
```

正式模型资产：

```text
models/nir-eye-yolo26n-best.onnx       # b1 reference / diagnostic
models/nir-eye-yolo26n-best-b8.onnx    # AMD formal YOLO
models/ritnet-b16-fp32.onnx            # AMD formal RITnet
models/ritnet-b16-fp32.onnx.data
```

基础 formal producer 的历史 `eyes.csv` 曾使用 320×160 analysis geometry。这一点仍属于历史 formal Schema；**当前 full-class 的 pupil/iris 几何不再从该 320×160结果拼接，而是统一从同一张 400×640 hard label 重新派生。**

基础正式批处理入口：

```text
run_formal_batch.py
```

单被试：

```text
run_formal_batched.py
```

`run_pipeline.py` 继续保留 diagnostic、discover、check-env 和历史兼容功能。

---

## 2. 环境与每次新终端

AMD 工作副本：

```text
D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-amd-DirectML
```

已有环境：

```text
D:\CondaEnvs\nir-amd
```

每次打开 PowerShell / VS Code Terminal：

```powershell
cd "D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-amd-DirectML"

git status --short --branch
git switch amd-DirectML
git pull --ff-only

git status --short --branch
conda activate "D:\CondaEnvs\nir-amd"
cd runtime\nir-formal

python -m pytest tests -q
python run_pipeline.py check-env
```

full-class 正式入口要求 Git working tree 干净；如果存在未提交代码，不要 `reset --hard`，先检查并处理本地修改。

DirectML 不可用时必须失败，不允许正式 session 静默退回 CPU。

新机器安装见 [`INSTALL.md`](INSTALL.md)，操作与 protocol gate 见 [`RUNBOOK_V1.md`](RUNBOOK_V1.md)。

---

## 3. 基础 formal 数据发现

当前 `config.yaml` 的候选数据根：

```text
E:/正式实验
F:/正式实验
E:/Data
F:/Data
```

程序忽略不存在的根；同一 subject 同时出现在多个有效根时必须报 duplicate，不静默选择。

检查：

```powershell
python run_pipeline.py discover --formal-only
python run_formal_batch.py --dry-run
```

已有历史 formal 结果时，不应为了 full-class 补全而重新执行 YOLO producer。

---

## 4. 当前唯一正式 RITnet full-class

详细数学定义、完整字段、官方/项目派生边界、恢复机制和验收标准见 [`RITNET_FULLCLASS_EXTENSION.md`](RITNET_FULLCLASS_EXTENSION.md)。当前版本标识：

```text
ritnet-fullclass-v2-native640
schema_version = 2
```

这是**当前唯一生产 Schema**，版本号不是“双轨运行”的含义。

full-class 数据流：

```text
已完成 formal eyes.csv 的 frame_idx + YOLO ROI
→ 原视频重新裁相同 ROI
→ RITnet 640×400 / FP32 / fixed-b16 / DirectML
→ 每个 eye row 保存 uint8 [400,640] 四分类 hard label
→ 同一 label 派生 pupil / iris_outer / ocular
→ pupil 与 iris ellipse、PIR、component、edge、OAR、原子 gate/diagnostic
→ class-3 pupil probability 条件摘要随 chunk checkpoint
→ sparse QC
→ manifest / completion / SHA256
```

不会重新运行 YOLO，也不会覆盖基础 formal `eyes.csv`。

### 4.1 原始证据与派生层级

正式 full-class 分为：

```text
原始可重算证据：*_labels/chunks/*.npz
索引/存储证明：label_index.csv + chunk_manifest.csv + store_manifest.json
派生数值：*_ritnet_fullclass_v2-native640.csv
人工检查：*_qc/ + *_qc_index.csv
运行说明：*_summary.json + *_manifest.json
最终证明：*_completion.json
```

完整 hard label 是后续重算几何/QC 的事实源，CSV 只是派生层。

### 4.2 当前正式入口

用户只调用：

```text
run_ritnet_fullclass_extension.py
run_ritnet_fullclass_batch.py
```

带 `native` 的 Python 文件属于内部实现，不是第二套用户生产入口。旧 fast 参数已经退出正式路径。

正式入口自动执行两项强 provenance：

- source video 计算内容 SHA256；
- 禁止 model mismatch override。

同时要求 Git working tree 干净，使 manifest 中记录的 commit 能完整代表执行代码。

---

## 5. 先验收 sub-031，再扩 AMD cohort

不要直接根据旧 full-class 的“已跑到 sub-031/从 sub-032 继续”状态启动新版本。完整证据版新增了全量 labels、概率 checkpoint、strict resume 和最终哈希链，必须重新以当前 completion 为完成依据。

### 5.1 dry-run

```powershell
python run_ritnet_fullclass_batch.py `
  --output "D:\_AttentionData\Beijing-NIR\amd-directml" `
  --subjects "sub-031" `
  --device 0 `
  --chunk-rows 128 `
  --dry-run
```

### 5.2 只跑 sub-031

```powershell
python run_ritnet_fullclass_batch.py `
  --output "D:\_AttentionData\Beijing-NIR\amd-directml" `
  --subjects "sub-031" `
  --device 0 `
  --chunk-rows 128
```

当前 `chunk_rows=128` 是存储打包默认值，只影响 chunk 大小，不改变科学结果；在正式 cohort 开始前根据 `sub-031` 的磁盘占用、压缩率、吞吐与恢复实测冻结。

### 5.3 sub-031 最小验收

至少确认：

```text
DirectML provider 正确
processed_rows == expected_rows == stored_label_rows
label shape == 400×640, dtype == uint8, value domain ⊆ {0,1,2,3}
label_index ordinal 连续，frame/eye 唯一
CSV key sequence == label_index key sequence
所有 chunk SHA256 正确
store_manifest.status == complete
store_manifest 内嵌 label_index/chunk_manifest SHA256 正确
CSV / label_index / chunk_manifest / store_manifest / summary / manifest / qc_index
    的 SHA256 全部通过 completion verifier
finalized store 重开后 store_manifest 内容和 SHA256 不变化
抽样 QC 语义正常
中断后 committed chunk 能恢复而不重复推理
```

通过以后再根据本机 discovery 生成 AMD 当前 cohort 的 subject list。

### 5.4 批量运行

例如先在 PowerShell 得到希望处理的 subject list，然后：

```powershell
python run_ritnet_fullclass_batch.py `
  --output "D:\_AttentionData\Beijing-NIR\amd-directml" `
  --subjects "$subjectArg" `
  --device 0 `
  --chunk-rows 128
```

不要再使用：

```text
--postprocess-workers
--validate-pupil
--force
--allow-model-mismatch
```

这些属于旧实现或不符合当前严格 provenance 的操作。

---

## 6. full-class resume 与完成判定

label store 使用事务式 chunk：

```text
临时 NPZ
→ flush + fsync
→ 结构校验
→ SHA256
→ os.replace
→ 更新 index / chunk manifest / store manifest
```

已经原子提交的 NPZ chunk 是恢复事实源。如果程序在 chunk rename 后、metadata 写入前中断，恢复器会从 committed chunk 重建 metadata，不删除该 chunk、不重跑对应 RITnet。

已经 complete 的 store 内容未变化时重新打开必须保持 byte-stable（字节稳定）。completion verifier 会在任何 store recovery 之后重新核验最终 artifact SHA256，因此 metadata 真正发生恢复时，旧 completion 会自然失效并要求重新 finalize；没有恢复时则不得产生 hash drift。

`status=complete` 不能只看文件存在，必须同时通过行数、shape/dtype/value-domain、chunk hash、index↔CSV key、store manifest 和全部顶层 artifact hash。

---

## 7. 官方 RITnet 与项目新增记录的边界

官方 RITnet 提供/定义：

- `DenseNet2D` 网络；
- 冻结 `best_model.pkl` 权重；
- background / sclera / iris / pupil 四分类任务语义；
- 网络 logits；
- gamma=0.8、CLAHE 1.5/8×8、Normalize([0.5],[0.5])；
- 官方 test 流程对 logits 做 argmax 得到 hard prediction。

Attention-Analysis 自己增加：

- YOLO ROI 复用与 ROI→640×400 resize；
- fixed-b16 FP32 ONNX / DirectML 运行适配；
- `labels_u8` ONNX 输出（logits 后 ArgMax）；
- `pupil_probability`（Softmax/Gather class 3）；
- pupil/iris ellipse、PIR、component、edge、OAR；
- probability summaries；
- `gate_*` / `diagnostic_*`；
- label store、QC、manifest、completion、SHA256。

所以 `native_*` 只表示“640×400 hard-label 坐标系”，不表示“RITnet 官方变量”。

---

## 8. 几何与 blink/PERCLOS 边界

ROI 会被调整到固定 640×400。如果 `scale_x != scale_y`，模型坐标存在非等比例形变。因此 full-class 保存 `source_roi_*`、`roi_to_ritnet_scale_*` 和 `geometry_coordinate_system`，不把模型坐标 PIR 宣称为物理尺度不变指标。完整 hard label 已保存，后续可映回 source ROI 坐标重算。

OAR（ocular aperture ratio，眼球可见开口比例）是项目派生几何量，不是 EAR（Eye Aspect Ratio，基于眼睑关键点的眼睛纵横比），也不是已经验证的 blink/closed/PERCLOS 标签。`ritnet_missing`、`yolo_missing`、瞳孔面积下降或低 confidence 也不能单独解释为 blink。

当前 full-class 只保存原子 QC 事实，不提前冻结未经验证的新 primary validity cutoff 或 blink cutoff。

---

## 9. 基础 formal producer 的既有参数

`config.yaml` 的基础 formal 主要参数包括：

- YOLO confidence：0.40；
- YOLO imgsz：640；
- YOLO fixed batch size：8；
- tracking：none；
- 历史 formal analysis geometry：320×160；
- RITnet 输入：640×400；
- RITnet fixed batch size：16；
- RITnet precision：fp32；
- FocusWave release：v3.1.3；
- formal subject 编号下限：31；
- phases：baseline / instructions / practice / block1 / block2；
- baseline：180 s；
- expected formal blocks：2。

YOLO/RITnet 尾批只在 fixed ONNX batch 需要时复制最后一个真实样本补齐，padding 输出被丢弃。

---

## 10. 当前不能在仓库里伪称完成的验收

以下必须由 AMD 本机实际运行确认：

- DirectML 端到端 full-class；
- 当前 `.onnx` + `.onnx.data` 实机加载；
- `sub-031` 完整 labels / CSV / QC / completion；
- `chunk_rows=128` 的压缩率、磁盘占用与吞吐；
- Windows 文件系统下真实中断恢复；
- 当前 AMD cohort 全量完成。

因此仓库现在可以冻结**唯一完整方法和代码路径**，但在 AMD 本机把这些验收项跑完之前，不能把运行状态写成“44 人已经完成”。

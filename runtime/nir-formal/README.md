# NIR Formal Runtime（AMD / DirectML）

这是 `Attention-Analysis` 的 AMD 正式 NIR 运行包。基础 formal producer 已经完成历史 YOLO + RITnet 提取；当前要补全的是 **最终 RITnet full-class 管线**。这条管线不会重新运行 YOLO，而是严格复用历史正式 `eyes.csv` 中已经保存的 YOLO 眼睛框，从原始 NIR AVI 重新构造固定 1.6 宽高比 ROI，再运行同一套冻结 RITnet 权重。

当前最终目标是：**每被试新增 full-class 输出 < 1 GiB，并保留后续分析和质量检查所需的完整标量/小向量原子信息，而不是全量保存每只眼 400×640 的标签图或四分类概率图。**

Issue #21 是当前唯一收口清单；AMD `sub-031` 真机验收通过前，不启动全 cohort。

---

## 1. 每次打开新终端

AMD 工作副本：

```text
D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-amd-DirectML
```

已有 Conda 环境：

```text
D:\CondaEnvs\nir-amd
```

PowerShell / VS Code Terminal：

```powershell
cd "D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-amd-DirectML"

git switch amd-DirectML
git pull --ff-only
git status --short --branch

conda activate "D:\CondaEnvs\nir-amd"
cd runtime\nir-formal
```

正式入口要求 Git working tree 干净。不要为了通过检查而使用 `git reset --hard`；如果有本地修改，先确认它们是什么。

---

## 2. 当前最终 full-class 数据流

```text
历史 formal completion / frames.csv / eyes.csv
        ↓ 严格完整性验证
历史 YOLO bbox + 原始 NIR AVI
        ↓ 不重跑 YOLO
扩大上下文并构造固定 8:5（1.6）ROI
        ↓ 超出原视频的位置只作为 RITnet 输入人工补边
统一缩放到 640×400
        ↓
RITnet FP32 / fixed batch=16 / DirectML
        ↓
硬四分类标签 + 临时四分类逐像素概率 + max probability + top1-top2 margin + entropy
        ↓
只在“真实原视频像素区域”计算正式指标
        ↓
逐眼标量/小向量 + frame coverage + temporal QC
        ↓
固定时间点/异常帧综合 QC 图
        ↓
summary.json + manifest.json + completion.json + SHA256 + <1 GiB 硬门槛
```

人工补边可以影响神经网络看到的上下文，但**人工补出来的像素本身不进入正式面积、比例、几何、四分类软比例和不确定性统计**。如果模型在人工补边里预测出瞳孔/虹膜/眼球，这些像素数以及结构是否碰到真实画面边界会作为质量检查事实单独保存。

---

## 3. 最终 RITnet ONNX 必须重新导出一次

旧的 final uncertainty ONNX 若输出 `soft_class_fraction [16,4]`，已经不能用于当前管线，因为它在模型内部对整张 640×400 ROI 提前求平均，无法排除人工补边。

当前最终模型接口必须是：

```text
labels              uint8   [16,400,640]
class_probability   float32 [16,4,400,640]
max_probability     float32 [16,400,640]
top1_top2_margin    float32 [16,400,640]
entropy              float32 [16,400,640]
```

这些逐像素概率/不确定性图只在当前 batch 的内存中存在，统计完成后释放，不逐眼写入硬盘。

在 AMD 机器拉到最新代码后执行：

```powershell
python export_ritnet_batch_variants.py `
  --final-uncertainty `
  --batches 16 `
  --force
```

期望生成：

```text
models/ritnet-b16-fp32-uncertainty.onnx
models/ritnet-b16-fp32-uncertainty.onnx.data
```

随后立即验证 DirectML 和输出合同：

```powershell
python validate_ritnet_fullclass_final_model.py --device 0
```

只有输出 `"status": "pass"`，并且第一 provider 为 `DmlExecutionProvider`，才能进入 `sub-031`。

---

## 4. 先跑代码测试

```powershell
python -m pytest tests -q
```

这里验证的是代码合同，包括：

- 固定 1.6 ROI 与人工补边；
- 真实原视频像素分析域；
- hard-class / pupil / iris / ocular 指标；
- 四分类软比例只在真实像素上计算；
- whole / ocular / boundary 不确定性统计；
- 同眼连续帧 temporal delta 与 rolling median/MAD 异常事实；
- 完整帧时间线 coverage；
- QC 固定/异常选样与空间预算；
- `qc_index.csv` 与图片 SHA256；
- `summary / manifest / completion` 完整性合同；
- 最终用户入口不再路由到旧 label-store 实现。

pytest 通过仍然不等于正式验收；AMD DirectML 和真实 `sub-031` 仍必须实跑。

---

## 5. `sub-031` 唯一验收入口

历史 formal 输出根沿用当前 AMD 数据目录。例如：

```text
D:\_AttentionData\Beijing-NIR\amd-directml
```

先只检查源选择，不运行 RITnet：

```powershell
python run_ritnet_fullclass_batch.py `
  --output "D:\_AttentionData\Beijing-NIR\amd-directml" `
  --subjects "sub-031" `
  --device 0 `
  --dry-run
```

确认选中的历史 formal run 正确以后，运行：

```powershell
python run_ritnet_fullclass_batch.py `
  --output "D:\_AttentionData\Beijing-NIR\amd-directml" `
  --subjects "sub-031" `
  --device 0
```

单被试也可以直接运行：

```powershell
python run_ritnet_fullclass_extension.py `
  --run-dir "<sub-031 对应的历史 formal run 目录>" `
  --config config.yaml `
  --device 0
```

**不要再使用：**

```text
--chunk-rows
--compression
--postprocess-workers
--validate-pupil
--force
--allow-model-mismatch
```

它们属于已经退出当前正式 full-class 路径的旧实现。

---

## 6. 最终输出结构

每个被试的新输出独立放置，不修改历史 formal 文件：

```text
ritnet-fullclass-final/
└─ sub-031/
   ├─ data/
   │  ├─ eye_metrics.csv.gz
   │  └─ frame_coverage.csv.gz
   ├─ qc/
   │  ├─ images/
   │  │  └─ *.png
   │  └─ qc_index.csv
   ├─ summary.json
   ├─ manifest.json
   └─ completion.json
```

中断恢复使用的 SQLite checkpoint 不属于最终科研数据，位于相邻工作目录：

```text
.ritnet-fullclass-work/sub-031.sqlite
```

它只用于恢复已经完成的逐眼数值计算；最终完成与否只由新的严格 `completion.json` 校验决定。

---

## 7. `eye_metrics.csv.gz` 保存什么

最终逐眼主表不是旧 `eyes.csv` 的拼接，而是固定版本 Schema。主要包括：

- subject / frame / eye / phase / 时间；
- 历史 YOLO bbox、置信度和 source status/reason；
- 新固定 1.6 ROI 的请求范围、真实来源范围、四侧人工补边量和 resize 信息；
- 背景、巩膜、虹膜、瞳孔、iris_outer、ocular 的 hard 像素数和比例；
- 瞳孔与 iris_outer 轮廓、椭圆、轴长、面积、中心；
- pupil-to-iris diameter / ellipse-area / contour-area ratio；
- ocular aperture；
- component fragmentation 与边界接触等原子质量事实；
- 四类 soft fraction；
- max probability、top1-top2 margin、entropy 在 whole / ocular / boundary 三个真实像素分析域中的分布摘要；
- 相邻连续同眼帧 delta、瞳孔中心位移和 temporal jump/anomaly 事实。

质量指标只用于后续筛选、敏感性分析和人工复核，**不会在提取阶段自动删除科研数据。**

---

## 8. `frame_coverage.csv.gz` 为什么必须单独保存

逐眼表只可能存在于历史 YOLO 找到眼睛的帧。为了避免把“YOLO 没找到眼睛”误当作“视频里没有这一帧”，最终另保存完整帧时间线。

每个历史 formal frame 都必须在 `frame_coverage.csv.gz` 中出现一次，并区分：

```text
historical video read failed
yolo no eye
single eye detected / success
both eyes success
final video decode failed
RITnet no success
source eye present but final result missing
```

固定时间 QC 锚点也从这张完整时间线产生，因此即使某帧 YOLO 两眼都没检测到，仍然可以被抽中查看原视频画面。

---

## 9. QC 与磁盘预算

当前配置：

```text
qc_image_max_count = 200
qc_artifact_budget_bytes = 268435456   # 256 MiB
final_output_limit_bytes = 1073741824  # 1 GiB
```

QC 一帧只保存一张综合图：

```text
原视频整帧（历史 YOLO 框 + 新 ROI 真实范围）
+ 左眼 RITnet 四分类叠加 / ellipse / 关键指标
+ 右眼 RITnet 四分类叠加 / ellipse / 关键指标
```

固定时间点优先保存；异常帧按 phase/reason 均匀抽样。同一帧多个异常合并为一张图。`qc_index.csv` 记录每张图的选择原因、对应眼睛、coverage status、SHA256 与大小。

如果异常图达到空间预算，后续异常图可以停止扩张；**固定时间锚点不能为了节省空间被静默丢弃**。如果固定锚点本身已经超预算，运行直接失败并要求重新审查配置。

---

## 10. 最终完成判定

存在 `completion.json` 不等于自动完成。再次运行时程序会重新验证：

1. `summary.json` 与 `manifest.json` SHA256；
2. `eye_metrics.csv.gz` 固定表头、Schema 版本、subject 和行数；
3. `frame_coverage.csv.gz` 固定表头、Schema 版本、subject 和行数；
4. `qc_index.csv` 固定表头；
5. 每张 QC 图片存在、大小一致、SHA256 一致；
6. manifest 中三个核心 artifact SHA256/大小一致；
7. manifest 的 work identity 与当前 Git commit / config / RITnet 模型 / source identity 完全一致；
8. 整个被试最终目录实际大小不超过 1 GiB。

只有全部通过才允许严格 skip。已有 completion 损坏、属于旧 Git/模型/配置，或者只有部分 QC/metadata 而没有有效 completion 时，程序会停止，不会自动删除或覆盖这些证据。

---

## 11. `sub-031` 最终人工验收

在扩展到 cohort 前至少确认：

- pytest 全通过；
- `validate_ritnet_fullclass_final_model.py` 在 AMD 上通过，主 provider 为 DirectML；
- `sub-031` 运行到有效 completion；
- `eye_metrics` 行数与历史 source eye rows 一致；
- `frame_coverage` 行数与历史 `frames.csv` 一致；
- 最终目录实际大小 `< 1 GiB`；
- 抽看清晰、一般、模糊、闭眼、半闭眼、反光、眼镜、画面边缘、YOLO 漏检、单眼、分割碎裂、temporal jump 等 QC；
- 人工补边中的预测没有进入正式 hard/soft/uncertainty 指标；
- 关闭后重新运行同一命令能够通过严格完整性检查并跳过，而不是再次推理。

这些证据回填 Issue #21 后，才进入剩余 AMD subjects 的正式 full-class 批处理。

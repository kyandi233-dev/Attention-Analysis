# 08-23-06｜artifacts与非正式模型资产删除工作记录

> 文档性质：仓库整理 / 已授权删除执行记录  
> 日期：08-23  
> 当天序号：06  
> 当前维护主线：`main`  
> 状态：已执行

## 一、本轮授权

用户明确批准：

1. `artifacts/` 全部删除，认为其中内容对当前项目不重要；
2. 删除所有不参与当前正式分析的 `models` 资产；
3. `scripts/` 暂不直接删除，先给出可删候选供用户审查。

本授权**不包括**删除任何 `docs/工作记录/` 文件。工作记录仍按科研 provenance 永久保留原则处理。

## 二、删除的 artifacts

从当前 `main` 删除整个：

```text
artifacts/
```

其中包括此前的：

```text
gate1-24eyes/
preview/
roi-compare/
roi-selection-smoke-sub011/
roi-selection-smoke3s-sub011/
```

删除原因：这些内容均属于已经退出当前正式路线的阶段性审核、比较、预览或冒烟输出；用户确认它们对当前项目不重要，不需要继续作为当前仓库一级资产保存。

这不影响当前正式 NIR 全量分析结果，因为正式全量结果本就保存在仓库外独立输出目录，正式运行也不从 `artifacts/` 读取任何输入。

## 三、删除的非正式 models

删除：

```text
models/external/
models/historical/
```

覆盖的历史资产包括：

```text
RITnet 完整上游仓库副本
DeepVOG
DeepVOG-3D
pye3d-detector
YOLO face-parts detector
MediaPipe face_landmarker.task
YOLO-face yolov8n-face.onnx
YuNet yunet_2023mar.onnx
```

### 删除原因

这些资产均不属于当前正式 NIR runtime 的直接依赖。

当前正式 YOLO26n 模型仍完整保留：

```text
datasets/
→ training/nir-eye-yolo/
→ training/.../weights/best.pt
→ runtime/nir-formal/models/nir-eye-yolo26n-best.pt
```

当前正式 RITnet 仍完整保留：

```text
runtime/nir-formal/ritnet/
runtime/nir-formal/ritnet_runtime.py
runtime/nir-formal/models/ritnet-best_model.pkl
runtime/nir-formal/ritnet/UPSTREAM.md
```

此前已经核对：正式 runtime 中冻结的 `densenet.py`、RITnet 权重和 license 可追溯到上游 RITnet；因此删除 `models/external/ritnet/` 不会破坏正式运行或来源说明。

DeepVOG、DeepVOG-3D、pye3d、MediaPipe、YuNet、YOLO-face、face-parts 等均已退出当前正式路线。继续在 `main` 保存完整源码和模型会增加仓库体积，并容易使新设备配置或后续维护者误判当前依赖。

## 四、历史复现边界

删除上述旧模型后，历史 `configs/formal.yaml` 和部分旧 `scripts/` 可能仍保留对这些路径的引用。这是历史状态的一部分，不代表当前 `main` 仍保证这些旧候选脚本可以直接运行。

若需要完整复现旧 tracking / 多候选算法阶段，应使用 Git 历史或：

```text
history/tracking-era-2026-08
```

而不是重新把旧第三方资产塞回当前正式主线。

## 五、本轮同步更新的当前说明

同步更新：

```text
AGENTS.md
models/README.md
docs/010-overview/013-仓库资产与复现关系.md
```

目的：避免删除后当前说明仍指向不存在的 `artifacts/`、`models/external/` 或 `models/historical/`。

历史工作记录不追溯改写；它们继续记录这些目录和资产在当时真实存在、被测试或被比较过的事实。

## 六、下一步：scripts 删除审查

下一步只对 `scripts/` 做删除候选审计，不自动删除。

判断标准：

1. 是否仍对应当前数据、训练、正式行为分析或正式评估；
2. 是否只是已删除模型的旧候选入口；
3. 是否存在仍被当前代码调用的共享模块；
4. 是否仅为一次性生成旧报告/旧 artifacts；
5. 历史工作记录与冻结历史版本是否已经足以保留其研究 provenance。

审计后将向用户提供分组清单，由用户确认后再删。

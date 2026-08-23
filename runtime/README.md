# runtime｜冻结运行环境与可移植分析包

## 当前定位（2026-08-23）

`runtime/` 用于保存项目在特定阶段冻结下来的可移植运行资产、环境依赖和模型副本。这里的文件与 `src/`、`scripts/` 的职责不同：

- `src/` / `scripts/` 保存持续演化的源码与研究工具；
- `runtime/` 保存为了跨电脑运行、环境复现和模型身份固定而打包的**冻结 runtime snapshot**。

项目当前已经完成 YOLO26n 训练和正式 NIR 全量分析；但 runtime 内部各包的 README/config 仍应按其**创建当时的真实状态**理解，不能因为项目后来完成全量就反向改写冻结包历史。

## 当前内容

```text
runtime/
├── README.md
├── PyPupilEXT-0.0.1-cp310-cp310-win_amd64.whl
├── requirements-main.txt
├── requirements-pupil.txt
├── nir-yolo-tracking-ritnet-v1/
├── nir-yolo-tracking-ritnet-v1.zip
└── nir-yolo-tracking-ritnet-v1.zip.sha256
```

## `nir-yolo-tracking-ritnet-v1`

该目录是 2026-08-22 建立的跨机 GPU portable package，内部实现：

```text
NIR AVI
→ YOLO26n 双眼检测
→ CSRT / KCF tracking 或逐帧 YOLO
→ 周期性 / 失败回退重检测
→ 眼 ROI 扩展与 320×160 裁剪
→ RITnet 分割
→ frames.csv / eyes.csv / overlay / summary / run_manifest
```

入口：

```text
runtime/nir-yolo-tracking-ritnet-v1/run_pipeline.py
```

该 runner 支持 `--full-video`，因此代码能力并不局限于 60 秒短片；默认 60 秒和 README 中的“短视频准入测试”反映的是**创建时的安全运行策略**。

### 模型身份已核验

portable package 并没有使用另一套独立模型。

YOLO 权重：

```text
yolotrain/runs/yolo26n_eye_100epoch/weights/best.pt
runtime/nir-yolo-tracking-ritnet-v1/models/nir-eye-yolo26n-best.pt
```

两者当前 Git blob SHA 完全相同：

```text
e9d818bdcdca41dc318b04512a4176e3db078e57
```

文件大小均为：

```text
5,403,247 bytes
```

RITnet 权重：

```text
models/RITnet-master/best_model.pkl
runtime/nir-yolo-tracking-ritnet-v1/models/ritnet-best_model.pkl
```

两者当前 Git blob SHA 完全相同：

```text
f0864e6651f578525a9101c7ca787e23d2d201d7
```

文件大小均为：

```text
1,018,397 bytes
```

因此 runtime 中两个模型文件应理解为**为跨机复现而冻结的同内容副本**，不是需要因为“重复”而删除的冗余资产。

## 历史状态与当前状态不要混淆

包内 `README.md` 和 `config.yaml` 在 2026-08-22 创建时明确把该包描述为：

```text
GPU short-video admission trial
not frozen production
```

同时 2026-08-22 工作记录也明确：当日只完成静态 test 评价和 portable package 构建，正式视频全量当时尚未启动。

这些描述是正确的历史记录，应继续保留。

项目级现实状态则已经推进到：

```text
正式 NIR 全量分析已完成
```

二者并不矛盾：前者描述 **8 月 22 日创建 runtime 时的状态**，后者描述 **当前项目状态**。

## 当前仍未闭合的 provenance 问题

当前仓库已经确认：

1. `run_pipeline.py` 是现有分支中明确实现 YOLO + tracking + RITnet 完整链路的 portable runner；
2. 它使用的 YOLO 和 RITnet 权重与仓库正式资产逐字节一致；
3. 它支持 `--full-video`；
4. 项目正式 NIR 全量已经实际完成。

但是，截至当前 GitHub 分支核验，尚未找到正式全量运行时最终保存的 `run_manifest.json`、完整命令记录或其他能够证明“最终全量就是使用该 package 的哪组参数运行”的 committed evidence。

因此当前最严谨的表述是：

> `runtime/nir-yolo-tracking-ritnet-v1/` 是目前仓库中已经核验出的 YOLO26n + tracking + RITnet 完整 portable implementation，也是正式路线的直接代码血缘候选；但在最终 full-run manifest 被找到之前，不应仅凭目录名把其 2026-08-22 默认参数自动宣布为正式全量最终冻结参数。

后续如找到正式运行输出中的 `run_manifest.json`，应优先用其中的 command、config、模型 SHA256 和环境信息闭合这条 provenance 链，而不是重新推测当时的运行方式。

## 冻结包维护原则

`nir-yolo-tracking-ritnet-v1/` 内已有 `SHA256SUMS.txt`，顶层还有同名 ZIP 和 ZIP SHA256。因此：

- 不为了更新项目状态而直接改写包内 README/config；
- 不随意替换包内模型；
- 不把包内模型副本当成普通重复文件删除；
- 如果未来确实需要修改实现，应建立新的 runtime 版本，而不是静默覆盖 v1；
- 任何涉及删除旧 runtime 资产的操作仍必须先取得用户明确同意。

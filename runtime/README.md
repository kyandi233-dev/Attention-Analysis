# runtime｜冻结运行环境与可移植分析包

## 当前定位

`runtime/` 保存为跨电脑运行、环境复现和模型身份固定而冻结的运行资产。持续演化的源码仍位于 `src/` 与 `scripts/`；runtime 不作为第二套开发源码目录。

当前结构：

```text
runtime/
├── README.md
├── NIR-formal/
├── PyPupilEXT-0.0.1-cp310-cp310-win_amd64.whl
├── requirements-main.txt
└── requirements-pupil.txt
```

原来的 `nir-yolo-tracking-ritnet-v1.zip` 与 `.zip.sha256` 已按用户批准删除；保留解包后的正式 runtime 目录即可。

## NIR-formal

`runtime/NIR-formal/` 是 2026-08-22 建立并冻结的 YOLO26n + tracking + RITnet portable implementation，内部实现：

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
runtime/NIR-formal/run_pipeline.py
```

该 runner 支持 `--full-video`。包内 YOLO 权重与 `yolotrain/runs/yolo26n_eye_100epoch/weights/best.pt` 为同一 Git blob；包内 RITnet 权重与 `models/RITnet-master/best_model.pkl` 也为同一 Git blob。因此两个模型副本属于 frozen runtime 的复现资产，不是普通重复文件。

## 历史状态与当前状态

`NIR-formal/` 内部 README/config 是 2026-08-22 当时冻结的 snapshot，其中仍可能描述 GPU short-video admission trial、not frozen production 等当时状态。这些包内历史内容不反向改写。

项目当前事实是：正式 NIR 全量分析已经完成。当前仍未闭合的是 final full-run provenance：仓库中尚未找到最终运行的 `run_manifest.json` 或等价命令记录，因此不能仅凭 runtime 默认配置把 tracker、重检测间隔、ROI expansion 等宣布为全量最终冻结参数。

## 维护原则

- 不静默替换 `NIR-formal/` 内的模型和实现；
- 不把 runtime 中模型副本当成普通重复文件删除；
- 新实现若与 frozen snapshot 有实质变化，应明确建立新版本或新的正式 runtime 资产；
- 找到正式 full-run manifest 后，优先补轻量 provenance，而不是重新猜测最终运行方式。

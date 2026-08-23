# 方法说明

方法层回答“具体怎么做、为什么这样做”，不替代 `010-nir/` 的研究过程记录，也不改写 `docs/工作记录/` 的历史正文。

| 编号 | 文件 | 内容 |
|---|---|---|
| 061 | [061-YOLO眼框检测方法.md](061-YOLO眼框检测方法.md) | NIR 专用 YOLO26n 眼框检测、训练划分、val/test、bbox→ROI 与失败状态 |
| 062 | [062-Tracking策略.md](062-Tracking策略.md) | CSRT/KCF、周期重检测、tracker 失败回退、双眼身份与计算收益 |
| 063 | [063-RITnet瞳孔与虹膜分割方法.md](063-RITnet瞳孔与虹膜分割方法.md) | ROI 预处理、RITnet 分割、pupil mask、椭圆拟合、confidence 与失败语义 |

后续正文从 `064` 顺延，不跳号。08-22 portable runtime 的默认参数与正式 full-run 最终参数必须分开；未找到最终 `run_manifest.json` 前，不把默认值改称最终冻结值。

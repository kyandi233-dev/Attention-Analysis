# attention-pipeline-v2

## 2026-08-21 当前项目状态

> 2026-08-21 17:30（Asia/Shanghai）｜北京已完成一百多名正式被试的数据采集；预计 2026-08-23 晚回珠海后启动正式分析。

- NIR 眼框数据集 v1 与 YOLO26n 100 epochs 训练已完成；冻结 test 为 7 名被试、85 张图、169 个眼框，mAP50=0.9913、mAP50-95=0.6589。`best.pt` 已进入静态 test 入围，但尚未通过正式视频生产准入。
- 正式视频处理路线暂定为：`OpenCV 读取视频 → YOLO 周期性重新检测 → tracking 更新中间帧 ROI → RITnet 处理眼睛 ROI → 时序质量控制与指标提取`。
- RX 6750 GRE 电脑当前不使用 PyTorch GPU 加速；训练完成后，计划视条件将权重和推理环境转移至 NVIDIA 电脑，处理一百多个约 25 分钟视频。
- tracking 算法、重新检测间隔和 RITnet 参数仍需在珠海用短视频冒烟测试确定；在此之前不运行全量正式提取。
- 已生成 `runtime/nir-yolo-tracking-ritnet-v1.zip`，用于 GPU 电脑的 YOLO26n + CSRT/KCF + RITnet 真实短视频联调；支持 `F:/正式实验` 与 `E:/Data` 两个根目录。

> 08-16（Asia/Shanghai）｜NIR高严重度修复、528眼轴角复核和历史连续序列复核已完成；正式三种人脸ROI在双眼特写上均未通过身份硬门，生产冻结按计划停止。

跨仓库项目状态与研究决策以[厚璨杯统一项目记忆](../../000-项目定位和进度/000-厚璨杯项目记忆.md)为准；本README只维护v2自身入口和运行状态。

唯一项目入口：[000-项目总览与架构.md](000-项目总览与架构.md)。脚本跳转：[scripts/00-目录与映射.md](scripts/00-目录与映射.md)。完整决策过程：[docs/工作记录/00-目录与映射.md](docs/工作记录/00-目录与映射.md)。

## 当前NIR结论

- 椭圆轴角已修复并测试；历史阶段4/4b重新运行后，六算法和18项调优仍全部未达准入门槛，最佳约17.2%。
- 新PuReST适配层历史复跑：可见覆盖0.5230、恢复192 ms，连续性仍优于PuRe；这不是正式准确率。
- 正式sub-011 Block1是双眼/鼻梁特写，不是完整人脸。60时点中MediaPipe、YuNet、当前YOLO-face正确双眼ROI均为0。
- 因ROI硬门失败，尚未冻结正式`minPx/maxPx`、门控、模型、runtime或`scripts/run_nir_pipeline.py`。
- RGB、跨模态与专注评分接口继续关闭。

详细记录：

- [阶段4与5修复复核](docs/工作记录/08-16-04-NIR阶段4与5修复复核工作记录.md)
- [正式ROI入围检查](docs/工作记录/08-16-05-NIR正式ROI入围检查工作记录.md)

```powershell
$env:PYTHONPATH='src'
& 'D:/Code/python/python.exe' -m pytest -q
& 'D:/Code/python/python.exe' scripts/gate1_contract_check.py
```

主分析环境：`D:/Code/python/python.exe`；PyPupilEXT环境：`D:/aaawork/07-竞赛/厚璨杯/venv-pupil/Scripts/python.exe`。主环境当前有`requests`依赖版本告警，已记录但未在本轮破坏性清理。

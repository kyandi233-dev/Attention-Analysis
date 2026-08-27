# 2026-08-28 NVIDIA NIR full-class 重跑交接记录

## 范围

- 分支：`nvidia-cuda`，本轮代码提交：`f2298a2`、`3c9da4d`、`a00dd08`。
- 不重新运行 YOLO；只复用已完成的历史 formal source，运行当前 RITnet four-class post-hoc labels-only 流程，不加 `--validate-pupil`。
- 正式输出根目录：`D:\Project\厚粲杯\11_数据\01_Attention-Analysis_nvidia-cuda_formal_NIR`。

## 修复

1. `--subjects` 单被试发现阶段先按被试名限制候选目录，再做 completion validation，不再扫描全部 `sub-*_formal_*`。
2. 缺失或空白的 legacy `completion.json.yolo_batch_size` 不再阻断；输出 provenance 明确标记 `source_yolo_batch_size_recorded=false`。真实历史 `eyes.csv` 同样缺少该列，按未记录处理；显式错误或不匹配值仍拒绝。
3. legacy source 缺失 `yolo_model_sha256` 按未保存处理，保留 `source_yolo_model_sha256_recorded=false`；不伪造哈希。
4. 远端同步中误删的 formal run guard 已恢复；四项 guard 测试失败属于远端基线缺陷，不是测试滞后。

## 模型来源与完整性

远端 `nvidia-cuda` 未包含 uncertainty ONNX，且不是 Git LFS 文件。本轮按仓库现有冻结模型 `runtime/nir-formal/models/ritnet-best_model.pkl` 和现有 `export_ritnet_batch_variants.py --final-uncertainty` 确定性导出；未改变模型语义或科研参数。

- `ritnet-b16-fp32-uncertainty.onnx` SHA-256：`599f79b89cae455ce3bf412dd28f00438140d2f739d585c2e2223681b03eae6d`
- `.onnx.data` SHA-256：`ff6389a1a8e3ef9ae8ba4349fde7fd7dc6b549691d0015eeed75c5ca7b28fdd6`
- CUDA provider、固定输入 `640x400`、五输出结构和 class probability mass 校验通过。

## 验证

- `runtime/nir-formal/tests`：34 passed。
- `run_pipeline.py check-env`：CUDA/RTX 5070/模型检查通过。
- `sub-056` 单被试 labels-only 验收：官方 `validate_final_completion` 通过；73,207 eye rows、38,903 frame rows、200 QC 图、16 条 QC pixel evidence、总输出 155,010,538 bytes，marker `status=complete`。
- sub-056 manifest 固定记录 Git commit `a00dd0802740662980080c28577dfbb72991deef`、模型哈希、source 哈希和 work identity；不把旧 8/26 产物计作本轮结果。

## 环境限制与队列交接

当前 `.venv_nir_gpu` 缺少 `scipy`，因此整仓 `pytest -q` 另有 6 个 behavior 测试在收集阶段失败；这些与本轮 NIR runtime 无关，未通过安装依赖掩盖。运行时必须把 `.venv_nir_gpu\Lib\site-packages\torch\lib` 临时置于 `PATH`，以加载 CUDA/cuDNN DLL。

全队列须由独立 PowerShell 终端启动，日志和 PID 位于输出根目录 `_runtime_logs`；不得绑定代理/Codex 进程，不得启动第二实例。启动前确认无有效 native batch/fullclass 进程和 `.run.lock`；已完成 sub-056 应由 valid completion 严格 skip。

## 队列启动尝试与当前阻塞

- 04:01 左右的 queue 工作树 `4d55514` 被 strict identity 拒绝 sub-056，已在 sub-057 前停止；日志 `native_batch_queue_20260828_040118.log`，PID 元数据已标记停止原因。
- 随后从验收一致的 `a00dd08` 工作树启动独立 PowerShell（shell PID 39376），dry-run 71；sub-056 已确认 `skipped_valid_completion`。
- sub-057 的历史 `.ritnet-fullclass-work\sub-057.sqlite` 与当前 identity 不同，queue 按 `continue_on_error` 继续到 sub-058；已在 sub-058 实际处理前停止。日志 `native_batch_queue_20260828_040500.log`，PID 元数据已标记 `stopped_before_sub058`。
- 未产生新的 sub-057/sub-058 final 输出；旧 workstore 未删除。重启前需将 stale SQLite 做可恢复归档或采用经确认的隔离 workstore 策略，不能静默覆盖。

# NVIDIA 完整性修复与 ORT CUDA 可选 Profile 工作记录

> 2026-08-24（Asia/Shanghai）｜本记录保存 NVIDIA `1.0.1` 的实际同步过程。已完成全量分析的历史 `1.0.0` 不被改写，由 tag `nvidia-v1.0.0` 保留。

## 背景

用户要求先修复“读帧失败或部分输出仍被批处理当成已完成”的漏洞，排除 `sub-9504`，并把 AMD 线已冻结的 ONNX 紧凑模型同步到 NVIDIA 分支作为可选高速 profile，便于多设备分开运行。

## 目标与边界

1. 保留 `.pt/.pkl` 和默认 `pytorch-cuda` 复现入口，不覆盖历史科研参数或已有结果。
2. 增加与 AMD 同源的 YOLO/RITnet ONNX 资产和显式 `ort-cuda` profile，固定 FP32、RITnet batch=16。
3. 两个 NVIDIA profile 都不允许正式模式静默退回 CPU。ORT 只注册 CUDA EP，关闭 session CPU EP fallback、Python 运行期 fallback 和 TF32。
4. 批处理只接受通过严格校验的 `completion.json: complete`；smoke、partial、读帧失败、损坏或身份不匹配不能触发 skip。

## 执行与决策过程

1. 从远端 `nvidia-cuda` 精确节点 `e63675ad15c17db6ea2ac7a3bb1c1ac6fc106e06` 创建独立 worktree，并在该节点创建、推送 annotated tag `nvidia-v1.0.0`。
2. 新增原子 `completion.json` 和统一校验器。完成判定核对运行身份、phase windows、帧主键集合、计数、CSV、summary 和 manifest。读帧失败返回 3，产物不完整返回 4。
3. `sub-9504` 写入默认排除表；当前 E 盘发现 45 人，排除后 batch 精确选择 44 人。
4. 保留 `pytorch-cuda` 默认路径，同时增加 `ort-cuda`。两者的输出目录分开：ORT 目录名带 `ort-cuda`，completion identity 也包含 backend，防止相互跳过。
5. 本机只有 AMD DirectML，没有 NVIDIA CUDA。因此本机只能验证两条拒绝路径：PyTorch CUDA 正式模式会因 CUDA 不可用直接失败；ORT 只见 `DmlExecutionProvider/CPUExecutionProvider` 时也会直接失败。真实 CUDA 速度、稳定性和跨后端 parity 必须在 NVIDIA 设备上补验，未验前不把 ORT 写成已证明更快。

## 校验结果

- runtime/formal 相关回归：32 项通过，1 项按原条件跳过；加上文档约定的 current behavior baseline 后整组为 35 项通过、1 项跳过。
- 新增覆盖：summary-only 拒绝、smoke 拒绝、身份不匹配、缺帧、读帧失败、子进程假成功、`sub-9504` 排除、ORT CUDA 不可用拒绝 CPU fallback、RITnet 尾批补位与 YOLO `conf=0.40` 坐标还原。
- ONNX 资产 SHA256 与 AMD 分支一致：YOLO `e38cc13b...`，RITnet ONNX `1933f44f...`，external data `1be9bc24...`。

## 已完成、未完成与待确认

- 已完成：代码移植、双 profile 隔离、静默 CPU fallback 拒绝、单元回归、本机失败路径验证。
- 未完成：NVIDIA 真机 `check-env`、300/600 帧短测、PyTorch CUDA 与 ORT CUDA parity/速度对照。
- 待确认：只有 ORT CUDA 在真机通过精度门和速度门后，才能作为该 worker 的正式 profile。

# 08-26-17 硬件主线 Behavior/RGB 同步阶段完成

## 本轮结论

项目已经开始从“按模态长期分支”收口为“按硬件长期分支”。截至本记录，`amd-DirectML` 与 `nvidia-cuda` 都已经包含 Behavior、NIR-Behavior 下游分析和 RGB 共享科学层；正式科研输出默认位于 Git working tree 之外。

## AMD 主线

`amd-DirectML` 当前已包含：

- 既有 AMD/DirectML NIR 正式 runtime 与 full-class 补充路线；
- FocusWave v3.1.3 BB Behavior 正式分析；
- `nir_behavior` trial/probe/phase 对齐、coverage/QC/diagnostics 与 stimulus visual covariates；
- `src/attention_pipeline/rgb/` 完整共享科学层；
- RGB Motion / Pose / Face dry-run、tracking/eyelid derived、QC；
- 已验证的 Py-Feat ONNX Runtime DirectML Face 工程工具；
- RGB 15 Hz、backend、第一档工程优化等 AMD 决策文档。

AMD 正式输出根继续位于仓库外，例如：

- NIR：`D:/_AttentionData/Beijing-NIR/...`
- Behavior：`D:/_AttentionData/Beijing-Behavior/formal-v1`
- RGB：`D:/_AttentionData/Beijing-RGB`

## NVIDIA 主线

`nvidia-cuda` 当前已包含：

- 既有 NVIDIA/CUDA NIR 稳定基线与 full-class 补充路线；
- FocusWave v3.1.3 BB Behavior 正式分析；
- 从 AMD 科学层同步的 `nir_behavior` 与 stimulus visual 分析；
- `src/attention_pipeline/rgb/` 共享科学层；
- RGB Motion / Pose / Face sampling、tracking/eyelid derived、QC 与 parity 辅助入口；
- RGB 15 Hz 与 primary/eyelid 等硬件无关决策文档。

NVIDIA Face 正式 CUDA runtime **尚未完成**：当前配置明确标记为 Py-Feat 2.1.1 native / PyTorch CUDA 路线待实现与 AMD↔NVIDIA parity。不得把共享 RGB 代码已经进入主线误写成“CUDA Face 正式推理已验证完成”。

NVIDIA 仓库外输出根当前包括：

- NIR：`D:/Project/厚粲杯/11_数据/01_Attention-Analysis_nvidia-cuda_formal_NIR`
- Behavior：`D:/Project/厚粲杯/11_数据/02_Attention-Analysis_nvidia-cuda_formal_Behavior`
- NIR-Behavior：`D:/Project/厚粲杯/11_数据/03_Attention-Analysis_nvidia-cuda_NIR-Behavior`
- RGB：`D:/Project/厚粲杯/11_数据/04_Attention-Analysis_nvidia-cuda_RGB`

## 分支状态

长期目标仍为只保留：

- `amd-DirectML`
- `nvidia-cuda`

`rgb-dev` 与 `rgb-nvidia-cuda` 暂时保留作为历史/开发保险，直到以下事项完成后再由用户明确同意删除：

1. NVIDIA PyTorch/CUDA Face runner 实现；
2. sub-031 AMD DirectML ↔ NVIDIA CUDA parity；
3. sub-033 timestamp/capture-gap stress test；
4. blink event / `perclos80_proxy` 规则冻结；
5. AMD/NVIDIA full-video formal runner 与 completion/resume/QC 收口。

`amd-DirectML-ritnet512` 已不再出现在 GitHub 远端分支列表。

另有 `analysis/nir-behavior-v2` 并行开发分支；当前相对 AMD 主线仅多一个协作规则文档，没有额外分析代码，因此本轮未删除、未覆盖、未强行合并。

## 输出与 Git 安全规则

- `/outputs/` 已加入 AMD/NVIDIA 两条主线 `.gitignore`；
- 正式结果默认写仓库外；
- `git pull`、切分支、后续选择性合并不会触碰仓库外已有正式 NIR/RGB/Behavior 结果；
- 重新运行具体 runner 时，仍需遵循各 runner 的 completion/resume/versioned-output 规则，不能把“Git 不覆盖结果”误解为“业务脚本永远不会覆盖用户主动指定的同名输出”。

## 下一步

1. 在 NVIDIA 侧实现与 Py-Feat 2.1.1 原生定义一致的 PyTorch/CUDA Face runtime；
2. 使用同一 sub-031 timestamp 集做 AMD↔NVIDIA parity；
3. 回到 sub-033 gap stress 与 blink/PERCLOS proxy 冻结；
4. 将后续 RGB 开发直接落到对应硬件主线，feature branch 仅在需要并行隔离时短期创建。

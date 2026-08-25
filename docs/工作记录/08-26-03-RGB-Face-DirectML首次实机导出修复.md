# 08-26-03｜RGB Face DirectML 首次实机导出修复

> 2026-08-26｜分支：`rgb-dev`｜承接 `08-26-02-RGB-Face-DirectML-Gate01实现.md`。本记录保存首次在 AMD 开发机实际执行 LibreFace Gate 0 时暴露的环境与 Windows 路径问题，以及对应修复；不改写既有 CPU benchmark 结果。

## 1. 实机执行结果

首次按 Gate 0 执行：

```powershell
conda activate "D:\CondaEnvs\attention-face-libreface"
python -m pip install onnx
```

pip 安装了 `onnx 1.19.1`，并把原有 `protobuf 3.20.3` 升级为 `protobuf 6.33.6`。该环境已有 `mediapipe 0.10.5`，其依赖要求为 `protobuf<4,>=3.11`，因此环境被破坏。随后导出启动时出现：

```text
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
```

这不是 ONNX 图或 DirectML 执行错误，而是 LibreFace reference 环境内 MediaPipe 与 protobuf 的依赖冲突。

## 2. 环境修复与固定版本

LibreFace reference/export 环境后续固定：

```powershell
python -m pip uninstall -y onnx protobuf
python -m pip install "protobuf==3.20.3" "onnx==1.16.2"
python -c "import google.protobuf, mediapipe, onnx; print('protobuf', google.protobuf.__version__); print('mediapipe', mediapipe.__version__); print('onnx', onnx.__version__)"
```

预期：

```text
protobuf 3.20.3
mediapipe 0.10.5
onnx 1.16.2
```

这里刻意只重装 `onnx/protobuf`，不使用 `--force-reinstall`，避免连带重装或升级已有稳定的 NumPy、Torch 等依赖。以后不得在 `attention-face-libreface` 中裸执行 `pip install onnx`。DirectML runtime 仍使用独立 `attention-face-directml` 环境，因此该 pin 只用于 LibreFace PyTorch checkpoint → ONNX 的 Gate 0 导出，不限制独立 DirectML 环境的 ORT 版本。

## 3. Windows checkpoint path 问题

环境冲突之外，首次导出还暴露：

```text
FileNotFoundError: [WinError 3] 系统找不到指定的路径。: ''
```

traceback 位于 LibreFace `download_weights()`。LibreFace 0.2.0 当前实现通过：

```python
model_dir = "/".join(model_path.split("/")[:-1])
```

手工计算 checkpoint 父目录，而不是使用 `os.path.dirname()` / `Path.parent`。原 Gate 0 脚本用 `Path` 在 Windows 上生成纯反斜杠路径，例如：

```text
D:\...\AU_Recognition\weights\combined_repvgg.pt
```

LibreFace 按 `/` 分割后无法得到父目录，最终调用 `os.makedirs("")` 并失败。

修复：`scripts/face_export_libreface_onnx.py` 新增 `_libreface_path()`，只在传入 LibreFace 内部 checkpoint API 时把路径解析为 Windows 同样可用的 forward-slash 形式：

```text
D:/.../AU_Recognition/weights/combined_repvgg.pt
```

用户命令行参数仍保持正常 Windows `D:\...` 写法，不需要手动修改数据目录。

## 4. Git 修复

- `fd28e32`：`fix(rgb): make LibreFace ONNX export Windows-safe`
- `b16335b`：`docs(rgb): pin LibreFace ONNX export dependencies`
- `f6360b8`：`docs(rgb): avoid force-reinstall in LibreFace export env`

045 已同步写入 LibreFace export 的 protobuf/ONNX 固定版本与 Windows 路径说明。

## 5. Gate 0 实机复测：已通过

完成上述修复后，LibreFace Gate 0 已在当前 AMD 开发机成功完成。实际 export manifest 为 `rgb-face-libreface2-onnx-export-v0.1`，运行环境：

```text
Python 3.9.25
Windows 10.0.19045
Torch 2.0.0+cpu
ONNX 1.16.2
LibreFace 0.2.0
opset 17
```

三套 ONNX 与源 checkpoint 均成功落盘并记录 SHA256：

| role | ONNX | ONNX SHA256 | source checkpoint SHA256 |
|---|---|---|---|
| AU joint | `libreface2_au_joint.onnx` | `49533731a55147a1e33f0aeb30de711b6a34694e933aa7390aa4499f03878568` | `d7d11d9f029afef0d36306f91f5e67fd20531390864ab014e08683dbaca3954b` |
| expression | `libreface2_expression.onnx` | `16b60d5943da9da8843a0a17f398f62a72e6f2f404a665cdedcdecce8cb412e2` | `bcf48160b13b881ea9a7a67080b2d0177b1ce0a1ea4845ca74c24eeb728e6b2d` |
| gaze MLP | `libreface2_gaze_mlp.onnx` | `22ef72133d1aa23486bd02a4a9f9018526d36e46e5f5dac0d9c2bb2557c24b5d` | `88cc628f085f758b918789f9b4c25765363811b9c54a3120f2f3727d1801052b` |

导出合同保持当前 LibreFace Python reference：AU 为 224×224 ImageNet normalize 后的联合 intensity/detection head；expression 为 8 类 score；gaze MLP 输入为 MediaPipe 前 468 landmarks 展平后的 1404-d 特征。MediaPipe alignment/head pose/landmark 与 gaze feature extraction 仍保留在 CPU 前处理，Gate 0 只迁移 learned heads。

因此当前状态正式更新为：**LibreFace Gate 0 = PASS**。没有重跑既有 300 帧 CPU benchmark，也没有冻结 Face backend。

## 6. 当前续接位置

下一步直接进入 LibreFace Gate 1：

1. 切换/创建独立 `attention-face-directml` 环境；
2. 确认 `DmlExecutionProvider` 可见；
3. 对上述 3 个 ONNX 运行 batch 1/8/16/32 synthetic provider/fallback/model-core probe；
4. 检查每个模型的 `status`、`session_providers`、DML/CPU kernel events 与吞吐；
5. Gate 1 通过后再做 Py-Feat Gate 0/1，然后进入同 300 帧真实输入 parity/end-to-end。

当前仍不运行新的 CPU benchmark，不运行 44 被试全量 Face，不改变 Pose/Motion 已有决策。
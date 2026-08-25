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
python -m pip install --force-reinstall "protobuf==3.20.3" "onnx==1.16.2"
python -c "import google.protobuf, mediapipe, onnx; print('protobuf', google.protobuf.__version__); print('mediapipe', mediapipe.__version__); print('onnx', onnx.__version__)"
```

预期：

```text
protobuf 3.20.3
mediapipe 0.10.5
onnx 1.16.2
```

以后不得在 `attention-face-libreface` 中裸执行 `pip install onnx`。DirectML runtime 仍使用独立 `attention-face-directml` 环境，因此该 pin 只用于 LibreFace PyTorch checkpoint → ONNX 的 Gate 0 导出，不限制独立 DirectML 环境的 ORT 版本。

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

045 已同步写入 LibreFace export 的 protobuf/ONNX 固定版本与 Windows 路径说明。

## 5. 当前续接位置

下一步仍然是 Gate 0，不回退、不重复 CPU benchmark：

1. 在 `attention-face-libreface` 修复 protobuf/ONNX 版本；
2. `git pull --ff-only` 拉取 Windows path 修复；
3. 重新执行同一条 `face_export_libreface_onnx.py`；
4. 成功生成 3 个 ONNX + export manifest 后，再进入 `attention-face-directml` Gate 1；
5. LibreFace Gate 1 通过后再做 Py-Feat Gate 0/1 与同 300 帧真实输入 parity/end-to-end。

本次失败没有产生新的 CPU benchmark 结果，不改变 Face backend 尚未冻结的状态。
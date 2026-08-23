# LabelImg 标注工具环境

本目录保存 NIR 眼框数据标注时真正需要长期保留的 LabelImg 配置与兼容补丁。不要把完整 Python venv 当作可移植资产提交到 Git；环境应按本文重新建立。

## 固定环境

- Windows 64-bit
- Python 3.10
- LabelImg 1.8.6
- PyQt5 5.15.11

依赖版本见 `requirements.txt`。

## 从零重建

建议把虚拟环境建在仓库外，或使用仓库内被 `.gitignore` 忽略的 `.venv-labelimg/`：

```powershell
py -3.10 -m venv .venv-labelimg
.\.venv-labelimg\Scripts\python.exe -m pip install --upgrade pip
.\.venv-labelimg\Scripts\python.exe -m pip install -r tools\labelimg\requirements.txt
.\.venv-labelimg\Scripts\python.exe tools\labelimg\patch_labelimg.py
```

补丁脚本会把 LabelImg 1.8.6 中 4 个可能向 Qt `setValue()` 传入 float 的位置改为 `int(...)`。脚本可重复运行；已经打过补丁时不会再次修改。

## 启动

```powershell
.\.venv-labelimg\Scripts\python.exe tools\labelimg\launch.py `
  datasets\nir-eye-v1\images\batch1 `
  datasets\nir-eye-v1\classes.txt `
  datasets\nir-eye-v1\labels_yolo\batch1
```

`launch.py` 在导入 PyQt5 前把 Qt platform plugins 复制到 ASCII-only 路径，规避仓库或 Python 环境位于中文路径时 `qwindows.dll` 加载失败的问题。默认复制到：

```text
%LOCALAPPDATA%\attention-analysis\qtplugins
```

也可通过环境变量 `QT_PLUGINS_ASCII` 自定义位置。

## 标注规则

- 格式：YOLO。
- 单类：`eye`，不区分左右眼。
- 框包含眼裂、上下眼睑和少量眼周。
- 闭眼仍标注；完全看不到眼睛时不造框，并在数据 manifest/annotation 状态中记录。
- 原始数据版本见 `datasets/nir-eye-v1/README.md`。

## 历史环境

仓库根目录当前仍暂存 `venv-labelimg/`，它是 2026-08 标注阶段使用过的完整 Windows venv。该目录不再作为推荐入口；待确认删除后，只保留本目录的重建配置和补丁即可。

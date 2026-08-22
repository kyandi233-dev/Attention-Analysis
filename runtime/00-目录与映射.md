# 00｜Runtime 目录与映射

`runtime/` 保存可复现运行环境、可运行分析包及其依赖快照。它不是正式分析结果输出目录。

## 当前正式 runtime

```text
nir-formal/
```

这是已经用于正式全量分析的 NIR runtime。它由历史目录 `nir-yolo-tracking-ritnet-v1/` 重命名而来；正式模式采用逐帧 YOLO，并保留 tracking 代码用于诊断/历史复现。

目录内包含：

- 正式运行配置 `config.yaml`
- 单被试/正式 batch 入口
- FocusWave v3.1.3 phase window 映射
- YOLO + RITnet 正式推理代码
- 当前正式模型权重副本
- requirements、测试与校验信息

## 历史 runtime 资产

| 路径 | 角色 |
|---|---|
| `nir-yolo-tracking-ritnet-v1.zip` | 较早阶段生成的可迁移压缩包快照，保留原历史名称 |
| `nir-yolo-tracking-ritnet-v1.zip.sha256` | 上述压缩包完整性校验 |
| `PyPupilEXT-0.0.1-cp310-cp310-win_amd64.whl` | PyPupilEXT 本地 wheel 依赖 |
| `requirements-main.txt` | 历史主环境依赖快照 |
| `requirements-pupil.txt` | 历史 pupil 环境依赖快照 |

## 命名原则

正式 unpacked runtime 使用用途名称 `nir-formal/`，不再把可替换算法组合或历史版本号写进目录名。历史 zip、工作记录和旧分支继续保留当时名称，不改写。

## 输出边界

正式分析结果应存放在独立分析输出位置，而不是作为长期产物堆在 `runtime/`。runtime 内部出现的 `outputs/` 仅属于运行配置/运行期路径，不改变“runtime = 可运行环境”的目录职责。

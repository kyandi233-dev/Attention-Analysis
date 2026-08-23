# 08-23-03｜RITnet来源与正式Runtime复现审计工作记录

> 文档性质：仓库整理 / 正式 runtime provenance 审计  
> 日期：08-23  
> 当天序号：03  
> 当前维护主线：`main`  
> 状态：已完成本轮审计

## 一、审计目的

确认当前正式 NIR runtime 中使用的 RITnet 源码和权重究竟来自哪里，判断：

1. `runtime/nir-formal/ritnet/` 是否足以作为正式运行的冻结实现；
2. `models/external/ritnet/` 的完整上游仓库是否仍是正式运行必要依赖；
3. RITnet 应放在 `runtime/`、`tools/` 还是其他位置。

## 二、正式 runtime 当前 RITnet 结构

当前正式 runtime 已包含：

```text
runtime/nir-formal/
├── ritnet/
│   ├── License.md
│   ├── densenet.py
│   └── UPSTREAM.md
├── ritnet_runtime.py
└── models/
    └── ritnet-best_model.pkl
```

其中：

- `densenet.py`：RITnet 网络定义；
- `ritnet-best_model.pkl`：正式冻结权重；
- `License.md`：上游 license；
- `ritnet_runtime.py`：Attention-Analysis 自己的正式运行适配层；
- `UPSTREAM.md`：本轮新增的来源与冻结版本说明。

## 三、上游来源确认

上游正式仓库确认 为：

```text
AayushKrChaudhary/RITnet
https://github.com/AayushKrChaudhary/RITnet
```

上游默认分支：

```text
master
```

本轮核对的上游 commit：

```text
6431c57ce7bf0eda935fb6178b926ae9440b50bf
```

对应论文为 Chaudhary et al. (2019) 的 RITnet ICCVW 论文。

## 四、文件一致性核对

### 1. `densenet.py`

正式 runtime：

```text
runtime/nir-formal/ritnet/densenet.py
```

Git blob SHA：

```text
9bc49f0e285a9dc26d4885ab7f74cf3c5fdbe59a
```

该 SHA 与上游 commit `6431c57...` 的 `densenet.py` 完全一致。

### 2. 正式权重

正式 runtime：

```text
runtime/nir-formal/models/ritnet-best_model.pkl
```

Git blob SHA：

```text
f0864e6651f578525a9101c7ca787e23d2d201d7
```

该 SHA 同时与：

```text
models/external/ritnet/best_model.pkl
```

以及上游 commit `6431c57...` 的 `best_model.pkl` 完全一致。

### 3. License

runtime 保留的 `License.md` 与完整上游副本使用同一 Git blob，因此正式 runtime 已保留必要的上游授权文件。

## 五、RITnet 应放在哪里

结论：**正式运行所需的冻结 RITnet 源码继续放在 `runtime/nir-formal/`。**

不建议放在 `tools/`。

原因是：

- `tools/` 用于 LabelImg 等独立辅助工具；
- RITnet 是正式 NIR pipeline 的直接算法依赖；
- 换电脑执行正式分析时，runtime 必须直接包含它；
- 把 RITnet 放到 `tools/` 会使“辅助工具”和“正式算法依赖”语义混淆。

## 六、完整上游仓库是否必须保留

当前：

```text
models/external/ritnet/
```

保存完整上游仓库副本，包括训练、测试、dataset、environment、augmentation 示例和正式权重。

但当前正式 runtime 实际只需要其中的：

```text
densenet.py
best_model.pkl
License.md
```

并通过本项目的：

```text
ritnet_runtime.py
```

接入正式 pipeline。

因此，**`models/external/ritnet/` 并不是当前正式 NIR 运行的必要依赖。**

它是否可以从 `main` 删除，下一步只需再确认一件事：是否仍有需要保留在 `main` 的历史脚本直接调用 `models/external/ritnet/` 内的 `infer_ritnet.py`、训练脚本或其他上游文件。

如果不存在这种仍需主线运行的依赖，则完整上游副本可以删除；正式复现依赖由 `runtime/nir-formal/` 保证，上游完整历史还可以由 Git 历史 / 冻结历史版本追溯。

任何实际删除仍需用户明确批准。

## 七、本轮新增文件

```text
runtime/nir-formal/ritnet/UPSTREAM.md
```

该文件记录：

- 上游仓库；
- 上游 commit；
- 正式 runtime 中关键文件的 blob SHA；
- 哪些文件来自上游；
- 哪些文件属于本项目适配层；
- 正式运行与完整上游仓库的边界。

## 八、下一步

继续审计 `models/external/` 与 `models/historical/`：

1. 列出每个模型 / 源码目录对应的历史脚本；
2. 判断该模型是否仍属于当前正式 pipeline；
3. 判断是否为新设备正式配置所必需；
4. 判断是否只需要通过历史版本与工作记录保留；
5. 形成明确的“保留 / 可删 / 待确认”清单后，再由用户授权删除。

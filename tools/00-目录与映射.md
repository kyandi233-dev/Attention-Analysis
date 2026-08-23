# Tools

`tools/` 保存项目开发、标注和调试使用的独立辅助工具。这里的内容不属于 `attention_pipeline` Python 包，也不属于正式 Behavior/NIR runtime。

## 当前工具

| 路径 | 用途 | 与正式分析关系 |
|---|---|---|
| `labelimg/` | NIR 眼框数据的离线标注工具 | 训练数据准备工具；正式推理不依赖 |

历史 `clipboard-vision-mcp/` 属于开发辅助工具，正式分析从未依赖，已按用户授权从 current branch 删除；需要追溯时使用 Git 历史。

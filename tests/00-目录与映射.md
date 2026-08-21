# 00｜测试目录与映射

> 2026-08-13 15:35（Asia/Shanghai）｜测试覆盖正式协议、行为证据、时间轴、NIR 几何、缺失语义和标注预览。

| 文件 | 覆盖范围 |
|---|---|
| `test_protocol_and_config.py` | ABCCBA、18×12、四探针、四状态、审批开关 |
| `test_behavior.py` | RT保留/QC、d′分母、固定位置、窗口边界与后验 |
| `test_io.py` | dropped 行与 AVI 位置映射、block 解析 |
| `test_nir.py` | 解剖眼别、固定 ROI、仿射、三点椭圆、缺失 EAR/PERCLOS |
| `test_review.py` | 264/528 设计、seed复现语义、盲标字段与保存/恢复接口 |

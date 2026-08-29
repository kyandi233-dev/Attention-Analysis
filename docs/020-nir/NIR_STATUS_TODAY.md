# NIR 正式队列进度

更新时间：2026-08-29（本机只读观测）

## 当前状态

- 状态：RUNNING
- 当前 subject：`sub-085`
- GPU 队列：1 条有效 parent-child 链；未启动第二条 GPU 队列
- 正式环境：`D:\CondaEnvs\nir-nvidia\python.exe`
- launcher commit：`d5fcbc9`（Conda 固定解释器、CUDA DLL 自动注入、CUDA EP fail-closed）
- Git 分支：`nvidia-cuda`
- 输出根：`D:\Project\厚粲杯\11_数据\01_Attention-Analysis_nvidia-cuda_formal_NIR`
- 运行边界：不重跑 YOLO；不修改 RITnet、Topology、QC、metrics 或 schema

## 已完成 subject

每个 `complete` 必须存在并通过检查：`completion.json`、eye rows、frame rows、
80 张 QC 图和 16 条 pixel evidence。

| Subject | Status | Eye rows | Frame rows | QC | Pixel |
|---|---:|---:|---:|---:|---:|
| sub-056 | complete | 73207 | 38903 | 80 | 16 |
| sub-057 | complete | 76298 | 38704 | 80 | 16 |
| sub-058 | complete | 81594 | 41058 | 80 | 16 |
| sub-059 | complete | 77725 | 39411 | 80 | 16 |
| sub-062 | complete | 78419 | 39482 | 80 | 16 |
| sub-064 | complete | 75145 | 38542 | 80 | 16 |
| sub-065 | complete | 82106 | 41527 | 80 | 16 |
| sub-067 | complete | 82529 | 41759 | 80 | 16 |
| sub-068 | complete | 76304 | 38964 | 80 | 16 |
| sub-070 | complete | 88718 | 45138 | 80 | 16 |
| sub-071 | complete | 82999 | 42537 | 80 | 16 |
| sub-072 | complete | 85061 | 43343 | 80 | 16 |
| sub-073 | complete | 79106 | 41099 | 80 | 16 |
| sub-074 | complete | 80061 | 40321 | 80 | 16 |
| sub-075 | complete | 79350 | 40200 | 80 | 16 |
| sub-076 | complete | 75140 | 38244 | 80 | 16 |
| sub-077 | complete | 82805 | 42124 | 80 | 16 |
| sub-078 | complete | 81340 | 41809 | 80 | 16 |
| sub-081 | complete | 76155 | 38801 | 80 | 16 |
| sub-082 | complete | 78140 | 39888 | 80 | 16 |
| sub-083 | complete | 61490 | 39875 | 80 | 16 |
| sub-084 | complete | 54800 | 41643 | 80 | 16 |

已完成：22 个 subject。`sub-085` 的 `completion.json` 尚未发布，仍按 RUNNING
处理；只有完成文件和全部产物通过检查后才登记为 complete。

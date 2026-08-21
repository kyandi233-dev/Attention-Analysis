# AGENTS.md｜attention-pipeline-v2入口

> 2026-08-13 18:53（Asia/Shanghai）｜本仓库是当前可审计分析管线；先验证测量和行为证据，再讨论专注评分。

## 必读入口

开始工作前完整读取：

1. `D:/AAAWORK/07-竞赛/厚璨杯/000-项目定位和进度/000-厚璨杯项目记忆.md`
2. 本仓库`README.md`与`000-项目总览与架构.md`
3. `docs/00-目录与映射.md`、相应模块入口、`docs/900-工作记录.md`
4. `configs/preexperiment.yaml`

## 仓库边界

- 原始数据、正式实验程序和v1管线只读；v2输出只写`D:/_AttentionData/output-v2/040-pre-experiment`。
- 当前NIR须从现状审计和小样本复核继续；未经批准不生成完整528眼真值集、不运行算法全基准、不进行11人正式提取。
- RGB和跨模态接口当前关闭；不得迁移未验证的rPPG/HRV结论。
- 专注总分尚未冻结；行为、NIR质量与评分模型必须分层。
- 正式修改前列出文件、字段、算法、参数、图表、测试和停止点，获得用户确认后执行。
- 准确率评价阶段禁止用插值掩盖失败；缺失、拒绝、观测和插值分轨保存。

## 环境

- 主流程：`D:/Code/python/python.exe`
- pupil-detectors与PyPupilEXT：`D:/AAAWORK/07-竞赛/厚璨杯/venv-pupil/Scripts/python.exe`
- 当前只使用上述两个已验证解释器；不自行新建或切换环境。

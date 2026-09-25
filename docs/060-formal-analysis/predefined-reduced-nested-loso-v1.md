# 预定义精简模型嵌套验证 v1

## 1. 研究角色

该分析是冻结二分类完整模型之后的次要精简验证，不修改确认性主分析。它回答：在完全未见参与者上，由训练参与者内部选择的少量预定义特征集，能否保持或改善当前完整模型的概率预测。

外层按 `participant_group_id` 做留一参与者验证（leave-one-subject-out [LOSO]）。每个外层训练集内部使用参与者分组五折交叉验证（participant-grouped five-fold cross-validation [CV]），同时选择候选特征集与 L2 正则逻辑回归的 `C`。外层测试参与者的标签、表现和消融结果不得进入选择、插补、标准化或模型拟合。

## 2. 冻结候选

配置入口为 `configs/supervised_predefined_reduced_v1.yaml`。候选仅有三套：

1. `full_11`：当前冻结的 11 个完整特征；
2. `behavior_core`：No-Go 抑制错误率与反应时水平；
3. `behavior_core_plus_ocular`：Behavior-core 加入当前完整眼部特征。

候选不是从本次外层结果继续增删得到。旧消融结果只用于形成有限候选，不作为跨全样本的直接筛选器。

## 3. 运行与核验

```powershell
$env:PYTHONPATH = 'src'
python scripts/supervised_predefined_reduced_nested.py `
  --design-config configs/supervised_predefined_reduced_v1.yaml `
  --input-table <AS.full.csv> `
  --output-root <output-root> `
  --run-id AS.full__predefined_reduced_nested__included_missing_aware

python scripts/verify_predefined_reduced_run.py `
  --run-dir <new-run-dir> `
  --frozen-run-dir <frozen-full-run-dir>
```

核验器逐行比较冻结完整模型的新旧折外概率，并检查外层与内层参与者互斥、60 个外层折完整、候选集合未越界及失败表为空。关键产物为：

- `predefined_reduced_summary.json`；
- `predefined_reduced_verification.json`；
- `outer_fold_candidate_selection.csv`；
- `candidate_selection_frequency.csv`；
- `paired_model_increments.csv`；
- `paired_increment_bootstrap.json`；
- `probe_predictions.csv` 与 `fold_audits.json`。

## 4. 解释边界

主要差值定义为“冻结完整模型损失 − 训练内选择模型损失”，正值才表示精简流程更好。比较必须基于相同参与者、相同探针的成对折外预测，并用固定折外参与者簇自助法计算 95% 置信区间（confidence interval [CI]）。

内层选择频率不能单独证明某个特征集更好；正式判断以外层成对差值及其区间为准。若区间跨 0，只能报告未发现精简流程改善泛化的证据，不能把内层频繁入选解释为完整模型存在可删除特征。

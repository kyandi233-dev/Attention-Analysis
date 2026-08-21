from attention_pipeline.behavior.reporting import (
    _design_audit,
    _field_dictionary,
    _program_recommendations,
    load_cohort,
)
from attention_pipeline.behavior.evidence import cohort_probe_evidence


def test_phase4_recommendations_are_proposals_not_implemented_changes():
    recommendations = _program_recommendations()
    assert set(recommendations["decision_class"]) == {
        "可直接纳入下一版",
        "需新实验版本验证",
        "需操作性定义讨论",
        "当前不应实施",
    }
    assert recommendations["recommendation_id"].is_unique
    assert recommendations["recommendation"].str.contains("专注分数").sum() == 0


def test_phase4_audit_records_fixed_design_without_claiming_default_probe():
    audit = _design_audit().set_index("audit_item")
    assert "ABCCBA" in audit.loc["Block顺序", "observed_design"]
    assert "current_choice初始为None" in audit.loc["探针界面", "observed_design"]
    assert "30/82/137/191" in audit.loc["探针位置", "observed_design"]


def test_field_dictionary_keeps_qc_and_candidate_evidence_separate(config):
    cohort = load_cohort(config)
    probes = cohort_probe_evidence(config, cohort)
    dictionary = _field_dictionary(cohort, probes)
    indexed = dictionary.set_index(["layer", "field"])
    assert "仅标记，不删除" in indexed.loc[("v2逐试次", "rt_qc_lt_150"), "missing_semantics"]
    assert indexed.loc[("探针前证据窗", "window_status"), "freeze_status"] == "候选字段，未冻结评分"
    assert indexed.loc[("建议新增", "schedule_hash"), "freeze_status"] == "待批准"

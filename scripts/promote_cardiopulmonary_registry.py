"""Promote the two cardiopulmonary features from supplementary to formal registration.

方法依据：
- `分析设计/1.15.9-毫米波生成链合同修复与time-legality处置_20260913.md` §3（四个前置条件）
- FocusWave-Formal-Analysis 默认分支已合并的裁决 PR #16（解除 time-legality 的正式记录）
- 上游 `greenboo26/focuswave-multimodal-attention-analysis` 已合并 PR #44，M1 成为 main 的
  canonical 状态（main `527ff09`，M1 执行提交 `01da845e`）

本脚本以**补充配置**为基底（其中已登记全部 13 个特征），只翻转两个心肺特征的字段：
- `time_legality_status` → `verified_pre_probe_only`
- `time_legality_evidence` → 指向已 canonical 的 M1 证据
- 五个正式资格标志打开；`supplementary_model_eligible` 关闭
- `allowed_device_packages` → 含毫米波的四套包 [M2, M4, M6, M7]
- **保持 `physiology_qualification: LIMITED_SUPPORTING_ONLY`**：time-legality（合同/来源）
  与生理效度是两条不同的轴，M1 只闭合前者；生产端对 HR/BR 的生理判定不变。

并断言**其余 11 个特征逐字段未变**。
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

WORKTREE = Path(r"D:\Project\.codex-worktrees\Attention-Analysis-q1-4class-20260913")
SOURCE = WORKTREE / "configs" / "supervised_learning_v2_supplementary_cardiopulmonary.yaml"
TARGET = WORKTREE / "configs" / "supervised_learning_v3_cardiopulmonary.yaml"

CARDIOPULMONARY_IDS = ("cardiopulmonary.hr.fused.v1", "cardiopulmonary.br.v1")
MMWAVE_PACKAGES = ["M2", "M4", "M6", "M7"]

EVIDENCE = (
    "MMWAVE M1 producer contract and provenance closure, now canonical on the mmWave repository "
    "main branch (merge of PR #44; main 527ff09). Producer execution commit 01da845e with a clean "
    "worktree and four source SHA-256 values independently rechecked. Frozen producer time "
    "contract: scientific clock = timestamp CSV zero-based column 1 (DLL host receive/enqueue "
    "time), column 2 is QC-only; formal frame membership is strictly right-open "
    "[window_effective_start_unix_ms, probe_onset_unix_ms) with window_end equal to probe_onset "
    "exactly and a frame exactly at probe onset excluded; window_effective_start_unix_ms is the "
    "actual slicing lower bound; J and E share one process_probe() implementation; "
    "mmwave_hr_usable_window_fraction now carries the producer course signal_quality.usable_ratio "
    "instead of a binary availability indicator. Real-cohort old-vs-new audit v2 PASS over all "
    "2320 probes: key and frame-audit conservation, zero strict new-frame membership violations, "
    "zero stateless same-membership violations, zero stateful HR violations when local membership "
    "and the incoming previous_bpm anchor are both equal, and all 54 same-membership HR "
    "differences explained by previous_bpm anchor divergence. Boundary: this closes the 1.15.9 §3 "
    "time-legality gate for contract and provenance ONLY. It is not a physiology validation: "
    "physiology_qualification remains LIMITED_SUPPORTING_ONLY."
)


def main() -> int:
    source = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    target = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))

    pipeline = dict(target["pipeline"])
    pipeline["name"] = "focuswave-supervised-learning-v3-cardiopulmonary"
    pipeline["version"] = "3.0.0"
    pipeline["inherits_frozen_registry"] = "configs/supervised_learning_v1.yaml"
    pipeline["cardiopulmonary_promotion"] = (
        "1.15.9 §3 time-legality gate closed for contract and provenance after the M1 producer "
        "closure became canonical on the mmWave main branch; the two cardiopulmonary features are "
        "promoted from supplementary to formal prediction eligibility. Physiology qualification is "
        "unchanged (LIMITED_SUPPORTING_ONLY) and is a separate axis."
    )
    target["pipeline"] = pipeline

    registry = target["feature_registry"]
    registry["status"] = "frozen_with_cardiopulmonary"
    registry["frozen_from"] = dict(registry.get("frozen_from", {}))
    registry["frozen_from"]["cardiopulmonary"] = (
        "mmWave/mmwave_cardiopulmonary_endpoint_guard_e352dce_20260913/"
        "mmwave_cardiopulmonary_taskb_source.csv"
    )
    registry["schema_version"] = "1.16.10-time-legality-v1"

    promoted: list[dict[str, object]] = []
    for feature in registry["features"]:
        if feature["feature_id"] not in CARDIOPULMONARY_IDS:
            continue
        feature["time_legality_status"] = "verified_pre_probe_only"
        feature["time_legality_evidence"] = EVIDENCE
        feature["standalone_eligible"] = True
        feature["behavior_increment_eligible"] = True
        feature["behavior_reference_eligible"] = False
        feature["modality_model_eligible"] = True
        feature["full_model_eligible"] = True
        feature["full_leave_one_out_eligible"] = True
        feature["allowed_device_packages"] = list(MMWAVE_PACKAGES)
        feature["supplementary_model_eligible"] = False
        feature["report_role"] = "formal_supporting"
        feature["physiology_qualification"] = "LIMITED_SUPPORTING_ONLY"
        feature["description"] = (
            "Millimetre-wave derived cardiopulmonary estimate. Formally registered after the M1 "
            "producer contract and provenance closure became canonical upstream: the 1.15.9 §3 "
            "time-legality gate is closed for contract and provenance. Physiology remains "
            "LIMITED_SUPPORTING_ONLY (HR/BR are HOLD / SUPPORTING_ONLY upstream); the time-legality "
            "closure is not a physiology validation, and this limitation must be carried with every "
            "result that uses this feature."
        )
        promoted.append(
            {
                "feature_id": feature["feature_id"],
                "columns": feature["columns"],
                "modality": feature["modality"],
                "source_namespace": feature["source_namespace"],
                "required_devices": feature["required_devices"],
                "temporal_anchor": feature["temporal_anchor"],
                "temporal_scope": feature["temporal_scope"],
                "time_legality_status": feature["time_legality_status"],
                "allowed_device_packages": feature["allowed_device_packages"],
                "physiology_qualification": feature["physiology_qualification"],
                "report_role": feature["report_role"],
            }
        )

    if len(promoted) != len(CARDIOPULMONARY_IDS):
        raise AssertionError(f"expected to promote {CARDIOPULMONARY_IDS}; promoted {len(promoted)}")

    # --- 其余特征必须逐字段未变 ---------------------------------------------------------
    source_features = {f["feature_id"]: f for f in source["feature_registry"]["features"]}
    changed: list[str] = []
    for feature in registry["features"]:
        feature_id = feature["feature_id"]
        if feature_id in CARDIOPULMONARY_IDS:
            continue
        if feature != source_features[feature_id]:
            changed.append(feature_id)
    if changed:
        raise AssertionError(f"non-cardiopulmonary features changed unexpectedly: {changed}")

    # 冻结段必须逐键相同（除 pipeline 与 feature_registry 之外）
    frozen_drift = {
        section: {"source": source.get(section), "target": target.get(section)}
        for section in ("paths", "task", "validation", "preprocessing", "models", "uncertainty")
        if source.get(section) != target.get(section)
    }
    if frozen_drift:
        raise AssertionError(f"frozen sections drifted: {sorted(frozen_drift)}")

    TARGET.write_text(
        yaml.safe_dump(target, allow_unicode=True, sort_keys=False, width=110),
        encoding="utf-8",
    )

    print(json.dumps(
        {
            "target": str(TARGET),
            "n_features_total": len(registry["features"]),
            "promoted": promoted,
            "non_cardiopulmonary_features_changed": changed,
            "frozen_sections_identical": not frozen_drift,
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

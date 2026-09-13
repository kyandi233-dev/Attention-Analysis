"""Generate the 1.16.26 Ocular head-motion sensitivity config from the frozen binary config.

方法权威：`分析设计/1.16.26-瞳孔头动协变量敏感性预注册_20260913.md` §5

本脚本不重打冻结值，而是**以冻结的二分类配置为基底**深拷贝，仅替换模型声明：
- 删除 `feature_registry`（非空 registry 优先，保留它就回不到兼容 `feature_schemes` 路径）；
- 写入两个 model family（各一条候选），使两个模型**各自独立**做内层 C 选择，而不是
  在同一个 family 内被内层 CV 择优；
- 其余 `pipeline` / `validation` / `preprocessing` / `models` / `uncertainty` / `outputs`
  逐键保持与冻结配置相同，并在写盘后**逐项断言相等**。
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

WORKTREE = Path(r"D:\Project\.codex-worktrees\Attention-Analysis-q1-4class-20260913")
SOURCE = WORKTREE / "configs" / "supervised_learning_v1.yaml"
TARGET = WORKTREE / "configs" / "supervised_learning_ocular_headmotion_sensitivity_v1.yaml"

OCULAR_COLUMNS = [
    "ocular__rseg_hard__rgb_nir_qc__level_mean__2s",
    "ocular__rseg_hard__rgb_nir_qc__variability_mad__2s",
    "ocular__rseg_hard__rgb_nir_qc__linear_slope_per_sec__2s",
    "ocular__rseg_hard__rgb_nir_qc__quadratic_curvature_per_sec2__2s",
    "ocular__blink_event_rate_per_min__pre30s",
]
HEADMOTION_COLUMNS = [
    "pose_lateral_right_per_sec_median",
    "pose_vertical_up_per_sec_median",
]
#: 必须与冻结配置逐键相等的段。
FROZEN_SECTIONS = ("validation", "preprocessing", "models", "uncertainty", "outputs")


def main() -> int:
    source = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))

    target = {key: value for key, value in source.items() if key != "feature_registry"}
    pipeline = dict(target["pipeline"])
    pipeline["name"] = "focuswave-supervised-learning-v1-ocular-headmotion-sensitivity"
    pipeline["evidence_method_entry"] = (
        "分析设计/1.16.26-瞳孔头动协变量敏感性预注册_20260913.md"
    )
    pipeline["sensitivity_note"] = (
        "1.16.26 Ocular head-motion covariate sensitivity. This config deliberately "
        "omits feature_registry and uses the compatibility feature_schemes path, which "
        "does NOT run the registry time-legality gate; the two pose covariates' pre-probe "
        "legality rests on the frozen producer handoff evidence strings quoted in the "
        "preregistration §3. The head-motion arm is a nuisance-adjusted sensitivity arm, "
        "never a scientific modality model."
    )
    target["pipeline"] = pipeline

    target["feature_schemes"] = {
        "model_families": {
            "ocular_reference": {
                "candidates": [
                    {
                        "feature_set_id": "ocular_reference",
                        "columns": list(OCULAR_COLUMNS),
                        "description": (
                            "1.16.26 sensitivity baseline: the five registered Ocular "
                            "predictors only."
                        ),
                        "modality_blocks": ["ocular"],
                        "required_devices": ["nir", "rgb"],
                    }
                ]
            },
            "ocular_plus_headmotion": {
                "candidates": [
                    {
                        "feature_set_id": "ocular_plus_headmotion",
                        "columns": list(OCULAR_COLUMNS) + list(HEADMOTION_COLUMNS),
                        "description": (
                            "1.16.26 sensitivity arm: the same five Ocular predictors plus "
                            "the two pre-specified head-motion covariates from the frozen "
                            "Movement handoff. Nuisance-adjusted sensitivity only; this is "
                            "NOT a modality model and must never be reported as Movement "
                            "contributing information."
                        ),
                        "modality_blocks": ["ocular", "movement"],
                        "required_devices": ["nir", "rgb"],
                    }
                ]
            },
        },
        "note": (
            "Compatibility interface by design (preregistration §5): the registry cannot "
            "express 'Ocular plus non-Ocular nuisance covariates'."
        ),
    }

    TARGET.write_text(
        yaml.safe_dump(target, allow_unicode=True, sort_keys=False, width=100),
        encoding="utf-8",
    )

    # --- 逐项断言冻结值未漂移 --------------------------------------------------------
    written = yaml.safe_load(TARGET.read_text(encoding="utf-8"))
    diffs: dict[str, object] = {}
    for section in FROZEN_SECTIONS:
        if written.get(section) != source.get(section):
            diffs[section] = {
                "frozen": source.get(section),
                "written": written.get(section),
            }
    leftovers = [key for key in source if key not in written]
    extra = [key for key in written if key not in source]

    report = {
        "target": str(TARGET),
        "frozen_sections_identical": not diffs,
        "frozen_section_diffs": diffs,
        "sections_dropped_from_source": leftovers,
        "sections_added": extra,
        "feature_registry_present_in_target": "feature_registry" in written,
        "families": sorted(written["feature_schemes"]["model_families"]),
        "candidate_counts": {
            name: len(family["candidates"])
            for name, family in written["feature_schemes"]["model_families"].items()
        },
    }
    if diffs:
        raise AssertionError(f"frozen sections drifted: {json.dumps(diffs, ensure_ascii=False)}")
    if leftovers != ["feature_registry"]:
        raise AssertionError(f"expected only feature_registry to be dropped; got {leftovers}")
    if report["feature_registry_present_in_target"]:
        raise AssertionError("feature_registry must be absent so the compatibility path is used")
    for name, count in report["candidate_counts"].items():
        if count != 1:
            raise AssertionError(
                f"family {name} must declare exactly one candidate so both models are fitted "
                f"independently; got {count}"
            )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

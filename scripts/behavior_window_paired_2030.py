"""Paired 20 s versus 30 s Behavior window validation on identical probes.

This is a post-freeze supplementary analysis. It never changes the frozen 30 s
main analysis or the historical 1.16.25 window-sensitivity outputs.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import yaml

from attention_pipeline.supervised_learning.evaluation import paired_log_loss_increment
from attention_pipeline.supervised_learning.feature_schemes import FeatureScheme
from attention_pipeline.supervised_learning.reporting import write_supervised_run
from attention_pipeline.supervised_learning.runner import run_nested_loso


FEATURES = (
    "go_correct_rt_median_ms",
    "go_correct_rt_cv",
    "go_correct_rt_theilsen_slope_ms_per_s",
    "raw_go_omission_rate",
    "commission_rate",
)
KEY = ("participant_group_id", "session_id", "block_id", "probe_event_id")
ANALYSIS_SET_ID = "AS.behavior_reference__paired_window20s_30s"
MEMBERSHIP = "included_missing_aware"
RUN_IDS = {20: "behavior_window20s__paired_2030", 30: "behavior_window30s__paired_2030"}
MODEL_IDS = {20: "behavior_window20s", 30: "behavior_window30s"}


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["block_id"] = out["block_id"].astype(str).str.lower()
    for column in KEY:
        if out[column].isna().any() or out[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"missing/blank probe key: {column}")
        out[column] = out[column].astype(str)
    if out.duplicated(list(KEY)).any():
        raise ValueError("duplicate participant/session/block/probe key")
    return out.sort_values(list(KEY), kind="stable").reset_index(drop=True)


def _finite(frame: pd.DataFrame) -> pd.Series:
    values = frame[list(FEATURES)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return pd.Series(np.isfinite(values).all(axis=1), index=frame.index)


def build_inputs(
    window20_path: Path, frozen30_path: Path, sensitivity_path: Path,
    *, expected_probes: int = 2306, expected_participants: int = 61,
    expected_sessions: int = 116,
) -> tuple[dict[int, pd.DataFrame], dict[str, object]]:
    """Validate source, labels, and exact paired membership before any write."""
    inputs = {20: _normalise(pd.read_csv(window20_path, low_memory=False)),
              30: _normalise(pd.read_csv(frozen30_path, low_memory=False))}
    raw = pd.read_csv(sensitivity_path, low_memory=False)
    source: dict[int, pd.DataFrame] = {}
    for window in (20, 30):
        slice_ = raw.loc[pd.to_numeric(raw["window_seconds_nominal"], errors="coerce").eq(window)]
        source[window] = _normalise(slice_.copy())
        source[window] = source[window].loc[_finite(source[window])].reset_index(drop=True)
        expected = inputs[window]
        observed = source[window]
        if len(expected) != len(observed) or not expected[list(KEY)].equals(observed[list(KEY)]):
            raise ValueError(f"{window} s input membership differs from the finite source slice")
        for column in (*FEATURES, "q1_nominal_4class", "q2_ordinal_4level"):
            left = pd.to_numeric(expected[column], errors="coerce").to_numpy(dtype=float)
            right = pd.to_numeric(observed[column], errors="coerce").to_numpy(dtype=float)
            if not np.allclose(left, right, rtol=0, atol=1e-9, equal_nan=True):
                raise ValueError(f"{window} s source mismatch: {column}")
        if not _finite(expected).all():
            raise ValueError(f"{window} s input contains nonfinite frozen predictors")
    twenty, thirty = inputs[20], inputs[30]
    paired = twenty[list(KEY)].merge(thirty[list(KEY)], on=list(KEY), how="left", indicator=True)
    if not paired["_merge"].eq("both").all():
        raise ValueError("20 s input is not a subset of the frozen 30 s input")
    thirty = thirty.merge(twenty[list(KEY)], on=list(KEY), how="inner", validate="one_to_one")
    thirty = _normalise(thirty)
    if not twenty[list(KEY)].equals(thirty[list(KEY)]):
        raise ValueError("paired inputs do not have identical ordered probe keys")
    for column in ("q1_nominal_4class", "q2_ordinal_4level"):
        if not twenty[column].equals(thirty[column]):
            raise ValueError(f"paired labels/context differ: {column}")
    if twenty["participant_group_id"].nunique() != expected_participants or len(twenty) != expected_probes:
        raise ValueError("paired probe/participant denominator drift")
    if twenty["session_id"].nunique() != expected_sessions:
        raise ValueError("paired session denominator drift")
    for window, frame in ((20, twenty), (30, thirty)):
        frame["analysis_set_id"] = ANALYSIS_SET_ID
        frame["membership_type"] = MEMBERSHIP
        inputs[window] = frame
    audit = {
        "common_probes": len(twenty),
        "participants": int(twenty["participant_group_id"].nunique()),
        "sessions": int(twenty["session_id"].nunique()),
        "window20_finite_probes": len(source[20]),
        "window30_finite_probes": len(source[30]),
        "window30_excluded_from_pair": len(source[30]) - len(thirty),
        "same_keys_and_labels": True,
        "source_feature_tolerance": 1e-9,
        "input_sha256": {"window20": _hash(window20_path), "frozen30": _hash(frozen30_path),
                         "sensitivity": _hash(sensitivity_path)},
    }
    return inputs, audit


def _config(path: Path) -> tuple[dict[str, object], dict[str, object]]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config["task"]["name"] != "q1_equals_1_vs_2_3_4":
        raise ValueError("binary Q1 task contract changed")
    if config["validation"]["outer"]["group_column"] != "participant_group_id":
        raise ValueError("outer grouping contract changed")
    if config["validation"]["inner"]["n_splits"] != 5:
        raise ValueError("inner grouped-fold contract changed")
    if config["models"]["primary"]["C_candidates"] != [0.01, 0.1, 1.0, 10.0]:
        raise ValueError("regularization grid changed")
    feature_map = {f["feature_id"]: f for f in config["feature_registry"]["features"]}
    frozen_ids = (
        "behavior.rt_level.median.v1", "behavior.rt_variability.cv.v1",
        "behavior.rt_trend.theilsen.v1", "behavior.go_omission.raw.v1",
        "behavior.nogo_commission.raw.v1",
    )
    if tuple(feature_map[name]["columns"][0] for name in frozen_ids) != FEATURES:
        raise ValueError("frozen Behavior feature registry changed")
    params = {
        "group_col": "participant_group_id",
        "c_candidates": config["models"]["primary"]["C_candidates"],
        "inner_splits": 5,
        "max_iter": int(config["models"]["primary"]["max_iter"]),
        "seed": int(config["pipeline"]["random_seed"]),
    }
    return config, params


def _git_sha(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def run(
    window20_path: Path, frozen30_path: Path, sensitivity_path: Path,
    config_path: Path, output_root: Path,
) -> dict[str, object]:
    if output_root.exists():
        raise FileExistsError(f"paired output root already exists: {output_root}")
    inputs, input_audit = build_inputs(window20_path, frozen30_path, sensitivity_path)
    _, params = _config(config_path)
    repo = Path(__file__).resolve().parents[1]
    provenance = {"code_sha": _git_sha(repo), "script_sha256": _hash(Path(__file__)),
                  "config_path": str(config_path.resolve()), "config_sha256": _hash(config_path),
                  **input_audit["input_sha256"]}
    scheme = FeatureScheme(feature_set_id="behavior_reference_5", columns=FEATURES,
                           modalities=("behavior",), description="frozen five-feature Behavior reference")
    output_root.mkdir(parents=True, exist_ok=False)
    input_dir = output_root / "inputs"
    input_dir.mkdir()
    for window in (20, 30):
        frame = inputs[window]
        frame.to_csv(input_dir / f"window{window}s_common.csv", index=False, encoding="utf-8-sig")
        result = run_nested_loso(
            frame, model_feature_schemes={MODEL_IDS[window]: (scheme,)},
            run_id=RUN_IDS[window], analysis_set_id=ANALYSIS_SET_ID,
            membership_type=MEMBERSHIP, **params,
        )
        if not result.failures.empty or len(result.predictions) != len(frame):
            raise RuntimeError(f"{window} s run has failed/incomplete outer folds")
        write_supervised_run(result, output_root=output_root,
                             provenance={**provenance, "window_seconds": window,
                                         "paired_input_sha256": _hash(input_dir / f"window{window}s_common.csv")})
    prediction_paths = {window: output_root / RUN_IDS[window] / "probe_predictions.csv" for window in (20, 30)}
    predictions = {window: pd.read_csv(path) for window, path in prediction_paths.items()}
    comparison = paired_log_loss_increment(predictions[30], predictions[20],
                                           replicates=1000, seed=20260830, confidence_level=0.95)
    comparison.participant_increments.to_csv(output_root / "paired_participant_delta.csv", index=False,
                                             encoding="utf-8-sig")
    summary = {
        "status": "complete", "analysis_role": "postfreeze_supplementary_paired_window_validation",
        "analysis_set_id": ANALYSIS_SET_ID, "input_audit": input_audit,
        "provenance": provenance, "model_parameters": params,
        "delta_definition": "participant_macro_log_loss_30s_minus_20s",
        "delta_log_loss": comparison.overall_increment,
        "delta_ci_lower_95": comparison.bootstrap["ci_lower"],
        "delta_ci_upper_95": comparison.bootstrap["ci_upper"],
        "participant_cluster_bootstrap": comparison.bootstrap,
        "window_scores": {
            str(window): json.loads(pd.read_csv(output_root / RUN_IDS[window] / "model_evaluation.csv")
                                    .iloc[0].to_json(double_precision=15))
            for window in (20, 30)
        },
        "outputs": {
            "window20_run": RUN_IDS[20], "window30_run": RUN_IDS[30],
            "window20_input": "inputs/window20s_common.csv",
            "window30_input": "inputs/window30s_common.csv",
            "paired_participant_delta": "paired_participant_delta.csv",
        },
    }
    (output_root / "paired_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window20-input", required=True, type=Path)
    parser.add_argument("--frozen30-input", required=True, type=Path)
    parser.add_argument("--sensitivity-source", required=True, type=Path)
    parser.add_argument("--config", default="configs/supervised_learning_v1.yaml", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    summary = run(args.window20_input, args.frozen30_input, args.sensitivity_source,
                  args.config, args.output_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Independently verify a completed paired 20 s / 30 s Behavior window run."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEY = ["participant_group_id", "session_id", "block_id", "probe_event_id"]
RUN_IDS = {20: "behavior_window20s__paired_2030", 30: "behavior_window30s__paired_2030"}
PROBABILITY = "p_q1_equals_1"


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _loss(frame: pd.DataFrame) -> np.ndarray:
    y = pd.to_numeric(frame["q1_binary"], errors="raise").to_numpy(dtype=int)
    p = pd.to_numeric(frame[PROBABILITY], errors="raise").to_numpy(dtype=float)
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("invalid Q1 labels or probabilities")
    p = np.clip(p, np.finfo(float).eps, 1 - np.finfo(float).eps)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def verify(root: Path) -> dict[str, object]:
    summary = json.loads((root / "paired_summary.json").read_text(encoding="utf-8"))
    predictions: dict[int, pd.DataFrame] = {}
    audits: dict[int, list[dict[str, object]]] = {}
    manifest: dict[int, dict[str, object]] = {}
    for window in (20, 30):
        run_dir = root / RUN_IDS[window]
        frame = pd.read_csv(run_dir / "probe_predictions.csv")
        frame = frame.sort_values(KEY, kind="stable").reset_index(drop=True)
        if len(frame) != 2306 or frame.duplicated(KEY).any() or frame["participant_group_id"].nunique() != 61:
            raise ValueError(f"{window} s prediction denominator or key failure")
        if frame["model_failed"].astype(str).str.lower().ne("false").any():
            raise ValueError(f"{window} s has failed prediction rows")
        if not frame["participant_group_id"].astype(str).equals(frame["outer_fold_group"].astype(str)):
            raise ValueError(f"{window} s outer-fold identity mismatch")
        if not pd.read_csv(run_dir / "failures.csv").empty:
            raise ValueError(f"{window} s failure table is not empty")
        manifest[window] = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        if manifest[window]["status"] != "complete" or manifest[window]["n_failed_folds"] != 0:
            raise ValueError(f"{window} s manifest is not complete")
        if (manifest[window]["inner_splits"] != 5
                or manifest[window]["c_candidates"] != [0.01, 0.1, 1.0, 10.0]
                or manifest[window]["provenance"]["window_seconds"] != window):
            raise ValueError(f"{window} s training configuration/provenance drift")
        audits[window] = json.loads((run_dir / "fold_audits.json").read_text(encoding="utf-8"))
        if len(audits[window]) != 61:
            raise ValueError(f"{window} s outer fold count is not 61")
        seen: set[str] = set()
        for audit in audits[window]:
            held_out = str(audit["outer_fold_group"])
            train = set(map(str, audit["outer_train_group_ids"]))
            test = set(map(str, audit["outer_test_group_ids"]))
            if audit["failed"] or held_out in seen or held_out in train or test != {held_out}:
                raise ValueError(f"{window} s outer participant leakage/failure: {held_out}")
            seen.add(held_out)
            for inner in audit["selection"]["inner_fold_audits"]:
                inner_train = set(map(str, inner["train_group_ids"]))
                inner_valid = set(map(str, inner["validation_group_ids"]))
                if inner_train & inner_valid or held_out in inner_train | inner_valid:
                    raise ValueError(f"{window} s inner participant leakage: {held_out}")
        if seen != set(frame["participant_group_id"].astype(str)):
            raise ValueError(f"{window} s missing outer participant folds")
        input_path = root / "inputs" / f"window{window}s_common.csv"
        if _hash(input_path) != manifest[window]["provenance"]["paired_input_sha256"]:
            raise ValueError(f"{window} s input hash mismatch")
        input_frame = pd.read_csv(input_path).sort_values(KEY, kind="stable").reset_index(drop=True)
        if not input_frame[KEY].equals(frame[KEY]):
            raise ValueError(f"{window} s input and prediction keys differ")
        if not np.allclose(pd.to_numeric(input_frame["q1_nominal_4class"]),
                           pd.to_numeric(frame["q1_nominal_4class"]), rtol=0, atol=0):
            raise ValueError(f"{window} s input and prediction labels differ")
        predictions[window] = frame
    a, b = predictions[20], predictions[30]
    if manifest[20]["provenance"]["config_sha256"] != manifest[30]["provenance"]["config_sha256"]:
        raise ValueError("window arms used different model configurations")
    if not a[KEY + ["q1_binary", "outer_fold_group"]].equals(
        b[KEY + ["q1_binary", "outer_fold_group"]]
    ):
        raise ValueError("paired OOF archives have unequal keys, labels or outer folds")
    for left, right in zip(audits[20], audits[30], strict=True):
        if left["outer_fold_group"] != right["outer_fold_group"]:
            raise ValueError("outer fold order differs")
        x, y = left["selection"]["inner_fold_audits"], right["selection"]["inner_fold_audits"]
        if [(v["train_group_ids"], v["validation_group_ids"]) for v in x] != [
            (v["train_group_ids"], v["validation_group_ids"]) for v in y
        ]:
            raise ValueError("inner participant partitions differ between windows")
    frame = pd.DataFrame({"participant_group_id": a["participant_group_id"],
                          "loss20": _loss(a), "loss30": _loss(b)})
    participant = frame.groupby("participant_group_id", sort=True).agg(
        n_probes=("loss20", "size"), loss20=("loss20", "mean"), loss30=("loss30", "mean")
    )
    participant["delta"] = participant["loss30"] - participant["loss20"]
    delta = participant["delta"].to_numpy(dtype=float)
    rng = np.random.default_rng(20260830)
    draws = np.array([np.mean(delta[rng.integers(0, len(delta), size=len(delta))])
                      for _ in range(1000)])
    low, high = np.quantile(draws, [0.025, 0.975])
    recorded = pd.read_csv(root / "paired_participant_delta.csv")
    if len(recorded) != 61 or not np.allclose(
        recorded.sort_values("participant_group_id")["mean_log_loss_increment"].to_numpy(dtype=float),
        delta, rtol=0, atol=1e-12,
    ):
        raise ValueError("participant-level paired difference mismatch")
    for observed, expected, name in (
        (summary["delta_log_loss"], delta.mean(), "point estimate"),
        (summary["delta_ci_lower_95"], low, "lower CI"),
        (summary["delta_ci_upper_95"], high, "upper CI"),
        (summary["window_scores"]["20"]["participant_equal_log_loss"], participant["loss20"].mean(), "20 s loss"),
        (summary["window_scores"]["30"]["participant_equal_log_loss"], participant["loss30"].mean(), "30 s loss"),
    ):
        if not np.isclose(float(observed), float(expected), rtol=0, atol=1e-12):
            raise ValueError(f"independent recomputation mismatch: {name}")
    return {
        "status": "PASS", "n_probes_per_window": 2306, "n_participants": 61,
        "n_outer_folds_per_window": 61, "n_failed_folds": 0,
        "same_probe_label_and_outer_fold": True, "same_inner_partitions": True,
        "independent_delta_log_loss": float(delta.mean()),
        "independent_ci_lower_95": float(low), "independent_ci_upper_95": float(high),
        "window20_log_loss": float(participant["loss20"].mean()),
        "window30_log_loss": float(participant["loss30"].mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args()
    result = verify(args.run_root)
    (args.run_root / "paired_verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

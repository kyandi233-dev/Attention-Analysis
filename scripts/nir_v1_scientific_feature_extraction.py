"""Extract the minimal, auditable NIR v1 scientific feature layer.

This script consumes the already frozen 71-session cohort table.  It does not
rediscover sessions, change coverage tiers, or run the NIR model.  The primary
signal is the full-class pupil-to-iris diameter ratio (PIR); absolute pupil
diameter is retained only for the requested correlation audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


PIR = "fullclass_pupil_to_iris_diameter_ratio"
PIR_VALID = "fullclass_normalization_valid"
OLD_DIAMETER = "pupil_equiv_diameter"
OAR = "fullclass_ocular_aperture_ratio_median"
WINDOWS = {"pre_10s": 10.0, "pre_20s": 20.0, "pre_30s": 30.0}
NUMERIC_FEATURES = ["pir_median", "pir_mad", "pir_robust_slope_per_s", "pir_valid_fraction"]


def normalize_subject(value) -> str:
    s = str(value).strip()
    if s.startswith("sub-"):
        return s
    m = re.search(r"(\d+)", s)
    if not m:
        raise ValueError(f"Cannot normalize subject: {value!r}")
    return f"sub-{int(m.group(1)):03d}"


def finite(values):
    a = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return a[np.isfinite(a)]


def mad(values) -> float:
    a = finite(values)
    return float(np.median(np.abs(a - np.median(a)))) if len(a) else np.nan


def robust_slope(times_ms, values) -> float:
    t = pd.to_numeric(pd.Series(times_ms), errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    if len(y) < 2 or np.ptp(t) <= 0:
        return np.nan
    # First reduce to at most eight time bins, matching the existing
    # robust_binned_slope convention and keeping the dry-run tractable.
    edges = np.linspace(t.min(), t.max(), min(8, len(t)) + 1)
    bt, by = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (t >= lo) & ((t < hi) if hi < edges[-1] else (t <= hi))
        if sel.any():
            bt.append(float(np.median(t[sel])))
            by.append(float(np.median(y[sel])))
    t, y = np.asarray(bt), np.asarray(by)
    if len(y) < 2 or np.ptp(t) <= 0:
        return np.nan
    # Theil-Sen slope is robust to isolated segmentation outliers.
    with np.errstate(divide="ignore", invalid="ignore"):
        slopes = (y[None, :] - y[:, None]) / (t[None, :] - t[:, None]) * 1000.0
    # Zero is a valid plateau slope; only remove non-finite values and the
    # zero-denominator diagonal.
    slopes = slopes[np.isfinite(slopes)]
    return float(np.median(slopes)) if len(slopes) else 0.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def behavior_probes(subject: str, data_root: Path) -> pd.DataFrame:
    rows = []
    for block in (1, 2):
        pattern = data_root / f"{subject}_" / "beh" / f"{subject}_Block{block}_B_beh.csv"
        files = [pattern] if pattern.exists() else list((data_root / f"{subject}_" / "beh").glob(f"{subject}_Block{block}_*_beh.csv"))
        if not files:
            raise FileNotFoundError(f"Behavior file not found for {subject} block {block}")
        d = pd.read_csv(files[0])
        d["absolute_onset_time"] = pd.to_numeric(d["absolute_onset_time"], errors="coerce")
        probes = d[(pd.to_numeric(d["is_probe"], errors="coerce") == 1) & d["absolute_onset_time"].notna()].copy()
        probes = probes.sort_values("absolute_onset_time")
        probes["block_num"] = block
        probes["probe_order_in_block"] = np.arange(1, len(probes) + 1)
        probes["probe_order"] = (block - 1) * 10 + probes["probe_order_in_block"]
        probes["probe_onset_ms"] = probes["absolute_onset_time"].round().astype("int64")
        probes["block_start_ms"] = int(d["absolute_onset_time"].min())
        # Keep the established formal-task 1.15 s trial tail for the block bound.
        probes["block_end_ms"] = int(d["absolute_onset_time"].max() + 1150)
        probes["time_on_task_sec"] = (probes["absolute_onset_time"] - probes["block_start_ms"]) / 1000.0
        probes["source_behavior_file"] = str(files[0])
        rows.append(probes[["block_num", "probe_order_in_block", "probe_order", "probe_onset_ms",
                            "probe_response", "probe_vigilance", "block_start_ms", "block_end_ms",
                            "time_on_task_sec", "source_behavior_file"]])
    out = pd.concat(rows, ignore_index=True)
    out["probe_id"] = out["probe_order"]
    out["previous_probe_onset_ms"] = out["probe_onset_ms"].shift(1)
    out.loc[out["block_num"].ne(out["block_num"].shift(1)), "previous_probe_onset_ms"] = np.nan
    out["seconds_since_previous_probe"] = (out["probe_onset_ms"] - out["previous_probe_onset_ms"]) / 1000.0
    return out


def eye_summary(frame: pd.DataFrame, start: int, end: int, eye: str) -> dict:
    x = frame[(frame["eye"] == eye) & (frame["unix_ms"] >= start) & (frame["unix_ms"] < end)].copy()
    valid = x[PIR_VALID].fillna(False).astype(bool) & pd.to_numeric(x[PIR], errors="coerce").notna()
    y = pd.to_numeric(x.loc[valid, PIR], errors="coerce")
    old = pd.to_numeric(x.loc[valid, OLD_DIAMETER], errors="coerce")
    out = {"source_rows": int(len(x)), "pir_valid_rows": int(len(y)),
           "pir_valid_fraction": float(len(y) / len(x)) if len(x) else np.nan,
           "roi_clipped_rows": int(x["roi_clipped"].fillna(False).astype(bool).sum()) if len(x) else 0,
           "ritnet_found_rows": int(x["ritnet_found"].fillna(False).astype(bool).sum()) if len(x) else 0,
           "pir_invalid_rows": int(len(x) - len(y))}
    out.update({"pir_median": float(y.median()) if len(y) else np.nan,
                "pir_mad": mad(y), "pir_robust_slope_per_s": robust_slope(x.loc[valid, "unix_ms"], y),
                "old_diameter_median": float(old.median()) if len(old) else np.nan,
                "oar_median": float(pd.to_numeric(x.loc[valid, OAR], errors="coerce").median()) if len(y) else np.nan})
    return out


def run(args):
    cohort_path = Path(args.cohort_input)
    nir_root, data_root, out_root = map(Path, (args.nir_root, args.data_root, args.output_root))
    out_root.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(cohort_path)
    cohort["subject"] = cohort["subject"].map(normalize_subject)
    cohort["probe_id"] = cohort["probe_id"].astype(int)
    cohort["coverage_tier"] = np.select([cohort["nir_usable_frame_rate"] >= 0.80, cohort["nir_usable_frame_rate"] >= 0.50], ["primary", "sensitivity-only"], default="excluded")
    expected_tiers = cohort["coverage_tier"].value_counts().to_dict()
    if expected_tiers != {"primary": 1174, "excluded": 208, "sensitivity-only": 38}:
        raise ValueError(f"Frozen cohort tier counts changed: {expected_tiers}")
    cohort_key = cohort[["subject", "probe_id", "nir_usable_frame_rate"]].drop_duplicates()
    all_rows, source_paths = [], []
    for subject in sorted(cohort["subject"].unique()):
        probes = behavior_probes(subject, data_root)
        run_dir = nir_root / f"{subject}_formal_v3.1.3_yolo8_b16_fp32"
        files = list(run_dir.glob(f"{subject}_ritnet_fullclass_v1-2-fast-qc.csv"))
        if not files:
            raise FileNotFoundError(f"fullclass output not found: {subject}")
        fullclass_path = files[0]
        source_paths.append(str(fullclass_path))
        fc = pd.read_csv(fullclass_path)
        fc["unix_ms"] = pd.to_numeric(fc["unix_ms"], errors="coerce")
        fc = fc[fc["phase"].isin(["block1", "block2"])].copy()
        fc["block_num"] = fc["phase"].map({"block1": 1, "block2": 2})
        for _, p in probes.iterrows():
            if not ((cohort_key["subject"] == subject) & (cohort_key["probe_id"] == p.probe_id)).any():
                continue
            block_fc = fc[fc["block_num"] == p.block_num]
            for window, duration in WINDOWS.items():
                requested_start = int(p.probe_onset_ms - duration * 1000)
                requested_end = int(p.probe_onset_ms)
                start, end = max(requested_start, int(p.block_start_ms)), min(requested_end, int(p.block_end_ms))
                prev = p.previous_probe_onset_ms
                crosses = bool(pd.notna(prev) and requested_start <= prev < requested_end)
                summaries = {eye: eye_summary(block_fc, start, end, eye) for eye in ("frame_left", "frame_right")}
                available_fraction = max(0.0, (end - start) / (requested_end - requested_start))
                base = {"subject": subject, "probe_id": int(p.probe_id), "block": int(p.block_num),
                        "probe_order_in_block": int(p.probe_order_in_block), "probe_order": int(p.probe_order),
                        "probe_onset_ms": int(p.probe_onset_ms), "time_on_task_sec": float(p.time_on_task_sec),
                        "probe_response_raw": p.probe_response, "probe_vigilance_raw": p.probe_vigilance,
                        "window": window, "window_seconds": duration, "requested_start_ms": requested_start,
                        "requested_end_ms": requested_end, "available_start_ms": start, "available_end_ms": end,
                        "window_truncated_by_block_start": requested_start < p.block_start_ms,
                        "window_truncated_by_block_end": requested_end > p.block_end_ms,
                        "boundary_truncated": requested_start < p.block_start_ms or requested_end > p.block_end_ms,
                        "available_duration_fraction": available_fraction,
                        "previous_probe_onset_ms": prev, "seconds_since_previous_probe": p.seconds_since_previous_probe,
                        "window_crosses_previous_probe": crosses,
                        "seconds_of_window_before_previous_probe": float((prev - requested_start) / 1000.0) if crosses else np.nan,
                        "nir_quality_tier": cohort.loc[(cohort.subject == subject) & (cohort.probe_id == p.probe_id), "coverage_tier"].iloc[0],
                        "cohort_30s_usable_frame_rate": cohort_key.loc[(cohort_key.subject == subject) & (cohort_key.probe_id == p.probe_id), "nir_usable_frame_rate"].iloc[0]}
                for eye, s in summaries.items():
                    prefix = "left" if eye == "frame_left" else "right"
                    for k, v in s.items(): base[f"{prefix}_{k}"] = v
                available = [summaries[e] for e in summaries if np.isfinite(summaries[e]["pir_median"])]
                for k in NUMERIC_FEATURES + ["old_diameter_median", "oar_median"]:
                    vals = [s[k] for s in available if np.isfinite(s[k])]
                    base[f"pir_fused_{k}" if k.startswith("pir_") else f"fused_{k}"] = float(np.median(vals)) if vals else np.nan
                base["n_eyes_with_pir"] = len(available)
                base["eyes_both_available"] = len(available) == 2
                base["pir_feature_available"] = len(available) > 0
                all_rows.append(base)
    result = pd.DataFrame(all_rows)
    for window in WINDOWS:
        mask = result["window"].eq(window) & result["pir_feature_available"]
        means = result.loc[mask].groupby("subject")["pir_fused_pir_median"].transform("mean")
        result.loc[mask, "pir_subject_mean"] = means
        result.loc[mask, "pir_within_subject_deviation"] = result.loc[mask, "pir_fused_pir_median"].to_numpy() - means.to_numpy()
    result.to_csv(out_root / "nir_v1_probe_features.csv", index=False, encoding="utf-8-sig")
    result[result.window.eq("pre_30s")].groupby("subject").size().rename("probe_rows_30s").reset_index().to_csv(out_root / "subject_summary.csv", index=False, encoding="utf-8-sig")
    audit = {"cohort_input": str(cohort_path), "cohort_rows": int(len(cohort)), "cohort_sessions": int(cohort.subject.nunique()),
             "windows_seconds": [10, 20, 30], "feature_primary": "PIR", "primary_variability": "MAD",
             "fusion": "median of available eye-level summaries; left/right columns retained",
             "coverage_tier_counts": expected_tiers,
             "primary_probes": int((cohort["coverage_tier"] == "primary").sum()),
             "pir_available_by_window": {w: int(result.loc[result.window.eq(w) & result.pir_feature_available].shape[0]) for w in WINDOWS},
             "both_eye_by_window": {w: int(result.loc[result.window.eq(w) & result.eyes_both_available].shape[0]) for w in WINDOWS},
             "boundary_truncation_by_window": {w: int(result.loc[result.window.eq(w) & result.boundary_truncated].shape[0]) for w in WINDOWS},
             "previous_probe_crossing_by_window": {w: int(result.loc[result.window.eq(w) & result.window_crosses_previous_probe].shape[0]) for w in WINDOWS},
             "pir_distribution_30s": distribution(result[result.window.eq("pre_30s")]["pir_fused_pir_median"]),
             "eye_validity_30s": {"left_any_valid": int(result.loc[result.window.eq("pre_30s"), "left_pir_valid_fraction"].gt(0).sum()), "right_any_valid": int(result.loc[result.window.eq("pre_30s"), "right_pir_valid_fraction"].gt(0).sum())},
             "subject_probe_counts": result[result.window.eq("pre_30s")].groupby("subject").size().to_dict(),
             "subject_imbalance": {"min": int(result[result.window.eq("pre_30s")].groupby("subject").size().min()), "max": int(result[result.window.eq("pre_30s")].groupby("subject").size().max()), "subjects_not_20": int((result[result.window.eq("pre_30s")].groupby("subject").size() != 20).sum())},
             "rows": int(len(result)), "source_fullclass_files": len(source_paths),
             "probe_semantics_status": "verified_from v3.1.3 final source images; raw fields retained",
             "probe_response_mapping": {1: "完全专注于分拣任务", 2: "关注实验本身，但没有聚焦于分拣任务", 3: "在想与实验无关的事情", 4: "大脑空白，没有明确想法"},
             "probe_vigilance_mapping": {1: "非常困倦", 2: "比较困倦", 3: "比较清醒", 4: "非常清醒"},
             "old_diameter_correlation": correlation(result),
             "source_sha256": {str(cohort_path): sha256(cohort_path)}}
    audit["source_git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    (out_root / "audit_summary.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    manifest = {"command": " ".join(args.command), "inputs": {str(cohort_path): sha256(cohort_path)}, "outputs": {p.name: sha256(p) for p in out_root.iterdir() if p.is_file()}}
    (out_root / "provenance_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(out_root, audit, result)


def correlation(result: pd.DataFrame) -> dict:
    out = {}
    for w, d in result.groupby("window"):
        x = pd.to_numeric(d["pir_fused_pir_median"], errors="coerce")
        y = pd.to_numeric(d["fused_old_diameter_median"], errors="coerce")
        ok = x.notna() & y.notna()
        out[w] = {"n": int(ok.sum()), "pearson_r": float(x[ok].corr(y[ok])) if ok.sum() >= 2 else None,
                  "spearman_r": float(x[ok].corr(y[ok], method="spearman")) if ok.sum() >= 2 else None}
    return out


def distribution(series: pd.Series) -> dict:
    x = pd.to_numeric(series, errors="coerce").dropna()
    if not len(x):
        return {"n": 0}
    q = x.quantile([.01, .05, .25, .5, .75, .95, .99])
    return {"n": int(len(x)), "min": float(x.min()), "max": float(x.max()), "mean": float(x.mean()), "sd": float(x.std()), **{f"q{int(k*100):02d}": float(v) for k, v in q.items()}, "outside_0_to_2": int(((x < 0) | (x > 2)).sum())}


def write_report(root: Path, audit: dict, result: pd.DataFrame):
    lines = ["# NIR v1 minimal scientific feature dry-run audit", "", "本报告只新增 PIR feature/QC layer，不修改既有 71-session cohort、coverage tier、NIR runtime 或正式统计。", "",
             "## Frozen cohort", "", f"输入行数 {audit['cohort_rows']}，session 数 {audit['cohort_sessions']}。coverage 分层直接继承输入表：primary 1174，sensitivity-only 38，excluded 208。", "",
             "## Features", "", "正式 primary pupil measure 为 full-class PIR。每只眼保留 PIR median、MAD、robust slope、valid fraction；fused 值是可用左右眼 eye-level summary 的中位数，不进行 frame-level pooling。left/right provenance 保留在同一行的独立字段中。旧 pupil_equiv_diameter 仅用于相关性审计。", "",
             "Windows: 10 s and 20 s sensitivity, 30 s primary. Each window is clipped to its current block; truncation and previous-probe crossing fields are retained.", "",
             "## Audit", "", f"PIR feature available rows: " + "; ".join(f"{w}={audit['pir_available_by_window'][w]}" for w in WINDOWS),
             f"Both-eye available rows: " + "; ".join(f"{w}={audit['both_eye_by_window'][w]}" for w in WINDOWS),
             f"30 s left/right any-valid probes: {audit['eye_validity_30s']['left_any_valid']}/{audit['eye_validity_30s']['right_any_valid']}; subject probe count range: {audit['subject_imbalance']['min']}–{audit['subject_imbalance']['max']}; subjects not equal to 20: {audit['subject_imbalance']['subjects_not_20']}",
             f"30 s PIR distribution: {json.dumps(audit['pir_distribution_30s'], ensure_ascii=False)}",
             f"Boundary truncation: " + "; ".join(f"{w}={audit['boundary_truncation_by_window'][w]}" for w in WINDOWS),
             f"Previous-probe crossing: " + "; ".join(f"{w}={audit['previous_probe_crossing_by_window'][w]}" for w in WINDOWS), "",
             "RITnet failure, ROI clipping, segmentation failure and PIR invalidity are QC/missingness only; no field is named or interpreted as blink/PERCLOS. OAR remains secondary exploratory only.", "",
             "## Behavior source semantics", "", "v3.1.3 final task source uses probe_response for the four attention-state options and probe_vigilance for the four vigilance options. Exact option text was verified from the final source PNGs and is recorded in audit_summary.json; raw numeric columns remain in the feature table.", "",
             "## Subject-aware and leakage rule", "", "Output includes subject, block, time_on_task_sec and probe order. subject_mean and within_subject_deviation are descriptive full-cohort fields. For any future machine-learning cross-validation, centering parameters must be calculated inside the training fold and applied to held-out subjects; these dry-run columns must not be reused as precomputed ML parameters.", "",
             "## Cohort and statistics boundary", "", "This report does not change cohort membership or formal statistical conclusions. It is not approval to start formal statistics until scientific review of this feature/QC layer is complete.", "",
             "## Reproducibility", "", "Inputs, command, row counts, source commit and SHA-256 values are recorded in provenance_manifest.json and audit_summary.json. Raw/private subject data are not copied into the repository."]
    (root / "NIR_V1_MINIMAL_SCIENTIFIC_FIX_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort-input", required=True)
    ap.add_argument("--nir-root", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--output-root", required=True)
    args = ap.parse_args()
    args.command = __import__("sys").argv
    run(args)


if __name__ == "__main__":
    main()

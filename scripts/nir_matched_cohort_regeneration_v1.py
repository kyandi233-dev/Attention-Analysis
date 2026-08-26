"""Regenerate threshold-free matched NIR probe cohorts from completed outputs.

This deterministic utility only reads completed fullclass ``eyes.csv`` files
and the corresponding formal behavior CSVs. It does not run NIR inference,
does not apply a session exclusion rule, and preserves the frozen coverage
tiers: primary >= 0.80, sensitivity-only 0.50-<0.80, excluded < 0.50.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


WINDOW_MS = 30_000
CONFIDENCE_MIN = 0.80


def behavior_probes(data_root: Path, subject: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted((data_root / f"sub-{subject}_" / "beh").glob("sub-*Block*_B_beh.csv")):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("is_probe") == "1" and row.get("probe_onset_time"):
                    rows.append(
                        {
                            "subject": subject,
                            "probe_id": str(int(float(row["trial_num"]))),
                            "block_num": int(float(row["block_num"])),
                            "probe_onset_ms": int(float(row["probe_onset_time"])),
                            "probe_attention_label": row.get("probe_response") or None,
                            "probe_vigilance_label": row.get("probe_vigilance") or None,
                        }
                    )
    result = pd.DataFrame(rows).sort_values(["block_num", "probe_onset_ms"]).reset_index(drop=True)
    result["probe_id"] = (result.index + 1).astype(str)
    return result


def usable_frame_table(eyes_path: Path, subject: str) -> pd.DataFrame:
    eyes = pd.read_csv(eyes_path)
    for column in ("unix_ms", "pupil_confidence", "pupil_equiv_diameter"):
        eyes[column] = pd.to_numeric(eyes[column], errors="coerce")
    usable = (
        eyes.status.eq("observed")
        & eyes.ritnet_found.fillna(False).astype(bool)
        & ~eyes.roi_clipped.fillna(True).astype(bool)
        & eyes.pupil_confidence.ge(CONFIDENCE_MIN)
        & eyes.pupil_equiv_diameter.gt(0)
        & eyes.unix_ms.notna()
    )
    eyes["usable_eye"] = usable
    rows: list[dict[str, object]] = []
    for unix_ms, group in eyes.groupby("unix_ms", sort=True):
        values = group.loc[group.usable_eye, "pupil_equiv_diameter"].dropna().to_numpy(dtype=float)
        rows.append(
            {
                "unix_ms": int(unix_ms),
                "usable_frame": int(values.size > 0),
                "pupil_diameter_frame_median": float(np.median(values)) if values.size else np.nan,
            }
        )
    frame = pd.DataFrame(rows)
    frame["subject"] = subject
    return frame


def slope_per_second(x: np.ndarray, y: np.ndarray) -> float:
    if len(y) < 2 or np.unique(x).size < 2:
        return np.nan
    return float(np.polyfit((x - x.min()) / 1000.0, y, 1)[0])


def build_probe_windows(frames: pd.DataFrame, probes: pd.DataFrame) -> list[dict[str, object]]:
    timestamp = frames.unix_ms.to_numpy(dtype=np.int64)
    usable = frames.usable_frame.to_numpy(dtype=int)
    diameter = frames.pupil_diameter_frame_median.to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    for probe in probes.itertuples(index=False):
        mask = (timestamp >= probe.probe_onset_ms - WINDOW_MS) & (timestamp < probe.probe_onset_ms)
        tv, uv, dv = timestamp[mask], usable[mask], diameter[mask]
        good = (uv == 1) & np.isfinite(dv)
        x, y = tv[good], dv[good]
        rows.append(
            {
                **probe._asdict(),
                "nir_window_requested_s": WINDOW_MS / 1000.0,
                "nir_frame_count": int(mask.sum()),
                "nir_usable_frame_count": int(good.sum()),
                "nir_usable_frame_rate": float(good.mean()) if mask.sum() else np.nan,
                "nir_pupil_diameter_median": float(np.median(y)) if len(y) else np.nan,
                "nir_pupil_diameter_iqr": float(np.quantile(y, 0.75) - np.quantile(y, 0.25)) if len(y) >= 4 else np.nan,
                "nir_pupil_diameter_slope_per_s": slope_per_second(x, y),
            }
        )
    return rows


def quality_tier(rate: float) -> str:
    if not np.isfinite(rate) or rate < 0.50:
        return "exclude_lt50"
    if rate < 0.80:
        return "sensitivity_50_to_lt80"
    return "primary_ge80"


def regenerate(nir_root: Path, data_root: Path, output_root: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    for run in sorted(nir_root.glob("sub-*_formal_v3.1.3_yolo8_b16_fp32")):
        completion = run / "completion.json"
        eyes = run / "eyes.csv"
        if not completion.is_file() or not eyes.is_file():
            continue
        marker = json.loads(completion.read_text(encoding="utf-8"))
        if marker.get("status") != "complete":
            continue
        subject = run.name.split("_")[0].replace("sub-", "")
        probes = behavior_probes(data_root, subject)
        frames = usable_frame_table(eyes, subject)
        features = build_probe_windows(frames, probes)
        rows.extend(features)
        audit.append(
            {
                "subject": subject,
                "probe_count": int(len(probes)),
                "frame_count": int(len(frames)),
                "probe_windows_with_any_usable_frame": int(
                    sum(row["nir_usable_frame_count"] > 0 for row in features)
                ),
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(rows)
    data.to_csv(output_root / "nir_probe_windows_unfiltered.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audit).to_csv(output_root / "nir_probe_window_build_audit.csv", index=False, encoding="utf-8-sig")
    summary = {
        "completed_subjects": int(len(audit)),
        "probe_windows": int(len(data)),
        "quality_rule_at_frame_level": "at least one observed, non-clipped RITnet eye at confidence>=0.80",
        "window_inclusion_threshold": "none; intentionally unfiltered pending user confirmation",
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nir-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(regenerate(args.nir_root, args.data_root, args.output_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

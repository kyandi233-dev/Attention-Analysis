from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def refresh_ocular_coverage(ocular_root: str | Path) -> None:
    root = Path(ocular_root).expanduser().resolve()
    wide_path = root / "tables/ocular_probe_features_wide.csv"
    handoff_path = root / "tables/ocular_feature_handoff.csv"
    coverage_path = root / "tables/ocular_feature_coverage.csv"

    wide = pd.read_csv(wide_path, encoding="utf-8-sig", low_memory=False)
    handoff = pd.read_csv(handoff_path, encoding="utf-8-sig", low_memory=False)
    rows: list[dict[str, object]] = []

    for index, row in handoff.iterrows():
        predictor = str(row["predictor_column"])
        if predictor not in wide.columns:
            continue
        values = pd.to_numeric(wide[predictor], errors="coerce")
        finite = np.isfinite(values)
        summary = {
            "probe_total_n": int(len(wide)),
            "finite_probe_n": int(finite.sum()),
            "finite_fraction": float(finite.mean()) if len(wide) else None,
            "participant_group_n": int(
                wide.loc[finite, "participant_group_id"].dropna().astype(str).nunique()
            ),
            "session_n": int(wide.loc[finite, "session_id"].dropna().astype(str).nunique()),
        }
        handoff.at[index, "coverage_summary"] = json.dumps(
            summary, ensure_ascii=False, sort_keys=True
        )
        handoff.at[index, "estimability_status"] = (
            "estimable_frozen_rule" if summary["finite_probe_n"] else "not_estimable"
        )
        rows.append(
            {
                "predictor_column": predictor,
                "report_role": row["report_role"],
                **summary,
            }
        )

    handoff.to_csv(handoff_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(coverage_path, index=False, encoding="utf-8-sig")

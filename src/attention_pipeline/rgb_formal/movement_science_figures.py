from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PRIMARY = "body_motion_energy_median"
AUXILIARY = {
    "pose_lateral_right_per_sec_median",
    "pose_vertical_up_per_sec_median",
    "pose_radial_proximity_direction_score_median",
}

TABLE_TO_FIGURE = {
    "movement_task_progression.csv": "movement_task_progression.png",
    "movement_q1_models.csv": "movement_q1_relationships.png",
    "movement_q2_models.csv": "movement_q2_relationships.png",
    "movement_behavior_links.csv": "movement_behavior_relationships.png",
}


def _first_existing(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def _feature_mask(frame: pd.DataFrame, wanted: set[str]) -> pd.Series:
    keep = pd.Series(False, index=frame.index)
    for column in frame.columns:
        keep = keep | frame[column].astype("string").str.strip().isin(wanted)
    return keep


def _row_label(frame: pd.DataFrame) -> pd.Series:
    label_cols = [
        name
        for name in (
            "feature",
            "predictor",
            "rgb_feature",
            "term",
            "contrast",
            "outcome",
            "behavior_metric",
            "level",
        )
        if name in frame.columns
    ]
    if not label_cols:
        return pd.Series([f"row_{i + 1}" for i in range(len(frame))], index=frame.index)
    labels = []
    for _, row in frame[label_cols].iterrows():
        parts = []
        for name in label_cols:
            value = row[name]
            if pd.notna(value):
                text = str(value).strip()
                if text and text not in parts:
                    parts.append(text)
        labels.append(" | ".join(parts) if parts else "model term")
    return pd.Series(labels, index=frame.index)


def _coefficient_plot(frame: pd.DataFrame, path: Path, title: str) -> bool:
    import matplotlib.pyplot as plt

    estimate_col = _first_existing(frame, ("estimate", "coef", "coefficient", "beta", "B"))
    if estimate_col is None:
        return False
    estimates = pd.to_numeric(frame[estimate_col], errors="coerce")
    mask = np.isfinite(estimates)
    data = frame.loc[mask].copy()
    estimates = estimates.loc[mask]
    if data.empty:
        return False

    low_col = _first_existing(data, ("ci_low", "conf_low", "lower_ci", "ci_95_low", "lower"))
    high_col = _first_existing(data, ("ci_high", "conf_high", "upper_ci", "ci_95_high", "upper"))
    labels = _row_label(data)
    if len(data) > 28:
        data = data.iloc[:28].copy()
        estimates = estimates.iloc[:28]
        labels = labels.iloc[:28]

    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(9.2, max(4.8, 0.38 * len(data) + 1.8)))
    if low_col and high_col:
        low = pd.to_numeric(data[low_col], errors="coerce")
        high = pd.to_numeric(data[high_col], errors="coerce")
        valid_ci = np.isfinite(low) & np.isfinite(high)
        xerr = np.vstack(
            [
                np.where(valid_ci, estimates.to_numpy(float) - low.to_numpy(float), 0.0),
                np.where(valid_ci, high.to_numpy(float) - estimates.to_numpy(float), 0.0),
            ]
        )
        ax.errorbar(estimates, y, xerr=xerr, fmt="o", capsize=3)
    else:
        ax.scatter(estimates, y)
    ax.axvline(0.0, linewidth=1)
    ax.set_yticks(y, labels=labels.tolist())
    ax.set_xlabel("Model coefficient")
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def build_movement_science_figures(movement_root: str | Path) -> list[str]:
    root = Path(movement_root).expanduser().resolve()
    tables = root / "tables"
    generated: list[str] = []
    for table_name, figure_name in TABLE_TO_FIGURE.items():
        path = tables / table_name
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        except pd.errors.EmptyDataError:
            continue
        if frame.empty:
            continue
        primary = frame.loc[_feature_mask(frame, {PRIMARY})].copy()
        if _coefficient_plot(primary, root / "figures/main" / figure_name, figure_name.removesuffix(".png").replace("_", " ").title()):
            generated.append(f"figures/main/{figure_name}")

        auxiliary = frame.loc[_feature_mask(frame, AUXILIARY)].copy()
        sensitivity_name = figure_name.replace(".png", "_pose_sensitivity.png")
        if _coefficient_plot(
            auxiliary,
            root / "figures/sensitivity" / sensitivity_name,
            "Pose / radial auxiliary sensitivity",
        ):
            generated.append(f"figures/sensitivity/{sensitivity_name}")
    return generated

"""Real-render regression test for Movement figure typography.

The Formal 1.16.12 contract requires Chinese in-image labels while digits and
Latin text still prefer Times New Roman. Matplotlib does not fall back to later
entries of ``rcParams['font.serif']``, so a "Times New Roman first + CJK fallback"
configuration silently renders every Chinese label as a tofu box while all tests
stay green. These tests therefore render for real and fail on any missing glyph.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def _force_agg() -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)


def _test_frame(rows: int = 48) -> pd.DataFrame:
    rng = np.random.default_rng(20260913)
    return pd.DataFrame(
        {
            "participant_group_id": [f"p{1 + (i % 6):02d}" for i in range(rows)],
            "session_id": [f"sub-{100 + (i % 8):03d}" for i in range(rows)],
            "block_id": ["B1" if i % 2 else "B2" for i in range(rows)],
            "body_motion_energy_median": rng.normal(0.08, 0.02, rows).clip(0.01),
            "exposure_change_abs_median": rng.normal(0.04, 0.01, rows).clip(0.001),
            "pose_lateral_right_per_sec_median": rng.normal(0.0, 0.02, rows),
            "pose_vertical_up_per_sec_median": rng.normal(0.0, 0.02, rows),
            "pose_radial_proximity_direction_score_median": rng.normal(0.0, 0.02, rows),
        }
    )


def _coverage_frame(rows: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "session_id": [f"sub-{100 + i:03d}" for i in range(rows)],
            "body_motion_observable_ratio": rng.uniform(0.7, 0.99, rows),
            "exposure_change_observable_ratio": rng.uniform(0.7, 0.99, rows),
            "pose_shoulders_observable_ratio": rng.uniform(0.6, 0.99, rows),
        }
    )


def _coefficient_frame(rows: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    return pd.DataFrame(
        {
            "term": [f"term_{i}" for i in range(rows)],
            "estimate": rng.normal(0.0, 0.1, rows),
            "ci_low": rng.normal(-0.2, 0.05, rows),
            "ci_high": rng.normal(0.2, 0.05, rows),
        }
    )


def test_contains_cjk_detects_chinese_and_mixed_strings() -> None:
    from attention_pipeline.rgb_formal.movement_science_figures import contains_cjk

    assert contains_cjk("身体动作能量中位数（无量纲）") is True
    assert contains_cjk("参与者均值 ± SEM") is True
    assert contains_cjk("区块内任务时间段（cycle bin）") is True
    assert contains_cjk("探针窗口数") is True
    assert contains_cjk("body_motion_energy_median") is False
    assert contains_cjk("Q1=1 | ref=0 | model term") is False
    assert contains_cjk("0.05 ± 0.01") is False


def test_movement_figures_render_chinese_without_missing_glyphs(tmp_path: Path) -> None:
    """Every shipped figure must render its Chinese labels with real glyphs."""
    _force_agg()
    import pytest

    from attention_pipeline.rgb_formal import movement_science_figures as figures

    probe = _test_frame()
    coverage = _coverage_frame()
    coefficients = _coefficient_frame()

    figs_dir = tmp_path / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    cases = (
        ("primary", lambda p: figures._primary_distribution_plot(probe, p)),
        ("exposure", lambda p: figures._exposure_qc_plot(probe, p)),
        ("pose", lambda p: figures._pose_distribution_plot(probe, p)),
        ("coverage", lambda p: figures._coverage_plot(coverage, p)),
        ("coefficients", lambda p: figures._coefficient_plot(coefficients, p)),
    )

    for name, run in cases:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            generated, reason = run(figs_dir / name)
        assert generated is True, f"{name} figure was not generated: {reason}"
        missing = [str(item.message) for item in caught if "missing from font" in str(item.message)]
        assert missing == [], f"{name} figure rendered missing glyphs: {missing[:3]}"
        assert (figs_dir / f"{name}.png").is_file()
        assert (figs_dir / f"{name}.svg").is_file()


def test_movement_figure_text_objects_receive_explicit_font_chain(tmp_path: Path) -> None:
    """Chinese text runs get a CJK-first chain; latin text stays Times-New-Roman-first."""
    _force_agg()
    import matplotlib.pyplot as plt

    from attention_pipeline.rgb_formal import movement_science_figures as figures

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("身体动作能量中位数（无量纲）")
    ax.set_ylabel("0.5 1.0 1.5")
    ax.plot([0, 1], [0, 1], label="参与者均值 ± SEM")
    ax.legend(frameon=False)

    from matplotlib.text import Text

    figures.apply_text_fonts(fig)
    by_text = {text.get_text(): list(text.get_fontfamily()) for text in fig.findobj(match=Text)}
    plt.close(fig)

    chinese = by_text["身体动作能量中位数（无量纲）"]
    latin = by_text["0.5 1.0 1.5"]
    mixed = by_text["参与者均值 ± SEM"]

    assert chinese, "Chinese label received no explicit font family"
    assert latin, "Latin label received no explicit font family"
    assert figures.contains_cjk(chinese[0]) or "SimSun" in chinese, chinese
    assert latin[0] == "Times New Roman", latin
    assert mixed[0] == chinese[0], (mixed, chinese)

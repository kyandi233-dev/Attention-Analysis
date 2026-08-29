from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

import attention_pipeline.behavior_formal.science_v3_figures as wrapper
from attention_pipeline.behavior_formal.science_v3_metric_figures import (
    FIGURE_FAMILIES,
    generate_complete_metric_figure_pack,
)


def test_legacy_overview_title_calls_are_suppressed(monkeypatch, tmp_path) -> None:
    observed: list[str] = []

    def fake_overview(*args, **kwargs):
        fig, ax = plt.subplots()
        ax.set_title("不应进入图片")
        observed.append(ax.get_title())
        plt.close(fig)
        return []

    monkeypatch.setattr(wrapper, "_generate_behavior_figures", fake_overview)
    monkeypatch.setattr(
        wrapper,
        "generate_complete_metric_figure_pack",
        lambda **kwargs: ([], pd.DataFrame(), pd.DataFrame()),
    )
    wrapper.generate_behavior_figures(pd.DataFrame(), pd.DataFrame(), tmp_path / "figures")
    assert observed == [""]
    assert wrapper.publication_figure_contract()["internal_title_allowed"] is False


def test_every_requested_metric_has_every_required_figure_family_in_coverage_audit(tmp_path) -> None:
    metrics = ["go_correct_rt_median_ms", "omission_rate"]
    files, manifest, audit = generate_complete_metric_figure_pack(
        session=pd.DataFrame(),
        block=pd.DataFrame(),
        cycle=pd.DataFrame(),
        probe=pd.DataFrame(),
        output_dir=tmp_path,
        metrics=metrics,
    )
    assert files == []
    assert manifest.empty
    assert len(audit) == len(metrics) * len(FIGURE_FAMILIES)
    assert set(audit["metric"]) == set(metrics)
    for metric in metrics:
        assert set(audit.loc[audit["metric"].eq(metric), "figure_family"]) == set(FIGURE_FAMILIES)
    assert audit["internal_title_allowed"].eq(False).all()
    assert audit["caption_is_external"].eq(True).all()

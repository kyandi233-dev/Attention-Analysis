from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

import attention_pipeline.behavior_formal.science_v3_metric_figures as figures
from attention_pipeline.formal_analysis.publication_style import (
    FONT_FAMILY,
    configure_publication_style,
    finalize_publication_figure,
)


def test_shared_style_is_times_new_roman_with_frameless_legends_and_external_titles():
    configure_publication_style()
    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2], label="Series")
    ax.set_title("Must remain in the external caption")
    legend = ax.legend()
    finalize_publication_figure(fig)

    assert mpl.rcParams["font.family"] == ["serif"]
    assert mpl.rcParams["font.serif"][0] == FONT_FAMILY
    assert ax.get_title() == ""
    assert legend.get_frame().get_visible() is False
    plt.close(fig)


def test_legacy_behavior_pack_uses_english_labels_frameless_legend_and_log_beta(monkeypatch, tmp_path):
    observed = {}

    def inspect_save(fig, path):
        ax = fig.axes[0]
        observed.update({
            "xlabel": ax.get_xlabel(),
            "ylabel": ax.get_ylabel(),
            "yscale": ax.get_yscale(),
            "legend_frame": ax.get_legend().get_frame().get_visible(),
        })
        plt.close(fig)
        return str(path)

    monkeypatch.setattr(figures, "_save", inspect_save)
    block = pd.DataFrame({
        "session_id": ["s1", "s1", "s2", "s2"],
        "repeat_participant_id": ["p1", "p1", "p2", "p2"],
        "block_id": ["B1", "B2", "B1", "B2"],
        "beta": [0.10, 0.20, 0.30, 13.0],
    })
    _, _, audit = figures.generate_complete_metric_figure_pack(
        session=pd.DataFrame(), block=block, cycle=pd.DataFrame(), probe=pd.DataFrame(),
        output_dir=tmp_path, metrics=["beta"],
    )
    assert observed == {
        "xlabel": "Block",
        "ylabel": "β",
        "yscale": "log",
        "legend_frame": False,
    }
    assert audit.loc[audit["status"].eq("generated"), "in_image_language"].eq("English").all()

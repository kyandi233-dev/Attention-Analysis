from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from attention_pipeline.config import Config
from attention_pipeline.nir_behavior.stimulus_visual import (
    build_stimulus_visual_table,
    srgb_relative_luminance,
)


def _config(tmp_path: Path) -> Config:
    return Config(
        path=tmp_path / "stimulus_visual.yaml",
        digest="test-digest",
        data={
            "pipeline": {"version": "stimulus-visual-test"},
            "focuswave": {
                "repository": "kyandi233-dev/FocusWave",
                "branch": "formaltest",
                "sart_task_path": "01-MainProgram/core/sart_task.py",
                "materials_subdir": "01-MainProgram/素材",
            },
            "paths": {
                "materials_dir": None,
                "output_csv": str(tmp_path / "out.csv"),
                "manifest_json": str(tmp_path / "manifest.json"),
            },
            "render": {
                "screen_width_px": 120,
                "screen_height_px": 80,
                "bg_fill_hex": "#808080",
                "base_size_px": 20,
                "stimulus_position_px": [0, -2],
                "stimulus_sizes_pct": [80, 100, 120],
                "resize_filter": "nearest",
                "fixed_central_roi_size_px": 40,
                "alpha_visible_threshold": 0.05,
            },
            "stimuli": {
                "go": [
                    "01_mango.png",
                    "02_grape.png",
                    "03_avocado.png",
                    "04_bitter_melon.png",
                    "05_cucumber.png",
                    "06_kiwi.png",
                    "07_kale.png",
                    "08_lime.png",
                ],
                "nogo": ["09_no_go.png"],
            },
            "references": {"background": "刺激.png", "mask": "掩蔽刺激.png"},
            "metrics": {
                "srgb_relative_luminance": True,
                "rms_contrast": True,
                "include_mask_delta": True,
            },
        },
    )


def _write_materials(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "GO").mkdir()
    (root / "NOGO").mkdir()
    (root.parent / "core").mkdir(parents=True, exist_ok=True)
    (root.parent / "core" / "sart_task.py").write_text("# synthetic\n", encoding="utf-8")

    Image.new("RGBA", (60, 40), (200, 200, 200, 255)).save(root / "刺激.png")
    # Opaque darker adaptive mask so stimulus-vs-mask deltas are non-zero.
    Image.new("RGBA", (60, 40), (100, 100, 100, 255)).save(root / "掩蔽刺激.png")

    go_names = [
        "01_mango.png",
        "02_grape.png",
        "03_avocado.png",
        "04_bitter_melon.png",
        "05_cucumber.png",
        "06_kiwi.png",
        "07_kale.png",
        "08_lime.png",
    ]
    for idx, name in enumerate(go_names):
        Image.new("RGBA", (32, 32), (20 + idx * 20, 220, 40, 255)).save(root / "GO" / name)
    Image.new("RGBA", (32, 32), (230, 220, 40, 255)).save(root / "NOGO" / "09_no_go.png")
    return root


def test_srgb_relative_luminance_endpoints() -> None:
    rgb = np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)
    y = srgb_relative_luminance(rgb)
    assert np.isclose(y[0, 0], 0.0)
    assert np.isclose(y[0, 1], 1.0)


def test_builds_exact_27_condition_table(tmp_path: Path) -> None:
    materials = _write_materials(tmp_path / "素材")
    table, manifest = build_stimulus_visual_table(_config(tmp_path), materials)
    assert len(table) == 27
    assert table["stimulus_name"].nunique() == 9
    assert sorted(table["stimulus_size_pct"].unique().tolist()) == [80, 100, 120]
    assert int(table["is_no_go"].sum()) == 3
    assert manifest["condition_count"] == 27


def test_rendered_size_and_area_increase_with_size(tmp_path: Path) -> None:
    materials = _write_materials(tmp_path / "素材")
    table, _ = build_stimulus_visual_table(_config(tmp_path), materials)
    mango = table[table["stimulus_name"] == "01_mango.png"].sort_values(
        "stimulus_size_pct"
    )
    assert mango["rendered_size_px"].tolist() == [16, 20, 24]
    assert mango["fruit_visible_area_px"].is_monotonic_increasing
    assert mango["fruit_visible_area_fraction_screen"].is_monotonic_increasing


def test_relative_luminance_deltas_are_recorded(tmp_path: Path) -> None:
    materials = _write_materials(tmp_path / "素材")
    table, _ = build_stimulus_visual_table(_config(tmp_path), materials)
    required = {
        "screen_rel_lum_mean",
        "central_rel_lum_mean",
        "fruit_support_rel_lum_mean",
        "background_screen_rel_lum_mean",
        "mask_screen_rel_lum_mean",
        "delta_screen_rel_lum_vs_background",
        "delta_central_rel_lum_vs_background",
        "delta_screen_rel_lum_vs_mask",
        "delta_central_rel_lum_vs_mask",
    }
    assert required.issubset(table.columns)
    assert table["delta_screen_rel_lum_vs_mask"].notna().all()

from __future__ import annotations

from pathlib import Path

from PIL import Image

from attention_pipeline.config import Config
from attention_pipeline.nir_behavior.stimulus_visual import build_stimulus_visual_table
from attention_pipeline.nir_behavior.stimulus_visual_report import (
    write_stimulus_visual_images,
)


def _config(tmp_path: Path) -> Config:
    return Config(
        path=tmp_path / "stimulus_visual.yaml",
        digest="test-report",
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
                "rendered_images_dir": str(tmp_path / "rendered"),
                "overview_full_png": str(tmp_path / "overview_full.png"),
                "overview_central_png": str(tmp_path / "overview_central.png"),
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
            "report_outputs": {"overview_tile_width_px": 60},
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


def _materials(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "GO").mkdir()
    (root / "NOGO").mkdir()
    (root.parent / "core").mkdir(parents=True, exist_ok=True)
    (root.parent / "core" / "sart_task.py").write_text("# synthetic\n", encoding="utf-8")
    Image.new("RGBA", (60, 40), (200, 200, 200, 255)).save(root / "刺激.png")
    Image.new("RGBA", (60, 40), (100, 100, 100, 255)).save(root / "掩蔽刺激.png")
    names = [
        "01_mango.png",
        "02_grape.png",
        "03_avocado.png",
        "04_bitter_melon.png",
        "05_cucumber.png",
        "06_kiwi.png",
        "07_kale.png",
        "08_lime.png",
    ]
    for idx, name in enumerate(names):
        Image.new("RGBA", (32, 32), (20 + idx * 20, 220, 40, 255)).save(root / "GO" / name)
    Image.new("RGBA", (32, 32), (230, 220, 40, 255)).save(root / "NOGO" / "09_no_go.png")
    return root


def test_writes_27_full_screen_pngs_and_two_overviews(tmp_path: Path) -> None:
    config = _config(tmp_path)
    materials = _materials(tmp_path / "素材")
    table, _ = build_stimulus_visual_table(config, materials)
    result = write_stimulus_visual_images(config, materials, table)

    assert result["full_resolution_condition_count"] == 27
    condition_dir = Path(result["conditions_dir"])
    files = sorted(condition_dir.glob("*.png"))
    assert len(files) == 27
    with Image.open(files[0]) as image:
        assert image.size == (120, 80)

    assert Path(result["references"]["background_screen"]["path"]).is_file()
    assert Path(result["references"]["mask_screen"]["path"]).is_file()
    assert Path(result["overview_full"]["path"]).is_file()
    assert Path(result["overview_central"]["path"]).is_file()

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from ..config import Config
from .stimulus_visual import (
    _alpha_composite_center,
    _load_rgba,
    _psychopy_center_to_pil,
    _render_reference_screens,
    _resample,
    _stimulus_paths,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(config: Config, raw: str | Path) -> Path:
    path = Path(str(raw))
    if not path.is_absolute():
        path = (config.path.parent.parent / path).resolve()
    return path


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _label_tile(image: Image.Image, label: str, *, width: int) -> Image.Image:
    rgb = image.convert("RGB")
    target_h = max(1, int(round(rgb.height * width / rgb.width)))
    thumb = rgb.resize((width, target_h), resample=Image.Resampling.LANCZOS)
    band_h = max(30, int(round(width * 0.08)))
    canvas = Image.new("RGB", (width, target_h + band_h), "white")
    canvas.paste(thumb, (0, band_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, max(2, band_h // 5)), label, fill="black", font=_font(max(12, band_h // 2)))
    return canvas


def _contact_sheet(
    rows: list[tuple[str, int, Image.Image]],
    *,
    stimulus_order: list[str],
    sizes: list[int],
    tile_width: int,
) -> Image.Image:
    by_key = {(name, size): image for name, size, image in rows}
    sample = next(iter(by_key.values()))
    sample_tile = _label_tile(sample, "sample", width=tile_width)
    tile_h = sample_tile.height
    sheet = Image.new(
        "RGB",
        (tile_width * len(sizes), tile_h * len(stimulus_order)),
        "white",
    )
    for row_idx, name in enumerate(stimulus_order):
        for col_idx, size in enumerate(sizes):
            image = by_key[(name, size)]
            label = f"{Path(name).stem} | {size}%"
            tile = _label_tile(image, label, width=tile_width)
            sheet.paste(tile, (col_idx * tile_width, row_idx * tile_h))
    return sheet


def write_stimulus_visual_images(
    config: Config,
    materials_dir: str | Path,
    table: pd.DataFrame,
) -> dict[str, Any]:
    """Write report-ready reconstructions of the 27 formal SART conditions.

    Outputs exact full-screen 2880x1920 reconstructions for the Beijing formal
    display profile, plus full-screen and central-crop contact sheets. This does
    not change the numeric CSV schema.
    """
    materials = Path(materials_dir).resolve()
    paths = config.section("paths")
    report = config.data.get("report_outputs", {})
    render = config.section("render")

    output_dir = _resolve_path(
        config,
        paths.get(
            "rendered_images_dir",
            "D:/_AttentionData/Beijing-NIR/analysis/nir-behavior-v1/stimulus_visual_rendered",
        ),
    )
    overview_full = _resolve_path(
        config,
        paths.get(
            "overview_full_png",
            "D:/_AttentionData/Beijing-NIR/analysis/nir-behavior-v1/stimulus_visual_overview_full.png",
        ),
    )
    overview_central = _resolve_path(
        config,
        paths.get(
            "overview_central_png",
            "D:/_AttentionData/Beijing-NIR/analysis/nir-behavior-v1/stimulus_visual_overview_central.png",
        ),
    )

    references_dir = output_dir / "references"
    conditions_dir = output_dir / "conditions"
    references_dir.mkdir(parents=True, exist_ok=True)
    conditions_dir.mkdir(parents=True, exist_ok=True)
    overview_full.parent.mkdir(parents=True, exist_ok=True)
    overview_central.parent.mkdir(parents=True, exist_ok=True)

    background, mask_screen, _ = _render_reference_screens(materials, config)
    background_path = references_dir / "00_background_screen.png"
    mask_path = references_dir / "00_mask_screen.png"
    background.convert("RGB").save(background_path, format="PNG", optimize=True)
    mask_screen.convert("RGB").save(mask_path, format="PNG", optimize=True)

    width = int(render["screen_width_px"])
    height = int(render["screen_height_px"])
    base_size = float(render["base_size_px"])
    pos_x, pos_y = [float(v) for v in render["stimulus_position_px"]]
    sizes = [int(v) for v in render["stimulus_sizes_pct"]]
    central_size = int(render["fixed_central_roi_size_px"])
    filt = _resample(str(render.get("resize_filter", "nearest")))
    center_x, center_y = _psychopy_center_to_pil(width, height, pos_x, pos_y)

    half = central_size // 2
    cx0 = max(0, int(round(center_x)) - half)
    cy0 = max(0, int(round(center_y)) - half)
    cx1 = min(width, cx0 + central_size)
    cy1 = min(height, cy0 + central_size)

    stimulus_order = [name for name, _, _ in _stimulus_paths(materials, config)]
    full_tiles: list[tuple[str, int, Image.Image]] = []
    central_tiles: list[tuple[str, int, Image.Image]] = []
    rendered_files: list[dict[str, Any]] = []

    for stimulus_name, is_no_go, stimulus_path in _stimulus_paths(materials, config):
        source = _load_rgba(stimulus_path)
        for size_pct in sizes:
            rendered_size = max(1, int(round(base_size * size_pct / 100.0)))
            stimulus = source.resize((rendered_size, rendered_size), resample=filt)
            screen = background.copy()
            _alpha_composite_center(
                screen,
                stimulus,
                center_x=center_x,
                center_y=center_y,
            )
            rgb = screen.convert("RGB")
            filename = f"{Path(stimulus_name).stem}_size{size_pct:03d}.png"
            image_path = conditions_dir / filename
            rgb.save(image_path, format="PNG", optimize=True)

            full_tiles.append((stimulus_name, size_pct, rgb.copy()))
            central_tiles.append(
                (stimulus_name, size_pct, rgb.crop((cx0, cy0, cx1, cy1)))
            )
            rendered_files.append(
                {
                    "stimulus_name": stimulus_name,
                    "stimulus_size_pct": size_pct,
                    "is_no_go": bool(is_no_go),
                    "path": str(image_path),
                    "sha256": _sha256(image_path),
                    "width_px": rgb.width,
                    "height_px": rgb.height,
                }
            )

    expected = len(table)
    if len(rendered_files) != expected:
        raise RuntimeError(
            f"rendered condition count {len(rendered_files)} != table rows {expected}"
        )

    tile_width = int(report.get("overview_tile_width_px", 480))
    full_sheet = _contact_sheet(
        full_tiles,
        stimulus_order=stimulus_order,
        sizes=sizes,
        tile_width=tile_width,
    )
    central_sheet = _contact_sheet(
        central_tiles,
        stimulus_order=stimulus_order,
        sizes=sizes,
        tile_width=tile_width,
    )
    full_sheet.save(overview_full, format="PNG", optimize=True)
    central_sheet.save(overview_central, format="PNG", optimize=True)

    return {
        "status": "complete",
        "full_resolution_condition_count": len(rendered_files),
        "full_resolution_size_px": [width, height],
        "rendered_images_dir": str(output_dir),
        "conditions_dir": str(conditions_dir),
        "references": {
            "background_screen": {
                "path": str(background_path),
                "sha256": _sha256(background_path),
            },
            "mask_screen": {
                "path": str(mask_path),
                "sha256": _sha256(mask_path),
            },
        },
        "overview_full": {
            "path": str(overview_full),
            "sha256": _sha256(overview_full),
            "description": "9 stimuli x 3 sizes, full-screen thumbnails",
        },
        "overview_central": {
            "path": str(overview_central),
            "sha256": _sha256(overview_central),
            "description": (
                "9 stimuli x 3 sizes, fixed central ROI thumbnails for report readability"
            ),
        },
        "condition_files": rendered_files,
        "note": (
            "Individual condition PNGs are exact reconstructed task-window images for the "
            "Beijing formal Surface Pro profile; overview sheets are downsampled report aids."
        ),
    }

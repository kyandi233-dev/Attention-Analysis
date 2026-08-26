from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from ..config import Config


STIMULUS_VISUAL_SCHEMA_VERSION = 1


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def srgb_relative_luminance(rgb: np.ndarray) -> np.ndarray:
    """Return linear-sRGB relative luminance Y in [0, 1].

    This is a digital image metric, not a calibrated display measurement in cd/m^2.
    """
    array = np.asarray(rgb, dtype=np.float64) / 255.0
    if array.shape[-1] != 3:
        raise ValueError("rgb input must end with three channels")
    linear = np.where(
        array <= 0.04045,
        array / 12.92,
        ((array + 0.055) / 1.055) ** 2.4,
    )
    return (
        0.2126 * linear[..., 0]
        + 0.7152 * linear[..., 1]
        + 0.0722 * linear[..., 2]
    )


def _rms_contrast(y: np.ndarray) -> float | None:
    values = np.asarray(y, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    mean = float(np.mean(values))
    if mean <= 0:
        return None
    return float(np.std(values, ddof=0) / mean)


def _stats(prefix: str, y: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(y, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            f"{prefix}_rel_lum_mean": None,
            f"{prefix}_rel_lum_median": None,
            f"{prefix}_rel_lum_sd": None,
            f"{prefix}_rms_contrast": None,
        }
    return {
        f"{prefix}_rel_lum_mean": float(np.mean(values)),
        f"{prefix}_rel_lum_median": float(np.median(values)),
        f"{prefix}_rel_lum_sd": float(np.std(values, ddof=0)),
        f"{prefix}_rms_contrast": _rms_contrast(values),
    }


def _hex_rgb(value: str) -> tuple[int, int, int]:
    text = str(value).strip().lstrip("#")
    if len(text) != 6:
        raise ValueError(f"expected #RRGGBB color, got {value!r}")
    return tuple(int(text[i : i + 2], 16) for i in (0, 2, 4))


def _resample(name: str) -> Image.Resampling:
    key = str(name).strip().lower()
    mapping = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }
    if key not in mapping:
        raise ValueError(f"unknown resize filter: {name}")
    return mapping[key]


def _load_rgba(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        return image.convert("RGBA")


def _adaptive_full_width(
    image: Image.Image,
    *,
    screen_width: int,
    resample: Image.Resampling,
) -> Image.Image:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("invalid image dimensions")
    scale = screen_width / float(width)
    target = (screen_width, max(1, int(round(height * scale))))
    return image.resize(target, resample=resample)


def _psychopy_center_to_pil(
    screen_width: int,
    screen_height: int,
    pos_x: float,
    pos_y: float,
) -> tuple[float, float]:
    # PsychoPy pix coordinates: +y is up. Pillow image coordinates: +y is down.
    return screen_width / 2.0 + pos_x, screen_height / 2.0 - pos_y


def _alpha_composite_center(
    base: Image.Image,
    overlay: Image.Image,
    *,
    center_x: float,
    center_y: float,
) -> tuple[int, int, int, int]:
    x0 = int(round(center_x - overlay.width / 2.0))
    y0 = int(round(center_y - overlay.height / 2.0))
    x1 = x0 + overlay.width
    y1 = y0 + overlay.height
    base.alpha_composite(overlay, dest=(x0, y0))
    return x0, y0, x1, y1


def _render_adaptive_layer(
    base: Image.Image,
    layer_path: Path,
    *,
    resample: Image.Resampling,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    layer = _adaptive_full_width(
        _load_rgba(layer_path),
        screen_width=base.width,
        resample=resample,
    )
    bbox = _alpha_composite_center(
        base,
        layer,
        center_x=base.width / 2.0,
        center_y=base.height / 2.0,
    )
    return base, bbox


def _central_bounds(
    screen_width: int,
    screen_height: int,
    center_x: float,
    center_y: float,
    size: int,
) -> tuple[int, int, int, int]:
    half = int(size) // 2
    x0 = max(0, int(round(center_x)) - half)
    y0 = max(0, int(round(center_y)) - half)
    x1 = min(screen_width, x0 + int(size))
    y1 = min(screen_height, y0 + int(size))
    return x0, y0, x1, y1


def _rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _render_reference_screens(
    materials_dir: Path,
    config: Config,
) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    render = config.section("render")
    refs = config.section("references")
    width = int(render["screen_width_px"])
    height = int(render["screen_height_px"])
    fill = _hex_rgb(str(render["bg_fill_hex"]))
    filt = _resample(str(render.get("resize_filter", "nearest")))

    base = Image.new("RGBA", (width, height), fill + (255,))
    background_path = materials_dir / str(refs["background"])
    mask_path = materials_dir / str(refs["mask"])

    background, background_bbox = _render_adaptive_layer(
        base.copy(), background_path, resample=filt
    )
    mask_screen = background.copy()
    mask_screen, mask_bbox = _render_adaptive_layer(
        mask_screen, mask_path, resample=filt
    )
    provenance = {
        "background_path": str(background_path),
        "background_sha256": sha256(background_path),
        "background_bbox_px": list(background_bbox),
        "mask_path": str(mask_path),
        "mask_sha256": sha256(mask_path),
        "mask_bbox_px": list(mask_bbox),
    }
    return background, mask_screen, provenance


def _stimulus_paths(materials_dir: Path, config: Config) -> list[tuple[str, bool, Path]]:
    stimuli = config.section("stimuli")
    rows: list[tuple[str, bool, Path]] = []
    for name in stimuli.get("go", []):
        rows.append((str(name), False, materials_dir / "GO" / str(name)))
    for name in stimuli.get("nogo", []):
        rows.append((str(name), True, materials_dir / "NOGO" / str(name)))
    return rows


def build_stimulus_visual_table(
    config: Config,
    materials_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    materials = Path(materials_dir).resolve()
    if not materials.is_dir():
        raise FileNotFoundError(f"FocusWave materials directory not found: {materials}")

    render = config.section("render")
    width = int(render["screen_width_px"])
    height = int(render["screen_height_px"])
    base_size = float(render["base_size_px"])
    pos_x, pos_y = [float(v) for v in render["stimulus_position_px"]]
    sizes = [int(v) for v in render["stimulus_sizes_pct"]]
    central_size = int(render["fixed_central_roi_size_px"])
    alpha_threshold = float(render.get("alpha_visible_threshold", 0.05))
    filt = _resample(str(render.get("resize_filter", "nearest")))

    center_x, center_y = _psychopy_center_to_pil(width, height, pos_x, pos_y)
    central_bbox = _central_bounds(width, height, center_x, center_y, central_size)

    background, mask_screen, ref_provenance = _render_reference_screens(
        materials, config
    )
    background_y = srgb_relative_luminance(_rgb_array(background))
    mask_y = srgb_relative_luminance(_rgb_array(mask_screen))
    cx0, cy0, cx1, cy1 = central_bbox
    background_central_y = background_y[cy0:cy1, cx0:cx1]
    mask_central_y = mask_y[cy0:cy1, cx0:cx1]

    reference_metrics: dict[str, Any] = {}
    reference_metrics.update(_stats("background_screen", background_y))
    reference_metrics.update(_stats("background_central", background_central_y))
    reference_metrics.update(_stats("mask_screen", mask_y))
    reference_metrics.update(_stats("mask_central", mask_central_y))

    records: list[dict[str, Any]] = []
    source_files: list[dict[str, Any]] = []
    for stimulus_name, is_no_go, stimulus_path in _stimulus_paths(materials, config):
        source_hash = sha256(stimulus_path)
        source_files.append(
            {
                "stimulus_name": stimulus_name,
                "is_no_go": bool(is_no_go),
                "path": str(stimulus_path),
                "sha256": source_hash,
            }
        )
        source = _load_rgba(stimulus_path)
        for size_pct in sizes:
            rendered_size = max(1, int(round(base_size * size_pct / 100.0)))
            stimulus = source.resize(
                (rendered_size, rendered_size),
                resample=filt,
            )
            screen = background.copy()
            bbox = _alpha_composite_center(
                screen,
                stimulus,
                center_x=center_x,
                center_y=center_y,
            )

            screen_rgb = _rgb_array(screen)
            screen_y = srgb_relative_luminance(screen_rgb)
            central_y = screen_y[cy0:cy1, cx0:cx1]

            x0, y0, x1, y1 = bbox
            clip_x0, clip_y0 = max(0, x0), max(0, y0)
            clip_x1, clip_y1 = min(width, x1), min(height, y1)
            alpha = np.asarray(stimulus.getchannel("A"), dtype=np.float64) / 255.0
            alpha_x0 = clip_x0 - x0
            alpha_y0 = clip_y0 - y0
            alpha_x1 = alpha_x0 + max(0, clip_x1 - clip_x0)
            alpha_y1 = alpha_y0 + max(0, clip_y1 - clip_y0)
            alpha_clip = alpha[alpha_y0:alpha_y1, alpha_x0:alpha_x1]
            visible = alpha_clip > alpha_threshold
            support_y = screen_y[clip_y0:clip_y1, clip_x0:clip_x1][visible]
            visible_area = int(np.count_nonzero(visible))

            record: dict[str, Any] = {
                "stimulus_name": stimulus_name,
                "stimulus_code": Path(stimulus_name).stem,
                "is_no_go": int(is_no_go),
                "stimulus_size_pct": size_pct,
                "base_size_px": base_size,
                "rendered_size_px": rendered_size,
                "stimulus_pos_x_psychopy_px": pos_x,
                "stimulus_pos_y_psychopy_px": pos_y,
                "stimulus_center_x_image_px": center_x,
                "stimulus_center_y_image_px": center_y,
                "screen_width_px": width,
                "screen_height_px": height,
                "fixed_central_roi_size_px": central_size,
                "fixed_central_roi_x0": cx0,
                "fixed_central_roi_y0": cy0,
                "fixed_central_roi_x1": cx1,
                "fixed_central_roi_y1": cy1,
                "stimulus_bbox_x0": x0,
                "stimulus_bbox_y0": y0,
                "stimulus_bbox_x1": x1,
                "stimulus_bbox_y1": y1,
                "fruit_source_sha256": source_hash,
                "fruit_visible_area_px": visible_area,
                "fruit_visible_area_fraction_screen": visible_area / float(width * height),
                "fruit_visible_area_fraction_central_roi": visible_area
                / float(max(1, (cx1 - cx0) * (cy1 - cy0))),
            }
            record.update(_stats("screen", screen_y))
            record.update(_stats("central", central_y))
            record.update(_stats("fruit_support", support_y))
            record.update(reference_metrics)
            record["delta_screen_rel_lum_vs_background"] = (
                record["screen_rel_lum_mean"]
                - reference_metrics["background_screen_rel_lum_mean"]
            )
            record["delta_central_rel_lum_vs_background"] = (
                record["central_rel_lum_mean"]
                - reference_metrics["background_central_rel_lum_mean"]
            )
            record["delta_screen_rel_lum_vs_mask"] = (
                record["screen_rel_lum_mean"]
                - reference_metrics["mask_screen_rel_lum_mean"]
            )
            record["delta_central_rel_lum_vs_mask"] = (
                record["central_rel_lum_mean"]
                - reference_metrics["mask_central_rel_lum_mean"]
            )
            records.append(record)

    table = pd.DataFrame(records).sort_values(
        ["is_no_go", "stimulus_name", "stimulus_size_pct"]
    ).reset_index(drop=True)

    expected = len(_stimulus_paths(materials, config)) * len(sizes)
    if len(table) != expected:
        raise RuntimeError(f"expected {expected} stimulus conditions, got {len(table)}")

    focuswave = config.section("focuswave")
    sart_task_path = materials.parent / "core" / "sart_task.py"
    manifest: dict[str, Any] = {
        "status": "complete",
        "schema_version": STIMULUS_VISUAL_SCHEMA_VERSION,
        "pipeline_version": config.section("pipeline").get("version"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_digest": config.digest,
        "materials_dir": str(materials),
        "focuswave_repository": focuswave.get("repository"),
        "focuswave_branch": focuswave.get("branch"),
        "focuswave_sart_task_path": str(sart_task_path),
        "focuswave_sart_task_sha256": sha256(sart_task_path)
        if sart_task_path.is_file()
        else None,
        "render": dict(render),
        "reference_sources": ref_provenance,
        "stimulus_sources": source_files,
        "condition_count": int(len(table)),
        "metric_semantics": {
            "relative_luminance": (
                "IEC/WCAG-style linear-sRGB Y = 0.2126R + 0.7152G + 0.0722B; "
                "digital relative metric, not physical cd/m^2"
            ),
            "rms_contrast": "population SD(Y) / mean(Y)",
            "screen": "entire reconstructed task window",
            "central": "fixed square ROI centered on PsychoPy stimulus position",
            "fruit_support": "composited pixels whose resized fruit alpha exceeds threshold",
            "mask_delta": (
                "stimulus-screen relative luminance minus rendered mask-screen relative luminance"
            ),
        },
    }
    return table, manifest


def _candidate_material_dirs(config: Config) -> list[Path]:
    candidates: list[Path] = []
    env = os.environ.get("FOCUSWAVE_MATERIALS_DIR")
    if env:
        candidates.append(Path(env))

    raw = config.section("paths").get("materials_dir")
    if raw:
        path = Path(str(raw))
        if not path.is_absolute():
            path = (config.path.parent.parent / path).resolve()
        candidates.append(path)

    repo_root = config.path.parent.parent
    subdir = Path(str(config.section("focuswave")["materials_subdir"]))
    for name in ("FocusWave", "FocusWave-formaltest", "focuswave"):
        candidates.append(repo_root.parent / name / subdir)
    return candidates


def resolve_materials_dir(
    config: Config,
    explicit: str | Path | None = None,
) -> Path:
    if explicit is not None:
        path = Path(explicit).resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        return path
    for candidate in _candidate_material_dirs(config):
        candidate = candidate.resolve()
        if candidate.is_dir():
            return candidate
    attempted = "\n".join(f"  - {p}" for p in _candidate_material_dirs(config))
    raise FileNotFoundError(
        "Could not find a local FocusWave materials directory. "
        "Pass --materials-dir or set FOCUSWAVE_MATERIALS_DIR. Tried:\n"
        + attempted
    )


def write_stimulus_visual_outputs(
    config: Config,
    table: pd.DataFrame,
    manifest: dict[str, Any],
) -> tuple[Path, Path]:
    paths = config.section("paths")
    csv_path = Path(str(paths["output_csv"]))
    manifest_path = Path(str(paths["manifest_json"]))
    if not csv_path.is_absolute():
        csv_path = (config.path.parent.parent / csv_path).resolve()
    if not manifest_path.is_absolute():
        manifest_path = (config.path.parent.parent / manifest_path).resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    payload = dict(manifest)
    payload["output_csv"] = str(csv_path)
    payload["output_csv_sha256"] = sha256(csv_path)
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return csv_path, manifest_path

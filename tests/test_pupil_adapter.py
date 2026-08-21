import csv
import importlib.util
import sys
import types
from pathlib import Path

import cv2
import numpy as np


def _load_sequence_adapter():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        path = scripts / "nir_sequence_detect.py"
        spec = importlib.util.spec_from_file_location("nir_sequence_detect_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


class FakePupil:
    def __init__(self, size=(20.0, 40.0), angle=70.0, valid=True):
        self.center = (100.0, 80.0)
        self.size = size
        self.angle = angle
        self.confidence = 0.8 if valid else -1.0
        self.outline_confidence = 0.9 if valid else -1.0
        self._valid = valid

    def valid(self, threshold):
        return self._valid


def test_sequence_record_preserves_raw_axes_and_applies_px_gate():
    module = _load_sequence_adapter()
    frame = {
        "subject": "sub", "block": "1", "sequence_id": "seq", "eye": "eye_right",
        "frame_offset": 0, "frame_idx": 1, "unix_ms": 1000,
        "face_status": "ok", "roi_status": "ok",
    }
    image = np.full((160, 320), 180, dtype=np.uint8)
    cv2.ellipse(image, (100, 80), (10, 20), 70, 0, 360, 20, -1)
    row, reset = module._record(
        frame, "PuReST", FakePupil(), image, 1.0, 6, 50, 0.5, "first_after_reset"
    )
    assert row["raw_axis_w_px"] == 20.0
    assert row["raw_axis_h_px"] == 40.0
    assert row["major_diameter"] == 40.0
    assert row["major_angle_deg"] == 160.0
    assert row["returned"] == 1
    assert reset is False

    rejected, reset = module._record(
        frame, "PuReST", FakePupil(size=(20.0, 60.0)), image, 1.0,
        6, 50, 0.5, "continuing_session"
    )
    assert rejected["algorithm_returned"] == 1
    assert rejected["returned"] == 0
    assert rejected["quality_status"] == "diameter_rejected"
    assert reset is True


def test_main_preserves_missing_roi_rows(tmp_path, monkeypatch):
    module = _load_sequence_adapter()

    class FakePuRe:
        def runWithConfidence(self, image, roi, pupil, min_px, max_px):
            pupil.__dict__.update(FakePupil().__dict__)

    class FakePuReST:
        def __init__(self):
            self.minPupilDiameterMM = 2.0
        def reset(self):
            pass
        def runWithConfidence(self, image):
            return FakePupil()

    fake = types.SimpleNamespace(Pupil=FakePupil, PuRe=FakePuRe, PuReST=FakePuReST)
    monkeypatch.setitem(sys.modules, "pypupilext", fake)

    image_path = tmp_path / "eye.png"
    cv2.imencode(".png", np.full((160, 320), 128, dtype=np.uint8))[1].tofile(str(image_path))
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "subject", "block", "sequence_id", "eye", "frame_offset",
            "frame_idx", "unix_ms", "face_detected", "roi_status", "roi_path",
        ])
        writer.writeheader()
        writer.writerow({
            "subject": "sub", "block": 1, "sequence_id": "seq", "eye": "eye_right",
            "frame_offset": 0, "frame_idx": 10, "unix_ms": 1000,
            "face_detected": 1, "roi_status": "ok", "roi_path": image_path.name,
        })
        writer.writerow({
            "subject": "sub", "block": 1, "sequence_id": "seq", "eye": "eye_right",
            "frame_offset": 1, "frame_idx": 11, "unix_ms": 1033,
            "face_detected": 0, "roi_status": "roi_missing", "roi_path": "",
        })
    out = tmp_path / "out.csv"
    assert module.main([
        "--manifest", str(manifest), "--roi-root", str(tmp_path), "--out", str(out),
    ]) == 0
    with out.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    missing = [row for row in rows if row["frame_offset"] == "1"]
    assert len(missing) == 2
    assert {row["error_code"] for row in missing} == {"roi_missing"}

def test_visibility_gate_is_external_and_explicit():
    module = _load_sequence_adapter()
    assert module._visibility_ok({"visible_proxy": "1", "p80_closed_proxy": "0", "openness": "0.8"}, 0.55)
    assert not module._visibility_ok({"visible_proxy": "0", "p80_closed_proxy": "0", "openness": "0.8"}, 0.55)
    assert not module._visibility_ok({"visible_proxy": "1", "p80_closed_proxy": "1", "openness": "0.8"}, 0.55)
    assert not module._visibility_ok({"visible_proxy": "1", "p80_closed_proxy": "0", "openness": "0.3"}, 0.55)


def test_purest_resets_only_after_consecutive_diameter_rejections(tmp_path, monkeypatch):
    module = _load_sequence_adapter()

    class FakePuReST:
        def __init__(self):
            self.minPupilDiameterMM = 2.0
        def reset(self):
            pass
        def runWithConfidence(self, image):
            return FakePupil(size=(20.0, 60.0))

    fake = types.SimpleNamespace(Pupil=FakePupil, PuRe=lambda: None, PuReST=FakePuReST)
    monkeypatch.setitem(sys.modules, "pypupilext", fake)
    image_path = tmp_path / "eye.png"
    cv2.imencode(".png", np.full((160, 320), 128, dtype=np.uint8))[1].tofile(str(image_path))
    manifest = tmp_path / "manifest.csv"
    fields = [
        "subject", "block", "sequence_id", "eye", "frame_offset", "frame_idx",
        "unix_ms", "face_detected", "roi_status", "roi_path", "openness",
        "visible_proxy", "p80_closed_proxy",
    ]
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for offset in range(3):
            writer.writerow({
                "subject": "sub", "block": 1, "sequence_id": "seq", "eye": "eye_right",
                "frame_offset": offset, "frame_idx": 10 + offset, "unix_ms": 1000 + 33 * offset,
                "face_detected": 1, "roi_status": "ok", "roi_path": image_path.name,
                "openness": 0.8, "visible_proxy": 1, "p80_closed_proxy": 0,
            })
    out = tmp_path / "out.csv"
    assert module.main([
        "--manifest", str(manifest), "--roi-root", str(tmp_path), "--out", str(out),
        "--algorithms", "PuReST", "--reset-after-quality-rejects", "3",
    ]) == 0
    with out.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["quality_status"] for row in rows] == ["diameter_rejected"] * 3
    assert [row["session_state"] for row in rows] == [
        "quality_rejected", "quality_rejected", "session_reset",
    ]


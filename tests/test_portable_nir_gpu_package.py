import importlib.util
from pathlib import Path
import sys

import numpy as np


PACKAGE = Path(__file__).resolve().parents[1] / "runtime" / "nir-yolo-tracking-ritnet-v1"
sys.path.insert(0, str(PACKAGE))
SPEC = importlib.util.spec_from_file_location("portable_nir_pipeline", PACKAGE / "run_pipeline.py")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_subject_normalization_and_known_paths():
    assert MODULE.normalize_subject("sub-056") == ("sub-056_", "sub-056")
    assert MODULE.normalize_subject("sub-056_") == ("sub-056_", "sub-056")


def test_center_jump_gate():
    previous = (10.0, 10.0, 30.0, 20.0)
    assert MODULE.center_jump_ok(previous, (15.0, 10.0, 35.0, 20.0), 0.5)
    assert not MODULE.center_jump_ok(previous, (30.0, 10.0, 50.0, 20.0), 0.5)


def test_expand_crop_clips_and_has_fixed_size():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    roi, box, clipped = MODULE.expand_crop(frame, (0.0, 10.0, 40.0, 30.0), 0.3, 0.45, (320, 160))
    assert roi.shape == (160, 320)
    assert box[0] == 0
    assert clipped


def test_valid_box_rejects_out_of_frame():
    shape = (100, 200, 3)
    assert MODULE.valid_box((1, 2, 20, 30), shape)
    assert not MODULE.valid_box((-1, 2, 20, 30), shape)


def test_tracker_box_is_integer_xywh_for_opencv():
    assert MODULE.xyxy_to_xywh((1.2, 2.4, 11.7, 22.8)) == (1, 2, 10, 20)

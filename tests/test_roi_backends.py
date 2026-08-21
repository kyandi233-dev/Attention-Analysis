import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd


def _scripts_modules():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        common = importlib.import_module("roi_common")
        yunet = importlib.import_module("roi_yunet")
        return common, yunet
    finally:
        sys.path.remove(str(scripts))


def test_yunet_landmarks_are_already_pixel_coordinates(monkeypatch):
    common, yunet = _scripts_modules()

    class Detector:
        def setInputSize(self, size):
            self.size = size
        def detect(self, frame):
            face = np.array([100, 100, 300, 300, 300, 220, 190, 220, 245, 270, 320, 310, 180, 310, 0.9], dtype=np.float32)
            return True, face.reshape(1, -1)

    monkeypatch.setattr(cv2.FaceDetectorYN, "create", lambda *args, **kwargs: Detector())
    monkeypatch.setattr(yunet, "ascii_model_path", lambda path: str(path))
    backend = yunet.YuNetRoi("model.onnx", 320, 160, corner_span=0.67)
    eyes = backend.eyes(np.zeros((720, 1280, 3), dtype=np.uint8))
    assert set(eyes) == {"eye_right", "eye_left"}
    assert eyes["eye_right"]["image"].shape == (160, 320)
    assert eyes["eye_left"]["image"].shape == (160, 320)
    assert np.asarray(eyes["eye_right"]["source_to_roi_affine"]).shape == (2, 3)
    assert np.asarray(eyes["eye_right"]["roi_to_source_affine"]).shape == (2, 3)
    assert backend.name == "yunet-span670"
    assert eyes["eye_right"]["reference_kind"] == "yunet_eye_center_estimated_scale"
    assert eyes["eye_right"]["canthi_source"] is None


def test_build_roi_preserves_failed_frame_eye_rows(tmp_path):
    common, _ = _scripts_modules()
    root = tmp_path / "formal"
    video_dir = root / "sub-001_" / "nir"
    video_dir.mkdir(parents=True)
    video = video_dir / "sub-001_nir.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for value in (0, 80, 160, 240):
        writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
    writer.release()

    class Provider:
        name = "fake"
        def __init__(self):
            self.count = 0
            self.closed = False
        def eyes(self, frame):
            self.count += 1
            if self.count == 2:
                return None
            payload = common.crop_resize_gray_payload(frame, (0, 0, 64, 48), 320, 160)
            return {"eye_right": payload, "eye_left": payload}
        def close(self):
            self.closed = True

    provider = Provider()
    args = SimpleNamespace(
        root=str(root), subject="sub-001", block=1, seg_starts="0",
        n_segments=1, frames_per_seg=4,
    )
    out = tmp_path / "out"
    manifest, successful = common.build_roi(provider, args, out)
    rows = pd.read_csv(manifest)
    assert len(rows) == 8
    assert successful == 6
    assert (rows["roi_status"] == "roi_missing").sum() == 2
    successful_rows = rows[rows["roi_status"] == "ok"]
    assert successful_rows["source_to_roi_affine"].notna().all()
    assert successful_rows["roi_to_source_affine"].notna().all()
    assert successful_rows["roi_corners_source"].notna().all()
    assert provider.closed is True

def test_axis_aligned_roi_mapping_round_trip():
    common, _ = _scripts_modules()
    frame = np.zeros((240, 400, 3), dtype=np.uint8)
    payload = common.crop_resize_gray_payload(frame, (40, 30, 240, 130), 320, 160)
    source = np.array([[40.0, 30.0], [240.0, 130.0], [120.0, 80.0]], dtype=np.float32)
    roi = cv2.transform(source[None, :, :], payload["source_to_roi_affine"])[0]
    restored = cv2.transform(roi[None, :, :], payload["roi_to_source_affine"])[0]
    assert np.allclose(restored, source, atol=1e-4)
    assert np.asarray(payload["roi_corners_source"]).shape == (4, 2)

def test_formal_config_populates_roi_defaults():
    common, _ = _scripts_modules()
    config_path = Path(__file__).resolve().parents[1] / "configs" / "formal.yaml"
    args = SimpleNamespace(
        config=str(config_path), root=None, out=None, venv_python=None, block=None,
        n_segments=None, frames_per_seg=None, roi_w=None, roi_h=None,
        px_min=None, px_max=None, pupil_min_mm=None, openness_visible=None,
        max_session_gap_ms=None, reset_after_quality_rejects=None, yunet_model=None,
    )
    model = common.resolve_common_args(args, "yunet_model")
    assert Path(args.root).as_posix() == "E:/正式实验"
    assert args.frames_per_seg == 120 * 30
    assert (args.roi_w, args.roi_h) == (320, 160)
    assert (args.px_min, args.px_max) == (6, 50)
    assert model.endswith("yunet_2023mar.onnx")



def test_shared_openness_gate_uses_one_mediapipe_source(tmp_path):
    common, _ = _scripts_modules()
    source = pd.DataFrame([
        {"frame_idx": frame, "frame_offset": frame, "eye": eye, "ear": ear}
        for frame, values in enumerate(((0.20, 0.30), (0.40, 0.50), (0.05, 0.05)))
        for eye, ear in zip(("eye_right", "eye_left"), values)
    ])
    target = source.drop(columns=["ear"]).copy()
    source_path = tmp_path / "mediapipe.csv"
    target_path = tmp_path / "alternative.csv"
    source.to_csv(source_path, index=False)
    target.to_csv(target_path, index=False)
    result = common.apply_openness_gate(
        source_path, calibration_frames=2, visible_threshold=0.55, closed_threshold=0.20
    )
    common.apply_openness_gate(
        target_path, calibration_frames=2, visible_threshold=0.55, closed_threshold=0.20,
        source_manifest=source_path,
    )
    gated = pd.read_csv(target_path)
    assert result["source"] == "mediapipe_self"
    assert gated["visibility_source"].eq("mediapipe_shared").all()
    assert gated["ear"].notna().all()
    assert gated["baseline_open_ear"].notna().all()
    assert gated.loc[gated["frame_idx"] == 2, "p80_closed_proxy"].eq(1).all()


def _scripts_mediapipe():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        return importlib.import_module("roi_mediapipe"), importlib.import_module("roi_common")
    finally:
        sys.path.remove(str(scripts))


class _FakeSession:
    """模拟 FaceLandmarkerSession：返回固定 478 点（右眼角 33/133、左眼角 362/263）。"""

    def __init__(self, model_path, confidence=0.5):
        self.conf = confidence

    def detect(self, frame):
        pts = np.zeros((478, 2), dtype=np.float32)
        pts[33] = [300, 300]
        pts[133] = [450, 300]
        pts[362] = [900, 300]
        pts[263] = [1050, 300]
        return pts

    def close(self):
        pass


def test_mediapipe_roi_name_and_params(monkeypatch):
    mediapipe, _ = _scripts_mediapipe()
    monkeypatch.setattr(mediapipe, "FaceLandmarkerSession", _FakeSession)
    r = mediapipe.MediaPipeRoi("model.task", 320, 160, corner_span=0.5, confidence=0.3, clahe=True)
    assert r.name == "mediapipe-span500c030clahe"
    assert r._session.conf == 0.3
    assert r._clahe is not None
    r.close()


def test_mediapipe_roi_clahe_enhances(monkeypatch):
    mediapipe, _ = _scripts_mediapipe()
    monkeypatch.setattr(mediapipe, "FaceLandmarkerSession", _FakeSession)
    base = np.full((720, 1280, 3), 60, dtype=np.uint8)
    noise = np.random.RandomState(0).randint(-5, 6, (720, 1280, 3)).astype(np.int16)
    frame = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    r_raw = mediapipe.MediaPipeRoi("model.task", 320, 160, confidence=0.5, clahe=False)
    r_cla = mediapipe.MediaPipeRoi("model.task", 320, 160, confidence=0.5, clahe=True)
    e_raw, e_cla = r_raw.eyes(frame), r_cla.eyes(frame)
    assert e_raw and e_cla
    std_raw = float(np.std(e_raw["eye_right"]["image"]))
    std_cla = float(np.std(e_cla["eye_right"]["image"]))
    assert std_cla > std_raw  # CLAHE 拉伸对比度


def test_roi_check_configs():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        roi_check = importlib.import_module("roi_check")
    finally:
        sys.path.remove(str(scripts))
    assert len(roi_check.CONFIGS) == 4
    assert [c[1] for c in roi_check.CONFIGS] == [False, False, True, True]
    assert [c[2] for c in roi_check.CONFIGS] == [0.5, 0.3, 0.5, 0.3]
    assert len(roi_check.SUBJECTS) >= 7  # E:\\正式实验 探测到的正式被试数


def _scripts_faceparts():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        return importlib.import_module("roi_faceparts"), importlib.import_module("roi_common")
    finally:
        sys.path.remove(str(scripts))


class _Arr:
    """模拟 ultralytics Boxes 的 .cpu().numpy() 链。"""
    def __init__(self, a):
        self._a = np.asarray(a)
    def cpu(self):
        return self
    def numpy(self):
        return self._a


class _FakeFacePartsBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = _Arr(xyxy)
        self.conf = _Arr(conf)
        self.cls = _Arr(cls)
    def __len__(self):
        return len(self.xyxy.numpy())


class _FakeFacePartsResult:
    def __init__(self, xyxy, conf, cls):
        self.boxes = _FakeFacePartsBoxes(xyxy, conf, cls)


class _FakeFacePartsYolo:
    """模拟 ultralytics.YOLO：类别映射 + 返回含 2 eye + 1 eyebrow 的检测。"""
    def __init__(self, path):
        self.names = {0: "eye", 1: "nose", 2: "mouth", 3: "eyebrow"}
    def __call__(self, frame, conf=0.25, verbose=False, agnostic_nms=True, imgsz=640):
        xyxy = np.array([[100, 80, 160, 140], [300, 80, 360, 140], [200, 60, 220, 90]], dtype=np.float32)
        confs = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        clss = np.array([0, 0, 3], dtype=np.int32)  # 两个 eye(0) + 一个 eyebrow(3)
        return [_FakeFacePartsResult(xyxy, confs, clss)]


def test_faceparts_roi_detects_two_eyes_sorted_by_x(monkeypatch):
    faceparts, common = _scripts_faceparts()
    monkeypatch.setattr(faceparts, "_make_model", lambda path: _FakeFacePartsYolo(path))
    r = faceparts.FacePartsRoi("yolov8n.pt", 320, 160, corner_span=0.5)
    eyes = r.eyes(np.zeros((720, 1280, 3), dtype=np.uint8))
    assert set(eyes) == {"eye_right", "eye_left"}
    assert eyes["eye_right"]["image"].shape == (160, 320)
    assert eyes["eye_left"]["image"].shape == (160, 320)
    # 单 eye 类靠 x 排序分左右：x 小=eye_right（框 100..160 中心 130），x 大=eye_left（框 300..360 中心 330）
    right_ref = np.asarray(eyes["eye_right"]["reference_points_source"])
    left_ref = np.asarray(eyes["eye_left"]["reference_points_source"])
    assert right_ref[0][0] < 200
    assert left_ref[0][0] > 200
    assert eyes["eye_right"]["reference_kind"] == "faceparts_eye_bbox"
    assert r.name == "faceparts-n"
    r.close()


def test_faceparts_roi_rejects_wrong_class_mapping(monkeypatch):
    faceparts, _ = _scripts_faceparts()
    class BadYolo(_FakeFacePartsYolo):
        def __init__(self, path):
            self.names = {0: "face"}  # 错误类别映射
    monkeypatch.setattr(faceparts, "_make_model", lambda path: BadYolo(path))
    import pytest
    with pytest.raises(ValueError):
        faceparts.FacePartsRoi("bad.pt", 320, 160)


def test_faceparts_roi_returns_none_when_fewer_than_two_eyes(monkeypatch):
    faceparts, _ = _scripts_faceparts()
    class OneEyeYolo(_FakeFacePartsYolo):
        def __call__(self, frame, conf=0.25, verbose=False, agnostic_nms=True, imgsz=640):
            xyxy = np.array([[100, 80, 160, 140]], dtype=np.float32)
            confs = np.array([0.9], dtype=np.float32)
            clss = np.array([0], dtype=np.int32)
            return [_FakeFacePartsResult(xyxy, confs, clss)]
    monkeypatch.setattr(faceparts, "_make_model", lambda path: OneEyeYolo(path))
    r = faceparts.FacePartsRoi("yolov8n.pt", 320, 160)
    assert r.eyes(np.zeros((720, 1280, 3), dtype=np.uint8)) is None
    r.close()


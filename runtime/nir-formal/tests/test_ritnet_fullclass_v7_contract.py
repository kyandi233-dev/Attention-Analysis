from __future__ import annotations

import cv2
import numpy as np

from ritnet_fullclass_final_engine import CORE_VERSION, _summarize_outputs, PreparedBatch
from ritnet_fullclass_final_runtime import COHORT_OUTPUT_NAMES, OUTPUT_NAMES, _DerivedUncertaintyBatch, RitnetFullClassFinalRuntime
from ritnet_fullclass_qc import build_qc_selections
from ritnet_fullclass_schema import EYE_METRIC_FIELDS, EYE_METRICS_SCHEMA_VERSION
from ritnet_fullclass_uncertainty import summarize_uncertainty


def _labels():
    labels = np.zeros((400, 640), dtype=np.uint8)
    cv2.ellipse(labels, (320, 200), (220, 90), 0, 0, 360, 1, -1)
    cv2.ellipse(labels, (320, 200), (90, 60), 0, 0, 360, 2, -1)
    cv2.ellipse(labels, (320, 200), (35, 25), 0, 0, 360, 3, -1)
    return labels


def _probability():
    p = np.empty((4, 400, 640), dtype=np.float32)
    p[0], p[1], p[2], p[3] = 0.50, 0.30, 0.15, 0.05
    return p


def test_v7_lean_schema_exact_contract():
    assert EYE_METRICS_SCHEMA_VERSION == 6
    assert "pupil_geom_mean_diameter" in EYE_METRIC_FIELDS
    assert "pupil_to_iris_diameter_ratio" not in EYE_METRIC_FIELDS
    assert "whole_max_probability_p50" not in EYE_METRIC_FIELDS
    assert "boundary_entropy_p95" not in EYE_METRIC_FIELDS
    assert "ocular_max_probability_mean" in EYE_METRIC_FIELDS


def test_pupil_only_qc_composite_and_no_iris_reasons():
    frame = {"phase": "block1", "phase_segment": 1, "frame_idx": 1, "coverage_status": "both_eyes_success", "fixed_qc_anchor": False}
    eye = {"phase": "block1", "phase_segment": 1, "frame_idx": 1, "eye": "frame_left", "qc_pupil_fragmented": True, "temporal_anomaly": False}
    selections = build_qc_selections(frame_coverage_rows=[frame], eye_metric_rows=[eye], anomaly_limit_per_reason_per_phase=2, max_image_count=2)
    assert selections[0].reasons == ("pupil_fragmented",)


def test_compact_cohort_keeps_soft_fractions_and_skips_distributions():
    labels, valid, p = _labels(), np.ones((400, 640), dtype=bool), _probability()
    full = summarize_uncertainty(labels=labels, valid_source_mask=valid, class_probability=p, max_probability=p.max(0), top1_top2_margin=np.sort(p, axis=0)[-1] - np.sort(p, axis=0)[-2], entropy=-np.sum(p*np.log(p), axis=0))
    compact = summarize_uncertainty(labels=labels, valid_source_mask=valid, class_probability=p, max_probability=_DerivedUncertaintyBatch(p[None], "max_probability")[0], top1_top2_margin=_DerivedUncertaintyBatch(p[None], "top1_top2_margin")[0], entropy=_DerivedUncertaintyBatch(p[None], "entropy")[0], inputs_validated=True)
    for field in ("soft_background_fraction", "soft_sclera_fraction", "soft_iris_fraction", "soft_pupil_fraction", "ocular_max_probability_mean", "ocular_top1_top2_margin_mean", "ocular_entropy_mean"):
        assert compact[field] == full[field] if field.startswith("soft_") else np.isclose(compact[field], full[field])
    assert "whole_max_probability_p50" not in compact and "boundary_entropy_p95" not in compact


def test_runtime_cohort_requests_labels_and_class_probability_only():
    class Session:
        def run(self, names, feeds):
            assert tuple(names) == COHORT_OUTPUT_NAMES
            return [np.zeros((16, 400, 640), np.uint8), np.broadcast_to(_probability(), (16, 4, 400, 640)).copy()]
    runtime = RitnetFullClassFinalRuntime.__new__(RitnetFullClassFinalRuntime)
    runtime.session, runtime.input_name, runtime.precision, runtime.cohort_compact_outputs = Session(), "image", "fp32", True
    outputs, timing = runtime.infer_prepared(np.zeros((16, 1, 400, 640), np.float32), 1)
    assert timing["output_contract"] == "labels+class_probability-cohort"
    assert tuple(COHORT_OUTPUT_NAMES) == ("labels", "class_probability")
    assert tuple(OUTPUT_NAMES) == ("labels", "class_probability", "max_probability", "top1_top2_margin", "entropy")
    assert len(outputs["max_probability"]) == 1


def test_full_output_path_remains_available_for_sparse_qc():
    assert tuple(OUTPUT_NAMES)[2:] == ("max_probability", "top1_top2_margin", "entropy")


def test_cuda_contract_identity_and_pipeline_contracts():
    assert CORE_VERSION == "fullclass-final-core-v7-pupil-only-lean-schema"
    assert RitnetFullClassFinalRuntime.FIXED_BATCH_SIZE == 16
    assert RitnetFullClassFinalRuntime.__dict__["FIXED_BATCH_SIZE"] == 16


def test_compact_summary_preserves_failed_rows():
    labels = _labels()
    prepared = PreparedBatch(items=[{"ordinal": 0, "base": {}, "roi": np.zeros((400,640), np.uint8), "valid_source_mask": np.ones((400,640), bool)}], successful_indices=(0,), tensor=None, valid_batch_size=1, timing={})
    p = np.broadcast_to(_probability(), (1,4,400,640)).copy()
    outputs = {"labels": labels[None], "class_probability": p, "max_probability": _DerivedUncertaintyBatch(p, "max_probability"), "top1_top2_margin": _DerivedUncertaintyBatch(p, "top1_top2_margin"), "entropy": _DerivedUncertaintyBatch(p, "entropy")}
    rows, _ = _summarize_outputs(prepared=prepared, outputs=outputs, boundary_band_px=5, low_max_probability_threshold=None)
    assert np.isclose(rows[0][1]["soft_pupil_fraction"], 0.05)

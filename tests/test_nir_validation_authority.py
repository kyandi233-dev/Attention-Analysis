from __future__ import annotations

from attention_pipeline import nir_pipeline_validation as package
from attention_pipeline.nir_pipeline_validation.pupil_validation import run_validation as pupil_run_validation
from attention_pipeline.nir_pipeline_validation.run import run_validation as legacy_pir_run_validation


def test_package_run_validation_is_pupil_only_authority() -> None:
    assert package.run_validation is pupil_run_validation
    assert package.legacy_pir_run_validation is legacy_pir_run_validation
    assert package.run_validation is not package.legacy_pir_run_validation

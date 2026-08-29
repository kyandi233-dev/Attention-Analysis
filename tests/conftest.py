from pathlib import Path

import pytest

from attention_pipeline.config import load_config


@pytest.fixture(scope="session")
def project_root():
    return Path(__file__).resolve().parents[1]


def pytest_collection_modifyitems(items):
    """Classify environment-bound historical tests without hiding portable failures."""
    local_marker = pytest.mark.requires_local_raw_data
    legacy_backend_marker = pytest.mark.legacy_optional_backend
    removed_roi_backend_tests = {
        "test_roi_check_configs",
        "test_faceparts_roi_detects_two_eyes_sorted_by_x",
        "test_faceparts_roi_rejects_wrong_class_mapping",
        "test_faceparts_roi_returns_none_when_fewer_than_two_eyes",
    }
    for item in items:
        if "config" in getattr(item, "fixturenames", ()):
            item.add_marker(local_marker)
        if (
            item.path.name == "test_roi_backends.py"
            and item.name in removed_roi_backend_tests
        ):
            item.add_marker(legacy_backend_marker)
            item.add_marker(
                pytest.mark.skip(
                    reason=(
                        "historical optional ROI backend module was removed from the current "
                        "repository; retained test is provenance-only until deletion is authorized"
                    )
                )
            )


@pytest.fixture(scope="session")
def config(project_root):
    """Historical preexperiment config, available only on its original data environment."""
    cfg = load_config(project_root / "configs" / "preexperiment.yaml")
    raw_value = cfg.section("paths").get("raw_root")
    if raw_value in (None, ""):
        pytest.skip("legacy preexperiment config has no raw_root")
    raw_root = Path(str(raw_value)).expanduser()
    if not raw_root.exists():
        pytest.skip(
            "requires machine-local legacy preexperiment raw data; "
            f"configured raw_root is unavailable: {raw_root}"
        )
    return cfg

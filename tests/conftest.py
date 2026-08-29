from pathlib import Path

import pytest

from attention_pipeline.config import load_config


@pytest.fixture(scope="session")
def project_root():
    return Path(__file__).resolve().parents[1]


def pytest_collection_modifyitems(items):
    """Classify environment-bound/orphaned historical tests without hiding current failures.

    Current formal tests must remain runnable in a clean checkout. Historical tests
    whose implementation scripts are no longer present are retained for provenance,
    but are explicit skips rather than unexplained ModuleNotFound/FileNotFound errors.
    """
    local_marker = pytest.mark.requires_local_raw_data
    legacy_backend_marker = pytest.mark.legacy_optional_backend
    orphaned_files = {"test_roi_backends.py", "test_pupil_adapter.py"}
    orphaned_benchmark_tests = {
        "test_apply_params_targets_swirski_params_object",
        "test_canonical_axes_preserve_ellipse_orientation",
    }

    for item in items:
        if "config" in getattr(item, "fixturenames", ()):
            item.add_marker(local_marker)

        orphaned = item.path.name in orphaned_files or (
            item.path.name == "test_benchmark.py" and item.name in orphaned_benchmark_tests
        )
        if orphaned:
            item.add_marker(legacy_backend_marker)
            item.add_marker(
                pytest.mark.skip(
                    reason=(
                        "historical regression target is not present in the current repository; "
                        "test retained as provenance until explicit deletion/archival is authorized"
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

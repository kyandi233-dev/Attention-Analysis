from pathlib import Path

import pytest

from attention_pipeline.config import load_config


@pytest.fixture(scope="session")
def project_root():
    return Path(__file__).resolve().parents[1]


def pytest_collection_modifyitems(items):
    """Label tests that consume the historical preexperiment ``config`` fixture.

    The marker documents an environment dependency; the fixture below performs the
    actual skip only when the legacy raw root is unavailable. Portable formal tests
    that create their own Config objects are unaffected.
    """
    marker = pytest.mark.requires_local_raw_data
    for item in items:
        if "config" in getattr(item, "fixturenames", ()):
            item.add_marker(marker)


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

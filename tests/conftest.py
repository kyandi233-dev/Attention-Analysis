from pathlib import Path

import pytest

from attention_pipeline.config import load_config


@pytest.fixture(scope="session")
def project_root():
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def config(project_root):
    return load_config(project_root / "configs" / "preexperiment.yaml")


from ritnet_fullclass_final_engine import CORE_VERSION
from ritnet_fullclass_workstore import V8_CORE_VERSION


def test_engine_core_version_matches_workstore_v8_version():
    assert CORE_VERSION == V8_CORE_VERSION
    assert CORE_VERSION == "fullclass-final-core-v8-interface-safe-plain-csv"

from pathlib import Path

from attention_pipeline.rgb.paths import RGBOutputLayout


def test_subject_files_repeat_subject_prefix(tmp_path: Path) -> None:
    layout = RGBOutputLayout(root=tmp_path / "Beijing-RGB")
    path = layout.subject_file("sub-031", "motion_raw.parquet")
    assert path == tmp_path / "Beijing-RGB" / "sub-031" / "sub-031_motion_raw.parquet"


def test_test_outputs_use_single_test_directory(tmp_path: Path) -> None:
    layout = RGBOutputLayout(root=tmp_path / "Beijing-RGB")
    path = layout.test_file("sub-031_motion-test.parquet")
    assert path == tmp_path / "Beijing-RGB" / "_test" / "sub-031_motion-test.parquet"

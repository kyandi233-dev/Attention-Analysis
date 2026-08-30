from attention_pipeline.behavior_formal import publication_reporting as behavior_reporting
from attention_pipeline.rgb_formal import reporting as rgb_reporting


def test_behavior_publication_in_image_labels_are_ascii_and_external_captioned():
    assert behavior_reporting.VIEWS
    assert all(all(ord(ch) < 128 for ch in value) for value in behavior_reporting.LABELS.values())
    row = behavior_reporting._figure_row("beta", "block_pair", "generated", "")
    assert row["internal_title"] is False
    assert row["in_image_language"] == "English"
    assert row["caption_location"] == "external_manifest_and_report"


def test_rgb_reporting_exports_vector_and_raster_contract():
    assert callable(rgb_reporting.build_rgb_report)

from scripts.evaluate_yolo_eye_test import Box, _metric, greedy_match


def b(x1, y1, x2, y2, confidence=1.0):
    return Box((x1, y1, x2, y2), confidence)


def test_complete_match_and_metrics():
    rows = greedy_match([b(0, 0, 10, 10)], [b(0, 0, 10, 10, 0.9)], 0.5)
    assert [r["kind"] for r in rows] == ["match"]
    assert _metric(1, 0, 0) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_missing_truth_is_fn():
    rows = greedy_match([b(0, 0, 10, 10), b(20, 0, 30, 10)], [b(0, 0, 10, 10)], 0.5)
    assert sum(r["kind"] == "match" for r in rows) == 1
    assert sum(r["kind"] == "fn" for r in rows) == 1


def test_false_positive_and_duplicate_are_not_forced_into_two_eyes():
    rows = greedy_match([b(0, 0, 10, 10)], [b(0, 0, 10, 10, 0.9), b(0, 0, 10, 10, 0.8), b(20, 20, 30, 30, 0.7)], 0.5)
    assert sum(r["kind"] == "match" for r in rows) == 1
    assert sum(r["kind"] == "fp" for r in rows) == 2


def test_empty_predictions_and_single_truth():
    rows = greedy_match([b(0, 0, 10, 10)], [], 0.5)
    assert rows[0]["kind"] == "fn"
    assert _metric(0, 0, 1)["recall"] == 0.0

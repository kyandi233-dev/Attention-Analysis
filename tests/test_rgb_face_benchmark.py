from attention_pipeline.rgb.face_benchmark import _evenly_spaced_positions


def test_evenly_spaced_positions_are_deterministic_and_cover_edges():
    positions = list(range(10, 110))
    selected = _evenly_spaced_positions(positions, 5)
    assert selected == [10, 35, 60, 84, 109]
    assert selected[0] == positions[0]
    assert selected[-1] == positions[-1]


def test_evenly_spaced_positions_return_all_when_short():
    assert _evenly_spaced_positions([3, 7, 9], 10) == [3, 7, 9]
    assert _evenly_spaced_positions([], 10) == []

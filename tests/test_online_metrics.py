import pytest
from online_metrics import ScalarHistory, PathHistory


def test_cumulative_mean_survives_display_history_eviction():
    h = ScalarHistory(maxlen=8)
    for i in range(10000):
        h.append(i)
    assert len(h) == 8 and h.count == 10000
    assert h.mean == pytest.approx(4999.5)
    h.clear()
    assert h.mean == 0 and h.count == 0


def test_path_statistics_retain_original_origin_after_history_eviction():
    h = PathHistory(maxlen=8)
    for i in range(10000):
        h.append((i * .01, 0))
    assert len(h) == 8 and h.count == 10000
    assert h.distance == pytest.approx(99.99)
    assert h.tortuosity == pytest.approx(1)
    h.clear()
    h.append((500, 500))
    assert h.distance == 0 and h.tortuosity is None

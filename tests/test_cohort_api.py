"""Contract tests for the cohort interface: atomic rejection, read-only graph."""
import numpy as np
import pytest

from brainlab.cohort import CpuLoopCohortEngine


def _graph(n=60, k=6, seed=3):
    rng = np.random.default_rng(seed)
    post = np.concatenate([rng.choice(n, k, replace=False) for _ in range(n)]).astype(np.int32)
    ptr = np.arange(0, n * k + 1, k, dtype=np.int64)
    weight = rng.uniform(-1.0, 2.0, n * k).astype(np.float32)
    return dict(ptr=ptr, post=post, weight=np.ascontiguousarray(weight), ids=np.arange(n, dtype=np.int64))


def _drive(eng, value=30.0):
    d = np.zeros((eng.n_flies, eng.n), dtype=np.float32)
    d[:, :5] = value
    return d


def _same(a, b):
    return all(np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in a if k != "dynamics")


def test_graph_is_read_only_and_shared():
    arrays = _graph()
    eng = CpuLoopCohortEngine(arrays, 2)
    assert not arrays["weight"].flags.writeable
    with pytest.raises(ValueError):
        arrays["weight"][0] = 5.0


def test_nan_in_one_fly_advances_no_fly():
    eng = CpuLoopCohortEngine(_graph(), 2)
    before = [eng.read_state(k) for k in range(2)]
    d = _drive(eng)
    d[1, 0] = np.nan
    with pytest.raises(ValueError):
        eng.step(d, 10)
    assert all(_same(before[k], eng.read_state(k)) for k in range(2))


@pytest.mark.parametrize("ticks", [1.5, 0, -1, True, "3"])
def test_non_integer_ticks_refused(ticks):
    eng = CpuLoopCohortEngine(_graph(), 1)
    before = eng.read_state(0)
    with pytest.raises(ValueError):
        eng.step(_drive(eng), ticks)
    assert _same(before, eng.read_state(0))


def test_bad_state_rejected_before_any_live_change():
    eng = CpuLoopCohortEngine(_graph(), 1)
    eng.step(_drive(eng), 20)
    live = eng.read_state(0)
    bad = dict(live)
    bad["v"] = np.full_like(live["v"], -30.0)       # valid field, would change state
    bad["g"] = np.zeros((3,), dtype=live["g"].dtype)  # invalid shape
    with pytest.raises(ValueError):
        eng.write_state(0, bad)
    assert _same(live, eng.read_state(0))
    for mutate in (lambda s: s.pop("queue"), lambda s: s.update(dynamics="v2"),
                   lambda s: s.update(cursor=-1), lambda s: s.update(sim_ms=float("nan"))):
        s = dict(live)
        mutate(s)
        with pytest.raises(ValueError):
            eng.write_state(0, s)
        assert _same(live, eng.read_state(0))


def test_cpu_resume_is_exact():
    arrays = _graph()
    a = CpuLoopCohortEngine(arrays, 1)
    a.step(_drive(a), 30)
    mid = a.read_state(0)
    a.step(_drive(a), 30)
    b = CpuLoopCohortEngine(arrays, 1)
    b.write_state(0, mid)
    b.step(_drive(b), 30)
    assert _same(a.read_state(0), b.read_state(0))

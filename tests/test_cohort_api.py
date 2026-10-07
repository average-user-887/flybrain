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


def test_state_semantic_holes_refused():
    eng = CpuLoopCohortEngine(_graph(), 1)
    eng.step(_drive(eng), 20)
    live = eng.read_state(0)
    bad_queue = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in live.items()}
    bad_queue["queue_count"][0] = 1
    bad_queue["queue"][0, 0] = eng.n + 939
    extra = dict(live, surprise=1)
    clocks = dict(live, sim_ms=float(live["sim_ms"]) + 100.0)
    for bad in (bad_queue, extra, clocks):
        with pytest.raises(ValueError):
            eng.write_state(0, bad)
        assert _same(live, eng.read_state(0))
    eng.write_state(0, live)   # the genuine state is still accepted


def _tiny_two_cell_engine():
    arrays = dict(ids=np.arange(2, dtype=np.int64), ptr=np.array([0, 1, 1], np.int64),
                  post=np.array([1], np.int32), weight=np.array([1], np.float32))
    return CpuLoopCohortEngine(arrays, 1)


def test_contradictory_active_state_is_refused_atomically():
    """A1 (Astra's exact control): rc1 accepted active_flag[0]=1 with nactive=0, which
    silently stopped a driven neuron from ever being activated."""
    fresh = _tiny_two_cell_engine()
    control = _tiny_two_cell_engine()
    out = control.step(np.array([[20., 0.]], np.float32), 200)
    assert int(out.sum()) == 2
    live = fresh.read_state(0)
    bad = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in live.items()}
    bad['active_flag'][0] = 1
    with pytest.raises(ValueError, match='active list and active_flag disagree'):
        fresh.write_state(0, bad)
    assert _same(live, fresh.read_state(0))


def test_duplicate_active_entries_and_negative_refractory_refused():
    eng = CpuLoopCohortEngine(_graph(), 1)
    eng.step(_drive(eng), 20)
    live = eng.read_state(0)

    def copy():
        return {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in live.items()}
    assert int(live['nactive'][0]) >= 2
    dup = copy()
    dup['active'][1] = dup['active'][0]
    unlisted = copy()
    unlisted['nactive'][0] -= 1
    refr = copy()
    refr['refractory'][0] = -1
    for bad, msg in ((dup, 'disagree'), (unlisted, 'disagree'), (refr, 'refractory')):
        with pytest.raises(ValueError, match=msg):
            eng.write_state(0, bad)
        assert _same(live, eng.read_state(0))
    eng.write_state(0, live)

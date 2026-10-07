"""v6 is additive: with no per-type parameters it is v4, bit for bit."""
import numpy as np

from brainlab.brain import Brain
from brainlab.v6_calibrated import BrainV6, Phototransduction
from tests.test_graded_v4 import random_graph


def _run(b, drive, ms=60):
    out = []
    for _ in range(ms):
        b.step(drive, 1.0)
        out.append(b.v.copy())
    return np.array(out)


def test_empty_v6_is_bit_identical_to_v4():
    arrays = random_graph()
    n = len(arrays['ids'])
    rng = np.random.default_rng(5)
    mask = (rng.random(n) < 0.5).astype(np.uint8)
    drive = np.zeros(n, np.float32)
    drive[rng.choice(n, n // 10, replace=False)] = 25.0
    a = Brain(arrays=dict(arrays), dynamics='v4', backend='cpu', graded_policy=mask)
    b = BrainV6(None, {'types': {}}, 'x' * 64, cell_type=np.array(['X'] * n), arrays=dict(arrays),
                graded_policy=mask)
    assert np.array_equal(_run(a, drive), _run(b, drive))


def test_sigmoid_release_and_light_change_the_result():
    arrays = random_graph()
    n = len(arrays['ids'])
    mask = np.ones(n, np.uint8)
    ct = np.array(['R1-R6'] * 20 + ['X'] * (n - 20))
    p = {'phototransduction': dict(G=5.0, Ka=0.5, tau_a_ms=200, tau_p0_ms=8, n_stages=4, dead_time_ms=10,
                                   encoder_to_intensity=0.01),
         'types': {'R1-R6': {'release': {'kind': 'sigmoid', 'gain': 2.0, 'vh_mV': -40.0, 's_mV': 4.0}}}}
    b = BrainV6(None, p, 'y' * 64, light_nodes=np.arange(20), cell_type=ct, arrays=dict(arrays), graded_policy=mask)
    drive = np.zeros(n, np.float32); drive[:20] = 100.0
    v = _run(b, drive, 80)
    assert v[-1, :20].mean() > -40.0          # light depolarises R
    s = b.snapshot_state(); x = _run(b, drive, 5); b.restore_state(s); y = _run(b, drive, 5)
    assert np.array_equal(x, y)


def test_phototransduction_adapts():
    pt = Phototransduction(dict(G=5.0, Ka=0.5, tau_a_ms=200, tau_p0_ms=8, n_stages=4, dead_time_ms=10), 1)
    g = [pt.step(np.array([1.0]))[0] for _ in range(2000)]
    assert max(g) > 1.5 * g[-1] > 0

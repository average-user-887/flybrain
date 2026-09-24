import os

import numpy as np
import pytest


@pytest.mark.skipif(
    os.environ.get("NEUROFLY_RUN_PHYSICS") != "1",
    reason="set NEUROFLY_RUN_PHYSICS=1 in the FlyGym 2.1 environment",
)
@pytest.mark.parametrize("use_numba", [True, False])
@pytest.mark.parametrize("retraction_height_threshold", [None, 0.0])
def test_fast_loop_is_bit_identical_to_stock_controller(retraction_height_threshold, use_numba):
    pytest.importorskip("flygym")
    from neurofly_body.flygym_body import FlyGymBody

    stock = FlyGymBody(physics_dt_s=0.0001, warmup_s=0.01, fast_loop=False)
    fast = FlyGymBody(physics_dt_s=0.0001, warmup_s=0.01, fast_loop=True)
    fast._fast_loop.use_numba = use_numba
    commands = [(1.0, 1.0), (0.2, 1.2), (1.2, 0.0), (0.0, 0.0), (0.8, 0.6)]
    corrected = set()
    if retraction_height_threshold is not None:
        # Flat ground rarely triggers the retraction rule; lower its threshold
        # on both controllers so that branch is compared too.
        for body in (stock, fast):
            body.controller.retraction_height_threshold = retraction_height_threshold
    try:
        assert stock.describe()["controller_loop"] == "stock"
        assert fast.describe()["controller_loop"] == "fast"
        assert stock.reset(seed=3) == fast.reset(seed=3)
        for index in range(150):  # 0.3 s of body time in 2 ms steps
            command = commands[(index // 30) % len(commands)]
            expected = stock.step(command, substeps=20)
            actual = fast.step(command, substeps=20)
            assert actual == expected, f"observation diverged at step {index}"
            for name in ("qpos", "qvel", "ctrl", "act"):
                assert np.array_equal(
                    getattr(stock.sim.mj_data, name), getattr(fast.sim.mj_data, name)
                ), f"mj_data.{name} diverged at step {index}"
            for name in ("retraction_correction", "stumbling_correction", "retraction_persistence_counter"):
                assert np.array_equal(getattr(stock.controller, name), getattr(fast.controller, name))
            info = stock.controller.last_info
            if info["leg_to_correct_retraction"] is not None:
                corrected.add("retraction")
            if info["stumbling_mask"].any():
                corrected.add("stumbling")
            assert info["leg_to_correct_retraction"] == fast.controller.last_info["leg_to_correct_retraction"]
        # A second episode with the command still cached from the first one:
        # reset() must not leave stale CPG drive in the fast loop.
        assert stock.reset(seed=11) == fast.reset(seed=11)
        for index in range(20):
            assert fast.step(commands[-1], substeps=20) == stock.step(commands[-1], substeps=20)
    finally:
        stock.close()
        fast.close()
    # The comparison is only meaningful if the correction branches actually ran.
    assert "stumbling" in corrected
    if retraction_height_threshold is not None:
        assert "retraction" in corrected


@pytest.mark.skipif(
    os.environ.get("NEUROFLY_RUN_PHYSICS") != "1",
    reason="set NEUROFLY_RUN_PHYSICS=1 in the FlyGym 2.1 environment",
)
def test_vectorised_spline_and_interp_match_scipy_and_numpy_bitwise():
    pytest.importorskip("flygym")
    from flygym_demo.complex_terrain import PreprogrammedSteps

    from neurofly_body.fast_controller import _interp_rows, _PeriodicCubicLegs

    steps = PreprogrammedSteps()
    splines = [steps._psi_funcs[leg] for leg in steps.legs]
    legs = _PeriodicCubicLegs(splines)
    rng = np.random.default_rng(0)
    grid = splines[0].x
    samples = [rng.uniform(-40.0, 200.0, 6) for _ in range(3000)]
    samples += [np.full(6, value) for value in np.concatenate([grid, grid + 2 * np.pi, [2 * np.pi, -1e-12]])]
    for phases in samples:
        expected = np.stack([s(np.asarray(p)[np.newaxis])[:, 0] for s, p in zip(splines, phases)])
        assert np.array_equal(legs(phases), expected)

    xp = np.stack([np.array([0.0, a / 2, a + np.pi / 4, (a + 2 * np.pi) / 2, 2 * np.pi]) for a in (2.37, 2.23, 1.95)])
    fp = np.array([0.0, 0.8, 0.0, -0.1, 0.0])
    for _ in range(3000):
        x = rng.uniform(-1.0, 7.0, 3)
        if rng.random() < 0.3:
            x = xp[np.arange(3), rng.integers(0, 5, 3)].copy()
        expected = np.array([np.interp(x[row], xp[row], fp) for row in range(3)])
        assert np.array_equal(_interp_rows(x, xp, fp), expected)

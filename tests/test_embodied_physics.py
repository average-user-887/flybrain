import os

import numpy as np
import pytest


@pytest.mark.skipif(
    os.environ.get("NEUROFLY_RUN_PHYSICS") != "1",
    reason="set NEUROFLY_RUN_PHYSICS=1 in the FlyGym 2.1 environment",
)
def test_flygym_body_advances_articulated_state():
    pytest.importorskip("flygym")
    from neurofly_body.flygym_body import FlyGymBody

    body = FlyGymBody(physics_dt_s=0.0001, warmup_s=0.001)
    try:
        initial = body.reset(seed=0)
        final = body.step((0.0, 0.8), substeps=20)
        assert final["body_sim_time_s"] > initial["body_sim_time_s"]
        assert len(final["joint_angles_rad"]) > 6
        assert np.isfinite(final["joint_angles_rad"]).all()
        assert len(final["contacts"]["found"]) == 6
    finally:
        body.close()


@pytest.mark.skipif(
    os.environ.get("NEUROFLY_RUN_PHYSICS") != "1",
    reason="set NEUROFLY_RUN_PHYSICS=1 in the FlyGym 2.1 environment",
)
def test_flygym_corrections_and_phases_are_observable_at_zero_command():
    """D3: the stock corrections and adhesion phases stay active at zero drive."""
    pytest.importorskip("flygym")
    from neurofly_body.flygym_body import FlyGymBody

    body = FlyGymBody(physics_dt_s=0.0001, warmup_s=0.001)
    try:
        initial = body.reset(seed=0)
        # Before the first controller substep nothing has been computed yet.
        assert initial["flygym_corrections"]["stepped"] is False

        zero = body.step((0.0, 0.0), substeps=200)
        corrections = zero["flygym_corrections"]
        assert corrections["stepped"] is True
        for key in (
            "net_corrections",
            "retraction_correction",
            "stumbling_correction",
        ):
            assert len(corrections[key]) == 6
            assert np.isfinite(corrections[key]).all()
        assert len(corrections["stumbling_mask"]) == 6
        assert all(isinstance(flag, bool) for flag in corrections["stumbling_mask"])
        leg = corrections["leg_to_correct_retraction"]
        assert leg is None or 0 <= leg < 6

        # The CPG magnitude is zero but its phase keeps advancing, which is what
        # keeps switching adhesion in the output-disconnected control.
        assert len(zero["cpg_phases_rad"]) == 6
        assert np.isfinite(zero["cpg_phases_rad"]).all()
        assert max(zero["cpg_magnitudes"]) == 0.0
        assert zero["cpg_phases_rad"] != initial["cpg_phases_rad"]

        described = body.describe()["always_active_flygym_machinery"]
        assert any("output-disconnected" in line for line in described)
    finally:
        body.close()

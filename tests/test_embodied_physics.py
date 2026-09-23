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

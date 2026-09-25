"""One steering sign convention across the hybrid bridge, the co-sim server and the daemon.

Arena frame: heading is measured counter-clockwise, the fly moves along
(cos h, sin h), and yaw_rate > 0 is a counter-clockwise (left) turn.  DNa02
activation predicts ipsilateral turning, so the steering signal is
dna02_diff = rate(DNa02_L) - rate(DNa02_R), the same as the daemon's
``yaw = 0.02 * (L - R)`` and neurofly_body's crossed CPG decoder.
"""

import math

import numpy as np
import pytest

from arena import FlyState
from brainlab.cosim_server import ConnectomeServer
from connectome_bridge import ConnectomeBridge

_STILL = dict(fly_pos=np.array([50.0, 50.0]), fly_heading=0.0, fly_speed=0.0, fly_yaw_rate=0.0)


def _step(bridge, **kwargs):
    args = dict(_STILL, odor_left=0.0, odor_right=0.0, wind_vector=np.zeros(2), dt=0.02,
                is_saccade=False)
    args.update(kwargs)
    return bridge.step(**args)


def test_arena_frame_left_is_counter_clockwise():
    fly = FlyState(0.0, 0.0, heading=0.0)
    left, right = fly.get_antennae_positions()
    assert left.y > 0 > right.y  # left antenna at +45 deg, right at -45 deg
    fly.update(dheading=0.5, dt=1.0)
    assert 0 < fly.heading < math.pi  # a positive yaw rate turns toward the left antenna


def test_surrogate_turns_toward_odor_on_the_right():
    bridge = ConnectomeBridge(seed=0)
    out = _step(bridge, odor_left=0.1, odor_right=0.9)
    assert bridge.dna02_rate_r > bridge.dna02_rate_l  # ipsilateral DNa02 recruited
    assert out["dna02_diff"] < 0.0
    assert out["yaw_rate"] < 0.0  # clockwise: toward the right antenna


def test_surrogate_turns_toward_odor_on_the_left():
    bridge = ConnectomeBridge(seed=0)
    out = _step(bridge, odor_left=0.9, odor_right=0.1)
    assert bridge.dna02_rate_l > bridge.dna02_rate_r
    assert out["dna02_diff"] > 0.0 and out["yaw_rate"] > 0.0


def test_surrogate_anemotaxis_turns_upwind():
    # Wind blowing toward -y comes FROM +y, i.e. from the fly's left at heading 0.
    bridge = ConnectomeBridge(seed=0)
    out = _step(bridge, odor_left=0.5, odor_right=0.5, wind_vector=np.array([0.0, -30.0]))
    assert out["wpn_wind_heading"] > 0.0
    assert out["yaw_rate"] > 0.0  # turns left, into the wind


def test_surrogate_optomotor_opposes_self_rotation():
    bridge = ConnectomeBridge(seed=0)
    out = _step(bridge, fly_yaw_rate=1.0)
    assert out["yaw_rate"] < 0.0
    bridge = ConnectomeBridge(seed=0)
    out = _step(bridge, fly_yaw_rate=-1.0)
    assert out["yaw_rate"] > 0.0


class _GraphReply:
    """RPC client stand-in: the graph fires DNa02_L only."""

    is_connected = True
    last_error = None

    def __init__(self, stale_diff):
        self.stale_diff = stale_diff

    def step(self, sensory, duration_ms=2.0):
        return dict(status="ok", dna02_diff=self.stale_diff, dna02_rate_l=40.0, dna02_rate_r=0.0,
                    dnp09_rate=0.0, bpn_rate=20.0, mdn_rate=0.0, dnp01_gf_spikes=0)


@pytest.mark.parametrize("stale_diff", [40.0, -40.0])
def test_rpc_left_dna02_turns_left_whatever_the_server_diff_field(stale_diff):
    bridge = ConnectomeBridge(mode="rpc", rpc_client=_GraphReply(stale_diff), seed=0)
    out = _step(bridge)
    assert out["motor_source"] == "graph-rpc"
    assert out["dna02_diff"] == 40.0
    assert out["yaw_rate"] > 0.0


def test_cosim_server_diff_is_left_minus_right(tmp_path):
    server = ConnectomeServer(graph_dir=tmp_path, allow_synthetic=True)
    left = [i for i in server.dn_indices["dna02_l"] if i < server.n_neurons]
    assert left, "synthetic graph must map DNa02_L"

    def fake_step(currents, duration_ms):
        counts = np.zeros(server.n_neurons, dtype=np.int64)
        counts[left] = 1
        return counts, 0.0

    server.brain.step = fake_step
    reply = server.step({}, duration_ms=2.0)
    assert reply["dna02_rate_l"] > 0.0 == reply["dna02_rate_r"]
    assert reply["dna02_diff"] == reply["dna02_rate_l"] - reply["dna02_rate_r"]
    # Same sign as the daemon's decoded yaw, 0.02 * (L - R), + = counter-clockwise.
    assert math.copysign(1.0, reply["dna02_diff"]) == math.copysign(1.0, 0.02 * (reply["dna02_rate_l"] - reply["dna02_rate_r"]))

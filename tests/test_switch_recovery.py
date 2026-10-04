"""Assay switching never leaves the live loop on a released instance.

Live defect, 2026-10-04 browser sign-off (docs/receipts/live-signoff-20261004-run2):
the observatory daemon froze on the SECOND visit to the optomotor assay in one
process.  The graph controller cached its WP5 ``OptomotorLoop`` keyed only by
``instance_id``.  Leaving optomotor checkpoints and releases that GraphInstance;
coming back activates a NEW GraphInstance object with the SAME id, so the cache
hit returned a loop still bound to the released object, whose ``step`` raises
"is not active; inactive instances never step".  ``step_once`` latched that error
and nothing advanced again until a restart, while switches still answered "ok"
and the dashboard pill still said LIVE DAEMON.

Runs on the synthetic test graph with the stub WP5 map used by
tests/test_wp5_live_loop.py; no 600 MB graph, no behavioural claim.
"""

from neurofly_daemon import NeuroflyHTTPHandler
from tests.test_wp5_live_loop import graph_runner, synthetic_io_map


def _switch(runner, paradigm):
    return runner._apply_command({"action": "switch_paradigm", "paradigm": paradigm})


def _status(runner):
    class Handler:
        gateway = type("Gateway", (), {"describe": lambda self: {}})()
    handler = Handler()
    handler.runner = runner
    return NeuroflyHTTPHandler._status_payload(handler)


def test_returning_to_optomotor_steps_the_reactivated_instance(tmp_path):
    runner = graph_runner(tmp_path)
    runner.graph_controller._optomotor_io = synthetic_io_map()
    with runner.lock:
        runner.step_once()                       # builds the WP5 loop for instance object A
        first = runner.registry.active
        assert _switch(runner, "t-maze")["status"] == "ok"   # A checkpointed and released
        runner.step_once()
        assert _switch(runner, "optomotor")["status"] == "ok"  # new object B, same instance_id
        second = runner.registry.active
        assert second is not first and second.instance_id == first.instance_id
        before = runner.total_steps
        for _ in range(3):
            runner.step_once()
    assert runner.last_error is None
    assert runner.total_steps == before + 3
    held_instance, loop = runner.graph_controller._optomotor_loop
    assert held_instance is second
    assert loop.instance is second
    assert runner.arena.fly.last_connectome_telemetry["optomotor"] is not None


def test_repeated_round_trips_through_optomotor_never_halt(tmp_path):
    runner = graph_runner(tmp_path)
    runner.graph_controller._optomotor_io = synthetic_io_map()
    with runner.lock:
        for target in ["optomotor", "looming-escape", "optomotor", "t-maze", "optomotor", "optomotor"] * 3:
            assert _switch(runner, target)["status"] == "ok"
            before = runner.total_steps
            runner.step_once()
            assert runner.last_error is None, target
            assert runner.total_steps == before + 1, target

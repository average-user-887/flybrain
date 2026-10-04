"""Assay switching never leaves the live loop on a released instance, and a halted
simulation is reported honestly and recovers on a rebuild.

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
import pytest

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


class _Boom(RuntimeError):
    pass


def _fail_next_step(runner, message="injected step failure"):
    original = runner.arena.step

    def failing(dt):
        runner.arena.step = original
        raise _Boom(message)
    runner.arena.step = failing


def test_step_error_halts_and_is_reported_honestly(tmp_path):
    runner = graph_runner(tmp_path, paradigm="t-maze")
    with runner.lock:
        runner.step_once()
        _fail_next_step(runner)
        steps = runner.total_steps
        runner.step_once()
        runner.step_once()                       # fail-safe: still halted, nothing advances
    assert runner.total_steps == steps
    assert runner.last_error == "injected step failure"
    assert runner._can_step() is False
    detail = runner.error_detail
    assert detail["message"] == "injected step failure" and detail["type"] == "_Boom"
    assert detail["paradigm"] == "t-maze" and detail["step"] == steps
    assert detail["instance_id"] == runner.registry.active.instance_id

    status = _status(runner)
    assert status["status"] == "error" and status["halted"] is True and status["paused"] is False
    assert status["error_detail"]["message"] == "injected step failure"
    assert "select" in status["error_detail"]["recover"].lower()

    with runner.lock:
        packet = runner._assemble_telemetry({})
    assert packet["error"] == "injected step failure" and packet["halted"] is True
    assert packet["timing"]["achieved_speed"] == 0.0

    # A command that does not rebuild anything is applied but says the run stays halted.
    with runner.lock:
        ack = runner._apply_command({"action": "set_speed", "speed": 5})
    assert ack["status"] == "ok"
    assert ack["ack"]["halted_by_error"] == "injected step failure"
    assert runner.last_error == "injected step failure"


@pytest.mark.parametrize("target", ["y-maze", "t-maze"])
def test_successful_switch_clears_the_halt_and_steps_resume(tmp_path, target):
    runner = graph_runner(tmp_path, paradigm="t-maze")
    with runner.lock:
        runner.step_once()
        _fail_next_step(runner)
        runner.step_once()
        steps = runner.total_steps
        result = _switch(runner, target)
    assert result["status"] == "ok"
    assert result["cleared_error"]["message"] == "injected step failure"
    assert result["ack"]["halted_by_error"] is None
    assert runner.last_error is None and runner.error_detail is None
    assert _status(runner)["status"] == "online" and _status(runner)["halted"] is False
    # The cleared error is not forgotten.
    assert runner.cleared_errors[-1]["message"] == "injected step failure"
    assert runner.cleared_errors[-1]["cleared_by"] == "switch_paradigm"
    with runner.lock:
        runner.step_once()
    assert runner.total_steps == steps + 1


def test_failed_switch_keeps_the_halt(tmp_path):
    runner = graph_runner(tmp_path, paradigm="t-maze")
    with runner.lock:
        _fail_next_step(runner)
        runner.step_once()
        result = _switch(runner, "no-such-assay")
    assert result["status"] == "error"
    assert runner.last_error == "injected step failure"
    assert _status(runner)["status"] == "error"


def test_backend_switch_clears_the_halt(tmp_path):
    runner = graph_runner(tmp_path, paradigm="t-maze")
    with runner.lock:
        _fail_next_step(runner)
        runner.step_once()
        result = runner._apply_command({"action": "switch_backend", "backend": "modular"})
    assert result["status"] == "ok", result
    assert runner.last_error is None
    assert runner.cleared_errors[-1]["cleared_by"] == "switch_backend"

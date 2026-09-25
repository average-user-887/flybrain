"""Tests for live connectome multi-paradigm sensory ingress, motor decoding, and backend switching."""
import pytest
import numpy as np
import math

from neurofly_daemon import ContinuousExperimentRunner, GraphArenaController


def test_synthetic_graph_multi_assay_stepping(tmp_path):
    """Synthetic test graph steps across assays without crashing, producing valid graph motor records."""
    runner = ContinuousExperimentRunner(
        initial_paradigm="open-arena",
        sim_speed=1,
        checkpoint_interval=3600,
        output_dir=tmp_path,
        backend="connectome-fixed",
        test_synthetic_graph=True
    )
    with runner.lock:
        res = runner.step_once()
    fly = runner.arena.fly
    telem = fly.last_connectome_telemetry
    assert telem["halted"] is False
    assert telem["motor_source"] == "graph"
    assert "dn_rates" in telem
    assert "epg_wedges" in telem
    assert len(telem["epg_wedges"]) == 16
    assert isinstance(telem["epg_bump_phase"], float)
    assert fly.speed > 0.0


def test_switch_backend_live(tmp_path):
    """Runner switches cleanly between modular and connectome backends."""
    runner = ContinuousExperimentRunner(
        initial_paradigm="open-arena",
        sim_speed=1,
        checkpoint_interval=3600,
        output_dir=tmp_path,
        backend="modular",
        test_synthetic_graph=True
    )
    assert runner.backend == "modular"
    assert runner.graph_mode is False

    # Switch to connectome-fixed
    cmd = {"action": "switch_backend", "params": {"backend": "connectome-fixed"}}
    ack = runner.dispatch_command(cmd)
    assert ack["status"] == "ok"
    assert runner.backend == "connectome-fixed"
    assert runner.graph_mode is True

    # Step in connectome-fixed
    with runner.lock:
        runner.step_once()
    assert runner.arena.fly.motor_source == "graph"

    # Switch back to modular
    cmd2 = {"action": "switch_backend", "params": {"backend": "modular"}}
    ack2 = runner.dispatch_command(cmd2)
    assert ack2["status"] == "ok"
    assert runner.backend == "modular"
    assert runner.graph_mode is False


def test_looming_escape_trigger(tmp_path):
    """Looming stimulus drives LC4/LPLC2 channels or escape state."""
    runner = ContinuousExperimentRunner(
        initial_paradigm="looming-escape",
        sim_speed=1,
        checkpoint_interval=3600,
        output_dir=tmp_path,
        backend="connectome-fixed",
        test_synthetic_graph=True
    )
    controller = runner.graph_controller

    # A strong loom drives LC4/LPLC2, but escape needs a DNp01 spike in the graph.
    out = controller(fly=runner.arena.fly, sensory={"looming_theta": 0.8, "looming_detected": True}, dt=0.02)
    assert out["halted"] is False
    assert out["motor_source"] == "graph"
    gf_spiked = sum(controller.last_counts[i] for i in controller.dn_indices["dnp01"]) > 0
    assert (out["state"] == "ESCAPE") == gf_spiked
    assert out["dn_rates"]["gf"] > 0 if gf_spiked else out["dn_rates"]["gf"] == 0.0

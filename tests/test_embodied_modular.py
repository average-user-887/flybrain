"""Modular (non-connectome) baseline controller for the embodied loop."""
import json

import pytest

from neurofly_body import cli
from neurofly_body.modular import ModularCommandDecoder, ModularOptomotorBackend


def _drive(backend, slip, steps=200):
    for _ in range(steps):
        reply = backend.step({"optomotor_slip_rad_s": slip, "optomotor_contrast": 1.0}, 2.0)
    return reply


def test_no_slip_walks_straight_with_tonic_drive():
    reply = _drive(ModularOptomotorBackend(), 0.0)
    assert reply["modular_command"] == {"left": 1.0, "right": 1.0}


def test_follows_the_drum_direction():
    # slip > 0: world turns counter-clockwise; the fly follows by shrinking its left legs.
    reply = _drive(ModularOptomotorBackend(), 4.0)
    assert reply["yaw_bias"] > 0 and reply["modular_command"]["left"] < reply["modular_command"]["right"] == 1.0
    reply = _drive(ModularOptomotorBackend(), -4.0)
    assert reply["modular_command"]["right"] < reply["modular_command"]["left"] == 1.0


def test_zero_contrast_gives_no_turn():
    backend = ModularOptomotorBackend()
    for _ in range(100):
        reply = backend.step({"optomotor_slip_rad_s": 4.0, "optomotor_contrast": 0.0}, 2.0)
    assert reply["modular_command"] == {"left": 1.0, "right": 1.0}


def test_reset_replays_exactly():
    backend = ModularOptomotorBackend()
    first = [backend.step({"optomotor_slip_rad_s": 2.0}, 2.0) for _ in range(20)]
    backend.reset()
    assert [backend.step({"optomotor_slip_rad_s": 2.0}, 2.0) for _ in range(20)] == first


def test_decoder_clips_and_labels():
    decoder = ModularCommandDecoder()
    out = decoder.decode_reply({"modular_command": {"left": 5.0, "right": -5.0}}, 2.0)
    assert (out["left_cpg_drive"], out["right_cpg_drive"]) == (1.2, -1.2)
    assert "no connectome" in decoder.describe()["classification"]


pytest.importorskip("flygym")


def test_modular_fly_walks_and_turns_with_the_drum(tmp_path):
    out = tmp_path / "run"
    assert cli.main(["run", "--controller", "modular", "--duration", "0.3", "--output", str(out),
                     "--seed", "1"]) == 0
    summary = json.loads((out / "summary.json").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["neural_backend"]["controller_kind"] == "modular-baseline"
    assert manifest["invocation"]["controller"] == "modular"
    assert summary["final_thorax"]["yaw_rad"] > 0.2     # counter-clockwise, with the 4 rad/s drum
    assert cli.main(["replay-check", str(out), "--output", str(tmp_path / "replay")]) == 0

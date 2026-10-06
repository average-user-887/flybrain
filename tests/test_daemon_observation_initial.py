import copy
import json

import pytest

from experiment_brains import PARADIGMS
from neurofly_daemon import ContinuousExperimentRunner
from observation_envelopes import validate_observation_envelope
from online_metrics import ConfigError, SCHEMA


def runner(tmp_path, paradigm, **kwargs):
    return ContinuousExperimentRunner(
        initial_paradigm=paradigm,
        output_dir=tmp_path / paradigm,
        checkpoint_interval=3600,
        **kwargs,
    )


@pytest.mark.parametrize("paradigm", PARADIGMS)
def test_all_14_initial_packets_have_actual_owner_identity_and_spec_policy(tmp_path, paradigm):
    active = runner(tmp_path, paradigm)
    packet = active.latest_telemetry
    owner = active.arena.observation_owner

    assert owner is not None
    assert packet["observation"]["schema"] == SCHEMA
    assert validate_observation_envelope(packet["observation"]) == packet["observation"]
    assert packet["observation"]["identity"] == {
        **active.identity(), "brain_id": active.active_brain.brain_id,
    }
    assert packet["observation"]["segment_id"] == active.segment_id
    assert packet["observation"]["segment_start_sim_s"] == 0.0
    assert packet["observation"]["config_id"] == f"segment:{active.segment_id}"
    assert packet["observation_lifecycle"]["producer"] == owner.observation_status()
    assert packet["observation_lifecycle"]["requested_policy"]["trial_seconds"] is None
    assert packet["observation_publication"]["last_terminal"] is None

    detached = copy.deepcopy(packet["observation"])
    detached["identity"]["run_id"] = "changed"
    assert active._live_observation()["identity"]["run_id"] == active.identity()["run_id"]
    json.dumps(packet, allow_nan=False)


def test_omitted_explicit_and_continuous_window_provenance(tmp_path):
    omitted = runner(tmp_path, "t-maze")
    assert omitted.observation_config["window_source"] == "spec"
    assert omitted.observation_config["override"] is None
    assert omitted.trial_length_s is None

    explicit = runner(tmp_path, "y-maze", trial_length_s=7.5)
    assert explicit.observation_config["effective_window_s"] == 7.5
    assert explicit.observation_config["override"]["source"] == "cli:--trial-seconds"
    assert explicit.latest_telemetry["observation_lifecycle"]["requested_policy"] == {
        "continuous": False, "trial_seconds": 7.5, "trial_seconds_applied": True,
    }

    continuous = runner(tmp_path, "heat-maze", trial_length_s=7.5, continuous=True)
    assert continuous.observation_config["automatic_end"] is False
    assert continuous.observation_config["override"] == {
        "source": "cli:--continuous", "value": None, "set_at_sim_s": 0.0,
        "manifest_run_id": continuous.manifest.run_id,
    }
    assert continuous.latest_telemetry["observation_lifecycle"]["requested_policy"] == {
        "continuous": True, "trial_seconds": 7.5, "trial_seconds_applied": False,
    }


@pytest.mark.parametrize("value", [0, -1, 1e-9, 4.9e-7, float("nan"), float("inf"),
                                    float.fromhex("0x1.fffffffffffffp+1023"), True, "5"])
def test_bad_window_is_rejected_before_output_mutation(tmp_path, value):
    output = tmp_path / "must-not-exist"
    with pytest.raises(ConfigError, match="trial-seconds"):
        ContinuousExperimentRunner(output_dir=output, trial_length_s=value)
    assert not output.exists()


def test_live_validity_never_promotes_an_incomplete_current_run(tmp_path):
    scientific = runner(tmp_path, "t-maze")
    scientific.incidents.append({
        "run_id": scientific._current_run_id(), "reason": "fixture", "at": 1.0,
    })
    assert scientific.result_validity()["state"] == "incomplete"
    assert scientific._live_observation()["validity"] == "invalidated"

    exploratory = runner(tmp_path, "y-maze", exploratory=True)
    exploratory.incidents.append({
        "run_id": exploratory._current_run_id(), "reason": "fixture", "at": 1.0,
    })
    assert exploratory.result_validity()["state"] == "incomplete"
    assert exploratory._live_observation()["validity"] == "exploratory_degraded"


def test_first_step_uses_producer_status_and_legacy_flags_do_not_reset_it(tmp_path):
    active = runner(tmp_path, "labyrinth", trial_length_s=0.025)
    segment = active.segment_id
    active.arena.fly.pos.x, active.arena.fly.pos.y = 130.0, 85.0
    active.step_once()

    assert active.segment_id == segment
    assert active.current_trial == 1
    assert active.trial_history == []
    assert active.latest_telemetry["observation_lifecycle"]["producer"] == \
        active.arena.observation_owner.observation_status()
    assert active.latest_telemetry["observation"]["presentation_elapsed_s"] == 0.02

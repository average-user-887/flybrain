"""The verdict mode: the stimulus-sign reversal and its rate-matched controls."""

import json
import math

import pytest

from neurofly_body import verdict as V
from neurofly_body.decoder import DNa02CPGDecoder
from neurofly_body.runner import EmbodiedConfig, run_embodied
from tests.test_embodied_runner import FakeBody, FakeGraph


def _summary(spikes_l, spikes_r, yaw, x=0.0, y=0.0):
    return {
        "total_dna02_spikes_l": spikes_l,
        "total_dna02_spikes_r": spikes_r,
        "dna02_spike_count": spikes_l + spikes_r,
        "dna02_spike_count_warning": None,
        "mean_applied_drive_l": 0.1,
        "mean_applied_drive_r": 0.2,
        "steps_with_applied_drive_above_1pct_of_max": 100,
        "decoder_produced_output": True,
        "motor_output_reached_body": True,
        "trajectory_sha256": "d" * 64,
        "records": 2500,
        "wall_time_s": 31.0,
        "final_thorax": {"position_mm": [x, y, 0.9], "yaw_rad": yaw},
    }


def _outcome(name, seed, omega, spikes_l, spikes_r, yaw, x=3.0, y=4.0, cumulative=None):
    """An Outcome without touching disk; `cumulative` defaults to the wrapped yaw."""
    return V.Outcome(
        condition=V.Condition(name, seed, omega, "reversal"),
        summary=_summary(spikes_l, spikes_r, yaw, x, y),
        final_yaw_rad=yaw,
        cumulative_yaw_rad=yaw if cumulative is None else cumulative,
        planar_displacement_mm=math.hypot(x, y),
        extras={"run_dir": name},
    )


def _reversal_outcomes(yaw_positive=2.6, yaw_negative=-1.2):
    return {
        "seed0-positive-w": _outcome("seed0-positive-w", 0, 4.0, 30, 3, yaw_positive),
        "seed0-negative-w": _outcome("seed0-negative-w", 0, -4.0, 1, 19, yaw_negative),
        "seed1-positive-w": _outcome("seed1-positive-w", 1, 4.0, 26, 0, 2.117),
        "seed1-negative-w": _outcome("seed1-negative-w", 1, -4.0, 3, 15, -1.017),
    }


def test_plan_runs_both_signs_for_every_seed_and_needs_two_seeds():
    conditions = V.plan([0, 1], 4.0, full_controls=False)
    assert [c.name for c in conditions] == [
        "seed0-positive-w", "seed0-negative-w",
        "seed1-positive-w", "seed1-negative-w",
    ]
    assert [c.world_angular_velocity_rad_s for c in conditions] == [4.0, -4.0, 4.0, -4.0]
    assert all(c.mode == "intact" for c in conditions)

    with pytest.raises(SystemExit, match="at least two seeds"):
        V.plan([0], 4.0, full_controls=False)
    with pytest.raises(SystemExit, match="must be positive"):
        V.plan([0, 1], -4.0, full_controls=False)


def test_full_controls_adds_the_baseline_and_the_three_controls():
    names = [c.name for c in V.plan([0, 1], 4.0, full_controls=True)]
    assert names[:4] == [
        "seed0-positive-w", "seed0-negative-w",
        "seed1-positive-w", "seed1-negative-w",
    ]
    assert names[4:] == [
        "seed0-zero-w",
        "seed0-output-disconnected",
        "seed0-drive-channel-swapped",
        "seed0-drive-time-shuffled",
    ]
    controls = {c.name: c for c in V.plan([0, 1], 4.0, full_controls=True)}
    # The rate-matched controls take their drive from the intact +w run, so that
    # run must come first in the plan.
    for name in ("seed0-drive-channel-swapped", "seed0-drive-time-shuffled"):
        assert controls[name].drive_from == "seed0-positive-w"
        assert controls[name].mode == "drive-rate-matched-control"
    assert controls["seed0-zero-w"].world_angular_velocity_rad_s == 0.0


def test_channel_swapped_control_is_exactly_rate_matched_and_mirrored():
    drive = [(0.0, 0.5), (0.1, 0.7), (0.2, 0.9)]
    override, described = V.build_drive_override("channel-swapped", drive)
    assert [override(i, (9.9, 9.9)) for i in range(3)] == [
        (0.5, 0.0), (0.7, 0.1), (0.9, 0.2)
    ]
    # The totals reaching the CPG are the same two multisets, exchanged.
    assert described["mean_left"] == pytest.approx(sum(r for _, r in drive) / 3)
    assert described["mean_right"] == pytest.approx(sum(l for l, _ in drive) / 3)
    assert described["rate_matched"] is True
    # Past the source sequence the control supplies nothing rather than guessing.
    assert override(99, (1.0, 1.0)) == (0.0, 0.0)


def test_time_shuffled_control_preserves_each_channel_mean_exactly():
    drive = [(i / 100.0, 1.0 - i / 100.0) for i in range(50)]
    override, described = V.build_drive_override("time-shuffled", drive)
    supplied = [override(i, (0.0, 0.0)) for i in range(50)]
    assert sorted(supplied) == sorted(drive)          # a permutation, nothing else
    assert supplied != drive                          # and a real one
    assert described["mean_left"] == pytest.approx(sum(l for l, _ in drive) / 50)
    assert described["mean_right"] == pytest.approx(sum(r for _, r in drive) / 50)
    assert described["shuffle_seed"] == V.SHUFFLE_SEED
    # Fixed seed, so the control is reproducible.
    again, _ = V.build_drive_override("time-shuffled", drive)
    assert [again(i, (0.0, 0.0)) for i in range(50)] == supplied


def test_verdict_passes_only_when_both_directions_follow_on_every_seed():
    judgement = V.judge(_reversal_outcomes(), [0, 1], V.YAW_PASS_THRESHOLD_RAD)
    assert judgement["verdict"] == "PASS"
    assert all(e["turn_follows_stimulus_sign"] for e in judgement["per_seed"])
    assert all(e["dna02_asymmetry_reverses"] for e in judgement["per_seed"])

    # A seed that turns the same way under both signs is a failure, however large.
    judgement = V.judge(
        _reversal_outcomes(yaw_negative=+1.2), [0, 1], V.YAW_PASS_THRESHOLD_RAD
    )
    assert judgement["verdict"] == "FAIL"

    # So is a turn that never leaves the noise floor.
    judgement = V.judge(
        _reversal_outcomes(yaw_positive=0.01, yaw_negative=-0.01),
        [0, 1],
        V.YAW_PASS_THRESHOLD_RAD,
    )
    assert judgement["verdict"] == "FAIL"


def test_too_few_spikes_is_inconclusive_not_a_pass():
    outcomes = _reversal_outcomes()
    outcomes["seed1-negative-w"] = _outcome("seed1-negative-w", 1, -4.0, 1, 2, -1.017)
    judgement = V.judge(outcomes, [0, 1], V.YAW_PASS_THRESHOLD_RAD)
    assert judgement["verdict"] == "INCONCLUSIVE"
    assert "seed1-negative-w" in judgement["reason"]


def test_planar_displacement_and_yaw_come_from_the_summary(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "telemetry.jsonl").write_text(
        "".join(
            json.dumps({"body": {"thorax": {"yaw_rad": yaw, "position_mm": [0.0, 0.0, 0.9]}}}) + "\n"
            for yaw in (0.5, 1.5, 2.5)
        )
    )
    outcome = V._measure(
        V.Condition("seed0-positive-w", 0, 4.0, "reversal"),
        run,
        _summary(30, 3, 2.5, x=3.0, y=4.0),
    )
    assert outcome.planar_displacement_mm == pytest.approx(5.0)
    assert outcome.final_yaw_rad == pytest.approx(2.5)
    assert outcome.cumulative_yaw_rad == pytest.approx(2.5)
    assert outcome.row()["condition"] == "seed0-positive-w"


def test_cumulative_yaw_unwraps_a_turn_past_half_a_revolution(tmp_path):
    """The bug this caught: a real +3.168 rad turn reads -3.116 rad wrapped."""
    run = tmp_path / "wrapped"
    run.mkdir()
    # A steady left turn through pi, as atan2 reports it.
    yaws = [0.0, 1.0, 2.0, 3.0, -3.1, -2.6]
    (run / "telemetry.jsonl").write_text(
        "".join(json.dumps({"body": {"thorax": {"yaw_rad": y}}}) + "\n" for y in yaws)
    )
    assert V.cumulative_yaw_rad(run) == pytest.approx(2 * math.pi - 2.6, abs=1e-9)
    assert V.cumulative_yaw_rad(run) > math.pi          # the real turn
    assert yaws[-1] < 0                                 # what the wrapped field says

    # And the judgement follows the cumulative value, not the wrapped one.
    outcomes = _reversal_outcomes()
    outcomes["seed1-positive-w"] = _outcome(
        "seed1-positive-w", 1, 4.0, 31, 0, -3.116, cumulative=+3.168
    )
    judgement = V.judge(outcomes, [0, 1], V.YAW_PASS_THRESHOLD_RAD)
    assert judgement["verdict"] == "PASS"
    entry = judgement["per_seed"][1]
    assert entry["cumulative_yaw_positive_w_rad"] == pytest.approx(3.168)
    assert entry["wrapped_final_yaw_positive_w_rad"] == pytest.approx(-3.116)


def test_markdown_receipt_states_the_verdict_and_the_limits():
    judgement = V.judge(_reversal_outcomes(), [0, 1], V.YAW_PASS_THRESHOLD_RAD)
    receipt = {
        "judgement": judgement,
        "created_at": "2026-09-26T00:00:00+00:00",
        "package_version": "0.1.0",
        "telemetry_format_version": 2,
        "conditions": [o.row() for o in _reversal_outcomes().values()],
        "controls": ["`x`: mirrored"],
        "reuse_check": {"note": "byte-identical"},
        "interpretation": ["two neurons wide"],
        "output_dir": "/tmp/x",
    }
    text = V._markdown(receipt)
    assert "Verdict: **PASS**" in text
    assert "seed0-positive-w" in text and "two neurons wide" in text
    assert "byte-identical" in text


def test_a_rate_matched_control_may_not_call_itself_intact(tmp_path):
    """The mode and the drive path can never disagree in a written artefact."""
    override, description = V.build_drive_override(
        "channel-swapped", [(0.0, 0.4)] * 3
    )
    with pytest.raises(ValueError, match="may not be used with mode 'intact'"):
        run_embodied(
            EmbodiedConfig(duration_s=0.006, output_dir=tmp_path / "a"),
            FakeGraph(),
            FakeBody(),
            drive_override=override,
        )
    with pytest.raises(ValueError, match="requires a drive_override"):
        run_embodied(
            EmbodiedConfig(
                duration_s=0.006,
                output_dir=tmp_path / "b",
                mode="drive-rate-matched-control",
            ),
            FakeGraph(),
            FakeBody(),
        )


def test_control_run_records_the_supplied_drive_and_disowns_the_output(tmp_path):
    body = FakeBody()
    override, description = V.build_drive_override("channel-swapped", [(0.0, 0.4)] * 3)
    summary = run_embodied(
        EmbodiedConfig(
            duration_s=0.006,
            output_dir=tmp_path / "control",
            mode="drive-rate-matched-control",
        ),
        FakeGraph(),
        body,
        decoder=DNa02CPGDecoder(),
        drive_override=override,
        drive_override_description=description,
    )
    assert body.commands == [(0.4, 0.0)] * 3
    assert summary["decoder_produced_output"] is True
    assert summary["motor_output_reached_body"] is False

    rows = [
        json.loads(line)
        for line in (tmp_path / "control/telemetry.jsonl").read_text().splitlines()
    ]
    assert rows[0]["motor"]["applied_cpg_drive"] == [0.4, 0.0]
    assert "drive-rate-matched-control" in rows[0]["motor"]["applied_drive_source"]
    assert rows[0]["motor"]["output_connected"] is False
    # The decoder's own output is still logged beside it.
    assert rows[0]["motor"]["decoded_cpg_drive"][1] > 0.0

    manifest = json.loads((tmp_path / "control/manifest.json").read_text())
    assert manifest["control"]["mode"] == "drive-rate-matched-control"
    assert manifest["control"]["drive_override"]["construction"] == "channel-swapped"


def test_verdict_subcommand_is_documented_in_the_cli():
    from neurofly_body import cli

    args = cli._parser().parse_args(
        ["verdict", "--output", "/tmp/v", "--seeds", "0", "1", "--full-controls"]
    )
    assert args.command == "verdict"
    assert args.duration == 5.0
    assert args.seeds == [0, 1]
    assert args.angular_velocity == 4.0
    assert args.full_controls is True
    assert args.reuse_check is True

    args = cli._parser().parse_args(
        ["verdict", "--output", "/tmp/v", "--no-reuse-check"]
    )
    assert args.reuse_check is False
    assert args.seeds == [0, 1]


def test_verdict_takes_the_same_decoder_choice_as_run():
    from neurofly_body import cli
    from neurofly_body.decoder import DNCommandDecoder

    args = cli._parser().parse_args(["verdict", "--output", "/tmp/v"])
    assert args.decoder == "dn-v2"
    assert isinstance(V.make_decoder(args), DNCommandDecoder)
    args = cli._parser().parse_args(
        ["verdict", "--output", "/tmp/v", "--decoder", "dna02-crossed-v1"]
    )
    assert isinstance(V.make_decoder(args), DNa02CPGDecoder)


def test_intact_conditions_record_a_replayable_run_invocation():
    """replay-check re-runs a verdict condition standalone from its manifest."""
    from neurofly_body import cli

    args = cli._parser().parse_args(["verdict", "--output", "/tmp/v"])
    for condition in V.plan([0, 1], 4.0, full_controls=True):
        invocation = V.run_invocation(args, condition)
        if condition.mode == "drive-rate-matched-control":
            assert invocation["replayable_by_run"] is False
            continue
        assert set(cli.RUN_ARGUMENTS) <= set(invocation)
        assert invocation["seed"] == condition.seed
        assert invocation["world_angular_velocity_rad_s"] == condition.world_angular_velocity_rad_s
        argv = ["run", "--output", "/tmp/r"]
        for name in cli.RUN_ARGUMENTS:
            value = invocation[name]
            if isinstance(value, bool):
                if value:
                    argv.append("--" + name.replace("_", "-"))
            elif value is not None:
                argv += ["--" + name.replace("_", "-"), str(value)]
        parsed = cli._parser().parse_args(argv)
        assert parsed.mode == condition.mode and parsed.decoder == "dn-v2"


def test_control_findings_and_interpretation_come_from_the_numbers():
    outcomes = _reversal_outcomes()
    plan = {c.name: c for c in V.plan([0, 1], 4.0, full_controls=True)}
    for name, yaw in (
        ("seed0-zero-w", 0.004),
        ("seed0-output-disconnected", 0.007),
        ("seed0-drive-channel-swapped", -2.4),
        ("seed0-drive-time-shuffled", 0.05),
    ):
        outcomes[name] = V.Outcome(
            condition=plan[name],
            summary=_summary(9, 10, yaw),
            final_yaw_rad=yaw,
            cumulative_yaw_rad=yaw,
            planar_displacement_mm=4.1,
        )
    findings = {f["condition"]: f for f in V.control_findings(outcomes, V.YAW_PASS_THRESHOLD_RAD)}
    assert findings["seed0-output-disconnected"]["turn_absent"] is True
    assert findings["seed0-drive-channel-swapped"]["turn_reversed"] is True
    assert findings["seed0-drive-time-shuffled"]["turn_survived"] is False
    assert "does not survive" in findings["seed0-drive-time-shuffled"]["reading"]

    judgement = V.judge(outcomes, [0, 1], V.YAW_PASS_THRESHOLD_RAD)
    lines = V.interpretation("dn-v2", judgement, outcomes, 5.0)
    assert any("DNp09" in line for line in lines)
    assert not any("two neurons wide" in line for line in lines)
    assert any("seed0-zero-w" in line and "+0.004" in line for line in lines)
    legacy = V.interpretation("dna02-crossed-v1", judgement, outcomes, 5.0)
    assert any("two neurons wide" in line for line in legacy)


def test_receipt_never_prints_an_absolute_output_path():
    judgement = V.judge(_reversal_outcomes(), [0, 1], V.YAW_PASS_THRESHOLD_RAD)
    receipt = {
        "judgement": judgement,
        "created_at": "2026-10-05T00:00:00+00:00",
        "package_version": "0.1.0",
        "telemetry_format_version": 2,
        "conditions": [o.row() for o in _reversal_outcomes().values()],
        "controls": [],
        "reuse_check": {"note": "byte-identical"},
        "interpretation": [],
        "output_dir": "/somewhere/private/verdict-run",
    }
    text = V._markdown(receipt)
    assert "/somewhere/private" not in text
    assert "<output>/verdict-run" in text


def test_displacement_is_measured_from_the_first_record_not_the_origin(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "telemetry.jsonl").write_text(
        json.dumps({"body": {"thorax": {"yaw_rad": 0.0, "position_mm": [0.6, 0.0, 0.9]}}}) + "\n"
    )
    outcome = V._measure(
        V.Condition("seed0-zero-w", 0, 0.0, "baseline"), run, _summary(1, 1, 0.0, x=0.6, y=0.0)
    )
    assert outcome.planar_displacement_mm == pytest.approx(0.0)


def test_a_control_built_from_a_run_that_did_not_turn_is_uninformative():
    outcomes = _reversal_outcomes(yaw_positive=0.01)
    plan = {c.name: c for c in V.plan([0, 1], 4.0, full_controls=True)}
    outcomes["seed0-drive-channel-swapped"] = V.Outcome(
        condition=plan["seed0-drive-channel-swapped"], summary=_summary(1, 1, 0.0),
        final_yaw_rad=0.0, cumulative_yaw_rad=0.0, planar_displacement_mm=0.0,
    )
    (finding,) = V.control_findings(outcomes, V.YAW_PASS_THRESHOLD_RAD)
    assert finding["informative"] is False and "uninformative" in finding["reading"]


class SeededGraph(FakeGraph):
    """Fake graph whose spike stream restarts from ``optomotor_seed`` on reset."""

    def __init__(self):
        super().__init__()
        self.optomotor_seed = 0
        self.reset()

    def reset(self):
        super().reset()
        import numpy as np

        self.rng = np.random.default_rng(self.optomotor_seed)

    def step(self, sensory, duration_ms=2.0):
        reply = super().step(sensory, duration_ms)
        reply["dna02_rate_l"] = float(self.rng.integers(0, 3)) * 500.0
        reply["dna02_rate_r"] = float(self.rng.integers(0, 3)) * 500.0
        return reply


def _reuse_setup(tmp_path):
    from neurofly_body import cli

    args = cli._parser().parse_args(
        ["verdict", "--output", str(tmp_path), "--decoder", "dna02-crossed-v1",
         "--reuse-check-duration", "0.01"]
    )
    neural, body = SeededGraph(), FakeBody()
    conditions = V.plan([0, 1], 4.0, full_controls=False)
    for condition in conditions:          # ends on seed 1, like the default plan
        V._set_seed(neural, condition.seed)
        run_embodied(
            V._config(args, condition, tmp_path / condition.name, 0.04),
            neural, body, decoder=V.make_decoder(args), close_body=False,
        )
    return args, conditions[0], neural, body


def test_reuse_check_repeats_the_first_condition_on_its_own_seed(tmp_path):
    """The false alarm: a plan ending on seed 1 re-ran seed 0 on seed 1's stream."""
    args, first, neural, body = _reuse_setup(tmp_path)
    check = V._reuse_check(args, first, tmp_path, neural, body)
    assert check["bit_identical"] is True and check["records_compared"] == 5


def test_a_failed_reuse_check_invalidates_the_verdict(tmp_path, monkeypatch):
    args, first, neural, body = _reuse_setup(tmp_path)
    monkeypatch.setattr(V, "_set_seed", lambda neural, seed: None)   # the old bug
    check = V._reuse_check(args, first, tmp_path, neural, body)
    assert check["bit_identical"] is False and "DIVERGENT" in check["note"]

    judgement = V.judge(_reversal_outcomes(), [0, 1], V.YAW_PASS_THRESHOLD_RAD, check)
    assert judgement["verdict"] == "INVALID"
    assert "INVALID" in V.VERDICTS
    # A passing or skipped reuse check leaves the verdict to the turns.
    ok = {"ran": True, "bit_identical": True, "note": "byte-identical"}
    assert V.judge(_reversal_outcomes(), [0, 1], V.YAW_PASS_THRESHOLD_RAD, ok)["verdict"] == "PASS"
    skipped = {"ran": False, "bit_identical": None, "note": "skipped"}
    assert V.judge(_reversal_outcomes(), [0, 1], V.YAW_PASS_THRESHOLD_RAD, skipped)["verdict"] == "PASS"


def _invocation(**overrides):
    invocation = {
        "duration_s": 5.0, "seeds": [0, 1], "angular_velocity_rad_s": 2.5,
        "full_controls": True, "reuse_check": True, "reuse_check_duration_s": 0.1,
        "decoder": "dna02-crossed-v1", "contrast": 0.5, "neural_dt_ms": 1.0,
        "physics_dt_s": 0.0002, "warmup_s": 0.05, "decoder_tau_ms": 50.0,
        "max_cpg_drive": 1.2, "p9_gain_per_hz": 0.02, "dna02_stride_k_per_hz": 0.01,
        "mdn_gain_per_hz": 0.02, "cpg_gain_per_hz": 0.07,
        "graph_dir": None, "connectome_dir": None,
    }
    invocation.update(overrides)
    return invocation


def test_reproduce_command_pins_the_backend_and_every_parameter():
    """CPU and CUDA diverge, so `auto` cannot reproduce a receipt."""
    from neurofly_body import cli

    command = V.reproduce_command(_invocation(), "cpu")
    assert command.startswith("NEUROFLY_BRAIN_BACKEND=cpu python -m neurofly_body verdict")
    for flag in ("--angular-velocity 2.5", "--contrast 0.5", "--neural-dt-ms 1",
                 "--physics-dt-s 0.0002", "--warmup-s 0.05", "--cpg-gain-per-hz 0.07",
                 "--decoder dna02-crossed-v1", "--decoder-tau-ms 50", "--max-cpg-drive 1.2",
                 "--p9-gain-per-hz 0.02", "--dna02-stride-k-per-hz 0.01",
                 "--mdn-gain-per-hz 0.02", "--full-controls",
                 "--reuse-check-duration 0.1", "--seeds 0 1"):
        assert flag in command
    # The command parses back to the same parameters.
    argv = command.split()[1:]
    assert argv[:3] == ["python", "-m", "neurofly_body"]
    args = cli._parser().parse_args(argv[3:])
    assert (args.angular_velocity, args.contrast, args.neural_dt_ms, args.physics_dt_s,
            args.cpg_gain_per_hz, args.decoder) == (2.5, 0.5, 1.0, 0.0002, 0.07,
                                                     "dna02-crossed-v1")
    skipped = V.reproduce_command(_invocation(reuse_check=False), "cuda")
    assert skipped.startswith("NEUROFLY_BRAIN_BACKEND=cuda ") and "--no-reuse-check" in skipped


def test_markdown_reproduce_section_states_backend_code_data_and_dynamics():
    judgement = V.judge(_reversal_outcomes(), [0, 1], V.YAW_PASS_THRESHOLD_RAD)
    command = V.reproduce_command(_invocation(), "cpu")
    receipt = {
        "judgement": judgement, "created_at": "2026-10-05T00:00:00+00:00",
        "package_version": "0.1.0", "telemetry_format_version": 2,
        "conditions": [o.row() for o in _reversal_outcomes().values()],
        "controls": [], "reuse_check": {"note": "byte-identical"}, "interpretation": [],
        "output_dir": "/tmp/x", "invocation": _invocation(),
        "reproduce": {
            "brain_backend": "cpu", "command": command,
            "code": {"commit": "c" * 40, "dirty": False, "neurofly_body_version": "0.1.0"},
            "data": {"graph_sha256": "g" * 64},
            "dynamics": {"lif_dynamics_version": "v3"},
        },
    }
    text = V._markdown(receipt)
    assert command in text
    assert "NEUROFLY_BRAIN_BACKEND=cpu python -m neurofly_body replay-check" in text
    assert "c" * 40 in text and "g" * 64 in text and "`lif_dynamics_version` = `v3`" in text

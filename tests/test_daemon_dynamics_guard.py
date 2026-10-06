"""The daemon refuses LIF dynamics it does not support, before touching any file.

argparse never checks a default against ``choices``, so NEUROFLY_LIF_DYNAMICS=v6a (or
any unknown string, or a research-only version such as v4) used to reach the daemon
through the environment-derived default.  The daemon supports v1-v3; research engines
stay usable through Brain(..., dynamics=...) outside the daemon.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

import neurofly_daemon as nd

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("value", ["v6a", "not-a-version", "v4"])
def test_environment_selected_unsupported_dynamics_is_refused_before_any_write(tmp_path, value):
    out = tmp_path / "out"
    pid = tmp_path / "pid" / "daemon.pid"
    env = dict(os.environ, NEUROFLY_LIF_DYNAMICS=value, PYTHONPATH=str(ROOT))
    proc = subprocess.run([sys.executable, str(ROOT / "neurofly_daemon.py"), "--port", "0",
                           "--backend", "connectome-fixed", "--test-synthetic-graph",
                           "--output-dir", str(out), "--pid-file", str(pid)],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 2
    assert f"{value!r}" in proc.stderr and "not supported by the daemon" in proc.stderr
    assert "Traceback" not in proc.stderr
    assert not out.exists() and not pid.parent.exists()          # nothing created or written


@pytest.mark.parametrize("value", ["v6a", "garbage"])
def test_programmatic_runner_refuses_unsupported_dynamics_before_any_write(tmp_path, monkeypatch, value):
    monkeypatch.setenv("NEUROFLY_LIF_DYNAMICS", value)
    out = tmp_path / "out"
    with pytest.raises(nd.UnsupportedDynamics):
        nd.ContinuousExperimentRunner(initial_paradigm="t-maze", output_dir=out, backend="connectome-fixed",
                                      test_synthetic_graph=True)
    assert not out.exists()


def test_switching_to_a_graph_backend_under_unsupported_dynamics_is_refused(tmp_path, monkeypatch):
    runner = nd.ContinuousExperimentRunner(initial_paradigm="t-maze", output_dir=tmp_path, backend="modular",
                                           test_synthetic_graph=True)
    monkeypatch.setenv("NEUROFLY_LIF_DYNAMICS", "v6a")
    before = sorted(p.name for p in tmp_path.iterdir())
    with runner.lock:
        result = runner._apply_command({"action": "switch_backend", "backend": "connectome-fixed"})
    assert result["status"] == "error" and "not supported by the daemon" in result["message"]
    assert runner.backend == "modular" and runner.registry is None
    assert sorted(p.name for p in tmp_path.iterdir()) == before


@pytest.mark.parametrize("value", ["v1", "v2", "v3"])
def test_supported_daemon_dynamics_are_unchanged(monkeypatch, value):
    assert nd.DAEMON_DYNAMICS == ("v1", "v2", "v3")
    assert nd.daemon_dynamics(value) == value
    monkeypatch.setenv("NEUROFLY_LIF_DYNAMICS", value)
    assert nd.daemon_dynamics() == value
    assert nd.build_arg_parser().parse_args([]).dynamics == value


def test_default_is_v3_and_research_engines_stay_usable_outside_the_daemon(monkeypatch):
    monkeypatch.delenv("NEUROFLY_LIF_DYNAMICS", raising=False)
    assert nd.daemon_dynamics() == "v3"
    from brainlab.graph_identity import DYNAMICS_VERSIONS
    assert {"v4", "v5"} <= set(DYNAMICS_VERSIONS)                    # still declared for research
    monkeypatch.setenv("NEUROFLY_LIF_DYNAMICS", "v4")
    from brainlab.graph_identity import active_dynamics_version
    assert active_dynamics_version() == "v4"                         # Brain(...) path unaffected
    with pytest.raises(nd.UnsupportedDynamics):
        nd.daemon_dynamics()


class _Resolved(Exception):
    pass


def _resolved_dynamics(monkeypatch, argv):
    """Run run_daemon() up to the first step after the dynamics are resolved."""
    seen = {}

    def stop_here(args):
        seen["dynamics"] = args.dynamics
        raise _Resolved
    monkeypatch.setattr(nd, "choose_default_backend", stop_here)
    monkeypatch.setattr(sys, "argv", ["neurofly_daemon.py"] + argv)
    with pytest.raises(_Resolved):
        nd.run_daemon()
    return seen["dynamics"]


@pytest.mark.parametrize("argv,expected", [(["--dynam", "v1"], "v1"), (["--dynam=v2"], "v2"),
                                           (["--dynamics", "v1"], "v1")])
def test_abbreviated_dynamics_option_wins_over_an_unsupported_environment(monkeypatch, argv, expected):
    monkeypatch.setenv("NEUROFLY_LIF_DYNAMICS", "v6a")
    assert _resolved_dynamics(monkeypatch, argv) == expected


def test_abbreviated_dynamics_option_is_used_when_the_environment_is_supported(monkeypatch):
    monkeypatch.setenv("NEUROFLY_LIF_DYNAMICS", "v3")
    assert _resolved_dynamics(monkeypatch, ["--dynam", "v2"]) == "v2"
    assert _resolved_dynamics(monkeypatch, []) == "v3"


def test_unsupported_environment_without_an_option_is_refused(monkeypatch, capsys):
    monkeypatch.setenv("NEUROFLY_LIF_DYNAMICS", "v6a")
    monkeypatch.setattr(sys, "argv", ["neurofly_daemon.py"])
    with pytest.raises(SystemExit) as stop:
        nd.run_daemon()
    assert stop.value.code == 2 and "'v6a'" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# E_inh sensitivity overrides (NEUROFLY_LIF_E_INH_MV)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value", ["-60", "-56.0", "not-a-number", "nan"])
def test_unsupported_e_inh_override_is_refused_before_any_write(tmp_path, value):
    out = tmp_path / "out"
    pid = tmp_path / "pid" / "daemon.pid"
    env = dict(os.environ, NEUROFLY_LIF_E_INH_MV=value, PYTHONPATH=str(ROOT))
    env.pop("NEUROFLY_LIF_DYNAMICS", None)
    proc = subprocess.run([sys.executable, str(ROOT / "neurofly_daemon.py"), "--port", "0",
                           "--backend", "connectome-fixed", "--test-synthetic-graph",
                           "--output-dir", str(out), "--pid-file", str(pid)],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 2
    assert "NEUROFLY_LIF_E_INH_MV" in proc.stderr and "sensitivity variant" in proc.stderr
    assert "Traceback" not in proc.stderr
    assert not out.exists() and not pid.parent.exists()


def test_programmatic_runner_refuses_an_e_inh_variant_before_any_write(tmp_path, monkeypatch):
    monkeypatch.delenv("NEUROFLY_LIF_DYNAMICS", raising=False)
    monkeypatch.setenv("NEUROFLY_LIF_E_INH_MV", "-60")
    out = tmp_path / "out"
    with pytest.raises(nd.UnsupportedSensitivityOverride):
        nd.ContinuousExperimentRunner(initial_paradigm="t-maze", output_dir=out, backend="connectome-fixed",
                                      test_synthetic_graph=True)
    assert not out.exists()


@pytest.mark.parametrize("value", [None, "", "-70", "-70.0"])
def test_baseline_or_unset_e_inh_is_accepted(tmp_path, monkeypatch, value):
    monkeypatch.delenv("NEUROFLY_LIF_DYNAMICS", raising=False)
    if value is None:
        monkeypatch.delenv("NEUROFLY_LIF_E_INH_MV", raising=False)
    else:
        monkeypatch.setenv("NEUROFLY_LIF_E_INH_MV", value)
    assert nd.daemon_configuration() == "v3"
    runner = nd.ContinuousExperimentRunner(initial_paradigm="t-maze", output_dir=tmp_path,
                                           backend="connectome-fixed", test_synthetic_graph=True)
    assert runner.registry is not None
    assert _resolved_dynamics(monkeypatch, []) == "v3"


def test_e_inh_research_path_stays_usable_outside_the_daemon(monkeypatch):
    monkeypatch.setenv("NEUROFLY_LIF_E_INH_MV", "-60")
    from brainlab.graph_identity import active_e_inh_mV, dynamics_variant
    assert active_e_inh_mV() == -60.0
    assert dynamics_variant("v3", -60.0) != dynamics_variant("v3")
    with pytest.raises(nd.UnsupportedSensitivityOverride):
        nd.daemon_e_inh()


@pytest.mark.parametrize("value", ["   ", "\t", " \n "])
def test_whitespace_e_inh_is_refused_before_any_write(tmp_path, value):
    out = tmp_path / "out"
    pid = tmp_path / "pid" / "daemon.pid"
    env = dict(os.environ, NEUROFLY_LIF_E_INH_MV=value, PYTHONPATH=str(ROOT))
    env.pop("NEUROFLY_LIF_DYNAMICS", None)
    proc = subprocess.run([sys.executable, str(ROOT / "neurofly_daemon.py"), "--port", "0",
                           "--backend", "connectome-fixed", "--test-synthetic-graph",
                           "--output-dir", str(out), "--pid-file", str(pid)],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 2 and "NEUROFLY_LIF_E_INH_MV" in proc.stderr
    assert "Traceback" not in proc.stderr and "corrupt" not in proc.stderr.lower()
    assert not out.exists() and not pid.parent.exists()

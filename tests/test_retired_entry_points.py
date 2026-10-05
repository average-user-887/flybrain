"""Retired entry points refuse to run, explain why, and write nothing (docs/RETIREMENT_INDEX.md)."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

RETIRED = {
    'cli-full-sim': ['-m', 'neurofly.cli', 'full-sim'],
    'full-sim-script': [str(ROOT / 'experiments/full_connectome_simulation.py')],
    'full-sim-module': ['-m', 'experiments.full_connectome_simulation'],
    'paradigm-battery-script': [str(ROOT / 'experiments/run_paradigm_battery.py')],
    'paradigm-battery-module': ['-m', 'experiments.run_paradigm_battery'],
    'whole-brain-battery-script': [str(ROOT / 'experiments/whole_brain_scientific_battery.py')],
    'whole-brain-battery-module': ['-m', 'experiments.whole_brain_scientific_battery'],
    'sync-ecosystem': [str(ROOT / 'sync_ecosystem.py')],
    'test-srv': [str(ROOT / 'scripts/test_srv.py')],
}


def _run(argv, cwd):
    env = dict(os.environ, PYTHONPATH=str(ROOT), CUDA_VISIBLE_DEVICES='')
    return subprocess.run([sys.executable] + argv, cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize('name', sorted(RETIRED))
def test_retired_entry_point_refuses_and_writes_nothing(tmp_path, name):
    proc = _run(RETIRED[name] + ['--output-dir', 'out'], tmp_path)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert 'RETIRED' in proc.stderr and 'docs/RETIREMENT_INDEX.md' in proc.stderr
    assert not list(tmp_path.iterdir()), f'{name} wrote files'


@pytest.mark.parametrize('name', sorted(RETIRED))
def test_retired_entry_point_help_explains_and_writes_nothing(tmp_path, name):
    proc = _run(RETIRED[name] + ['--help'], tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert 'usage:' in proc.stdout and 'RETIRED' in proc.stdout
    assert not list(tmp_path.iterdir()), f'{name} --help wrote files'


def test_cli_full_sim_does_not_import_the_experiment(capsys):
    from neurofly import cli
    sys.modules.pop('experiments.full_connectome_simulation', None)
    assert cli.main(['full-sim', '--steps-per-task', '1']) == 2
    assert 'experiments.full_connectome_simulation' not in sys.modules
    assert 'RETIRED' in capsys.readouterr().err


def test_test_srv_import_is_inert(tmp_path):
    code = ("import importlib.util, sys\n"
            f"spec = importlib.util.spec_from_file_location('test_srv', {str(ROOT / 'scripts/test_srv.py')!r})\n"
            "spec.loader.exec_module(importlib.util.module_from_spec(spec))\n"
            "assert 'brainlab.cosim_server' not in sys.modules and 'brainlab.brain' not in sys.modules\n")
    proc = _run(['-c', code], tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == '' and proc.stderr == ''


def test_retired_mains_have_no_silent_alias(tmp_path, capsys):
    from experiments import full_connectome_simulation, run_paradigm_battery, whole_brain_scientific_battery
    assert full_connectome_simulation.main([]) == 2
    assert run_paradigm_battery.main([]) == 2
    out = tmp_path / 'battery'
    with pytest.raises(RuntimeError, match='RETIRED'):
        whole_brain_scientific_battery.run_complete_scientific_battery(output_dir=str(out))
    assert not out.exists()

"""Exploratory transfer thresholds must not masquerade as fit validation."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


@pytest.mark.parametrize('diagnostic,response,expected', [
    (True, -6.0, 'DIAGNOSTIC PASS'),
    (True, -4.0, 'DIAGNOSTIC FAIL'),
    (False, -6.0, 'PASS'),
])
def test_analysis_scope(tmp_path, diagnostic, response, expected):
    contract = {'pass_criteria': {'L2_ON': {'type': 'L2', 'sign': '-', 'min_abs_mV': 5}},
                'falsified_if_below_0p5_mV': []}
    if diagnostic:
        contract['transfer_diagnostics_are_not_fit_objectives'] = True
    pin = tmp_path / 'contract.json'
    pin.write_text(json.dumps(contract))
    np.savez(tmp_path / 'cell_columns.npz', node=[0], cell_type=['L2'], eye=['R'], column_state=['COMPLETE'])
    for condition in ('C1_fullfield_ON', 'C2_fullfield_OFF'):
        np.savez(tmp_path / (condition + '.npz'), node=[0], dV_E=[response], dV_S=[response], V_B=[-52.0])
    script = Path(__file__).resolve().parents[1] / 'scripts/v6/analyse_v6.py'
    done = subprocess.run([sys.executable, str(script), '--contract', str(pin), '--contract-sha256',
                           hashlib.sha256(pin.read_bytes()).hexdigest(), '--phase1', str(tmp_path),
                           '--run', str(tmp_path)], text=True, capture_output=True, check=True)
    result = json.loads((tmp_path / 'analysis.json').read_text())
    assert result['verdict'] == expected
    assert json.loads(done.stdout.splitlines()[-1])['verdict'] == expected
    if diagnostic:
        assert result['stage_validation'] == 'NOT EVALUABLE'
        assert result['motion'] == 'UNOPENED'
    else:
        assert 'stage_validation' not in result

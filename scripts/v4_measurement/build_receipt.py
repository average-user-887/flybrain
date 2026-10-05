"""Assemble docs/receipts/lif_dynamics_v4.json from the predeclared measurements."""
import hashlib
import json
import pathlib
import subprocess
import sys

W = pathlib.Path('<redacted-path>/Documents/ChatGPT/flybrain/<redacted-path>/.wt-graded')
T = pathlib.Path('<redacted-path>/tmp/graded')
sys.path.insert(0, str(W))
from brainlab.graph_identity import LIF_DYNAMICS_V4, dynamics_pin

a = json.load(open(T / 'measure_a.json'))
bc = json.load(open(T / 'measure_bc.json'))
locked = W / 'docs/receipts/graded_transmission_v4_declaration.locked.md'

dsi = bc['direction_selectivity']


def gate_c():
    t45 = [k for k in dsi if k.startswith(('T4', 'T5'))]
    best = None
    for k in t45:
        d = dsi[k]
        if d['dsi'] is not None and d['magnitude'] >= 0.5 and d['dsi'] >= 0.2:
            if best is None or d['dsi'] > dsi[best]['dsi']:
                best = k
    dn = {k: dsi[k] for k in ('DNa02_L', 'DNa02_R') if k in dsi}
    dn_signal = any(abs(v) > 0 for k in dn for v in dn[k]['tuning'].values())
    return dict(
        t4t5_subtype_with_dsi_ge_0_2_and_magnitude_ge_0_5mV=best,
        dna02_direction_dependent_signal=bool(dn_signal),
        passed=bool(best is not None and dn_signal),
        rule='spec §7.10 declared gate on (d)')


receipt = dict(
    schema='flybrain.lif-dynamics-receipt.v4',
    dynamics=LIF_DYNAMICS_V4,
    dynamics_pin=dynamics_pin('v4'),
    declaration_lock=dict(
        path='docs/receipts/graded_transmission_v4_declaration.locked.md',
        sha256=hashlib.sha256(locked.read_bytes()).hexdigest(),
        locked_at='2026-09-27T11:15:16+02:00',
        note='written and committed before any v4 measurement; commit ce8a66f'),
    code_revision=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=W, capture_output=True,
                                 text=True).stdout.strip(),
    unit_level_measurements=a,
    real_graph_measurement=bc,
    declared_gate_on_d=gate_c(),
)
out = W / 'docs/receipts/lif_dynamics_v4.json'
out.write_text(json.dumps(receipt, indent=2, default=str) + '\n')
print('wrote', out, out.stat().st_size, 'bytes')
print(json.dumps(receipt['declared_gate_on_d'], indent=2))

"""Static helpers for the bounded escape diagnostic; no dynamics imports."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
THREAD_ENV = dict(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMBA_NUM_THREADS='1')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def strict_json(path):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError('duplicate JSON key')
            out[key] = value
        return out
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def require_environment():
    if any(os.environ.get(k) != v for k, v in THREAD_ENV.items()):
        raise ValueError('CPU-only single-thread environment required')


def input_data(condition, seed, protocol):
    nodes = np.asarray(condition['activate_nodes'], dtype=np.int64)
    ticks = int(round(protocol['duration_ms'] / protocol['dt_ms']))
    seq = [seed, int(round(10 * condition['rate_hz'])), len(nodes)]
    rng = np.random.default_rng(seq)
    rng_sha = hashlib.sha256(json.dumps(rng.bit_generator.state, sort_keys=True, default=int).encode()).hexdigest()
    present = bool(len(nodes) and condition['rate_hz'] > 0)
    events = rng.random((ticks, len(nodes))) < condition['rate_hz'] * protocol['dt_ms'] * 1e-3 if present else None
    packed = np.packbits(events) if present else np.zeros(0, np.uint8)
    h = hashlib.sha256(nodes.tobytes())
    h.update(packed.tobytes() if present else b'none')
    return dict(activation_nodes=nodes, packed_events=packed,
                events_shape=np.array([ticks, len(nodes)], np.int64), events_present=np.array(present)), dict(
                    input_sha256=h.hexdigest(), rng_seed_seq=seq, rng_state_sha256=rng_sha,
                    event_ticks=int(events.any(axis=1).sum()) if present else 0,
                    events_per_cell=events.sum(axis=0).tolist() if present else [])


def validate_manifest(contract, prereg):
    if contract['conditions'] != prereg['conditions']:
        raise ValueError('contract/prereg conditions disagree')
    p = contract['protocol']
    if any(p.get(k) != v for k, v in dict(dt_ms=0.1, duration_ms=1000.0, dynamics='v3', backend='cpu',
                                        pulse_drive=10000.0, silence_drive=-10000.0,
                                        synaptic_scale_mV_per_contact=0.275).items()):
        raise ValueError('frozen dynamics/drive changed')
    if 'shuffled_input' in p or 'timing_probe_set' in p or p['compute']['timeout_s'] != 3500 or p['compute']['budget_wall_hours'] != 1.0:
        raise ValueError('irrelevant protocol or wrong budget')
    if contract['readouts'] != ['GF','TTMn','GFC2','PSI','DLMn','PVLP151']:
        raise ValueError('readouts differ from fixed design')
    gf, ttm, gfc = (contract['sets'][k]['node_index'] for k in ('GF','TTMn','GFC2'))
    if len(gf)!=2 or len(ttm)!=2 or len(gfc)!=10:
        raise ValueError('target set cardinality differs from frozen design')
    fixed=[]
    for name,nodes in (('GF10001',[gf[0]]),('GF10010',[gf[1]]),('GF_BOTH',gf)):
        for arm,clamp in (('intact',[]),('clamp_GFC2',gfc)):
            fixed.append(dict(id=name+'_200_'+arm,activate_nodes=nodes,rate_hz=200,clamp_nodes=clamp,seeds=list(range(8)),arm=arm))
        fixed.append(dict(id=name+'_200_empty_clamp_sham',activate_nodes=nodes,rate_hz=200,clamp_nodes=[],
                          seeds=[0],arm='empty_clamp_sham',requires_clamp_path=True))
    for arm,clamp in (('intact',[]),('clamp_GFC2',gfc)):
        fixed.append(dict(id='NO_INPUT_'+arm,activate_nodes=[],rate_hz=0,clamp_nodes=clamp,seeds=list(range(8)),arm=arm))
    fixed.append(dict(id='TTMn_BOTH_200_direct_control',activate_nodes=ttm,rate_hz=200,clamp_nodes=[],seeds=list(range(8)),arm='direct_activation_control'))
    if contract['conditions']!=fixed:
        raise ValueError('conditions differ from accepted fixed design')
    expected = {}
    for c in contract['conditions']:
        if c['id'] in {k[0] for k in expected}:
            raise ValueError('duplicate condition')
        if c['rate_hz'] not in (0, 200) or isinstance(c['rate_hz'], bool):
            raise ValueError('rate outside fixed design')
        for field in ('activate_nodes', 'clamp_nodes'):
            nodes = c[field]
            if not all(integer(i) and i >= 0 for i in nodes) or len(set(nodes)) != len(nodes):
                raise ValueError('invalid node identity')
        if set(c['activate_nodes']) & set(c['clamp_nodes']):
            raise ValueError('activation/clamp overlap')
        if c['seeds'] != ([0] if c['arm'] == 'empty_clamp_sham' else list(range(8))):
            raise ValueError('seeds differ from fixed design')
        for seed in c['seeds']:
            if not integer(seed):
                raise ValueError('seed must be integer, never bool/float')
            expected[c['id'], seed] = c
    if len(expected) != 75:
        raise ValueError('expected exactly 75 rows')
    if 'expected_inputs' in prereg:
        inputs={c['id']+'_s'+str(seed):input_data(c,seed,p)[1] for (name,seed),c in expected.items()}
        if prereg['expected_inputs']!=inputs:
            raise ValueError('frozen exact input manifest differs')
        for record in prereg['expected_inputs'].values():
            if not integer(record['event_ticks']) or not all(integer(v) for v in record['rng_seed_seq']+record['events_per_cell']):
                raise ValueError('input manifest count/seed types malformed')
    return expected


def verify_sources(prereg, expected_code):
    import subprocess
    if len(expected_code) != 40 or any(c not in '0123456789abcdef' for c in expected_code):
        raise ValueError('external source pin malformed')
    head = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain', '--untracked-files=all'], text=True)
    if head != expected_code or dirty.strip():
        raise ValueError('source is not the clean externally pinned commit')
    for name, digest in prereg['pins']['source_files_sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('source file differs from prereg pin: ' + name)
    h=hashlib.sha256()
    for name in ('brainlab/engine.py','brainlab/brain.py','brainlab/transmitter_policy.py'):
        h.update(name.encode()+b'\0'+(ROOT/name).read_bytes()+b'\0')
    if h.hexdigest()!=prereg['pins']['engine_sha256']:
        raise ValueError('canonical engine identity differs from accepted pin')


def load_plan(contract_path, contract_sha, prereg_path, prereg_sha, expected_code):
    if sha(contract_path) != contract_sha or sha(prereg_path) != prereg_sha:
        raise ValueError('external contract/prereg identity mismatch')
    contract, prereg = strict_json(contract_path), strict_json(prereg_path)
    if prereg['pins']['derived_contract_sha256'] != contract_sha:
        raise ValueError('prereg contract pin mismatch')
    validate_manifest(contract, prereg)
    verify_sources(prereg, expected_code)
    import importlib.metadata
    import platform
    if prereg['runtime_versions']!=dict(python=platform.python_version(),numpy=np.__version__,numba=importlib.metadata.version('numba')):
        raise ValueError('runtime library versions differ from frozen preparation')
    return contract, prereg

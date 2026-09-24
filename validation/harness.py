"""Load a preregistered spec, run its paradigm on the brain, write a receipt.

    python -m validation run validation/specs/optomotor_v1.json --out outputs/validation/<stamp>

A receipt carries everything needed to judge and reproduce the run: the spec
and its sha256 (and the commit that last touched it), the graph identity and
transmitter policy, the dynamics declaration and its pin, the brain backend,
the code revision, seeds, per-trial results, every gate with its statistic,
the physiology checks, and the verdict.

Verdict rule (``VERDICTS``): validity is the behaviour gate AND the
physiology gate (owner decision, 2026-09-24).  A run can only be
``PASS``/``FAIL`` if it is confirmatory: real graph, spec status
``preregistered``, spec seeds unchanged, clean committed source tree.  A
PASS whose gating targets or bounds rest on any value marked unverified is
``PASS_PROVISIONAL``.  Synthetic graphs never produce a scientific verdict.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import stats

ROOT = Path(__file__).resolve().parents[1]
SPEC_SCHEMA = 'flybrain.validation-spec.v1'
BOUNDS_SCHEMA = 'flybrain.firing-rate-bounds.v1'
RECEIPT_SCHEMA = 'flybrain.validation-receipt.v1'
SPEC_STATUSES = ('draft', 'preregistered', 'retired')
REQUIRED_SPEC_KEYS = ('schema', 'id', 'paradigm', 'status', 'declared_at', 'dynamics', 'stimulus', 'encoder',
                      'seeds', 'behaviour', 'physiology', 'citations', 'decision_rule')
VERDICTS = {
    'PASS': 'confirmatory; every behaviour gate and physiology check passed; all gating values verified',
    'PASS_PROVISIONAL': 'confirmatory; everything passed, but a gating target or bound is marked unverified',
    'FAIL': 'confirmatory; at least one behaviour gate or physiology check failed',
    'INCONCLUSIVE': 'confirmatory; a gate could not be evaluated (e.g. undefined interval)',
    'EXPLORATORY': 'not confirmatory (draft spec, changed seeds, or uncommitted code); results only',
    'SYNTHETIC_PLUMBING_ONLY': 'synthetic test graph; exercises the code path, no scientific claim',
}
GATE_TYPES = ('ci_lower_above', 'ci_upper_below', 'ci_within', 'ratio_ci_lower_above', 'ratio_ci_within',
              'all_equal', 'slope_ci_upper_below', 'proportion_ci_lower_above', 'proportion_ci_upper_below')


class SpecError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------
def load_spec(path: Path) -> dict:
    raw = Path(path).read_bytes()
    spec = json.loads(raw)
    check_spec(spec)
    spec['_sha256'] = hashlib.sha256(raw).hexdigest()
    spec['_path'] = str(Path(path).resolve())
    return spec


def check_spec(spec: dict) -> None:
    missing = [k for k in REQUIRED_SPEC_KEYS if k not in spec]
    if missing:
        raise SpecError(f'spec missing keys: {missing}')
    if spec['schema'] != SPEC_SCHEMA:
        raise SpecError(f"spec schema {spec['schema']!r} != {SPEC_SCHEMA!r}")
    if spec['status'] not in SPEC_STATUSES:
        raise SpecError(f"status {spec['status']!r} not in {SPEC_STATUSES}")
    cites = spec['citations']
    for gate in spec['behaviour']['gates']:
        for key in ('id', 'type', 'claim', 'basis', 'verified', 'sources'):
            if key not in gate:
                raise SpecError(f"gate {gate.get('id')} missing {key}")
        if gate['type'] not in GATE_TYPES:
            raise SpecError(f"gate {gate['id']} type {gate['type']!r} unknown")
        for src in gate['sources']:
            if src not in cites:
                raise SpecError(f"gate {gate['id']} cites {src!r}, not in citations")
    for key, cite in cites.items():
        for field_ in ('citation', 'doi', 'doi_status', 'read'):
            if field_ not in cite:
                raise SpecError(f'citation {key} missing {field_}')


def load_bounds(spec: dict) -> dict:
    path = (Path(spec['_path']).parent / spec['physiology']['bounds_file']).resolve()
    raw = path.read_bytes()
    bounds = json.loads(raw)
    if bounds.get('schema') != BOUNDS_SCHEMA:
        raise SpecError(f'{path}: schema {bounds.get("schema")!r} != {BOUNDS_SCHEMA!r}')
    bounds['_sha256'] = hashlib.sha256(raw).hexdigest()
    bounds['_path'] = str(path)
    return bounds


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------
def evaluate_gate(gate: dict, samples: dict, boot_seed: int, n_boot: int) -> dict:
    t = gate['type']
    out = dict(id=gate['id'], type=t, claim=gate['claim'], verified=gate['verified'])
    try:
        if t in ('ci_lower_above', 'ci_upper_below', 'ci_within'):
            s = stats.bootstrap_mean(samples[gate['sample']], seed=boot_seed, n_boot=n_boot)
            lo, hi = s['ci']
            out['statistic'] = s
            if lo is None:
                out['passed'] = None
            elif t == 'ci_lower_above':
                out['passed'] = lo > gate['threshold']
            elif t == 'ci_upper_below':
                out['passed'] = hi < gate['threshold']
            else:
                a, b = gate['interval']
                out['passed'] = a <= lo and hi <= b
        elif t in ('ratio_ci_lower_above', 'ratio_ci_within'):
            if 'sample' in gate:
                num, den = samples[gate['sample']]['num'], samples[gate['sample']]['den']
            else:
                num, den = samples[gate['num']], samples[gate['den']]
            s = stats.bootstrap_ratio_of_means(num, den, seed=boot_seed, n_boot=n_boot)
            out['statistic'] = s
            lo, hi = s['ci']
            if lo is None:
                out['passed'] = None
            elif t == 'ratio_ci_lower_above':
                out['passed'] = lo > gate['threshold']
            else:
                a, b = gate['interval']
                out['passed'] = a <= lo and hi <= b
        elif t == 'all_equal':
            values = samples[gate['sample']]
            out['statistic'] = dict(values=values)
            out['passed'] = all(v == gate['value'] for v in values)
        elif t == 'slope_ci_upper_below':
            s = stats.bootstrap_slope(samples[gate['sample']], seed=boot_seed, n_boot=n_boot)
            s['spearman_condition_means'] = stats.spearman(s['x'], s['means'])
            out['statistic'] = s
            out['passed'] = s['ci'][1] < gate['threshold']
        elif t in ('proportion_ci_lower_above', 'proportion_ci_upper_below'):
            kn = samples[gate['sample']]
            if isinstance(kn, list):
                kn = dict(k=int(sum(kn)), n=len(kn))
            s = stats.proportion(kn['k'], kn['n'])
            out['statistic'] = s
            lo, hi = s['ci']
            out['passed'] = (lo > gate['threshold']) if t == 'proportion_ci_lower_above' else (hi < gate['threshold'])
    except (KeyError, ValueError) as exc:
        out['passed'] = None
        out['error'] = f'{type(exc).__name__}: {exc}'
    return out


# ---------------------------------------------------------------------------
# Physiology
# ---------------------------------------------------------------------------
def evaluate_physiology(spec: dict, bounds: dict, rates: dict) -> list:
    """Each check compares the across-trial mean population rate with a bound.

    ``rates`` maps condition -> [per-trial SpikeLedger summaries].  A check
    names the condition(s), the window ('baseline' = spontaneous, 'stimulus'
    = driven) and either a population or the whole brain.
    """
    table = bounds['bounds']
    checks = []
    for check in spec['physiology']['checks']:
        bound = table[check['bound']]
        state = 'spontaneous_hz' if check['window'] == 'baseline' else 'driven_hz'
        interval = bound.get(state)
        values = []
        for condition in check['conditions']:
            for trial in rates.get(condition, []):
                window = trial.get(check['window'])
                if window is None:
                    continue
                if check.get('population') == '__brain__':
                    values.append(window['brain'][check.get('measure', 'mean_hz')])
                elif check['population'].endswith('*'):
                    # Cell-weighted mean over every population with this prefix (e.g. all ORN_*).
                    prefix = check['population'][:-1]
                    pops = [p for name, p in window['populations'].items()
                            if name.startswith(prefix) and p['n'] and p['mean_hz'] is not None]
                    if pops:
                        values.append(sum(p['mean_hz'] * p['n'] for p in pops) / sum(p['n'] for p in pops))
                else:
                    pop = window['populations'].get(check['population'])
                    if pop and pop['n'] and pop['mean_hz'] is not None:
                        values.append(pop[check.get('measure', 'mean_hz')])
        result = dict(id=check['id'], population=check['population'], window=check['window'],
                      bound=check['bound'], interval_hz=interval, verified=bound['verified'],
                      n_trials=len(values))
        if interval is None or not values:
            result.update(passed=None, reason='no bound for this state' if interval is None else 'no data')
        else:
            mean = float(np.mean(values))
            lo, hi = interval
            result.update(mean_hz=mean, min_trial_hz=float(np.min(values)), max_trial_hz=float(np.max(values)),
                          passed=bool(lo <= mean <= hi))
        checks.append(result)
    return checks


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def code_revision() -> dict:
    env = dict(os.environ, GIT_OPTIONAL_LOCKS='0')

    def git(*args):
        try:
            r = subprocess.run(['git', '-C', str(ROOT), *args], env=env, capture_output=True, text=True, timeout=10)
            return r.stdout.strip() if r.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None
    status = git('--no-optional-locks', 'status', '--porcelain', '--untracked-files=no')
    return dict(commit=git('rev-parse', 'HEAD'), dirty=(bool(status) if status is not None else None),
                describe=git('describe', '--always', '--dirty'))


def spec_commit(spec_path: str) -> Optional[dict]:
    try:
        r = subprocess.run(['git', '-C', str(ROOT), 'log', '-1', '--format=%H %cI', '--', spec_path],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            sha, date = r.stdout.strip().split(' ', 1)
            return dict(commit=sha, committed_at=date)
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def cuda_device() -> Optional[str]:
    try:
        from numba import cuda
        if cuda.is_available():
            return cuda.get_current_device().name.decode() if isinstance(
                cuda.get_current_device().name, bytes) else str(cuda.get_current_device().name)
    except Exception:   # noqa: BLE001 - optional information only
        return None
    return None


def decide(gates: list, physiology: list, reasons: list, *, synthetic: bool):
    """Return ``(verdict, components, unverified_ids)``; see :data:`VERDICTS`."""
    def component(values):
        if values and all(v is True for v in values):
            return 'PASS'
        return 'FAIL' if any(v is False for v in values) else 'INCONCLUSIVE'
    components = dict(behaviour=component([g['passed'] for g in gates]),
                      physiology=component([c['passed'] for c in physiology]))
    unverified = sorted({g['id'] for g in gates if g['verified'] is not True}
                        | {c['id'] for c in physiology if c['verified'] is not True})
    if synthetic:
        verdict = 'SYNTHETIC_PLUMBING_ONLY'
    elif reasons:
        verdict = 'EXPLORATORY'
    elif 'FAIL' in components.values():
        verdict = 'FAIL'
    elif 'INCONCLUSIVE' in components.values():
        verdict = 'INCONCLUSIVE'
    else:
        verdict = 'PASS_PROVISIONAL' if unverified else 'PASS'
    return verdict, components, unverified


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
@dataclass
class RunContext:
    shared: object
    cells: object
    seeds: list
    synthetic: bool
    make_runner: Callable
    connectome_dir: Optional[Path] = None
    log: Callable = print
    runners: list = field(default_factory=list)


def paradigm_module(name: str):
    from .paradigms import looming, optomotor, tmaze_odor
    return dict(optomotor=optomotor, looming=looming, tmaze_odor=tmaze_odor)[name]


def load_graph(spec: dict, *, synthetic: bool, graph_dir=None, connectome_dir=None):
    """Return ``(shared, cells, transmitter_report)`` for the spec's dynamics."""
    from .cells import CellTable
    dyn = spec['dynamics']
    if synthetic:
        from .synthetic import typed_synthetic_graph
        shared, cells = typed_synthetic_graph(seed=dyn.get('synthetic_seed', 0))
        return shared, cells, None
    from experiment_registry import SharedGraph
    shared = SharedGraph.load(graph_dir, connectome_dir)
    report = None
    policy = dyn.get('transmitter_policy')
    if policy:
        from brainlab import transmitter_policy as tp
        shared, report = tp.apply_to_shared(shared, policy=policy, unclear_mode=dyn.get('unclear_mode', 'excitatory'),
                                            connectome_dir=connectome_dir)
    return shared, CellTable.from_connectome(connectome_dir), report


def run(spec_path, out_dir: Path, *, synthetic: bool = False, backend: str = 'auto', seeds=None,
        graph_dir=None, connectome_dir=None, log: Callable = print) -> dict:
    spec = load_spec(spec_path)
    bounds = load_bounds(spec)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    dyn = spec['dynamics']
    # Process-wide as well, so anything that reads the active declaration agrees.
    os.environ['NEUROFLY_LIF_DYNAMICS'] = dyn['version']
    from brainlab.graph_identity import DYNAMICS_VERSIONS, dynamics_pin
    from .runner import BrainRunner

    started = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    clock = time.perf_counter()
    shared, cells, policy_report = load_graph(spec, synthetic=synthetic, graph_dir=graph_dir,
                                              connectome_dir=connectome_dir)
    load_s = time.perf_counter() - clock
    run_seeds = list(spec['seeds'] if seeds is None else seeds)
    backends_used = []

    def make_runner(graph):
        runner = BrainRunner(graph, dynamics=dyn['version'], backend=backend, e_inh_mV=dyn.get('e_inh_mV'))
        backends_used.append(runner.describe())
        return runner

    ctx = RunContext(shared=shared, cells=cells, seeds=run_seeds, synthetic=synthetic, make_runner=make_runner,
                     connectome_dir=connectome_dir, log=log)
    clock = time.perf_counter()
    result = paradigm_module(spec['paradigm']).run(spec, ctx)
    wall_s = time.perf_counter() - clock

    analysis = spec['behaviour'].get('analysis', {})
    boot_seed, n_boot = analysis.get('bootstrap_seed', 20260924), analysis.get('bootstrap_resamples', 10000)
    gates = [evaluate_gate(g, result['samples'], boot_seed, n_boot) for g in spec['behaviour']['gates']]
    reported = [evaluate_gate(g, result['samples'], boot_seed, n_boot) for g in spec['behaviour'].get('reported', [])]
    physiology = evaluate_physiology(spec, bounds, result['rates'])

    revision = code_revision()
    committed = spec_commit(spec['_path'])
    reasons = []
    if synthetic:
        reasons.append('synthetic graph')
    if spec['status'] != 'preregistered':
        reasons.append(f"spec status is {spec['status']!r}")
    if seeds is not None and list(seeds) != list(spec['seeds']):
        reasons.append('seeds differ from the spec')
    if revision['dirty'] is not False:
        reasons.append('source tree not clean (or unknown)')
    if committed is None:
        reasons.append('spec file is not committed')
    verdict, components, unverified = decide(gates, physiology, reasons, synthetic=synthetic)

    public_spec = {k: v for k, v in spec.items() if not k.startswith('_')}
    receipt = dict(
        schema=RECEIPT_SCHEMA, verdict=verdict, verdict_meaning=VERDICTS[verdict], components=components,
        not_confirmatory_because=reasons, unverified_gating_items=unverified,
        spec=dict(id=spec['id'], path=os.path.relpath(spec['_path'], ROOT), sha256=spec['_sha256'],
                  status=spec['status'], commit=committed, content=public_spec),
        bounds=dict(path=os.path.relpath(bounds['_path'], ROOT), sha256=bounds['_sha256']),
        graph=dict(identity=shared.identity.to_dict(), transmitter_policy=policy_report, load_s=load_s),
        dynamics=dict(version=dyn['version'], pin=dynamics_pin(dyn['version']),
                      declaration=DYNAMICS_VERSIONS[dyn['version']], e_inh_mV=dyn.get('e_inh_mV')),
        brain=dict(requested_backend=backend, runners=backends_used, cuda_device=cuda_device()),
        code=revision, seeds=run_seeds, io=result['io'], paradigm_extra=result.get('extra'),
        gates=gates, reported=reported, physiology=physiology,
        compute=dict(sim_s=result['sim_ms'] / 1000, wall_s=wall_s,
                     sim_s_per_wall_s=(result['sim_ms'] / 1000 / wall_s) if wall_s > 0 else None,
                     peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        host=platform.node(), started_at=started, finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    (out_dir / 'receipt.json').write_text(json.dumps(_jsonable(receipt), indent=2) + '\n')
    (out_dir / 'trials.json').write_text(json.dumps(_jsonable(result['trials'])) + '\n')
    log(f"VERDICT {verdict}  behaviour {components['behaviour']}  physiology {components['physiology']}")
    for g in gates:
        log(f"  gate {g['id']}: {g['passed']}")
    for c in physiology:
        log(f"  physiology {c['id']}: {c['passed']} ({c.get('mean_hz', c.get('reason'))})")
    if result.get('extra', {}).get('wp5_rule'):
        log(f"  WP5 rule (for comparison with v1): {result['extra']['wp5_rule']['verdict']}")
    if reasons:
        log('  not confirmatory: ' + '; '.join(reasons))
    return receipt


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value

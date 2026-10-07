"""Cohort runner: B independent fixed-v3 brains in the existing Arena + io_map loop.

Each fly k has seed ``seed_base + k``, its own ``Arena`` (the existing optomotor
paradigm world, seeded with that seed), its own ``OptomotorEncoder`` RNG
(``default_rng(seed)``, as the daemon's GraphArenaController builds it), its own
``DNa02YawDecoder`` and its own input cursor.  The graph and io map are loaded
once and shared read-only.

One arena step (``step_ms`` of brain time, 20 ms by default = the daemon's
graph step) is batched across flies::

    for every fly: slip_k = drum velocity - own yaw   (exactly what the Arena
                   will hand its graph controller this step)
                   drive[k] = OptomotorEncoder_k.encode(slip_k, contrast_k)
    counts = engine.step(drive, step_ms / 0.1 ms)      (ONE call for all flies)
    for every fly: yaw_k = DNa02YawDecoder_k.decode(counts[k])
                   Arena_k.step(dt): its graph controller receives the Arena's own
                   optomotor sensory packet, which must equal slip_k / contrast_k
                   bit for bit, and returns yaw_k (tethered, speed 0)

The batched encode -> graph -> decode is the same computation as
``brainlab.io_map.OptomotorLoop.step`` per fly.  A mismatch between the slip the
Arena delivers and the slip that was encoded is a hard error; no fixed or
substitute stimulus is ever used.

Scientific disclosure (also written into cohort_manifest.json as
``provenance.graph_io_declaration`` and printed by ``cohort run``): the encoder
injects motion-selective drive directly into T4/T5, bypassing photoreceptor
motion computation, and the DNa02 yaw decoder is an engineered linear readout.
Nothing here claims native motion computation or validated fly behaviour.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from .api import COHORT_SCHEMA, TICK_MS, CohortEngine, CpuLoopCohortEngine
from . import store

DEFAULT_STEP_MS = 20.0
DYNAMICS_ID = 'v3_fixed'
# Modules whose loaded files identify the executing writer (provenance.running_code_identity
# hashes only those actually imported; an installed wheel is hashed against its RECORD).
COHORT_WRITER_MODULES = ('brainlab.cohort', 'brainlab.cohort.api', 'brainlab.cohort.runner',
                         'brainlab.cohort.store', 'brainlab.cohort.gpu', 'brainlab.io_map', 'maze')
DISCLOSURE = ('Declared I/O, not native computation: the OptomotorEncoder injects motion-selective drive '
              'directly into T4/T5 (direction imposed by the encoder; photoreceptor motion computation is '
              'bypassed) and the DNa02YawDecoder is an engineered linear readout. Fixed weights, no learning. '
              'No claim of validated fly behaviour.')
ASSAYS = ('optomotor',)
RESUME_HELP = ('Verify and continue a cohort directory: same-engine resume is byte-identical; '
               'CPU<->GPU continuation is NOT exact, only verified within the contract bounds')
RESUME_DESCRIPTION = ('Verify a cohort directory and continue it. Resume on the same engine (CPU, or GPU '
                      'on the same device) is byte-identical to an uninterrupted run. CPU<->GPU '
                      'continuation is NOT exact: it is verified only within the preregistered contract '
                      'bounds (g relative error <= 1e-6 (preregistered), 0 spike mismatches over ticks '
                      '100-199); observed max g relative error about 4.53e-7.')


class CohortError(RuntimeError):
    pass


class EngineUnavailable(CohortError):
    pass


def writer_identity() -> dict:
    """Identity of the code executing NOW (never inherited from a checkpoint)."""
    import provenance
    return provenance.running_code_identity(tuple(provenance.WRITER_MODULES) + COHORT_WRITER_MODULES)


def _payload(writer: dict) -> dict:
    ident = writer.get('ident') or (f"source:{writer.get('commit')}" + ('+dirty' if writer.get('dirty') else ''))
    return {'ident': ident, 'version': writer.get('version')}


def dynamics_signature(engine: CohortEngine) -> dict:
    """Canonical signature of the model the RUNNING code executes: values + sha256.

    Read from the live modules and the live engine (never from a checkpoint), so a
    resume under different constants is refused even when the code version is equal.
    """
    from brainlab import engine as lif
    from brainlab.cohort.gpu import CPU_DELIVERY, DELIVERY
    from brainlab.graph_identity import dynamics_pin
    from brainlab.transmitter_policy import POLICY_V3
    if hasattr(engine, 'brains'):
        live = engine.brains[0]
        e_inh, g_exc, g_inh = live.e_inh_mV, live.g_unit_exc, live.g_unit_inh
    else:
        e_inh, g_exc, g_inh = (getattr(engine, 'e_inh', None), getattr(engine, 'g_unit_exc', None),
                               getattr(engine, 'g_unit_inh', None))
    values = {
        'dynamics': 'v3', 'plastic': False, 'tick_ms': TICK_MS,
        'constants': {name: float(getattr(lif, name)) for name in (
            'V_REST_MV', 'V_RESET_MV', 'V_THRESHOLD_MV', 'TAU_M_MS', 'TAU_SYN_MS', 'REFRACTORY_MS',
            'DELAY_MS', 'E_EXC_MV', 'E_INH_MV')},
        'g_unit': {'per_weight': float(lif.G_UNIT_PER_WEIGHT), 'exc_v3': float(lif.G_UNIT_EXC_V3),
                   'engine_exc': None if g_exc is None else float(g_exc),
                   'engine_inh': None if g_inh is None else float(g_inh)},
        'reversal': {'E_EXC_MV': float(lif.E_EXC_MV), 'engine_E_INH_MV': None if e_inh is None else float(e_inh)},
        'transmitter_policy': POLICY_V3,
        'declared_dynamics_pin': dynamics_pin('v3'),
        # Arrival arithmetic of BOTH engines (one signature, so cross-engine
        # resume stays possible). Cohorts from the earlier fixed-point GPU
        # kernel carried 'gpu_fixed_point_scale' instead and are refused.
        'arrival_arithmetic': {'cpu': CPU_DELIVERY, 'gpu': DELIVERY},
    }
    return {'values': values, 'sha256': _sha256_json(values)}


def _signature_diff(a: dict, b: dict, prefix: str = '') -> str:
    out = []
    for key in sorted(set(a) | set(b)):
        x, y = a.get(key), b.get(key)
        if isinstance(x, dict) and isinstance(y, dict):
            sub = _signature_diff(x, y, f'{prefix}{key}.')
            if sub:
                out.append(sub)
        elif x != y:
            out.append(f'{prefix}{key}: recorded {x!r}, running {y!r}')
    return '; '.join(out)


def graph_io_disclosure() -> dict:
    import provenance
    return {'graph_io': provenance.graph_io_declaration(include_config=True), 'notice': DISCLOSURE}


def _package_version() -> str:
    try:
        from neurofly import __version__
        return str(__version__)
    except Exception:
        return 'unknown'


def _sha256_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Graph: loaded once, shared read-only
# ---------------------------------------------------------------------------
@dataclass
class CohortGraph:
    arrays: dict
    io: object                      # brainlab.io_map.OptomotorIOMap
    graph_sha256: str
    io_map_sha256: str
    identity: dict
    source: dict
    synthetic: bool = False

    @property
    def n(self) -> int:
        return int(len(self.arrays['ptr']) - 1)


def load_real_graph(graph_dir=None, connectome_dir=None) -> CohortGraph:
    """Verified MaleCNS graph with the v3 transmitter policy, plus the pinned WP5 io map."""
    from experiment_registry import SharedGraph
    from brainlab.io_map import resolve_optomotor_io
    shared = SharedGraph.load_for_dynamics(graph_dir, connectome_dir, dynamics='v3')
    io = resolve_optomotor_io(connectome_dir)
    return CohortGraph(arrays=shared.arrays, io=io, graph_sha256=shared.identity.graph_sha256,
                       io_map_sha256=io.sha256, identity=shared.identity.to_dict(),
                       source={'kind': 'malecns_v1'}, synthetic=False)


def synthetic_graph(n: int = 64, k_out: int = 6, seed: int = 0, n_per: int = 5) -> CohortGraph:
    """SYNTHETIC TEST GRAPH (tests only): random CSR graph and a small io map
    laid out like ``tests/test_wp5_optomotor.small_io``.  Never MaleCNS."""
    from brainlab.graph_identity import synthetic_test_graph
    from brainlab.io_map import OptomotorIOMap
    arrays, identity, _ = synthetic_test_graph(n=n, k_out=k_out, seed=seed)
    pops = {}
    for k, name in enumerate(('ftb_L', 'btf_L', 'ftb_R', 'btf_R')):
        pops[name] = np.arange(20 + k * n_per, 20 + (k + 1) * n_per, dtype=np.int64) % n
    # Read DNa02 from one directly driven cell per eye so the closed loop
    # (yaw -> retinal slip -> drive) is exercised on this tiny test graph.
    pops['DNa02_L'] = pops['ftb_L'][:1].copy()
    pops['DNa02_R'] = pops['ftb_R'][:1].copy()
    monitors = {'HS_L': np.array([1], np.int64), 'HS_R': np.array([2], np.int64)}
    digest = _sha256_json({k: [int(i) for i in v] for k, v in {**pops, **monitors}.items()})
    io = OptomotorIOMap(populations=pops, source_ids={k: [int(i) for i in v] for k, v in pops.items()},
                        matched_counts={'T4': n_per, 'T5': 0}, available_counts={},
                        monitors=monitors, sha256=digest)
    from dataclasses import asdict
    return CohortGraph(arrays=arrays, io=io, graph_sha256=identity.graph_sha256, io_map_sha256=digest,
                       identity=asdict(identity),
                       source={'kind': 'synthetic', 'n': n, 'k_out': k_out, 'seed': seed, 'n_per': n_per},
                       synthetic=True)


def graph_from_source(source: dict, *, graph_dir=None, connectome_dir=None,
                      allow_synthetic: bool = False) -> CohortGraph:
    if source.get('kind') == 'synthetic':
        if not allow_synthetic:
            raise CohortError('This cohort ran on a SYNTHETIC TEST GRAPH; resuming it needs the explicit '
                              'test option (--test-synthetic-graph)')
        return synthetic_graph(**{k: source[k] for k in ('n', 'k_out', 'seed', 'n_per')})
    if source.get('kind') == 'malecns_v1':
        return load_real_graph(graph_dir, connectome_dir)
    raise CohortError(f'Unknown graph source {source!r}; refused')


# ---------------------------------------------------------------------------
# Engine selection
# ---------------------------------------------------------------------------
def make_engine(name: str, arrays: dict, n_flies: int) -> CohortEngine:
    if name == 'cpu':
        return CpuLoopCohortEngine(arrays, n_flies)
    if name == 'gpu':
        try:
            from brainlab.cohort.gpu import GpuCohortEngine
        except Exception as exc:
            raise EngineUnavailable(f'--engine gpu is unavailable: cannot import brainlab.cohort.gpu '
                                    f'({type(exc).__name__}: {exc}). Use --engine cpu.') from exc
        try:
            return GpuCohortEngine(arrays, n_flies)
        except Exception as exc:
            raise EngineUnavailable(f'--engine gpu is unavailable on this machine '
                                    f'({type(exc).__name__}: {exc}). Use --engine cpu.') from exc
    raise EngineUnavailable(f'Unknown engine {name!r}; expected cpu or gpu')


# ---------------------------------------------------------------------------
# One fly's world: Arena + encoder + decoder + cursor
# ---------------------------------------------------------------------------
class _BatchedGraphController:
    """The Arena's graph controller for one cohort fly.

    The brain step for this arena step has already happened (batched across
    flies); this hands the Arena the decoded command after checking that the
    Arena's own optomotor sensory packet is exactly what was encoded.
    """

    def __init__(self):
        self.pending = None

    def __call__(self, fly=None, sensory=None, dt=0.02, **kwargs):
        if self.pending is None:
            raise CohortError('Arena asked for a motor command with no batched brain step pending')
        slip, contrast, record = self.pending
        self.pending = None
        got = (kwargs.get('optomotor_slip_rad_s'), kwargs.get('optomotor_contrast'))
        if got[0] is None or float(got[0]) != slip or float(got[1]) != contrast:
            raise CohortError(f'Arena delivered optomotor input {got} but the brain was driven with '
                              f'({slip}, {contrast}); the cohort loop is out of step - refused, no substitute')
        yaw = float(record['yaw_rad_s'])
        return {'halted': False, 'forward_speed': 0.0, 'yaw_rate': yaw, 'motor_source': 'graph',
                'controller_fault': None, 'state': 'OPTOMOTOR-TETHERED',
                'dn_rates': {'dna02_l': record['rate_l'], 'dna02_r': record['rate_r'],
                             'dnp09': None, 'mdn': None, 'gf': None},
                'optomotor': {'slip_rad_s': slip, 'contrast': contrast, 'yaw_rad_s': yaw,
                              'contributions': record['contributions']},
                'engineered_assistance_enabled': False, 'engineered_assistance_applied': []}


def _phase_sha(encoder) -> str:
    h = hashlib.sha256()
    for name in sorted(encoder.phase):
        h.update(name.encode() + np.ascontiguousarray(encoder.phase[name], dtype='<f8').tobytes())
    return h.hexdigest()


class FlyWorld:
    def __init__(self, fly_id: int, seed: int, graph: CohortGraph, assay: str, step_ms: float):
        from arena import Arena
        from brainlab.io_map import DNa02YawDecoder, OptomotorEncoder
        if assay not in ASSAYS:
            raise CohortError(f'Assay {assay!r} is not supported by the cohort runner (supported: {ASSAYS})')
        self.fly_id, self.seed, self.assay, self.step_ms = int(fly_id), int(seed), assay, float(step_ms)
        self.io = graph.io
        self.rng = np.random.default_rng(self.seed)
        self.encoder = OptomotorEncoder(graph.io, self.rng)
        self.decoder = DNa02YawDecoder(graph.io)
        self.controller = _BatchedGraphController()
        self.arena = Arena(paradigm=assay, brain_type='modular', seed=self.seed, num_flies=1, num_predators=0,
                           fly_ablations=[{'ablate_mb': True}], controller_backend='connectome-fixed',
                           graph_controller=self.controller)
        self.cursor = 0                       # arena steps completed == input cursor
        self.total_spikes = 0
        self.yaw_sum = 0.0
        self.phase_sha256 = _phase_sha(self.encoder)

    # The Arena computes slip = radians(drum velocity) - fly.angular_velocity and
    # passes the paradigm contrast (arena.compute_steering); the controller checks it.
    def next_input(self):
        p = self.arena.paradigm
        slip = math.radians(float(p.sample_stimuli(self.arena.fly.pos.to_tuple(), self.arena.fly.heading)
                                  ['drum_velocity_deg_s'])) - float(getattr(self.arena.fly, 'angular_velocity', 0.0))
        contrast = float(p.sample_stimuli(self.arena.fly.pos.to_tuple(), self.arena.fly.heading).get('contrast', 1.0))
        return slip, contrast

    def schedule(self) -> dict:
        p = self.arena.paradigm
        return {'assay': self.assay, 'drum_velocity_deg_s': float(p.drum_velocity_deg_s),
                'contrast': float(p.contrast), 'step_ms': self.step_ms,
                'encoder': self.encoder.describe(), 'decoder': self.decoder.describe(),
                'deliver_sensory': True, 'silence': []}

    def world_summary(self) -> dict:
        f = self.arena.fly
        return {'x': f.pos.x, 'y': f.pos.y, 'heading': f.heading, 'speed': f.speed,
                'angular_velocity': f.angular_velocity,
                'drum_angle_deg': getattr(self.arena.paradigm, 'drum_angle_deg', None)}


def schedule_sha256(world: FlyWorld) -> str:
    return _sha256_json(world.schedule())


# ---------------------------------------------------------------------------
# Cohort run / resume
# ---------------------------------------------------------------------------
@dataclass
class _Output:
    path: Path
    fh: object = None
    hasher: object = field(default_factory=hashlib.sha256)
    nbytes: int = 0

    def write(self, record: dict):
        line = (json.dumps(record, sort_keys=True, separators=(',', ':')) + '\n').encode()
        self.fh.write(line)
        self.hasher.update(line)
        self.nbytes += len(line)

    def sync(self):
        self.fh.flush()
        os.fsync(self.fh.fileno())


class Cohort:
    def __init__(self, root: Path, graph: CohortGraph, engine: CohortEngine, worlds: List[FlyWorld],
                 manifest: dict, progress: Callable[[str], None]):
        self.root, self.graph, self.engine, self.worlds = Path(root), graph, engine, worlds
        self.manifest, self.progress = manifest, progress
        self.ticks_per_step = int(round(worlds[0].step_ms / TICK_MS))
        self.outputs: List[_Output] = []
        self.parents: List[Optional[str]] = [None] * len(worlds)
        self.drive = np.zeros((engine.n_flies, engine.n), dtype=np.float32)
        self.writer = writer_identity()
        self.signature = dynamics_signature(engine)

    # -- identity common to every checkpoint
    def _identity(self) -> dict:
        return {'graph_sha256': self.graph.graph_sha256, 'io_map_sha256': self.graph.io_map_sha256,
                'dynamics': DYNAMICS_ID, 'engine_backend_id': self.engine.backend_id,
                'payload': _payload(self.writer), 'writer': self.writer,
                'dynamics_signature': self.signature}

    def fly_dir(self, k: int) -> Path:
        return self.root / 'flies' / f'fly-{k:02d}'

    def open_outputs(self):
        for k in range(len(self.worlds)):
            path = self.fly_dir(k) / 'steps.jsonl'
            path.parent.mkdir(parents=True, exist_ok=True)
            out = _Output(path)
            if path.exists():
                data = path.read_bytes()
                out.hasher.update(data)
                out.nbytes = len(data)
            out.fh = open(path, 'ab')
            self.outputs.append(out)

    def close_outputs(self):
        for out in self.outputs:
            if out.fh is not None:
                out.sync()
                out.fh.close()
                out.fh = None

    def step(self):
        worlds, io = self.worlds, self.graph.io
        self.drive.fill(0.0)
        inputs = []
        for k, w in enumerate(worlds):
            slip, contrast = w.next_input()
            t_ms = w.cursor * w.step_ms
            totals = w.encoder.encode(self.drive[k], t_ms, slip, contrast)
            inputs.append((slip, contrast, totals))
        counts = self.engine.step(self.drive, self.ticks_per_step)
        for k, w in enumerate(worlds):
            slip, contrast, totals = inputs[k]
            c = counts[k]
            motor = w.decoder.decode(c, w.step_ms)
            w.controller.pending = (slip, contrast, motor)
            w.arena.step(w.step_ms / 1000.0)
            if w.controller.pending is not None:
                raise CohortError(f'fly {k}: the Arena did not consume the graph command this step')
            w.cursor += 1
            total = int(c.sum())
            w.total_spikes += total
            w.yaw_sum += float(motor['yaw_rad_s'])
            pops = {name: int(c[idx].sum()) for name, idx in io.populations.items()}
            pops.update({name: int(c[idx].sum()) for name, idx in io.monitors.items()})
            self.outputs[k].write({
                'step': w.cursor, 't_ms': w.cursor * w.step_ms, 'slip_rad_s': slip, 'contrast': contrast,
                'encoder_totals': totals, 'population_spikes': pops, 'total_spikes': total,
                'motor': {key: motor[key] for key in ('yaw_rad_s', 'rate_l', 'rate_r', 'spikes_l', 'spikes_r')},
                'world': w.world_summary()})

    def checkpoint(self):
        tick = self.worlds[0].cursor * self.ticks_per_step
        entries = {e['fly_id']: e for e in self.manifest['flies']}
        for k, w in enumerate(self.worlds):
            out = self.outputs[k]
            out.sync()
            meta = dict(self._identity(), fly_id=w.fly_id, seed=w.seed,
                        rng={'bit_generator': type(w.rng.bit_generator).__name__,
                             'state': w.rng.bit_generator.state},
                        input_cursor=w.cursor, stimulus_schedule_sha256=schedule_sha256(w),
                        encoder_phase_sha256=w.phase_sha256,
                        decoder={'rate_l': w.decoder.rate_l, 'rate_r': w.decoder.rate_r},
                        assay=w.assay,
                        # A string: the world encoding is order-sensitive (aliases), so it must
                        # not be re-sorted with the rest of the metadata.
                        world_state=json.dumps(w.arena.snapshot_world()),
                        tick=tick, sim_ms=tick * TICK_MS, step_ms=w.step_ms,
                        totals={'spikes': w.total_spikes, 'yaw_sum_rad_s': w.yaw_sum},
                        outputs={'steps_jsonl_bytes': out.nbytes, 'steps_jsonl_sha256': out.hasher.hexdigest()},
                        parent_checkpoint_sha256=self.parents[k])
            data = store.encode_checkpoint(meta, self.engine.read_state(k))
            name = store.checkpoint_name(w.fly_id, tick)
            digest = store.atomic_write_bytes(self.root / 'ckpt' / name, data)
            self.parents[k] = digest
            chain = entries[w.fly_id].setdefault('checkpoints', [])
            chain.append({'file': f'ckpt/{name}', 'sha256': digest, 'tick': tick})
        self.manifest['tick'] = tick
        self.manifest['sim_ms'] = tick * TICK_MS
        store.atomic_write_json(store.manifest_path(self.root), self.manifest)

    def run_steps(self, n_steps: int, checkpoint_every_steps: int, stop_after_steps: Optional[int] = None):
        t0 = time.perf_counter()
        done = 0
        last_report = t0
        for _ in range(n_steps):
            if stop_after_steps is not None and done >= stop_after_steps:
                raise KeyboardInterrupt('test hook: simulated crash')
            self.step()
            done += 1
            if self.worlds[0].cursor % checkpoint_every_steps == 0 or done == n_steps:
                self.checkpoint()
            now = time.perf_counter()
            if now - last_report >= 5.0 or done == n_steps:
                last_report = now
                for w in self.worlds:
                    self.progress(f'  fly-{w.fly_id:02d}  step {w.cursor:6d}  sim {w.cursor * w.step_ms / 1000:.3f} s  '
                                  f'spikes {w.total_spikes}  mean yaw {w.yaw_sum / max(w.cursor, 1):+.4f} rad/s')
        return done, time.perf_counter() - t0

    def summary(self, steps: int, wall_s: float) -> List[dict]:
        rows = []
        for w in self.worlds:
            sim_s = steps * w.step_ms / 1000.0
            rows.append({'fly': w.fly_id, 'seed': w.seed, 'steps': w.cursor, 'spikes': w.total_spikes,
                         'mean_yaw_rad_s': w.yaw_sum / max(w.cursor, 1),
                         'steps_per_s': steps / wall_s if wall_s > 0 else None,
                         'x_real_time': sim_s / wall_s if wall_s > 0 else None})
        return rows


def _print_table(rows: List[dict], wall_s: float, n_flies: int, progress):
    progress(f'{"fly":>4} {"seed":>6} {"steps":>7} {"spikes":>12} {"mean yaw rad/s":>15} '
             f'{"steps/s":>9} {"x real time":>12}')
    for r in rows:
        progress(f'{r["fly"]:>4} {r["seed"]:>6} {r["steps"]:>7} {r["spikes"]:>12} {r["mean_yaw_rad_s"]:>+15.5f} '
                 f'{(r["steps_per_s"] or 0):>9.2f} {(r["x_real_time"] or 0):>12.5f}')
    if rows and rows[0]['steps_per_s']:
        progress(f'cohort: {n_flies} flies, total {rows[0]["steps_per_s"] * n_flies:.2f} fly-steps/s, '
                 f'wall {wall_s:.2f} s (each fly sees the same wall time: the brains step as one batch)')


def _steps_for(seconds: float, step_ms: float) -> int:
    n = seconds * 1000.0 / step_ms
    if n < 1 or abs(n - round(n)) > 1e-9:
        raise CohortError(f'--seconds {seconds} is not a positive whole number of {step_ms} ms arena steps')
    return int(round(n))


def _every_steps(checkpoint_every_ms: Optional[float], step_ms: float, n_steps: int) -> int:
    if checkpoint_every_ms is None:
        return max(n_steps, 1)
    k = checkpoint_every_ms / step_ms
    if k < 1 or abs(k - round(k)) > 1e-9:
        raise CohortError(f'--checkpoint-every-ms {checkpoint_every_ms} must be a whole multiple of {step_ms} ms')
    return int(round(k))


def run_cohort(out, *, flies: int = 8, assay: str = 'optomotor', seconds: float = 2.0, seed_base: int = 1,
               engine: str = 'cpu', checkpoint_every_ms: Optional[float] = None, step_ms: float = DEFAULT_STEP_MS,
               graph: Optional[CohortGraph] = None, graph_dir=None, connectome_dir=None,
               progress: Callable[[str], None] = print, stop_after_steps: Optional[int] = None) -> List[dict]:
    root = Path(out).expanduser()
    store.refuse_legacy_path(root)
    if root.exists() and any(root.iterdir()):
        raise CohortError(f'{root} is not empty; use "neurofly cohort resume {root}" or a new --out directory')
    if flies < 1:
        raise CohortError('--flies must be >= 1')
    if assay not in ASSAYS:
        raise CohortError(f'Assay {assay!r} is not supported by the cohort runner (supported: {", ".join(ASSAYS)})')
    n_steps = _steps_for(seconds, step_ms)
    every = _every_steps(checkpoint_every_ms, step_ms, n_steps)
    with store.CohortLock(root):
        t_load = time.perf_counter()
        graph = graph or load_real_graph(graph_dir, connectome_dir)
        t_graph = time.perf_counter() - t_load
        t0 = time.perf_counter()
        eng = make_engine(engine, graph.arrays, flies)
        eng.reset(list(range(flies)))
        worlds = [FlyWorld(k, seed_base + k, graph, assay, step_ms) for k in range(flies)]
        t_setup = time.perf_counter() - t0
        sched = schedule_sha256(worlds[0])
        writer = writer_identity()
        manifest = {
            'schema': COHORT_SCHEMA, 'graph_sha256': graph.graph_sha256, 'io_map_sha256': graph.io_map_sha256,
            'dynamics': DYNAMICS_ID, 'engine': eng.describe(), 'engine_backend_id': eng.backend_id,
            'payload': _payload(writer), 'writer': writer,
            'scientific_disclosure': graph_io_disclosure(),
            'dynamics_signature': dynamics_signature(eng),
            'graph_identity': graph.identity, 'graph_source': graph.source, 'synthetic_graph': graph.synthetic,
            'assay': assay, 'step_ms': step_ms, 'tick_ms': TICK_MS, 'n_flies': flies, 'seed_base': seed_base,
            'stimulus_schedule': worlds[0].schedule(), 'stimulus_schedule_sha256': sched,
            'checkpoint_every_steps': every, 'target_steps': n_steps, 'tick': 0, 'sim_ms': 0.0,
            'flies': [{'fly_id': w.fly_id, 'seed': w.seed, 'dir': f'flies/fly-{w.fly_id:02d}', 'checkpoints': []}
                      for w in worlds],
            'segments': []}
        for w in worlds:
            store.atomic_write_json(root / 'flies' / f'fly-{w.fly_id:02d}' / 'manifest.json', {
                'schema': COHORT_SCHEMA, 'fly_id': w.fly_id, 'seed': w.seed, 'assay': assay,
                'graph_sha256': graph.graph_sha256, 'io_map_sha256': graph.io_map_sha256, 'dynamics': DYNAMICS_ID,
                'encoder_rng': 'numpy default_rng(seed)', 'arena_seed': w.seed,
                'recorded_populations': sorted(list(graph.io.populations) + list(graph.io.monitors)),
                'outputs': {'steps.jsonl': 'one JSON record per arena step: population spike counts, '
                                           'decoded motor command, world state'}})
        cohort = Cohort(root, graph, eng, worlds, manifest, progress)
        cohort.open_outputs()
        progress(f'cohort run: {flies} flies, assay {assay}, {seconds} s ({n_steps} steps of {step_ms} ms), '
                 f'engine {eng.backend_id}, graph {graph.graph_sha256[:12]}'
                 + (' [SYNTHETIC TEST GRAPH]' if graph.synthetic else ''))
        progress(f'NOTE: {DISCLOSURE}')
        try:
            cohort.checkpoint()
            steps, wall = cohort.run_steps(n_steps, every, stop_after_steps)
        finally:
            cohort.close_outputs()
        rows = cohort.summary(steps, wall)
        manifest['segments'].append({'kind': 'run', 'writer': cohort.writer, 'engine': eng.backend_id, 'steps': steps, 'wall_s': wall,
                                     'graph_load_s': t_graph, 'setup_s': t_setup, 'summary': rows})
        store.atomic_write_json(store.manifest_path(root), manifest)
        _print_table(rows, wall, flies, progress)
        return rows


def resume_cohort(out, *, seconds: Optional[float] = None, engine: Optional[str] = None,
                  graph: Optional[CohortGraph] = None, graph_dir=None, connectome_dir=None,
                  allow_synthetic: bool = False, progress: Callable[[str], None] = print,
                  stop_after_steps: Optional[int] = None) -> List[dict]:
    """Verify a cohort directory and continue it.

    Same-engine resume is byte-identical to an uninterrupted run.  CPU<->GPU
    continuation is NOT exact; it is verified only within the preregistered
    contract bounds (see RESUME_DESCRIPTION).
    """
    root = Path(out).expanduser()
    store.refuse_legacy_path(root)
    if not store.manifest_path(root).is_file():
        raise CohortError(f'{root} has no {store.MANIFEST_NAME}; not a cohort directory, refused')
    with store.CohortLock(root):
        manifest = store.read_manifest(root)
        if manifest.get('dynamics') != DYNAMICS_ID:
            raise CohortError(f'Cohort dynamics {manifest.get("dynamics")!r} is not {DYNAMICS_ID}; refused')
        # Verify every checkpoint chain before anything is built or written.
        last = {}
        for entry in manifest['flies']:
            last[entry['fly_id']] = store.verify_chain(root, entry)
        ticks = {meta['tick'] for meta, _, _ in last.values()}
        if len(ticks) != 1:
            raise CohortError(f'Flies stopped at different ticks {sorted(ticks)}; refused')
        graph = graph or graph_from_source(manifest['graph_source'], graph_dir=graph_dir,
                                           connectome_dir=connectome_dir, allow_synthetic=allow_synthetic)
        for key, have in (('graph_sha256', graph.graph_sha256), ('io_map_sha256', graph.io_map_sha256)):
            if manifest.get(key) != have:
                raise CohortError(f'{key} mismatch: the cohort was run on {manifest.get(key)}, the loaded graph '
                                  f'is {have}; refused')
        for fid, (meta, _, _) in last.items():
            for key, want in (('graph_sha256', graph.graph_sha256), ('io_map_sha256', graph.io_map_sha256),
                              ('dynamics', DYNAMICS_ID), ('assay', manifest['assay']),
                              ('stimulus_schedule_sha256', manifest['stimulus_schedule_sha256'])):
                if meta.get(key) != want:
                    raise CohortError(f'fly {fid}: checkpoint {key} {meta.get(key)!r} != {want!r}; refused')
        flies = int(manifest['n_flies'])
        step_ms = float(manifest['step_ms'])
        recorded_engine = manifest['engine_backend_id']
        eng = make_engine(engine or ('gpu' if 'gpu' in recorded_engine else 'cpu'), graph.arrays, flies)
        # The model the running code executes must be the one the cohort was run
        # under (code version may differ).  Checked before any restore or write.
        running = dynamics_signature(eng)
        recorded = manifest.get('dynamics_signature') or {}
        if recorded.get('sha256') != running['sha256'] or recorded.get('values') != running['values']:
            diff = _signature_diff(recorded.get('values') or {}, running['values'])
            raise CohortError(f'Dynamics signature mismatch: the cohort was run under {recorded.get("sha256")}, the '
                              f'running model is {running["sha256"]} ({diff}); refused, nothing restored or written')
        for fid, (meta, _, _) in last.items():
            if (meta.get('dynamics_signature') or {}).get('sha256') != running['sha256']:
                raise CohortError(f'fly {fid}: checkpoint dynamics signature differs from the running model; refused')
        # Cross-engine resume (cpu <-> gpu) is allowed: the preregistered restore
        # proof passed in both directions.  Source and target are recorded.
        # Phase 1: validate the WHOLE cohort in memory.  Nothing live or on disk
        # changes until every fly has passed every check.
        lasts = [last[e['fly_id']] for e in manifest['flies']]
        worlds, staged = [], []
        for k, (entry, (meta, state, digest)) in enumerate(zip(manifest['flies'], lasts)):
            w = FlyWorld(entry['fly_id'], entry['seed'], graph, manifest['assay'], step_ms)
            if w.phase_sha256 != meta['encoder_phase_sha256'] or schedule_sha256(w) != meta['stimulus_schedule_sha256']:
                raise CohortError(f'fly {k}: rebuilt encoder/schedule differs from the checkpoint; refused')
            if meta['rng']['bit_generator'] != type(w.rng.bit_generator).__name__:
                raise CohortError(f'fly {k}: RNG {meta["rng"]["bit_generator"]} differs; refused')
            if meta.get('engine_backend_id') != recorded_engine:
                raise CohortError(f'fly {k}: checkpoint engine {meta.get("engine_backend_id")} differs; refused')
            try:
                staged.append(eng._check_state(k, state, eng.read_state(k)))
                w.rng.bit_generator.state = meta['rng']['state']
                w.arena.restore_world(json.loads(meta['world_state']))
            except Exception as exc:
                raise CohortError(f'fly {k}: checkpoint state refused ({type(exc).__name__}: {exc})') from exc
            w.decoder.rate_l = float(meta['decoder']['rate_l'])
            w.decoder.rate_r = float(meta['decoder']['rate_r'])
            w.cursor = int(meta['input_cursor'])
            w.total_spikes = int(meta['totals']['spikes'])
            w.yaw_sum = float(meta['totals']['yaw_sum_rad_s'])
            path = root / 'flies' / f'fly-{w.fly_id:02d}' / 'steps.jsonl'
            want_n, want_sha = meta['outputs']['steps_jsonl_bytes'], meta['outputs']['steps_jsonl_sha256']
            data = path.read_bytes() if path.exists() else b''
            if len(data) < want_n or store.sha256_bytes(data[:want_n]) != want_sha:
                raise CohortError(f'{path}: recorded outputs do not match the checkpoint; refused')
            worlds.append(w)
        # Phase 2: install.  Steps written after the last checkpoint (an
        # interrupted run) are copied aside, never deleted, then cut off.
        for k, st in enumerate(staged):
            eng.write_state(k, st)
        for (meta, _, _), w in zip(lasts, worlds):
            path = root / 'flies' / f'fly-{w.fly_id:02d}' / 'steps.jsonl'
            want_n = meta['outputs']['steps_jsonl_bytes']
            if path.stat().st_size > want_n:
                orphan = path.with_name(f'steps.after-t{meta["tick"]:06d}.{int(time.time())}.orphan.jsonl')
                shutil.copyfile(path, orphan)
                with open(path, 'r+b') as fh:
                    fh.truncate(want_n)
                    os.fsync(fh.fileno())
                store.fsync_dir(path.parent)
                progress(f'  fly-{w.fly_id:02d}: steps after the last checkpoint moved to {orphan.name}')
        cursor = worlds[0].cursor
        n_steps = (_steps_for(seconds, step_ms) if seconds is not None
                   else int(manifest['target_steps']) - cursor)
        if seconds is not None:
            manifest['target_steps'] = cursor + n_steps
        if n_steps <= 0:
            progress(f'cohort {root}: already at step {cursor} (target {manifest["target_steps"]}); nothing to do. '
                     'Pass --seconds S to continue further.')
            return []
        manifest['engine_backend_id'] = eng.backend_id
        manifest['engine'] = eng.describe()
        cohort = Cohort(root, graph, eng, worlds, manifest, progress)
        cohort.parents = [last[e['fly_id']][2] for e in manifest['flies']]
        cohort.open_outputs()
        progress(f'cohort resume: {flies} flies from step {cursor} (sim {cursor * step_ms / 1000:.3f} s), '
                 f'+{n_steps} steps, engine {recorded_engine} -> {eng.backend_id}')
        progress(f'NOTE: {DISCLOSURE}')
        every = int(manifest['checkpoint_every_steps'])
        try:
            steps, wall = cohort.run_steps(n_steps, every, stop_after_steps)
        finally:
            cohort.close_outputs()
        rows = cohort.summary(steps, wall)
        # Compatibility was decided by schema, model and data hashes above, never by
        # code version; the continuation records both writers.
        manifest['segments'].append({'kind': 'resume', 'from_step': cursor, 'engine': eng.backend_id,
                                     'source_writer': lasts[0][0].get('writer'), 'target_writer': cohort.writer,
                                     'source_engine': recorded_engine, 'target_engine': eng.backend_id,
                                     'steps': steps, 'wall_s': wall, 'summary': rows})
        store.atomic_write_json(store.manifest_path(root), manifest)
        _print_table(rows, wall, flies, progress)
        return rows


# ---------------------------------------------------------------------------
# Numerical contract (Lane A owns the contract file and comparison logic)
# ---------------------------------------------------------------------------
def verify_contract(engine: str = 'gpu', progress: Callable[[str], None] = print) -> int:
    """Run the preregistered CPU/GPU contract and print PASS/FAIL per check.

    Expects ``brainlab.cohort.contract.run_contract(engine=...)`` to yield
    dicts with ``name``, ``passed`` (bool) and ``detail``.
    """
    try:
        from brainlab.cohort import contract
    except Exception as exc:
        progress(f'FAIL  contract-available: brainlab.cohort.contract cannot be imported ({type(exc).__name__}: {exc})')
        return 1
    try:
        results = list(contract.run_contract(engine=engine))
    except EngineUnavailable as exc:
        progress(f'FAIL  engine-available: {exc}')
        return 1
    if not results:
        progress('FAIL  contract: no checks were run')
        return 1
    ok = True
    for r in results:
        ok &= bool(r['passed'])
        progress(f'{"PASS" if r["passed"] else "FAIL"}  {r["name"]}: {r.get("detail", "")}')
    progress(f'contract: {"PASS" if ok else "FAIL"} ({sum(bool(r["passed"]) for r in results)}/{len(results)})')
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# CLI: neurofly cohort run|resume|verify
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog='neurofly cohort',
                                     epilog=DISCLOSURE,
                                     description='Many independent fixed-v3 MaleCNS brains in the Arena + io_map '
                                                 'optomotor loop, with per-fly recording and checkpoint/resume.')
    sub = parser.add_subparsers(dest='action', required=True)
    run = sub.add_parser('run', help='Run a new cohort into an empty directory')
    run.add_argument('--flies', type=int, default=8)
    run.add_argument('--assay', default='optomotor', choices=ASSAYS)
    run.add_argument('--seconds', type=float, default=2.0, help='simulated seconds per fly')
    run.add_argument('--seed-base', type=int, default=1, help='fly k gets seed SEED_BASE + k')
    run.add_argument('--out', required=True)
    run.add_argument('--engine', choices=('cpu', 'gpu'), default='cpu')
    run.add_argument('--checkpoint-every-ms', type=float, default=None,
                     help='simulated ms between checkpoints (default: start and end only)')
    res = sub.add_parser('resume', help=RESUME_HELP, description=RESUME_DESCRIPTION)
    res.add_argument('dir')
    res.add_argument('--seconds', type=float, default=None,
                     help='further simulated seconds (default: finish the original target)')
    res.add_argument('--engine', choices=('cpu', 'gpu'), default=None)
    ver = sub.add_parser('verify', help='Run the preregistered CPU/GPU numerical contract')
    ver.add_argument('--engine', choices=('cpu', 'gpu'), default='gpu')
    for p in (run, res):
        p.add_argument('--graph-dir', default=None)
        p.add_argument('--connectome-dir', default=None)
        p.add_argument('--test-synthetic-graph', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    from brainlab.graph_identity import GraphUnavailable
    try:
        if args.action == 'verify':
            return verify_contract(args.engine)
        graph = synthetic_graph() if args.test_synthetic_graph else None
        if args.action == 'run':
            run_cohort(args.out, flies=args.flies, assay=args.assay, seconds=args.seconds,
                       seed_base=args.seed_base, engine=args.engine,
                       checkpoint_every_ms=args.checkpoint_every_ms, graph=graph,
                       graph_dir=args.graph_dir, connectome_dir=args.connectome_dir)
        else:
            resume_cohort(args.dir, seconds=args.seconds, engine=args.engine,
                          graph_dir=args.graph_dir, connectome_dir=args.connectome_dir,
                          allow_synthetic=args.test_synthetic_graph)
        return 0
    except (CohortError, store.CohortStoreError, GraphUnavailable) as exc:
        print(f'neurofly cohort: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())

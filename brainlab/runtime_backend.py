"""Source-level compute selection and fixed-AMD capability admission.

No device imports or probing occur here. CPU/CUDA auto resolution remains in
Brain; a registry freezes its actual first resolved engine for later instances.
"""
import json
import math
import os
from pathlib import Path

COMPUTE_BACKENDS = ('auto', 'cpu', 'cuda', 'wgpu-amd')


class UnsupportedRuntimeCapability(ValueError):
    """Refusal before store, owner or learning-state mutation."""


def compute_selector(value=None):
    selected = value if value is not None else os.environ.get('NEUROFLY_BRAIN_BACKEND', 'auto')
    if type(selected) is not str or selected not in COMPUTE_BACKENDS:
        raise UnsupportedRuntimeCapability(f'Choose a brain backend from {COMPUTE_BACKENDS}')
    return selected


def require_fixed_amd(compute, controller, *, dynamics=None, learning=False, teaching=False,
                      continue_io_state=False, step_ms=None):
    if compute != 'wgpu-amd':
        return
    from .graph_identity import active_dynamics_version, active_e_inh_mV
    version = dynamics or active_dynamics_version()
    if version != 'v3' or active_e_inh_mV() not in (None, -70.):
        raise UnsupportedRuntimeCapability('wgpu-amd supports baseline fixed v3 only')
    if controller != 'connectome-fixed':
        raise UnsupportedRuntimeCapability('wgpu-amd requires connectome-fixed; plastic/readout/modular controllers are unsupported')
    if learning or teaching or continue_io_state:
        raise UnsupportedRuntimeCapability('wgpu-amd does not support learning, teaching or graph-I/O state migration')
    if step_ms is not None and (type(step_ms) not in (int, float) or not math.isfinite(step_ms)
            or not 0 < step_ms <= 102.4
            or not math.isclose(round(step_ms / .1) * .1, step_ms, rel_tol=0, abs_tol=1e-9)):
        raise UnsupportedRuntimeCapability('wgpu-amd graph step must be 1–1024 exact 0.1 ms ticks')


def require_fixed_graph_io(recorded):
    from provenance import graph_io_declaration
    expected = graph_io_declaration()
    if (not isinstance(recorded, dict) or recorded.get('version') != expected['version']
            or recorded.get('sha256') != expected['sha256']):
        raise UnsupportedRuntimeCapability(
            'wgpu-amd requires the exact current graph-I/O version and hash in both saved manifest '
            'and checkpoint. Saved state is retained; continue with its compatible CPU/CUDA engine. '
            'AMD graph-I/O migration is unsupported.')


def require_fixed_payload(meta, arrays):
    """Empty fixed-checkpoint plastic placeholders are not learned parameters."""
    import numpy as np
    from .brain import STATE_ARRAYS
    require_fixed_graph_io(meta.get('graph_io'))
    if (meta.get('backend') != 'connectome-fixed' or meta.get('rule') is not None
            or meta.get('readout_rate') is not None or meta.get('learning_state_resets')):
        raise UnsupportedRuntimeCapability('wgpu-amd cannot restore learned checkpoint state')
    for name in arrays:
        if name in {f'brain.{field}' for field in STATE_ARRAYS}:
            continue
        value = arrays[name]
        if (name not in ('plastic.edges', 'plastic.delta')
                or not isinstance(value, np.ndarray) or value.shape != (0,)):
            raise UnsupportedRuntimeCapability(f'wgpu-amd cannot restore learned checkpoint field {name}')


def require_fixed_manifest(manifest):
    from .graph_identity import active_dynamics
    require_fixed_graph_io(manifest.get('graph_io'))
    locations = manifest.get('learned_parameter_locations') or {}
    if (manifest.get('backend') != 'connectome-fixed' or not isinstance(locations, dict)
            or locations.keys() - {'brain_transients', 'rng', 'world_state'}):
        raise UnsupportedRuntimeCapability('wgpu-amd cannot restore a learned-state manifest')
    saved = manifest.get('dynamics') or {}
    if saved.get('plasticity_rule') is not None or any(
            saved.get(key) != value for key, value in active_dynamics().items()):
        raise UnsupportedRuntimeCapability('wgpu-amd requires exact baseline v3 saved dynamics; no conversion')


def preflight_fixed_store(root, assay=None):
    """Read-only learned-state admission before any target store/PID is touched.

    Inspect fixed instances only. Read metadata and learning fields lazily; do
    not allocate full saved transient arrays. Normal registry restore still
    validates graph identity, hashes and complete transient state afterwards.
    """
    import numpy as np
    from .brain import STATE_ARRAYS
    state_names = {f'brain.{field}' for field in STATE_ARRAYS}
    root = Path(root)
    index_path = root / 'registry.json'
    if not index_path.exists():
        return
    index = json.loads(index_path.read_text())
    entries = index.get('instances', {})
    selected = index.get('current_instances', {})
    for instance_id, entry in entries.items():
        if entry.get('backend') != 'connectome-fixed' or (assay is not None and entry.get('assay') != assay):
            continue
        key = f"{entry['assay']}|connectome-fixed"
        if key in selected and selected[key] != instance_id:
            continue
        directory = root / entry['assay'] / 'connectome-fixed' / instance_id
        manifest = json.loads((directory / 'manifest.json').read_text())
        require_fixed_manifest(manifest)
        for checkpoint in (directory / 'checkpoints').glob('ckpt-*.npz'):
            with np.load(checkpoint, allow_pickle=False) as archive:
                meta = json.loads(str(archive['meta']))
                learning = {name: archive[name] if name in ('plastic.edges', 'plastic.delta') else None
                            for name in archive.files if name != 'meta' and name not in state_names}
                require_fixed_payload(meta, learning)

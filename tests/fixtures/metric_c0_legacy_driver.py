"""Scripted legacy-view driver for metric contract S-12 (C0 baseline fixtures).

``drive(maze_module)`` runs a fixed, deterministic script against every registered
paradigm and returns, per paradigm, the normalised ``step()`` outputs at checkpoints
and ``get_metrics()`` after each checkpoint. It imports nothing from the repository:
the caller passes the ``maze`` module, so the same script runs against the baseline
(``713ba82``) and against the branch under test.

Regenerate the fixture from the BASELINE tree only (never from the branch):
    git show 713ba82:maze.py > <dir>/maze.py
    git show 713ba82:online_metrics.py > <dir>/online_metrics.py
    PYTHONPATH=<dir> python tests/fixtures/metric_c0_legacy_driver.py <out.json>
"""
import json
import math
import sys
from types import SimpleNamespace

PARADIGMS = ['t_maze', 'y_maze', 'heat_maze', 'buridan', 'visual_operant', 'wind_tunnel', 'looming_escape',
             'optomotor', 'gap_crossing', 'circadian_dam', 'courtship', 'labyrinth', 'multisensory_benchmark']

# Per-paradigm waypoints (x, y) the scripted fly walks through, chosen to cross each zone.
WAYPOINTS = {
    't_maze': [(70, 20), (70, 50), (20, 50), (70, 50), (120, 50), (70, 50), (20, 50)],
    'y_maze': [(60, 60), (60, 100), (60, 60), (25, 40), (60, 60), (95, 40), (60, 60), (60, 100)],
    'heat_maze': [(40, 40), (90, 90), (60, 60), (82, 78)],
    'buridan': [(60, 60), (100, 60), (60, 100), (30, 60), (60, 60)],
    'visual_operant': [(40, 40), (41, 40)],
    'wind_tunnel': [(20, 30), (60, 20), (120, 40), (180, 30)],
    'looming_escape': [(40, 40), (42, 40)],
    'optomotor': [(45, 45), (45, 46)],
    'gap_crossing': [(20, 10), (44, 10), (60, 10)],
    'circadian_dam': [(10, 5), (40, 5), (40.1, 5), (20, 5)],
    'courtship': [(4, 4), (11, 11), (15, 6)],
    'labyrinth': [(10, 10), (10, 80), (40, 80), (60, 75), (90, 60), (130, 85)],
    'multisensory_benchmark': [(0, 0), (40, 40), (-40, 40), (45, 45)],
}
STEPS_PER_LEG = 37


def _norm(obj):
    """JSON-normalise a legacy output exactly (floats keep their repr through json)."""
    if isinstance(obj, dict):
        return {str(k): _norm(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_norm(v) for v in obj]
    if hasattr(obj, 'tolist') and not isinstance(obj, (str, bytes)):
        return _norm(obj.tolist())
    if hasattr(obj, 'x') and hasattr(obj, 'y') and not isinstance(obj, (int, float, str)):
        return [_norm(obj.x), _norm(obj.y)]
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else repr(obj)
    try:
        return _norm(list(obj))
    except TypeError:
        return repr(obj)


def _fly(kind, x, y, heading, speed, angular_velocity, i):
    if kind == 'dict':
        return {'x': x, 'y': y, 'heading': heading, 'speed': speed, 'angular_velocity': angular_velocity}
    state = 'COURTSHIP' if i % 3 == 0 else ('SURGE' if i % 3 == 1 else 'CAST')
    return SimpleNamespace(pos=SimpleNamespace(x=x, y=y), heading=heading, speed=speed,
                           angular_velocity=angular_velocity, yaw_torque=0.3 * math.sin(i / 7.0),
                           behavioral_state=state, is_saccade=(i % 50 == 0))


def _run(maze, key):
    p = maze.ExperimentRegistry.get(key)
    pts = WAYPOINTS[key]
    out = {'checkpoints': [], 'metrics': []}
    i = 0
    for leg in range(len(pts) - 1):
        (x0, y0), (x1, y1) = pts[leg], pts[leg + 1]
        heading = math.atan2(y1 - y0, x1 - x0)
        for k in range(STEPS_PER_LEG):
            f = (k + 1) / STEPS_PER_LEG
            x, y = x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
            dt = 0.02 if i % 5 else 0.013
            speed = 0.0 if (key == 'circadian_dam' and leg == 1) else 1.0 + 0.5 * math.sin(i / 9.0)
            ang = 0.4 * math.sin(i / 11.0)
            kind = 'dict' if i % 2 else 'obj'
            if key == 'multisensory_benchmark' and i % 17 == 0:
                res = p.step(_fly(kind, x, y, heading, speed, ang, i), dt, override_thrust=0.2)
            else:
                res = p.step(_fly(kind, x, y, heading, speed, ang, i), dt)
            if key == 'looming_escape' and i == 40:
                p.gf_source = 'connectome'
                p.record_gf_spike()
            if k % 12 == 0:
                out['checkpoints'].append(_norm(res))
                out['metrics'].append(_norm(p.get_metrics()))
            i += 1
        if leg == 1:
            p.reset_trial()
            out['metrics'].append(_norm(p.get_metrics()))
    out['metrics'].append(_norm(p.get_metrics()))
    return out


def drive(maze):
    return {key: _run(maze, key) for key in PARADIGMS}


if __name__ == '__main__':
    import maze as _maze  # the baseline tree, via PYTHONPATH
    data = {'source': '713ba82:maze.py', 'driver': 'tests/fixtures/metric_c0_legacy_driver.py',
            'paradigms': drive(_maze)}
    with open(sys.argv[1], 'w') as fh:
        json.dump(data, fh, indent=1, sort_keys=True, allow_nan=False)
        fh.write('\n')

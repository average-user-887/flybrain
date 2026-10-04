import sys, tempfile, pathlib
sys.path.insert(0, '.'); sys.path.insert(0, 'tests')
from test_wp5_live_loop import graph_runner, synthetic_io_map
tmp = pathlib.Path(tempfile.mkdtemp())
r = graph_runner(tmp)
r.graph_controller._optomotor_io = synthetic_io_map()
with r.lock:
    r.step_once(); print('A optomotor', r.total_steps, r.last_error)
    print(r._apply_command({'action': 'switch_paradigm', 'paradigm': 't-maze'}).get('status'))
    r.step_once(); print('B t-maze', r.total_steps, r.last_error)
    print(r._apply_command({'action': 'switch_paradigm', 'paradigm': 'optomotor'}).get('status'))
    r.step_once(); print('C optomotor again', r.total_steps, r.last_error)
    print(r._apply_command({'action': 'switch_paradigm', 'paradigm': 't-maze'}))
    r.step_once(); print('D t-maze', r.total_steps, r.last_error, r.paused)

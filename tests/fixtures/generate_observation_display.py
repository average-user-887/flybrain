"""Regenerate browser compatibility fixtures from real Python producers/ledger.

No measurement values are inserted or repaired. These are synthetic harness
samples of actual producers, not biological validation or independent results.
"""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import maze
from learning_recorder import LearningRecorder
from observation_envelopes import build_observation_envelope
from observation_publication import ObservationPublicationQueue
from online_metrics import OpenArenaObserver, build_observation_config

ASSAYS = {
    'open_arena': 'open-arena', 't_maze': 't-maze', 'y_maze': 'y-maze',
    'heat_maze': 'heat-maze', 'buridan': 'buridan', 'visual_operant': 'visual-operant',
    'wind_tunnel': 'wind-tunnel', 'looming_escape': 'looming-escape', 'optomotor': 'optomotor',
    'gap_crossing': 'gap-crossing', 'circadian_dam': 'circadian-dam', 'courtship': 'courtship',
    'labyrinth': 'labyrinth', 'multisensory': 'multisensory-sandbox',
}


def generate():
    fixtures = []
    with TemporaryDirectory() as directory:
        recorder = LearningRecorder(directory)
        try:
            for producer_name, assay_id in ASSAYS.items():
                producer = OpenArenaObserver() if producer_name == 'open_arena' else maze.ExperimentRegistry.get(producer_name)
                identity = dict(run_id=f'run-{producer_name}', instance_id=f'instance-{producer_name}',
                                assay=assay_id, backend='modular', controller_version='modular/fixture',
                                daemon_run_id='daemon-fixture', brain_id=f'brain-{producer_name}',
                                activation=7, synthetic=True, test_mode=True)
                context = dict(identity=identity, provenance=dict(gf_source='geometric',
                    stimulus_entry_stage=None, motor_assists_enabled=False, controller_states=[]),
                    segment_id=f'segment-{producer_name}', segment_start_sim_s=123.25, validity='valid')
                live = build_observation_envelope(producer.snapshot_observation(), **context)
                fly = SimpleNamespace(pos=SimpleNamespace(x=30.0, y=30.0), heading=0.0,
                                      speed=1.0, angular_velocity=0.0,
                                      assay_escape_remaining=0.0, behavioral_state='SURGE')
                if producer_name == 'open_arena':
                    producer.observe(fly, False, 0.02)
                else:
                    producer.step(fly, 0.02)
                frozen = build_observation_envelope(producer.freeze_observation('manual_reset'),
                    terminal_pose_post_step=dict(x_mm=fly.pos.x, y_mm=fly.pos.y, heading_rad=fly.heading), **context)
                queue = ObservationPublicationQueue()
                queue.enqueue(frozen)
                claim = queue.claim_oldest()
                receipt = recorder.record_observation(claim['observation'])
                queue.acknowledge(claim['attempt_token'], receipt)
                fixtures.append(dict(identity=identity, brainId=identity['brain_id'], observation=live,
                    observationPublication=dict(last_terminal=queue.last_terminal(assay=assay_id,
                        backend='modular', instance_id=identity['instance_id'], brain_id=identity['brain_id']))))
            producer = maze.ExperimentRegistry.get('gap_crossing')
            producer.configure_observation(build_observation_config(
                producer.observation_spec(), config_id='deadline-fixture', override_window_s=0.025))
            identity = dict(fixtures[9]['identity'], run_id='run-deadline')
            context = dict(context, identity=identity, segment_id='segment-deadline')
            for _ in range(2):
                producer.step(fly, 0.02)
            terminal = build_observation_envelope(producer.freeze_observation(),
                terminal_pose_post_step=dict(x_mm=fly.pos.x, y_mm=fly.pos.y, heading_rad=fly.heading), **context)
            queue = ObservationPublicationQueue()
            queue.enqueue(terminal)
            claim = queue.claim_oldest()
            queue.acknowledge(claim['attempt_token'], recorder.record_observation(claim['observation']))
            deadline = dict(identity=identity, brainId=identity['brain_id'], observation=None,
                observationPublication=dict(last_terminal=queue.last_terminal(assay='gap-crossing',
                    backend='modular', instance_id=identity['instance_id'], brain_id=identity['brain_id'])))
        finally:
            recorder.close()
    return dict(generator='actual Python producers, envelope builder, publication queue and LearningRecorder', fixtures=fixtures, deadlineFixture=deadline)


if __name__ == '__main__':
    print(json.dumps(generate(), sort_keys=True, indent=2, allow_nan=False))

import json
import numpy as np
import pytest
from brainlab.brain import Brain
from brainlab.runs import Run


def test_logged_input_and_spikes_reconstruct_response(tmp_path):
    graph=tmp_path/'graph.npz'
    np.savez(graph,ptr=np.array([0,0],dtype=np.int64),post=np.array([],dtype=np.int32),
        weight=np.array([],dtype=np.float32),ids=np.array([720575940000000001],dtype=np.int64))
    with Run(graph, {'experiment':'test'}, 7, tmp_path/'runs') as run:
        counts,_=run.step(Brain(graph),[20],10,trial_id=0,phase='cue')
        run.trial(trial_id=0,condition='test',success=False,reward=0)
    assert json.loads((run.path/'run.json').read_text())['status']=='completed'
    events=[json.loads(line) for line in (run.path/'events.jsonl').read_text().splitlines()]
    artifact=np.load(run.path/events[0]['artifact'])
    assert artifact['input_ids'][0]==720575940000000001
    repeat,_=Brain(graph).step(artifact['input_current'],10)
    np.testing.assert_array_equal(counts,repeat)
    assert int(artifact['spike_counts'].sum())==events[0]['total_spikes']
    assert events[1]['success'] is False


def test_failed_run_keeps_prior_events_and_does_not_overwrite(tmp_path):
    graph=tmp_path/'graph.npz';graph.write_bytes(b'provenance-only')
    with pytest.raises(RuntimeError):
        with Run(graph,{},0,tmp_path/'runs') as run:
            run.event('trial_start',trial_id=1)
            raise RuntimeError('test interruption')
    meta=json.loads((run.path/'run.json').read_text())
    assert meta['status']=='failed' and 'test interruption' in meta['error']
    assert (run.path/'events.jsonl').exists()
    with Run(graph,{},0,tmp_path/'runs') as other:
        pass
    assert other.path != run.path

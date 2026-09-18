import numpy as np
import pytest
from brainlab.brain import Brain


def graph(tmp_path):
    path = tmp_path/'graph.npz'
    np.savez(path, ptr=np.array([0,1,1],dtype=np.int64),
             post=np.array([1],dtype=np.int32), weight=np.array([100],dtype=np.float32),
             ids=np.array([10,20],dtype=np.int64))
    return path


def test_late_stimulation_and_chunking(tmp_path):
    path = graph(tmp_path)
    whole, chunks = Brain(path), Brain(path)
    for brain in (whole, chunks):
        silent,_ = brain.step([0,0],10)
        assert not silent.any()
    expected,_ = whole.step([20,0],100)
    observed = sum((chunks.step([20,0],10)[0] for _ in range(10)))
    np.testing.assert_array_equal(observed, expected)
    np.testing.assert_allclose(chunks.v, whole.v)
    assert expected[0] > 0 and expected[1] > 0


@pytest.mark.parametrize('current,duration', [([float('nan'),0],10), ([0],10),
    ([0,0],0), ([0,0],float('inf')), ([0,0],.15)])
def test_invalid_input_does_not_advance(tmp_path,current,duration):
    brain = Brain(graph(tmp_path))
    with pytest.raises(ValueError):
        brain.step(current,duration)
    assert brain.cursor == 0

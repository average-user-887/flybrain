import numpy as np
from brainlab.learning import Readout, balanced, features_from_counts


def test_reward_changes_sampled_action_probability():
    x=np.array([1.,0.]); positive=Readout(2);negative=Readout(2)
    positive.update(x,1,1); negative.update(x,1,-1)
    assert positive.probability(x)>.5>negative.probability(x)


def test_readout_learns_without_target_argument():
    rng=np.random.default_rng(8); model=Readout(2)
    for target in balanced(rng,64):
        x=np.array([1.,0.]) if target==0 else np.array([0.,1.])
        action=int(rng.random()<model.probability(x))
        model.update(x,action,1 if action==target else -1)
    assert model.probability(np.array([1.,0.]))<.1
    assert model.probability(np.array([0.,1.]))>.9
    before=model.weights.copy()
    for _ in range(10):model.probability(np.array([1.,0.]))
    np.testing.assert_array_equal(before,model.weights)


def test_features_exclude_stimulated_neurons():
    baseline=np.array([0,2,2]);eligible=np.array([1,2])
    a=features_from_counts(np.array([999,3,2]),baseline,eligible)
    b=features_from_counts(np.array([0,3,2]),baseline,eligible)
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(a,[1,0])

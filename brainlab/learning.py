"""Reward-trained action readout of a fixed full MaleCNS graph.

An engineering demonstration, not connectome synaptic learning. The policy
receives only downstream spike features; task labels reach it only as reward.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pyarrow.feather as feather
from .brain import Brain
from .connectome import ROOT, file_digest
from .runs import Run, atomic_json


class Readout:
    def __init__(self, size, rate=4.):
        self.weights = np.zeros(size, dtype=np.float64)
        self.rate = rate

    def probability(self, features):
        score = float(np.dot(self.weights, features))
        return float(1/(1+np.exp(-np.clip(score,-30,30))))

    def update(self, features, action, reward):
        # Bernoulli policy-gradient update; action 1 = right. Negative reward
        # discourages the sampled action. No target label is passed here.
        p = self.probability(features)
        self.weights += self.rate*reward*(action-p)*features


def features_from_counts(counts, baseline, eligible):
    x = (counts.astype(np.float64)-baseline)[eligible]
    return x/max(float(np.linalg.norm(x)),1.)


def balanced(rng, count):
    if count < 2 or count % 2:
        raise ValueError('Trial counts must be positive even numbers')
    labels = np.tile([0,1],count//2)
    rng.shuffle(labels)
    return labels


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--train',type=int,default=64)
    p.add_argument('--evaluate',type=int,default=32)
    args=p.parse_args()
    rng=np.random.default_rng(args.seed)
    train=balanced(rng,args.train); evaluation=balanced(rng,args.evaluate)
    graph=ROOT/'outputs/brainlab/malecns_v1/graph.npz'
    nodes=feather.read_table(ROOT/'connectome_data/malecns_v1/normalized/neurons.feather').to_pandas()
    annotations=feather.read_table(ROOT/'connectome_data/malecns_v1/annotations.feather').to_pandas().set_index('bodyId').loc[nodes.source_id]
    retinal=nodes.cell_type.eq('R1-R6').to_numpy()
    lamina=nodes.cell_type.isin(['L1','L2','L3','L5']).to_numpy()
    eyes=[np.flatnonzero(retinal & annotations.rootSide.eq(side).to_numpy()) for side in ['L','R']]
    eligible=np.flatnonzero(~(retinal|lamina))
    head=Readout(len(eligible))
    config=dict(experiment='reward-readout-v1',graph_fixed=True,learning_location='external linear action readout',
        train_trials=args.train,evaluation_trials=args.evaluate,trial_reset=True,
        cue_encoding='Blue: left-eye R1-R6 stimulation; orange: right-eye R1-R6 stimulation. Symbolic labels, not color vision.',
        duration_ms=100.,tonic_current=12.,retinal_current_range=[18.,24.],
        retinal_dropout=.1,feature='L2-normalized downstream spike counts minus tonic-only baseline; directly driven cell classes excluded',
        reward='Correct sampled action +1, incorrect -1',learning_rate=head.rate,
        task='Immediate binary choice; no navigation or delay memory',
        expected_baseline_accuracy=.5,seed=args.seed)
    trials=[]
    with Run(graph,config,args.seed) as run:
        print(f'Learning run: {run.path}',flush=True)
        brain=Brain(graph)
        np.testing.assert_array_equal(brain.ids,nodes.source_id.to_numpy(dtype=np.int64))
        tonic=np.zeros(brain.n,dtype=np.float32);tonic[lamina]=12.
        baseline,_=run.step(brain,tonic,100.,trial_id=-1,phase='tonic_calibration')
        np.savez_compressed(run.path/'feature-map.npz',eligible_ids=brain.ids[eligible],baseline_counts=baseline)
        policy_rng=np.random.default_rng(args.seed+1000)
        final_weights=None
        eval_x=[]
        for number,target in enumerate(np.r_[train,evaluation]):
            split='training' if number<len(train) else 'evaluation'
            if number==len(train):
                final_weights=head.weights.copy()
                np.savez_compressed(run.path/'trained-readout.npz',weights=final_weights)
            brain=Brain(graph)
            drive=tonic.copy()
            indices=eyes[int(target)]
            selected=indices[rng.random(len(indices))>=.1]
            amplitude=float(rng.uniform(18,24))
            drive[selected]=amplitude
            counts,wall=run.step(brain,drive,100.,trial_id=number,phase='cue')
            x=features_from_counts(counts,baseline,eligible)
            probability=head.probability(x)
            action=int(policy_rng.random()<probability) if split=='training' else int(probability>=.5)
            reward=1 if action==target else -1
            delta=0.
            if split=='training':
                before=head.weights.copy()
                head.update(x,action,reward)
                delta=float(np.linalg.norm(head.weights-before))
            else:
                np.testing.assert_array_equal(head.weights,final_weights)
                eval_x.append(x.astype(np.float32))
            trial=dict(trial_id=number,condition='reward_readout',split=split,cue=['blue','orange'][target],
                target=['left','right'][target],action=['left','right'][action],reward=reward,success=bool(action==target),
                probability_right=probability,amplitude=amplitude,stimulated_neurons=len(selected),
                stimulus_spikes=int(counts.sum()),wall_seconds=wall,weight_update_norm=delta)
            run.trial(**trial)
            trials.append(trial)
            if (number+1)%16==0:
                print(f'{number+1}/{len(train)+len(evaluation)} trials',flush=True)
        predicted=np.array([int(t['action']=='right') for t in trials[len(train):]])
        learned=float(np.mean(predicted==evaluation))
        permutation_rng=np.random.default_rng(args.seed+2000)
        shuffled=np.array([np.mean(predicted==permutation_rng.permutation(evaluation)) for _ in range(1000)])
        np.savez_compressed(run.path/'evaluation-features.npz',features=np.asarray(eval_x),labels=evaluation,predictions=predicted)
        summary=dict(schema_version=1,run_id=run.path.name,seed=args.seed,training_trials=len(train),evaluation_trials=len(evaluation),
            learned_accuracy=learned,frozen_untrained_accuracy=float(np.mean(evaluation==1)),
            zero_feature_accuracy=float(np.mean(evaluation==int(head.probability(np.zeros(len(eligible)))>=.5))),
            shuffled_label_mean=float(shuffled.mean()),shuffled_label_95_interval=np.quantile(shuffled,[.025,.975]).tolist(),
            permutation_p_value=float((1+np.count_nonzero(shuffled>=learned))/(len(shuffled)+1)),
            training_first16_accuracy=float(np.mean([t['success'] for t in trials[:16]])),
            training_last16_accuracy=float(np.mean([t['success'] for t in trials[len(train)-16:len(train)]])),
            graph_sha256=run.metadata['graph_sha256'],neurons=brain.n,edges=len(brain.post),
            trainable_parameters=len(head.weights),graph_weights_changed=False,evaluation_weights_frozen=True,
            criterion='Held-out accuracy >= 0.8 and >= 0.2 above untrained control',
            passed=bool(learned>=.8 and learned-float(np.mean(evaluation==1))>=.2),
            caveat='External readout learns a symbolic immediate choice from fixed-brain activity. No biological learning, color vision, movement, or delay memory demonstrated.')
        atomic_json(run.path/'learning-summary.json',summary)
        payload=dict(summary=summary,config=config,trials=trials)
        atomic_json(run.path/'demo-data.json',payload)
    # Publish only a completed run to the local demo.
    atomic_json(ROOT/'learning-data.json',payload)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    main()

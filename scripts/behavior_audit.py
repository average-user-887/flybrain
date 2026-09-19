#!/usr/bin/env python3
"""Record all-assay trajectories; causal response checks live in test_assay_responses.

No target success scores are enforced. Failure to reach a goal is retained as a
measurement, while nonfinite motion or an unmarked jump fails this audit.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import subprocess
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from arena import Arena
from assay_controls import describe
from experiment_brains import PARADIGMS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps',type=int,default=3000)
    p.add_argument('--seeds',type=int,nargs='+',default=[4,5,6])
    p.add_argument('--output',type=Path,default=PROJECT/'outputs'/'behavior-review'/'after.json')
    args=p.parse_args();rows=[]
    for pid in PARADIGMS:
        for seed in args.seeds:
            a=Arena(paradigm=None if pid=='open-arena' else pid,seed=seed,num_flies=1,num_predators=0)
            initial=prev=a.fly.pos.to_tuple();distance=max_step=0.0;states=Counter();trajectory=[]
            for i in range(args.steps):
                r=a.step(.02);f=a.fly;point=f.pos.to_tuple();d=math.dist(prev,point)
                if not all(math.isfinite(v) for v in (*point,f.heading,f.speed)) or d>.32:
                    raise ValueError(f'Invalid motion in {pid}, seed {seed}, step {i}: {d}')
                distance+=d;max_step=max(max_step,d);prev=point;states[r['state']]+=1
                if i%50==0 or i==args.steps-1:
                    trajectory.append(dict(step=i+1,sim_time_s=(i+1)*.02,x=f.pos.x,y=f.pos.y,heading=f.heading,speed=f.speed,state=f.behavioral_state,stimuli=r.get('stimuli',{}),drives=getattr(f,'sensorimotor_drives',{})))
            row=dict(paradigm=pid,seed=seed,sim_seconds=args.steps*.02,start=initial,end=prev,distance_mm=distance,max_step_mm=max_step,states=dict(states),metrics=a.paradigm.get_metrics() if a.paradigm else {},limitation=describe(a)['limitation'],trajectory=trajectory)
            rows.append(row)
            print(json.dumps({k:row[k] for k in ('paradigm','seed','distance_mm','states','metrics')}),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),steps=args.steps,seeds=args.seeds,rows=rows),indent=2,allow_nan=False))


if __name__=='__main__':main()

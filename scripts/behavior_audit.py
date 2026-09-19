#!/usr/bin/env python3
"""Record all-assay trajectories; causal response checks live in test_assay_responses.

No target success scores are enforced. Failure to reach a goal is retained as a
measurement, while nonfinite motion or an unmarked jump fails this audit.

Every row reports three things separately (remediation plan, work package 3):
``arena_containment`` (physics: escapes, unexplained jumps, numerical trapping),
``controller_outcome`` (attempted vs realized motion, wall pushing, rest, time near
walls, rolling progress, repeated stalls, paradigm metrics) and
``biological_validity`` (not assessed by this audit). ``motor_provenance`` records
which engineered motor assists were enabled and how often each acted.
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
sys.path.insert(0,str(PROJECT/'scripts'))
from arena import Arena
from assay_controls import describe
from experiment_brains import PARADIGMS
import containment_audit as audit

BIOLOGY=dict(status='not_assessed',reason='Hand-built modular controller with heuristic gains; trajectories are not compared with measured Drosophila behaviour.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps',type=int,default=3000)
    p.add_argument('--seeds',type=int,nargs='+',default=[4,5,6])
    p.add_argument('--assists',choices=('on','off','both'),default='on',help='engineered motor assists (wall_avoidance_reflex, contact_turn)')
    p.add_argument('--output',type=Path,default=PROJECT/'outputs'/'behavior-review'/'after.json')
    args=p.parse_args();rows=[]
    modes={'on':[True],'off':[False],'both':[True,False]}[args.assists]
    for pid in PARADIGMS:
        for assists in modes:
            for seed in args.seeds:
                a=Arena(paradigm=None if pid=='open-arena' else pid,seed=seed,num_flies=1,num_predators=0,motor_assists=None if assists else dict(audit.ASSISTS_OFF))
                walls=list(getattr(a.paradigm,'walls',[]) or []) if a.paradigm else []
                probe=audit.MotionProbe(.02,region=audit.legal_region(pid),walls=walls,radius=a.fly.radius)
                initial=prev=a.fly.pos.to_tuple();distance=max_step=0.0;states=Counter();trajectory=[]
                for i in range(args.steps):
                    r=a.step(.02);f=a.fly;point=f.pos.to_tuple();d=math.dist(prev,point)
                    if not all(math.isfinite(v) for v in (*point,f.heading,f.speed)) or d>.32:
                        raise ValueError(f'Invalid motion in {pid}, seed {seed}, step {i}: {d}')
                    kind=probe.observe(dict(f.motor_record),*point)
                    distance+=d;max_step=max(max_step,d);prev=point;states[r['state']]+=1
                    if i%50==0 or i==args.steps-1:
                        m=f.motor_record
                        trajectory.append(dict(step=i+1,sim_time_s=(i+1)*.02,x=f.pos.x,y=f.pos.y,heading=f.heading,speed=f.speed,state=f.behavioral_state,stimuli=r.get('stimuli',{}),drives=getattr(f,'sensorimotor_drives',{}),
                                               motion_kind=kind,controller_yaw=m.get('controller_yaw'),controller_speed=m.get('controller_speed'),wall_reflex_yaw=m.get('wall_reflex_yaw'),
                                               attempted_mm=m.get('attempted_mm'),realized_mm=m.get('realized_mm'),wall_gap_mm=m.get('wall_gap_mm'),contact_normals=m.get('contact_normals')))
                controller=probe.controller_report()
                controller.update(states=dict(states),metrics=a.paradigm.get_metrics() if a.paradigm else {})
                row=dict(paradigm=pid,seed=seed,assists='on' if assists else 'off',sim_seconds=args.steps*.02,start=initial,end=prev,distance_mm=distance,max_step_mm=max_step,states=dict(states),metrics=controller['metrics'],limitation=describe(a)['limitation'],
                         arena_containment=probe.containment_report(),controller_outcome=controller,biological_validity=BIOLOGY,motor_provenance=a.motor_provenance(),trajectory=trajectory)
                rows.append(row)
                print(json.dumps(dict(paradigm=pid,seed=seed,assists=row['assists'],physics_ok=probe.physics_ok,distance_mm=round(distance,2),
                                      wall_pushing_s=controller['time_s'].get('wall_pushing',0.0),stalls=controller['stall_episodes'],metrics=row['metrics']),default=str),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),steps=args.steps,seeds=args.seeds,assists=args.assists,rows=rows),indent=2,allow_nan=False,default=str))
    return 0 if all(r['arena_containment']['ok'] for r in rows) else 1


if __name__=='__main__':sys.exit(main())

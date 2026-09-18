"""Rebuild a CSV comparison from run files, including failed/incomplete runs."""
import argparse
import csv
import json
from pathlib import Path
from .connectome import ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-dir',type=Path,default=ROOT/'runs')
    args=parser.parse_args()
    rows=[]
    for path in sorted(args.runs_dir.glob('*/run.json')):
        meta=json.loads(path.read_text())
        trials=[]
        events=path.parent/'events.jsonl'
        if events.exists():
            lines=events.read_text().splitlines()
            for i,line in enumerate(lines):
                try:
                    event=json.loads(line)
                except json.JSONDecodeError:
                    if i == len(lines)-1:
                        continue  # A killed process may leave one partial final event.
                    raise
                if event['event']=='trial_end':
                    trials.append(event)
        for trial in trials or [{}]:
            rows.append(dict(run_id=meta['run_id'],status=meta['status'],seed=meta['seed'],
                graph_sha256=meta.get('graph_sha256'),experiment=meta['config'].get('experiment'),
                trial_id=trial.get('trial_id'),condition=trial.get('condition'),
                stimulus_spikes=trial.get('stimulus_spikes'),success=trial.get('success'),
                reward=trial.get('reward'),split=trial.get('split')))
    if not rows:
        print('No recorded runs.'); return
    output=args.runs_dir/'comparison.csv'
    with output.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    print(f'{len(rows)} rows written to {output}')
    for row in rows:
        print(f"{row['run_id']} {row['status']} {row['condition']}: {row['stimulus_spikes']}")

if __name__ == '__main__':
    main()

#!/bin/bash
WT=<repo>
S=<scratch>/studio
cd "$WT" || exit 2
export PYTHONPATH=$WT NEUROFLY_BRAIN_BACKEND=cpu CUDA_VISIBLE_DEVICES= MUJOCO_GL=egl
export NEUROFLY_GRAPH_DIR=$WT/outputs/brainlab/malecns_v1
for r in 20261005-142913-auditd-intact-s22 20261005-142913-auditd-intact-s22-control; do
  <home>/Documents/ChatGPT/flybrain/.venv/bin/python -m neurofly_body replay-check "$S/queue/runs/$r" --output "$S/rc-$r" 2>&1 | tail -3
  cp "$S/rc-$r/replay_check.json" "$S/queue/runs/$r/" && grep -o '"verdict"[^,]*' "$S/queue/runs/$r/replay_check.json"
done
<home>/Documents/ChatGPT/flybrain/.venv/bin/python -m neurofly_studio curate \
  "$S/queue/runs/20261005-142913-auditd-intact-s22" "$S/queue/runs/20261005-142913-auditd-intact-s22-control" \
  --name auditd-intro --title "Audit D optomotor pair" --explanation "Audit test pair." \
  --curated "$S/curated" 2>&1 | tail -20
ls -R "$S/curated" | head -20

#!/bin/bash
# Review studio for PR #30 on 127.0.0.1:8796: real MaleCNS graph, CPU brain backend, real FlyGym body.
WT=<repo>
S=<scratch>/pr30-studio
mkdir -p "$S"
cd "$WT" || exit 2
export PYTHONPATH=$WT PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=2
export NEUROFLY_GRAPH_DIR=$WT/outputs/brainlab/malecns_v1
export NEUROFLY_BRAIN_BACKEND=cpu CUDA_VISIBLE_DEVICES= MUJOCO_GL=egl
exec <home>/Documents/ChatGPT/flybrain/.venv/bin/python -u -m neurofly_studio serve --port 8796 \
  --queue "$S/queue" --curated "$S/curated" \
  --graph-dir "$WT/outputs/brainlab/malecns_v1" --connectome-dir "$WT/connectome_data/malecns_v1" \
  > "$S/studio.log" 2>&1

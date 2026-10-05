#!/bin/bash
WT=<repo>
ST=<scratch>/studio
mkdir -p "$ST"
cd "$WT" || exit 2
export PYTHONPATH=$WT PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=2
export NEUROFLY_GRAPH_DIR=$WT/outputs/brainlab/malecns_v1
export NEUROFLY_BRAIN_BACKEND=cpu CUDA_VISIBLE_DEVICES= MUJOCO_GL=egl
exec <home>/Documents/ChatGPT/flybrain/.venv/bin/python -u -m neurofly_studio serve --port 8842 \
  --queue "$ST/queue" --curated "$ST/curated" \
  --graph-dir "$WT/outputs/brainlab/malecns_v1" --connectome-dir "$WT/connectome_data/malecns_v1" \
  > "$ST/studio.log" 2>&1

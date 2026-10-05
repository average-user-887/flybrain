#!/bin/bash
# usage: start_daemon.sh <tag> <backend> <paradigm>
WT=<repo>
OUT=<scratch>/daemon-$1
mkdir -p "$OUT"
cd "$WT" || exit 2
export PYTHONPATH=$WT PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=2
export NEUROFLY_GRAPH_DIR=$WT/outputs/brainlab/malecns_v1
export NEUROFLY_BRAIN_BACKEND=cpu CUDA_VISIBLE_DEVICES=
exec <home>/Documents/ChatGPT/flybrain/.venv/bin/python -u "$WT/neurofly_daemon.py" --host 127.0.0.1 --port 8841 \
  --backend $2 --paradigm $3 --speed 1 --trial-seconds 60 \
  --checkpoint-interval 60 --output-dir "$OUT/out" --data-dir "$OUT/learning" --pid-file "$OUT/daemon.pid" \
  > "$OUT/daemon.log" 2>&1

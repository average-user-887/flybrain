#!/bin/bash
# Review daemon for PR #30 on 127.0.0.1:8793 (CPU brain backend), real MaleCNS graph.
WT=<repo>
OUT=<scratch>/pr30-daemon-$1
mkdir -p "$OUT"
cd "$WT" || exit 2
export PYTHONPATH=$WT PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1
export NEUROFLY_GRAPH_DIR=$WT/outputs/brainlab/malecns_v1
export NEUROFLY_BRAIN_BACKEND=cpu CUDA_VISIBLE_DEVICES=
exec <home>/Documents/ChatGPT/flybrain/.venv/bin/python -u "$WT/neurofly_daemon.py" --host 127.0.0.1 --port 8793 \
  --backend connectome-fixed --paradigm t-maze --speed 3 --continuous --trial-seconds 120 \
  --checkpoint-interval 30 --output-dir "$OUT/out" --data-dir "$OUT/learning" --pid-file "$OUT/daemon.pid" \
  > "$OUT/daemon.log" 2>&1

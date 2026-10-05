#!/bin/bash
# start_daemon.sh <master|fixed> <run-name> : starts a connectome-fixed test daemon on 8791 (foreground)
set -e
WT=<redacted-path>/Documents/ChatGPT/flybrain/<redacted-path>/.wt-race
RACE=<redacted-path>/tmp/race
if [ "$1" = master ]; then SCRIPT=$RACE/master_daemon/neurofly_daemon.py; else SCRIPT=$WT/neurofly_daemon.py; fi
OUT=$RACE/runs/$2
mkdir -p "$OUT"
cd "$WT"
export PYTHONPATH=$WT PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1
export NEUROFLY_GRAPH_DIR=$WT/outputs/brainlab/malecns_v1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1   # GTX 1660 Ti only
exec <redacted-path>/Documents/ChatGPT/flybrain/.venv/bin/python -u <redacted-path>/tmp/race/inject_daemon.py --host 127.0.0.1 --port 8791 \
  --backend connectome-fixed --paradigm t-maze --speed 3 --continuous --trial-seconds 120 \
  --checkpoint-interval 30 --output-dir "$OUT/out" --data-dir "$OUT/learning" --pid-file "$OUT/daemon.pid" \
  > "$OUT/daemon.log" 2>&1

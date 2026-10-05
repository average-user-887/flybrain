#!/bin/bash
# suite.sh <worktree> <logfile> [pytest args...]
P=$1; LOG=$2; shift 2
cd "$P" || exit 2
export PYTHONPATH=$P NEUROFLY_GRAPH_DIR=$P/outputs/brainlab/malecns_v1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=2
if [ $# -eq 0 ]; then set -- tests/; fi
<home>/Documents/ChatGPT/flybrain/.venv/bin/python -m pytest -q -p no:cacheprovider -rs "$@" > "$LOG" 2>&1
echo "exit=$?" >> "$LOG"
tail -40 "$LOG"

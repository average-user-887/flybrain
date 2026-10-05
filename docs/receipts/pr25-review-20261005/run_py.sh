#!/bin/bash
# run_py.sh <script> : run a python script inside the PR #25 review worktree with the real graph
WT=<repo>
cd "$WT" || exit 2
export PYTHONPATH=$WT NEUROFLY_GRAPH_DIR=$WT/outputs/brainlab/malecns_v1 CUDA_VISIBLE_DEVICES=
exec <home>/Documents/ChatGPT/flybrain/.venv/bin/python "$@"

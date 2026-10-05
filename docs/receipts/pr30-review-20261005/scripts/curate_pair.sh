#!/bin/bash
# Replay-check the real intact / output-disconnected pair (seed 12) and curate it the real way.
WT=<repo>
S=<scratch>/pr30-studio
PY=<home>/Documents/ChatGPT/flybrain/.venv/bin/python
cd "$WT" || exit 2
export PYTHONPATH=$WT NEUROFLY_GRAPH_DIR=$WT/outputs/brainlab/malecns_v1 NEUROFLY_BRAIN_BACKEND=cpu CUDA_VISIBLE_DEVICES= MUJOCO_GL=egl
A=$S/queue/runs/20261005-133231-review-intact-vs-disconnected-s12
B=$A-control
for R in "$A" "$B"; do
  "$PY" -m neurofly_body replay-check "$R" --output "$S/replay-$(basename "$R")" > "$S/replay-$(basename "$R").log" 2>&1
  echo "$(basename "$R"): exit $? $(grep -o '"verdict": "[A-Z_]*"' "$S/replay-$(basename "$R")/replay_check.json")"
  cp "$S/replay-$(basename "$R")/replay_check.json" "$R/replay_check.json"
done
"$PY" -m neurofly_studio curate "$A" "$B" --curated "$S/curated" --name review-optomotor-intro \
  --title "Review: the fly with its brain connected vs disconnected" \
  --explanation "0.5 s review run, seed 12: the connectome drives the legs." \
  --control-explanation "Same seed; the brain runs but never reaches the legs."
echo "curate exit $?"

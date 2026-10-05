#!/bin/bash
# API switching loops against the review daemon: 2 settled passes + 3 rapid passes over all 14 assays
S=<scratch>
L=<repo>/docs/receipts/switch-race-20261004/scripts/switch_loop.py
PY=<home>/Documents/ChatGPT/flybrain/.venv/bin/python
for i in $(seq 1 120); do curl -sf http://127.0.0.1:8793/api/status >/dev/null && break; sleep 1; done
"$PY" "$L" http://127.0.0.1:8793 2 settled "$S/pr30-switch-settled.json"
"$PY" "$L" http://127.0.0.1:8793 3 rapid "$S/pr30-switch-rapid.json"
"$PY" - <<EOF
import json
for m in ('settled', 'rapid'):
    d = json.load(open('$S/pr30-switch-' + m + '.json'))
    print(m, {k: d[k] for k in ('passes', 'switches', 'freezes', 'final_status', 'final_error', 'final_steps', 'elapsed_s')})
EOF

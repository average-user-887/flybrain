#!/bin/bash
# signoff.sh <run-name> : live_ui_signoff.py against the review daemon (8793) and web (8795)
WT=<repo>
cd "$WT" || exit 2
for i in $(seq 1 300); do curl -sf http://127.0.0.1:8793/api/status >/dev/null && break; sleep 1; done
timeout 1200 <home>/Documents/ChatGPT/flybrain/.venv/bin/python scripts/live_ui_signoff.py \
  --web "http://127.0.0.1:8795/?daemon=http://127.0.0.1:8793" --daemon http://127.0.0.1:8793 \
  --out <scratch>/browser/$1 --restore-assay t-maze --restore-speed 3 > <scratch>/browser/$1.log 2>&1
echo "exit=$?"
tail -15 <scratch>/browser/$1.log

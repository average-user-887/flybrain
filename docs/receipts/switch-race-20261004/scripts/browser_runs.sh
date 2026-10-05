#!/bin/bash
# browser_runs.sh <prefix> <count> : harness runs against the daemon on 8791, same process
R=<redacted-path>/tmp/race
cd <redacted-path>/Documents/ChatGPT/flybrain/<redacted-path>/.wt-race
for i in $(seq 1 120); do curl -sf http://127.0.0.1:8791/api/status >/dev/null && break; sleep 1; done
for n in $(seq 1 "$2"); do
  timeout 900 <redacted-path>/Documents/ChatGPT/flybrain/.venv/bin/python scripts/live_ui_signoff.py \
    --web "http://127.0.0.1:8790/?daemon=http://127.0.0.1:8791" --daemon http://127.0.0.1:8791 \
    --out $R/browser/$1-run$n --restore-assay t-maze --restore-speed 3 > $R/browser/$1-run$n.log 2>&1
  <redacted-path>/Documents/ChatGPT/flybrain/.venv/bin/python $R/summ.py $R/browser/$1-run$n
done

#!/bin/bash
# trial.sh <master|fixed> <run-name> <passes> <settled|rapid> : fresh daemon, one loop, stop
R=<redacted-path>/tmp/race
$R/start_daemon.sh "$1" "$2" &
for i in $(seq 1 180); do curl -sf http://127.0.0.1:8791/api/status >/dev/null && break; sleep 1; done
<redacted-path>/Documents/ChatGPT/flybrain/.venv/bin/python $R/switch_loop.py http://127.0.0.1:8791 "$3" "$4" $R/runs/$2/loop.json
echo "  not-active lines in daemon log: $(grep -c 'not active' $R/runs/$2/daemon.log)"
kill "$(cat $R/runs/$2/daemon.pid)"; wait
for i in $(seq 1 30); do ss -ltn | grep -q ':8791 ' || break; sleep 1; done

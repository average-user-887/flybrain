#!/usr/bin/env bash
# Project NeuroFly — Stop Continuous Learning Daemon Gracefully

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PID_FILE="$DIR/outputs/neurofly_daemon.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "[NeuroFly] No running daemon found (PID file missing)."
    exit 0
fi

PID=$(cat "$PID_FILE")
if kill -0 "$PID" 2>/dev/null; then
    echo "[NeuroFly] Sending SIGTERM to Daemon (PID: $PID)..."
    kill "$PID"
    for i in {1..15}; do
        if ! kill -0 "$PID" 2>/dev/null; then
            echo "[NeuroFly] Daemon shut down cleanly."
            rm -f "$PID_FILE"
            exit 0
        fi
        sleep 0.5
    done
    echo "[NeuroFly] Force killing PID $PID..."
    kill -9 "$PID" 2>/dev/null
    rm -f "$PID_FILE"
else
    echo "[NeuroFly] Stale PID file removed (process not active)."
    rm -f "$PID_FILE"
fi

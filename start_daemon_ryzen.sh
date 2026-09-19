#!/usr/bin/env bash
# Project NeuroFly — Start Continuous Learning Daemon (Headless 24/7)
# Runs portably in any Linux / remote workstation environment.

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR"

PORT="${NEUROFLY_PORT:-8769}"
SPEED="${NEUROFLY_SPEED:-15.0}"
PARADIGM="${NEUROFLY_PARADIGM:-multisensory-sandbox}"

PID_FILE="$DIR/outputs/neurofly_daemon.pid"
LOG_FILE="$DIR/outputs/neurofly_daemon.log"
mkdir -p "$DIR/outputs"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "[NeuroFly] Daemon is already running (PID: $PID, Port: $PORT)."
        exit 0
    else
        rm -f "$PID_FILE"
    fi
fi

# Detect Python interpreter (prefer local .venv if present)
if [ -f "$DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$DIR/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    PYTHON_BIN="python"
fi

echo "[NeuroFly] Launching Continuous Learning Daemon in background..."
echo "[NeuroFly] Interpreter: $PYTHON_BIN | Port: $PORT | Speed: ${SPEED}x | Assay: $PARADIGM"

nohup "$PYTHON_BIN" neurofly_daemon.py \
    --port "$PORT" \
    --speed "$SPEED" \
    --paradigm "$PARADIGM" \
    --pid-file "$PID_FILE" > "$LOG_FILE" 2>&1 &

DAEMON_PID=$!
echo "$DAEMON_PID" > "$PID_FILE"
sleep 1

if kill -0 "$DAEMON_PID" 2>/dev/null; then
    echo "[NeuroFly] Daemon successfully active (PID: $DAEMON_PID)."
    echo "[NeuroFly] Streaming API: http://localhost:$PORT/api/stream"
    echo "[NeuroFly] Log file: $LOG_FILE"
else
    echo "[NeuroFly] ERROR: Daemon failed to start. Check log:"
    cat "$LOG_FILE"
    exit 1
fi

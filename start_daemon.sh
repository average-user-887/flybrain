#!/usr/bin/env bash
# Project NeuroFly — Start Continuous Learning Daemon (Headless 24/7)
# Runs portably on any Linux host with a local .venv (or python3 on PATH).
#
# Environment:
#   NEUROFLY_PORT / NEUROFLY_SPEED / NEUROFLY_PARADIGM   launch parameters
#   NEUROFLY_HOST                                        bind address (default 127.0.0.1, this machine only).
#                                                        NEUROFLY_HOST=0.0.0.0 exposes the API to your LAN;
#                                                        commands are then open to anyone on it unless
#                                                        NEUROFLY_PUBLIC/NEUROFLY_ADMIN_TOKEN are used.
#                                                        (neurofly_daemon.py itself defaults to 127.0.0.1.)
#   NEUROFLY_PUBLIC=1                                    read-only public mode (see docs/PUBLIC_STREAMING.md)
#   NEUROFLY_ADMIN_TOKEN                                 bearer token that re-enables commands in public mode
#   NEUROFLY_DATA_DIR                                    where trials.jsonl / telemetry_summary.jsonl go
#   NEUROFLY_BACKEND                                     controller backend. Unset (default): the
#                                                        daemon's own fresh-launch default, which is
#                                                        connectome-fixed when the prepared MaleCNS graph
#                                                        verifies, otherwise the hand-built modular
#                                                        controller with a printed and dashboard-visible
#                                                        "not the connectome" notice. Set explicitly to
#                                                        connectome-fixed, connectome-plastic,
#                                                        connectome-with-trained-readout or modular.
#                                                        Saved brains keep their own backend: a modular,
#                                                        plastic or readout store is never converted.
#   NEUROFLY_GRAPH_DIR                                   prepared graph for graph backends, e.g.
#                                                        NEUROFLY_GRAPH_DIR=/path/to/malecns_v1
#                                                        (an explicit graph backend without it stops the
#                                                        daemon; no fallback)
# Process layout: two processes by default (a headless simulation process that owns the brain,
# recorder and checkpoints, plus a web process). Append --single-process for the one-process
# daemon. Stop with ./stop_daemon.sh, which lets the simulation save first.
# Extra daemon flags can be appended: ./start_daemon.sh --public --stream-hz 5

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR"

PORT="${NEUROFLY_PORT:-8769}"
HOST="${NEUROFLY_HOST:-127.0.0.1}"
SPEED="${NEUROFLY_SPEED:-15.0}"
PARADIGM="${NEUROFLY_PARADIGM:-multisensory-sandbox}"
BACKEND="${NEUROFLY_BACKEND:-}"
BACKEND_ARGS=()
if [ -n "$BACKEND" ]; then
    BACKEND_ARGS=(--backend "$BACKEND")
fi
GRAPH_ARGS=()
if [ -n "${NEUROFLY_GRAPH_DIR:-}" ]; then
    GRAPH_ARGS=(--graph-dir "$NEUROFLY_GRAPH_DIR")
fi

PID_FILE="$DIR/outputs/neurofly_daemon.pid"
# The launcher (the process started here) supervises the simulation and web processes;
# in the default split layout the simulation process writes its own PID to PID_FILE.
LAUNCHER_FILE="$DIR/outputs/neurofly_launcher.pid"
LOG_FILE="$DIR/outputs/neurofly_daemon.log"
mkdir -p "$DIR/outputs"

for f in "$LAUNCHER_FILE" "$PID_FILE"; do
    if [ -f "$f" ]; then
        PID=$(cat "$f")
        if kill -0 "$PID" 2>/dev/null; then
            echo "[NeuroFly] Daemon is already running (PID: $PID, Port: $PORT)."
            exit 0
        else
            rm -f "$f"
        fi
    fi
done

# Detect Python interpreter (prefer local .venv if present)
if [ -f "$DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$DIR/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    PYTHON_BIN="python"
fi

echo "[NeuroFly] Launching Continuous Learning Daemon in background..."
echo "[NeuroFly] Interpreter: $PYTHON_BIN | Bind: $HOST:$PORT | Speed: ${SPEED}x | Assay: $PARADIGM | Backend: ${BACKEND:-default (connectome-fixed if the graph is prepared)}"
if [ "$HOST" = "127.0.0.1" ]; then
    echo "[NeuroFly] Note: reachable from this machine only. For LAN access set NEUROFLY_HOST=0.0.0.0 (commands are then open to your LAN)."
else
    echo "[NeuroFly] Note: listening on $HOST (reachable from your LAN). Unset NEUROFLY_HOST for this machine only."
fi
if [ -n "$BACKEND" ] && [ "$BACKEND" != "modular" ] && [ -z "${NEUROFLY_GRAPH_DIR:-}" ]; then
    echo "[NeuroFly] Note: $BACKEND needs the prepared graph (default location outputs/brainlab/malecns_v1, or NEUROFLY_GRAPH_DIR / --graph-dir)."
fi
if [ "${NEUROFLY_PUBLIC:-0}" != "0" ]; then
    echo "[NeuroFly] Public mode requested via NEUROFLY_PUBLIC (commands need NEUROFLY_ADMIN_TOKEN)."
fi

nohup "$PYTHON_BIN" neurofly_daemon.py \
    --host "$HOST" \
    --port "$PORT" \
    --speed "$SPEED" \
    --paradigm "$PARADIGM" \
    "${BACKEND_ARGS[@]}" "${GRAPH_ARGS[@]}" \
    --pid-file "$PID_FILE" "$@" > "$LOG_FILE" 2>&1 &

DAEMON_PID=$!
echo "$DAEMON_PID" > "$LAUNCHER_FILE"
sleep 1

if kill -0 "$DAEMON_PID" 2>/dev/null; then
    echo "[NeuroFly] Daemon successfully active (PID: $DAEMON_PID)."
    echo "[NeuroFly] Streaming API: http://localhost:$PORT/api/stream"
    echo "[NeuroFly] Log file: $LOG_FILE (the startup backend choice and its reason are printed there)"
else
    echo "[NeuroFly] ERROR: Daemon failed to start. Check log:"
    cat "$LOG_FILE"
    rm -f "$LAUNCHER_FILE"
    exit 1
fi

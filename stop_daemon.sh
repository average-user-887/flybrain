#!/usr/bin/env bash
# Project NeuroFly — Stop Continuous Learning Daemon Gracefully
#
# Default (split) layout: SIGTERM goes to the launcher started by start_daemon.sh, which
# stops the simulation process first (its final checkpoint save), then the web process,
# and removes the private IPC socket. One-process daemons (--single-process) and older
# starts without a launcher PID file are stopped through the daemon PID file as before.
#
# Environment:
#   NEUROFLY_STOP_TIMEOUT   seconds to wait for the final save before force-killing (default 120)

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PID_FILE="$DIR/outputs/neurofly_daemon.pid"
LAUNCHER_FILE="$DIR/outputs/neurofly_launcher.pid"
LOG_FILE="$DIR/outputs/neurofly_daemon.log"
TIMEOUT="${NEUROFLY_STOP_TIMEOUT:-120}"
# Exit status: 0 only if every signalled process exited on SIGTERM; 1 after any forced
# kill. A process exiting is not proof that its final checkpoint is durable: that
# evidence is the daemon's own shutdown log and receipts.
RC=0

is_neurofly() {
    # Guard against a recycled PID: only signal a NeuroFly daemon process.
    kill -0 "$1" 2>/dev/null || return 1
    if [ -r "/proc/$1/cmdline" ]; then
        tr '\0' ' ' < "/proc/$1/cmdline" | grep -q "neurofly" || return 1
    fi
    return 0
}

stop_pid() {
    local pid="$1" what="$2"
    echo "[NeuroFly] Sending SIGTERM to $what (PID: $pid); waiting up to ${TIMEOUT}s for the final save..."
    kill "$pid"
    local waited=0
    while [ "$waited" -lt $((TIMEOUT * 2)) ]; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "[NeuroFly] Process $pid exited after SIGTERM (check the daemon log for its final save)."
            return 0
        fi
        sleep 0.5
        waited=$((waited + 1))
    done
    echo "[NeuroFly] Force killing PID $pid after ${TIMEOUT}s (the final save may be incomplete)..."
    kill -9 "$pid" 2>/dev/null
    return 1
}

if [ -f "$LAUNCHER_FILE" ]; then
    LPID=$(cat "$LAUNCHER_FILE")
    if is_neurofly "$LPID"; then
        if ! stop_pid "$LPID" "the launcher (the simulation saves first, then the web process stops)"; then
            RC=1
            # A force-killed launcher cannot remove its private IPC socket directory.
            IPC=$(grep -o 'IPC [^ ]*/neurofly-split-[^ /]*/sim.sock' "$LOG_FILE" 2>/dev/null | tail -1 | cut -d' ' -f2)
            if [ -n "$IPC" ]; then
                rm -f "$IPC"
                rmdir "$(dirname "$IPC")" 2>/dev/null
            fi
        fi
        rm -f "$LAUNCHER_FILE"
        # The simulation removes PID_FILE on its own clean shutdown; a leftover is stale.
        if [ -f "$PID_FILE" ] && ! is_neurofly "$(cat "$PID_FILE")"; then
            rm -f "$PID_FILE"
        fi
        if [ ! -f "$PID_FILE" ]; then
            exit "$RC"
        fi
    else
        rm -f "$LAUNCHER_FILE"
    fi
fi

if [ ! -f "$PID_FILE" ]; then
    echo "[NeuroFly] No running daemon found (PID file missing)."
    exit "$RC"
fi

PID=$(cat "$PID_FILE")
if is_neurofly "$PID"; then
    stop_pid "$PID" "Daemon" || RC=1
    rm -f "$PID_FILE"
else
    echo "[NeuroFly] Stale PID file removed (process not active)."
    rm -f "$PID_FILE"
fi
exit "$RC"

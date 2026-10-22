#!/usr/bin/env bash
# Server in the background, survives an ssh drop: start | stop | status | log.
# Port 8100: 8000 on this machine is taken by docker. Listens on 127.0.0.1 only — it's reached from outside via ssh (scripts/ask.py
# on this machine or ssh -L 8100:127.0.0.1:8100 from a laptop).
#
#   scripts/llm-server.sh start [extra serve arguments]
#
# start with different arguments than the running server restarts it.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN="$ROOT/target/llm-server"
PID="$RUN/serve.pid"
LOG="$RUN/serve.log"
ARGS="$RUN/serve.args"
PORT="${QWC_PORT:-8100}"
URL="http://127.0.0.1:$PORT"
mkdir -p "$RUN"

running() { [[ -f $PID ]] && kill -0 "$(cat "$PID")" 2>/dev/null; }

stop() {
    if running; then
        kill "$(cat "$PID")"
        # Wait for exit: a new server won't fit in VRAM while the old one holds it.
        while running; do sleep 0.5; done
        echo "stopped"
    else
        echo "not running"
    fi
    rm -f "$PID" "$ARGS"
}

case "${1:-status}" in
start)
    shift || true
    if running; then
        if [[ "$(cat "$ARGS" 2>/dev/null)" == "$*" ]]; then
            echo "already running, pid $(cat "$PID")"; exit 0
        fi
        echo "different mode, restarting"
        stop
    fi
    export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
    cargo build --release -q -p qwc-engine --bin serve --manifest-path "$ROOT/Cargo.toml"
    setsid nohup "$ROOT/target/release/serve" --bind "127.0.0.1:$PORT" "$@" >"$LOG" 2>&1 </dev/null &
    echo $! >"$PID"
    echo "$*" >"$ARGS"
    echo -n "loading model"
    until curl -sf "$URL/v1/models" >/dev/null; do
        if ! running; then echo; echo "server crashed:"; tail -20 "$LOG"; exit 1; fi
        echo -n .; sleep 2
    done
    echo; echo "ready: $URL (pid $(cat "$PID"), log $LOG)"
    ;;
stop) stop ;;
status)
    if running && curl -sf "$URL/v1/models" >/dev/null; then echo "running, pid $(cat "$PID"), arguments: $(cat "$ARGS" 2>/dev/null)"
    elif running; then echo "loading, pid $(cat "$PID")"
    else echo "not running"; fi
    ;;
log) tail -f "$LOG" ;;
*) echo "usage: $0 start|stop|status|log"; exit 2 ;;
esac

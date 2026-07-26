#!/bin/bash
# Control the live Tektronix display.  Usage: tekctl.sh start|stop|restart|status
#
# Lives in a script deliberately: running pkill from an inline `bash -c ...`
# whose own command line mentions the target is a self-kill - pkill -f matches
# the wrapper shell too, and the wrapper dies before it can restart anything.
# Here the shell's cmdline is just this file's path, so there is nothing to
# accidentally match. We also kill by recorded PID rather than by pattern.
PID_FILE=/home/super/tekscreen.pid
LOG=/home/super/tekscreen.log
export DISPLAY=:0
export XAUTHORITY=/home/super/.Xauthority
export OPENBLAS_CORETYPE=ARMV8      # numpy SIGILLs on Cortex-A57 without this

stop() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        kill "$(cat "$PID_FILE")" 2>/dev/null
        sleep 1
        kill -9 "$(cat "$PID_FILE")" 2>/dev/null
        echo "stopped $(cat "$PID_FILE")"
    else
        echo "not running (no live pid file)"
    fi
    rm -f "$PID_FILE"
}

start() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "already running: $(cat "$PID_FILE")"; return
    fi
    : > "$LOG"
    cd /home/super || exit 1
    nohup python3 tekscreen.py "$@" >> "$LOG" 2>&1 &
    echo $! > "$PID_FILE"
    echo "started $(cat "$PID_FILE")"
}

case "${1:-status}" in
    start)   shift; start "$@" ;;
    stop)    stop ;;
    restart) stop; shift; start "$@" ;;
    status)
        if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            echo "running: $(cat "$PID_FILE")"
        else
            echo "not running"
        fi
        [ -s "$LOG" ] && { echo "--- log ---"; tail -6 "$LOG"; }
        ;;
    *) echo "usage: $0 start|stop|restart|status"; exit 1 ;;
esac

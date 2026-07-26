#!/bin/bash
# Prove the display comes back from being killed - repeatedly, and more times
# than the old start limit (5 in 300s) would ever have allowed.
echo "=== kill/recover test ==="
for i in 1 2 3 4 5 6 7; do
    PID=$(systemctl show tek-display -p MainPID --value)
    kill -9 "$PID" 2>/dev/null
    sleep 5
    NEW=$(systemctl show tek-display -p MainPID --value)
    ACT=$(systemctl is-active tek-display)
    echo "  kill $i: pid $PID -> $NEW   active=$ACT"
done
echo
echo "final state: $(systemctl is-active tek-display)"
echo "restarts:    $(systemctl show tek-display -p NRestarts --value)"

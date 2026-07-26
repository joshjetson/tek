#!/bin/bash
# Verify the display will come up on a cold boot - WITHOUT rebooting.
#
# Checks the things that actually differ at boot: unit correctness, enablement,
# dependency ordering, whether anything it needs is only present because of
# this login session, and a genuine cold start with every cache cleared.
echo "=== 1. unit file valid ==="
systemd-analyze verify /etc/systemd/system/tek-display.service 2>&1 \
    | grep -v "Failed to create\|Failed to initialize" | head -5
echo "   (no output above = valid)"

echo
echo "=== 2. enabled for boot ==="
echo "   is-enabled : $(systemctl is-enabled tek-display)"
echo "   default    : $(systemctl get-default)"
LINK=/etc/systemd/system/multi-user.target.wants/tek-display.service
[ -L "$LINK" ] && echo "   wants-link : OK -> $(readlink $LINK)" \
               || echo "   wants-link : *** MISSING - will NOT start at boot ***"

echo
echo "=== 3. no start limit (must retry forever) ==="
echo "   StartLimitIntervalUSec : $(systemctl show tek-display -p StartLimitIntervalUSec --value)"
echo "   Restart                : $(systemctl show tek-display -p Restart --value)"

echo
echo "=== 4. paths the unit depends on ==="
for p in $(grep -oE '/[A-Za-z0-9_./-]+\.py' /etc/systemd/system/tek-display.service) \
         $(grep -oE 'WorkingDirectory=\S+' /etc/systemd/system/tek-display.service | cut -d= -f2); do
    [ -e "$p" ] && echo "   OK      $p" || echo "   MISSING $p"
done

echo
echo "=== 5. environment: is OPENBLAS_CORETYPE in the UNIT, not just the shell? ==="
grep -q "OPENBLAS_CORETYPE" /etc/systemd/system/tek-display.service \
    && echo "   OK - set in the unit (numpy SIGILLs without it)" \
    || echo "   *** NOT in the unit - it will crash at boot ***"

echo
echo "=== 6. runs as a user that can reach the framebuffer ==="
U=$(systemctl show tek-display -p User --value)
echo "   User=$U   groups: $(id -nG $U 2>/dev/null)"
id -nG "$U" 2>/dev/null | tr ' ' '\n' | grep -qx video \
    && echo "   OK - in 'video', can open /dev/fb0" \
    || echo "   *** not in 'video' group ***"

echo
echo "=== 7. COLD START (all caches cleared, exactly as first boot) ==="
# sudo needs the password on a non-interactive stdin, and the log window must
# start AFTER the restart - an earlier version matched fps lines from the
# still-running old process and reported a 0.03s "cold start".
SUDO="echo ${TEKPW:?set TEKPW} | sudo -S"
eval "$SUDO systemctl stop tek-display" 2>/dev/null
rm -rf ~/.cache/tekdromo
# Report by PID, not by timestamp. journalctl --since has one-second
# granularity, so a window opened right after a restart still contains the
# OUTGOING process's lines - which is how an earlier run "measured" a warm
# start using the old process's output.
report() {
    local pid="" i
    for i in $(seq 1 120); do
        pid=$(systemctl show tek-display -p MainPID --value)
        [ "$pid" != "0" ] && [ "$pid" != "$1" ] && \
            journalctl _PID=$pid --no-pager 2>/dev/null | grep -q "poses warm" && break
        sleep 0.5
    done
    echo "   PID was $1, now $pid"
    journalctl _PID=$pid --no-pager \
        | grep -oE '\[.*|geometry .*|camera .*' | sed 's/^/   /'
}

OLDPID=$(systemctl show tek-display -p MainPID --value)
eval "$SUDO systemctl start tek-display" 2>/dev/null
report "$OLDPID"

echo
echo "=== 8. WARM start (caches present, as on every boot after the first) ==="
OLDPID=$(systemctl show tek-display -p MainPID --value)
eval "$SUDO systemctl restart tek-display" 2>/dev/null
report "$OLDPID"

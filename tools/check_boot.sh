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

# --------------------------------------------------------------------------
# 9 and 10 exist because of a real lockout: the box rebooted, the display
# covered the console, and the network did not come up because the wifi
# profile was user-scoped and nobody had logged in. Both are boot-only
# failures - everything looks perfect from an established session, which is
# exactly why they need checking from here rather than by eye.
# --------------------------------------------------------------------------
echo
echo "=== 9. network comes up WITHOUT a login session ==="
BAD=0
for f in /etc/NetworkManager/system-connections/*; do
    [ -e "$f" ] || continue
    NAME=$(basename "$f")
    PERM=$(eval "$SUDO grep -h '^permissions=' \"$f\"" 2>/dev/null | cut -d= -f2-)
    AUTO=$(nmcli -g connection.autoconnect con show "$NAME" 2>/dev/null)
    # psk-flags absent or 0 means NetworkManager holds the secret itself. Any
    # other value means it asks a secret agent, and there is no agent until
    # somebody logs in.
    FLAGS=$(eval "$SUDO grep -h '^psk-flags=' \"$f\"" 2>/dev/null | cut -d= -f2-)
    FLAGS=${FLAGS:-0}
    if [ -n "$PERM" ]; then
        echo "   *** $NAME: permissions=$PERM - will NOT connect before login ***"
        BAD=1
    elif [ "$FLAGS" != "0" ]; then
        echo "   *** $NAME: psk-flags=$FLAGS - needs a logged-in secret agent ***"
        BAD=1
    else
        echo "   OK      $NAME (system-scoped, secret stored, autoconnect=$AUTO)"
    fi
done
[ "$BAD" = 0 ] && echo "   -> SSH will be reachable before anyone logs in"

echo
echo "=== 10. the panic key (ESC x3) will be running ==="
echo "   is-enabled : $(systemctl is-enabled tek-panic 2>&1)"
echo "   is-active  : $(systemctl is-active tek-panic 2>&1)"
LINK=/etc/systemd/system/multi-user.target.wants/tek-panic.service
[ -L "$LINK" ] && echo "   wants-link : OK" \
               || echo "   wants-link : *** MISSING - no escape hatch at boot ***"
# An empty User= means root, which is what this unit needs: reading
# /dev/input and calling systemctl both require it.
PU=$(systemctl show tek-panic -p User --value)
echo "   User       : ${PU:-root} $([ -z "$PU" ] && echo '(needed for /dev/input + systemctl)' || echo '*** must be root ***')"
PPID_=$(systemctl show tek-panic -p MainPID --value)
NDEV=$(eval "$SUDO ls -l /proc/$PPID_/fd" 2>/dev/null | grep -c 'input/event')
echo "   watching   : $NDEV input device(s)"
[ "${NDEV:-0}" -gt 0 ] || echo "   *** watching nothing - the chord cannot fire ***"
# It must not depend on the thing it exists to escape from.
if systemctl show tek-panic -p After --value | grep -q tek-display; then
    echo "   *** ordered After=tek-display - it must NOT be ***"
else
    echo "   ordering   : OK (independent of tek-display)"
fi

echo
echo "=== 10b. PulseAudio must not exit when idle (it takes A2DP with it) ==="
IDLE=$(pulseaudio --dump-conf 2>/dev/null | grep -oE "exit-idle-time = -?[0-9]+" | awk '{print $3}')
echo "   exit-idle-time : ${IDLE:-unknown}"
if [ "$IDLE" = "-1" ]; then
    echo "   OK - the daemon never exits, so the speaker stays paired"
else
    echo "   *** ${IDLE}s: any moment with no client kills PulseAudio, and the"
    echo "       Bluetooth speaker disconnects with it (96 restarts in one day) ***"
fi
N=$(journalctl --no-pager --since today 2>/dev/null | grep -oE "pulseaudio\[[0-9]+\]" | sort -u | wc -l)
echo "   distinct PulseAudio processes today: $N  (want a small number)"

echo
echo "=== 11. tty1 autologin (a shell is waiting after a panic) ==="
if systemctl show getty@tty1 -p ExecStart --value | grep -q -- "--autologin"; then
    echo "   OK - agetty --autologin: $(systemctl show getty@tty1 -p ExecStart --value | grep -oE '\-\-autologin [a-z]+')"
else
    echo "   *** no autologin - a panic leaves a login prompt to type blind ***"
fi

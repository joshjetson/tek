#!/bin/bash
# Keep the Bluetooth speaker connected and keep audio routed to it.
#
# Three separate things can silently drop the speaker, and all three have bitten
# this machine already:
#
#  1. The A2DP link idles out. PulseAudio's module-suspend-on-idle suspends the
#     sink when nothing is playing; the speaker then sees no stream and powers
#     the link down. Unloading it keeps the transport open.
#  2. PulseAudio restarts. Every restart tears down the A2DP transport, and the
#     module has to be unloaded again afterwards.
#  3. The speaker is simply switched off and on again.
#
# So this does not "connect once" - it continuously reasserts all three.
#
# Uses D-Bus directly rather than driving bluetoothctl. bluetoothctl is an
# interactive REPL: piping commands into it means racing its startup, and it
# reports a permission denial as "No default controller available", which sends
# you looking at the wrong thing entirely. The D-Bus calls either work or return
# a real error.
set -u

MAC="${TEK_BT_MAC:-A0:E9:DB:16:01:E7}"           # Marley Get Together
DEV="/org/bluez/hci0/dev_${MAC//:/_}"
SINK="bluez_sink.${MAC//:/_}.a2dp_sink"
PERIOD="${TEK_BT_PERIOD:-15}"

say() { echo "$(date '+%H:%M:%S') $*"; }

connected() {
    dbus-send --system --dest=org.bluez --print-reply "$DEV" \
        org.freedesktop.DBus.Properties.Get \
        string:org.bluez.Device1 string:Connected 2>/dev/null \
        | grep -q "boolean true"
}

was=unknown
while true; do
    if connected; then
        [ "$was" = yes ] || say "connected to $MAC"
        was=yes

        # Reassert on every pass: these are cheap, and PulseAudio may have
        # restarted since the last one without us noticing.
        if pactl list modules short 2>/dev/null | grep -q suspend-on-idle; then
            pactl unload-module module-suspend-on-idle 2>/dev/null \
                && say "unloaded module-suspend-on-idle (was letting the link idle out)"
        fi
        if pactl list sinks short 2>/dev/null | grep -q "$SINK"; then
            cur=$(pactl info 2>/dev/null | sed -n 's/^Default Sink: //p')
            if [ "$cur" != "$SINK" ]; then
                pactl set-default-sink "$SINK" 2>/dev/null \
                    && say "routed audio to $SINK"
            fi
        fi
    else
        [ "$was" = no ] || say "not connected - reconnecting $MAC"
        was=no
        dbus-send --system --dest=org.bluez --print-reply --reply-timeout=20000 \
            "$DEV" org.bluez.Device1.Connect >/dev/null 2>&1 \
            && say "reconnect issued OK" || true
    fi
    sleep "$PERIOD"
done

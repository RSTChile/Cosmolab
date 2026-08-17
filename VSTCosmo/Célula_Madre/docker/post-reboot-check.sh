#!/usr/bin/env bash
# Verificación post-reinicio (ejecutar desde Mac: ssh rpi bash /home/ubuntu/anima/celula_madre/docker/post-reboot-check.sh)
set -uo pipefail
CM="/home/ubuntu/anima/celula_madre"
echo "=== framebuffer ==="
ls -la /dev/fb* 2>/dev/null || echo "sin fb"
for f in /sys/class/graphics/fb*; do
  [ -f "$f/virtual_size" ] && echo "$(basename $f): $(cat $f/virtual_size) $(cat $f/name 2>/dev/null)"
done
echo "=== systemd ==="
systemctl --user is-active anima-organismo.service anima-watchdog.service anima-pi-screen.service 2>/dev/null || true
echo "=== organismo ==="
"$CM/docker/run_native_pi.sh" status 2>&1 | head -2 || true
echo "=== pi-screen ==="
pgrep -af anima-pi-screen || echo "no corre"
echo "=== cabeza web ==="
curl -sf http://127.0.0.1:7788/cabeza | head -c 80 || echo "sin /cabeza"
echo ""
#!/usr/bin/env bash
# Pantalla EXTENDIDA: PiScreen en fb1 (cabeza) + HDMI a resolución nativa.
# El modo drm=1 duplica ambas a 480x320. Este script vuelve a fbtft clásico.
#   sudo bash /home/ubuntu/anima/celula_madre/docker/fix_piscreen_extended.sh && sudo reboot
set -euo pipefail
CFG="/boot/firmware/config.txt"
[ -f "$CFG" ] || CFG="/boot/config.txt"
cp "$CFG" "${CFG}.bak.extended.$(date +%Y%m%d%H%M%S)"

sed -i 's/^dtoverlay=spi1-3cs/#dtoverlay=spi1-3cs/' "$CFG"
sed -i 's/^#dtoverlay=spi1-3cs/#dtoverlay=spi1-3cs/' "$CFG"

# PiScreen SIN drm → fb1 directo (anima-pi-screen pinta la cabeza)
sed -i 's/^dtoverlay=piscreen.*/dtoverlay=piscreen,speed=16000000,rotate=90/' "$CFG"

# HDMI independiente a 1080p
grep -q '^hdmi_force_mode=' "$CFG" || echo "hdmi_force_mode=1" >>"$CFG"
grep -q '^hdmi_group=' "$CFG" || echo "hdmi_group=2" >>"$CFG"
grep -q '^hdmi_mode=' "$CFG" || echo "hdmi_mode=82" >>"$CFG"
sed -i 's/^#hdmi_force_mode=/hdmi_force_mode=/' "$CFG"
sed -i 's/^#hdmi_group=/hdmi_group=/' "$CFG"
sed -i 's/^#hdmi_mode=/hdmi_mode=/' "$CFG"

echo "Config aplicada:"
grep -E "piscreen|spi1|hdmi" "$CFG" | grep -v "^#" | head -10
echo "Reinicia: sudo reboot"
echo "Tras reinicio: pantalla pequeña = cabeza (fb1), HDMI = escritorio extendido."
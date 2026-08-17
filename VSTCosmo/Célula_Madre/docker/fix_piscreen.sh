#!/usr/bin/env bash
# Arregla PiScreen (pantalla pequeña SPI) en la Pi. Ejecutar EN la Pi:
#   sudo bash /home/ubuntu/anima/celula_madre/docker/fix_piscreen.sh
# Luego reiniciar.
set -euo pipefail

CFG="/boot/firmware/config.txt"
[ -f "$CFG" ] || CFG="/boot/config.txt"
BAK="${CFG}.bak.$(date +%Y%m%d%H%M%S)"

cp "$CFG" "$BAK"
echo "Backup: $BAK"

# spi1-3cs suele chocar con PiScreen en spi0
sed -i 's/^dtoverlay=spi1-3cs/#dtoverlay=spi1-3cs  # desactivado: conflicto PiScreen/' "$CFG"

# HDMI forzado puede impedir que la TFT arranque como fb1
sed -i 's/^hdmi_force_mode=/#hdmi_force_mode=/' "$CFG"
sed -i 's/^hdmi_group=/#hdmi_group=/' "$CFG"
sed -i 's/^hdmi_mode=/#hdmi_mode=/' "$CFG"

# PiScreen SIN drm → fb1 directo (cabeza vía headless) + HDMI extendido
if grep -q '^dtoverlay=piscreen' "$CFG"; then
  sed -i 's/^dtoverlay=piscreen.*/dtoverlay=piscreen,speed=16000000,rotate=90/' "$CFG"
else
  echo "dtoverlay=piscreen,speed=16000000,rotate=90" >>"$CFG"
fi
# HDMI 1080p independiente (evita logo Ubuntu duplicado en monitor externo)
grep -q '^hdmi_force_mode=' "$CFG" || echo "hdmi_force_mode=1" >>"$CFG"
grep -q '^hdmi_group=' "$CFG" || echo "hdmi_group=2" >>"$CFG"
grep -q '^hdmi_mode=' "$CFG" || echo "hdmi_mode=82" >>"$CFG"
sed -i 's/^#hdmi_force_mode=/hdmi_force_mode=/' "$CFG"
sed -i 's/^#hdmi_group=/hdmi_group=/' "$CFG"
sed -i 's/^#hdmi_mode=/hdmi_mode=/' "$CFG"

echo ""
echo "config.txt actualizado. Cambios:"
grep -E "piscreen|spi1|hdmi_force|hdmi_group|hdmi_mode" "$CFG" | grep -v "^#" || true
echo ""
echo "Reinicia: sudo reboot"
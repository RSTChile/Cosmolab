#!/usr/bin/env bash
# HDMI como pantalla PRINCIPAL (escritorio usable). Desactiva PiScreen drm que bloquea a 480x320.
# Ejecutar desde el Mac (te pedirá contraseña sudo una vez):
#   ssh -t rpi 'sudo bash /home/ubuntu/anima/celula_madre/docker/fix_hdmi_primary.sh'
set -euo pipefail

CM="/home/ubuntu/anima/celula_madre"
CFG="/boot/firmware/config.txt"
[ -f "$CFG" ] || CFG="/boot/config.txt"
cp "$CFG" "${CFG}.bak.hdmi.$(date +%Y%m%d%H%M%S)"

# PiScreen off → X usa solo HDMI
sed -i 's/^dtoverlay=piscreen/#dtoverlay=piscreen  # off: HDMI principal/' "$CFG"

# HDMI 1080p
grep -q '^hdmi_force_mode=' "$CFG" || echo "hdmi_force_mode=1" >>"$CFG"
grep -q '^hdmi_group=' "$CFG" || echo "hdmi_group=2" >>"$CFG"
grep -q '^hdmi_mode=' "$CFG" || echo "hdmi_mode=82" >>"$CFG"
sed -i 's/^#hdmi_force_mode=/hdmi_force_mode=/' "$CFG"
sed -i 's/^#hdmi_group=/hdmi_group=/' "$CFG"
sed -i 's/^#hdmi_mode=/hdmi_mode=/' "$CFG"

# Tras reinicio: escritorio libre en HDMI (sin kiosk ni pantalla chica)
mkdir -p /home/ubuntu/.config/environment.d
cat >/home/ubuntu/.config/environment.d/anima.conf <<'EOF'
ANIMA_DISPLAY=desktop
EOF
chown ubuntu:ubuntu /home/ubuntu/.config/environment.d/anima.conf

systemctl --user -M ubuntu@ disable anima-pi-headless.service 2>/dev/null || true

echo ""
echo "Config aplicada:"
grep -E "piscreen|hdmi" "$CFG" | grep -v "^#" | head -8
echo ""
echo "REINICIANDO en 5 s… (HDMI = escritorio normal 1080p)"
sleep 5
reboot
#!/usr/bin/env bash
# Plan B: PiScreen sin DRM (framebuffer clásico fb1). Si fix_piscreen.sh con drm=1 falla.
#   sudo bash /home/ubuntu/anima/celula_madre/docker/fix_piscreen_fbtft.sh && sudo reboot
set -euo pipefail
CFG="/boot/firmware/config.txt"
[ -f "$CFG" ] || CFG="/boot/config.txt"
cp "$CFG" "${CFG}.bak.fbtft.$(date +%Y%m%d%H%M%S)"
sed -i 's/^dtoverlay=spi1-3cs/#dtoverlay=spi1-3cs/' "$CFG"
sed -i 's/^hdmi_force_mode=/#hdmi_force_mode=/' "$CFG"
sed -i 's/^hdmi_group=/#hdmi_group=/' "$CFG"
sed -i 's/^hdmi_mode=/#hdmi_mode=/' "$CFG"
sed -i 's/^dtoverlay=piscreen.*/dtoverlay=piscreen,speed=8000000,rotate=90/' "$CFG"
grep -E "piscreen|spi1|hdmi" "$CFG" | grep -v "^#" | head -10
echo "Reinicia: sudo reboot"
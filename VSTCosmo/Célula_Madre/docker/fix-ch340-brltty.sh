#!/usr/bin/env bash
# Evita que brltty robe el CH340 (ATmega USB-serial). Ejecutar UNA vez en la Pi con sudo.
set -euo pipefail

echo "==> Deteniendo brltty (robaba ttyUSB0 del CH340)..."
systemctl stop brltty-udev.service 2>/dev/null || true
systemctl stop brltty.service 2>/dev/null || true
systemctl disable brltty-udev.service 2>/dev/null || true
systemctl disable brltty.service 2>/dev/null || true
systemctl mask brltty-udev.service 2>/dev/null || true

RULE=/etc/udev/rules.d/99-anima-ch340-brltty.rules
echo "==> Regla udev: CH340 no es dispositivo braille"
cat > "$RULE" <<'EOF'
# ANIMA cloroplasto: CH340 ATmega — no reclamar como braille
ACTION=="add", SUBSYSTEM=="usb", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="7523", ENV{ID_MM_DEVICE_IGNORE}="1", ENV{ID_MM_PORT_IGNORE}="1"
ACTION=="add", SUBSYSTEM=="usb", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="7523", TAG+="uaccess", GROUP="dialout", MODE="0666"
EOF

udevadm control --reload-rules
udevadm trigger

echo "==> Reconecta el USB de la ATmega (desenchufa y enchufa)..."
sleep 2
ls -la /dev/ttyUSB* /dev/serial/by-id/* 2>/dev/null || echo "AVISO: aún sin ttyUSB — desenchufa/enchufa el cable USB"

echo "OK — Arduino IDE → Herramientas → Puerto → /dev/ttyUSB0"
echo "     Placa → Arduino Mega or Mega 2560 (no hace falta flashear)"
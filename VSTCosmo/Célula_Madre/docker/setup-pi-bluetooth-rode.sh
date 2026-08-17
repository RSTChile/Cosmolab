#!/usr/bin/env bash
# Audio Bluetooth: Pi recibe el monitor mix del RØDECaster Pro → organismo escucha.
# Ejecutar en la Pi (sudo para paquetes):
#   sudo bash /home/ubuntu/anima/celula_madre/docker/setup-pi-bluetooth-rode.sh
set -euo pipefail

CM="/home/ubuntu/anima/celula_madre"
USER_NAME="${SUDO_USER:-ubuntu}"

echo "=== 1. Paquetes (Bluetooth A2DP + PortAudio + sounddevice) ==="
apt-get update -qq
apt-get install -y bluez pulseaudio pulseaudio-module-bluetooth \
  libportaudio2 portaudio19-dev python3-pip libsndfile1

sudo -u "$USER_NAME" pip3 install --user sounddevice 2>/dev/null || \
  pip3 install --user sounddevice

echo "=== 2. PulseAudio: perfil A2DP sink (la Pi actúa como altavoz BT) ==="
PA_USER="/home/$USER_NAME/.config/pulse"
mkdir -p "$PA_USER"
cat >"$PA_USER/default.pa.d/bluetooth-rode.pa" <<'EOF'
load-module module-switch-on-connect
load-module module-bluetooth-discover
EOF
chown -R "$USER_NAME:$USER_NAME" "/home/$USER_NAME/.config/pulse"

BT_MAIN="/etc/bluetooth/main.conf"
if [ -f "$BT_MAIN" ]; then
  grep -q '^Class ' "$BT_MAIN" || echo 'Class = 0x200414' >>"$BT_MAIN"
  sed -i 's/^#Class = .*/Class = 0x200414/' "$BT_MAIN" 2>/dev/null || true
  sed -i 's/^DiscoverableTimeout = .*/DiscoverableTimeout = 0/' "$BT_MAIN" 2>/dev/null || \
    echo 'DiscoverableTimeout = 0' >>"$BT_MAIN"
fi

systemctl enable bluetooth
systemctl restart bluetooth
systemctl --user -M "$USER_NAME@" restart pulseaudio 2>/dev/null || true

echo "=== 3. Variables del organismo (audio en vivo) ==="
ENV_FILE="$CM/docker/.env.pi"
grep -q '^ANIMA_PI_AUDIO_DEVICE=' "$ENV_FILE" 2>/dev/null || \
  echo 'ANIMA_PI_AUDIO_DEVICE=bluez' >>"$ENV_FILE"
grep -q '^VST_DISABLE_DIRECT_AUDIO=' "$ENV_FILE" 2>/dev/null || \
  echo 'VST_DISABLE_DIRECT_AUDIO=0' >>"$ENV_FILE"
chown "$USER_NAME:$USER_NAME" "$ENV_FILE" 2>/dev/null || true

cat >"$CM/docker/pair-rodecaster.sh" <<'PAIR'
#!/usr/bin/env bash
# Emparejar RØDECaster → Pi (monitor mix por Bluetooth)
set -euo pipefail
echo "En el RØDECaster: Ajustes → Bluetooth → conectar auriculares inalámbricos / monitor."
echo "Busca este dispositivo: $(hostname)"
echo ""
bluetoothctl <<'BT'
power on
agent on
default-agent
discoverable on
pairable on
scan on
BT
echo ""
echo "Cuando aparezca el Rodecaster:  pair XX:XX:XX:XX:XX:XX  →  trust XX:XX:XX:XX:XX:XX  →  connect XX:XX:XX:XX:XX:XX"
echo "Verifica fuente de audio:  pactl list sources short | grep -i blue"
PAIR
chmod +x "$CM/docker/pair-rodecaster.sh"
chown "$USER_NAME:$USER_NAME" "$CM/docker/pair-rodecaster.sh"

echo ""
echo "LISTO."
echo "  1) Reinicia sesión o:  systemctl --user restart pulseaudio"
echo "  2) Empareja:  bash $CM/docker/pair-rodecaster.sh"
echo "  3) Reinicia organismo:  bash $CM/docker/run_native_pi.sh stop && bash $CM/docker/run_native_pi.sh start"
echo "  4) Comprueba:  curl -s http://127.0.0.1:7788/dispositivos | python3 -m json.tool"
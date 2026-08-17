#!/usr/bin/env bash
# OBSOLETO para la arquitectura Mac — usa el servidor TCP:
#   bash anima-pi-rode-via-server.sh IP_DEL_MAC 0 1
# Este script (captura directa BT) queda como respaldo experimental.
# Alimenta al organismo con el audio del RØDECaster Pro II por Bluetooth.
# NO toca los defaults de GNOME (evita el parpadeo entrada/salida).
# Uso: bash anima-pi-rode-capture.sh
#      bash anima-pi-rode-capture.sh --wait   # espera hasta 45 s a que haya señal
set -euo pipefail

CM="/home/ubuntu/anima/celula_madre"
PORT="${VST_PUERTO:-7788}"
WAIT="${1:-}"

CARD=$(pactl list cards short 2>/dev/null | grep -i bluez_card | awk '{print $2}' | head -1)
if [ -z "$CARD" ]; then
  echo "ERROR: Rodecaster BT no emparejado. Ejecuta: bash $CM/docker/pair-rodecaster.sh"
  exit 1
fi

# Guardar defaults del usuario (restaurar al salir)
OLD_SRC=$(pactl get-default-source 2>/dev/null || true)
OLD_SNK=$(pactl get-default-sink 2>/dev/null || true)
OLD_PROFILE=$(pactl list cards 2>/dev/null | awk -v c="$CARD" '$0 ~ c {p=1} p && /Active Profile:/ {print $3; exit}')
cleanup() {
  [ -n "${OLD_PROFILE:-}" ] && pactl set-card-profile "$CARD" "$OLD_PROFILE" 2>/dev/null || true
  [ -n "${OLD_SRC:-}" ] && pactl set-default-source "$OLD_SRC" 2>/dev/null || true
  [ -n "${OLD_SNK:-}" ] && pactl set-default-sink "$OLD_SNK" 2>/dev/null || true
}
trap cleanup EXIT

pick_source() {
  # A2DP: captura el monitor del sink (lo que el Rode ENVÍA a la Pi para escuchar)
  local mon
  mon=$(pactl list sources short 2>/dev/null | grep -i 'bluez_sink.*a2dp_sink.*monitor' | awk '{print $2}' | head -1)
  if [ -n "$mon" ]; then
    echo "$mon"
    return 0
  fi
  # HFP: captura directa (mono 16 kHz)
  local src
  src=$(pactl list sources short 2>/dev/null | grep -i 'bluez_source.*handsfree' | awk '{print $2}' | head -1)
  [ -n "$src" ] && echo "$src"
}

# Perfil A2DP sink = Rode → Pi (monitor inalámbrico / playback en la Pi)
pactl set-card-profile "$CARD" a2dp_sink 2>/dev/null || true
sleep 1
PULSE_SRC=$(pick_source)
if [ -z "$PULSE_SRC" ]; then
  echo "Probando perfil Handsfree (captura directa)…"
  pactl set-card-profile "$CARD" handsfree_head_unit 2>/dev/null || true
  sleep 2
  PULSE_SRC=$(pick_source)
fi

if [ -z "$PULSE_SRC" ]; then
  echo "ERROR: no hay fuente de captura BT del Rodecaster."
  exit 1
fi

echo "[rode] Captura desde: $PULSE_SRC"
echo "[rode] (sin cambiar tus defaults de GNOME — solo esta sesión)"

measure_peak() {
  PULSE_SOURCE="$PULSE_SRC" python3 <<'PY'
import os, sounddevice as sd, numpy as np
src = os.environ["PULSE_SOURCE"]
# sample rate según fuente
sr = 16000 if "handsfree" in src else 44100
ch = 1 if "handsfree" in src else 2
rec = sd.rec(int(3 * sr), samplerate=sr, channels=ch, device="pulse", dtype="float64")
sd.wait()
peak = float(np.max(np.abs(rec)))
rms = float(np.sqrt(np.mean(rec**2)))
print(f"{peak:.6f} {rms:.6f} {sr} {ch}")
PY
}

echo "[rode] Reproduce audio en el Rode AHORA (monitor BT → ubuntu)…"
PEAK=0
TRIES=1
if [ "$WAIT" = "--wait" ]; then
  TRIES=15
fi

for i in $(seq 1 "$TRIES"); do
  read -r PEAK RMS SR CH < <(measure_peak)
  echo "[rode] intento $i/$TRIES · peak=$PEAK rms=$RMS (${SR}Hz ${CH}ch)"
  awk "BEGIN {exit !($PEAK > 0.003)}" && break
  [ "$TRIES" -gt 1 ] && sleep 3
done

if awk "BEGIN {exit !($PEAK <= 0.003)}"; then
  echo ""
  echo "Sin señal capturada. Comprueba en el RØDECaster Pro II:"
  echo "  • El mix debe salir por MONITOR INALÁMBRICO → dispositivo 'ubuntu' (no solo altavoces del Rode)"
  echo "  • En Ubuntu: ENTRADA = Rodecaster (eso es lo que come el organismo)"
  echo "  • SALIDA = Rodecaster solo si quieres oír en el Rode; para alimentar al animalito importa la ENTRADA"
  echo "  • Vuelve a probar con: bash $0 --wait"
  exit 2
fi

# Persistir para el organismo (sin tocar defaults de GNOME al arrancar)
cat >"$CM/docker/.env.pi" <<EOF
VST_DISABLE_DIRECT_AUDIO=0
ANIMA_PI_AUDIO_DEVICE=pulse
PULSE_SOURCE=$PULSE_SRC
ANIMA_PI_AUDIO_SR=$SR
ANIMA_PI_AUDIO_CH=$CH
EOF

DI=$(PULSE_SOURCE="$PULSE_SRC" python3 -c "
import os, sounddevice as sd
os.environ['PULSE_SOURCE'] = open('$CM/docker/.env.pi').read().split('PULSE_SOURCE=')[1].split()[0]
for i, d in enumerate(sd.query_devices()):
    if d.get('max_input_channels', 0) > 0 and d.get('name', '').lower() == 'pulse':
        print(i)
        break
")

echo "[rode] device_index=$DI · PULSE_SOURCE=$PULSE_SRC"
curl -sf -X POST "http://127.0.0.1:${PORT}/start" \
  -H "Content-Type: application/json" \
  -d "{\"cfg\":{\"left_src\":{\"tipo\":\"dispositivo\",\"device_index\":${DI},\"channel_index\":0},\"right_src\":{\"tipo\":\"dispositivo\",\"device_index\":${DI},\"channel_index\":$(( CH > 1 ? 1 : 0 ))},\"binaural\":true,\"segundos\":2,\"continuo\":true,\"criterio_duracion\":\"min\"}}"

echo ""
echo "[rode] Organismo alimentado. Verifica energía:"
echo "  curl -s http://127.0.0.1:${PORT}/estado | python3 -c \"import sys,json; d=json.load(sys.stdin); print('L',d.get('fuente_L'),d.get('energia_L'),'R',d.get('energia_R'))\""
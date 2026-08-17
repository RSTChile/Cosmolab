#!/usr/bin/env bash
# Conecta el organismo al Rode vía VST_AudioServer (igual que en el Mac).
#
# Caso A — Rode en el Mac (USB), servidor en el Mac:
#   En el Mac:  python3 audio/VST_AudioServer.py --device "Main Multitrack"
#   En la Pi:   bash anima-pi-rode-via-server.sh 192.168.86.XX 0 1
#
# Caso B — Rode en la Pi, servidor local en la Pi:
#   bash anima-audio-server.sh start
#   bash anima-pi-rode-via-server.sh 127.0.0.1 0 1
#
# Canales: índices 0-based (0=ch1 Main L, 1=ch2 Main R, 2=ch3 Combo1 L, …)
set -euo pipefail

CM="/home/ubuntu/anima/celula_madre"
PORT="${VST_SERVIDOR_PORT:-8765}"
HOST="${1:-${VST_SERVIDOR_HOST:-127.0.0.1}}"
CH_L="${2:-${ANIMA_RODE_CH_L:-0}}"
CH_R="${3:-${ANIMA_RODE_CH_R:-1}}"

cd "$CM"
export PYTHONPATH="${CM}:${CM}/audio:${PYTHONPATH:-}"
PYTHON="python3"
if [ -x "${CM}/.venv-pi/bin/python" ] && "${CM}/.venv-pi/bin/python" -c "import numpy" 2>/dev/null; then
  PYTHON="${CM}/.venv-pi/bin/python"
fi

echo "[rode] Comprobando AudioServer en ${HOST}:${PORT}…"
"$PYTHON" <<PY
import sys
sys.path[:0] = ["${CM}", "${CM}/audio"]
from VST_AudioServer import AudioStreamClient
c = AudioStreamClient("${HOST}", ${PORT})
hs = c.handshake()
print("[rode] OK:", hs.get("device"), "·", hs.get("channels"), "canales @", hs.get("sample_rate"), "Hz")
c.cerrar()
PY

cat >"$CM/docker/.env.pi" <<EOF
VST_DISABLE_DIRECT_AUDIO=1
VST_SERVIDOR_HOST=${HOST}
VST_SERVIDOR_PORT=${PORT}
ANIMA_RODE_CH_L=${CH_L}
ANIMA_RODE_CH_R=${CH_R}
EOF

echo "[rode] .env.pi actualizado"
curl -sf -X POST "http://127.0.0.1:${VST_PUERTO:-7788}/start" \
  -H "Content-Type: application/json" \
  -d "{\"cfg\":{\"left_src\":{\"tipo\":\"servidor\",\"host\":\"${HOST}\",\"port\":${PORT},\"channel_index\":${CH_L},\"nombre\":\"Rødecaster L\"},\"right_src\":{\"tipo\":\"servidor\",\"host\":\"${HOST}\",\"port\":${PORT},\"channel_index\":${CH_R},\"nombre\":\"Rødecaster R\"},\"binaural\":true,\"segundos\":2,\"continuo\":true,\"criterio_duracion\":\"min\"}}"

echo ""
echo "[rode] .env.pi guardado — tras reinicio el organismo escuchará solo (run_native_pi + systemd)."
echo "  Reinicia ahora si quieres probar: bash run_native_pi.sh stop && bash run_native_pi.sh start"
echo ""
echo "  curl -s http://127.0.0.1:7788/estado | python3 -c \"import sys,json; d=json.load(sys.stdin); print('L',d.get('fuente_L'),d.get('energia_L'),'RC',d.get('RC_total'))\""
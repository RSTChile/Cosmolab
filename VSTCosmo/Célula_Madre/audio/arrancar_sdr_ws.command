#!/bin/bash
# arrancar_sdr_ws.command — activa la API WebSocket de SDRconnect y lanza el headless.
# Por qué headless: si abres la GUI de SDRconnect, TOMA el dispositivo (RSPduo) y el 5454
# no queda usable. El headless levanta SOLO la API WS en :5454 sin tomar la GUI; el cliente
# (VST_LectorSDRServidor / organismo A) abre y sintoniza el dispositivo por la propia API.
# Uso:  bash audio/arrancar_sdr_ws.command [IDX_AUDIO]   (Ctrl+C para detener) · déjalo corriendo.
#   IDX_AUDIO (opcional) = índice del dispositivo de SALIDA de audio de SDRconnect. Sin él,
#   deja el actual. Para mandar el sonido de la radio al "RODECaster Pro II Chat" (y NO al Main),
#   prueba índices hasta oírlo ahí:  bash audio/arrancar_sdr_ws.command 2   (luego 3, 1, 4…).
set -e
AUDIO_IDX="${1:-}"
CFG="/Volumes/LaCie/Library/Application Support/com.sdrplay.sdrconnect/config.json"
[ -f "$CFG" ] || CFG="$(find "$HOME/Library/Application Support/com.sdrplay.sdrconnect" /Volumes/*/Library/Application\ Support/com.sdrplay.sdrconnect -maxdepth 1 -name config.json 2>/dev/null | head -1)"
HEADLESS="/Applications/SDRconnect.app/Contents/MacOS/SDRconnect_headless"

# 1) activar el WebSocket (siempre) + opcionalmente fijar el dispositivo de salida de audio
if [ -n "$CFG" ] && [ -f "$CFG" ]; then
  /usr/bin/python3 - "$CFG" "$AUDIO_IDX" <<'PY'
import json, sys
p = sys.argv[1]
aidx = sys.argv[2] if len(sys.argv) > 2 else ""
c = json.load(open(p))
st = c.setdefault("application_state", {})
st["WebSocketInterfaceEnabled"] = True
if aidx.strip():
    st["AudioDeviceIndex"] = int(aidx)
    print("[sdr-ws] AudioDeviceIndex =", aidx, "(salida de audio de la radio)")
json.dump(c, open(p, "w"), indent=4)
print("[sdr-ws] WebSocketInterfaceEnabled=true escrito en", p)
PY
else
  echo "[sdr-ws] AVISO: no encontré config.json de SDRconnect — el 5454 podría no abrir."
fi

# 2) quitar cuarentena por si acaso (idempotente)
xattr -d com.apple.quarantine "$HEADLESS" 2>/dev/null || true

# 3) lanzar el headless (deja este proceso corriendo; el 5454 queda escuchando)
echo "[sdr-ws] lanzando SDRconnect_headless → API WebSocket en ws://127.0.0.1:5454"
exec "$HEADLESS"

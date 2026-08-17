#!/bin/bash
# escuchar_radio_chat.command — hace SONAR la radio SDR por el canal "RODECaster Pro II Chat".
# ─────────────────────────────────────────────────────────────────────────────────────────
# POR QUÉ EXISTE:
#   En SDRconnect headless (5454) el dispositivo NO se abre solo: hay que abrirlo por la WS API.
#   Al abrir/habilitar el device PRIMARIO, SDRconnect reproduce él mismo el demod (WFM) hacia el
#   dispositivo de salida configurado (AudioDeviceIndex=4 = RODECaster Chat). El audio NO viaja por
#   la WS: lo toca SDRconnect a la tarjeta. Por eso basta con MANTENER el device abierto.
#   Este lanzador reusa el LectorSDRServidor (el mismo cliente del organismo A) y lo deja corriendo.
#
# REQUISITO: tener el headless lanzado antes (audio/arrancar_sdr_ws.command) y el AudioDeviceIndex
#   ya en el canal Chat (se fija eligiéndolo en la GUI antes de cerrarla; hoy quedó en 4 = Chat).
#
# USO:   bash audio/escuchar_radio_chat.command [MHz]     (Ctrl+C para parar)
#        bash audio/escuchar_radio_chat.command 94.5      (default 94.5 FM)
#   Nota: si el organismo A está corriendo, YA abre el device por su cuenta y la radio suena en
#   Chat sin este script; úsalo cuando quieras oírla SIN levantar el organismo.
set -e
VENV_PY="$HOME/.venvs/vstcosmo/bin/python"
[ -x "$VENV_PY" ] || VENV_PY="python3"
ORG="$(cd "$(dirname "$0")/../organelos" && pwd)"
MHZ="${1:-94.5}"
export ANIMA_SDRWS_URI="${ANIMA_SDRWS_URI:-ws://127.0.0.1:5454}"
exec "$VENV_PY" -u -c "
import sys, time, os
sys.path.insert(0, '$ORG')
# frecuencia: acepta MHz (94.5) o Hz crudo (94500000)
mhz = float('$MHZ')
hz = int(mhz*1e6) if mhz < 100000 else int(mhz)
os.environ['ANIMA_SDRWS_FREQ_HZ'] = str(hz)
from VST_LectorSDRServidor import LectorSDRServidor
l = LectorSDRServidor()
print('[radio-chat] abriendo RSPduo en %s @ %.3f MHz → audio al canal Chat (idx 4)' % (l.uri, hz/1e6))
if not l.arrancar():
    raise SystemExit('[radio-chat] no arrancó (¿5454 cerrado o falta websockets?)')
print('[radio-chat] device abierto; suena por el canal Chat. Ctrl+C para parar.')
try:
    while True:
        time.sleep(2.0)
        if not l.vivo:
            print('[radio-chat] (sin espectro reciente; reintentando en segundo plano...)')
except KeyboardInterrupt:
    print('\n[radio-chat] cerrando; el device se libera y el audio para.')
    l.cerrar()
"

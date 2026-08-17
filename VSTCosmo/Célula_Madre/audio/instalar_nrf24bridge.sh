#!/bin/bash
# instalar_nrf24bridge.sh — despliega el puente de radio digital de A como LaunchAgent.
# El puente lee el Arduino Uno (nRF24) y lo expone en :8772 para el organismo A (Docker).
# Se corre una copia en ~/bin porque el Desktop esta protegido por TCC y launchd no lo lee.
# Uso:  bash audio/instalar_nrf24bridge.sh   (re-ejecutar para actualizar el codigo)
set -e
AQUI="$(cd "$(dirname "$0")" && pwd)"                 # .../Célula_Madre/audio
DEST_BIN="$HOME/bin/vstcosmo_nrf24bridge.py"
PLIST="$HOME/Library/LaunchAgents/com.vstcosmo.nrf24bridge.plist"
LABEL="com.vstcosmo.nrf24bridge"

mkdir -p "$HOME/bin" "$HOME/Library/Logs/vstcosmo-nrf24bridge"
cp "$AQUI/VST_NRF24Bridge.py" "$DEST_BIN"             # despliega el puente (self-contained)
cp "$AQUI/com.vstcosmo.nrf24bridge.plist" "$PLIST"    # instala el LaunchAgent

# recargar el agente (idempotente)
launchctl unload "$PLIST" 2>/dev/null || true
pkill -f "vstcosmo_nrf24bridge.py" 2>/dev/null || true
pkill -f "VST_NRF24Bridge.py" 2>/dev/null || true
sleep 1
launchctl load "$PLIST"

echo "Puente nRF24 instalado y arrancado."
echo "  binario:  $DEST_BIN"
echo "  agente:   $PLIST  (RunAtLoad + KeepAlive)"
echo "  prueba:   curl -s http://127.0.0.1:8772/nrf"
echo "  logs:     ~/Library/Logs/vstcosmo-nrf24bridge/"

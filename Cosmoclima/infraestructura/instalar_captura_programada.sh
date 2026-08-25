#!/bin/bash
# Deja la captura de la Minuta Técnica corriendo sola, en el sistema.
#
# POR QUÉ HACE FALTA
# ------------------
# La Minuta Técnica de peligro de remoción en masa de SERNAGEOMIN se sobrescribe
# a sí misma: no guarda historia. Y se emite POR EVENTO, no por calendario — o
# sea, puede cambiar y volver atrás dentro del mismo día. Cada cambio que no
# capturamos es historia que no se recupera nunca, y sin historia no hay con qué
# validar el modelo contra lo que la fuente decía en su momento.
#
# QUÉ INSTALA
# -----------
# Un LaunchAgent de macOS (el programador nativo del sistema) que corre la foto
# liviana cuatro veces al día. Cada foto pesa ~17 KB: son unos 25 MB al año.
# Sobrevive a cerrar la terminal y a reiniciar el computador.
#
# Se eligieron minutos "raros" (13 en vez de 00) a propósito: los servicios
# públicos reciben una avalancha de peticiones en las horas en punto.
#
# PARA DESINSTALARLO
#   bash instalar_captura_programada.sh --quitar

set -euo pipefail

AQUI="$(cd "$(dirname "$0")" && pwd)"
PROYECTO="$(cd "$AQUI/.." && pwd)"
PYTHON="$PROYECTO/.venv-esa/bin/python"
ETIQUETA="cl.cosmolab.infraestructura.minuta"
PLIST="$HOME/Library/LaunchAgents/$ETIQUETA.plist"
LOG="$AQUI/datos/crudo/sernageomin/minuta_diaria/_captura.log"

if [[ "${1:-}" == "--quitar" ]]; then
    launchctl unload "$PLIST" 2>/dev/null || true
    rm -f "$PLIST"
    echo "Captura programada desinstalada. El histórico ya capturado NO se borra."
    exit 0
fi

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: no encuentro el intérprete en $PYTHON" >&2
    exit 1
fi

mkdir -p "$(dirname "$LOG")" "$HOME/Library/LaunchAgents"

cat > "$PLIST" <<PLISTFIN
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$ETIQUETA</string>

    <key>ProgramArguments</key>
    <array>
        <string>$PYTHON</string>
        <string>$AQUI/traer_capas_sernageomin.py</string>
        <string>--minuta</string>
    </array>

    <key>WorkingDirectory</key>
    <string>$PROYECTO</string>

    <!-- Cuatro veces al día. La minuta se emite por evento, así que una sola
         foto diaria puede perderse un cambio que apareció y se revirtió. -->
    <key>StartCalendarInterval</key>
    <array>
        <dict><key>Hour</key><integer>6</integer><key>Minute</key><integer>13</integer></dict>
        <dict><key>Hour</key><integer>12</integer><key>Minute</key><integer>13</integer></dict>
        <dict><key>Hour</key><integer>18</integer><key>Minute</key><integer>13</integer></dict>
        <dict><key>Hour</key><integer>23</integer><key>Minute</key><integer>13</integer></dict>
    </array>

    <!-- Si el computador estaba apagado a la hora prevista, corre al prender.
         Es preferible una foto tarde que un agujero en la serie. -->
    <key>RunAtLoad</key>
    <true/>

    <key>StandardOutPath</key>
    <string>$LOG</string>
    <key>StandardErrorPath</key>
    <string>$LOG</string>
</dict>
</plist>
PLISTFIN

launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"

echo "Captura programada instalada."
echo "  agente : $ETIQUETA"
echo "  corre  : 06:13, 12:13, 18:13 y 23:13, todos los días"
echo "  guarda : $AQUI/datos/crudo/sernageomin/minuta_diaria/"
echo "  bitácora: $LOG"
echo
echo "Para ver si está activo:  launchctl list | grep $ETIQUETA"
echo "Para desinstalar:         bash $0 --quitar"

#!/usr/bin/env bash
# Arregla SDRplay RSP1 en la Pi: quita contención USB y reinicia servicios.
# Ejecutar en la Pi: sudo bash /home/ubuntu/anima/celula_madre/docker/fix-sdr-hardware.sh
set -euo pipefail

echo "==> 1. Detener SoapyRemote (compite con sdrplay_apiService por el USB)"
systemctl stop soapyremote-server.service 2>/dev/null || true
systemctl disable soapyremote-server.service 2>/dev/null || true

echo "==> 2. Reinstalar API 3.14 si hay instalador en Descargas"
RUN="/home/ubuntu/Descargas/SDRplay_RSP_API-Linux-3.14.0.run"
if [ -x "$RUN" ] && [ ! -f /usr/local/lib/libsdrplay_api.so.3.14 ]; then
  echo "    Ejecutando $RUN (primera instalación)..."
  yes y | sh "$RUN" >/tmp/sdrplay_install.log 2>&1 || {
    echo "    AVISO: instalador devolvió error — ver /tmp/sdrplay_install.log"
    tail -20 /tmp/sdrplay_install.log || true
  }
else
  echo "    (API ya instalada — se omite reinstalación)"
fi

ldconfig 2>/dev/null || true

echo "==> 3. Reiniciar servicio SDRplay"
systemctl restart sdrplay.service
sleep 2
systemctl is-active sdrplay.service

echo "==> 4. USB"
lsusb | grep -i 1df7 || echo "AVISO: RSP1 no visible en USB"

echo "==> 5. Prueba API (GetDevices)"
export LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH:-}
python3 /home/ubuntu/anima/celula_madre/docker/test_sdrplay_api.py || true

echo ""
echo "OK — si ves 1 dispositivo RSP, el hardware está operativo."
echo "SoapyRemote quedó deshabilitado (no necesario para E; evita conflicto USB)."
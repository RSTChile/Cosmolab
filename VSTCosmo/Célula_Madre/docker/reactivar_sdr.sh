#!/bin/bash
# reactivar_sdr.sh — desatasca el RSP1 de la Pi cuando "abre pero no entrega muestras".
# ─────────────────────────────────────────────────────────────────────────────────────
# El RSP1 (clon Mirics con DIP dañado + mismatch sdrplay 3.14/3.15) se cuelga: enumera y
# abre, pero readStream da timeout (ret=-1) → espectro TODO EN CERO. Un reinicio del
# servicio NO basta; hace falta un RESET del USB. Este script hace ambos, en orden.
#
# Se corre como ROOT (vía sudo NOPASSWD) desde el botón "Reactivar radio" de la UI de E,
# o a mano:  sudo bash reactivar_sdr.sh
#
# El lector de E (LectorSDR) se auto-sana: al resetear el USB su handle muere, readStream
# lanza, y el barrido reabre el device fresco. No hace falta reiniciar el organismo.
set -u
RSP_ID="1df7:2500"          # SDRplay RSP1 (ajustar si cambia el modelo)

echo "[reactivar] $(date '+%H:%M:%S') — desatascando el RSP1 ($RSP_ID)"

# 1) reset del USB (lo que de verdad reanuda el streaming)
BUSDEV=$(lsusb 2>/dev/null | awk -v id="$RSP_ID" '$0 ~ id {gsub(/:/,"",$4); print $2, $4}')
if [ -n "$BUSDEV" ]; then
  read -r BUS DEV <<< "$BUSDEV"
  python3 - "$BUS" "$DEV" <<'PY'
import fcntl, os, sys
bus, dev = sys.argv[1], sys.argv[2]
p = "/dev/bus/usb/%s/%s" % (bus, dev)
try:
    fd = os.open(p, os.O_WRONLY)
    fcntl.ioctl(fd, (ord('U') << 8) | 20, 0)   # USBDEVFS_RESET
    os.close(fd)
    print("[reactivar] USB reset OK ->", p)
except Exception as e:
    print("[reactivar] USB reset FALLÓ:", e)
PY
else
  echo "[reactivar] AVISO: no encontré el RSP1 en lsusb (¿desconectado?)"
fi

# 2) reiniciar el servicio de la API sdrplay (re-detecta el device tras el reset)
sleep 1
systemctl restart sdrplay.service && echo "[reactivar] sdrplay.service reiniciado"
sleep 2

# 3) confirmar que enumera de nuevo
LD_LIBRARY_PATH=/usr/local/lib python3 -c "import SoapySDR; d=SoapySDR.Device.enumerate({'driver':'sdrplay'}); print('[reactivar] enumera:', len(d), 'device(s)')" 2>/dev/null || true
echo "[reactivar] listo — el lector de E reabrirá el device en unos segundos."

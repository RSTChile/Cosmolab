#!/bin/bash
# Servidor SDRconnect para el RSPduo — centrado en 94.5 MHz (FM del campo).
# Lo lanza el LaunchAgent cl.cosmolab.sdrserver al iniciar sesión.
# KeepAlive lo reinicia si cae. Logs en ~/Library/Logs/sdr_server*.log
cd /Applications/SDRconnect.app/Contents/MacOS || { echo "no existe SDRconnect.app"; exit 1; }
exec ./SDRconnect --server \
  --hwser=2007054F32 \
  --port=50000 \
  --samplerate=2000000 \
  --centerfrequency=94500000 \
  --antenna=0 \
  --ifagc=1 \
  --lnastate=0 \
  --ifgr=59

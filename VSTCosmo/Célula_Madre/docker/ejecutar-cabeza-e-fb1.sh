#!/usr/bin/env bash
set -euo pipefail

BASE="/home/ubuntu/anima/celula_madre/docker"
CONF="xorg-fb1-cabeza-e.conf"
APP="$BASE/abrir-cabeza-e-fb1.sh"
LOG="$BASE/cabeza-e-fb1-service.log"

mkdir -p "$BASE"
exec >>"$LOG" 2>&1

echo "[$(date -Is)] esperando /dev/fb1..."
for _ in $(seq 1 120); do
  [ -e /dev/fb1 ] && break
  sleep 1
done
[ -e /dev/fb1 ] || { echo "[$(date -Is)] ERROR: no existe /dev/fb1"; exit 1; }

echo "[$(date -Is)] esperando Organismo E /cabeza..."
for _ in $(seq 1 240); do
  if curl -fsS --max-time 2 http://127.0.0.1:7788/cabeza >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
curl -fsS --max-time 2 http://127.0.0.1:7788/cabeza >/dev/null

echo "[$(date -Is)] limpiando sesion fb1 previa..."
pkill -f "Xorg :1 .*xorg-fb1-cabeza-e.conf" 2>/dev/null || true
pkill -f "anima-cabeza-e-fb1" 2>/dev/null || true
sleep 1

# Xorg resuelve -config RELATIVO solo bajo /etc/X11 (ignora $BASE/xorg-*.conf).
# Por eso el anti-blank va en CLI: -s 0 -dpms (no depende de sudo ni de /etc/X11).
# La conf del repo sigue documentando ServerFlags; copiar a /etc/X11 cuando haya sudo.
cd "$BASE" || exit 1
if [ ! -f "$CONF" ]; then
  echo "[$(date -Is)] AVISO: falta $BASE/$CONF — X usará defaults de /etc/X11 si existen"
fi
echo "[$(date -Is)] lanzando cabeza E en /dev/fb1 (cwd=$BASE config=$CONF -s 0 -dpms)..."
exec xinit "./abrir-cabeza-e-fb1.sh" -- /usr/bin/Xorg :1 -config "$CONF" -s 0 -dpms -nolisten tcp -noreset -novtswitch -sharevts vt3

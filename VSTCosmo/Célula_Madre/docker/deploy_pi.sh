#!/usr/bin/env bash
# Sincroniza Célula_Madre Mac → Raspberry Pi y levanta organismo A.
# Modo: native (default, sin sudo) | docker (requiere Docker instalado en la Pi)
set -euo pipefail

MODE="${1:-native}"
PI_HOST="${PI_HOST:-rpi}"
PI_DIR="${PI_DIR:-/home/ubuntu/anima/celula_madre}"
SRC="$(cd "$(dirname "$0")/.." && pwd)"

TAR="/tmp/anima_pi_deploy.tar.gz"
echo "==> Empaquetando y enviando $SRC → $PI_HOST:$PI_DIR"
COPYFILE_DISABLE=1 tar czf "$TAR" -C "$SRC" \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='.DS_Store' \
  --exclude='*.png' --exclude='*.zip' --exclude='*.bak' \
  --exclude='experimentos' --exclude='sintesis_output' --exclude='estado_persistente' \
  --exclude='.venv-pi' --exclude='docker/history_pi' --exclude='docker/anima-a.log' .
scp -o Compression=no "$TAR" "$PI_HOST:/home/ubuntu/anima/anima_pi_deploy.tar.gz"
ssh "$PI_HOST" "mkdir -p $PI_DIR && cd $PI_DIR && tar xzf /home/ubuntu/anima/anima_pi_deploy.tar.gz"

if [ "$MODE" = "docker" ]; then
  echo "==> Build + up (Docker)"
  ssh "$PI_HOST" "mkdir -p $PI_DIR/docker/history_pi && cd $PI_DIR/docker && docker compose -f docker-compose.pi.yml up --build -d"
else
  echo "==> Arranque nativo (venv)"
  ssh "$PI_HOST" "chmod +x $PI_DIR/docker/run_native_pi.sh && $PI_DIR/docker/run_native_pi.sh start"
fi

echo "==> Esperando /estado..."
for i in $(seq 1 36); do
  if ssh "$PI_HOST" "curl -sf http://127.0.0.1:7788/estado >/dev/null 2>&1"; then
    echo "OK — organismo vivo en http://192.168.86.33:7788"
    ssh "$PI_HOST" "curl -s http://127.0.0.1:7788/estado | head -c 600"
    echo ""
    exit 0
  fi
  sleep 5
done

echo "TIMEOUT — revisa en la Pi:"
if [ "$MODE" = "docker" ]; then
  echo "  ssh $PI_HOST 'docker logs anima-a --tail 80'"
else
  echo "  ssh $PI_HOST 'tail -80 $PI_DIR/docker/anima-a.log'"
fi
exit 1
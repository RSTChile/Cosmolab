#!/bin/bash
# Díada Docker A/B/C/D — organismos locales en Mac (:7788–:7820)
cd "$(dirname "$0")"
if ! docker info >/dev/null 2>&1; then
  echo "Docker no responde. Abre Docker Desktop, espera a que arranque y vuelve a ejecutar este script."
  open -a Docker 2>/dev/null || true
  exit 1
fi
echo "Levantando anima-a anima-b anima-c anima-d…"
docker compose up -d anima-a anima-b anima-c anima-d
echo ""
docker compose ps anima-a anima-b anima-c anima-d
echo ""
for p in 7788 7799 7810 7820; do
  code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 2 "http://127.0.0.1:$p/estado" 2>/dev/null || echo "000")
  echo "  :$p/estado → HTTP $code"
done
echo ""
echo "Listo. Sociedad: http://localhost:9101/"
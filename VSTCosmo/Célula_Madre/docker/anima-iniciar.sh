#!/usr/bin/env bash
# Inicia Organismo E - Planta (ANIMA en Pi) y abre el observatorio.
# Tras reinicio: doble-clic en «Iniciar ANIMA» reanuda desde ESTADO_SESION_GROK_2026-07-02.md.
set -euo pipefail

CM="/home/ubuntu/anima/celula_madre"
URL="${ANIMA_URL:-http://127.0.0.1:7788/}"
# shellcheck source=/dev/null
[ -f "$CM/docker/.env.pi" ] && set -a && source "$CM/docker/.env.pi" && set +a
ORG_LABEL="${VST_ORGANISMO_LABEL:-Organismo E - Planta}"

notify() {
  command -v notify-send >/dev/null 2>&1 && notify-send "ANIMA — $ORG_LABEL" "$1" || true
}

bash "$CM/docker/run_native_pi.sh" start
notify "Arrancando…"

for _ in $(seq 1 45); do
  curl -sf http://127.0.0.1:7788/estado >/dev/null 2>&1 && break
  sleep 1
done

if curl -sf http://127.0.0.1:7788/estado >/dev/null 2>&1; then
  notify "Vivo en :7788 — abriendo observatorio"
else
  notify "AVISO: /estado no responde aún"
fi

export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"
[ -f "/run/user/$(id -u)/gdm/Xauthority" ] && XAUTHORITY="/run/user/$(id -u)/gdm/Xauthority"

if [ -n "${DISPLAY:-}" ] && command -v firefox >/dev/null 2>&1; then
  firefox --new-window "$URL" >/dev/null 2>&1 &
elif [ -n "${DISPLAY:-}" ] && command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$URL" >/dev/null 2>&1 &
fi
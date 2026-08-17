#!/bin/bash
# Sociedad ANIMA — observatorio dinámico (:9101). No toca el observatorio de la díada (:9100).
cd "$(dirname "$0")/.."
export ANIMA_SEED_URLS="${ANIMA_SEED_URLS:-http://127.0.0.1:7788,http://127.0.0.1:7799,http://127.0.0.1:7810,http://127.0.0.1:7820,http://192.168.86.22:7788,http://192.168.86.33:7788}"
export ANIMA_SOCIEDAD_PORT="${ANIMA_SOCIEDAD_PORT:-9101}"
export ANIMA_SOCIEDAD_CATALOGO="${ANIMA_SOCIEDAD_CATALOGO:-1}"
# Versión pública (oculta IPs; país + bandera): ANIMA_SOCIEDAD_PUBLIC=1
# Países por organismo (JSON): ANIMA_ORG_PAISES='{"ANIMA_ANIMA_PI":"CL","ANIMA_E_PI":"CL"}'
# Apariencia cabeza al instalar: ANIMA_CARA_GENERO=masculino|femenino  ANIMA_CARA_TONO=blanco|celeste|rosado|amarillo|cafe
# Por organismo (JSON): ANIMA_ORG_ASPECTO='{"ANIMA_E_PI":{"genero":"femenino","tono":"rosado"}}'
export ANIMA_SOCIEDAD_PUBLIC="${ANIMA_SOCIEDAD_PUBLIC:-1}"
if [ -x ".venv/bin/python3" ]; then PY=".venv/bin/python3"
elif [ -x "venv/bin/python3" ]; then PY="venv/bin/python3"
else PY="python3"
fi
echo "Sociedad ANIMA → http://localhost:${ANIMA_SOCIEDAD_PORT}/"
echo "Semilla: ${ANIMA_SEED_URLS}"
exec "$PY" conversacion/vst_sociedad.py
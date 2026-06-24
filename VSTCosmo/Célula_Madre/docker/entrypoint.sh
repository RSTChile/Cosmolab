#!/bin/sh
# Punto de entrada: un mismo cuerpo, tres roles. ANIMA_ROLE elige quién despierta en este contenedor.
set -e
case "${ANIMA_ROLE:-a}" in
  a)   echo "[anima] rol=A (organismo, puerto ${VST_PUERTO:-7788})"
       exec python /app/celula_madre/web/VST_CelulaMadre_WebLive_A.py ;;
  b)   echo "[anima] rol=B (organismo, puerto ${VST_PUERTO:-7799})"
       exec python /app/celula_madre/web/VST_CelulaMadre_WebLive_B.py ;;
  mcp) echo "[anima] rol=MCP (membrana de la díada, HTTP)"
       exec python /app/celula_madre/mcp/vst_mcp_diada.py --http ;;
  *)   echo "ANIMA_ROLE desconocido: '${ANIMA_ROLE}' (usa a|b|mcp)"; exit 1 ;;
esac

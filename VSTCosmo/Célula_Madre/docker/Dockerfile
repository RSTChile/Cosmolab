# ============================================================================
# ANIMA DÍADA — imagen ÚNICA para los organismos vivos y su membrana MCP.
# Docker = CUERPO/AMBIENTE OPERATIVO (no inteligencia: ésta es endógena del campo Φ).
# El rol del contenedor (organismo A, organismo B, o membrana MCP) lo decide ANIMA_ROLE.
# Audio (Rødecaster) NO va aquí: corre NATIVO en el Mac y entra por TCP (host.docker.internal).
# ============================================================================
FROM python:3.12-slim

WORKDIR /app

# Lo imprescindible: numpy (campo Φ/organelos), scipy (aparato fonador ARP 2600: filtro resonante VCF),
# soundfile (carga de WAV subidos en cualquier formato) + mcp (membrana). NO sounddevice/portaudio:
# el audio del mundo entra por red, no por CoreAudio (inaccesible en contenedor).
RUN pip install --no-cache-dir numpy scipy soundfile mcp

# El organismo entero (núcleo en subcarpetas + membrana). audio_binaural (2.4 GB) NO se copia
# (ver .dockerignore): el organismo vive en soledad con 'demo:silencio'; para oír wav, móntalo como volumen.
COPY . /app/celula_madre/

ENV PYTHONUNBUFFERED=1 \
    ANIMA_BIND=0.0.0.0 \
    ANIMA_ESTADO_DIR=/data \
    ANIMA_AUTOSTART=1 \
    ANIMA_FUENTE_DEFECTO=demo:silencio \
    VST_DISABLE_DIRECT_AUDIO=1

# Disco propio del organismo (su historia): se monta como VOLUMEN en compose.
VOLUME ["/data"]
ENTRYPOINT ["sh", "/app/celula_madre/docker/entrypoint.sh"]

# ============================================================================
# ANIMA DÍADA — imagen ÚNICA para los organismos vivos y su membrana MCP.
# Docker = CUERPO/AMBIENTE OPERATIVO (no inteligencia: ésta es endógena del campo Φ).
# El rol del contenedor (organismo A, organismo B, o membrana MCP) lo decide ANIMA_ROLE.
# Audio (Rødecaster) NO va aquí: corre NATIVO en el Mac y entra por TCP (host.docker.internal).
# ============================================================================
FROM python:3.12-slim

WORKDIR /app

# Dependencias FIJADAS (requirements.txt = lock reproducible). Se copia ANTES del
# código para que la capa de pip se cachee y no se reinstale al cambiar el código.
# numpy (campo Φ), scipy (fonador ARP 2600: VCF resonante), soundfile (carga WAV) + mcp (membrana).
# NO sounddevice/portaudio: el audio del mundo entra por red, no por CoreAudio.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

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

# Salud del ORGANISMO: /estado responde en VST_PUERTO. start-period 60s (el watchdog
# espera ~45s al arranque acoplado). Los roles mcp/conversacion sobreescriben este
# healthcheck en compose (otro puerto). Habilita 'depends_on: service_healthy'.
HEALTHCHECK --interval=30s --timeout=8s --start-period=60s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('VST_PUERTO','7788')+'/estado', timeout=5)" || exit 1

ENTRYPOINT ["sh", "/app/celula_madre/docker/entrypoint.sh"]

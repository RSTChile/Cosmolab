# ============================================================================
# COSMOGÉNESIS CG001 — imagen única para universos A/B y observatorio.
# Docker = cuerpo operativo del laboratorio (análogo a ANIMA díada).
# El rol del contenedor lo decide CG_ROLE: a | b | observatorio
# ============================================================================
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY CG001 /app/CG001
COPY docker/entrypoint.sh /app/docker/entrypoint.sh
RUN chmod +x /app/docker/entrypoint.sh

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    CG_BIND=0.0.0.0 \
    CG_AUTOSTART=1 \
    CG_HISTORY_DIR=/data \
    CG_HISTORY_ENABLE=true \
    CG_TICK_HZ=10

VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=8s --start-period=45s --retries=3 \
  CMD python -c "import os,urllib.request; p=os.environ.get('CG_PUERTO','7888'); urllib.request.urlopen('http://127.0.0.1:'+p+'/estado', timeout=5)" || exit 1

ENTRYPOINT ["sh", "/app/docker/entrypoint.sh"]
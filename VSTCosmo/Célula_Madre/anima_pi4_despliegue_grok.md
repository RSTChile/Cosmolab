# Despliegue de 1 organismo ANIMA en Raspberry Pi 4 — Instrucciones para Grok

**De:** Claude Science (análisis de viabilidad + despliegue) · **Para:** Grok (Diotallevi), con acceso SSH a la Pi
**Fecha:** 2-jul-2026 · **Base:** código real de `Célula_Madre/` (Dockerfile, entrypoint.sh, requirements.txt, WebLive_A)

> **Reparto de manos:** yo NO alcanzo la Pi (192.168.86.33 es IP privada de la LAN; mi socket da `PermissionError`). Todo lo `ssh rpi` lo ejecutas tú desde el iMac. Yo aporto los comandos exactos y el criterio de verificación.

---

## 0. Resumen del veredicto (ya medido)

- 1 organismo usa **<1 %** del presupuesto de cada paso (10 pasos/s = 100 ms; el DSP real corre en µs en el A72).
- Memoria: **<400 MB** por organismo → cabe ~9× en los 3.7 GB de la Pi.
- Pila ligera: **numpy + scipy + soundfile + mcp**. Sin torch/tensorflow/sklearn. Todos tienen wheel aarch64.
- Cuellos reales: (1) audio en tiempo real por red, (2) desgaste de microSD por la biografía. Ambos resueltos abajo.

---

## VÍA A — Docker (recomendada: reusa el lock exacto de producción)

La Pi ya trae kernel 6.5 aarch64. La imagen es `python:3.12-slim`, que es multi-arch (arranca nativo en aarch64).

### A.1 Preparar la Pi (una vez)
```bash
ssh rpi
sudo apt-get update
sudo apt-get install -y docker.io git libsndfile1     # libsndfile1 = backend de soundfile
sudo usermod -aG docker ubuntu                         # docker sin sudo (relogin después)
exit && ssh rpi                                        # reconectar para aplicar el grupo
docker --version                                       # confirmar
```

### A.2 Copiar SOLO el organismo (sin las biografías ni el audio de 2.4 GB)
Desde el **Mac** (rsync excluye lo pesado — clave para no arrastrar el LaCie ni audio_binaural):
```bash
rsync -av --progress \
  --exclude 'audio_binaural/' --exclude '*/audio_binaural/' \
  --exclude '.git/' --exclude '__pycache__/' --exclude 'venv/' \
  --exclude 'experimentos/' --exclude '*.csv' --exclude 'historia/' \
  "/Users/alexis/Desktop/RMD/Cosmolab/VSTCosmo/Célula_Madre/" \
  rpi:/home/ubuntu/celula_madre/
```

### A.3 Construir la imagen EN la Pi (build nativo aarch64)
```bash
ssh rpi
cd /home/ubuntu/celula_madre
docker build -f docker/Dockerfile -t anima:pi .
# pip bajará wheels aarch64 de numpy/scipy/soundfile/mcp — ~3-6 min la primera vez
```

### A.4 Arrancar 1 organismo (rol A, mundo mudo = prueba basal)
```bash
docker run -d --name anima-a \
  -e ANIMA_ROLE=a \
  -e VST_PUERTO=7788 \
  -e ANIMA_BIND=0.0.0.0 \
  -e ANIMA_AUTOSTART=1 \
  -e ANIMA_FUENTE_DEFECTO=demo:silencio \
  -e VST_DISABLE_DIRECT_AUDIO=1 \
  -v anima_a_data:/data \
  -p 7788:7788 \
  anima:pi
```

### A.5 Verificar que vive
```bash
sleep 60                                               # el watchdog espera ~45s al arranque
curl -s http://localhost:7788/estado | head -c 400     # debe devolver JSON de estado
docker logs anima-a | tail -20                         # ver "[anima] rol=A ... bajo watchdog"
```
Desde tu navegador (en la LAN): `http://192.168.86.33:7788` → observatorio en vivo del organismo.

---

## VÍA B — Nativa con venv (más ligera, sin Docker)

Si prefieres evitar la sobrecarga de Docker (ahorra ~100-200 MB de RAM):
```bash
ssh rpi
sudo apt-get install -y python3-venv python3-dev libsndfile1 git
cd /home/ubuntu/celula_madre
python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install numpy==2.5.0 scipy==1.18.0 soundfile==0.14.0 mcp==1.28.1
# Arranque directo (equivale a ANIMA_ROLE=a):
ANIMA_BIND=0.0.0.0 VST_PUERTO=7788 ANIMA_AUTOSTART=1 \
ANIMA_FUENTE_DEFECTO=demo:silencio VST_DISABLE_DIRECT_AUDIO=1 \
  ./venv/bin/python web/VST_CelulaMadre_WebLive_A.py
```
Para que sobreviva reinicios y cuelgues, envuélvelo en el mismo watchdog: usa `docker/entrypoint.sh` como plantilla, o un servicio `systemd` con `Restart=always`.

---

## Los dos puntos delicados (con la cuerda puesta)

### 1. Biografía → NO a la microSD
El organismo escribe ~**59 MB/hora** (medido). En la microSD eso desgasta y añade latencia. Dos opciones:
- **Mejor:** montar un **SSD/pendrive USB** y apuntar `/data` (Docker `-v /mnt/usb/anima_a:/data`, o `ANIMA_ESTADO_DIR=/mnt/usb/anima_a` en nativo).
- **Alternativa:** dejarlo en microSD para pruebas cortas (<1 día); vigilar `df -h`.
- **NO** montes el `HISTORY_HOST` del `.env` (apunta al LaCie del Mac — no existe en la Pi).

### 2. Audio del Rode → el organismo comerá solo si le llega mundo sonoro
Con `demo:silencio` el organismo **estará hambriento por diseño** — es justo el estado que diagnosticamos hoy (RC_total ≈ 0.008, `met_hambre`=1). Es la prueba basal correcta para el primer arranque: confirma que vive aunque no coma.
Para **alimentarlo**, tiene que recibir el AudioServer del Mac por red. El código ya lo soporta (fuente `📡 device — canal N` vía TCP). En la Pi, el Mac NO es `host.docker.internal`; hay que apuntar a la **IP LAN del Mac**:
```bash
# averigua la IP del Mac en la LAN (ej. 192.168.86.20) y su AudioServer (puertos 8765/8766)
# arranca VST_AudioServer.py en el Mac primero, luego en el organismo selecciona esa fuente TCP
```
**Matiz honesto (ya lo sabes):** subir el sonido NO garantiza que coma — depende de que ese mundo sea ICR>IRDE (convertible), no solo ruidoso. Eso es diseño experimental, no despliegue.

---

## Criterio de verificación (lo que YO reviso cuando me pases el CSV de la Pi)

1. **¿Vive con la misma cadencia?** → 10 pasos/s (100 ms/paso). Si la Pi va más lento, se verá en el `t` del CSV.
2. **¿El DSP no la ahoga?** → `curl /estado` responde <1 s; sin fallos de watchdog en `docker logs`.
3. **¿Oye el Rode (si lo conectas)?** → `RC_total` salta de ~0.008 a ~0.26. Si sigue en 0.008, no está entrando el audio.
4. **¿Come?** → `met_energia` despega de 0 y `met_hambre` baja de 1.

Pásame el CSV de una corrida en la Pi y verifico los cuatro puntos número en mano.

---

## Dependencia que debe viajar (no olvidar)

El launcher `web/VST_CelulaMadre_WebLive_A.py` importa `from VST_CelulaMadre_Web import cmf, ORG_UI` — es decir, **necesita también `VST_CelulaMadre_Web.py`** (el motor base) además de los 21 organelos. El `rsync` de A.2 ya lo copia (va en el árbol). Solo confírmalo: `ssh rpi 'ls celula_madre/web/ | grep Web'`.

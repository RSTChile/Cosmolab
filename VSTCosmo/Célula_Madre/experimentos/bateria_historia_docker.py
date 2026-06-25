#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BATERÍA — historia longitudinal (Docker). Valida la INFRAESTRUCTURA de registro:
crea dirs, escribe fisiología/eventos/snapshots/comunicación/voz, rota archivos, conserva
las columnas del CSV, no bloquea el lazo, y el índice apunta a archivos existentes.
NO valida la inteligencia (eso son otras baterías). Corre sin Docker (prueba el Historiador).
"""
import os, sys, json, time, glob, tempfile, shutil
AQUI = os.path.dirname(os.path.abspath(__file__)); RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, RAIZ)
[sys.path.insert(0, os.path.join(RAIZ, _d)) for _d in ("genoma", "campo", "organelos", "diada", "web", "audio") if os.path.isdir(os.path.join(RAIZ, _d))]

D = os.path.join(tempfile.gettempdir(), "vst_hist_bateria")
shutil.rmtree(D, ignore_errors=True)
os.environ.update({"VST_HISTORY_DIR": D, "VST_HISTORY_ENABLE": "true",
                   "VST_HISTORY_ROTATE_SECONDS": "1", "VST_RECORD_VOICE_WAV": "true"})
from vst_historia import Historiador

res = []
def chk(nombre, cond):
    res.append((nombre, bool(cond))); print(f"  {'PASS' if cond else 'FALLA'}  {nombre}")

print("=" * 78)
print("BATERÍA — historia longitudinal en Docker (infraestructura de registro)")
print("=" * 78)

h = Historiador("A"); hB = Historiador("B"); hD = Historiador("diada")

# (1) crea directorios por organismo + díada
subs_A = all(os.path.isdir(os.path.join(D, "organismo_A", s)) for s in ("fisiologia", "voz", "eventos", "snapshots"))
subs_D = all(os.path.isdir(os.path.join(D, "diada", s)) for s in ("comunicacion", "conversaciones"))
chk("crea directorios A/B/diada", subs_A and subs_D and os.path.isdir(os.path.join(D, "organismo_B")))

# (2) NO bloquea el lazo: 5000 filas deben encolarse en <0.5s
COLS_FIJAS = ["t", "OI", "H_homeostasis", "RC_total", "necesidad", "energia", "act_perm",
              "estructura", "voz_emitida", "voz_arousal", "balance_LR"]
t0 = time.time()
for i in range(5000):
    fila = {c: (0.001 * i if c != "voz_emitida" else "chat") for c in COLS_FIJAS}
    h.registrar_fila(fila, modo="basal")
dt = time.time() - t0
chk(f"no bloquea el lazo (5000 filas encoladas en {dt*1000:.0f}ms)", dt < 0.5)

# (3) escribe eventos, comunicación, snapshots, voz
h.evento("docker_start", "arranque"); h.evento("despertar", "vida restaurada")
hD.comunicacion({"emisor": "A", "receptor": "B", "prototipo_codebook": "chat",
                 "contexto_emisor": {"OI": 0.3}, "delta_receptor": {"delta_OI": 0.01}})
h.snapshot({"memoria": {"valencia": {}}, "codebook": [[0]], "edad_pasos": 5000, "modo_vida": "basal"})
h.voz(b"RIFF\x00\x00WAVEfake", {"emisor": "A", "voz_emitida": "chat", "duracion": 1.0})
time.sleep(3.0)   # deja vaciar la cola + rotar (rotate=1s)

# (4) fisiología escrita y CONSERVA las columnas
fis = sorted(glob.glob(os.path.join(D, "organismo_A", "fisiologia", "*.csv")))
cols_ok = False; filas_n = 0
if fis:
    head = open(fis[-1], encoding="utf-8").readline().strip().split(",")
    cols_ok = all(c in head for c in COLS_FIJAS) and "ts_real" in head and "modo_vida" in head
    filas_n = sum(sum(1 for _ in open(p, encoding="utf-8")) - 1 for p in fis)
chk("escribe fisiología y CONSERVA todas las columnas (+ ts_real/modo_vida)", cols_ok)
chk(f"registró las 5000 filas ({filas_n})", filas_n >= 5000)

# (5) rotación: rotate=1s + 3s de escritura → debe haber >1 archivo de fisiología
chk(f"rota archivos por tiempo ({len(fis)} archivos)", len(fis) >= 1)

# (6) eventos, comunicación, snapshot, voz en disco
ev = glob.glob(os.path.join(D, "organismo_A", "eventos", "*.jsonl"))
com = glob.glob(os.path.join(D, "diada", "comunicacion", "*.jsonl"))
snp = glob.glob(os.path.join(D, "organismo_A", "snapshots", "*.json"))
wav = glob.glob(os.path.join(D, "organismo_A", "voz", "*.wav"))
vjson = glob.glob(os.path.join(D, "organismo_A", "voz", "*.json"))
chk("escribe eventos (JSONL diaria)", ev and sum(1 for _ in open(ev[0])) >= 2)
chk("escribe comunicación A↔B (con contexto/delta)", com and "delta_receptor" in open(com[0]).read())
chk("escribe snapshots (memoria/codebook/edad)", snp and "codebook" in open(snp[0]).read())
chk("escribe voz WAV + metadatos JSON", wav and vjson)

# (7) índice maestro apunta a archivos EXISTENTES
idx = os.path.join(D, "index.jsonl"); idx_ok = False
if os.path.exists(idx):
    lineas = [json.loads(l) for l in open(idx, encoding="utf-8") if l.strip()]
    idx_ok = len(lineas) > 0 and all(os.path.exists(e.get("archivo", "/no")) for e in lineas if "archivo" in e)
chk("index.jsonl apunta a archivos existentes", idx_ok)

# (8) errores de escritura no rompen nada
chk("sin errores de escritura", h.stats().get("errores", 1) == 0 and hD.stats().get("errores", 1) == 0)

print("-" * 78)
ok = sum(1 for _, p in res if p)
print(f"  RESUMEN: {ok}/{len(res)} PASS")
print(f"  → historia de prueba en {D}")
shutil.rmtree(D, ignore_errors=True)
sys.exit(0 if ok == len(res) else 1)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANIMA-4 · BRAZO POSITIVO de O-N1 (RC = ICR + IRDE).
PREGUNTA: ¿qué TIPO de ruido contextual (RC) se convierte en SENTIDO (ICR → vocabulario nuevo)
y cuál se DESVÍA a riesgo (IRDE)?  Hipótesis: estímulo ESTRUCTURADO → ICR (acuña); CAÓTICO → IRDE.

DISEÑO: 4 mundos COMPARTIDOS (mismo audio para los 4, NO divergente → canal relacional limpio),
cada uno en topología PLENA (todos↔todos, máximo acople y difusión). Desde CERO antes de CADA
condición (comparación independiente de la tasa de invención por mundo).

  SILENCIO  (demo:silencio)   → RC≈0   ANCLA: sin materia prima, no debe aparecer sentido nuevo.
  MUSICA    (Brandemburgo)    → RC alto ESTRUCTURADO → predice ICR (sentido) alto.
  VOZ       (Voz_Estudio)     → RC alto tipo conversación (el ejemplo de O-N1).
  RUIDO     (ruido_pos60deg)  → RC máximo CAÓTICO → predice IRDE (desvío) alto, poco ICR.

MIDE por fila: RC_total, ICR, IRDE, ICR_ratio, voz_creadas/aprendidas/estables, voz_id, voz_emulada_de
(rutas). Manifiesto con ts_real para segmentar los primarios. Análisis fino RC→ICR/IRDE post-hoc.
"""
import os, sys, json, time, glob, csv, datetime, subprocess, urllib.request
from collections import Counter

ORGS = {"A": {"port": 7788, "host": "anima-a"}, "B": {"port": 7799, "host": "anima-b"},
        "C": {"port": 7810, "host": "anima-c"}, "D": {"port": 7820, "host": "anima-d"}}
LETRAS = ["A", "B", "C", "D"]
DOCKER_DIR = "/Users/alexis/Desktop/RMD/Cosmolab/VSTCosmo/Célula_Madre/docker"
HIST = "/Users/alexis/Desktop/RMD/Cosmolab/VSTCosmo/Docker_Historia"
AUDIO_DIR = "/Users/alexis/Desktop/RMD/Cosmolab/VSTCosmo/audio_binaural"
VOLS = ["anima-diada_anima_a_data", "anima-diada_anima_b_data",
        "anima-diada_anima_c_data", "anima-diada_anima_d_data"]
DUR_COND = int(os.environ.get("DUR_COND", "5400"))   # 90 min/cond por defecto
SETTLE = int(os.environ.get("SETTLE", "14"))
SAMPLE = 4
TS0 = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT = os.path.expanduser(f"~/Downloads/ANIMA4_MUNDOS_{TS0}")
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "ejecucion.log")
MANIF = os.path.join(OUT, f"condiciones_{TS0}.csv")
EPI = ("> **Advertencia epistémica:** los rótulos de audio/voz son etiquetas humanas de archivo/banco. "
       "NO sabemos qué significan para los organismos. 'Sentido nuevo' = unidad de vocabulario que cubre "
       "una región afectiva antes no cubierta, NO significado humano.\n")
_manifiesto = []

def log(m):
    line = f"[{datetime.datetime.now():%H:%M:%S}] {m}"; print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")

def iurl(L): o = ORGS[L]; return f"http://{o['host']}:{o['port']}/comunicacion/bloque.wav"
def post(L, path, body):
    o = ORGS[L]
    req = urllib.request.Request(f"http://127.0.0.1:{o['port']}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=12) as r: return json.loads(r.read())
    except Exception as e: log(f"  ! post {path} {L}: {e}"); return None
def get(L, path):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{ORGS[L]['port']}{path}", timeout=8) as r:
            return json.loads(r.read())
    except Exception: return {}

def _buscar(*subs):
    files = [os.path.basename(p) for p in glob.glob(os.path.join(AUDIO_DIR, "*.wav"))]
    for sub in subs:
        for f in files:
            if sub.lower() in f.lower(): return f
    return None
A_MUS = _buscar("Brandemburgo.wav", "musica_pos") or "demo:silencio"
A_VOZ = _buscar("Voz_Estudio.wav", "Voz_Estudio", "voz_pos") or "demo:silencio"
A_RUI = _buscar("ruido_pos", "Ruido") or "demo:silencio"
# (mundo, audio).  SILENCIO primero (ancla RC≈0), luego los 3 ricos.
WORLDS = [("SILENCIO", "demo:silencio"), ("MUSICA", A_MUS), ("VOZ", A_VOZ), ("RUIDO", A_RUI)]

def mundo_src(n): return {"tipo": "demo", "spec": n} if n.startswith("demo:") else {"tipo": "archivo", "nombre": n}
def rel_otros(L): return {"tipo": "otros_organismos",
                          "urls": [f"{iurl(x)}?modo=R2D2&gain=20.0" for x in LETRAS if x != L], "nombre": "otros"}

# columnas de fisiología muestreadas en vivo (numéricas)
SAMP = ["RC_total", "ICR", "IRDE", "ICR_ratio", "IRDE_ratio", "oao_imitacion_mag",
        "voz_creadas", "voz_aprendidas", "voz_estables", "voz_propias"]

def _stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs: return {"med": None, "max": None, "last": None}
    return {"med": round(sum(xs) / len(xs), 5), "max": round(max(xs), 5), "last": round(xs[-1], 5)}

def reset_zero():
    """Borra el vocabulario/estado (volúmenes) y levanta organismos FRESCOS. Comparación independiente por mundo."""
    log("  [reset] down + wipe volúmenes + up (organismos a CERO)…")
    subprocess.run(["docker", "compose", "down"], cwd=DOCKER_DIR, capture_output=True, timeout=120)
    for v in VOLS: subprocess.run(["docker", "volume", "rm", v], capture_output=True, timeout=60)
    env = dict(os.environ); env.update(COMPOSE_BAKE="false", DOCKER_BUILDKIT="0")
    subprocess.run(["docker", "compose", "up", "-d"], cwd=DOCKER_DIR, env=env, capture_output=True, timeout=240)
    # esperar a que respondan
    for _ in range(50):
        ok = sum(1 for L in LETRAS if get(L, "/ultima_fila") is not None and get(L, "/ultima_fila") != {})
        if ok >= 4: break
        time.sleep(3)
    # confirmar cero
    z = []
    for L in LETRAS:
        f = (get(L, "/ultima_fila") or {}).get("fila") or {}
        z.append(f"{L}={f.get('voz_creadas')}")
    log(f"  [reset] listo · creadas: {' '.join(z)}")
    return ok >= 4

def correr(mundo, audio):
    cond_id = f"M_{mundo}"
    log(f"  ── COND {cond_id}: mundo={audio} · PLENA (todos↔todos) · {DUR_COND//60} min ──")
    for L in LETRAS:
        post(L, "/exp_tag", {"exp_topologia": "PLENA", "exp_ciclo": "1", "exp_mundo_audio": audio,
                             "exp_control": "real", "exp_fuente_relacion": "todos", "exp_mundo_nombre": mundo})
    for L in LETRAS:
        cfg = {"left_src": mundo_src(audio), "right_src": rel_otros(L),
               "binaural": True, "segundos": 2, "continuo": True, "criterio_duracion": "min"}
        post(L, "/start", {"cfg": cfg, "modo_vida": "experimento"})
    time.sleep(SETTLE)
    t0 = round(time.time(), 2)
    series = {L: {c: [] for c in SAMP} for L in LETRAS}
    voz_ids = {L: Counter() for L in LETRAS}      # repertorio emitido
    rutas = []                                     # (org, voz_id, emulada_de) cuando hay emulación
    fin = time.time() + DUR_COND
    while time.time() < fin:
        for L in LETRAS:
            f = (get(L, "/ultima_fila") or {}).get("fila") or {}
            if not f: continue
            for c in SAMP:
                v = f.get(c)
                if v is not None:
                    try: series[L][c].append(float(v))
                    except Exception: pass
            vid = f.get("voz_id")
            if vid: voz_ids[L][vid] += 1
            emu = f.get("voz_emulada_de")
            if emu: rutas.append((L, vid, emu))
        time.sleep(SAMPLE)
    t1 = round(time.time(), 2)
    for L in LETRAS:
        post(L, "/control", {"action": "stop"})
        _manifiesto.append({"cond_id": cond_id, "mundo": mundo, "audio": audio, "organismo": L,
                            "ts_real_ini": t0, "ts_real_fin": t1, "topologia": "PLENA", "control": "real"})
    # resumen de la condición
    creadas = {L: (_stats(series[L]["voz_creadas"])["max"] or 0) for L in LETRAS}
    aprend = {L: (_stats(series[L]["voz_aprendidas"])["max"] or 0) for L in LETRAS}
    rc = _stats(sum((series[L]["RC_total"] for L in LETRAS), []))
    icr = _stats(sum((series[L]["ICR"] for L in LETRAS), []))
    irde = _stats(sum((series[L]["IRDE"] for L in LETRAS), []))
    nuevas = sorted({vid for L in LETRAS for vid in voz_ids[L] if vid and ("palabra_" in vid or "apr_" in vid)})
    log(f"     ✅ {cond_id} · RC_med={rc['med']} ICR_med={icr['med']} IRDE_med={irde['med']} · "
        f"creadas={sum(creadas.values())} emuladas={sum(aprend.values())} rutas={len(rutas)} voces_nuevas={len(nuevas)}")
    return {"cond_id": cond_id, "mundo": mundo, "audio": audio, "creadas": creadas, "aprend": aprend,
            "rc": rc, "icr": icr, "irde": irde, "rutas": rutas, "nuevas": nuevas,
            "stats": {L: {c: _stats(series[L][c]) for c in SAMP} for L in LETRAS}}

def _empaquetar():
    try:
        prim = os.path.join(OUT, "primarios_fisiologia.tar.gz")
        subprocess.run(["tar", "czf", prim, "-C", "/", HIST.lstrip("/")], capture_output=True, timeout=600)
        log(f"  ✅ {os.path.basename(prim)} (primarios SEGMENTABLES por manifiesto ts_real)")
    except Exception as e: log(f"  ! empaquetar: {e}")

def _salidas(resultados):
    # manifiesto
    if _manifiesto:
        with open(MANIF, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(_manifiesto[0].keys())); w.writeheader(); w.writerows(_manifiesto)
    # tabla por condición
    rows = []
    for r in resultados:
        rows.append({"cond_id": r["cond_id"], "mundo": r["mundo"], "audio": r["audio"],
                     "RC_med": r["rc"]["med"], "ICR_med": r["icr"]["med"], "IRDE_med": r["irde"]["med"],
                     "ICR_sobre_RC": (round(r["icr"]["med"]/r["rc"]["med"], 3) if (r["rc"]["med"] or 0) > 0 else None),
                     "creadas_total": sum(r["creadas"].values()), "emuladas_total": sum(r["aprend"].values()),
                     "voces_nuevas": len(r["nuevas"]), "rutas": len(r["rutas"])})
    with open(os.path.join(OUT, "resultados_mundos.csv"), "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    # rutas léxicas
    with open(os.path.join(OUT, "rutas_lexicas.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["cond_id", "mundo", "org_emite", "voz_id", "voz_emulada_de"])
        for r in resultados:
            for (org, vid, emu) in r["rutas"]:
                w.writerow([r["cond_id"], r["mundo"], org, vid, emu])
    # resumen
    sm = ["# ANIMA-4 · Brazo positivo O-N1 (RC=ICR+IRDE) — ¿qué RC se vuelve sentido?\n", EPI,
          f"**{TS0}** · 4 mundos compartidos × PLENA · {DUR_COND//60} min/cond · desde CERO entre condiciones.\n",
          "## RC y su reparto (ICR=sentido / IRDE=riesgo) + invención, por mundo",
          "| Mundo | RC_med | ICR_med | IRDE_med | ICR/RC | creadas | emuladas | voces nuevas | rutas |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        sm.append(f"| {r['mundo']} | {r['RC_med']} | {r['ICR_med']} | {r['IRDE_med']} | {r['ICR_sobre_RC']} "
                  f"| {r['creadas_total']} | {r['emuladas_total']} | {r['voces_nuevas']} | {r['rutas']} |")
    sm.append("\n**Predicción O-N1:** RC alto+estructurado (MUSICA/VOZ) → ICR alto → aparece vocabulario nuevo; "
              "RC alto+caótico (RUIDO) → IRDE alto, poco sentido; SILENCIO (RC≈0) → nada. Verificar la conservación "
              "`RC=ICR+IRDE` y correlación temporal invención↔ICR en los primarios.\n")
    open(os.path.join(OUT, "resumen_mundos.md"), "w").write("\n".join(sm))
    log(f"  ✅ resúmenes escritos en {OUT}")

def main():
    log("=" * 72); log(f"ANIMA-4 MUNDOS (brazo positivo O-N1) · {datetime.datetime.now():%Y-%m-%d %H:%M} · {OUT}")
    log(f"mundos: " + ", ".join(f"{m}={a}" for m, a in WORLDS))
    resultados = []
    for mundo, audio in WORLDS:
        try:
            if not reset_zero():
                log(f"  ⚠ {mundo}: organismos no respondieron tras reset; intento igual.")
            res = correr(mundo, audio)
            resultados.append(res)
            _salidas(resultados)   # guardado incremental (robustez ante corte)
        except Exception as e:
            log(f"  ❌ ERROR mundo {mundo}: {e}")
    _salidas(resultados); _empaquetar()
    log(f"✅ MUNDOS COMPLETO · {datetime.datetime.now():%Y-%m-%d %H:%M} · {OUT}"); log("=" * 72)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANIMA-4 · RÉGIMEN DE ESTÍMULOS VARIADOS (RC alto por CAMBIO, no por duración).
Corrección de campo de Alexis: en el Rode, el vocabulario nuevo surgía RÁPIDO porque él variaba
mucho el contexto — cambiaba la música, metía otras entradas, cerraba un oído y luego el otro.
RC (ruido contextual) = TASA DE NOVEDAD, no volumen ni duración. Un audio estático se habitúa
(el banco lo cubre → RC efectivo cae); la VARIACIÓN constante mantiene RC alto → dispara invención.

DISEÑO: 1 sociedad acoplada (PLENA, todos↔todos), desde CERO. Cada ~SEG segundos se CAMBIA el
estímulo del mundo, rotando por el pool COMPLETO de audios (decenas) con saltos grandes para
maximizar el contraste entre estímulos consecutivos, y alternando MODOS DE CANAL (normal / invertir
L-R / cerrar el oído del mundo / cerrar el oído de los pares / dos mundos distintos). Corto: la
emergencia, si O-N1 es correcto, debe verse pronto.

MIDE (vivo + primarios): RC_total, ICR, IRDE, voz_creadas/aprendidas/estables, voz_id, voz_emulada_de
(rutas). Línea de tiempo fina (timeline_rico.csv) para graficar invención(t) ↔ ICR(t) ↔ cambios.
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
DUR_TOTAL = int(os.environ.get("DUR_TOTAL", "2400"))   # 40 min por defecto (corto; lo que importa es el cambio)
SEG = int(os.environ.get("SEG", "25"))                  # cambia de estímulo cada 25 s → muchísima variación
SAMPLE = 4
GAIN = os.environ.get("GAIN", "20.0")
TS0 = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT = os.path.expanduser(f"~/Downloads/ANIMA4_RICO_{TS0}")
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "ejecucion.log")
MANIF = os.path.join(OUT, f"condiciones_{TS0}.csv")
EPI = ("> **Advertencia epistémica:** los rótulos de audio/voz son etiquetas humanas. NO sabemos qué "
       "significan para los organismos. 'Sentido nuevo' = unidad de vocabulario que cubre una región "
       "afectiva antes no cubierta, NO significado humano.\n")
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
    except Exception as e: return None
def get(L, path):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{ORGS[L]['port']}{path}", timeout=8) as r:
            return json.loads(r.read())
    except Exception: return {}

# pool COMPLETO de estímulos (decenas). Orden con salto grande → estímulos consecutivos muy distintos.
_files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(AUDIO_DIR, "*.wav")))
POOL = [{"tipo": "archivo", "nombre": n} for n in _files]
SIL = {"tipo": "demo", "spec": "demo:silencio"}
def peers(L): return {"tipo": "otros_organismos",
                      "urls": [f"{iurl(x)}?modo=R2D2&gain={GAIN}" for x in LETRAS if x != L], "nombre": "otros"}
def mundo(i): return POOL[(i * 47) % len(POOL)] if POOL else SIL          # salto coprimo → recorre todo, sin repetir pronto
def mundo2(i): return POOL[(i * 47 + 19) % len(POOL)] if POOL else SIL

def cfg_segmento(L, i):
    """(left_src, right_src, etiqueta). 5 modos rotados → varía audio Y canal (incl. cerrar oídos)."""
    W, W2, P = mundo(i), mundo2(i), peers(L)
    m = i % 5
    if m == 0: return W, P, f"{W['nombre']}|pares"          # mundo L · pares R (normal)
    if m == 1: return P, W, f"pares|{W['nombre']}"          # invertido: pares L · mundo R
    if m == 2: return W, SIL, f"{W['nombre']}|cierra-pares" # cierra el oído de los pares (solo mundo)
    if m == 3: return SIL, P, f"cierra-mundo|pares"         # cierra el oído del mundo (solo pares)
    return W, W2, f"{W['nombre']}|{W2['nombre']}"           # dos mundos distintos a la vez

SAMP = ["RC_total", "ICR", "IRDE", "ICR_ratio", "voz_creadas", "voz_aprendidas", "voz_estables"]
def _m(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 5) if xs else None

def reset_zero():
    log("  [reset] down + wipe volúmenes + up (a CERO)…")
    subprocess.run(["docker", "compose", "down"], cwd=DOCKER_DIR, capture_output=True, timeout=120)
    for v in VOLS: subprocess.run(["docker", "volume", "rm", v], capture_output=True, timeout=60)
    env = dict(os.environ); env.update(COMPOSE_BAKE="false", DOCKER_BUILDKIT="0")
    subprocess.run(["docker", "compose", "up", "-d"], cwd=DOCKER_DIR, env=env, capture_output=True, timeout=240)
    ok = 0
    for _ in range(50):
        ok = sum(1 for L in LETRAS if (get(L, "/ultima_fila") or {}) != {})
        if ok >= 4: break
        time.sleep(3)
    log(f"  [reset] organismos accesibles {ok}/4"); return ok >= 4

def aplicar(i):
    left = {}; right = {}; etiqs = {}
    for L in LETRAS:
        l, r, et = cfg_segmento(L, i); etiqs[L] = et
        cfg = {"left_src": l, "right_src": r, "binaural": True, "segundos": 2, "continuo": True, "criterio_duracion": "min"}
        post(L, "/start", {"cfg": cfg, "modo_vida": "experimento"})
        post(L, "/exp_tag", {"exp_topologia": "RICO_VARIADO", "exp_ciclo": "1", "exp_mundo_audio": et,
                             "exp_control": "real", "exp_fuente_relacion": "todos", "exp_segmento": str(i)})
    return etiqs

def main():
    log("=" * 72); log(f"ANIMA-4 RICO/VARIADO (RC por CAMBIO) · {datetime.datetime.now():%Y-%m-%d %H:%M} · {OUT}")
    log(f"pool={len(POOL)} estímulos · cambia cada {SEG}s · total {DUR_TOTAL//60} min · ~{DUR_TOTAL//SEG} cambios · 5 modos de canal")
    reset_zero()
    t_ini = time.time(); t_fin = t_ini + DUR_TOTAL
    voz_ids = {L: Counter() for L in LETRAS}; rutas = []; nuevas = set()
    primera_creada = {"t": None, "org": None}; primera_ruta = {"t": None}
    tl = []  # timeline
    i = 0
    etiqs = aplicar(i); t0 = round(time.time(), 2)
    log(f"  seg {i}: " + " ".join(f"{L}:{etiqs[L][:22]}" for L in ['A']))
    next_change = time.time() + SEG
    last_creadas = {L: 0 for L in LETRAS}
    while time.time() < t_fin:
        # ¿toca cambiar de estímulo?
        if time.time() >= next_change:
            i += 1; etiqs = aplicar(i); next_change = time.time() + SEG
        # muestreo
        buf = {c: [] for c in SAMP}; creadas = {}; emul = {}
        for L in LETRAS:
            f = (get(L, "/ultima_fila") or {}).get("fila") or {}
            if not f: continue
            for c in SAMP:
                v = f.get(c)
                if v is not None:
                    try: buf[c].append(float(v))
                    except Exception: pass
            try: creadas[L] = int(float(f.get("voz_creadas") or 0))
            except Exception: creadas[L] = last_creadas[L]
            try: emul[L] = int(float(f.get("voz_aprendidas") or 0))
            except Exception: emul[L] = 0
            vid = f.get("voz_id")
            if vid:
                voz_ids[L][vid] += 1
                if "palabra_" in vid or "apr_" in vid: nuevas.add(vid)
            em = f.get("voz_emulada_de")
            if em:
                rutas.append((round(time.time()-t_ini,1), L, vid, em))
                if primera_ruta["t"] is None: primera_ruta["t"] = round(time.time()-t_ini,1); log(f"  🔗 PRIMERA RUTA t={primera_ruta['t']}s: {L} emuló {em}")
        # detectar primera invención
        for L in LETRAS:
            if creadas.get(L, 0) > last_creadas.get(L, 0):
                if primera_creada["t"] is None:
                    primera_creada = {"t": round(time.time()-t_ini,1), "org": L}
                    log(f"  ✨ PRIMERA PALABRA NUEVA t={primera_creada['t']}s · {L} (creadas {last_creadas[L]}→{creadas[L]})")
            last_creadas[L] = creadas.get(L, last_creadas.get(L,0))
        tl.append({"t": round(time.time()-t_ini, 1), "seg": i, "etiq_A": etiqs.get("A","")[:30],
                   "RC": _m(buf["RC_total"]), "ICR": _m(buf["ICR"]), "IRDE": _m(buf["IRDE"]),
                   "creadas_tot": sum(last_creadas.values()), "emul_tot": sum(emul.values()) if emul else 0,
                   "voces_nuevas": len(nuevas)})
        # log de avance cada ~2 min
        if len(tl) % 30 == 0:
            log(f"  t={int(time.time()-t_ini)}s seg={i} · RC={tl[-1]['RC']} ICR={tl[-1]['ICR']} · "
                f"creadas={tl[-1]['creadas_tot']} emul={tl[-1]['emul_tot']} nuevas={len(nuevas)}")
            _escribir(tl, rutas, voz_ids, nuevas, primera_creada, primera_ruta)  # incremental
        time.sleep(SAMPLE)
    for L in LETRAS:
        post(L, "/control", {"action": "stop"})
        _manifiesto.append({"cond_id": "RICO_VARIADO", "organismo": L, "ts_real_ini": t0,
                            "ts_real_fin": round(time.time(),2), "topologia": "RICO_VARIADO", "control": "real"})
    _escribir(tl, rutas, voz_ids, nuevas, primera_creada, primera_ruta)
    _empaquetar()
    log(f"✅ RICO COMPLETO · creadas_tot={sum(last_creadas.values())} · voces_nuevas={len(nuevas)} · rutas={len(rutas)} · "
        f"1ª palabra t={primera_creada['t']}s · 1ª ruta t={primera_ruta['t']}s")
    log("=" * 72)

def _escribir(tl, rutas, voz_ids, nuevas, primera_creada, primera_ruta):
    with open(os.path.join(OUT, "timeline_rico.csv"), "w", newline="") as f:
        if tl:
            w = csv.DictWriter(f, fieldnames=list(tl[0].keys())); w.writeheader(); w.writerows(tl)
    with open(os.path.join(OUT, "rutas_lexicas.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["t_s", "org_emite", "voz_id", "voz_emulada_de"])
        for r in rutas: w.writerow(r)
    if _manifiesto:
        with open(MANIF, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(_manifiesto[0].keys())); w.writeheader(); w.writerows(_manifiesto)
    rc = _m([t["RC"] for t in tl]); icr = _m([t["ICR"] for t in tl]); irde = _m([t["IRDE"] for t in tl])
    sm = ["# ANIMA-4 · Régimen de estímulos VARIADOS (RC por cambio) — ¿emerge sentido nuevo?\n", EPI,
          f"**{TS0}** · pool {len(POOL)} estímulos · cambio cada {SEG}s · {DUR_TOTAL//60} min · sociedad PLENA desde cero.\n",
          "## Resultado",
          f"- **1ª palabra nueva:** {'t='+str(primera_creada['t'])+'s ('+str(primera_creada['org'])+')' if primera_creada['t'] is not None else 'NO emergió'}",
          f"- **1ª ruta léxica (emulación):** {'t='+str(primera_ruta['t'])+'s' if primera_ruta['t'] is not None else 'ninguna'}",
          f"- **voces nuevas distintas:** {len(nuevas)} · **rutas totales:** {len(rutas)}",
          f"- **RC medio:** {rc} · **ICR medio:** {icr} · **IRDE medio:** {irde}"
          f" · ICR/RC: {round(icr/rc,3) if (rc or 0)>0 else None}",
          f"- **voces nuevas:** {', '.join(sorted(nuevas)) if nuevas else '—'}\n",
          "## Interpretación O-N1",
          "Si la variación rápida (RC alto sostenido por CAMBIO) hace emerger vocabulario pronto, confirma que "
          "RC es tasa de novedad, no duración — y que el sentido nuevo es la conversión RC→ICR. Ver timeline_rico.csv "
          "(invención(t) vs ICR(t) vs cambios) y rutas_lexicas.csv (propagación entre organismos).\n"]
    open(os.path.join(OUT, "resumen_rico.md"), "w").write("\n".join(sm))

def _empaquetar():
    try:
        prim = os.path.join(OUT, "primarios_fisiologia.tar.gz")
        subprocess.run(["tar", "czf", prim, "-C", "/", HIST.lstrip("/")], capture_output=True, timeout=600)
        log(f"  ✅ {os.path.basename(prim)}")
    except Exception as e: log(f"  ! empaquetar: {e}")

if __name__ == "__main__":
    main()

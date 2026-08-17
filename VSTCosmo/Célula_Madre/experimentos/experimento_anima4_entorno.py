#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANIMA-4 · ENTORNO COMPLEJO Y DINÁMICO (método de campo de Alexis, completo).
Máxima variación en TODAS las dimensiones, para hacer emerger sentido (vocabulario) y poder
discriminar atención-mutua vs seguir-el-contexto.

VARIACIÓN EN 3 EJES SIMULTÁNEOS:
  (A) AUDIO: rota mezclando TODAS las categorías (notas, frecuencias, tonos, música, voz, ruido,
      viento, texturas), nunca dos categorías iguales seguidas. Dwell variable: cortos ~10s; largos
      hasta 60s ('1 minuto y cambio'). Variantes espaciales (±60°) entran en la mezcla.
  (B) RUTEO (oído por el que entra par/mundo), secuencia CÍCLICA de regímenes:
        R1 PARES      : PAREJAS A↔B C↔D, A y C INVERTIDOS (par por oído R), B,D normales. Mundo COMÚN.
        R2 FLIP       : inversión GLOBAL de los 4 (anti-habituación + prueba causal). Mundo COMÚN.
        R3 UNO_X_UNO  : invierte un organismo a la vez, 5 temas cada uno, rota A→B→C→D. Mundo COMÚN.
        R4 AZAR       : ruteo aleatorio por organismo + CADA UNO SU PROPIO sonido (máxima variación).
        → vuelve a R1 y cicla.
  (C) MUNDO: COMÚN en R1-R3 (para que el test contexto-vs-atención sea limpio) y PROPIO por organismo
      en R4 (máxima divergencia).

DISCRIMINACIÓN (crítica adversarial válida): con A,C invertidos, si las cabezas SIGUEN EL CONTEXTO,
A,C se orientan al revés que B,D (el mundo entra por oídos opuestos); si hay ATENCIÓN MUTUA, A mira a
B y C mira a D (convergencia por pareja). Cada transición de régimen y cada FLIP es una prueba causal.

MIDE: act_orientacion_deg de los 4 (correlaciones por pareja, por régimen) + RC/ICR/IRDE + vocabulario
+ rutas léxicas (voz_id/voz_emulada_de). Desde CERO.
"""
import os, sys, json, time, glob, csv, datetime, subprocess, urllib.request, math, wave, random
from collections import Counter, defaultdict

ORGS = {"A": {"port": 7788, "host": "anima-a"}, "B": {"port": 7799, "host": "anima-b"},
        "C": {"port": 7810, "host": "anima-c"}, "D": {"port": 7820, "host": "anima-d"}}
LETRAS = ["A", "B", "C", "D"]
PAR = {"A": "B", "B": "A", "C": "D", "D": "C"}
DOCKER_DIR = "/Users/alexis/Desktop/RMD/Cosmolab/VSTCosmo/Célula_Madre/docker"
HIST = "/Users/alexis/Desktop/RMD/Cosmolab/VSTCosmo/Docker_Historia"
AUDIO_DIR = "/Users/alexis/Desktop/RMD/Cosmolab/VSTCosmo/audio_binaural"
VOLS = ["anima-diada_anima_a_data", "anima-diada_anima_b_data",
        "anima-diada_anima_c_data", "anima-diada_anima_d_data"]
DUR_TOTAL = int(os.environ.get("DUR_TOTAL", "3600"))     # 60 min (≈3 ciclos)
GAIN = os.environ.get("GAIN", "20.0")
SEED = int(os.environ.get("SEED", "42"))
RNG = random.Random(SEED)
SAMPLE = 4
# conteo de temas por régimen (confirmado por Alexis)
N_R1, N_R2, N_R3_C, N_R4 = 10, 10, 5, 10                 # R3: 5 temas por organismo (×4)
TS0 = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT = os.path.expanduser(f"~/Downloads/ANIMA4_ENTORNO_{TS0}")
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "ejecucion.log")
MANIF = os.path.join(OUT, f"condiciones_{TS0}.csv")
EPI = ("> **Advertencia epistémica:** los rótulos de audio/voz son etiquetas humanas. NO sabemos qué "
       "significan para los organismos.\n")
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
    except Exception: return None
def get(L, path):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{ORGS[L]['port']}{path}", timeout=8) as r:
            return json.loads(r.read())
    except Exception: return {}

# ---------- pool categorizado ----------
def _dur(p):
    try:
        with wave.open(p) as w: return w.getnframes() / float(w.getframerate())
    except Exception: return 35.0
def _cat(n):
    l = n.lower()
    import re
    if re.search(r'freq_\d', l): return "frecuencia"
    if re.search(r'\b(do|re|mi|fa|sol|la|si)[_\.]', l) or 'escala' in l or 'piano' in l: return "nota"
    if 'tono' in l: return "tono"
    if 'brandemburgo' in l or 'blue_monday' in l or 'bigbang' in l: return "musica"
    if 'voz' in l and 'viento' in l: return "vozviento"
    if 'voz' in l: return "voz"
    if 'viento' in l: return "viento"
    if 'ruido' in l: return "ruido"
    return "textura"
_files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(AUDIO_DIR, "*.wav")))
CAT = {}; DUR = {}
for n in _files:
    CAT[n] = _cat(n); DUR[n] = _dur(os.path.join(AUDIO_DIR, n))
PORCAT = defaultdict(list)
for n in _files: PORCAT[CAT[n]].append(n)
CATS = sorted(PORCAT)
def src(n): return {"tipo": "archivo", "nombre": n}
def dwell(n): return max(10.0, min(60.0, DUR.get(n, 35.0)))   # cortos ~10s, largos hasta 60s

_cat_i = 0; _en_cat = {c: 0 for c in CATS}
def siguiente_audio():
    """Rota por categorías (round-robin) para mezclar tipos y nunca repetir categoría seguida."""
    global _cat_i
    c = CATS[_cat_i % len(CATS)]; _cat_i += 1
    lst = PORCAT[c]; n = lst[_en_cat[c] % len(lst)]; _en_cat[c] += 1
    return n

# ---------- ruteo ----------
def par_src(L): return {"tipo": "comunicacion", "url": f"{iurl(PAR[L])}?modo=R2D2&gain={GAIN}",
                        "nombre": f"par {PAR[L]}", "modo": "R2D2"}
def ear(L, W, invertido):
    """(left, right). NORMAL: L=par R=mundo. INVERTIDO: L=mundo R=par."""
    p = par_src(L)
    return (src(W), p) if invertido else (p, src(W))

# plan de regímenes (cíclico). Cada entrada: (regimen, n_temas, fn_invertidos(seg_local)->set)
def plan_cycle():
    return [
        ("R1_PARES",  N_R1,      lambda s: {"A", "C"}),                  # A,C invertidos
        ("R2_FLIP",   N_R2,      lambda s: {"B", "D"}),                  # flip global (B,D invertidos)
        ("R3_UNOxUNO", N_R3_C*4, lambda s: {LETRAS[s // N_R3_C]}),       # uno a la vez, 5 temas c/u
        ("R4_AZAR",   N_R4,      None),                                  # aleatorio + mundo propio
    ]

SAMP = ["RC_total", "ICR", "IRDE", "voz_creadas", "voz_aprendidas", "act_orientacion_deg"]
def _m(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 5) if xs else None
def _pearson(xs, ys):
    pr = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    n = len(pr)
    if n < 8: return None
    mx = sum(p[0] for p in pr)/n; my = sum(p[1] for p in pr)/n
    num = sum((p[0]-mx)*(p[1]-my) for p in pr)
    dx = math.sqrt(sum((p[0]-mx)**2 for p in pr)); dy = math.sqrt(sum((p[1]-my)**2 for p in pr))
    return round(num/(dx*dy), 3) if dx and dy else None

def reset_zero():
    log("  [reset] down + wipe + up (a CERO)…")
    subprocess.run(["docker", "compose", "down"], cwd=DOCKER_DIR, capture_output=True, timeout=120)
    for v in VOLS: subprocess.run(["docker", "volume", "rm", v], capture_output=True, timeout=60)
    env = dict(os.environ); env.update(COMPOSE_BAKE="false", DOCKER_BUILDKIT="0")
    subprocess.run(["docker", "compose", "up", "-d"], cwd=DOCKER_DIR, env=env, capture_output=True, timeout=240)
    ok = 0
    for _ in range(50):
        ok = sum(1 for L in LETRAS if (get(L, "/ultima_fila") or {}) != {})
        if ok >= 4: break
        time.sleep(3)
    log(f"  [reset] accesibles {ok}/4"); return ok >= 4

def aplicar(regimen, invset, seg_local):
    """Configura los 4 oídos según el régimen. Devuelve (etiqs, dwell_max, info)."""
    etiqs = {}; dwells = []
    if regimen == "R4_AZAR":
        for L in LETRAS:
            W = siguiente_audio(); inv = bool(RNG.getrandbits(1))      # cada uno SU sonido + ruteo aleatorio
            l, r = ear(L, W, inv); dwells.append(dwell(W))
            _start(L, l, r, regimen, W, inv); etiqs[L] = f"AZAR {'INV' if inv else 'NRM'} {W[:14]}"
    else:
        W = siguiente_audio()                                          # mundo COMÚN
        for L in LETRAS:
            inv = L in (invset or set())
            l, r = ear(L, W, inv); dwells.append(dwell(W))
            _start(L, l, r, regimen, W, inv)
            etiqs[L] = f"{'INV' if inv else 'NRM'} {W[:16]}"
    return etiqs, max(dwells)
def _start(L, l, r, regimen, W, inv):
    cfg = {"left_src": l, "right_src": r, "binaural": True, "segundos": 2, "continuo": True, "criterio_duracion": "min"}
    post(L, "/start", {"cfg": cfg, "modo_vida": "experimento"})
    post(L, "/exp_tag", {"exp_topologia": "ENTORNO", "exp_ciclo": "1", "exp_regimen": regimen,
                         "exp_mundo_audio": W if isinstance(W, str) else "?", "exp_control": "real",
                         "exp_fuente_relacion": f"par:{PAR[L]}", "exp_invertido": str(inv)})

def main():
    log("=" * 72); log(f"ANIMA-4 ENTORNO complejo/dinámico · {datetime.datetime.now():%Y-%m-%d %H:%M} · {OUT}")
    log(f"pool {len(_files)} sonidos · {len(CATS)} categorías ({', '.join(CATS)}) · seed={SEED} · {DUR_TOTAL//60} min")
    log(f"regímenes/ciclo: R1_PARES({N_R1}) R2_FLIP({N_R2}) R3_UNOxUNO({N_R3_C}×4) R4_AZAR({N_R4})")
    reset_zero()
    t_ini = time.time(); t_fin = t_ini + DUR_TOTAL
    ori = {L: [] for L in LETRAS}; ori_reg = []                # orientación + régimen por muestra
    serie = {L: {c: [] for c in SAMP} for L in LETRAS}
    voz_ids = {L: Counter() for L in LETRAS}; rutas = []; nuevas = set()
    last_creadas = {L: 0 for L in LETRAS}; tl = []
    cyc = 0; t0 = round(time.time(), 2)
    while time.time() < t_fin:
        cyc += 1
        for (regimen, n_temas, invfn) in plan_cycle():
            for s in range(n_temas):
                if time.time() >= t_fin: break
                invset = invfn(s) if invfn else None
                etiqs, dwell_max = aplicar(regimen, invset, s)
                if s == 0: log(f"  ▸ ciclo{cyc} {regimen}: " + " · ".join(f"{L}:{etiqs[L]}" for L in LETRAS))
                fin_seg = time.time() + min(dwell_max, 60.0)
                while time.time() < fin_seg and time.time() < t_fin:
                    snap = {}
                    for L in LETRAS:
                        f = (get(L, "/ultima_fila") or {}).get("fila") or {}
                        o = f.get("act_orientacion_deg")
                        try: o = float(o) if o is not None else None
                        except Exception: o = None
                        ori[L].append(o); snap[L] = o
                        for c in SAMP:
                            v = f.get(c)
                            if v is not None:
                                try: serie[L][c].append(float(v))
                                except Exception: pass
                        try:
                            cr = int(float(f.get("voz_creadas") or 0))
                            if cr > last_creadas[L]:
                                log(f"  ✨ palabra nueva t={round(time.time()-t_ini)}s · {L} ({last_creadas[L]}→{cr}) [{regimen}]")
                            last_creadas[L] = cr
                        except Exception: pass
                        vid = f.get("voz_id")
                        if vid:
                            voz_ids[L][vid] += 1
                            if "palabra_" in vid or "apr_" in vid: nuevas.add(vid)
                        em = f.get("voz_emulada_de")
                        if em:
                            rutas.append((round(time.time()-t_ini, 1), regimen, L, vid, em))
                            log(f"  🔗 ruta t={round(time.time()-t_ini)}s {L}←{em} [{regimen}]")
                    ori_reg.append(regimen)
                    tl.append({"t": round(time.time()-t_ini, 1), "ciclo": cyc, "regimen": regimen,
                               "A": snap.get("A"), "B": snap.get("B"), "C": snap.get("C"), "D": snap.get("D"),
                               "creadas": sum(last_creadas.values()), "nuevas": len(nuevas)})
                    if len(tl) % 40 == 0:
                        _escribir(tl, ori, ori_reg, rutas, nuevas, last_creadas, serie)
                        log(f"  t={int(time.time()-t_ini)}s {regimen} · creadas={sum(last_creadas.values())} nuevas={len(nuevas)} rutas={len(rutas)}")
                    time.sleep(SAMPLE)
    for L in LETRAS:
        post(L, "/control", {"action": "stop"})
        _manifiesto.append({"cond_id": "ENTORNO", "organismo": L, "par": PAR[L],
                            "ts_real_ini": t0, "ts_real_fin": round(time.time(), 2)})
    _escribir(tl, ori, ori_reg, rutas, nuevas, last_creadas, serie); _empaquetar()
    log(f"✅ ENTORNO COMPLETO · creadas={sum(last_creadas.values())} nuevas={len(nuevas)} rutas={len(rutas)}")
    log("=" * 72)

def _corrs_por_regimen(ori, ori_reg):
    """Correlación de orientación por pareja, separada por régimen (R1/R2 = test limpio)."""
    out = {}
    for reg in ("R1_PARES", "R2_FLIP", "R3_UNOxUNO", "R4_AZAR"):
        idx = [i for i, r in enumerate(ori_reg) if r == reg]
        sub = {L: [ori[L][i] for i in idx] for L in LETRAS}
        out[reg] = {f"{a}-{b}": _pearson(sub[a], sub[b]) for a, b in
                    [("A", "B"), ("C", "D"), ("A", "C"), ("B", "D"), ("A", "D"), ("B", "C")]}
    return out

def _escribir(tl, ori, ori_reg, rutas, nuevas, last_creadas, serie):
    with open(os.path.join(OUT, "timeline_entorno.csv"), "w", newline="") as f:
        if tl:
            w = csv.DictWriter(f, fieldnames=list(tl[0].keys())); w.writeheader(); w.writerows(tl)
    with open(os.path.join(OUT, "rutas_lexicas.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["t_s", "regimen", "org_emite", "voz_id", "voz_emulada_de"])
        for r in rutas: w.writerow(r)
    if _manifiesto:
        with open(MANIF, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(_manifiesto[0].keys())); w.writeheader(); w.writerows(_manifiesto)
    cr = _corrs_por_regimen(ori, ori_reg)
    rc = _m(sum((serie[L]["RC_total"] for L in LETRAS), [])); icr = _m(sum((serie[L]["ICR"] for L in LETRAS), []))
    sm = ["# ANIMA-4 · Entorno complejo y dinámico — atención mutua vs contexto + invención\n", EPI,
          f"**{TS0}** · pool {len(_files)} sonidos ({len(CATS)} cat.) · ruteo cíclico R1→R2→R3→R4 · desde cero.\n",
          "## Correlación de ORIENTACIÓN por pareja y por régimen (R1/R2 = test limpio, mundo común)",
          "| régimen | A–B | C–D | A–C | B–D | A–D | B–C |", "|---|---|---|---|---|---|---|"]
    for reg in ("R1_PARES", "R2_FLIP", "R3_UNOxUNO", "R4_AZAR"):
        c = cr[reg]; sm.append(f"| {reg} | {c['A-B']} | {c['C-D']} | {c['A-C']} | {c['B-D']} | {c['A-D']} | {c['B-C']} |")
    sm += ["\n## Lectura del control de atención (R1: A,C invertidos · B,D normales · mundo común)",
           "- **CONTEXTO (artefacto):** mundo entra por oídos opuestos en A/C vs B/D → invertidos juntos, normales "
           "juntos, unos al revés → **A-C(+), B-D(+), A-B(−)**. Y en **R2 (flip global) la orientación se INVIERTE**.",
           "- **ATENCIÓN MUTUA:** A mira a B, C mira a D → acoplamiento DENTRO de pareja, **indiferente al flip R2**.",
           f"\n## Léxico (entorno rico): creadas={sum(last_creadas.values())} · voces_nuevas={len(nuevas)} · "
           f"rutas={len(rutas)} · RC_med={rc} · ICR_med={icr}",
           f"voces nuevas: {', '.join(sorted(nuevas)) if nuevas else '—'}\n",
           "Datos: timeline_entorno.csv · rutas_lexicas.csv · primarios (act_* completos, segmentables por exp_regimen).\n"]
    open(os.path.join(OUT, "resumen_entorno.md"), "w").write("\n".join(sm))

def _empaquetar():
    try:
        prim = os.path.join(OUT, "primarios_fisiologia.tar.gz")
        subprocess.run(["tar", "czf", prim, "-C", "/", HIST.lstrip("/")], capture_output=True, timeout=600)
        log(f"  ✅ {os.path.basename(prim)}")
    except Exception as e: log(f"  ! empaquetar: {e}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRÍADA · ¿una relación de a TRES produce diferencia estructurada sin inyectarla? (díada A↔B + palabra)
======================================================================================================
Pregunta del equipo Cosmogénesis, contestada por los propios organismos:
  De a DOS  = la díada A↔B (se hablan por la voz) → relación simétrica, refuerzo SIN estructura.
  De a TRES = un TERCERO que media y PERSISTE y CAMBIA: la PALABRA (S>0 propia, se re-usa, se re-acuña).
  A — palabra — B.

Hipótesis: si A y B parten IDÉNTICOS compartiendo UNA palabra-semilla NEUTRA (sin sentido precargado),
¿la palabra se DIFERENCIA —se usa/significa distinto para A que para B— por pura HISTORIA, sin que nadie
inyecte la diferencia?  SÍ → el de-a-tres genera estructura desde la vida.  NO → pegamento sin estructura.

Mecanismo (aporte de Gemini, Plano C): la diferencia EMERGE si especializar la palabra le BAJA el IRDE a
cada organismo respecto de usarla uniforme (estructura como atractor de menor energía). No se diseña la
diferencia; se diseña la pendiente.

DISEÑO LIMPIO — vocabulario base MÍNIMO (vía ANIMA_VOCES_DIR) para que la semilla sea el TERCERO dominante
y no quede ahogada por el banco de 70 palabras. Cada condición parte de CERO y simétrica (down+wipe A,B):
  C1  TRIADA_VIVA  banco base = SOLO la semilla (compartida, idéntica en A y B). ANIMA_CONTROL=real.
  C2  DIADA_SOLA   banco base = VACÍO (sin tercero compartido; coinan desde cero). real.  [contraste]
  C4  SHUFFLED     como C1 + ANIMA_CONTROL=shuffled → rompe la contingencia de la historia. FALSADOR:
                   la diferenciación DEBE desaparecer (si persiste, entró el Pastor).
  (C3 PALABRA_CONGELADA — palabra externa sin S propia — refinamiento siguiente si C1 diferencia.)

En todas, los organismos pueden ACUÑAR palabras nuevas (voces_creadas) cuando la semilla no cubre su
necesidad: el tercero vivo engendra un ecosistema. Se mide si la semilla (y ese ecosistema) DIVERGE.

MIDE (muestreo /ultima_fila: voz_id + estado, A y B): especialización funcional (estado en la emisión),
roles emergentes (quién emite/acuña/emula), atractor IRDE. Análisis → analizar_triada.py.

SALIDAS ~/Downloads/ANIMA_TRIADA_<ts>/: timeline_triada.csv, condiciones.csv, log.
ENV: DUR_COND (s, def 900), SAMPLE (s, def 2), GAIN (def 20.0), CONDS (def "C1,C2,C4").
"""
import os, sys, json, time, csv, datetime, subprocess, urllib.request

AQUI = os.path.dirname(os.path.abspath(__file__))
SEMILLA_WAV = os.path.join(AQUI, "semilla_raiz.wav")
DOCKER_DIR = "/Users/alexis/Desktop/RMD/Cosmolab/VSTCosmo/Célula_Madre/docker"
BANCO_DIR = "/app/celula_madre/voces_r2d2"          # banco base DENTRO del contenedor (se vacía y se siembra)
ORGS = {"A": {"port": 7788, "host": "anima-a", "cont": "anima-a", "vol": "anima-diada_anima_a_data"},
        "B": {"port": 7799, "host": "anima-b", "cont": "anima-b", "vol": "anima-diada_anima_b_data"}}
PAR = {"A": "B", "B": "A"}
DUR_COND = int(os.environ.get("DUR_COND", "900"))
SAMPLE = float(os.environ.get("SAMPLE", "2"))
GAIN = os.environ.get("GAIN", "20.0")
CONDS = [c.strip() for c in os.environ.get("CONDS", "C1,C3,C4").split(",") if c.strip()]
# (nombre, banco_base: "semilla"|"vacio", ANIMA_CONTROL, no_acunar: tercero ESTÉRIL)
COND_DEF = {"C1": ("TRIADA_VIVA",       "semilla", "real",     False),
            "C2": ("DIADA_SOLA",        "vacio",   "real",     False),
            "C3": ("PALABRA_CONGELADA", "semilla", "real",     True),
            "C4": ("SHUFFLED",          "semilla", "shuffled", False)}
SEED_LABEL = "semilla_raiz"

TS0 = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT = os.path.expanduser(f"~/Downloads/ANIMA_TRIADA_{TS0}")
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "ejecucion.log")
TIMELINE = os.path.join(OUT, "timeline_triada.csv")
MANIF = os.path.join(OUT, "condiciones.csv")

def log(m):
    line = f"[{datetime.datetime.now():%H:%M:%S}] {m}"; print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")

# ---------------- HTTP ----------------
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
def fila(L):
    f = (get(L, "/ultima_fila") or {}).get("fila") or {}
    est = get(L, "/estado") or {}
    m = dict(f)
    for k in ("voz_titulo", "vivo", "voz_creadas", "voz_propias", "voz_aprendidas", "oido_par"):
        if k in est: m[k] = est[k]
    return m

# ---------------- DOCKER / setup ----------------
def _dc(args, timeout=180):
    env = dict(os.environ); env.update(DOCKER_BUILDKIT="0", COMPOSE_DOCKER_CLI_BUILD="0")
    return subprocess.run(["docker", "compose"] + args, cwd=DOCKER_DIR, env=env, capture_output=True, timeout=timeout)

def _esperar_vivos(intentos=50):
    for _ in range(intentos):
        if all((get(L, "/ultima_fila") or {}) != {} for L in ORGS): return True
        time.sleep(3)
    return False

def reset_diada(control, no_acunar=False):
    """A y B a CERO (down+wipe volúmenes+up) con ANIMA_CONTROL y ANIMA_NO_ACUNAR dados. Arranque simétrico."""
    log(f"  [reset] down+wipe(A,B)+up · ANIMA_CONTROL={control} · no_acunar={no_acunar}")
    _dc(["stop", "anima-a", "anima-b"], timeout=90); _dc(["rm", "-f", "anima-a", "anima-b"], timeout=90)
    for L in ORGS.values():
        subprocess.run(["docker", "volume", "rm", L["vol"]], capture_output=True, timeout=60)
    env = dict(os.environ); env.update(DOCKER_BUILDKIT="0", COMPOSE_DOCKER_CLI_BUILD="0",
                                       ANIMA_CONTROL=control, ANIMA_NO_ACUNAR=("1" if no_acunar else ""))
    subprocess.run(["docker", "compose", "up", "-d", "anima-a", "anima-b"], cwd=DOCKER_DIR, env=env,
                   capture_output=True, timeout=240)
    return _esperar_vivos()

def preparar_banco(L, modo_banco):
    """Reduce el banco base DENTRO del contenedor a lo mínimo: SOLO la semilla, o VACÍO. Se vacía
    /app/celula_madre/voces_r2d2 (las 70 palabras) para que la semilla sea el TERCERO dominante."""
    cont = ORGS[L]["cont"]
    subprocess.run(["docker", "exec", cont, "sh", "-c", f"rm -f {BANCO_DIR}/*.wav"], capture_output=True, timeout=30)
    if modo_banco == "semilla":
        subprocess.run(["docker", "cp", SEMILLA_WAV, f"{cont}:{BANCO_DIR}/semilla_raiz.wav"],
                       capture_output=True, timeout=30)

def _src_par(L):
    return {"tipo": "comunicacion", "modo": "R2D2",
            "url": f"http://{ORGS[PAR[L]]['host']}:{ORGS[PAR[L]]['port']}/comunicacion/bloque.wav?modo=R2D2&gain={GAIN}",
            "nombre": f"voz de {PAR[L]}"}
def _start_diada():
    """Canal de voz A↔B: cada uno oye SÓLO al otro por su oído de par; el otro oído en silencio."""
    for L in ORGS:
        par = _src_par(L); sil = {"tipo": "demo", "spec": "silencio", "nombre": "silencio"}
        l, r = (sil, par) if L == "A" else (par, sil)     # A: par a la derecha · B: par a la izquierda
        cfg = {"left_src": l, "right_src": r, "binaural": True, "segundos": 2, "continuo": True, "criterio_duracion": "min"}
        post(L, "/start", {"cfg": cfg, "modo_vida": "experimento"})
        post(L, "/exp_tag", {"exp_topologia": "TRIADA", "exp_fuente_relacion": f"voz:{PAR[L]}"})

SAMP_COLS = ["voz_id", "voz_titulo", "voz_origen", "voz_emulada_de", "voz_creadas", "voz_aprendidas",
             "voz_propias", "prop_bienestar", "IRDE", "ICR", "met_energia", "necesidad", "OI",
             "voz_arousal", "voz_valence", "expectativa", "oao_imitacion_mag", "H_homeostasis", "RC_total"]
_buffer = []; _cols = ["ts", "t_cond", "cond", "org"] + SAMP_COLS; _manifiesto = []

def _volcar():
    with open(TIMELINE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_cols, extrasaction="ignore"); w.writeheader()
        for r in _buffer: w.writerow(r)
    with open(MANIF, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cond", "nombre", "banco", "control", "dur_s", "n_semilla", "muestras_A", "muestras_B"])
        w.writeheader()
        for r in _manifiesto: w.writerow(r)

def correr(cond):
    nombre, banco, control, no_acunar = COND_DEF[cond]
    log(f"=== {cond} · {nombre} · banco={banco} · control={control} · no_acunar={no_acunar} · {DUR_COND}s ===")
    if not reset_diada(control, no_acunar):
        log(f"  [!] {cond}: A/B inaccesibles; salto"); return
    for L in ORGS: preparar_banco(L, banco)
    log(f"  [banco] preparado ({banco}); reinicio proceso para cargarlo")
    _dc(["restart", "anima-a", "anima-b"], timeout=120); _esperar_vivos()
    _start_diada(); time.sleep(12)
    if banco == "semilla":
        ok = sum(1 for L in ORGS if SEED_LABEL in json.dumps(get(L, "/voces") or [], ensure_ascii=False))
        log(f"  [semilla] presente en banco: {ok}/2")
    t0 = time.time(); muestras = {"A": 0, "B": 0}; n_sem = 0
    while time.time() - t0 < DUR_COND:
        ts = datetime.datetime.now().isoformat(timespec="seconds"); tc = round(time.time() - t0, 1)
        for L in ORGS:
            f = fila(L)
            if not f: continue
            row = {"ts": ts, "t_cond": tc, "cond": cond, "org": L}
            for c in SAMP_COLS: row[c] = f.get(c)
            _buffer.append(row); muestras[L] += 1
            if f.get("voz_id") == SEED_LABEL: n_sem += 1
        if int(tc) % 120 < SAMPLE:
            log(f"  {cond} t={int(tc)}s · semilla emitida {n_sem}× · A={fila('A').get('voz_id')} B={fila('B').get('voz_id')}")
        time.sleep(SAMPLE)
    _manifiesto.append({"cond": cond, "nombre": nombre, "banco": banco, "control": control, "dur_s": DUR_COND,
                        "n_semilla": n_sem, "muestras_A": muestras["A"], "muestras_B": muestras["B"]})
    _volcar()
    log(f"  {cond} fin · A={muestras['A']} B={muestras['B']} · semilla {n_sem}×")

def main():
    log("=" * 76); log(f"ANIMA · TRÍADA (díada + palabra) · {datetime.datetime.now():%Y-%m-%d %H:%M} · {OUT}")
    log(f"condiciones={CONDS} · dur/cond={DUR_COND}s · sample={SAMPLE}s · banco base MÍNIMO")
    for cond in CONDS:
        if cond in COND_DEF: correr(cond)
        else: log(f"  [!] condición desconocida: {cond}")
    _volcar()
    log(f"FIN · timeline+manifiesto en {OUT}")

if __name__ == "__main__":
    main()

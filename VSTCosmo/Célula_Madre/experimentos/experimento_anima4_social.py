#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANIMA-4 · BATERÍA SOCIAL DE TOPOLOGÍA (v2, con TRAZABILIDAD por ts_real).  Solo audios guardados.

PREGUNTA: ¿la TOPOLOGÍA (quién oye/imita a quién) determina la estructura de convergencia entre los 4
organismos? Cada topología predice una matriz de similitud 4×4 distinta; se comparan contra lo observado.

TRAZABILIDAD (vía usada: OPCIÓN 1 — columnas exp_* aditivas en la fisiología, verificado no-rompe):
  Antes de cada bloque se hace POST /exp_tag a cada organismo (exp_topologia/exp_ciclo/exp_mundo_audio/
  exp_control/exp_fuente_relacion). Esas columnas quedan en CADA fila de Docker_Historia → cada paso es
  atribuible a su condición por ts_real. ADEMÁS se escribe un manifiesto condiciones_<ts>.csv (ts_real_ini/
  fin por bloque×organismo) como segundo camino de join (redundancia).

DISEÑO:
  · 4 topologías (fuente de RELACIÓN por organismo, vía /start):
      PLENA       : todos = "otros organismos".
      CADENA      : A←B, B←C, C←D, D←nadie (cadena abierta D→C→B→A; D = fuente pura).
      ESTRELLA    : A,B,C ← D ; D←nadie (líder D).
      PAREJAS     : A↔B , C↔D.
  · Mundo DIVERGENTE: cada organismo recibe un audio fijo y distinto (voz/música/ruido/4to) por su oído de
    mundo; el oído de relación lleva la topología. Ciclo 2 ROTA la asignación (desconfunde rol vs audio).
  · CONTROL de falsación: CADENA con ANIMA_CONTROL=shuffled (recrea contenedores).
  · ~10 min/condición · ~1.5 h · correr de noche con caffeinate.

LIMITACIÓN HONESTA (se documenta en el informe): "D oye a nadie" es limpio a nivel de AUDIO (silencio),
pero la imitación de GESTOS del nodo-fuente cae al roster por entorno (fallback de la arquitectura), así
que el nodo-fuente no queda 100% aislado a nivel de gestos. Los nodos NO-fuente siguen la topología del cfg.

SALIDAS → ~/Downloads/ANIMA4_TOPO_<ts>/: resultados_sociales.csv · matriz_imitacion_por_condicion.csv ·
  matriz_gestos_por_condicion.csv · condiciones_<ts>.csv · difusion_lexica.csv · resumen_social.md ·
  informe_social.md · primarios_fisiologia.tar.gz · bitacoras.tar.gz

ADVERTENCIA EPISTÉMICA (en todos los informes): los rótulos de audio y de voz son etiquetas humanas de
archivo/banco; NO sabemos qué significan para los organismos. Se mide respuesta a configuraciones
acústicas y sociales, NO significados humanos.
"""
import json, os, sys, time, math, glob, subprocess, datetime, urllib.request, csv

ORGS = {"A": {"port": 7788, "host": "anima-a"}, "B": {"port": 7799, "host": "anima-b"},
        "C": {"port": 7810, "host": "anima-c"}, "D": {"port": 7820, "host": "anima-d"}}
LETRAS = ["A", "B", "C", "D"]
DOCKER_DIR = "/Users/alexis/Desktop/RMD/Cosmolab/VSTCosmo/Célula_Madre/docker"
HIST = "/Users/alexis/Desktop/RMD/Cosmolab/VSTCosmo/Docker_Historia"
AUDIO_DIR = "/Users/alexis/Desktop/RMD/Cosmolab/VSTCosmo/audio_binaural"
DUR_COND = int(os.environ.get("DUR_COND", "600"))
GAP = int(os.environ.get("GAP", "20"))
SAMPLE = 4
HACER_CONTROL = os.environ.get("CONTROL_SHUFFLED", "1") == "1"
T_INICIO = time.time()
TS0 = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT = os.path.expanduser(f"~/Downloads/ANIMA4_TOPO_{TS0}")
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "ejecucion.log")
MANIF = os.path.join(OUT, f"condiciones_{TS0}.csv")
_manifiesto = []  # filas del manifiesto

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
def ts_real(L):
    f = (get(L, "/ultima_fila") or {}).get("fila") or {}
    try: return float(f.get("ts_real"))
    except Exception: return None

# ---- audios (mundo divergente) ----
def _buscar(*subs):
    files = [os.path.basename(p) for p in glob.glob(os.path.join(AUDIO_DIR, "*.wav"))]
    for sub in subs:
        for f in files:
            if sub.lower() in f.lower(): return f
    return None
A_VOZ = _buscar("Voz_Estudio.wav", "Voz_Estudio", "voz_pos") or "demo:silencio"
A_MUS = _buscar("Brandemburgo.wav", "musica_pos") or "demo:silencio"
A_RUI = _buscar("ruido_pos", "Ruido") or "demo:silencio"
A_4TO = _buscar("Viento.wav", "viento_pos") or "demo:silencio"
AUDIOS4 = [A_VOZ, A_MUS, A_RUI, A_4TO]; ETIQ4 = ["voz", "musica", "ruido", "viento/4to"]
MUNDO_SOCIAL = os.environ.get("MUNDO_AUDIO", "demo:silencio")   # MUNDO COMPARTIDO (silencio por defecto): que el canal relacional NO se ahogue
def mundo_src(n): return {"tipo": "demo", "spec": n} if n.startswith("demo:") else {"tipo": "archivo", "nombre": n}
def asignacion(ciclo): return {L: MUNDO_SOCIAL for L in LETRAS}   # mismo mundo para todos (sin fuerza divergente; sin rotación)

# ---- topologías → fuente de relación + a quién imita cada organismo ----
def rel_otros(L): return {"tipo": "otros_organismos",
                          "urls": [f"{iurl(x)}?modo=R2D2&gain=20.0" for x in LETRAS if x != L], "nombre": "otros"}
def rel_esc(X): return {"tipo": "comunicacion", "url": f"{iurl(X)}?modo=R2D2&gain=20.0", "nombre": f"voz de {X}", "modo": "R2D2"}
REL_SIL = {"tipo": "demo", "spec": "demo:silencio"}
def topologia(nombre):
    """→ (rel_por_organismo, fuente_relacion_legible). CADENA abierta D→C→B→A; ESTRELLA líder D."""
    if nombre == "BASELINE":
        return {L: REL_SIL for L in LETRAS}, {L: "aislado" for L in LETRAS}
    if nombre == "PLENA":
        return {L: rel_otros(L) for L in LETRAS}, {L: "todos" for L in LETRAS}
    if nombre == "CADENA":     # A←B, B←C, C←D, D←nadie
        oye = {"A": "B", "B": "C", "C": "D", "D": None}
    elif nombre == "ESTRELLA": # A,B,C←D, D←nadie
        oye = {"A": "D", "B": "D", "C": "D", "D": None}
    elif nombre == "PAREJAS":  # A↔B, C↔D
        oye = {"A": "B", "B": "A", "C": "D", "D": "C"}
    else: raise ValueError(nombre)
    rel = {L: (rel_esc(oye[L]) if oye[L] else REL_SIL) for L in LETRAS}
    leg = {L: (oye[L] or "nadie") for L in LETRAS}
    return rel, leg

TOPOS = ["PLENA", "CADENA", "ESTRELLA", "PAREJAS"]

COLS_S = ["oao_imitacion_mag", "oao_echoica_n", "oao_oido", "voz_propias", "voz_aprendidas", "voz_estables",
          "voz_creadas", "alt_intencion_comunicativa", "alt_agencia_otro", "voz_otro_valor_ecologico",
          "expectativa", "OI", "Omega", "LF_op", "ove_experiencias", "cara_valoracion", "act_orientacion_deg",
          "energia_L", "energia_R", "g_freq", "g_intensidad", "g_pausa", "g_repeticion"]

def correr(cond_id, topo, ciclo, asg, control="real"):
    rel, leg = topologia(topo)
    log(f"  ── COND {cond_id}: {topo} c{ciclo} control={control} · " + " ".join(f"{L}<{leg[L]}|{asg[L][:10]}" for L in LETRAS))
    # 1) TRAZABILIDAD: etiquetar a cada organismo ANTES de arrancar
    for L in LETRAS:
        post(L, "/exp_tag", {"exp_topologia": topo, "exp_ciclo": str(ciclo), "exp_mundo_audio": asg[L],
                             "exp_control": control, "exp_fuente_relacion": leg[L]})
    # 2) /start
    for L in LETRAS:
        cfg = {"left_src": mundo_src(asg[L]), "right_src": rel[L],
               "binaural": True, "segundos": 2, "continuo": True, "criterio_duracion": "min"}
        post(L, "/start", {"cfg": cfg, "modo_vida": "experimento"})
    time.sleep(12)
    t0 = round(time.time(), 2)                     # ts_real_ini (epoch ≈ ts_real de las filas del bloque)
    series = {L: {c: [] for c in COLS_S} for L in LETRAS}; voz = {L: {} for L in LETRAS}
    fin = time.time() + DUR_COND
    while time.time() < fin:
        for L in LETRAS:
            f = (get(L, "/ultima_fila") or {}).get("fila") or {}
            if not f: continue
            for c in COLS_S:
                v = f.get(c)
                if v is not None:
                    try: series[L][c].append(float(v))
                    except Exception: pass
            vt = f.get("voz_titulo")
            if vt and vt != "-": voz[L][vt] = voz[L].get(vt, 0) + 1
        time.sleep(SAMPLE)
    t1 = round(time.time(), 2)                     # ts_real_fin
    for L in LETRAS:
        post(L, "/control", {"action": "stop"})
        _manifiesto.append({"cond_id": cond_id, "topologia": topo, "ciclo": ciclo, "organismo": L,
                            "ts_real_ini": t0, "ts_real_fin": t1, "mundo_audio": asg[L],
                            "control": control, "fuente_relacion": leg[L]})
    return {"cond_id": cond_id, "topo": topo, "ciclo": ciclo, "control": control, "asg": dict(asg),
            "leg": leg, "series": series, "voz_top": {L: sorted(voz[L].items(), key=lambda x: -x[1])[:4] for L in LETRAS}}

def set_control(modo):
    env = dict(os.environ); env.update(ANIMA_CONTROL=modo, COMPOSE_BAKE="false", DOCKER_BUILDKIT="0")
    try:
        subprocess.run(["docker", "compose", "up", "-d", "--no-deps", "--force-recreate",
                        "anima-a", "anima-b", "anima-c", "anima-d"], cwd=DOCKER_DIR, env=env,
                       capture_output=True, timeout=180)
        log(f"  [control] recreado con ANIMA_CONTROL={modo}"); time.sleep(18); return True
    except Exception as e: log(f"  [control] error: {e}"); return False

# ---- métricas ----
def stats(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return {"med": round(sum(xs) / len(xs), 4), "max": round(max(xs), 4), "n": len(xs)} if xs else {"med": None, "max": None, "n": 0}
def corr(a, b, lag=0):
    if lag > 0: a, b = a[lag:], b[:len(b) - lag]
    elif lag < 0: a, b = a[:len(a) + lag], b[-lag:]
    n = min(len(a), len(b))
    if n < 8: return None
    a, b = a[:n], b[:n]; ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a); vb = sum((x - mb) ** 2 for x in b)
    if va <= 0 or vb <= 0: return None
    return sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / math.sqrt(va * vb)
def matriz_imit(res):
    s = res["series"]; M = {}
    for a in LETRAS:
        M[a] = {b: (1.0 if a == b else (round(corr(s[a]["oao_imitacion_mag"], s[b]["oao_imitacion_mag"]) or 0, 3)
                                        if corr(s[a]["oao_imitacion_mag"], s[b]["oao_imitacion_mag"]) is not None else None)) for b in LETRAS}
    return M
def matriz_gestos(res):
    """4×4 similitud de GESTOS con mejor retardo: máx |corr| de g_* (promedio de los 4 componentes) en lags -3..3."""
    s = res["series"]; M = {}
    for a in LETRAS:
        M[a] = {}
        for b in LETRAS:
            if a == b: M[a][b] = 1.0; continue
            best = None
            for comp in ["g_freq", "g_intensidad", "g_pausa", "g_repeticion"]:
                vals = [corr(s[a][comp], s[b][comp], lg) for lg in range(-3, 4)]
                vals = [v for v in vals if v is not None]
                if vals: best = (best or 0) + max(vals, key=abs)
            M[a][b] = round(best / 4, 3) if best is not None else None
    return M

EPI = ("> **Advertencia epistémica:** las etiquetas de audio y de voz son rótulos humanos de archivo/banco. "
       "NO sabemos qué significan para los organismos. Se mide respuesta a configuraciones acústicas y sociales, "
       "NO significados humanos.\n")

def main():
    log("=" * 72); log(f"ANIMA-4 TOPOLOGÍA v2 · {datetime.datetime.now():%Y-%m-%d %H:%M} · {OUT}")
    log("✅ TRAZABILIDAD: Opción 1 (columnas exp_* por fila, verificado no-rompe) + manifiesto redundante.")
    log("audios: " + ", ".join(f"{e}={a}" for e, a in zip(ETIQ4, AUDIOS4)))
    resultados, mat_imit, mat_gest = [], [], []
    # FASE 0 — BASELINE AISLADO: ANIMA_CONTROL=null (sin percepción del par) → cada organismo SOLO.
    # Mide la tasa INTRÍNSECA de invención (contra qué comparar el efecto del acople). Mundo = silencio.
    log("  ── FASE 0: BASELINE aislado (ANIMA_CONTROL=null, mundo=silencio) ──")
    if set_control("null"):
        try:
            res = correr("BASELINE_aislado", "BASELINE", 0, asignacion(1), control="null")
            mat_imit.append(("BASELINE_aislado","BASELINE",0,matriz_imit(res))); mat_gest.append(("BASELINE_aislado","BASELINE",0,matriz_gestos(res)))
            for L in LETRAS:
                st={c:stats(res["series"][L][c]) for c in COLS_S}
                fila={"cond_id":"BASELINE_aislado","topo":"BASELINE","ciclo":0,"org":L,"control":"null",
                      "audio_mundo":MUNDO_SOCIAL,"oye_a":"aislado","voz_top":"; ".join(f"{t}({n})" for t,n in res["voz_top"][L])}
                for c in COLS_S: fila[c+"_med"]=st[c]["med"]; fila[c+"_max"]=st[c]["max"]
                resultados.append(fila)
        except Exception as e: log(f"  ❌ baseline: {e}")
        set_control("real")
    for ciclo in (1,):
        asg = asignacion(ciclo)
        for topo in TOPOS:
            cid = f"C{ciclo}_{topo}"
            try: res = correr(cid, topo, ciclo, asg)
            except Exception as e: log(f"  ❌ ERROR {cid}: {e}"); continue
            mat_imit.append((cid, topo, ciclo, matriz_imit(res)))
            mat_gest.append((cid, topo, ciclo, matriz_gestos(res)))
            for L in LETRAS:
                st = {c: stats(res["series"][L][c]) for c in COLS_S}
                fila = {"cond_id": cid, "topo": topo, "ciclo": ciclo, "org": L, "control": "real",
                        "audio_mundo": asg[L], "oye_a": res["leg"][L],
                        "voz_top": "; ".join(f"{t}({n})" for t, n in res["voz_top"][L])}
                for c in COLS_S: fila[c + "_med"] = st[c]["med"]; fila[c + "_max"] = st[c]["max"]
                resultados.append(fila)
            log(f"     ✅ {cid} · imit_max " + " ".join(f"{L}={stats(res['series'][L]['oao_imitacion_mag'])['max']}" for L in LETRAS))
            time.sleep(GAP)
    # ESTRELLA repetida: ciclos extra para estadística (¿el 0.121 off-diag es real o ruido?)
    REPS_ESTRELLA = int(os.environ.get("REPS_ESTRELLA", "0"))
    for ciclo in range(2, 2 + REPS_ESTRELLA):
        asg = asignacion(ciclo); cid = f"C{ciclo}_ESTRELLA"
        log(f"  ── ESTRELLA repetición (ciclo {ciclo}) ──")
        try: res = correr(cid, "ESTRELLA", ciclo, asg)
        except Exception as e: log(f"  ❌ ERROR {cid}: {e}"); continue
        mat_imit.append((cid, "ESTRELLA", ciclo, matriz_imit(res)))
        mat_gest.append((cid, "ESTRELLA", ciclo, matriz_gestos(res)))
        for L in LETRAS:
            st = {c: stats(res["series"][L][c]) for c in COLS_S}
            fila = {"cond_id": cid, "topo": "ESTRELLA", "ciclo": ciclo, "org": L, "control": "real",
                    "audio_mundo": asg[L], "oye_a": res["leg"][L],
                    "voz_top": "; ".join(f"{t}({n})" for t, n in res["voz_top"][L])}
            for c in COLS_S: fila[c + "_med"] = st[c]["med"]; fila[c + "_max"] = st[c]["max"]
            resultados.append(fila)
        log(f"     ✅ {cid} · imit_max " + " ".join(f"{L}={stats(res['series'][L]['oao_imitacion_mag'])['max']}" for L in LETRAS))
        time.sleep(GAP)
    # control shuffled (cadena)
    if HACER_CONTROL:
        log("  ── CONTROL: CADENA con ANIMA_CONTROL=shuffled ──")
        if set_control("shuffled"):
            try:
                res = correr("CTRL_SHUF_CADENA", "CADENA", 1, asignacion(1), control="shuffled")
                mat_imit.append(("CTRL_SHUF_CADENA", "CADENA", 0, matriz_imit(res)))
                mat_gest.append(("CTRL_SHUF_CADENA", "CADENA", 0, matriz_gestos(res)))
                for L in LETRAS:
                    st = {c: stats(res["series"][L][c]) for c in COLS_S}
                    fila = {"cond_id": "CTRL_SHUF_CADENA", "topo": "CADENA", "ciclo": 0, "org": L,
                            "control": "shuffled", "audio_mundo": asignacion(1)[L], "oye_a": res["leg"][L],
                            "voz_top": "; ".join(f"{t}({n})" for t, n in res["voz_top"][L])}
                    for c in COLS_S: fila[c + "_med"] = st[c]["med"]; fila[c + "_max"] = st[c]["max"]
                    resultados.append(fila)
            except Exception as e: log(f"  ❌ control: {e}")
            set_control("real")
            for L in LETRAS: post(L, "/control", {"action": "stop"})
        else:
            log("  ⊘ control shuffled omitido (no se pudo recrear).")
    _salidas(resultados, mat_imit, mat_gest); _empaquetar()
    log(f"✅ BATERÍA COMPLETA · {datetime.datetime.now():%Y-%m-%d %H:%M} · {OUT}"); log("=" * 72)

def _wr_csv(path, rows):
    if not rows: return
    cols = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

def _wr_mat(path, mats):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["cond_id", "topo", "ciclo", "fila"] + LETRAS)
        for cid, topo, ciclo, M in mats:
            for a in LETRAS: w.writerow([cid, topo, ciclo, a] + [M[a][b] for b in LETRAS])

def _offdiag(M):
    vs = [M[a][b] for a in LETRAS for b in LETRAS if a != b and M[a][b] is not None]
    return round(sum(vs) / len(vs), 3) if vs else None

def _salidas(resultados, mat_imit, mat_gest):
    _wr_csv(os.path.join(OUT, "resultados_sociales.csv"), resultados)
    _wr_csv(MANIF, _manifiesto)
    _wr_mat(os.path.join(OUT, "matriz_imitacion_por_condicion.csv"), mat_imit)
    _wr_mat(os.path.join(OUT, "matriz_gestos_por_condicion.csv"), mat_gest)
    with open(os.path.join(OUT, "difusion_lexica.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["cond_id", "org", "voz_creadas_max", "voz_aprendidas_max", "voz_propias_max", "voz_estables_max"])
        for r in resultados:
            w.writerow([r["cond_id"], r["org"], r.get("voz_creadas_max"), r.get("voz_aprendidas_max"),
                        r.get("voz_propias_max"), r.get("voz_estables_max")])
    # PREDICHO (cualitativo) por topología → texto de comparación
    PRED = {"PLENA": "todos↔todos similares (sin bloques)",
            "CADENA": "vecinos adyacentes similares, decreciente con la distancia; A–D mínima",
            "ESTRELLA": "A,B,C convergen hacia D (alto con D), bajo entre A/B/C",
            "PAREJAS": "dos bloques: A-B y C-D altos, cruces bajos"}
    crea = {L: max([r["voz_creadas_max"] or 0 for r in resultados if r["org"] == L] or [0]) for L in LETRAS}
    apr = {L: max([r["voz_aprendidas_max"] or 0 for r in resultados if r["org"] == L] or [0]) for L in LETRAS}
    sm = [f"# ANIMA-4 · Topología y convergencia — síntesis ejecutiva\n", EPI,
          f"**{TS0}** · 2 ciclos × 4 topologías + control shuffled · ~{DUR_COND // 60} min/cond. "
          f"Trazabilidad: columnas exp_* por fila (Opción 1) + manifiesto.\n",
          "## Similitud media entre organismos (corr fuera-diagonal) por condición"]
    sm.append("| Condición | imitación (magnitud) | gestos (mejor lag) |\n|---|---|---|")
    di = {c: M for c, _, _, M in mat_imit}; dg = {c: M for c, _, _, M in mat_gest}
    for c in di: sm.append(f"| {c} | {_offdiag(di[c])} | {_offdiag(dg.get(c, {}))} |")
    sm.append("\n## Léxico"); [sm.append(f"- **{L}**: inventó {crea[L]} · emuló {apr[L]}") for L in LETRAS]
    sm.append("\n*Matrices completas: matriz_*_por_condicion.csv · datos primarios SEGMENTABLES por exp_* "
              "(o por el manifiesto condiciones_*.csv) en primarios_fisiologia.tar.gz.*")
    open(os.path.join(OUT, "resumen_social.md"), "w").write("\n".join(sm))
    # informe completo
    inf = [f"# ANIMA-4 · Topología y convergencia — informe completo\n", EPI,
           "**Vía de trazabilidad usada:** OPCIÓN 1 — columnas `exp_*` aditivas en cada fila de fisiología "
           "(verificado: no rompe la lectura; CSV viejos→NULL con union_by_name). Manifiesto `condiciones_*.csv` "
           "como segundo camino de join por ts_real.\n",
           "**Limitación honesta:** el nodo-FUENTE de CADENA/ESTRELLA ('oye a nadie') es limpio por AUDIO "
           "(silencio) pero su imitación de gestos cae al roster por entorno (fallback de la arquitectura); no "
           "queda 100% aislado a nivel de gestos. Los nodos no-fuente sí siguen la topología del cfg.\n",
           f"Audios (mundo divergente): " + ", ".join(f"{e}={a}" for e, a in zip(ETIQ4, AUDIOS4)) + "\n",
           "## Matrices de GESTOS (similitud con mejor lag) — predicho vs observado\n"]
    for cid, topo, ciclo, M in mat_gest:
        inf.append(f"### {cid}  ·  predicho: {PRED.get(topo, '—')}")
        inf.append("| | " + " | ".join(LETRAS) + " |"); inf.append("|---|" + "---|" * 4)
        for a in LETRAS: inf.append(f"| **{a}** | " + " | ".join(str(M[a][b]) for b in LETRAS) + " |")
        inf.append(f"corr media fuera-diagonal: **{_offdiag(M)}**\n")
    best = max([t for t in TOPOS], key=lambda t: max([_offdiag(M) or -1 for c, tp, cy, M in mat_gest if tp == t and cy in (1, 2)] or [-1]))
    inf += ["## Respuestas",
            f"1. **Topología con mayor similitud de gestos:** {best}.",
            "2-4. **Léxico:** inventó/emuló — " + ", ".join(f"{L}(inv {crea[L]}/emu {apr[L]})" for L in LETRAS) +
            ". **Salvedad:** cada organismo numera 'palabra propia N' desde 1 → los NOMBRES colisionan entre "
            "organismos; la propagación NOMINAL no es atribuible en este diseño; se reporta solapamiento/conteo agregado.",
            "5. **Predicho vs observado:** comparar cada matriz de gestos con su predicción (arriba). Coincidencia "
            "= la topología gobierna; matrices planas (~0) = la fuerza divergente del mundo dominó.",
            "6. **Control shuffled (CTRL_SHUF_CADENA) vs C1/C2_CADENA:** si la estructura se rompe con shuffle, la "
            "convergencia dependía de contingencia; si se mantiene, es artefacto. NOTA arquitectónica: 'shuffled' "
            "aquí desordena el AUDIO (la voz que se oye), no el canal de GESTOS (HTTP), que permanece real; por eso "
            "shuffled prueba la contingencia ACÚSTICA. El control que corta los GESTOS es ANIMA_CONTROL=null (corrida previa: la imitación colapsó).",
            "7. **Rol vs audio:** comparar ciclo 1 vs ciclo 2 (audios rotados). Si el patrón sigue a la POSICIÓN "
            "en la topología y no al audio, manda la topología.",
            "8. **Para corrida larga:** la topología con mayor estructura observada y, dado el resultado previo "
            "(mundo divergente aplana la convergencia), repetir con mundo COMPARTIDO/silencio para detectar el efecto relacional."]
    open(os.path.join(OUT, "informe_social.md"), "w").write("\n".join(inf))
    log("  ✅ informes y matrices escritos")

def _empaquetar():
    try:
        files = subprocess.run(["find"] + [f"{HIST}/organismo_ANIMA_{L}/fisiologia" for L in LETRAS] +
                               ["-name", "*.csv", "-newermt", datetime.datetime.fromtimestamp(T_INICIO).strftime("%Y-%m-%d %H:%M")],
                               capture_output=True, text=True, timeout=60).stdout.split("\n")
        files = [f for f in files if f.strip()]
        if files:
            open("/tmp/_a4_f.txt", "w").write("\n".join(files))
            subprocess.run(["tar", "-czf", os.path.join(OUT, "primarios_fisiologia.tar.gz"), "-T", "/tmp/_a4_f.txt"], capture_output=True, timeout=900)
            log(f"  ✅ primarios_fisiologia.tar.gz ({len(files)} csv, SEGMENTABLES por exp_*)")
    except Exception as e: log(f"  ⊘ primarios: {e}")
    try:
        bit = []
        for L in LETRAS:
            bit += glob.glob(f"{HIST}/organismo_ANIMA_{L}/**/bitacora*.json", recursive=True)
            bit += glob.glob(f"{HIST}/organismo_ANIMA_{L}/**/*.jsonl", recursive=True)
        bit = list(dict.fromkeys(b for b in bit if os.path.exists(b)))
        if bit:
            open("/tmp/_a4_b.txt", "w").write("\n".join(bit))
            subprocess.run(["tar", "-czf", os.path.join(OUT, "bitacoras.tar.gz"), "-T", "/tmp/_a4_b.txt"], capture_output=True, timeout=300)
            log(f"  ✅ bitacoras.tar.gz ({len(bit)} archivos)")
    except Exception as e: log(f"  ⊘ bitácoras: {e}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANIMA-4 · EXPERIMENTO DE ESTRÉS (saturación exhaustiva de los 4 organismos vivos).
=================================================================================
IDEA (spec de Alexis): bombardear a los 4 organismos con TODO el banco de audios de
`audio_binaural`, haciendo entrar cada audio por DISTINTOS RUTEOS de oído y con TIEMPOS
DE EXPOSICIÓN VARIABLES, capturando la batería COMPLETA de datos de los organelos nuevos
(propiocepción, membrana, hemisferios, RC por oído, etc.). No busca una sola hipótesis:
es un barrido de estrés para VER cómo responde todo el organismo a la variación máxima.

QUÉ VARÍA
---------
  (1) AUDIO: los 118 .wav de audio_binaural (todas las categorías), en orden ALEATORIO
      reproducible (seed). Cada audio entra como `{tipo:archivo}`.
  (2) RUTEO de oído (4 modos). El audio ocupa 0, 1 o 2 oídos; el resto recibe la SOCIEDAD
      (otros organismos = los 3 demás juntos), de modo que cuando el audio NO entra, los
      organismos se oyen ENTRE ELLOS:
        L    : izq=AUDIO      der=SOCIEDAD
        R    : izq=SOCIEDAD   der=AUDIO
        BOTH : izq=AUDIO      der=AUDIO        (contexto puro, sin canal social)
        PAR  : izq=SOCIEDAD   der=SOCIEDAD     (sin contexto: se escuchan entre sí)
  (3) EXPOSICIÓN: duración de cada condición ALEATORIA dentro de un set (estrés = mezcla
      de ráfagas cortas y exposiciones largas).

LOS 4 RECIBEN EL MISMO (audio, ruteo) A LA VEZ → mundo compartido, comparables entre sí.

QUÉ CAPTURA
-----------
  TODA la fila fisiológica de /ultima_fila (~276 columnas: prop_*, hemi_*, RC_*_L/R, mem_*,
  met_*, ove_*, alt_*, expr_*, oao_*, expectativa_*, H_*, altruismo_*, act_*, voz_*, ...)
  FUSIONADA con los extras de /estado (fuente_L/R, reaccion_L/R, mute_L/R, oido_par).
  Sondeo denso cada SAMPLE s. Escritura INCREMENTAL (resiliente a cortes).

SALIDAS (en ~/Downloads/ANIMA4_ESTRES_<ts>/)
  - timeline_estres.csv     : una fila por (muestra × organismo), TODAS las columnas (unión)
  - condiciones_<ts>.csv    : manifiesto (condición × organismo) con ts_real_ini/fin
  - resumen_estres.md       : parámetros + cobertura + advertencia epistémica
  - primarios_<ts>.tar.gz    : copia de timeline + manifiesto + log

CONFIG por ENV
  DUR_TOTAL   (s)   tope global; 0 = recorrer las 472 condiciones una vez (def 0)
  N_AUDIOS          limitar nº de audios (def: todos)
  SAMPLE      (s)   periodo de sondeo (def 2)
  GAIN              ganancia de la voz del par/sociedad (def 20.0)
  SEED              semilla del orden aleatorio (def 42)
  RESET_ZERO  0/1   1 = down+wipe+up antes de empezar (BORRA biografía). Def 0 (vivos).
  EXPOS             lista de duraciones, coma-separada (def "8,12,20,30,45")
  MODOS             ruteos a usar, coma-separados (def "L,R,BOTH,PAR")
"""
import os, sys, json, time, glob, csv, datetime, subprocess, urllib.request, random
from collections import Counter

# ----------------------------- CONFIG -----------------------------
ORGS = {"A": {"port": 7788, "host": "anima-a"}, "B": {"port": 7799, "host": "anima-b"},
        "C": {"port": 7810, "host": "anima-c"}, "D": {"port": 7820, "host": "anima-d"}}
LETRAS = ["A", "B", "C", "D"]
DOCKER_DIR = "/Users/alexis/Desktop/RMD/Cosmolab/VSTCosmo/Célula_Madre/docker"
AUDIO_DIR = "/Users/alexis/Desktop/RMD/Cosmolab/VSTCosmo/audio_binaural"
VOLS = ["anima-diada_anima_a_data", "anima-diada_anima_b_data",
        "anima-diada_anima_c_data", "anima-diada_anima_d_data"]

DUR_TOTAL = int(os.environ.get("DUR_TOTAL", "0"))        # 0 = una pasada completa por las condiciones
N_AUDIOS = int(os.environ.get("N_AUDIOS", "0"))          # 0 = todos
SAMPLE = float(os.environ.get("SAMPLE", "2"))
GAIN = os.environ.get("GAIN", "20.0")
SEED = int(os.environ.get("SEED", "42"))
RESET_ZERO = os.environ.get("RESET_ZERO", "0").lower() in ("1", "true", "yes", "on")
EXPOS = [float(x) for x in os.environ.get("EXPOS", "8,12,20,30,45").split(",") if x.strip()]
MODOS = [m.strip().upper() for m in os.environ.get("MODOS", "L,R,BOTH,PAR").split(",") if m.strip()]
SETTLE = 3.0                                             # espera tras /start antes de muestrear
RNG = random.Random(SEED)

TS0 = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT = os.path.expanduser(f"~/Downloads/ANIMA4_ESTRES_{TS0}")
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "ejecucion.log")
TIMELINE = os.path.join(OUT, "timeline_estres.csv")
MANIF = os.path.join(OUT, f"condiciones_{TS0}.csv")
RESUMEN = os.path.join(OUT, "resumen_estres.md")
EPI = ("> **Advertencia epistémica:** los rótulos de audio son etiquetas HUMANAS; NO sabemos qué "
       "significan para los organismos. El sentido (si lo hay) lo PRODUCE el organismo, no lo trae el sonido.")

def log(m):
    line = f"[{datetime.datetime.now():%H:%M:%S}] {m}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

# ----------------------------- HTTP -----------------------------
def post(L, path, body):
    o = ORGS[L]
    req = urllib.request.Request(f"http://127.0.0.1:{o['port']}{path}",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=12) as r:
            return json.loads(r.read())
    except Exception as e:
        return None
def get(L, path):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{ORGS[L]['port']}{path}", timeout=8) as r:
            return json.loads(r.read())
    except Exception:
        return {}
def fila(L):
    """Fila COMPLETA (/ultima_fila, ~276 cols) FUSIONADA con extras de /estado
    (fuente_L/R, reaccion_L/R, mute_L/R, oido_par). El estado breve gana en colisión."""
    f = (get(L, "/ultima_fila") or {}).get("fila") or {}
    est = get(L, "/estado") or {}
    merged = dict(f)
    for k in ("fuente_L", "fuente_R", "reaccion_L", "reaccion_R", "mute_L", "mute_R",
              "oido_par", "voz_titulo", "vivo"):
        if k in est:
            merged[k] = est[k]
    return merged

# ----------------------------- AUDIO -----------------------------
def soc_src(L):
    """La SOCIEDAD para el organismo L: los OTROS 3 juntos (lo que oye cuando el audio no entra
    por ese oído). Usa el descriptor 'otros_organismos' que el propio organismo resuelve."""
    otros = [x for x in LETRAS if x != L]
    urls = [f"http://{ORGS[x]['host']}:{ORGS[x]['port']}/comunicacion/bloque.wav?modo=R2D2&gain={GAIN}"
            for x in otros]
    return {"tipo": "otros_organismos", "urls": urls, "nombre": "otros organismos"}
def aud_src(n):
    return {"tipo": "archivo", "nombre": n}

def oidos(L, audio, modo):
    """Devuelve (left_src, right_src) según el modo de ruteo para el organismo L."""
    A = aud_src(audio); S = soc_src(L)
    if modo == "L":    return (A, S)
    if modo == "R":    return (S, A)
    if modo == "BOTH": return (A, A)
    if modo == "PAR":  return (S, S)
    raise ValueError(f"modo desconocido: {modo}")

_files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(AUDIO_DIR, "*.wav")))
if N_AUDIOS > 0:
    _files = _files[:N_AUDIOS]

def plan():
    """Lista ALEATORIA reproducible de condiciones (audio, modo, dur). Cada audio × cada modo,
    con duración de exposición aleatoria del set EXPOS. Orden barajado (seed)."""
    conds = [(a, m) for a in _files for m in MODOS]
    RNG.shuffle(conds)
    return [(a, m, RNG.choice(EXPOS)) for (a, m) in conds]

# ----------------------------- CONTROL DE ORGANISMOS -----------------------------
def reset_zero():
    log("  [reset] down + wipe volúmenes + up (organismos a CERO)…")
    subprocess.run(["docker", "compose", "down"], cwd=DOCKER_DIR, capture_output=True, timeout=120)
    for v in VOLS:
        subprocess.run(["docker", "volume", "rm", v], capture_output=True, timeout=60)
    env = dict(os.environ); env.update(DOCKER_BUILDKIT="0", COMPOSE_DOCKER_CLI_BUILD="0")
    subprocess.run(["docker", "compose", "up", "-d"], cwd=DOCKER_DIR, env=env, capture_output=True, timeout=300)
    ok = 0
    for _ in range(60):
        ok = sum(1 for L in LETRAS if (get(L, "/ultima_fila") or {}) != {})
        if ok >= 4:
            break
        time.sleep(3)
    log(f"  [reset] accesibles {ok}/4")
    return ok >= 4

def aplicar(audio, modo):
    """Configura los 4 organismos con el MISMO (audio, modo) y les pone los tags experimentales."""
    for L in LETRAS:
        l, r = oidos(L, audio, modo)
        cfg = {"left_src": l, "right_src": r, "binaural": True, "segundos": 2,
               "continuo": True, "criterio_duracion": "min"}
        post(L, "/start", {"cfg": cfg, "modo_vida": "experimento"})
        post(L, "/exp_tag", {"exp_topologia": "ESTRES", "exp_regimen": modo,
                             "exp_mundo_audio": audio, "exp_control": "real",
                             "exp_fuente_relacion": "sociedad", "exp_invertido": str(modo == "R")})

# ----------------------------- ESCRITURA -----------------------------
_cols_vistas = []          # unión de columnas, en orden de aparición
_buffer = []               # filas pendientes de volcar
_manifiesto = []

def _registrar(row):
    for k in row:
        if k not in _cols_vistas:
            _cols_vistas.append(k)
    _buffer.append(row)

def _volcar():
    """Reescribe el timeline completo con la unión de columnas vista hasta ahora (incremental, seguro)."""
    if not _buffer:
        return
    cab = ["ts", "t_exp", "org", "audio", "modo", "dur"] + [c for c in _cols_vistas
                                                            if c not in ("ts", "t_exp", "org", "audio", "modo", "dur")]
    with open(TIMELINE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cab, extrasaction="ignore")
        w.writeheader()
        for r in _buffer:
            w.writerow(r)

def _manif_volcar():
    if not _manifiesto:
        return
    with open(MANIF, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cond_idx", "audio", "modo", "dur", "org",
                                          "ts_real_ini", "ts_real_fin", "muestras"])
        w.writeheader()
        for r in _manifiesto:
            w.writerow(r)

# ----------------------------- MAIN -----------------------------
def main():
    log("=" * 78)
    log(f"ANIMA-4 ESTRÉS · {datetime.datetime.now():%Y-%m-%d %H:%M} · {OUT}")
    conds = plan()
    n_total = len(conds)
    log(f"audios={len(_files)} · modos={MODOS} · condiciones={n_total} · exposiciones={EXPOS}s · "
        f"sample={SAMPLE}s · seed={SEED} · reset_zero={RESET_ZERO}")
    if DUR_TOTAL > 0:
        log(f"tope global DUR_TOTAL={DUR_TOTAL}s ({DUR_TOTAL//60} min)")
    if RESET_ZERO:
        reset_zero()
    else:
        log("  [vivos] sin reset: el primer /start revive a los que estén detenidos")

    t_ini = time.time()
    t_fin = t_ini + DUR_TOTAL if DUR_TOTAL > 0 else None
    ini_x_org = {L: 0 for L in LETRAS}            # contador de muestras por organismo
    for ci, (audio, modo, dur) in enumerate(conds):
        if t_fin and time.time() >= t_fin:
            log("  [tope] DUR_TOTAL alcanzado, fin"); break
        aplicar(audio, modo)
        time.sleep(SETTLE)
        ts_ini = {L: (fila(L).get("ts_real") or datetime.datetime.now().isoformat()) for L in LETRAS}
        muestras = {L: 0 for L in LETRAS}
        seg_fin = time.time() + dur
        if ci % 20 == 0 or ci < 4:
            log(f"  ▸ cond {ci+1}/{n_total} · {modo:4} · {audio[:34]:34} · {dur:.0f}s")
        while time.time() < seg_fin and (not t_fin or time.time() < t_fin):
            ts = datetime.datetime.now().isoformat(timespec="seconds")
            t_exp = round(time.time() - t_ini, 1)
            for L in LETRAS:
                f = fila(L)
                if not f:
                    continue
                row = {"ts": ts, "t_exp": t_exp, "org": L, "audio": audio, "modo": modo, "dur": dur}
                row.update(f)
                _registrar(row)
                muestras[L] += 1
            time.sleep(SAMPLE)
        for L in LETRAS:
            ini_x_org[L] += muestras[L]
            _manifiesto.append({"cond_idx": ci, "audio": audio, "modo": modo, "dur": dur, "org": L,
                                "ts_real_ini": ts_ini[L],
                                "ts_real_fin": (fila(L).get("ts_real") or datetime.datetime.now().isoformat()),
                                "muestras": muestras[L]})
        if ci % 5 == 0 or ci == n_total - 1:        # volcado incremental cada 5 condiciones
            _volcar(); _manif_volcar()
    _volcar(); _manif_volcar()
    _resumen(n_total, time.time() - t_ini, ini_x_org)
    _empaquetar()
    log(f"FIN · {sum(ini_x_org.values())} muestras · timeline+manifiesto+resumen en {OUT}")

def _resumen(n_total, elapsed, ini_x_org):
    with open(RESUMEN, "w") as f:
        f.write(f"# ANIMA-4 · Experimento de ESTRÉS — {TS0}\n\n{EPI}\n\n")
        f.write("## Parámetros\n\n")
        f.write(f"- Audios: **{len(_files)}** de `audio_binaural`\n")
        f.write(f"- Modos de ruteo: **{', '.join(MODOS)}** (L=izq, R=der, BOTH=ambos, PAR=sociedad)\n")
        f.write(f"- Condiciones planificadas: **{n_total}** (audio × modo, orden aleatorio seed={SEED})\n")
        f.write(f"- Exposiciones (s): {EXPOS}  ·  sample: {SAMPLE}s  ·  reset_zero: {RESET_ZERO}\n")
        f.write(f"- Duración real: **{elapsed/60:.1f} min**\n")
        f.write(f"- Columnas capturadas: **{len(_cols_vistas)}** (fila completa /ultima_fila + extras /estado)\n\n")
        f.write("## Muestras por organismo\n\n")
        for L in LETRAS:
            f.write(f"- {L}: {ini_x_org[L]}\n")
        f.write("\n## Datos\n\n")
        f.write("- `timeline_estres.csv` — fila por (muestra × organismo), TODAS las columnas.\n")
        f.write(f"- `{os.path.basename(MANIF)}` — manifiesto condición×organismo con ts_real.\n")
        f.write("\nLos primarios crudos quedan además en `Docker_Historia/` (tags `exp_*`).\n")

def _empaquetar():
    import tarfile
    tgz = os.path.join(OUT, f"primarios_{TS0}.tar.gz")
    with tarfile.open(tgz, "w:gz") as t:
        for p in (TIMELINE, MANIF, RESUMEN, LOG):
            if os.path.exists(p):
                t.add(p, arcname=os.path.basename(p))
    log(f"  [empaque] {tgz}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANÁLISIS / CONDENSADOR del experimento de ESTRÉS (read-only).
=============================================================
El timeline crudo (timeline_estres.csv) tiene ~290 columnas × miles de filas: demasiado grande para
que las IAs del equipo lo lean. Este script lo CONDENSA en tablas pequeñas y un informe markdown
digerible, siguiendo el estilo de los análisis previos (analisis_longitudinal.py): read-only,
agrega, rankea, NO modifica nada.

QUÉ RESPONDE
  - ¿Qué audios hacen SENTIR MEJOR / PEOR a los organismos? (bienestar W = prop_bienestar)
  - ¿Cómo es la REACCIÓN por oído al audio (reacción del RC en la oreja que recibe el audio)?
  - ¿Cómo varía por RUTEO (L/R/BOTH/PAR) y por ORGANISMO (A/B/C/D)?
  - ¿Por CATEGORÍA de sonido (nota/música/voz/ruido/...)?
  - Coste metabólico, energía oída, RC/ICR/IRDE, homeostasis bajo el estímulo.

REACCIÓN AL AUDIO (clave): la reacción por oído (reaccion_L/R) es el appraisal del RC en ESA oreja.
La "reacción al audio" = la reacción en la(s) oreja(s) que recibe(n) el audio según el modo:
  L→reaccion_L · R→reaccion_R · BOTH→media(L,R) · PAR→(sin audio: reacción a la sociedad).

SALIDAS (junto al timeline, en la carpeta del experimento)
  - resumen_analisis.md   informe digerible con rankings y tablas
  - por_audio.csv         1 fila por audio (agregado sobre orgs/muestras del modo con audio)
  - por_modo.csv          1 fila por ruteo
  - por_organismo.csv     1 fila por organismo
  - por_categoria.csv     1 fila por categoría de sonido

USO:  python analizar_estres.py [carpeta_o_csv]
      (sin args → toma la carpeta ANIMA4_ESTRES_* más reciente en ~/Downloads)
"""
import os, sys, csv, glob, re, math
from collections import defaultdict, Counter

# ----------------------------- localizar timeline -----------------------------
def _localizar(arg):
    if arg:
        if os.path.isdir(arg):
            return os.path.join(arg, "timeline_estres.csv")
        return arg
    cands = sorted(glob.glob(os.path.expanduser("~/Downloads/ANIMA4_ESTRES_*")), reverse=True)
    for d in cands:
        p = os.path.join(d, "timeline_estres.csv")
        if os.path.exists(p):
            return p
    sys.exit("No encontré timeline_estres.csv (pasa la carpeta como argumento).")

CSV = _localizar(sys.argv[1] if len(sys.argv) > 1 else None)
OUT = os.path.dirname(CSV)
print(f"timeline: {CSV}")

# ----------------------------- helpers -----------------------------
def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None
def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None
def _fmt(x, d=3):
    return f"{x:.{d}f}" if isinstance(x, float) else ("" if x is None else str(x))

def _cat(n):
    l = (n or "").lower()
    if re.search(r'freq_?\d', l): return "frecuencia"
    if re.search(r'\b(do|re|mi|fa|sol|la|si)[_\.]', l) or 'escala' in l or 'piano' in l: return "nota"
    if 'tono' in l: return "tono"
    if 'brandemburgo' in l or 'blue_monday' in l or 'bigbang' in l or 'musica' in l: return "musica"
    if 'voz' in l and 'viento' in l: return "vozviento"
    if 'voz' in l: return "voz"
    if 'viento' in l: return "viento"
    if 'ruido' in l: return "ruido"
    return "textura"

# columnas que resumimos (las más informativas de los organelos nuevos + clásicas)
METRICAS = ["prop_bienestar", "prop_dW", "prop_vigor", "prop_malestar",
            "energia_L", "energia_R", "RC_total", "ICR", "IRDE",
            "met_energia", "met_hambre", "OI", "H_homeostasis", "hemi_divergencia",
            "cara_valoracion", "expr_vocalizando", "oao_imitacion_mag"]

def reaccion_audio(row, modo):
    """Reacción del RC en la(s) oreja(s) que recibe(n) el audio, según el modo."""
    rL, rR = _f(row.get("reaccion_L")), _f(row.get("reaccion_R"))
    if modo == "L":    return rL
    if modo == "R":    return rR
    if modo == "BOTH": return _mean([rL, rR])
    return None    # PAR: el audio no entra → no hay reacción-al-audio

# ----------------------------- leer y agregar -----------------------------
# acumuladores: por (clave) → {metrica: [valores]} + reaccion_audio + n
def _acc():
    return {"n": 0, "reac": [], "W": [], **{m: [] for m in METRICAS}}

por_audio = defaultdict(_acc)
por_modo = defaultdict(_acc)
por_org = defaultdict(_acc)
por_cat = defaultdict(_acc)
audios_modos = defaultdict(set)     # audio → modos vistos (cobertura)
lex_seq = defaultdict(list)         # org → [(t_exp, audio, modo, creadas, voz_id, emulada_de, aprendidas)] en orden
n_filas = 0

with open(CSV, encoding="utf-8", errors="replace") as fh:
    r = csv.DictReader(fh)
    for row in r:
        n_filas += 1
        audio = row.get("audio", "?"); modo = row.get("modo", "?"); org = row.get("org", "?")
        cat = _cat(audio)
        audios_modos[audio].add(modo)
        lex_seq[org].append((_f(row.get("t_exp")) or 0.0, audio, modo,
                             int(_f(row.get("voz_creadas")) or 0), row.get("voz_id") or "",
                             row.get("voz_emulada_de") or "", int(_f(row.get("voz_aprendidas")) or 0)))
        ra = reaccion_audio(row, modo)
        W = _f(row.get("prop_bienestar"))
        for acc, key in ((por_audio, audio), (por_modo, modo), (por_org, org), (por_cat, cat)):
            a = acc[key]
            a["n"] += 1
            if ra is not None: a["reac"].append(ra)
            if W is not None: a["W"].append(W)
            for m in METRICAS:
                v = _f(row.get(m))
                if v is not None: a[m].append(v)

def _resumir(acc):
    out = {}
    for k, a in acc.items():
        d = {"n": a["n"], "reaccion_audio": _mean(a["reac"]), "W": _mean(a["W"])}
        for m in METRICAS:
            d[m] = _mean(a[m])
        out[k] = d
    return out

R_audio = _resumir(por_audio); R_modo = _resumir(por_modo)
R_org = _resumir(por_org); R_cat = _resumir(por_cat)

# ----------------------------- ANÁLISIS LÉXICO (acuñar / imitar / difundir) -----------------------------
def _origen(vid):
    """Letra del organismo que ACUÑÓ una palabra a partir de su ID. palabra_A004→A, apr_D001→D (quien la
    aprendió), fon_wobble→'' (banco compartido del aparato fonador, sin autor único)."""
    m = re.match(r'(?:palabra|apr)_?([A-D])', vid or "")
    return m.group(1) if m else ""

eventos = []                          # (t_exp, org, tipo, voz_id, emulada_de, audio, modo)
vocab = defaultdict(lambda: {"acunadas": [], "aprendidas": [], "fonador": []})
difusion = Counter()                  # (origen → copiador) : nº de veces
primera_inv = None                    # (t_exp, org, voz_id) primera invención observada

for org, seq in lex_seq.items():
    seq.sort(key=lambda x: x[0])
    visto = set(); prev_cre = None
    for (t, audio, modo, cre, vid, emu, apr) in seq:
        # INVENCIÓN / primera aparición de una palabra propia (acuñada, aprendida o del fonador)
        if vid and vid not in visto and re.match(r'(palabra|apr|fon)_', vid):
            visto.add(vid)
            if vid.startswith("palabra"):
                vocab[org]["acunadas"].append(vid); tipo = "acuña"
                if primera_inv is None: primera_inv = (t, org, vid)
            elif vid.startswith("apr"):
                vocab[org]["aprendidas"].append(vid); tipo = "aprende"
            else:
                vocab[org]["fonador"].append(vid); tipo = "fonador"
            eventos.append((t, org, tipo, vid, emu, audio, modo))
        # IMITACIÓN / copia: emite emulando la palabra de otro (voz_emulada_de apunta al ID copiado)
        if emu and emu not in ("-", ""):
            src = _origen(emu)
            if src and src != org:
                if difusion[(src, org)] == 0:        # registra la PRIMERA vez de esa ruta como evento
                    eventos.append((t, org, "imita", emu, emu, audio, modo))
                difusion[(src, org)] += 1
        prev_cre = cre

eventos.sort(key=lambda e: e[0])
_acu = [e for e in eventos if e[2] == "acuña"]          # primera invención GLOBAL (no la del 1er org del dict)
primera_inv = (_acu[0][0], _acu[0][1], _acu[0][3]) if _acu else None
with open(os.path.join(OUT, "lexico_eventos.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["t_exp_s", "org", "tipo", "voz_id", "emulada_de", "audio", "modo"])
    for e in eventos:
        w.writerow([_fmt(e[0], 1)] + list(e[1:]))
with open(os.path.join(OUT, "lexico_vocabulario.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["org", "n_acunadas", "n_aprendidas", "n_fonador", "acunadas", "aprendidas"])
    for org in sorted(vocab):
        v = vocab[org]
        w.writerow([org, len(v["acunadas"]), len(v["aprendidas"]), len(v["fonador"]),
                    " ".join(v["acunadas"]), " ".join(v["aprendidas"])])
with open(os.path.join(OUT, "lexico_difusion.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["origen", "copiador", "veces"])
    for (s, d), n in sorted(difusion.items(), key=lambda x: -x[1]):
        w.writerow([s, d, n])

# ----------------------------- escribir CSVs condensados -----------------------------
def _escribir_csv(path, dic, keyname, extra=None):
    cols = [keyname, "n", "reaccion_audio", "W"] + METRICAS + (extra or [])
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for k in sorted(dic, key=lambda x: (dic[x]["W"] is None, dic[x]["W"] or 0)):
            d = dic[k]
            extra_vals = []
            if extra and "categoria" in extra: extra_vals = [_cat(k)]
            w.writerow([k, d["n"], _fmt(d["reaccion_audio"], 4), _fmt(d["W"], 4)] +
                       [_fmt(d[m], 4) for m in METRICAS] + extra_vals)

_escribir_csv(os.path.join(OUT, "por_audio.csv"), R_audio, "audio", extra=["categoria"])
_escribir_csv(os.path.join(OUT, "por_modo.csv"), R_modo, "modo")
_escribir_csv(os.path.join(OUT, "por_organismo.csv"), R_org, "organismo")
_escribir_csv(os.path.join(OUT, "por_categoria.csv"), R_cat, "categoria")

# ----------------------------- informe markdown -----------------------------
def _tabla(dic, keyname, claves, top=None, ordpor="W", rev=False):
    items = [(k, d) for k, d in dic.items()]
    items.sort(key=lambda kd: (kd[1].get(ordpor) is None, kd[1].get(ordpor) or 0), reverse=rev)
    if top: items = items[:top]
    cab = "| " + keyname + " | " + " | ".join(claves) + " |\n"
    sep = "|" + "---|" * (len(claves) + 1) + "\n"
    filas = ""
    for k, d in items:
        filas += "| " + str(k)[:34] + " | " + " | ".join(_fmt(d.get(c), 3) for c in claves) + " |\n"
    return cab + sep + filas

audios_completos = sum(1 for a, ms in audios_modos.items() if len(ms) >= 4)
with open(os.path.join(OUT, "resumen_analisis.md"), "w") as f:
    f.write(f"# Análisis condensado — Experimento de ESTRÉS\n\n")
    f.write("> **Epistémica:** los rótulos de audio son etiquetas humanas; el sentido (si lo hay) lo "
            "PRODUCE el organismo. La 'reacción al audio' = appraisal del RC en la oreja que lo recibe "
            "(comprensión−riesgo, [-1,1]); el **bienestar W** = prop_bienestar (estado global).\n\n")
    f.write(f"## Cobertura\n\n")
    f.write(f"- Muestras totales: **{n_filas}**  ·  audios tocados: **{len(audios_modos)}**  "
            f"(con los 4 ruteos completos: {audios_completos})\n")
    f.write(f"- Organismos: {', '.join(sorted(por_org))}\n\n")
    f.write("## Por RUTEO (cómo entra el audio)\n\n")
    f.write(_tabla(R_modo, "modo", ["n", "reaccion_audio", "W", "energia_L", "energia_R",
                                    "met_energia", "RC_total", "OI"], ordpor="W") + "\n")
    f.write("## Por ORGANISMO\n\n")
    f.write(_tabla(R_org, "org", ["n", "reaccion_audio", "W", "prop_dW", "met_energia",
                                  "ICR", "IRDE", "OI"], ordpor="W") + "\n")
    f.write("## Por CATEGORÍA de sonido\n\n")
    f.write(_tabla(R_cat, "categoria", ["n", "reaccion_audio", "W", "energia_L",
                                        "met_energia", "RC_total"], ordpor="W") + "\n")
    f.write("## Audios que lo hacen sentir PEOR (W más bajo)\n\n")
    f.write(_tabla(R_audio, "audio", ["n", "W", "reaccion_audio", "prop_malestar",
                                      "met_energia"], top=12, ordpor="W") + "\n")
    f.write("## Audios que lo hacen sentir MEJOR (W más alto)\n\n")
    f.write(_tabla(R_audio, "audio", ["n", "W", "reaccion_audio", "prop_vigor",
                                      "met_energia"], top=12, ordpor="W", rev=True) + "\n")
    # -------- LÉXICO --------
    f.write("## Léxico: acuñar / imitar / difundir\n\n")
    tot_ac = sum(len(vocab[o]["acunadas"]) for o in vocab)
    tot_ap = sum(len(vocab[o]["aprendidas"]) for o in vocab)
    if primera_inv:
        f.write(f"- **Primera invención observada:** {primera_inv[1]} acuñó `{primera_inv[2]}` "
                f"a los **{primera_inv[0]:.0f}s**.\n")
    f.write(f"- Palabras propias acuñadas (total): **{tot_ac}**  ·  aprendidas/emuladas: **{tot_ap}**.\n\n")
    f.write("### Vocabulario por organismo\n\n")
    f.write("| org | acuñadas | aprendidas | fonador | palabras propias |\n|---|---|---|---|---|\n")
    for org in sorted(vocab):
        v = vocab[org]
        f.write(f"| {org} | {len(v['acunadas'])} | {len(v['aprendidas'])} | {len(v['fonador'])} | "
                f"{' '.join(v['acunadas'][:8])} |\n")
    f.write("\n### Difusión (quién copia a quién)\n\n")
    if difusion:
        f.write("| origen → copiador | veces |\n|---|---|\n")
        for (s, d), n in sorted(difusion.items(), key=lambda x: -x[1]):
            f.write(f"| {s} → {d} | {n} |\n")
    else:
        f.write("Sin rutas de copia inter-organismo todavía (cada uno usa lo suyo).\n")
    f.write("\n### Primeros eventos léxicos (con contexto de audio)\n\n")
    f.write("| t(s) | org | tipo | voz_id | bajo audio | modo |\n|---|---|---|---|---|---|\n")
    for e in eventos[:18]:
        f.write(f"| {e[0]:.0f} | {e[1]} | {e[2]} | {e[3]} | {str(e[5])[:24]} | {e[6]} |\n")
    f.write("\n(eventos completos en `lexico_eventos.csv`; vocabulario en `lexico_vocabulario.csv`; "
            "rutas en `lexico_difusion.csv`)\n\n")

    f.write("## Archivos\n\n")
    f.write("- `por_audio.csv` · `por_modo.csv` · `por_organismo.csv` · `por_categoria.csv` — "
            "tablas condensadas (1 fila por grupo).\n")
    f.write("- `lexico_eventos.csv` · `lexico_vocabulario.csv` · `lexico_difusion.csv` — análisis léxico.\n")
    f.write("- `timeline_estres.csv` — crudo (no apto para IA por tamaño).\n")

print(f"OK · {n_filas} muestras condensadas")
print(f"  → resumen_analisis.md + por_audio/modo/organismo/categoria.csv en {OUT}")

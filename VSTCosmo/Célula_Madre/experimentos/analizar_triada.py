#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANÁLISIS de la TRÍADA (read-only): ¿la palabra-semilla SE DIFERENCIÓ por historia, o quedó uniforme?
=====================================================================================================
Condensa timeline_triada.csv y emite un veredicto FALSABLE a la pregunta de Cosmogénesis:
¿una relación de a tres (A + B + palabra) produce diferencia estructurada sin inyectarla?

OBSERVABLES (todos emergentes, anti-Shannon):
  (1) ESPECIALIZACIÓN FUNCIONAL — el ESTADO interno en el instante en que cada organismo emite la
      semilla. Si A y B la emiten desde estados internos DISTINTOS, y la distancia CRECE con la
      historia (mitad tardía > mitad temprana), la palabra se especializó distinto en cada uno.
  (2) ROLES EMERGENTES — asimetría en quién emite/acuña vs adopta/emula, nacida de un inicio simétrico.
  (3) ATRACTOR IRDE (mecanismo de Gemini) — ¿usar la semilla deja el IRDE por DEBAJO del basal? La
      diferenciación se "paga" si especializar baja el riesgo.

VEREDICTO:
  C1 diferencia (índice alto + CRECE) · C4(shuffled) lo aplana · C2 simétrica  → SÍ: el de-a-tres
     genera diferencia estructurada por pura historia.
  C1 se queda uniforme como C2  → NO: pegamento sin estructura; la frontera es honda.

USO: python analizar_triada.py [carpeta]   (sin args → ANIMA_TRIADA_* más reciente en ~/Downloads)
"""
import os, sys, csv, glob, math
from collections import defaultdict

def _loc(arg):
    if arg:
        return os.path.join(arg, "timeline_triada.csv") if os.path.isdir(arg) else arg
    for d in sorted(glob.glob(os.path.expanduser("~/Downloads/ANIMA_TRIADA_*")), reverse=True):
        p = os.path.join(d, "timeline_triada.csv")
        if os.path.exists(p): return p
    sys.exit("No encontré timeline_triada.csv")

CSV = _loc(sys.argv[1] if len(sys.argv) > 1 else None)
OUT = os.path.dirname(CSV)
SEED = "semilla_raiz"
ESTADO = ["prop_bienestar", "IRDE", "met_energia", "necesidad", "OI", "voz_arousal", "voz_valence", "expectativa"]

def _f(x):
    try: return float(x)
    except (TypeError, ValueError): return None
def _mean(xs):
    xs = [x for x in xs if x is not None]; return sum(xs) / len(xs) if xs else None
def _std(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2: return None
    m = sum(xs) / len(xs); return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
conds = sorted(set(r["cond"] for r in rows))

# normalización (z-score) por dimensión sobre TODO el dataset → distancias escala-libres
norm = {}
for c in ESTADO:
    vals = [_f(r.get(c)) for r in rows]
    norm[c] = (_mean(vals) or 0.0, _std(vals) or 1.0)
def zvec(r):
    return [((_f(r.get(c)) or norm[c][0]) - norm[c][0]) / (norm[c][1] or 1.0) for c in ESTADO]
def dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def emite_semilla(r):
    return (r.get("voz_id") == SEED) or (str(r.get("voz_titulo")).strip().lower() == "raíz")

res = {}
for cond in conds:
    rc = [r for r in rows if r["cond"] == cond]
    seed_rows = {"A": [], "B": []}
    for r in rc:
        if emite_semilla(r) and r["org"] in seed_rows:
            seed_rows[r["org"]].append(r)
    nA, nB = len(seed_rows["A"]), len(seed_rows["B"])
    # (1) especialización funcional: distancia entre estado-medio-de-emisión de A vs B
    def vmean(rs):
        if not rs: return None
        vs = [zvec(r) for r in rs]; return [sum(col) / len(col) for col in zip(*vs)]
    mA, mB = vmean(seed_rows["A"]), vmean(seed_rows["B"])
    diff = dist(mA, mB) if (mA and mB) else None
    # crecimiento: mitad temprana vs tardía (por t_cond)
    def mitad(rs, tardia):
        if len(rs) < 4: return None
        rs = sorted(rs, key=lambda r: _f(r.get("t_cond")) or 0)
        h = rs[len(rs)//2:] if tardia else rs[:len(rs)//2]
        return vmean(h)
    dEarly = dist(mitad(seed_rows["A"], False), mitad(seed_rows["B"], False)) if (nA >= 4 and nB >= 4) else None
    dLate = dist(mitad(seed_rows["A"], True), mitad(seed_rows["B"], True)) if (nA >= 4 and nB >= 4) else None
    # (2) roles emergentes: asimetría de uso de la semilla + creadas/aprendidas
    rol_uso = (nA - nB) / (nA + nB) if (nA + nB) else None
    def last(L, col):
        xs = [_f(r.get(col)) for r in rc if r["org"] == L and _f(r.get(col)) is not None]
        return xs[-1] if xs else None
    crA, crB = last("A", "voz_creadas"), last("B", "voz_creadas")
    apA, apB = last("A", "voz_aprendidas"), last("B", "voz_aprendidas")
    # emulación de la semilla (alguien la emuló del otro)
    emul = sum(1 for r in rc if str(r.get("voz_emulada_de") or "").strip() not in ("", "-"))
    # (3) atractor IRDE: IRDE al emitir semilla vs IRDE global (por org)
    def irde(rs): return _mean([_f(r.get("IRDE")) for r in rs])
    irde_seedA, irde_seedB = irde(seed_rows["A"]), irde(seed_rows["B"])
    irde_globA = irde([r for r in rc if r["org"] == "A"]); irde_globB = irde([r for r in rc if r["org"] == "B"])
    # baseline de simetría general (para C2 sin semilla): divergencia del estado-medio A vs B
    gA = vmean([r for r in rc if r["org"] == "A"]); gB = vmean([r for r in rc if r["org"] == "B"])
    sim_global = dist(gA, gB) if (gA and gB) else None
    res[cond] = dict(nA=nA, nB=nB, diff=diff, dEarly=dEarly, dLate=dLate, rol_uso=rol_uso,
                     crA=crA, crB=crB, apA=apA, apB=apB, emul=emul,
                     irde_seedA=irde_seedA, irde_seedB=irde_seedB, irde_globA=irde_globA,
                     irde_globB=irde_globB, sim_global=sim_global)

def fx(x, d=3): return f"{x:.{d}f}" if isinstance(x, float) else ("—" if x is None else str(x))

# ---- veredicto ----
def crece(c):
    e, l = res[c]["dEarly"], res[c]["dLate"]
    return (e is not None and l is not None and l > e * 1.15)
c1 = res.get("C1", {}); c2 = res.get("C2", {}); c4 = res.get("C4", {})
# DISCRIMINADORES de estructura (NO la distancia de estado, que deriva por ruido en ambas condiciones):
#   (a) ROLES emergentes: asimetría de uso de la semilla nacida de un inicio simétrico.
#   (b) ATRACTOR IRDE: usar la semilla deja al organismo en MENOR riesgo (≥30% bajo el global) = la
#       especialización se paga metabólicamente (mecanismo de Gemini).
def _rolasim(c):
    r = res.get(c, {}).get("rol_uso"); return abs(r) if r is not None else None
def _attractor(c, L):
    s = res.get(c, {}).get(f"irde_seed{L}"); g = res.get(c, {}).get(f"irde_glob{L}")
    return (s is not None and g is not None and g > 0 and s < 0.7 * g)
c1_roles = (_rolasim("C1") is not None and _rolasim("C1") > 0.2)
c1_attr = (_attractor("C1", "A") or _attractor("C1", "B"))
c1_estructura = c1_roles and c1_attr and (c1.get("nA", 0) >= 8 and c1.get("nB", 0) >= 8)
c4_colapsa = ("C4" not in res) or (((_rolasim("C4") is None) or _rolasim("C4") < 0.15) and
                                   not (_attractor("C4", "A") or _attractor("C4", "B")))
if c1_estructura and c4_colapsa:
    veredicto = ("**SÍ (provisional)** — una relación de a TRES (A + B + palabra-semilla) generó **diferencia "
                 "estructurada por pura historia**, y sin que nadie la inyectara:\n"
                 f"- **Roles emergentes**: de un inicio simétrico, la palabra se volvió de uno más que del otro "
                 f"(asimetría de uso {fx(c1.get('rol_uso'))}; A={c1.get('nA')} emisiones, B={c1.get('nB')}).\n"
                 f"- **Atractor metabólico (mecanismo de Gemini)**: usar la palabra deja al organismo en MENOR "
                 f"riesgo (IRDE@semilla {fx(c1.get('irde_seedA'),4)} vs global {fx(c1.get('irde_globA'),4)}). La "
                 f"especialización se *paga* — es un atractor de menor energía.\n"
                 f"- **Falsador OK**: barajar la historia (C4/shuffled) **borra** ambos signos (asimetría "
                 f"{fx(c4.get('rol_uso'))}, atractor desaparece). La diferencia dependía de la historia compartida.\n\n"
                 "La pared se rompe **desde la vida**, no desde la física: el tercero con S>0 diferencia.")
elif (c1.get("nA", 0) >= 8 and c1.get("nB", 0) >= 8) and not c1_estructura:
    veredicto = ("**NO (provisional)** — con la semilla muy usada, NO emergió estructura (sin roles claros ni "
                 "atractor metabólico): pegamento sin estructura. La frontera de Cosmogénesis es honda.")
else:
    veredicto = "**INCONCLUSO** — la semilla no se emitió lo suficiente por AMBOS organismos en C1. Subir DUR_COND."
# nota de honestidad sobre la distancia de estado
_nota_dist = ("Nota metodológica: la *distancia de estado en la emisión* crece en C1 **y** en C4 (deriva del "
              "estado global, ruido), por eso NO se usa como discriminador; los signos que SÍ separan estructura "
              "de ruido son los roles emergentes y el atractor IRDE, que colapsan bajo shuffle.")

with open(os.path.join(OUT, "resumen_triada.md"), "w") as f:
    f.write("# Tríada · ¿la palabra se diferenció por historia? — veredicto\n\n")
    f.write("> El sentido es la consecuencia, no el sonido. Aquí medimos si una palabra NEUTRA, sembrada "
            "idéntica en A y B, llegó a **usarse desde estados internos distintos** en cada uno, **sin** "
            "que nadie inyectara la diferencia.\n\n")
    f.write(f"## Veredicto\n\n{veredicto}\n\n> {_nota_dist}\n\n")
    f.write("## Especialización funcional (¿desde qué estado emiten la semilla?)\n\n")
    f.write("| cond | emis. A | emis. B | distancia A↔B | temprana | tardía | ¿crece? |\n|---|---|---|---|---|---|---|\n")
    for c in conds:
        d = res[c]; f.write(f"| {c} | {d['nA']} | {d['nB']} | {fx(d['diff'])} | {fx(d['dEarly'])} | "
                            f"{fx(d['dLate'])} | {'sí' if crece(c) else 'no'} |\n")
    f.write("\n*(distancia en desviaciones estándar; >0,5 = estados de emisión netamente distintos)*\n\n")
    f.write("## Roles emergentes y léxico\n\n")
    f.write("| cond | asim. uso (A−B) | creadas A/B | aprendidas A/B | emulaciones |\n|---|---|---|---|---|\n")
    for c in conds:
        d = res[c]; f.write(f"| {c} | {fx(d['rol_uso'])} | {fx(d['crA'],0)}/{fx(d['crB'],0)} | "
                            f"{fx(d['apA'],0)}/{fx(d['apB'],0)} | {d['emul']} |\n")
    f.write("\n## Atractor IRDE (mecanismo de Gemini: ¿especializar BAJA el riesgo?)\n\n")
    f.write("| cond | IRDE@semilla A | IRDE global A | IRDE@semilla B | IRDE global B |\n|---|---|---|---|---|\n")
    for c in conds:
        d = res[c]; f.write(f"| {c} | {fx(d['irde_seedA'],4)} | {fx(d['irde_globA'],4)} | "
                            f"{fx(d['irde_seedB'],4)} | {fx(d['irde_globB'],4)} |\n")
    f.write("\n*(si IRDE@semilla < IRDE global, usar la palabra deja al organismo en menor riesgo = la "
            "diferenciación se 'paga' metabólicamente)*\n\n")
    f.write("## Baseline de simetría (C2 = díada sin palabra-tercero)\n\n")
    for c in conds:
        f.write(f"- {c}: divergencia general de estado A↔B = {fx(res[c]['sim_global'])}\n")

print("OK · veredicto en resumen_triada.md ·", OUT)
print(veredicto.replace("**", ""))

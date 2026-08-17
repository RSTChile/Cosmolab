"""
CS066 confirmatorio Nivel 1 — ajuste del exponente diam~N^(1/d) con umbrales PRE-REGISTRADOS.
Lee las celdas cs066conf_k{k}_N{N}.csv, ajusta log(diam) vs log(N) por cada k (brazo local y barajado),
y adjudica contra el pre-registro de DISENO_CS066_confirmatorio_Nivel1_CS.md:
  CONFIRMA tejido 3D-espacial : slope ∈ [0.29,0.40] (d∈[2.5,3.5]) Y R²>0.9 Y monótona (diam crece con N).
  TEJIDO DÉBIL                : slope < 0.15 o no-monótona.
  NULO                        : barajado da el mismo slope que local (no específico).
Corre parcial o completo (usa las celdas que existan). No acomoda: imprime el veredicto mecánico.
"""
import csv, glob, math, os
from collections import defaultdict
import statistics as st

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
KS = [3, 4, 5, 6, 8, 10]
NS = [1500, 2500, 3500, 5000]
ARMS = ["local", "sin_local", "local_barajado", "local_marco_congelado"]

# cell[(k,N)][arm][field] = [valores]
cell = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
ncell = {}
for k in KS:
    for N in NS:
        f = os.path.join(_HERE, f"cs066conf_k{k}_N{N}.csv")
        if not os.path.exists(f):
            continue
        n = 0
        with open(f) as fh:
            for r in csv.DictReader(fh):
                a = r["arm"]
                for fld in ("diam_fin", "d_s", "clustering", "gigante", "n_ejes"):
                    v = r[fld]
                    try:
                        x = float(v)
                        if not math.isnan(x):
                            cell[(k, N)][a][fld].append(x)
                    except ValueError:
                        pass
                if a == "local":
                    n += 1
        ncell[(k, N)] = n

def mean(k, N, arm, fld):
    xs = cell[(k, N)][arm][fld]
    return st.mean(xs) if xs else float("nan")

def fit(k, arm):
    """log(diam) vs log(N) sobre los N disponibles para este k. Devuelve (slope, R2, monot, pts)."""
    pts = [(N, mean(k, N, arm, "diam_fin")) for N in NS if not math.isnan(mean(k, N, arm, "diam_fin"))]
    if len(pts) < 3:
        return None
    xs = [math.log(N) for N, _ in pts]; ys = [math.log(d) for _, d in pts]
    mx = st.mean(xs); my = st.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    inter = my - slope * mx
    ss_res = sum((y - (slope * x + inter)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    diams = [d for _, d in pts]
    monot = all(diams[i + 1] >= diams[i] - 1e-9 for i in range(len(diams) - 1))
    return slope, r2, monot, pts

print("=" * 92)
print("CS066 CONFIRMATORIO NIVEL 1 — exponente diam~N^(1/d) por k_local (pre-registro fijo)")
print("=" * 92)
print("celdas presentes (parches 'local' por celda):")
for k in KS:
    row = "  k=%2d: " % k + "  ".join(f"N{N}={ncell.get((k,N),0)}" for N in NS)
    print(row)

print("\n%-8s %-16s %8s %8s %8s %10s %s" % ("k_local", "arm", "slope", "d=1/slope", "R2", "monótona", "diam por N"))
verdict_por_k = {}
for k in KS:
    for arm in ("local", "local_barajado"):
        r = fit(k, arm)
        if r is None:
            print("%-8d %-16s   (faltan N; celdas incompletas)" % (k, arm)); continue
        slope, r2, monot, pts = r
        d = (1 / slope) if slope > 1e-6 else float("inf")
        diams = " ".join(f"{N}:{dd:.1f}" for N, dd in pts)
        print("%-8d %-16s %8.3f %8.1f %8.3f %10s   %s" % (k, arm, slope, d, r2, "sí" if monot else "NO", diams))
        if arm == "local":
            verdict_por_k[k] = (slope, r2, monot)

print("\n" + "=" * 92)
print("ADJUDICACIÓN pre-registrada (régimen decisivo k∈{3,4,5,6}):")
print("=" * 92)
fuerte = [k for k in (3, 4, 5, 6) if k in verdict_por_k]
conf = deb = 0
for k in fuerte:
    slope, r2, monot = verdict_por_k[k]
    if 0.29 <= slope <= 0.40 and r2 > 0.9 and monot:
        tag = "CONFIRMA tejido d≈3 (slope∈[0.29,0.40], R²>0.9, monótona)"; conf += 1
    elif slope < 0.15 or not monot:
        tag = "TEJIDO DÉBIL (slope<0.15 o no-monótona)"; deb += 1
    else:
        tag = "ZONA INTERMEDIA (slope 0.15–0.29: tejido parcial, d>3.5)"
    # especificidad vs barajado
    rb = fit(k, "local_barajado")
    esp = ""
    if rb:
        esp = f" · barajado slope={rb[0]:.3f} (Δ={slope-rb[0]:+.3f})"
    print(f"  k={k}: {tag}{esp}")
if fuerte:
    print(f"\n  RESUMEN régimen fuerte: {conf} celdas CONFIRMA · {deb} DÉBIL · de {len(fuerte)} con datos.")
    print("  (El (B) global de CS066 —espacio ≠ direcciones— NO depende de esto; solo endurece/matiza 'hay tejido'.)")
else:
    print("  aún sin celdas de k fuerte completas (≥3 N). Correr más malla.")

#!/usr/bin/env python
"""
VALIDACION kappa_P / kappa_Delta contra controles CON MALLA CAUSAL.
Re-analisis puro de archivos .sink de Phantom ya existentes. NO corre nada.

Definiciones (identicas a la tabla publicada en
DISENO_EXPERIMENTOS_NODOS_ABIERTOS_desde_2.5.5_CS.md §2):

  kappa_P      = media_sobre_sumideros[ (t_ultimo - t_primero) / (T_fin - T_ini) ]
                 donde T_ini/T_fin = primer/ultimo instante presente en el .sink
  kappa_P_abs  = misma vida, pero normalizada por tmax=0.5 (lectura literal de
                 "duracion total de la corrida"); se reporta como control de definicion
  kappa_D      = media_sobre_sumideros[ masa_final / masa_inicial ]
  kappa_D_alt  = media_sobre_sumideros[ masa_final - masa_inicial ]  (masa acretada)
  kappa_V      = media_sobre_sumideros[ acrecion(ultimo tercio de vida) /
                                        acrecion(primer tercio de vida) ]
  frac_sobrev  = fraccion de sumideros nacidos que siguen presentes en el ultimo
                 instante del archivo (C-N1.3: ¿la persistencia FILTRA?)

Columnas del .sink de Phantom:
  0 time  1 x  2 y  3 z  4 mass ... 11 macc ... 18 sinkID  19 nptmass
"""
import numpy as np, os, csv, math
from itertools import combinations

RAIZ = "/Users/alexis/phantom_cs073"
TMAX = 0.5
TOL_TMAX = 1e-6

BRAZOS = {
    "REAL":        [("ic_real",        f"{RAIZ}/bateria_n2000/ic_real")],
    "REAL_extra":  [(f"ic_real_s{s}",  f"{RAIZ}/bateria_real_extra_n2000/ic_real_s{s}") for s in range(301, 306)],
    "NULL_orig":   [(f"ic_null{i}",    f"{RAIZ}/bateria_n2000/ic_null{i}") for i in range(1, 9)],
    "NULL3":       [(f"ic_null3_s{s}", f"{RAIZ}/bateria_null3_n2000/ic_null3_s{s}") for s in range(501, 509)],
    "RANDOM_ER":   [(f"ic_random_s{s}",f"{RAIZ}/bateria_grafo_random_n2000/ic_random_s{s}") for s in range(701, 709)],
    "NULL4":       [(f"ic_null4_s{s}", f"{RAIZ}/bateria_null4_n2000/ic_null4_s{s}") for s in range(601, 604)],
    "NULL5":       [(f"ic_null5_s{s}", f"{RAIZ}/bateria_null5_n2000/ic_null5_s{s}") for s in range(801, 803)],
}

def sinkpath(d):
    for n in ("cosmog01.sink", "cosmog1.sink"):
        p = os.path.join(d, n)
        if os.path.exists(p):
            return p
    c = [f for f in os.listdir(d) if f.endswith(".sink")] if os.path.isdir(d) else []
    return os.path.join(d, c[0]) if c else None


def analiza(d):
    """Devuelve dict de metricas de una corrida, o {'error': msg}."""
    p = sinkpath(d)
    if p is None:
        return {"error": "sin archivo .sink"}
    try:
        a = np.loadtxt(p, skiprows=2)
    except Exception as e:
        return {"error": f"ilegible: {e}"}
    if a.ndim != 2 or a.shape[0] < 2:
        return {"error": "archivo vacio o de una sola fila"}

    t, m, macc, sid = a[:, 0], a[:, 4], a[:, 11], a[:, 18].astype(int)
    T0, T1 = t.min(), t.max()
    span = T1 - T0
    ids = np.unique(sid)

    vidas, ratios, acret, kv, sobreviven = [], [], [], [], 0
    m_ini_tot, m_fin_tot = 0.0, 0.0
    for s in ids:
        k = sid == s
        ts, ms = t[k], m[k]
        o = np.argsort(ts)
        ts, ms = ts[o], ms[o]
        tb, td = ts[0], ts[-1]          # nacimiento, ultima aparicion
        vidas.append(td - tb)
        m0, m1 = ms[0], ms[-1]
        m_ini_tot += m0
        m_fin_tot += m1
        # GUARDA: masa inicial siempre > 0 en Phantom; si no, se excluye y se avisa
        ratios.append(m1 / m0 if m0 > 0 else np.nan)
        acret.append(m1 - m0)
        if td >= T1 - 1e-9:
            sobreviven += 1
        # kappa_V: acrecion en el ultimo tercio de vida / primer tercio
        vida = td - tb
        if vida > 0:
            c1, c2 = tb + vida / 3.0, tb + 2.0 * vida / 3.0
            m_c1 = np.interp(c1, ts, ms)
            m_c2 = np.interp(c2, ts, ms)
            a1 = m_c1 - m0          # acrecion primer tercio
            a3 = m1 - m_c2          # acrecion ultimo tercio
            kv.append(a3 / a1 if a1 > 0 else np.nan)
        else:
            kv.append(np.nan)

    vidas = np.array(vidas)
    return {
        "n_sinks": len(ids),
        "T0": T0, "T1": T1,
        "llego_tmax": abs(T1 - TMAX) < 1e-6,
        "kappa_P": float(np.mean(vidas) / span) if span > 0 else np.nan,
        "kappa_P_abs": float(np.mean(vidas) / TMAX),
        "t_nac_medio": float(np.mean(t[np.array([np.argmax(sid == s) for s in ids])])) if len(ids) else np.nan,
        "kappa_D": float(np.nanmean(ratios)),
        "kappa_D_alt": float(np.nanmean(acret)),
        "kappa_V": float(np.nanmean(kv)),
        "masa_ini_tot": float(m_ini_tot),
        "masa_fin_tot": float(m_fin_tot),
        "frac_sobrev": sobreviven / len(ids) if len(ids) else np.nan,
        "n_sobrev": sobreviven,
    }


# --------------------------------------------------------------------------
filas = []
errores = []
for brazo, corridas in BRAZOS.items():
    for nombre, d in corridas:
        r = analiza(d)
        if "error" in r:
            errores.append((brazo, nombre, r["error"]))
            continue
        r["brazo"], r["corrida"] = brazo, nombre
        filas.append(r)

print("=" * 100)
print("CORRIDAS EXCLUIDAS:", errores if errores else "ninguna")
print("=" * 100)

# tabla por corrida
hdr = ("brazo", "corrida", "n_sinks", "n_sobrev", "frac_sobrev", "T0", "T1", "llego_tmax",
       "kappa_P", "kappa_P_abs", "kappa_D", "kappa_D_alt", "kappa_V",
       "masa_ini_tot", "masa_fin_tot", "t_nac_medio")
print(f"{'brazo':<11}{'corrida':<16}{'n':>3}{'sob':>5}{'frac':>6}{'T0':>8}{'T1':>7}{'kP':>8}{'kPabs':>8}{'kD':>7}{'kDalt':>9}{'kV':>8}{'M_fin':>10}")
for f in filas:
    print(f"{f['brazo']:<11}{f['corrida']:<16}{f['n_sinks']:>3}{f['n_sobrev']:>5}{f['frac_sobrev']:>6.2f}"
          f"{f['T0']:>8.3f}{f['T1']:>7.3f}{f['kappa_P']:>8.4f}{f['kappa_P_abs']:>8.4f}"
          f"{f['kappa_D']:>7.3f}{f['kappa_D_alt']:>9.1f}{f['kappa_V']:>8.3f}{f['masa_fin_tot']:>10.1f}")

# CSV
CSV = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/VALIDACION_kappaP_kappaDelta_por_corrida.csv"
with open(CSV, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(hdr))
    w.writeheader()
    for f in filas:
        w.writerow({k: f[k] for k in hdr})
print("\nCSV ->", CSV)

# --------------------------------------------------------------------------
# z de REAL (n=1, la corrida original) contra cada brazo control
def arr(brazo, campo):
    return np.array([f[campo] for f in filas if f["brazo"] == brazo], float)

real1 = [f for f in filas if f["brazo"] == "REAL"][0]
METR = ["kappa_P", "kappa_P_abs", "kappa_D", "kappa_D_alt", "kappa_V", "masa_fin_tot", "n_sinks", "frac_sobrev"]
CTRL = ["NULL_orig", "NULL3", "RANDOM_ER", "NULL4", "NULL5"]

print("\n" + "=" * 100)
print("Z DE LA CORRIDA REAL ORIGINAL (n=1) CONTRA CADA BRAZO CONTROL")
print("=" * 100)
print(f"{'metrica':<14}{'REAL':>10}" + "".join(f"{c:>26}" for c in CTRL))
for mm in METR:
    fila = f"{mm:<14}{real1[mm]:>10.4f}"
    for c in CTRL:
        v = arr(c, mm)
        mu, sd = v.mean(), v.std(ddof=1) if len(v) > 1 else np.nan
        z = (real1[mm] - mu) / sd if sd and sd > 0 else np.nan
        fila += f"{mu:>10.3f}±{sd:<7.3f}z={z:>6.2f}"
    print(fila)

# --------------------------------------------------------------------------
# REAL con las 6 semillas (original + 5 extra) vs controles: Mann-Whitney exacta
def mw_exacta(x, y):
    """U de Mann-Whitney exacta por enumeracion; devuelve (U, p dos colas, p minimo)."""
    n1, n2 = len(x), len(y)
    todos = np.concatenate([x, y])
    rangos = todos.argsort().argsort().astype(float) + 1
    # empates -> rangos promedio
    for v in np.unique(todos):
        k = todos == v
        if k.sum() > 1:
            rangos[k] = rangos[k].mean()
    R1 = rangos[:n1].sum()
    U = R1 - n1 * (n1 + 1) / 2.0
    # distribucion nula exacta por permutacion de asignaciones
    idx = list(range(n1 + n2))
    cnt = 0
    tot = 0
    Us = []
    for comb in combinations(idx, n1):
        r = rangos[list(comb)].sum()
        Us.append(r - n1 * (n1 + 1) / 2.0)
        tot += 1
    Us = np.array(Us)
    Umed = n1 * n2 / 2.0
    p2 = np.mean(np.abs(Us - Umed) >= abs(U - Umed))
    pmin = 2.0 / tot
    return U, p2, pmin


print("\n" + "=" * 100)
print("REAL n=6 (original + 5 semillas extra) vs CONTROLES  —  Mann-Whitney exacta")
print("=" * 100)
for mm in METR:
    x = np.array([f[mm] for f in filas if f["brazo"] in ("REAL", "REAL_extra")], float)
    linea = f"{mm:<14} REAL={x.mean():.4f}±{x.std(ddof=1):.4f} (n={len(x)})"
    for c in CTRL:
        y = arr(c, mm)
        if len(y) < 2:
            continue
        mu, sd = y.mean(), y.std(ddof=1)
        z = (x.mean() - mu) / sd if sd > 0 else np.nan
        U, p2, pmin = mw_exacta(x, y)
        linea += f" | {c}: {mu:.4f}±{sd:.4f} z={z:.2f} p={p2:.5f}(min {pmin:.5f})"
    print(linea)

# --------------------------------------------------------------------------
# GUARDA 2: identidades algebraicas — correlacion kappa vs masa
from scipy.stats import spearmanr, pearsonr
print("\n" + "=" * 100)
print("GUARDA 2 · ¿SON LAS KAPPA UNA COPIA DE LA MASA?  (todas las corridas juntas, n=%d)" % len(filas))
print("=" * 100)
M = np.array([f["masa_fin_tot"] for f in filas])
for mm in ["kappa_P", "kappa_P_abs", "kappa_D", "kappa_D_alt", "kappa_V", "t_nac_medio"]:
    v = np.array([f[mm] for f in filas])
    ok = np.isfinite(v) & np.isfinite(M)
    rs, ps = spearmanr(v[ok], M[ok])
    rp, pp = pearsonr(v[ok], M[ok])
    print(f"  {mm:<14} vs masa_fin_tot :  Spearman rho={rs:+.3f} (p={ps:.2e})   Pearson r={rp:+.3f}")
# kappa_P vs tiempo de nacimiento medio
tn = np.array([f["t_nac_medio"] for f in filas])
kp = np.array([f["kappa_P"] for f in filas])
rs, ps = spearmanr(kp, tn)
print(f"  kappa_P        vs t_nacimiento_medio : Spearman rho={rs:+.3f} (p={ps:.2e})")

# --------------------------------------------------------------------------
# GUARDA 1: tautologia — ¿muere alguna vez algun sumidero?
print("\n" + "=" * 100)
print("GUARDA 1 · TAUTOLOGIA: ¿podia kappa_P salir distinto? ¿muere algun sumidero?")
print("=" * 100)
tot_s = sum(f["n_sinks"] for f in filas)
tot_v = sum(f["n_sobrev"] for f in filas)
print(f"  sumideros nacidos en TODOS los brazos: {tot_s}")
print(f"  sumideros presentes en el ultimo instante: {tot_v}  ({100.0*tot_v/tot_s:.2f}%)")
print(f"  sumideros que se APAGAN: {tot_s - tot_v}")
for b in BRAZOS:
    fs = [f for f in filas if f["brazo"] == b]
    if not fs:
        continue
    ns, nv = sum(f["n_sinks"] for f in fs), sum(f["n_sobrev"] for f in fs)
    print(f"    {b:<12} nacidos={ns:>3}  sobreviven={nv:>3}  frac={nv/ns:.4f}")

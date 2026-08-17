"""Re-analisis C-N2.5.5: recupera las cantidades CONTINUAS de CS070 detras del booleano direccion_real."""
from __future__ import annotations
import json, csv, os, sys
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis")
from cs067_habitacion_completa import cuenta_ejes_gap

AQUI = os.path.dirname(os.path.abspath(__file__))
DEST = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
ARMS = ["semilla_coherente", "semilla_barajada", "sin_semilla", "semilla_sustrato_local"]

d = json.load(open(os.path.join(AQUI, "cs070_espectros.json")))
orig = {(r["arm"], r["N"], r["seed"]): r for r in
        json.load(open(os.path.join(DEST, "cs070_tanda_resultados.json")))}

# ---- 0. verificacion de determinismo: la re-corrida reproduce la tanda original? ----
dif = [k for r in d for k in [(r["arm"], r["N"], r["seed"])]
       if abs(orig[k]["pico_medio"] - r["pico_medio"]) > 1e-12 or orig[k]["n_ejes"] != r["n_ejes"]]
print(f"[determinismo] corridas que NO reproducen la tanda original: {len(dif)}/{len(d)}")

# ---- 1. cantidades continuas por corrida ----
filas = []
for r in d:
    ev = np.array(r["ev"], float)
    lam = np.clip(ev, 0, None); s = lam.sum(); p = lam / s
    K = r["K_sorteado"]; D = r["D"]
    n_pobl = int((p >= 0.02).sum())            # ejes con >=2% de los nodos
    n_vivos = int((p > 0).sum())               # ejes con al menos 1 nodo
    PR = 1.0 / float((p ** 2).sum())
    # CONTRAFACTUAL: mismo juez, pero con el embedding D=8 de CS067 (rellenar con ceros)
    ev8 = np.concatenate([lam, np.zeros(8 - D)]) if D < 8 else lam
    nej8, PR8, gap8, rthr8 = cuenta_ejes_gap(ev8)
    filas.append(dict(arm=r["arm"], N=r["N"], seed=r["seed"], K_sorteado=K, D_embedding=D,
                      pico_medio=r["pico_medio"], frac_picados=r["frac_picados"],
                      certificado=int(r["certificado"]), n_ejes=r["n_ejes"], PR=round(PR, 4),
                      gap_interno=r["gap_interno"], n_pobl=n_pobl, n_vivos=n_vivos,
                      rango_pleno=int(n_pobl == D), peso_semilla=r["peso_semilla"],
                      direccion_real=int(r["direccion_real"]),
                      n_ejes_D8=nej8, direccion_real_D8=int(r["certificado"] and nej8 > 1),
                      p_ordenado=";".join(f"{x:.5f}" for x in p)))

csv_out = os.path.join(DEST, "REANALISIS_CS070_continuas.csv")
with open(csv_out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(filas[0].keys())); w.writeheader(); w.writerows(filas)
print(f"[csv] {csv_out}  ({len(filas)} filas)")

pm = np.array([f["pico_medio"] for f in filas])
ne = np.array([f["n_ejes"] for f in filas])
npo = np.array([f["n_pobl"] for f in filas])
cert = np.array([f["certificado"] for f in filas])
rp = np.array([f["rango_pleno"] for f in filas])
ne8 = np.array([f["n_ejes_D8"] for f in filas])
dr8 = np.array([f["direccion_real_D8"] for f in filas])

print("\n=== 1. LAS DOS CONDICIONES, POR SEPARADO (96 corridas) ===")
print(f"certificado (pico_medio>0.85)  : {cert.sum()}/96 = {cert.mean():.3f}")
print(f"n_ejes>1                       : {(ne>1).sum()}/96 = {(ne>1).mean():.3f}")
print(f"AMBAS (direccion_real)         : {((ne>1)&(cert==1)).sum()}/96")
print(f"pico_medio  min/med/max        : {pm.min():.4f} / {np.median(pm):.4f} / {pm.max():.4f}")
print(f"n_pobl (ejes con >=2% nodos)   : {dict(zip(*np.unique(npo, return_counts=True)))}")
print(f"rango pleno (n_pobl==D)        : {rp.sum()}/96")
print(f"n_ejes                          : {dict(zip(*np.unique(ne, return_counts=True)))}")

print("\n=== 2. ANTI-CORRELACION entre las dos condiciones ===")
rho, pv = spearmanr(pm, npo)
print(f"Spearman(pico_medio, n_pobl) TODAS  : rho={rho:+.3f} p={pv:.2e}")
sub = ~rp.astype(bool)
if sub.sum() > 2:
    r2, p2 = spearmanr(pm[sub], ne[sub])
    print(f"Spearman(pico_medio, n_ejes) solo donde el contador esta ACTIVO (n={sub.sum()}): rho={r2:+.3f} p={p2:.2e}")
print(f"pico_medio en corridas con n_ejes>1 : {pm[ne>1]}")
print(f"pico_medio en corridas con n_ejes==1: {pm[ne==1]}")

print("\n=== 3. CONTRAFACTUAL: mismo juez, embedding D=8 de CS067 ===")
print(f"n_ejes_D8                       : {dict(zip(*np.unique(ne8, return_counts=True)))}")
print(f"n_ejes_D8>1                     : {(ne8>1).sum()}/96")
print(f"direccion_real con D=8          : {dr8.sum()}/96 = {dr8.mean():.3f}")
for arm in ARMS:
    idx = np.array([f["arm"] == arm for f in filas])
    print(f"  {arm:24s}: cert={cert[idx].sum():2d}/24  n_ejes>1={(ne[idx]>1).sum():2d}/24  "
          f"D8: n_ejes>1={(ne8[idx]>1).sum():2d}/24  dir_real_D8={dr8[idx].sum():2d}/24")

print("\n=== 4. por brazo x N: medianas continuas ===")
for arm in ARMS:
    for N in [900, 1500, 2500]:
        idx = np.array([f["arm"] == arm and f["N"] == N for f in filas])
        print(f"  {arm:24s} N={N:5d}: pico med={np.median(pm[idx]):.4f} [{pm[idx].min():.3f},{pm[idx].max():.3f}] "
              f"n_pobl med={np.median(npo[idx]):.1f} K={sorted(set(f['K_sorteado'] for f,i in zip(filas,idx) if i))}")

# ---- graficos ----
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
ax = axes[0, 0]
for arm, c in zip(ARMS, ["#2b6cb0", "#c05621", "#2f855a", "#6b46c1"]):
    v = pm[np.array([f["arm"] == arm for f in filas])]
    ax.hist(v, bins=np.arange(0.40, 1.02, 0.04), alpha=0.55, label=arm, color=c)
ax.axvline(0.85, color="k", ls="--", lw=2); ax.text(0.855, ax.get_ylim()[1]*0.9, "umbral 0.85", fontsize=8)
ax.set_xlabel("pico_medio (= media de conf del Potts)"); ax.set_ylabel("corridas")
ax.set_title("A. pico_medio: la condicion 'certificado' SI se cumple (18/96)"); ax.legend(fontsize=7)

ax = axes[0, 1]
vals, cnts = np.unique(ne, return_counts=True)
ax.bar([str(v) for v in vals], cnts, color="#c05621")
ax.set_xlabel("n_ejes (contador cuenta_ejes_gap)"); ax.set_ylabel("corridas")
ax.set_title("B. n_ejes: 87/96 en CERO por la guarda de rango pleno")
for x, c in zip(range(len(vals)), cnts):
    ax.text(x, c + 1, str(c), ha="center", fontsize=9)

ax = axes[1, 0]
ax.scatter(pm[rp == 1], npo[rp == 1], c="#c05621", s=45, label="rango PLENO -> n_ejes=0 forzado")
ax.scatter(pm[rp == 0], npo[rp == 0], c="#2b6cb0", s=45, marker="s", label="contador activo")
ax.axvline(0.85, color="k", ls="--"); ax.axhline(1.5, color="gray", ls=":")
ax.set_xlabel("pico_medio"); ax.set_ylabel("n_pobl (ejes con >=2% de nodos)")
ax.set_title("C. las dos condiciones: cuadrante superior-derecho poblado,\npero marcado 'rango pleno'")
ax.legend(fontsize=7)

ax = axes[1, 1]
w = 0.35; x = np.arange(len(ARMS))
a = [ (np.array([f["arm"]==arm for f in filas]) & (ne>1)).sum() for arm in ARMS ]
b = [ dr8[np.array([f["arm"]==arm for f in filas])].sum() for arm in ARMS ]
ax.bar(x - w/2, a, w, label="n_ejes>1 (juez tal cual)", color="#c05621")
ax.bar(x + w/2, b, w, label="direccion_real con embedding D=8 (CS067)", color="#2f855a")
ax.set_xticks(x); ax.set_xticklabels([a_[:14] for a_ in ARMS], rotation=15, fontsize=7)
ax.set_ylabel("corridas de 24"); ax.set_title("D. contrafactual del embedding"); ax.legend(fontsize=7)

fig.suptitle("Re-analisis C-N2.5.5 — las cantidades continuas que el booleano direccion_real tapaba", fontsize=12)
fig.tight_layout()
png = os.path.join(DEST, "REANALISIS_CS070_continuas.png")
fig.savefig(png, dpi=140)
print(f"\n[png] {png}")

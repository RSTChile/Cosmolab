"""
cs090_fase6_outliers_diagnostico_diam_gigante.py -- FASE VI, diagnostico del PASO 1 (lectura, sin Phantom).

DE DONDE SALE ESTE SCRIPT: el Paso 1 (cs090_fase6_outliers_paso1_curvas.py) mostro que las 3 reglas de
pendiente muy negativa tienen diametro = 1 en la escala nativa b=1, con la medicion cayendo sobre una
componente conexa de TAMANO 2 (un par de nodos suelto), mientras la componente gigante de ese mismo grafo
tiene 84-95% de los nodos. La causa es el nodo de arranque de `_diam` en cs055 (congelado):

    src = next((i for i in range(N) if adj[i]), 0)   # PRIMER nodo con vecinos, por indice

Si el nodo de indice mas bajo que conserva alguna arista quedo en un pedacito colgante, el "diametro"
reportado es el de ese pedacito, no el del grafo. Al agrupar (b>=2) las cajas se renumeran y la medicion
suele caer en la componente grande, asi que el diametro SALTA hacia arriba -- y el ajuste lineal sobre los
5 puntos sale negativo por ese primer punto, no por la geometria.

QUE HACE ESTE SCRIPT: vuelve a medir el diametro de cada escala SOBRE LA COMPONENTE MAS GRANDE (eleccion
determinista, independiente de como esten numerados los nodos), recalcula la pendiente con ese diametro
corregido, y compara:
   (a) cuantas de las reglas medidas tenian la medicion "descarrilada" (la componente donde cayo `_diam`
       es mucho mas chica que la gigante), y
   (b) si la correlacion pendiente<->fraccion de masa en Phantom (el analisis de
       FASE6_reanalisis_azar_continuo_CS.md 3.3) cambia al usar la pendiente corregida.

IMPORTANTE -- esto es un DIAGNOSTICO, no un reemplazo: `cs055_proceso_acoplado.py`, `cs090_fase5_motor.py`
y `cs090_fase5_clasificador.py` NO se tocan; la pendiente oficial de todas las reglas sigue siendo la que
esta en los CSV. Aca solo se calcula una medicion alternativa para ver que explica.

Poblacion medida: las 76 reglas distintas que corrieron Phantom en Fase V-B (las 80 filas de
cs090_fase5b_TOTAL_40pares.csv tienen 4 reglas reutilizadas en dos pares) + las 14 reglas de pendiente
muy negativa detectadas en el Paso 2. Salidas: cs090_fase6_outliers_diam_gigante.csv y
cs090_fase6_outliers_diam_gigante.png. No corre Phantom. No declara veredicto.
"""
from __future__ import annotations
import csv
import sys
import time
from collections import deque

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, HERE)

import cs090_fase5_generador as GEN
import cs090_fase5_motor as MOT
import cs080_renormalizacion as CS80
from cs090_fase5_clasificador import _pendiente_loglog

N_PILOTO, N_SWEEPS, ESCALAS = 2000, 14, (1, 2, 4, 8, 16)


def _componentes_lista(adj, N):
    """Devuelve la lista de componentes (cada una, lista de nodos) de los nodos NO aislados."""
    seen = np.zeros(N, bool)
    comps = []
    for s in range(N):
        if seen[s] or not adj[s]:
            continue
        q = deque([s]); seen[s] = True; c = [s]
        while q:
            u = q.popleft()
            for w in adj[u]:
                if not seen[w]:
                    seen[w] = True; q.append(int(w)); c.append(int(w))
        comps.append(c)
    return comps


def _diam_doble_bfs(adj, nodos):
    """Mismo algoritmo de doble-BFS que `_diam` de cs055, pero arrancando en un nodo ELEGIDO (no en el
    de indice mas bajo del grafo entero) y recorriendo solo su componente."""
    if len(nodos) < 2:
        return 0
    def lejano(s):
        d = {s: 0}; q = deque([s]); f = s
        while q:
            u = q.popleft()
            for w in adj[u]:
                if w not in d:
                    d[w] = d[u] + 1; q.append(int(w))
                    if d[w] > d[f]:
                        f = w
        return f, d[f]
    a, _ = lejano(nodos[0])
    _, dd = lejano(a)
    return dd


def curva_corregida(seed, rule_id):
    """Devuelve dict con la pendiente original (medicion `_diam` tal cual) y la corregida (diametro de la
    componente MAS GRANDE en cada escala), mas el diagnostico de si la medicion original descarrilo."""
    p = GEN.generar_regla("A2", "B0", "C2", idx=0, seed=seed)
    rng = np.random.default_rng(p["seed"] * 5000 + N_PILOTO)
    sustrato = MOT.construir_A2(N_PILOTO, rng, p)
    sustrato = MOT.dinamica_B0(sustrato, p, rng, N_SWEEPS, "C2")
    m = MOT.medir(sustrato, p, rng)
    adj_real = m["adj_final"]

    n_cajas_l, diam_orig, diam_gig, tam_src_l, tam_gig_l = [], [], [], [], []
    for b in ESCALAS:
        if b == 1:
            adj_g, n_cajas = adj_real, N_PILOTO
        else:
            rng_b = np.random.default_rng(p["seed"] * 7000 + b * 31)
            asign, n_cajas = CS80.cajas_bfs(adj_real, N_PILOTO, b, rng_b)
            adj_g = CS80.grafo_grueso(adj_real, N_PILOTO, asign, n_cajas)
        d_orig = float(MOT._diam(adj_g, n_cajas)) if n_cajas > 1 else float("nan")
        comps = _componentes_lista(adj_g, n_cajas)
        if not comps:
            d_gig, tam_gig, tam_src = float("nan"), 0, 0
        else:
            gig = max(comps, key=len)
            d_gig = float(_diam_doble_bfs(adj_g, gig))
            tam_gig = len(gig)
            src = next((i for i in range(n_cajas) if adj_g[i]), None)
            tam_src = next((len(c) for c in comps if src in set(c)), 0)
        n_cajas_l.append(n_cajas); diam_orig.append(d_orig); diam_gig.append(d_gig)
        tam_src_l.append(tam_src); tam_gig_l.append(tam_gig)

    pend_orig = _pendiente_loglog(n_cajas_l, diam_orig)
    pend_corr = _pendiente_loglog(n_cajas_l, diam_gig)
    # "descarrilo" = en alguna escala la medicion original cayo en una componente < 10% de la gigante
    descarrila = [i for i in range(len(ESCALAS))
                  if tam_gig_l[i] > 0 and tam_src_l[i] < 0.10 * tam_gig_l[i]]
    return dict(rule_id=rule_id, seed=seed, pend_orig=pend_orig, pend_corr=pend_corr,
                giant_b1=m["giant"], n_aristas_b1=m["n_aristas"],
                K=p["K"], J=p["J"], noise=p["noise"], meandeg=p["meandeg"], kcap=p["kcap"],
                diam_orig=diam_orig, diam_gig=diam_gig, n_cajas=n_cajas_l,
                tam_comp_src=tam_src_l, tam_comp_gigante=tam_gig_l,
                escalas_descarriladas=len(descarrila),
                descarrila_en_b1=(0 in descarrila))


def main():
    # poblacion 1: las reglas que corrieron Phantom (con su fraccion de masa)
    fm, clase_of = {}, {}
    with open(f"{HERE}/cs090_fase5b_TOTAL_40pares.csv") as f:
        for row in csv.DictReader(f):
            fm[int(row["seed"])] = float(row["fraccion_masa_en_sumideros"])
            clase_of[int(row["seed"])] = row["clase"]
    # poblacion 2: las 14 de pendiente muy negativa
    negs = {}
    with open(f"{HERE}/cs090_fase6_outliers_430_todas.csv") as f:
        for row in csv.DictReader(f):
            if float(row["pendiente"]) < -0.3:
                negs[int(row["seed"])] = row["rule_id"]
    # nombres de regla por seed (desde la tabla de 430)
    nombre = {}
    with open(f"{HERE}/cs090_fase6_outliers_430_todas.csv") as f:
        for row in csv.DictReader(f):
            nombre[int(row["seed"])] = row["rule_id"]

    objetivo = sorted(set(fm) | set(negs))
    print(f"[diagnostico] {len(objetivo)} reglas a re-medir "
          f"({len(fm)} con Phantom, {len(negs)} de pendiente muy negativa, "
          f"{len(set(fm) & set(negs))} en ambos grupos)", flush=True)

    filas, t0 = [], time.time()
    for k, s in enumerate(objetivo):
        d = curva_corregida(s, nombre.get(s, f"seed{s}"))
        d["fraccion_masa"] = fm.get(s, float("nan"))
        d["corrio_phantom"] = s in fm
        d["clase"] = clase_of.get(s, "")
        filas.append(d)
        if (k + 1) % 15 == 0:
            print(f"   ...{k+1}/{len(objetivo)} ({time.time()-t0:.0f}s)", flush=True)

    n_desc = sum(1 for d in filas if d["descarrila_en_b1"])
    print(f"\n[cuantas mediciones descarrilaron en b=1] {n_desc}/{len(filas)} "
          f"(la componente donde cayo _diam es <10% de la gigante)")
    print("   de las de pendiente ORIGINAL < -0.3: "
          f"{sum(1 for d in filas if d['pend_orig'] < -0.3 and d['descarrila_en_b1'])}/"
          f"{sum(1 for d in filas if d['pend_orig'] < -0.3)}")
    print("   de las de pendiente ORIGINAL >= -0.3: "
          f"{sum(1 for d in filas if d['pend_orig'] >= -0.3 and d['descarrila_en_b1'])}/"
          f"{sum(1 for d in filas if d['pend_orig'] >= -0.3)}")

    print("\n[las 14 de pendiente muy negativa: original -> corregida]")
    for d in sorted([x for x in filas if x["pend_orig"] < -0.3], key=lambda x: x["pend_orig"]):
        print(f"   {d['rule_id']:<24} pend_orig={d['pend_orig']:+.4f} -> pend_corr={d['pend_corr']:+.4f} "
              f"| diam_orig={[int(v) for v in d['diam_orig']]} diam_gigante={[int(v) for v in d['diam_gig']]} "
              f"| comp_src_b1={d['tam_comp_src'][0]} vs gigante_b1={d['tam_comp_gigante'][0]}")

    # ---- correlacion pendiente(original vs corregida) contra fraccion de masa, sobre las que corrieron ----
    ph = [d for d in filas if d["corrio_phantom"]]
    y = np.array([d["fraccion_masa"] for d in ph])
    print(f"\n[correlacion con la fraccion de masa en Phantom, n={len(ph)} reglas distintas]")
    for etq, x in (("pendiente ORIGINAL ", np.array([d["pend_orig"] for d in ph])),
                   ("pendiente CORREGIDA", np.array([d["pend_corr"] for d in ph]))):
        rho, pv = stats.spearmanr(x, y)
        r2 = np.corrcoef(x, y)[0, 1] ** 2
        print(f"   {etq}: Spearman rho={rho:+.4f} (p={pv:.2e})   R2 lineal={r2:.4f}")
    sin3 = [d for d in ph if d["pend_orig"] >= -0.3]
    x = np.array([d["pend_orig"] for d in sin3]); y2 = np.array([d["fraccion_masa"] for d in sin3])
    rho, pv = stats.spearmanr(x, y2)
    print(f"   pendiente ORIGINAL sin los 3 outliers (n={len(sin3)}): rho={rho:+.4f} (p={pv:.2e})  "
          f"R2 lineal={np.corrcoef(x,y2)[0,1]**2:.4f}")

    # ---- CSV ----
    campos = ["rule_id", "seed", "clase", "corrio_phantom", "fraccion_masa", "pend_orig", "pend_corr",
              "giant_b1", "n_aristas_b1", "K", "J", "noise", "meandeg", "kcap",
              "escalas_descarriladas", "descarrila_en_b1", "n_cajas", "diam_orig", "diam_gig",
              "tam_comp_src", "tam_comp_gigante"]
    with open(f"{HERE}/cs090_fase6_outliers_diam_gigante.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos); w.writeheader()
        for d in sorted(filas, key=lambda x: x["pend_orig"]):
            w.writerow({k: d[k] for k in campos})
    print(f"\n[csv] cs090_fase6_outliers_diam_gigante.csv ({len(filas)} filas)")

    # ---- grafico ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax = axes[0]
    xs = [d["pend_orig"] for d in filas]; ys = [d["pend_corr"] for d in filas]
    cs = ["#cc3311" if d["descarrila_en_b1"] else "#4477aa" for d in filas]
    ax.scatter(xs, ys, c=cs, s=26, alpha=0.8)
    lim = [min(xs + ys) - 0.1, max(xs + ys) + 0.1]
    ax.plot(lim, lim, "k:", lw=1)
    ax.set_xlabel("pendiente ORIGINAL (_diam tal cual)")
    ax.set_ylabel("pendiente CORREGIDA (diam de la componente gigante)")
    ax.set_title("rojo = la medicion original cayo\nen un pedazo <10% de la gigante (b=1)")
    ax.grid(alpha=0.25)

    for k, (campo, tit) in enumerate((("pend_orig", "pendiente ORIGINAL"),
                                      ("pend_corr", "pendiente CORREGIDA"))):
        ax = axes[k + 1]
        for d in ph:
            ax.scatter(d[campo], d["fraccion_masa"], s=26, alpha=0.8,
                       color="#cc3311" if d["pend_orig"] < -0.3 else
                             ("#ee7733" if d["clase"] == "III" else "#4477aa"))
        rho, _ = stats.spearmanr([d[campo] for d in ph], [d["fraccion_masa"] for d in ph])
        ax.set_xlabel(tit); ax.set_ylabel("fraccion de masa en sumideros (Phantom)")
        ax.set_title(f"{tit} vs masa  (Spearman {rho:+.3f}, n={len(ph)})")
        ax.grid(alpha=0.25)
    fig.suptitle("¿La forma de U venia de la geometria o de donde arranca la medicion del diametro?",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{HERE}/cs090_fase6_outliers_diam_gigante.png", dpi=130)
    print("[png] cs090_fase6_outliers_diam_gigante.png")


if __name__ == "__main__":
    main()

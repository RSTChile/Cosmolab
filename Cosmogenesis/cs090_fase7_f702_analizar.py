"""
cs090_fase7_f702_analizar.py — FASE VII F7-02: ¿la masa acretada SIGUE a la escalera de clustering?

Lee los dumps de Phantom de todos los escalones, los une con la estructura medida por
`cs090_fase7_f702_escalera.py` y contesta la pregunta central con un DISEÑO PAREADO: cada grafo base es
su propio control a través de sus escalones (mismo N, mismo nº de aristas, misma secuencia de grados
nodo por nodo — lo único que cambia entre escalones de un mismo grafo es el clustering).

Reusa `cs090_fase5b_analizar.analizar_carpeta` TAL CUAL (sólo import): la misma extracción de métricas
de toda la línea (fracción de masa en sumideros como observable principal, κ_V agregado como
secundario). No se reimplementa nada de eso.

VERIFICACIÓN CRUZADA OBLIGATORIA (lección del bug de colisión de nombres de Fase V-B), antes de que
ninguna corrida entre en la estadística, contra el `meta_regla.json` de CADA carpeta:
  - la tarea declarada es FASE7_F702_escalera_clustering,
  - el escalón declarado coincide con el sufijo del nombre de la carpeta,
  - el (rule_id, seed) declarado coincide con el del nombre de la carpeta,
  - todos los escalones de un mismo grafo tienen el MISMO nº de aristas y la MISMA seed_layout,
  - el meta declara `grados_identicos_al_original = true`,
  - la carpeta declarada dentro del meta es la carpeta donde está el meta.
LA UNIÓN CON LA ESTRUCTURA ES POR (rule_id, seed, escalon) — nunca por rule_id solo: hay reglas
distintas con el mismo rule_id en lotes distintos (bug documentado en FASE6_O3B §2.1).

QUÉ ESTADÍSTICA SE HACE Y POR QUÉ
----------------------------------
  1. **Dentro de cada grafo** (lo que de verdad contesta la pregunta): Spearman entre clustering y
     fracción de masa sobre los escalones de ESE grafo, y si la secuencia es o no monótona creciente.
  2. **Global pareado**: test de Friedman (¿hay alguna diferencia entre escalones, tratando cada grafo
     como bloque?) y test de tendencia de Page (L), que es el test específico para la hipótesis
     ORDENADA "e0 <= e1 <= e2 <= e3 <= e4" en un diseño de bloques. Friedman dice "algo cambia", Page
     dice "cambia en el orden predicho".
  3. **Global crudo**: Spearman sobre todos los puntos juntos (menos informativo: mezcla la variación
     entre grafos con la de dentro de cada grafo) y Spearman sobre los valores CENTRADOS por grafo
     (a cada corrida se le resta la media de su grafo), que es el equivalente pareado.
  4. **Por qué vía**: correlación entre clustering y pendiente corregida, y correlación parcial de la
     masa con el clustering controlando la pendiente — para ver si el clustering mueve la masa a través
     de la geometría o por otro camino.

Escribe CSV crudo, CSV por grafo, CSV de estadística y el PNG de la escalera. No declara cierre ni
veredicto: sólo números.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, HERE)

from cs090_fase5b_analizar import analizar_carpeta          # sólo import, script congelado

BASE = Path("/Users/alexis/phantom_cs073/bateria_fase7_f702_escalera")
RUTA_CRUDO = f"{HERE}/cs090_fase7_f702_phantom_crudo.csv"
RUTA_POR_GRAFO = f"{HERE}/cs090_fase7_f702_por_grafo.csv"
RUTA_ESTAD = f"{HERE}/cs090_fase7_f702_estadistica.csv"
RUTA_PNG = f"{HERE}/cs090_fase7_f702_escalera.png"
RUTA_ESTRUCTURA = f"{HERE}/cs090_fase7_f702_estructura.csv"

ESCALONES = ("e0", "e1", "e2", "e3", "e4")     # el orden de la escalera (creciente en clustering)


# =============================================================================================
# Test de tendencia de Page (L) — el test para "los tratamientos crecen en el orden predicho"
# =============================================================================================
def page_L(matriz):
    """`matriz` es (n_bloques x k_tratamientos) con los tratamientos YA en el orden predicho creciente.
    Dentro de cada bloque se rankean los k valores (1 = el más chico). L = Σ_j j·R_j donde R_j es la
    suma de rangos de la columna j. Bajo H0 (sin tendencia) L tiene media y varianza conocidas; se usa
    la aproximación normal estándar, válida ya con n>=8 bloques."""
    m = np.asarray(matriz, dtype=float)
    n, k = m.shape
    rangos = np.vstack([stats.rankdata(fila) for fila in m])
    R = rangos.sum(axis=0)
    L = float(np.sum([(j + 1) * R[j] for j in range(k)]))
    mu = n * k * (k + 1) ** 2 / 4.0
    var = n * (k ** 3 - k) ** 2 / (144.0 * (k - 1))
    z = (L - mu) / np.sqrt(var)
    p_una_cola = float(stats.norm.sf(z))
    return dict(L=L, z=float(z), p_una_cola_creciente=p_una_cola,
                p_dos_colas=float(2 * min(p_una_cola, 1 - p_una_cola)))


def correlacion_parcial(x, y, z):
    """Spearman parcial de x con y controlando z: se rankea todo, se saca de x y de y la parte que
    explica z (regresión lineal sobre los rangos) y se correlacionan los residuos.

    Devuelve (nan, nan) si la covariable z no está disponible (p.ej. la estructura todavía no fue
    escrita a disco): mejor un hueco declarado que un número inventado."""
    x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
    if not np.isfinite(z).all() or not np.isfinite(x).all() or not np.isfinite(y).all():
        return float("nan"), float("nan")
    rx, ry, rz = stats.rankdata(x), stats.rankdata(y), stats.rankdata(z)
    def resid(a, b):
        A = np.vstack([b, np.ones_like(b)]).T
        coef, *_ = np.linalg.lstsq(A, a, rcond=None)
        return a - A @ coef
    ex, ey = resid(rx, rz), resid(ry, rz)
    r, p = stats.pearsonr(ex, ey)
    return float(r), float(p)


# =============================================================================================
def cargar_estructura():
    """Une TODOS los shards de estructura. Clave: (rule_id, seed, escalon)."""
    est = {}
    for p in sorted(Path(HERE).glob("cs090_fase7_f702_estructura*.csv")):
        if "_piloto" in p.name:
            continue
        for r in csv.DictReader(open(p)):
            est[(r["rule_id"], int(r["seed"]), r["escalon"])] = r
    return est


def main():
    estructura = cargar_estructura()
    print(f"[f702] estructura: {len(estructura)} filas (rule_id, seed, escalon)")

    # ---------------- lectura + verificación cruzada de cada carpeta ----------------
    metas, problemas = {}, []
    for carpeta in sorted(c for c in BASE.iterdir() if c.is_dir()):
        mp = carpeta / "meta_regla.json"
        if not mp.exists():
            problemas.append(f"{carpeta.name}: sin meta_regla.json")
            continue
        m = json.loads(mp.read_text())
        if m.get("tarea") != "FASE7_F702_escalera_clustering":
            problemas.append(f"{carpeta.name}: tarea declarada = {m.get('tarea')}")
            continue
        if Path(m.get("carpeta", "")).name != carpeta.name:
            problemas.append(f"{carpeta.name}: el meta declara carpeta={m.get('carpeta')}")
            continue
        esperado = f"{m['rule_id']}_s{m['seed']}_f702_{m['escalon']}"
        if carpeta.name != esperado:
            problemas.append(f"{carpeta.name}: el meta corresponde a {esperado}")
            continue
        if not m.get("grados_identicos_al_original", False):
            problemas.append(f"{carpeta.name}: el meta NO declara grados idénticos al original")
            continue
        if not (carpeta / "cosmog_00500").exists():
            problemas.append(f"{carpeta.name}: sin dump final cosmog_00500 (¿todavía corriendo?)")
            continue
        metas[carpeta.name] = (carpeta, m)

    # agrupar por grafo base y verificar coherencia interna del bloque
    grupos = defaultdict(dict)
    for nombre, (carpeta, m) in metas.items():
        grupos[(m["rule_id"], m["seed"])][m["escalon"]] = (carpeta, m)

    filas = []
    grafos_completos = []
    for (rid, seed), esc in sorted(grupos.items()):
        faltan = [e for e in ESCALONES if e not in esc]
        if faltan:
            problemas.append(f"{rid} s{seed}: faltan escalones {faltan} -- no entra en la estadística")
            continue
        aristas = {esc[e][1]["n_aristas_grafo_final"] for e in esc}
        layouts = {esc[e][1]["seed_layout"] for e in esc}
        if len(aristas) != 1 or len(layouts) != 1:
            problemas.append(f"{rid} s{seed}: aristas={aristas} seed_layout={layouts} (no uniformes)")
            continue
        grafos_completos.append((rid, seed))
        for nombre_esc, (carpeta, m) in sorted(esc.items()):
            f = analizar_carpeta(carpeta)
            # chequeo anti-IC-truncado: el generador y el runner pueden correr en paralelo, así que se
            # confirma que la corrida arrancó con las 2000 partículas de gas y no con un archivo a medio
            # escribir (un IC truncado daría menos partículas y una corrida no comparable)
            if f.get("n_gas_inicial") not in (None, 2000):
                problemas.append(f"{carpeta.name}: n_gas_inicial={f['n_gas_inicial']} (¿IC truncado?)")
            st = estructura.get((rid, seed, nombre_esc), {})
            filas.append(dict(
                rule_id=rid, seed=seed, lote=m.get("lote"), K=m.get("K"), kcap=m.get("kcap"),
                escalon=nombre_esc, n_aristas=m["n_aristas_grafo_final"],
                clustering=float(m["clustering_local"]), transitividad=float(m["transitividad"]),
                n_triangulos=int(m["n_triangulos"]), gigante=int(m["gigante"]),
                asortatividad=(float(st["asortatividad"]) if st.get("asortatividad") else None),
                n_componentes=(int(st["n_componentes"]) if st.get("n_componentes") else None),
                solapamiento_aristas=float(m["solapamiento_aristas"]),
                pendiente_corr=float(m["pendiente_corregida"]),
                frac_masa=f["fraccion_masa_en_sumideros"],
                kappa_v=f["kappa_v_agregado"], n_sumideros=f["n_sumideros"],
                t_primer_sumidero=f["t_primer_sumidero"],
                masa_acretada=f["masa_acretada_total"], dump_final=f.get("n_dump_final"),
                carpeta=carpeta.name,
            ))

    print(f"[f702] {len(grafos_completos)} grafos con la escalera completa; {len(filas)} corridas; "
          f"{len(problemas)} avisos")
    for pr in problemas:
        print(f"   !! {pr}")

    campos = []
    for f in filas:
        for c in f:
            if c not in campos:
                campos.append(c)
    with open(RUTA_CRUDO, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)
    print(f"[csv] {RUTA_CRUDO.split('/')[-1]} ({len(filas)} filas)")

    if not grafos_completos:
        print("[f702] todavía no hay ningún grafo con la escalera completa -- nada que analizar")
        return

    # ---------------- 1) relación DENTRO de cada grafo ----------------
    por_grafo, matriz_masa, matriz_kv, matriz_pend = [], [], [], []
    for (rid, seed) in grafos_completos:
        sub = {f["escalon"]: f for f in filas if f["rule_id"] == rid and f["seed"] == seed}
        c = np.array([sub[e]["clustering"] for e in ESCALONES])
        y = np.array([sub[e]["frac_masa"] for e in ESCALONES])
        kv = np.array([(sub[e]["kappa_v"] if sub[e]["kappa_v"] is not None else np.nan)
                       for e in ESCALONES])
        pend = np.array([sub[e]["pendiente_corr"] for e in ESCALONES])
        rho, p = stats.spearmanr(c, y)
        matriz_masa.append(y); matriz_kv.append(kv); matriz_pend.append(pend)
        # pendiente de la recta masa ~ clustering dentro del grafo (unidades: fracción de masa por
        # unidad de clustering; el rango de clustering es ~0 a ~0.4, así que es un número interpretable)
        b = float(np.polyfit(c, y, 1)[0])
        por_grafo.append(dict(
            rule_id=rid, seed=seed, lote=sub["e0"]["lote"], n_aristas=sub["e0"]["n_aristas"],
            clustering_min=float(c.min()), clustering_max=float(c.max()),
            **{f"C_{e}": float(sub[e]["clustering"]) for e in ESCALONES},
            **{f"masa_{e}": float(sub[e]["frac_masa"]) for e in ESCALONES},
            **{f"pend_{e}": float(sub[e]["pendiente_corr"]) for e in ESCALONES},
            **{f"gigante_{e}": int(sub[e]["gigante"]) for e in ESCALONES},
            **{f"asort_{e}": sub[e]["asortatividad"] for e in ESCALONES},
            **{f"kv_{e}": sub[e]["kappa_v"] for e in ESCALONES},
            # el grafo ORIGINAL sin tocar corre también, como punto de referencia: NO es un escalón de
            # la escalera (tiene otra historia de recableo), pero permite ver dónde cae su clustering
            # natural dentro del rango barrido y si su masa se sienta sobre la relación de la escalera
            clustering_orig_natural=(sub["orig"]["clustering"] if "orig" in sub else None),
            masa_orig_natural=(sub["orig"]["frac_masa"] if "orig" in sub else None),
            spearman_rho=float(rho), spearman_p=float(p),
            pendiente_masa_vs_clustering=b,
            monotona_creciente=bool(np.all(np.diff(y) >= 0)),
            monotona_decreciente=bool(np.all(np.diff(y) <= 0)),
            masa_max_en=ESCALONES[int(np.argmax(y))], masa_min_en=ESCALONES[int(np.argmin(y))],
            delta_masa_e4_e0=float(y[-1] - y[0]),
            delta_masa_max_min=float(y.max() - y.min()),
        ))
    with open(RUTA_POR_GRAFO, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(por_grafo[0].keys()))
        w.writeheader()
        w.writerows(por_grafo)
    print(f"[csv] {RUTA_POR_GRAFO.split('/')[-1]} ({len(por_grafo)} grafos)")

    M = np.vstack(matriz_masa)
    print("\n--- masa (fracción en sumideros) por escalón, un grafo por fila ---")
    for g, fila in zip(por_grafo, M):
        print(f"   {g['rule_id']:<24} s{g['seed']}  " + "  ".join(f"{v:.4f}" for v in fila)
              + f"   ρ={g['spearman_rho']:+.2f}  monótona↑={g['monotona_creciente']}")
    print("   media por escalón:      " + "  ".join(f"{v:.4f}" for v in M.mean(axis=0)))

    # ---------------- 2) global pareado ----------------
    resumen = []
    friedman = stats.friedmanchisquare(*[M[:, j] for j in range(M.shape[1])])
    page = page_L(M)
    n_creciente = sum(g["monotona_creciente"] for g in por_grafo)
    n_rho_pos = sum(g["spearman_rho"] > 0 for g in por_grafo)
    signos_e4_e0 = stats.binomtest(sum(g["delta_masa_e4_e0"] > 0 for g in por_grafo),
                                   len(por_grafo), 0.5, alternative="two-sided").pvalue
    try:
        wil_e4_e0 = stats.wilcoxon([g["delta_masa_e4_e0"] for g in por_grafo],
                                   alternative="two-sided")
        wil_p = float(wil_e4_e0.pvalue)
    except Exception:
        wil_p = float("nan")

    # ---------------- 3) global crudo y centrado ----------------
    # sólo los ESCALONES: la corrida `orig` es un punto de referencia con otra historia de recableo,
    # no un peldaño de la escalera, así que no entra en las correlaciones de la intervención
    filas_esc = [f for f in filas if f["escalon"] in ESCALONES]
    c_all = np.array([f["clustering"] for f in filas_esc])
    y_all = np.array([f["frac_masa"] for f in filas_esc])
    rho_global, p_global = stats.spearmanr(c_all, y_all)
    # centrado por grafo = equivalente pareado del Spearman global
    idx_g = {}
    for f in filas_esc:
        idx_g.setdefault((f["rule_id"], f["seed"]), []).append(f)
    c_cen, y_cen, p_cen, a_cen, g_cen = [], [], [], [], []
    for k, fs in idx_g.items():
        cc = np.array([f["clustering"] for f in fs]); yy = np.array([f["frac_masa"] for f in fs])
        pp = np.array([f["pendiente_corr"] for f in fs])
        # covariables que los swaps dirigidos arrastran sin querer y que hay que descontar:
        # asortatividad de grados y tamaño de la componente gigante
        aa = np.array([(f["asortatividad"] if f["asortatividad"] is not None else np.nan) for f in fs])
        gg = np.array([float(f["gigante"]) for f in fs])
        if np.all(np.isnan(aa)):
            aa = np.zeros_like(gg) * np.nan
        c_cen += list(cc - cc.mean()); y_cen += list(yy - yy.mean()); p_cen += list(pp - pp.mean())
        a_cen += list(aa - (np.nanmean(aa) if np.any(np.isfinite(aa)) else 0.0))
        g_cen += list(gg - gg.mean())
    rho_cen, p_cen_p = stats.spearmanr(c_cen, y_cen)

    # ---------------- 4) ¿por qué vía? clustering -> pendiente -> masa ----------------
    rho_cp, p_cp = stats.spearmanr(c_cen, p_cen)          # clustering vs pendiente (pareado)
    rho_pm, p_pm = stats.spearmanr(p_cen, y_cen)          # pendiente vs masa (pareado)
    r_parcial, p_parcial = correlacion_parcial(np.array(c_cen), np.array(y_cen), np.array(p_cen))
    # covariables arrastradas: asortatividad de grados y componente gigante
    a_arr = np.asarray(a_cen, dtype=float)
    if np.isfinite(a_arr).all():
        rho_ca, p_ca = stats.spearmanr(c_cen, a_cen)
        rho_am, p_am = stats.spearmanr(a_cen, y_cen)
    else:
        rho_ca = p_ca = rho_am = p_am = float("nan")
    r_parc_a, p_parc_a = correlacion_parcial(np.array(c_cen), np.array(y_cen), np.array(a_cen))
    rho_cg, p_cg = stats.spearmanr(c_cen, g_cen)
    rho_gm, p_gm = stats.spearmanr(g_cen, y_cen)
    r_parc_g, p_parc_g = correlacion_parcial(np.array(c_cen), np.array(y_cen), np.array(g_cen))

    resumen = [
        dict(prueba="Friedman (¿algún escalón difiere?)", estadistico=float(friedman.statistic),
             p=float(friedman.pvalue), n=len(por_grafo), detalle="bloques=grafos, tratamientos=5 escalones"),
        dict(prueba="Page L (tendencia creciente e0<=..<=e4)", estadistico=page["L"],
             p=page["p_una_cola_creciente"], n=len(por_grafo), detalle=f"z={page['z']:.3f}"),
        dict(prueba="grafos con masa monótona creciente", estadistico=n_creciente,
             p=float("nan"), n=len(por_grafo), detalle=f"{n_creciente}/{len(por_grafo)}"),
        dict(prueba="grafos con Spearman intra-grafo > 0", estadistico=n_rho_pos,
             p=float("nan"), n=len(por_grafo), detalle=f"{n_rho_pos}/{len(por_grafo)}"),
        dict(prueba="signos e4 vs e0", estadistico=sum(g["delta_masa_e4_e0"] > 0 for g in por_grafo),
             p=float(signos_e4_e0), n=len(por_grafo), detalle="binomial exacto 2 colas"),
        dict(prueba="Wilcoxon e4 vs e0", estadistico=float("nan"), p=wil_p, n=len(por_grafo),
             detalle="rangos con signo, 2 colas"),
        dict(prueba="Spearman GLOBAL crudo clustering-masa", estadistico=float(rho_global),
             p=float(p_global), n=len(filas_esc), detalle="mezcla variación entre y dentro de grafos"),
        dict(prueba="Spearman GLOBAL centrado por grafo", estadistico=float(rho_cen),
             p=float(p_cen_p), n=len(filas_esc), detalle="equivalente pareado"),
        dict(prueba="Spearman centrado clustering-pendiente", estadistico=float(rho_cp),
             p=float(p_cp), n=len(filas_esc), detalle="¿el clustering mueve la geometría?"),
        dict(prueba="Spearman centrado pendiente-masa", estadistico=float(rho_pm),
             p=float(p_pm), n=len(filas_esc), detalle="¿la geometría mueve la masa?"),
        dict(prueba="parcial clustering-masa | pendiente", estadistico=r_parcial,
             p=p_parcial, n=len(filas_esc), detalle="sobre rangos, centrado por grafo"),
        dict(prueba="Spearman centrado clustering-asortatividad", estadistico=float(rho_ca),
             p=float(p_ca), n=len(filas_esc), detalle="covariable arrastrada por los swaps dirigidos"),
        dict(prueba="Spearman centrado asortatividad-masa", estadistico=float(rho_am),
             p=float(p_am), n=len(filas_esc), detalle="¿la asortatividad explica la masa?"),
        dict(prueba="parcial clustering-masa | asortatividad", estadistico=r_parc_a,
             p=p_parc_a, n=len(filas_esc), detalle="descontando la asortatividad"),
        dict(prueba="Spearman centrado clustering-gigante", estadistico=float(rho_cg),
             p=float(p_cg), n=len(filas_esc), detalle="covariable arrastrada (conectividad)"),
        dict(prueba="Spearman centrado gigante-masa", estadistico=float(rho_gm),
             p=float(p_gm), n=len(filas_esc), detalle="¿la conectividad explica la masa?"),
        dict(prueba="parcial clustering-masa | gigante", estadistico=r_parc_g,
             p=p_parc_g, n=len(filas_esc), detalle="descontando el tamaño de la componente gigante"),
        dict(prueba="media rho intra-grafo", estadistico=float(np.mean([g["spearman_rho"] for g in por_grafo])),
             p=float("nan"), n=len(por_grafo), detalle="promedio simple de los ρ por grafo"),
    ]
    with open(RUTA_ESTAD, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(resumen[0].keys()))
        w.writeheader()
        w.writerows(resumen)
    print(f"\n[csv] {RUTA_ESTAD.split('/')[-1]}")
    for r in resumen:
        print(f"   {r['prueba']:<42} est={r['estadistico']:<10.4f} p={r['p']:<10.4g} n={r['n']}  "
              f"({r['detalle']})")

    graficar(filas, por_grafo, M)
    return filas, por_grafo, resumen


# =============================================================================================
def graficar(filas, por_grafo, M):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(16.5, 5.2))
    colores = plt.cm.viridis(np.linspace(0, 0.92, len(por_grafo)))

    ax = axs[0]
    for g, col in zip(por_grafo, colores):
        cs = [g[f"C_{e}"] for e in ESCALONES]
        ys = [g[f"masa_{e}"] for e in ESCALONES]
        ax.plot(cs, ys, "o-", color=col, lw=1.4, ms=4, alpha=0.85,
                label=f"{g['rule_id'].replace('A2-B0-C2-','')} (ρ={g['spearman_rho']:+.2f})")
        # el grafo ORIGINAL sin tocar, como referencia (no es un escalón: tiene otra historia de
        # recableo). Se dibuja con una cruz para ver dónde cae su clustering natural en el rango
        # barrido y si su masa se sienta o no sobre la curva de la escalera.
        if g.get("clustering_orig_natural") is not None:
            ax.plot([g["clustering_orig_natural"]], [g["masa_orig_natural"]], "X", color=col,
                    ms=11, mec="k", mew=0.8, zorder=5)
    ax.plot([], [], "X", color="0.4", mec="k", ms=10, label="× = grafo original sin tocar")
    ax.set_xlabel("clustering local medio del grafo (escalón)")
    ax.set_ylabel("fracción de masa en sumideros")
    ax.set_title("Cada línea = un grafo base recorriendo su escalera\n"
                 "(N, aristas y grados nodo-por-nodo idénticos dentro de cada línea)", fontsize=10)
    ax.legend(fontsize=6.2, ncol=2)
    ax.grid(alpha=0.25)

    ax = axs[1]
    xs = np.arange(len(ESCALONES))
    for fila, col in zip(M, colores):
        ax.plot(xs, fila - fila.mean(), "o-", color=col, lw=1.0, ms=3.5, alpha=0.55)
    ax.plot(xs, (M - M.mean(axis=1, keepdims=True)).mean(axis=0), "s-", color="crimson", lw=2.6, ms=8,
            label="media de los grafos")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels(ESCALONES)
    ax.set_xlabel("escalón de la escalera (clustering creciente →)")
    ax.set_ylabel("fracción de masa − media de su propio grafo")
    ax.set_title("Diseño pareado: cada grafo centrado en sí mismo", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axs[2]
    for g, col in zip(por_grafo, colores):
        cs = [g[f"C_{e}"] for e in ESCALONES]
        ps = [g[f"pend_{e}"] for e in ESCALONES]
        ax.plot(cs, ps, "o-", color=col, lw=1.2, ms=4, alpha=0.8)
    ax.set_xlabel("clustering local medio del grafo (escalón)")
    ax.set_ylabel("pendiente corregida log(diám)–log(N_cajas)")
    ax.set_title("Control de vía: ¿el clustering arrastra la geometría?", fontsize=10)
    ax.grid(alpha=0.25)

    fig.suptitle("F7-02 · Escalera de clustering con secuencia de grados clavada — "
                 "¿la masa acretada sigue a la perilla?", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RUTA_PNG, dpi=140)
    print(f"[png] {RUTA_PNG.split('/')[-1]}")


if __name__ == "__main__":
    main()

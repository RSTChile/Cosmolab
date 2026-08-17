"""
cs090_fase7_f703_analizar.py — FASE VII F7-03: con GRADOS Y TRIÁNGULOS FIJOS, ¿la masa cambia según
cómo estén ORGANIZADOS esos triángulos?

Lee los dumps de Phantom de los cinco brazos, los une con la estructura medida por
`cs090_fase7_f703_organizacion.py` y contesta con un DISEÑO PAREADO: cada grafo base es su propio
control a través de sus cinco brazos (mismo N, mismo nº de aristas, misma secuencia de grados nodo por
nodo, MISMO nº de triángulos; lo único que cambia es dónde están puestos esos triángulos).

Reusa `cs090_fase5b_analizar.analizar_carpeta` TAL CUAL (sólo import): la misma extracción de métricas
de toda la línea (fracción de masa en sumideros como observable principal, κ_V agregado como
secundario). No se reimplementa nada de eso.

VERIFICACIÓN CRUZADA OBLIGATORIA contra el `meta_regla.json` de CADA carpeta, antes de que ninguna
corrida entre en la estadística (lección del bug de colisión de nombres de Fase V-B):
  - la tarea declarada es FASE7_F703_organizacion_triangulos,
  - el brazo declarado coincide con el sufijo del nombre de la carpeta,
  - el (rule_id, seed) declarado coincide con el del nombre de la carpeta,
  - los cinco brazos de un grafo tienen el MISMO nº de aristas, la MISMA seed_layout y — lo específico
    de esta tarea — el MISMO nº de triángulos,
  - el meta declara `grados_identicos_al_original = true`,
  - la carpeta declarada dentro del meta es la carpeta donde está el meta.
LA UNIÓN CON LA ESTRUCTURA ES POR (rule_id, seed, brazo) — nunca por rule_id solo (bug documentado en
FASE6_O3B §2.1: hay reglas distintas con el mismo rule_id en lotes distintos).

QUÉ ESTADÍSTICA SE HACE Y POR QUÉ
----------------------------------
A diferencia de F7-02, acá los cinco brazos NO están ordenados en una escalera: son cinco maneras
distintas de repartir la misma cantidad de triángulos. Por eso:
  1. **Friedman** (bloques = grafos, tratamientos = 5 brazos): ¿hay ALGUNA diferencia entre brazos?
  2. **Wilcoxon pareado** de los contrastes con sentido físico declarado de antemano:
     solap vs disj (el eje de solapamiento de aristas) y conc vs disp (el eje de concentración en
     nodos), más cada brazo contra `libre`.
  3. **Spearman centrado por grafo** de la masa contra CADA medida de organización, para responder la
     segunda mitad de la pregunta: si la masa cambia, ¿qué propiedad la sigue mejor?
  4. Todo el tamaño de efecto se reporta también **en partículas**: con N=2000 y masa total 18800, una
     partícula = 0.0005 de fracción de masa (grano del instrumento medido en
     `FASE7_F704_cortar_bien_vs_azar_CS.md`).

Escribe CSV crudo, CSV por grafo, CSV de estadística y el PNG. No declara cierre ni veredicto.
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
from cs090_fase7_f702_analizar import correlacion_parcial   # sólo import (misma parcial que F7-02)

BASE = Path("/Users/alexis/phantom_cs073/bateria_fase7_f703_organizacion")
RUTA_CRUDO = f"{HERE}/cs090_fase7_f703_phantom_crudo.csv"
RUTA_POR_GRAFO = f"{HERE}/cs090_fase7_f703_por_grafo.csv"
RUTA_ESTAD = f"{HERE}/cs090_fase7_f703_estadistica.csv"
RUTA_CORREL = f"{HERE}/cs090_fase7_f703_correlaciones.csv"
RUTA_PARCIAL = f"{HERE}/cs090_fase7_f703_parciales.csv"
RUTA_PNG = f"{HERE}/cs090_fase7_f703_organizacion.png"

BRAZOS = ("libre", "conc", "disp", "solap", "disj")
GRANO_PARTICULA = 0.0005        # 1 partícula de 2000 = 0.0005 de fracción de masa (F7-04)

# medidas de organización que se enfrentan a la masa (todas salen del CSV de estructura)
MEDIDAS = [
    ("frac_aristas_multi_tri", "solapamiento: aristas en >1 triángulo"),
    ("gini_tri_nodo", "concentración: Gini de triángulos por nodo"),
    ("frac_nodos_en_triangulo", "soporte: nodos que tocan algún triángulo"),
    ("frac_aristas_en_triangulo", "soporte: aristas en algún triángulo"),
    ("n_comp_tri", "nº de cúmulos de triángulos"),
    ("frac_mayor_comp_tri", "fracción de triángulos en el cúmulo mayor"),
    ("modularidad_tri", "modularidad de la partición inducida"),
    ("dist_media_tri", "distancia media entre triángulos (saltos)"),
    ("tri_por_arista_media", "triángulos por arista con triángulo"),
    ("tri_por_nodo_max", "máximo de triángulos en un solo nodo"),
    ("clustering_local", "clustering local medio (Watts-Strogatz)"),
    ("transitividad", "transitividad global (fija por diseño)"),
    ("asortatividad", "asortatividad de grados"),
    ("pendiente_corr", "pendiente corregida log(diám)-log(N_cajas)"),
    ("gigante", "tamaño de la componente gigante"),
    ("solapamiento_aristas", "aristas compartidas con el grafo original"),
]


def cargar_estructura():
    est = {}
    for p in sorted(Path(HERE).glob("cs090_fase7_f703_estructura*.csv")):
        if "_piloto" in p.name:
            continue
        for r in csv.DictReader(open(p)):
            est[(r["rule_id"], int(r["seed"]), r["brazo"])] = r
    return est


def _f(d, k):
    v = d.get(k)
    if v in (None, "", "nan"):
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def main():
    estructura = cargar_estructura()
    print(f"[f703] estructura: {len(estructura)} filas (rule_id, seed, brazo)")

    metas, problemas = {}, []
    for carpeta in sorted(c for c in BASE.iterdir() if c.is_dir()):
        mp = carpeta / "meta_regla.json"
        if not mp.exists():
            problemas.append(f"{carpeta.name}: sin meta_regla.json")
            continue
        m = json.loads(mp.read_text())
        if m.get("tarea") != "FASE7_F703_organizacion_triangulos":
            problemas.append(f"{carpeta.name}: tarea declarada = {m.get('tarea')}")
            continue
        if Path(m.get("carpeta", "")).name != carpeta.name:
            problemas.append(f"{carpeta.name}: el meta declara carpeta={m.get('carpeta')}")
            continue
        esperado = f"{m['rule_id']}_s{m['seed']}_f703_{m['brazo']}"
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

    grupos = defaultdict(dict)
    for nombre, (carpeta, m) in metas.items():
        grupos[(m["rule_id"], m["seed"])][m["brazo"]] = (carpeta, m)

    filas, grafos_completos = [], []
    for (rid, seed), br in sorted(grupos.items()):
        faltan = [b for b in BRAZOS if b not in br]
        if faltan:
            problemas.append(f"{rid} s{seed}: faltan brazos {faltan} -- no entra en la estadística")
            continue
        aristas = {br[b][1]["n_aristas_grafo_final"] for b in br}
        layouts = {br[b][1]["seed_layout"] for b in br}
        tris = {br[b][1]["n_triangulos"] for b in br}
        if len(aristas) != 1 or len(layouts) != 1:
            problemas.append(f"{rid} s{seed}: aristas={aristas} seed_layout={layouts} (no uniformes)")
            continue
        if len(tris) != 1:
            problemas.append(f"{rid} s{seed}: nº de triángulos NO idéntico entre brazos: {sorted(tris)}"
                             f" (rango {max(tris)-min(tris)}) -- entra igual, se reporta la diferencia")
        grafos_completos.append((rid, seed))
        for nombre_br, (carpeta, m) in sorted(br.items()):
            f = analizar_carpeta(carpeta)
            if f.get("n_gas_inicial") not in (None, 2000):
                problemas.append(f"{carpeta.name}: n_gas_inicial={f['n_gas_inicial']} (¿IC truncado?)")
            st = estructura.get((rid, seed, nombre_br), {})
            fila = dict(
                rule_id=rid, seed=seed, lote=m.get("lote"), K=m.get("K"), kcap=m.get("kcap"),
                brazo=nombre_br, n_aristas=m["n_aristas_grafo_final"],
                n_triangulos=int(m["n_triangulos"]), T_objetivo=m.get("T_objetivo"),
                dif_max_triangulos=m.get("dif_max_triangulos"),
                clustering_local=float(m["clustering_local"]),
                transitividad=float(m["transitividad"]),
                gigante=int(m["gigante"]),
                solapamiento_aristas=float(m["solapamiento_aristas"]),
                pendiente_corr=float(m["pendiente_corregida"]),
                frac_masa=f["fraccion_masa_en_sumideros"],
                kappa_v=f["kappa_v_agregado"], n_sumideros=f["n_sumideros"],
                t_primer_sumidero=f["t_primer_sumidero"],
                masa_acretada=f["masa_acretada_total"], dump_final=f.get("n_dump_final"),
                carpeta=carpeta.name,
            )
            for col, _ in MEDIDAS:
                if col not in fila:
                    fila[col] = _f(st, col)
            fila["asortatividad"] = _f(st, "asortatividad")
            fila["n_componentes"] = _f(st, "n_componentes")
            fila["dist_media_azar"] = _f(st, "dist_media_azar")
            fila["t_piso"] = _f(st, "t_piso")
            fila["n_triangulos_original"] = _f(st, "n_triangulos_original")
            fila["clustering_original"] = _f(st, "clustering_original")
            filas.append(fila)

    print(f"[f703] {len(grafos_completos)} grafos con los 5 brazos; {len(filas)} corridas; "
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
        print("[f703] todavía no hay ningún grafo con los 5 brazos -- nada que analizar")
        return

    # ================= 1) por grafo =================
    por_grafo, M = [], []
    for (rid, seed) in grafos_completos:
        sub = {f["brazo"]: f for f in filas if f["rule_id"] == rid and f["seed"] == seed}
        y = np.array([sub[b]["frac_masa"] for b in BRAZOS])
        M.append(y)
        tris = [sub[b]["n_triangulos"] for b in BRAZOS]
        por_grafo.append(dict(
            rule_id=rid, seed=seed, lote=sub["libre"]["lote"], n_aristas=sub["libre"]["n_aristas"],
            n_triangulos_min=min(tris), n_triangulos_max=max(tris),
            dif_triangulos=max(tris) - min(tris),
            **{f"masa_{b}": float(sub[b]["frac_masa"]) for b in BRAZOS},
            **{f"C_{b}": float(sub[b]["clustering_local"]) for b in BRAZOS},
            **{f"multi_{b}": float(sub[b]["frac_aristas_multi_tri"]) for b in BRAZOS},
            **{f"gini_{b}": float(sub[b]["gini_tri_nodo"]) for b in BRAZOS},
            **{f"pend_{b}": float(sub[b]["pendiente_corr"]) for b in BRAZOS},
            **{f"gigante_{b}": int(sub[b]["gigante"]) for b in BRAZOS},
            rango_masa=float(y.max() - y.min()),
            rango_masa_particulas=float((y.max() - y.min()) / GRANO_PARTICULA),
            brazo_masa_max=BRAZOS[int(np.argmax(y))], brazo_masa_min=BRAZOS[int(np.argmin(y))],
            d_solap_disj=float(sub["solap"]["frac_masa"] - sub["disj"]["frac_masa"]),
            d_conc_disp=float(sub["conc"]["frac_masa"] - sub["disp"]["frac_masa"]),
            rango_C=float(max(sub[b]["clustering_local"] for b in BRAZOS)
                          - min(sub[b]["clustering_local"] for b in BRAZOS)),
            rango_multi=float(max(sub[b]["frac_aristas_multi_tri"] for b in BRAZOS)
                              - min(sub[b]["frac_aristas_multi_tri"] for b in BRAZOS)),
        ))
    with open(RUTA_POR_GRAFO, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(por_grafo[0].keys()))
        w.writeheader()
        w.writerows(por_grafo)
    print(f"[csv] {RUTA_POR_GRAFO.split('/')[-1]} ({len(por_grafo)} grafos)")

    M = np.vstack(M)
    print("\n--- fracción de masa en sumideros por brazo (una fila = un grafo) ---")
    print("   " + " " * 30 + "  ".join(f"{b:>8}" for b in BRAZOS))
    for g, fila in zip(por_grafo, M):
        print(f"   {g['rule_id'].replace('A2-B0-C2-',''):<12} s{g['seed']} tri={g['n_triangulos_min']:<5}"
              + "  ".join(f"{v:8.4f}" for v in fila)
              + f"   rango={g['rango_masa_particulas']:.1f} part.")
    print("   media:" + " " * 26 + "  ".join(f"{v:8.4f}" for v in M.mean(axis=0)))
    print("   media centrada:" + " " * 17
          + "  ".join(f"{v:+8.4f}" for v in (M - M.mean(axis=1, keepdims=True)).mean(axis=0)))

    # ================= 2) tests pareados =================
    n = len(por_grafo)
    resumen = []
    fried = stats.friedmanchisquare(*[M[:, j] for j in range(M.shape[1])])
    resumen.append(dict(prueba="Friedman (¿algún brazo difiere?)", estadistico=float(fried.statistic),
                        p=float(fried.pvalue), n=n, efecto_particulas=float("nan"),
                        detalle="bloques=grafos, tratamientos=5 brazos"))

    def par(a, b, etiqueta):
        d = np.array([por_grafo[i][f"masa_{a}"] - por_grafo[i][f"masa_{b}"] for i in range(n)])
        try:
            w = float(stats.wilcoxon(d, alternative="two-sided").pvalue)
        except Exception:
            w = float("nan")
        sg = int((d > 0).sum())
        pb = float(stats.binomtest(sg, n, 0.5, alternative="two-sided").pvalue)
        resumen.append(dict(prueba=f"Wilcoxon {etiqueta}", estadistico=float(d.mean()), p=w, n=n,
                            efecto_particulas=float(d.mean() / GRANO_PARTICULA),
                            detalle=f"media Δ={d.mean():+.5f} ({d.mean()/GRANO_PARTICULA:+.1f} part.), "
                                    f"signos {sg}/{n}, binomial p={pb:.4g}"))
        return d

    par("solap", "disj", "solap vs disj  (eje SOLAPAMIENTO de aristas)")
    par("conc", "disp", "conc  vs disp  (eje CONCENTRACIÓN en nodos)")
    for b in ("conc", "disp", "solap", "disj"):
        par(b, "libre", f"{b} vs libre")

    rangos = np.array([g["rango_masa_particulas"] for g in por_grafo])
    resumen.append(dict(prueba="rango masa max-min dentro de cada grafo",
                        estadistico=float(rangos.mean()), p=float("nan"), n=n,
                        efecto_particulas=float(rangos.mean()),
                        detalle=f"mediana={np.median(rangos):.1f} part., min={rangos.min():.1f}, "
                                f"max={rangos.max():.1f} (grano = 1 partícula = {GRANO_PARTICULA})"))

    # ¿cuánto de ese rango es simplemente ruido? vara: el rango esperado si los 5 brazos fueran
    # intercambiables se estima permutando las etiquetas DENTRO de cada grafo (no cambia nada: el
    # rango es invariante a permutar). La vara honesta es comparar el rango contra el grano y contra
    # la dispersión entre brazos que F7-02 vio al mover el clustering de verdad.
    resumen.append(dict(prueba="contraste de clustering logrado (rango C entre brazos)",
                        estadistico=float(np.mean([g["rango_C"] for g in por_grafo])),
                        p=float("nan"), n=n, efecto_particulas=float("nan"),
                        detalle="media del (C_max - C_min) dentro de cada grafo"))
    resumen.append(dict(prueba="contraste de solapamiento logrado (rango frac_aristas_multi)",
                        estadistico=float(np.mean([g["rango_multi"] for g in por_grafo])),
                        p=float("nan"), n=n, efecto_particulas=float("nan"),
                        detalle="media del (multi_max - multi_min) dentro de cada grafo"))
    resumen.append(dict(prueba="diferencia de triángulos entre brazos (control del diseño)",
                        estadistico=float(np.mean([g["dif_triangulos"] for g in por_grafo])),
                        p=float("nan"), n=n, efecto_particulas=float("nan"),
                        detalle=f"máx entre grafos = {max(g['dif_triangulos'] for g in por_grafo)} "
                                f"triángulos"))

    with open(RUTA_ESTAD, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(resumen[0].keys()))
        w.writeheader()
        w.writerows(resumen)
    print(f"\n[csv] {RUTA_ESTAD.split('/')[-1]}")
    for r in resumen:
        print(f"   {r['prueba']:<52} est={r['estadistico']:<11.5f} p={r['p']:<10.4g}  ({r['detalle']})")

    # ================= 3) ¿qué propiedad sigue mejor la masa? =================
    # Spearman CENTRADO POR GRAFO: a cada valor se le resta la media de su propio grafo, de modo que
    # sólo queda la variación ENTRE BRAZOS (que es lo que la intervención movió).
    idx_g = defaultdict(list)
    for f in filas:
        idx_g[(f["rule_id"], f["seed"])].append(f)
    correls = []
    for col, etiqueta in MEDIDAS:
        xs, ys = [], []
        for k, fs in idx_g.items():
            xx = np.array([_f(f, col) for f in fs], dtype=float)
            yy = np.array([f["frac_masa"] for f in fs], dtype=float)
            if not np.isfinite(xx).all() or np.nanstd(xx) < 1e-12:
                continue
            xs += list(xx - xx.mean()); ys += list(yy - yy.mean())
        if len(xs) < 10:
            correls.append(dict(medida=col, descripcion=etiqueta, rho=float("nan"), p=float("nan"),
                                n=len(xs), nota="sin variación entre brazos o sin datos"))
            continue
        rho, p = stats.spearmanr(xs, ys)
        correls.append(dict(medida=col, descripcion=etiqueta, rho=float(rho), p=float(p), n=len(xs),
                            nota=""))
    correls.sort(key=lambda d: (-abs(d["rho"]) if np.isfinite(d["rho"]) else 0))
    with open(RUTA_CORREL, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(correls[0].keys()))
        w.writeheader()
        w.writerows(correls)
    print(f"\n[csv] {RUTA_CORREL.split('/')[-1]} — masa vs cada medida, centrado por grafo")
    for c in correls:
        print(f"   {c['medida']:<28} ρ={c['rho']:+.3f}  p={c['p']:<10.4g} n={c['n']}  ({c['descripcion']})")

    # ================= 4) descontar las covariables arrastradas =================
    # Los brazos no cambian sólo la organización de los triángulos: al reorganizarlos también se mueven,
    # sin querer, el tamaño de la componente gigante, la asortatividad, la pendiente corregida y cuántas
    # aristas quedan compartidas con el grafo original. Se calcula la correlación PARCIAL de la masa con
    # la medida de organización que mejor correlaciona, descontando cada una de esas covariables.
    def _centrado(col):
        v = []
        for k, fs in idx_g.items():
            x = np.array([_f(f, col) for f in fs], dtype=float)
            if not np.isfinite(x).all():
                return None
            v += list(x - x.mean())
        return np.array(v, dtype=float)

    y_cen = _centrado("frac_masa")
    mejor = next((c for c in correls if np.isfinite(c["rho"])), None)
    parciales = []
    if mejor is not None and y_cen is not None:
        x_cen = _centrado(mejor["medida"])
        for cov in ("gigante", "n_componentes", "asortatividad", "pendiente_corr",
                    "solapamiento_aristas", "clustering_local", "frac_aristas_multi_tri",
                    "gini_tri_nodo"):
            if cov == mejor["medida"]:
                continue
            z = _centrado(cov)
            if z is None or x_cen is None or np.std(z) < 1e-12:
                continue
            r, p = correlacion_parcial(x_cen, y_cen, z)
            rz, pz = stats.spearmanr(z, y_cen)
            rxz, pxz = stats.spearmanr(x_cen, z)
            parciales.append(dict(
                medida=mejor["medida"], covariable=cov, rho_parcial=float(r), p_parcial=float(p),
                rho_cov_masa=float(rz), p_cov_masa=float(pz),
                rho_medida_cov=float(rxz), n=len(y_cen)))
        if parciales:
            with open(RUTA_PARCIAL, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(parciales[0].keys()))
                w.writeheader()
                w.writerows(parciales)
            print(f"\n[csv] {RUTA_PARCIAL.split('/')[-1]} — parciales de `{mejor['medida']}` "
                  f"(ρ crudo {mejor['rho']:+.3f}) descontando cada covariable")
            for pa in parciales:
                print(f"   | {pa['covariable']:<24} ρ_parcial={pa['rho_parcial']:+.3f} "
                      f"p={pa['p_parcial']:<10.4g}  (covariable↔masa ρ={pa['rho_cov_masa']:+.3f}, "
                      f"medida↔covariable ρ={pa['rho_medida_cov']:+.3f})")

    graficar(filas, por_grafo, M, correls)
    return filas, por_grafo, resumen, correls


# =============================================================================================
def graficar(filas, por_grafo, M, correls):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(16.5, 5.2))
    colores = plt.cm.viridis(np.linspace(0, 0.92, len(por_grafo)))
    xs = np.arange(len(BRAZOS))

    ax = axs[0]
    for g, fila, col in zip(por_grafo, M, colores):
        ax.plot(xs, fila, "o-", color=col, lw=1.2, ms=4, alpha=0.8,
                label=f"{g['rule_id'].replace('A2-B0-C2-','')} (T={g['n_triangulos_min']})")
    ax.plot(xs, M.mean(axis=0), "s-", color="crimson", lw=2.6, ms=8, label="media")
    ax.set_xticks(xs); ax.set_xticklabels(BRAZOS)
    ax.set_xlabel("organización de los triángulos")
    ax.set_ylabel("fracción de masa en sumideros")
    ax.set_title("Mismos grados nodo-por-nodo y MISMO nº de triángulos\nen los cinco brazos", fontsize=10)
    ax.legend(fontsize=6.2, ncol=2); ax.grid(alpha=0.25)

    ax = axs[1]
    Mc = M - M.mean(axis=1, keepdims=True)
    for fila, col in zip(Mc, colores):
        ax.plot(xs, fila / 0.0005, "o-", color=col, lw=1.0, ms=3.5, alpha=0.55)
    ax.plot(xs, Mc.mean(axis=0) / 0.0005, "s-", color="crimson", lw=2.6, ms=8, label="media")
    ax.axhline(0, color="k", lw=0.8)
    ax.axhspan(-1, 1, color="0.75", alpha=0.5, label="±1 partícula (grano)")
    ax.set_xticks(xs); ax.set_xticklabels(BRAZOS)
    ax.set_ylabel("masa − media de su propio grafo  [partículas]")
    ax.set_title("Diseño pareado, en unidades del grano del instrumento", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axs[2]
    mejor = next((c for c in correls if np.isfinite(c["rho"])), None)
    if mejor is not None:
        col_x = mejor["medida"]
        for k, g in enumerate(por_grafo):
            fs = [f for f in filas if f["rule_id"] == g["rule_id"] and f["seed"] == g["seed"]]
            xx = np.array([_f(f, col_x) for f in fs], float)
            yy = np.array([f["frac_masa"] for f in fs], float)
            ax.plot(xx - xx.mean(), (yy - yy.mean()) / 0.0005, "o", color=colores[k], ms=5, alpha=0.85)
        ax.axhline(0, color="k", lw=0.8); ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel(f"{col_x} − media de su grafo")
        ax.set_ylabel("masa − media de su grafo  [partículas]")
        ax.set_title(f"La medida que mejor sigue a la masa\nρ={mejor['rho']:+.3f} (p={mejor['p']:.3g})",
                     fontsize=10)
        ax.grid(alpha=0.25)

    fig.suptitle("F7-03 · Grados Y nº de triángulos fijos — ¿importa cómo están organizados?",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RUTA_PNG, dpi=140)
    print(f"[png] {RUTA_PNG.split('/')[-1]}")


if __name__ == "__main__":
    main()

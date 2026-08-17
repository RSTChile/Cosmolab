"""
cs090_fase8_f801_analizar.py — FASE VIII F8-01: ¿qué medida de apiñamiento sigue la masa cuando las
otras quedan fijas?

Lee los dumps de Phantom de los cinco brazos de `cs090_fase8_f801_desacople.py`, los une con la
estructura medida por ese mismo script y contesta con un DISEÑO PAREADO: cada grafo base es su propio
control a través de sus cinco brazos (mismo N, mismo nº de aristas, misma secuencia de grados nodo por
nodo, mismo nº de triángulos; lo único que cambia es la ORGANIZACIÓN de esos triángulos).

Reusa `cs090_fase5b_analizar.analizar_carpeta` TAL CUAL (sólo import): la misma extracción de métricas
de toda la línea. `correlacion_parcial` se importa de `cs090_fase7_f702_analizar`, también tal cual.

VERIFICACIÓN CRUZADA OBLIGATORIA contra el `meta_regla.json` de CADA carpeta (misma lista que F7-03):
tarea declarada, brazo y (rule_id, seed) coincidentes con el nombre de la carpeta, carpeta declarada =
carpeta real, `grados_identicos_al_original = true`, mismo nº de aristas / seed_layout / nº de
triángulos entre los cinco brazos de un grafo, y 2000 partículas de gas iniciales (anti-IC-truncado).
LA UNIÓN CON LA ESTRUCTURA ES POR (rule_id, seed, brazo), nunca por rule_id solo.

LO ESPECÍFICO DE ESTA TAREA — la MATRIZ DE DESACOPLE
-----------------------------------------------------
Antes de mirar la masa se reporta, para cada contraste, cuánto se desacopló DE VERDAD: el valor de las
cuatro medidas de apiñamiento en los dos brazos y su diferencia, expresada además como fracción del
rango total que esa medida recorre entre los cinco brazos. Un contraste sólo "aísla" una medida si la
diferencia de las OTRAS tres queda cerca de cero en esa escala. Un desacople fallido se reporta como
fallido: es información sobre qué está ligado por construcción.

Todo tamaño de efecto se reporta también EN PARTÍCULAS: 1 partícula = 0.0005 de fracción de masa
(grano medido en `FASE7_F704_cortar_bien_vs_azar_CS.md`). Endpoints continuos, nunca "% Clase III".
No declara cierre ni veredicto.
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
from cs090_fase7_f702_analizar import correlacion_parcial   # sólo import

BASE = Path("/Users/alexis/phantom_cs073/bateria_fase8_f801_desacople")
RUTA_CRUDO = f"{HERE}/cs090_fase8_f801_phantom_crudo.csv"
RUTA_POR_GRAFO = f"{HERE}/cs090_fase8_f801_por_grafo.csv"
RUTA_ESTAD = f"{HERE}/cs090_fase8_f801_estadistica.csv"
RUTA_DESACOPLE = f"{HERE}/cs090_fase8_f801_matriz_desacople.csv"
RUTA_CORREL = f"{HERE}/cs090_fase8_f801_correlaciones.csv"
RUTA_PNG = f"{HERE}/cs090_fase8_f801_desacople.png"

BRAZOS = ("abanico", "disp", "cola", "malla", "solap")
GRANO_PARTICULA = 0.0005

# las cuatro medidas que la tarea pide desacoplar, más la cola (que F7-03 no medía)
CUATRO = [
    ("tri_por_arista_media", "A  triángulos por arista con triángulo"),
    ("gini_tri_nodo", "B  Gini de concentración por nodo"),
    ("frac_aristas_multi_tri", "C  solapamiento: aristas en >1 triángulo"),
    ("frac_aristas_en_triangulo", "D  soporte: fracción de aristas con triángulo"),
    ("tri_por_arista_max", "E  cola: máximo de triángulos en una arista"),
]

MEDIDAS = CUATRO + [
    ("frac_nodos_en_triangulo", "soporte: nodos que tocan algún triángulo"),
    ("tri_por_nodo_max", "máximo de triángulos en un solo nodo"),
    ("gini_tri_arista", "Gini de la carga por arista"),
    ("frac_aristas_carga3mas", "fracción de aristas del soporte con >=3"),
    ("n_comp_tri", "nº de cúmulos de triángulos"),
    ("frac_mayor_comp_tri", "fracción de triángulos en el cúmulo mayor"),
    ("modularidad_tri", "modularidad de la partición inducida"),
    ("dist_media_tri", "distancia media entre triángulos (saltos)"),
    ("clustering_local", "clustering local medio (NO explicativa: falla en el signo)"),
    ("transitividad", "transitividad global (fija por diseño)"),
    ("asortatividad", "asortatividad de grados"),
    ("pendiente_corr", "pendiente corregida log(diám)-log(N_cajas)"),
    ("gigante", "tamaño de la componente gigante"),
    ("solapamiento_aristas", "aristas compartidas con el grafo original"),
]

# los contrastes declarados ANTES de mirar la masa, con qué pretende aislar cada uno
CONTRASTES = [
    ("abanico", "disp", "C1  Gini por nodo (las tres medidas de arista fijas POR CONSTRUCCIÓN)"),
    ("cola", "malla", "C2  cola (máximo por arista) contra media (fracción con >=2)"),
    ("abanico", "malla", "C3  solapamiento, con la concentración por nodo lo más pareja posible"),
    ("solap", "disp", "C4  ancla: el eje de F7-03 sobre estos mismos grafos base"),
    ("cola", "disp", "C5  control de ruido: dos brazos que salieron casi IDÉNTICOS"),
]


def cargar_estructura():
    est = {}
    for p in sorted(Path(HERE).glob("cs090_fase8_f801_estructura*.csv")):
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
    print(f"[f801] estructura: {len(estructura)} filas (rule_id, seed, brazo)")

    metas, problemas = {}, []
    for carpeta in sorted(c for c in BASE.iterdir() if c.is_dir()):
        mp = carpeta / "meta_regla.json"
        if not mp.exists():
            problemas.append(f"{carpeta.name}: sin meta_regla.json")
            continue
        m = json.loads(mp.read_text())
        if m.get("tarea") != "FASE8_F801_desacople_apinamiento":
            problemas.append(f"{carpeta.name}: tarea declarada = {m.get('tarea')}")
            continue
        if Path(m.get("carpeta", "")).name != carpeta.name:
            problemas.append(f"{carpeta.name}: el meta declara carpeta={m.get('carpeta')}")
            continue
        esperado = f"{m['rule_id']}_s{m['seed']}_f801_{m['brazo']}"
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
            for col, _e in MEDIDAS:
                if col not in fila:
                    fila[col] = _f(st, col)
            for col in ("asortatividad", "n_componentes", "dist_media_azar", "t_piso",
                        "n_triangulos_original", "clustering_original", "identidad_A_D_resid",
                        "n_aristas_soporte", "tri_por_arista_p90", "frac_carga_en_top1pct"):
                fila[col] = _f(st, col)
            filas.append(fila)

    print(f"[f801] {len(grafos_completos)} grafos con los 5 brazos; {len(filas)} corridas; "
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
        print("[f801] todavía no hay ningún grafo con los 5 brazos -- nada que analizar")
        return

    sub_de = {}
    for (rid, seed) in grafos_completos:
        sub_de[(rid, seed)] = {f["brazo"]: f for f in filas
                               if f["rule_id"] == rid and f["seed"] == seed}
    n = len(grafos_completos)

    # ================= 0) CONTROL DE LA IDENTIDAD ALGEBRAICA A·D·m = 3T =================
    resid = np.array([abs(_f(sub_de[k][b], "identidad_A_D_resid")) for k in sub_de for b in BRAZOS])
    print(f"\n[control] identidad A·D·m = 3T: |residuo| máximo = {np.nanmax(resid):.3g} "
          f"(si es ~0, A y D son la MISMA variable y su inversa: no se pueden separar)")

    # ================= 1) MATRIZ DE DESACOPLE (antes de mirar la masa) =================
    # rango total que cada medida recorre entre los cinco brazos, promediado sobre grafos: la vara
    # contra la que se juzga si una medida "quedó fija" en un contraste.
    rango_medida = {}
    for col, _e in MEDIDAS:
        rr = []
        for k in sub_de:
            v = np.array([_f(sub_de[k][b], col) for b in BRAZOS], dtype=float)
            if np.isfinite(v).all():
                rr.append(v.max() - v.min())
        rango_medida[col] = float(np.mean(rr)) if rr else float("nan")

    desacople = []
    for a, b, etiqueta in CONTRASTES:
        for col, nombre in MEDIDAS:
            va = np.array([_f(sub_de[k][a], col) for k in sub_de], dtype=float)
            vb = np.array([_f(sub_de[k][b], col) for k in sub_de], dtype=float)
            d = va - vb
            rng = rango_medida[col]
            desacople.append(dict(
                contraste=f"{a} - {b}", objetivo=etiqueta, medida=col, descripcion=nombre,
                media_a=float(np.nanmean(va)), media_b=float(np.nanmean(vb)),
                delta=float(np.nanmean(d)),
                delta_rel_rango=float(np.nanmean(d) / rng) if rng and np.isfinite(rng) and rng > 0
                else float("nan"),
                rango_entre_brazos=rng,
            ))
    with open(RUTA_DESACOPLE, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(desacople[0].keys()))
        w.writeheader()
        w.writerows(desacople)
    print(f"[csv] {RUTA_DESACOPLE.split('/')[-1]}")

    print("\n--- MATRIZ DE DESACOPLE: |Δ| relativo al rango entre los 5 brazos (0 = quedó fija) ---")
    print(f"    {'contraste':<20}" + "".join(f"{c.split('_')[0][:9]:>11}" for c, _ in CUATRO))
    for a, b, etiqueta in CONTRASTES:
        linea = f"    {a+' - '+b:<20}"
        for col, _e in CUATRO:
            fila = next(r for r in desacople if r["contraste"] == f"{a} - {b}" and r["medida"] == col)
            linea += f"{fila['delta_rel_rango']:>11.3f}"
        print(linea + f"   [{etiqueta}]")
    print("    (columnas: A=tri/arista  B=gini_nodo  C=frac_multi  D=frac_aristas  E=max_arista)")

    # ================= 2) por grafo + tests pareados =================
    por_grafo, M = [], []
    for (rid, seed) in grafos_completos:
        sub = sub_de[(rid, seed)]
        y = np.array([sub[b]["frac_masa"] for b in BRAZOS])
        M.append(y)
        tris = [sub[b]["n_triangulos"] for b in BRAZOS]
        fila = dict(
            rule_id=rid, seed=seed, lote=sub[BRAZOS[0]]["lote"],
            n_aristas=sub[BRAZOS[0]]["n_aristas"],
            n_triangulos_min=min(tris), n_triangulos_max=max(tris),
            dif_triangulos=max(tris) - min(tris),
            **{f"masa_{b}": float(sub[b]["frac_masa"]) for b in BRAZOS},
            **{f"gini_{b}": float(sub[b]["gini_tri_nodo"]) for b in BRAZOS},
            **{f"multi_{b}": float(sub[b]["frac_aristas_multi_tri"]) for b in BRAZOS},
            **{f"A_{b}": float(sub[b]["tri_por_arista_media"]) for b in BRAZOS},
            **{f"maxE_{b}": float(sub[b]["tri_por_arista_max"]) for b in BRAZOS},
            **{f"C_{b}": float(sub[b]["clustering_local"]) for b in BRAZOS},
            **{f"gigante_{b}": int(sub[b]["gigante"]) for b in BRAZOS},
            rango_masa=float(y.max() - y.min()),
            rango_masa_particulas=float((y.max() - y.min()) / GRANO_PARTICULA),
            brazo_masa_max=BRAZOS[int(np.argmax(y))], brazo_masa_min=BRAZOS[int(np.argmin(y))],
        )
        for a, b, _e in CONTRASTES:
            fila[f"d_{a}_{b}"] = float(sub[a]["frac_masa"] - sub[b]["frac_masa"])
        por_grafo.append(fila)
    with open(RUTA_POR_GRAFO, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(por_grafo[0].keys()))
        w.writeheader()
        w.writerows(por_grafo)
    print(f"[csv] {RUTA_POR_GRAFO.split('/')[-1]} ({len(por_grafo)} grafos)")

    M = np.vstack(M)
    print("\n--- fracción de masa en sumideros por brazo (una fila = un grafo) ---")
    print("    " + " " * 26 + "  ".join(f"{b:>8}" for b in BRAZOS))
    for g, fl in zip(por_grafo, M):
        print(f"    {g['rule_id'].replace('A2-B0-C2-',''):<10} s{g['seed']} T={g['n_triangulos_min']:<5}"
              + "  ".join(f"{v:8.4f}" for v in fl)
              + f"   rango={g['rango_masa_particulas']:.1f} part.")
    print("    media:" + " " * 22 + "  ".join(f"{v:8.4f}" for v in M.mean(axis=0)))

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
                                    f"signos {sg}/{n}, binomial p={pb:.4g}, "
                                    f"|Δ| min={np.abs(d).min()/GRANO_PARTICULA:.1f} part."))
        return d

    for a, b, etiqueta in CONTRASTES:
        par(a, b, f"{a} vs {b}  [{etiqueta}]")
    # todos los pares restantes, para no elegir a dedo
    for i in range(len(BRAZOS)):
        for j in range(i + 1, len(BRAZOS)):
            a, b = BRAZOS[i], BRAZOS[j]
            if not any((a, b) == (x, y) or (b, a) == (x, y) for x, y, _e in CONTRASTES):
                par(a, b, f"{a} vs {b}  (par adicional)")

    rangos = np.array([g["rango_masa_particulas"] for g in por_grafo])
    resumen.append(dict(prueba="rango masa max-min dentro de cada grafo",
                        estadistico=float(rangos.mean()), p=float("nan"), n=n,
                        efecto_particulas=float(rangos.mean()),
                        detalle=f"mediana={np.median(rangos):.1f} part., min={rangos.min():.1f}, "
                                f"max={rangos.max():.1f} (grano = 1 partícula = {GRANO_PARTICULA})"))
    resumen.append(dict(prueba="diferencia de triángulos entre brazos (control del diseño)",
                        estadistico=float(np.mean([g["dif_triangulos"] for g in por_grafo])),
                        p=float("nan"), n=n, efecto_particulas=float("nan"),
                        detalle=f"máx entre grafos = {max(g['dif_triangulos'] for g in por_grafo)}"))
    with open(RUTA_ESTAD, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(resumen[0].keys()))
        w.writeheader()
        w.writerows(resumen)
    print(f"\n[csv] {RUTA_ESTAD.split('/')[-1]}")
    for r in resumen:
        print(f"   {r['prueba']:<62} est={r['estadistico']:<11.5f} p={r['p']:<10.4g} ({r['detalle']})")

    # ================= 3) ¿qué medida sigue la masa? Spearman centrado por grafo =================
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
        if len(xs) < 6:
            correls.append(dict(medida=col, descripcion=etiqueta, rho=float("nan"), p=float("nan"),
                                n=len(xs), nota="sin variación entre brazos"))
            continue
        rho, p = stats.spearmanr(xs, ys)
        correls.append(dict(medida=col, descripcion=etiqueta, rho=float(rho), p=float(p),
                            n=len(xs), nota=""))
    correls.sort(key=lambda d: -abs(d["rho"]) if np.isfinite(d["rho"]) else 0)
    with open(RUTA_CORREL, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(correls[0].keys()))
        w.writeheader()
        w.writerows(correls)
    print(f"\n[csv] {RUTA_CORREL.split('/')[-1]}")
    print("--- Spearman centrado por grafo (masa contra cada medida) ---")
    for c in correls:
        print(f"   {c['descripcion']:<52} rho={c['rho']:+.3f}  p={c['p']:.3g}  n={c['n']} {c['nota']}")

    # ================= 4) PNG =================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 2, figsize=(13.5, 9.5))
        # (a) masa por brazo, pareado
        a0 = ax[0, 0]
        for fl in M:
            a0.plot(range(len(BRAZOS)), fl, "o-", color="0.7", lw=1, ms=4, alpha=0.8)
        a0.plot(range(len(BRAZOS)), M.mean(axis=0), "o-", color="crimson", lw=2.5, ms=8, label="media")
        a0.set_xticks(range(len(BRAZOS))); a0.set_xticklabels(BRAZOS)
        a0.set_ylabel("fracción de masa en sumideros")
        a0.set_title(f"F8-01: masa por brazo (n={n} grafos; grados y nº de triángulos fijos)")
        a0.legend(); a0.grid(alpha=0.3)

        # (b) matriz de desacople
        a1 = ax[0, 1]
        etiquetas = [c for c, _e in CUATRO]
        Mat = np.array([[next(r["delta_rel_rango"] for r in desacople
                              if r["contraste"] == f"{a} - {b}" and r["medida"] == col)
                         for col in etiquetas] for a, b, _e in CONTRASTES])
        im = a1.imshow(np.abs(Mat), cmap="magma_r", vmin=0, vmax=1.0)
        a1.set_xticks(range(len(etiquetas)))
        a1.set_xticklabels(["A tri/ar", "B gini_n", "C multi", "D soporte", "E max_ar"], rotation=20)
        a1.set_yticks(range(len(CONTRASTES)))
        a1.set_yticklabels([f"{a}-{b}" for a, b, _e in CONTRASTES])
        for i in range(Mat.shape[0]):
            for j in range(Mat.shape[1]):
                a1.text(j, i, f"{Mat[i, j]:+.2f}", ha="center", va="center",
                        color="white" if abs(Mat[i, j]) > 0.5 else "black", fontsize=9)
        a1.set_title("Δ de cada medida / rango entre brazos\n(0 = la medida quedó FIJA en ese contraste)")
        fig.colorbar(im, ax=a1, shrink=0.8)

        # (c) efecto de cada contraste en partículas
        a2 = ax[1, 0]
        etis, vals, errs = [], [], []
        for a, b, _e in CONTRASTES:
            d = np.array([g[f"d_{a}_{b}"] for g in por_grafo]) / GRANO_PARTICULA
            etis.append(f"{a}\n−{b}"); vals.append(d.mean())
            errs.append(d.std(ddof=1) / np.sqrt(len(d)))
        a2.bar(range(len(etis)), vals, yerr=errs, color=["#3b7dd8", "#d8863b", "#57a773", "#a3546e"],
               capsize=4)
        a2.axhline(0, color="k", lw=1)
        a2.axhline(1, color="0.5", ls=":", lw=1)
        a2.axhline(-1, color="0.5", ls=":", lw=1, label="grano = 1 partícula")
        a2.set_xticks(range(len(etis))); a2.set_xticklabels(etis, fontsize=9)
        a2.set_ylabel("Δ masa (partículas de 2000)")
        a2.set_title("efecto de cada contraste, en partículas (±EE)")
        a2.legend(); a2.grid(alpha=0.3, axis="y")

        # (d) correlaciones
        a3 = ax[1, 1]
        cc = [c for c in correls if np.isfinite(c["rho"])][:12][::-1]
        a3.barh(range(len(cc)), [c["rho"] for c in cc],
                color=["#57a773" if c["rho"] > 0 else "#c0504d" for c in cc])
        a3.set_yticks(range(len(cc)))
        a3.set_yticklabels([c["medida"][:30] for c in cc], fontsize=8)
        a3.axvline(0, color="k", lw=1)
        a3.set_xlabel("Spearman con la masa (centrado por grafo)")
        a3.set_title("qué medida sigue la masa")
        a3.grid(alpha=0.3, axis="x")

        fig.tight_layout()
        fig.savefig(RUTA_PNG, dpi=140)
        print(f"[png] {RUTA_PNG.split('/')[-1]}")
    except Exception as e:      # el PNG no puede tumbar el análisis
        print(f"[png] no se pudo dibujar: {e}")


if __name__ == "__main__":
    main()

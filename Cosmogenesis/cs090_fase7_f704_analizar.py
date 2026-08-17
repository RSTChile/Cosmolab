"""
cs090_fase7_f704_analizar.py — FASE VII, tarea F7-04: lee los volcados de Phantom de los 5 brazos,
los une con las métricas estructurales, VERIFICA que el nº de aristas sea idéntico entre brazos y
corre la estadística pareada (signos + Wilcoxon) grafo por grafo.

NO REIMPLEMENTA LA LECTURA: importa `analizar_carpeta` de `cs090_fase5b_analizar.py` (que a su vez usa
`leer_volcado_phantom.py`, congelado) — misma extracción de masa de sumideros, fracción de masa, κ_V y
número de sumideros que toda la línea Fase V-B / Fase VI. Requiere `sarracen` => se corre con
`./venv/bin/python`.

QUÉ ESCRIBE
-----------
  · `cs090_fase7_f704_phantom_crudo.csv`  — una fila por corrida (grafo × brazo), CSV CRUDO.
  · `cs090_fase7_f704_pares.csv`          — una fila por grafo, con las 5 masas lado a lado y sus Δ.
  · `cs090_fase7_f704_estadistica.csv`    — un contraste pareado por fila.

ENDPOINTS (decisión previa, no post-hoc): `fraccion_masa_en_sumideros` (endpoint primario, el mismo de
toda la línea), `kappa_v_agregado`, `n_sumideros`; y del lado estructural, **pendiente continua** y
**clustering**. En ningún punto se usa "% Clase III" — hay cuatro mediciones que muestran que ese
umbral fabrica discontinuidades inexistentes (INFORME_EQUIPO_FASE6_11ago2026_CS.md §1.2).

No declara cierre ni veredicto: tabula números y sus p-valores.
"""
from __future__ import annotations
import csv
import itertools
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, HERE)
from cs090_fase5b_analizar import analizar_carpeta          # sólo import
from leer_volcado_phantom import listar_dumps               # congelado, sólo import

BASE = Path("/Users/alexis/phantom_cs073/bateria_fase7_f704_cortar_bien")
BRAZOS = ("c2", "azar", "anticosto", "soporte", "antisoporte")
SUFIJO = "f704"

RUTA_CRUDO = f"{HERE}/cs090_fase7_f704_phantom_crudo.csv"
RUTA_PARES = f"{HERE}/cs090_fase7_f704_pares.csv"
RUTA_ESTAD = f"{HERE}/cs090_fase7_f704_estadistica.csv"
RUTA_ESTRUCTURA = f"{HERE}/cs090_fase7_f704_estructura.csv"

METRICAS = ("fraccion_masa_en_sumideros", "kappa_v_agregado", "n_sumideros", "masa_acretada_total")


def leer_estructura():
    """Consolida los CSV de estructura de todos los shards (`_piloto`, `_shard0..3`) en uno solo y lo
    devuelve indexado por (rule_id, seed) — la clave de unión correcta en esta línea; unir por
    `rule_id` solo pisa reglas distintas que comparten patrón de nombre entre lotes v1/v2 (bug O3-B).
    Acá cada rule_id aparece una vez, pero la clave se mantiene por disciplina y se verifica."""
    filas, campos = [], []
    for ruta in sorted(Path(HERE).glob("cs090_fase7_f704_estructura_*.csv")):
        with open(ruta) as f:
            for r in csv.DictReader(f):
                filas.append(r)
                for c in r:
                    if c not in campos:
                        campos.append(c)
    with open(RUTA_ESTRUCTURA, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)
    print(f"[f704] estructura consolidada: {len(filas)} grafos base -> "
          f"{RUTA_ESTRUCTURA.split('/')[-1]}")
    claves = [(r["rule_id"], r["seed"]) for r in filas]
    assert len(set(claves)) == len(claves), f"rule_id+seed duplicados en la estructura: {claves}"
    return {r["rule_id"]: r for r in filas}


def main():
    estructura = leer_estructura()
    filas = []
    for carpeta in sorted(c for c in BASE.iterdir() if c.is_dir()):
        if not listar_dumps(carpeta):
            print(f"[{carpeta.name}] sin dumps -- ¿corriste cs090_fase7_f704_correr.py?", flush=True)
            continue
        fila = analizar_carpeta(carpeta)
        # `analizar_carpeta` lee meta_regla.json; acá se agregan las claves propias de F7-04
        import json
        meta = json.loads((carpeta / "meta_regla.json").read_text())
        fila.update(brazo=meta["brazo"], lote=meta["lote"], M_pre_poda=meta["M_pre_poda"],
                    n_cut=meta["n_cut"], n_aristas_brazo=meta["n_aristas_grafo_final"],
                    grado_medio=meta["grado_medio_grafo_final"],
                    clustering_local=meta["clustering_local"], transitividad=meta["transitividad"],
                    pendiente_corregida=meta["pendiente_corregida"])
        filas.append(fila)
        print(f"[{carpeta.name}] brazo={fila['brazo']} aristas={fila['n_aristas_brazo']} "
              f"frac_masa={fila['fraccion_masa_en_sumideros']} kappaV={fila['kappa_v_agregado']}",
              flush=True)

    campos = []
    for f in filas:
        for c in f:
            if c not in campos:
                campos.append(c)
    with open(RUTA_CRUDO, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)
    print(f"\n[f704] {len(filas)} corridas -> {RUTA_CRUDO.split('/')[-1]}")

    # ---------------- una fila por grafo base, los 5 brazos lado a lado ----------------
    por_regla = {}
    for f in filas:
        por_regla.setdefault(f["rule_id"], {})[f["brazo"]] = f

    pares = []
    for rid, d in sorted(por_regla.items()):
        if set(d) != set(BRAZOS):
            print(f"   !! {rid}: faltan brazos {set(BRAZOS) - set(d)} -- queda fuera del pareado")
            continue
        e = estructura.get(rid, {})
        fila = dict(rule_id=rid, lote=d["c2"]["lote"], seed=e.get("seed"),
                    K=d["c2"]["K"], kcap=d["c2"]["kcap"],
                    M_pre_poda=d["c2"]["M_pre_poda"], n_cut=d["c2"]["n_cut"])
        # VERIFICACIÓN de M idéntico, leída del meta de cada corrida (no del script que las generó)
        aristas = {b: d[b]["n_aristas_brazo"] for b in BRAZOS}
        fila.update({f"aristas_{b}": v for b, v in aristas.items()})
        fila["M_identico_en_los_5"] = len(set(aristas.values())) == 1
        for b in BRAZOS:
            for met in METRICAS:
                fila[f"{met}_{b}"] = d[b][met]
            fila[f"clustering_{b}"] = d[b]["clustering_local"]
            fila[f"pendiente_{b}"] = d[b]["pendiente_corregida"]
        # Δ contra c2 y contra azar en el endpoint primario
        for b in BRAZOS:
            if b != "c2":
                fila[f"d_frac_masa_c2_menos_{b}"] = (fila["fraccion_masa_en_sumideros_c2"]
                                                     - fila[f"fraccion_masa_en_sumideros_{b}"])
        fila["orden_c2_gt_azar_gt_anticosto"] = bool(
            fila["fraccion_masa_en_sumideros_c2"] > fila["fraccion_masa_en_sumideros_azar"] >
            fila["fraccion_masa_en_sumideros_anticosto"])
        fila["orden_c2_gt_azar_gt_antisoporte"] = bool(
            fila["fraccion_masa_en_sumideros_c2"] > fila["fraccion_masa_en_sumideros_azar"] >
            fila["fraccion_masa_en_sumideros_antisoporte"])
        fila["orden_soporte_gt_azar_gt_antisoporte"] = bool(
            fila["fraccion_masa_en_sumideros_soporte"] > fila["fraccion_masa_en_sumideros_azar"] >
            fila["fraccion_masa_en_sumideros_antisoporte"])
        pares.append(fila)

    with open(RUTA_PARES, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pares[0].keys()))
        w.writeheader()
        w.writerows(pares)
    print(f"[f704] {len(pares)} grafos con los 5 brazos -> {RUTA_PARES.split('/')[-1]}")

    # ---------------- estadística pareada ----------------
    def pareado(met, a, b):
        xa = np.array([p[f"{met}_{a}"] for p in pares], dtype=float)
        xb = np.array([p[f"{met}_{b}"] for p in pares], dtype=float)
        ok = np.isfinite(xa) & np.isfinite(xb)
        xa, xb = xa[ok], xb[ok]
        d = xa - xb
        n = len(d)
        n_pos = int((d > 0).sum()); n_neg = int((d < 0).sum()); n_cero = int((d == 0).sum())
        p_signos = (stats.binomtest(n_pos, n_pos + n_neg, 0.5).pvalue
                    if (n_pos + n_neg) > 0 else float("nan"))
        try:
            p_wil = float(stats.wilcoxon(xa, xb, zero_method="wilcox").pvalue)
        except ValueError:
            p_wil = float("nan")
        return dict(metrica=met, brazo_a=a, brazo_b=b, n=n,
                    media_a=float(np.mean(xa)), media_b=float(np.mean(xb)),
                    delta_medio=float(np.mean(d)), delta_mediano=float(np.median(d)),
                    delta_rel_pct=float(100.0 * np.mean(d) / np.mean(xb)) if np.mean(xb) else float("nan"),
                    n_a_gana=n_pos, n_b_gana=n_neg, n_empates=n_cero,
                    p_signos=float(p_signos), p_wilcoxon=p_wil)

    contrastes = []
    for met in ("fraccion_masa_en_sumideros", "kappa_v_agregado", "n_sumideros"):
        for a, b in itertools.combinations(BRAZOS, 2):
            contrastes.append(pareado(met, a, b))

    with open(RUTA_ESTAD, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(contrastes[0].keys()))
        w.writeheader()
        w.writerows(contrastes)

    print(f"\n=== contrastes pareados, endpoint primario (fracción de masa en sumideros), "
          f"n={len(pares)} grafos ===")
    for c in contrastes:
        if c["metrica"] != "fraccion_masa_en_sumideros":
            continue
        print(f"  {c['brazo_a']:12s} vs {c['brazo_b']:12s}  "
              f"medias {c['media_a']:.5f} / {c['media_b']:.5f}  Δ={c['delta_medio']:+.5f} "
              f"({c['delta_rel_pct']:+.2f}%)  signos {c['n_a_gana']}/{c['n_a_gana']+c['n_b_gana']} "
              f"p={c['p_signos']:.4f}  Wilcoxon p={c['p_wilcoxon']:.4f}")

    n_ord1 = sum(p["orden_c2_gt_azar_gt_anticosto"] for p in pares)
    n_ord2 = sum(p["orden_c2_gt_azar_gt_antisoporte"] for p in pares)
    n_ord3 = sum(p["orden_soporte_gt_azar_gt_antisoporte"] for p in pares)
    print(f"\n  orden c2 > azar > anticosto   se cumple en {n_ord1}/{len(pares)} grafos "
          f"(esperado por azar: {len(pares)/6:.1f})")
    print(f"  orden c2 > azar > antisoporte se cumple en {n_ord2}/{len(pares)}")
    print(f"  orden soporte > azar > antisop se cumple en {n_ord3}/{len(pares)}")
    print(f"  M idéntico en los 5 brazos: {sum(p['M_identico_en_los_5'] for p in pares)}/{len(pares)} grafos")

    # ---- ¿la manipulación estructural funcionó? (chequeo de que los brazos SON distintos) ----
    print("\n=== la manipulación, comprobada: medias por brazo (mismo nº de aristas en los 5) ===")
    print(f"  {'brazo':13s} {'aristas':>8s} {'clustering':>11s} {'pendiente':>10s} "
          f"{'frac_masa':>10s} {'kappa_V':>8s} {'n_sink':>7s}")
    for b in BRAZOS:
        ar = np.mean([p[f"aristas_{b}"] for p in pares])
        cl = np.mean([float(p[f"clustering_{b}"]) for p in pares])
        pe = np.mean([float(p[f"pendiente_{b}"]) for p in pares])
        fm = np.mean([p[f"fraccion_masa_en_sumideros_{b}"] for p in pares])
        kv = np.nanmean([p[f"kappa_v_agregado_{b}"] if p[f"kappa_v_agregado_{b}"] is not None
                         else np.nan for p in pares])
        ns = np.mean([p[f"n_sumideros_{b}"] for p in pares])
        print(f"  {b:13s} {ar:8.1f} {cl:11.5f} {pe:10.4f} {fm:10.5f} {kv:8.4f} {ns:7.2f}")

    # ---- omnibus: ¿hay CUALQUIER efecto de brazo? (Friedman, bloques = grafo base) ----
    print("\n=== omnibus Friedman (bloques = grafo base, 5 brazos) ===")
    for met in ("fraccion_masa_en_sumideros", "n_sumideros"):
        mat = [[p[f"{met}_{b}"] for p in pares] for b in BRAZOS]
        fr = stats.friedmanchisquare(*mat)
        print(f"  {met:32s} chi2={fr.statistic:.3f} p={fr.pvalue:.4f}")

    # ---- resolución del endpoint: la masa está cuantizada en unidades de partícula ----
    todas = np.array([p[f"fraccion_masa_en_sumideros_{b}"] for p in pares for b in BRAZOS])
    print(f"\n  resolución del endpoint: 1 partícula = 1/2000 = 0.0005 de fracción de masa; "
          f"rango observado {todas.min():.4f}-{todas.max():.4f}; "
          f"|Δ| medio entre brazos ≈ {np.mean([abs(p['d_frac_masa_c2_menos_azar']) for p in pares]):.5f} "
          f"(≈{np.mean([abs(p['d_frac_masa_c2_menos_azar']) for p in pares])/0.0005:.1f} partículas)")

    # ---- test directo de la hipótesis de O3-B: a nº de aristas FIJO, ¿la masa sigue al clustering? ----
    # Se centra cada variable DENTRO de su grafo base (efecto fijo por grafo): así el contraste es
    # puramente entre brazos del mismo grafo, que es donde el nº de aristas está fijado por diseño.
    xs, ys, zs = [], [], []
    rhos_intra = []
    for p in pares:
        cl = np.array([float(p[f"clustering_{b}"]) for b in BRAZOS])
        pe = np.array([float(p[f"pendiente_{b}"]) for b in BRAZOS])
        fm = np.array([p[f"fraccion_masa_en_sumideros_{b}"] for b in BRAZOS])
        xs.append(cl - cl.mean()); ys.append(fm - fm.mean()); zs.append(pe - pe.mean())
        rhos_intra.append(stats.spearmanr(cl, fm).statistic)
    xs = np.concatenate(xs); ys = np.concatenate(ys); zs = np.concatenate(zs)
    r_cl = stats.spearmanr(xs, ys); r_pe = stats.spearmanr(zs, ys)
    rhos_intra = np.array(rhos_intra, dtype=float)
    print("\n=== ¿la masa sigue al CLUSTERING con nº de aristas fijo? (centrado por grafo, 60 corridas) ===")
    print(f"  clustering vs fracción de masa: Spearman ρ={r_cl.statistic:+.3f} (p={r_cl.pvalue:.4f})")
    print(f"  pendiente  vs fracción de masa: Spearman ρ={r_pe.statistic:+.3f} (p={r_pe.pvalue:.4f})")
    ok = np.isfinite(rhos_intra)
    print(f"  ρ intra-grafo (5 brazos c/u): mediana {np.median(rhos_intra[ok]):+.3f}, "
          f"positivo en {int((rhos_intra[ok] > 0).sum())}/{int(ok.sum())} grafos, "
          f"signos p={stats.binomtest(int((rhos_intra[ok] > 0).sum()), int(ok.sum()), 0.5).pvalue:.4f}")

    # correlación parcial (rangos) para separar clustering de pendiente, que en estos brazos NO están
    # fuertemente ligados entre sí -- se informa su ρ para que se vea
    def _parcial(a, b, c):
        ra, rb, rc = stats.rankdata(a), stats.rankdata(b), stats.rankdata(c)
        res = lambda u, v: u - np.polyval(np.polyfit(v, u, 1), v)
        return stats.pearsonr(res(ra, rc), res(rb, rc))
    print(f"  (clustering vs pendiente, centrados: ρ={stats.spearmanr(xs, zs).statistic:+.3f})")
    r1 = _parcial(xs, ys, zs); r2 = _parcial(zs, ys, xs)
    print(f"  parcial clustering vs masa | pendiente : r={r1[0]:+.3f} (p={r1[1]:.4f})")
    print(f"  parcial pendiente  vs masa | clustering: r={r2[0]:+.3f} (p={r2[1]:.4f})")

    # test de permutación que RESPETA los bloques: baraja la masa sólo DENTRO de cada grafo base
    # (el null correcto acá; el p de Spearman sobre 60 puntos supone independencia que no hay)
    rng = np.random.default_rng(7)
    obs = stats.spearmanr(xs, ys).statistic
    n_perm, cnt = 20000, 0
    for _ in range(n_perm):
        yp = np.concatenate([rng.permutation(ys[i * len(BRAZOS):(i + 1) * len(BRAZOS)])
                             for i in range(len(pares))])
        if stats.spearmanr(xs, yp).statistic >= obs:
            cnt += 1
    print(f"  permutación intra-grafo ({n_perm}): p_una_cola = {cnt / n_perm:.4f}")

    # correlación del Δ con la variable que O3-B dejó como candidata (clustering) y con la pendiente
    print("\n=== ¿qué acompaña al Δ de masa? (Spearman sobre los mismos grafos) ===")
    for b in ("azar", "anticosto", "soporte", "antisoporte"):
        d = np.array([p[f"fraccion_masa_en_sumideros_c2"] - p[f"fraccion_masa_en_sumideros_{b}"]
                      for p in pares], dtype=float)
        dcl = np.array([float(p["clustering_c2"]) - float(p[f"clustering_{b}"]) for p in pares])
        dpe = np.array([float(p["pendiente_c2"]) - float(p[f"pendiente_{b}"]) for p in pares])
        r_cl = stats.spearmanr(d, dcl)
        r_pe = stats.spearmanr(d, dpe)
        print(f"  c2−{b:12s}: Δmasa vs Δclustering ρ={r_cl.statistic:+.3f} (p={r_cl.pvalue:.3f}) | "
              f"Δmasa vs Δpendiente ρ={r_pe.statistic:+.3f} (p={r_pe.pvalue:.3f})")

    return pares, contrastes


if __name__ == "__main__":
    main()

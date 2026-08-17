"""
cs090_fase7_f706_analizar.py — FASE VII, tarea F7-06: lee los volcados de Phantom de los 5 brazos, los
une con las métricas estructurales y corre la estadística PAREADA (signos + Wilcoxon) grafo por grafo.

LA PREGUNTA
-----------
Con exactamente el MISMO multiconjunto de capacidades por nodo, ¿importa **a qué nodo** le toca cada
cupo? Tres asignaciones: alineada con el grado inicial (`alineado`), barajada (`permutado`), invertida
(`anti`). Y, porque la asignación cambia cuántas aristas sobreviven (confound medido: `alineado` >
`permutado` > `anti` en 12/12 grafos), dos brazos con la densidad igualada por dilución al azar
(`alin_dil`, `perm_dil`), que son el contraste LIMPIO.

NO REIMPLEMENTA LA LECTURA: importa `analizar_carpeta` de `cs090_fase5b_analizar.py` (que a su vez usa
`leer_volcado_phantom.py`, congelado) — misma extracción de masa de sumideros, fracción de masa, κ_V y
número de sumideros que toda la línea Fase V-B / Fase VI / Fase VII. Requiere `sarracen` => se corre
con `./venv/bin/python`.

QUÉ ESCRIBE
-----------
  · `cs090_fase7_f706_phantom_crudo.csv`  — una fila por corrida (grafo × brazo), CSV CRUDO.
  · `cs090_fase7_f706_pares.csv`          — una fila por grafo, los 5 brazos lado a lado.
  · `cs090_fase7_f706_estadistica.csv`    — un contraste pareado por fila.

ENDPOINTS (decisión previa, no post-hoc): `fraccion_masa_en_sumideros` (primario, el mismo de toda la
línea), `kappa_v_agregado`, `n_sumideros`; del lado estructural, **pendiente continua corregida** y
clustering. En ningún punto se usa "% Clase III".

GRANO DEL INSTRUMENTO: la masa está cuantizada en partículas — 1 partícula = 1/2000 = 0.0005 de
fracción de masa (`FASE7_F704_cortar_bien_vs_azar_CS.md`). Todo Δ se reporta también en unidades de
partícula, para poder decir "esto es más chico que lo que el instrumento resuelve" en vez de vender un
nulo que es falta de resolución.

No declara cierre ni veredicto: tabula números y sus p-valores.
"""
from __future__ import annotations
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, HERE)
from cs090_fase5b_analizar import analizar_carpeta          # sólo import
from leer_volcado_phantom import listar_dumps               # congelado, sólo import

BASE = Path("/Users/alexis/phantom_cs073/bateria_fase7_f706_cupos_alineados")
BRAZOS = ("alineado", "permutado", "anti", "alin_dil", "perm_dil")
GRANO = 0.0005          # 1 partícula = 1/2000 de la masa total (masa fija 18800 repartida en N=2000)

RUTA_CRUDO = f"{HERE}/cs090_fase7_f706_phantom_crudo.csv"
RUTA_PARES = f"{HERE}/cs090_fase7_f706_pares.csv"
RUTA_ESTAD = f"{HERE}/cs090_fase7_f706_estadistica.csv"
RUTA_ESTRUCTURA = f"{HERE}/cs090_fase7_f706_estructura.csv"

METRICAS = ("fraccion_masa_en_sumideros", "kappa_v_agregado", "n_sumideros", "masa_acretada_total")

# contrastes pareados de interés, declarados ANTES de mirar los datos
CONTRASTES_CLAVE = [
    ("alin_dil", "perm_dil", "LIMPIO: misma densidad exacta, mismo tratamiento (ambos diluidos)"),
    ("alineado", "permutado", "CRUDO: confundido con densidad (alineado tiene más aristas)"),
    ("alineado", "anti", "CRUDO: confundido con densidad"),
    ("permutado", "anti", "CRUDO: confundido con densidad"),
    ("alin_dil", "anti", "misma densidad, pero `anti` no fue diluido"),
    ("perm_dil", "anti", "misma densidad, pero `anti` no fue diluido"),
]


def leer_estructura():
    """Consolida los CSV de estructura de todos los shards en uno solo, indexado por (rule_id, seed) —
    la clave de unión correcta en esta línea; unir por `rule_id` solo pisa reglas distintas que
    comparten patrón de nombre entre lotes v1/v2 (bug O3-B)."""
    filas, campos = [], []
    for ruta in sorted(Path(HERE).glob("cs090_fase7_f706_estructura_shard*.csv")):
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
    print(f"[f706] estructura consolidada: {len(filas)} grafos base -> {RUTA_ESTRUCTURA.split('/')[-1]}")
    claves = [(r["rule_id"], r["seed"]) for r in filas]
    assert len(set(claves)) == len(claves), f"rule_id+seed duplicados: {claves}"
    return {r["rule_id"]: r for r in filas}


def pareado(pares, met, a, b):
    xa = np.array([p[f"{met}_{a}"] for p in pares], dtype=float)
    xb = np.array([p[f"{met}_{b}"] for p in pares], dtype=float)
    ok = np.isfinite(xa) & np.isfinite(xb)
    xa, xb = xa[ok], xb[ok]
    d = xa - xb
    n = len(d)
    n_pos = int((d > 0).sum()); n_neg = int((d < 0).sum()); n_cero = int((d == 0).sum())
    p_signos = (stats.binomtest(n_pos, n_pos + n_neg, 0.5).pvalue if (n_pos + n_neg) > 0 else float("nan"))
    try:
        p_wil = float(stats.wilcoxon(xa, xb, zero_method="wilcox").pvalue)
    except ValueError:
        p_wil = float("nan")
    return dict(metrica=met, brazo_a=a, brazo_b=b, n=n,
                media_a=float(np.mean(xa)), media_b=float(np.mean(xb)),
                delta_medio=float(np.mean(d)), delta_mediano=float(np.median(d)),
                delta_sd=float(np.std(d, ddof=1)) if n > 1 else float("nan"),
                delta_rel_pct=float(100.0 * np.mean(d) / np.mean(xb)) if np.mean(xb) else float("nan"),
                delta_en_particulas=(float(np.mean(d) / GRANO)
                                     if met == "fraccion_masa_en_sumideros" else float("nan")),
                n_a_gana=n_pos, n_b_gana=n_neg, n_empates=n_cero,
                p_signos=float(p_signos), p_wilcoxon=p_wil)


def main():
    estructura = leer_estructura()
    filas = []
    for carpeta in sorted(c for c in BASE.iterdir() if c.is_dir()):
        if not listar_dumps(carpeta):
            print(f"[{carpeta.name}] sin dumps -- ¿corriste cs090_fase7_f706_correr.py?", flush=True)
            continue
        fila = analizar_carpeta(carpeta)
        meta = json.loads((carpeta / "meta_regla.json").read_text())
        fila.update(brazo=meta["brazo"], brazo_origen=meta["brazo_origen"], diluido=meta["diluido"],
                    lote=meta["lote"], n_aristas_brazo=meta["n_aristas_grafo_final"],
                    grado_medio=meta["grado_medio_grafo_final"],
                    clustering_local=meta["clustering_local"], transitividad=meta["transitividad"],
                    pendiente_corregida=meta["pendiente_corregida"],
                    cupo_cv=meta["cupo_cv"], rho_cupo_grado=meta["rho_cupo_grado"],
                    n_quitadas_al_azar=meta["n_quitadas_al_azar"])
        filas.append(fila)
        print(f"[{carpeta.name}] brazo={fila['brazo']:10s} aristas={fila['n_aristas_brazo']} "
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
    print(f"\n[f706] {len(filas)} corridas -> {RUTA_CRUDO.split('/')[-1]}")

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
        fila = dict(rule_id=rid, lote=d["alineado"]["lote"], seed=e.get("seed"),
                    K=d["alineado"]["K"], kcap=d["alineado"]["kcap"],
                    grado_inicial_medio=e.get("grado_inicial_medio"),
                    M_objetivo_diluido=e.get("M_objetivo_diluido"),
                    n_aristas_unif=e.get("n_aristas_unif"),
                    unif_reproduce_motor=e.get("unif_reproduce_motor"),
                    meta5b_n_aristas_coincide=e.get("meta5b_n_aristas_coincide"),
                    multiconjunto_ok=(str(e.get("multiconjunto_alineado_vs_permutado")) == "True" and
                                      str(e.get("multiconjunto_alineado_vs_anti")) == "True"),
                    frac_masa_fase5b_unif=e.get("frac_masa_fase5b"))
        aristas = {b: d[b]["n_aristas_brazo"] for b in BRAZOS}
        fila.update({f"aristas_{b}": v for b, v in aristas.items()})
        fila["densidad_igualada_trio"] = (aristas["alin_dil"] == aristas["perm_dil"] == aristas["anti"])
        for b in BRAZOS:
            for met in METRICAS:
                fila[f"{met}_{b}"] = d[b][met]
            fila[f"clustering_{b}"] = float(d[b]["clustering_local"])
            fila[f"pendiente_{b}"] = float(d[b]["pendiente_corregida"])
            fila[f"rho_cupo_grado_{b}"] = float(d[b]["rho_cupo_grado"])
        fila["d_frac_masa_limpio"] = (fila["fraccion_masa_en_sumideros_alin_dil"]
                                      - fila["fraccion_masa_en_sumideros_perm_dil"])
        fila["d_frac_masa_crudo"] = (fila["fraccion_masa_en_sumideros_alineado"]
                                     - fila["fraccion_masa_en_sumideros_permutado"])
        pares.append(fila)

    with open(RUTA_PARES, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pares[0].keys()))
        w.writeheader()
        w.writerows(pares)
    print(f"[f706] {len(pares)} grafos con los 5 brazos -> {RUTA_PARES.split('/')[-1]}")

    n = len(pares)
    # ---------------- verificaciones agregadas ----------------
    print("\n=== verificaciones (contra disco, no contra el script que generó los datos) ===")
    print(f"  multiconjunto de cupos idéntico alineado/permutado/anti : "
          f"{sum(p['multiconjunto_ok'] for p in pares)}/{n} grafos")
    print(f"  brazo `unif` reproduce MOT.dinamica_B0 arista por arista : "
          f"{sum(str(p['unif_reproduce_motor']) == 'True' for p in pares)}/{n}")
    print(f"  nº de aristas de `unif` == meta_regla.json de Fase V-B   : "
          f"{sum(str(p['meta5b_n_aristas_coincide']) == 'True' for p in pares)}/{n}")
    print(f"  densidad EXACTAMENTE igual en alin_dil/perm_dil/anti     : "
          f"{sum(p['densidad_igualada_trio'] for p in pares)}/{n}")
    print(f"  ρ(cupo, grado inicial) medio: alineado={np.mean([p['rho_cupo_grado_alineado'] for p in pares]):+.3f}  "
          f"permutado={np.mean([p['rho_cupo_grado_permutado'] for p in pares]):+.3f}  "
          f"anti={np.mean([p['rho_cupo_grado_anti'] for p in pares]):+.3f}")

    # ---------------- el confound, cuantificado ----------------
    print("\n=== el confound de densidad, en números ===")
    print(f"  {'brazo':11s} {'aristas':>9s} {'clustering':>11s} {'pendiente':>10s} "
          f"{'frac_masa':>10s} {'kappa_V':>8s} {'n_sink':>7s}")
    for b in BRAZOS:
        ar = np.mean([p[f"aristas_{b}"] for p in pares])
        cl = np.mean([p[f"clustering_{b}"] for p in pares])
        pe = np.mean([p[f"pendiente_{b}"] for p in pares])
        fm = np.mean([p[f"fraccion_masa_en_sumideros_{b}"] for p in pares])
        kv = np.nanmean([p[f"kappa_v_agregado_{b}"] if p[f"kappa_v_agregado_{b}"] is not None
                         else np.nan for p in pares])
        ns = np.mean([p[f"n_sumideros_{b}"] for p in pares])
        print(f"  {b:11s} {ar:9.1f} {cl:11.5f} {pe:10.4f} {fm:10.5f} {kv:8.4f} {ns:7.2f}")

    todas_ar = np.array([p[f"aristas_{b}"] for p in pares for b in BRAZOS], dtype=float)
    todas_fm = np.array([p[f"fraccion_masa_en_sumideros_{b}"] for p in pares for b in BRAZOS])
    r_glob = stats.spearmanr(todas_ar, todas_fm)
    print(f"\n  masa vs aristas sobre las {len(todas_ar)} corridas: Spearman ρ={r_glob.statistic:+.3f} "
          f"(p={r_glob.pvalue:.2e})  <- por esto hacen falta los brazos diluidos")

    # ---------------- estadística pareada ----------------
    contrastes = []
    for met in ("fraccion_masa_en_sumideros", "kappa_v_agregado", "n_sumideros"):
        for a, b in itertools.combinations(BRAZOS, 2):
            contrastes.append(pareado(pares, met, a, b))
    with open(RUTA_ESTAD, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(contrastes[0].keys()))
        w.writeheader()
        w.writerows(contrastes)

    def buscar(met, a, b):
        """`itertools.combinations` sólo produce los pares en el orden de `BRAZOS`; los contrastes
        clave están escritos en el orden en que se leen mejor, así que si el par no está tabulado en
        ese sentido se calcula al vuelo (mismo cálculo, sentido invertido)."""
        for x in contrastes:
            if x["metrica"] == met and x["brazo_a"] == a and x["brazo_b"] == b:
                return x
        return pareado(pares, met, a, b)

    print(f"\n=== contrastes pareados, endpoint primario (fracción de masa en sumideros), n={n} ===")
    print(f"  grano del instrumento: 1 partícula = {GRANO} de fracción de masa\n")
    for a, b, nota in CONTRASTES_CLAVE:
        c = buscar("fraccion_masa_en_sumideros", a, b)
        print(f"  {a:10s} vs {b:10s}  medias {c['media_a']:.5f}/{c['media_b']:.5f}  "
              f"Δ={c['delta_medio']:+.5f} ({c['delta_en_particulas']:+.2f} partículas, "
              f"{c['delta_rel_pct']:+.1f}%)  signos {c['n_a_gana']}/{c['n_a_gana']+c['n_b_gana']} "
              f"p={c['p_signos']:.4f}  Wilcoxon p={c['p_wilcoxon']:.4f}")
        print(f"      {nota}")

    print(f"\n=== los mismos contrastes en κ_V y en nº de sumideros ===")
    for met in ("kappa_v_agregado", "n_sumideros"):
        for a, b, _ in CONTRASTES_CLAVE[:4]:
            c = buscar(met, a, b)
            print(f"  {met:20s} {a:10s} vs {b:10s}  Δ={c['delta_medio']:+.4f}  "
                  f"signos {c['n_a_gana']}/{c['n_a_gana']+c['n_b_gana']}  p={c['p_signos']:.4f}  "
                  f"Wilcoxon p={c['p_wilcoxon']:.4f}")

    # ---------------- ¿el Δ limpio supera el grano del instrumento? ----------------
    d_limpio = np.array([p["d_frac_masa_limpio"] for p in pares], dtype=float)
    d_crudo = np.array([p["d_frac_masa_crudo"] for p in pares], dtype=float)
    print("\n=== tamaño del efecto contra el grano del instrumento ===")
    for nombre, d in (("LIMPIO alin_dil−perm_dil", d_limpio), ("CRUDO alineado−permutado", d_crudo)):
        print(f"  {nombre:26s}: |Δ| medio={np.mean(np.abs(d)):.5f} ({np.mean(np.abs(d))/GRANO:.2f} part.) "
              f"| Δ medio={np.mean(d):+.5f} ({np.mean(d)/GRANO:+.2f} part.) "
              f"| |Δ|>1 partícula en {int((np.abs(d) > GRANO).sum())}/{len(d)} grafos "
              f"| IC95 de la media [{np.mean(d)-1.96*np.std(d,ddof=1)/np.sqrt(len(d)):+.5f}, "
              f"{np.mean(d)+1.96*np.std(d,ddof=1)/np.sqrt(len(d)):+.5f}]")

    # ---------------- ajuste por densidad del contraste CRUDO (por si el lector lo quiere así) ----
    # Regresión de la masa contra log(aristas) usando TODAS las corridas, y comparación de residuos.
    print("\n=== contraste crudo, descontando la densidad por regresión (control secundario) ===")
    x = np.log(todas_ar); y = todas_fm
    b1, b0 = np.polyfit(x, y, 1)
    r2 = 1 - np.sum((y - (b0 + b1 * x)) ** 2) / np.sum((y - y.mean()) ** 2)
    print(f"  frac_masa = {b0:+.4f} {b1:+.4f}·log(aristas)   R²={r2:.3f}  (n={len(x)})")
    res = {}
    for b in BRAZOS:
        ar = np.array([p[f"aristas_{b}"] for p in pares], dtype=float)
        fm = np.array([p[f"fraccion_masa_en_sumideros_{b}"] for p in pares], dtype=float)
        res[b] = fm - (b0 + b1 * np.log(ar))
    for a, b in (("alineado", "permutado"), ("alineado", "anti"), ("permutado", "anti")):
        d = res[a] - res[b]
        pw = float(stats.wilcoxon(res[a], res[b]).pvalue)
        ps = stats.binomtest(int((d > 0).sum()), int((d != 0).sum()), 0.5).pvalue
        print(f"  residuo {a:10s} − {b:10s}: Δ={np.mean(d):+.5f} ({np.mean(d)/GRANO:+.2f} part.)  "
              f"signos {int((d>0).sum())}/{int((d!=0).sum())} p={ps:.4f}  Wilcoxon p={pw:.4f}")

    # ---------------- ¿el efecto de la alineación sigue a la pendiente/clustering? ----------------
    print("\n=== ¿con qué se mueve el Δ limpio? (Spearman sobre los mismos grafos) ===")
    for nombre, v in (
        ("Δ pendiente (alin_dil−perm_dil)",
         np.array([p["pendiente_alin_dil"] - p["pendiente_perm_dil"] for p in pares])),
        ("Δ clustering (alin_dil−perm_dil)",
         np.array([p["clustering_alin_dil"] - p["clustering_perm_dil"] for p in pares])),
        ("kcap de la regla", np.array([float(p["kcap"]) for p in pares])),
        ("nº de aristas del trío igualado", np.array([float(p["aristas_anti"]) for p in pares])),
    ):
        r = stats.spearmanr(d_limpio, v)
        print(f"  Δmasa_limpio vs {nombre:34s} ρ={r.statistic:+.3f} (p={r.pvalue:.3f})")

    # ---------------- LA ASIMETRÍA QUE QUEDA: la DOSIS de dilución ----------------
    # Igualar la densidad tuvo un precio: para llegar al mismo nº de aristas hubo que quitarle MUCHO
    # más a `alineado` (que tenía más) que a `permutado`, y nada a `anti`. Como quitar aristas al azar
    # es en sí una intervención (el brazo `azar` de F7-04), hay que preguntarse si el orden
    # alin_dil > perm_dil > anti no es simplemente "cuántas aristas se quitaron al azar".
    print("\n=== control de la asimetría: ¿el orden a densidad igual es DOSIS DE DILUCIÓN? ===")
    dosis = {b: np.array([float(estructura[p["rule_id"]].get(f"n_quitadas_{b}", 0) or 0)
                          for p in pares]) for b in ("alin_dil", "perm_dil")}
    dosis["anti"] = np.zeros(n)
    print(f"  aristas quitadas al azar (media): alin_dil={dosis['alin_dil'].mean():.0f}  "
          f"perm_dil={dosis['perm_dil'].mean():.0f}  anti=0")
    xs, ys = [], []
    for k, p in enumerate(pares):
        d = np.array([dosis[b][k] for b in ("alin_dil", "perm_dil", "anti")])
        f = np.array([p[f"fraccion_masa_en_sumideros_{b}"] for b in ("alin_dil", "perm_dil", "anti")])
        xs.append(d - d.mean()); ys.append(f - f.mean())
    xs = np.concatenate(xs); ys = np.concatenate(ys)
    r = stats.spearmanr(xs, ys)
    print(f"  dosis vs masa, centrado por grafo (36 corridas a densidad igualada): "
          f"ρ={r.statistic:+.3f} (p={r.pvalue:.4f})")
    ddos = dosis["alin_dil"] - dosis["perm_dil"]
    r2_ = stats.spearmanr(ddos, d_limpio)
    print(f"  Δdosis (alin_dil−perm_dil) vs Δmasa limpio: ρ={r2_.statistic:+.3f} (p={r2_.pvalue:.4f})")
    # el contraste con la dosis igualada NO existe en este diseño; se declara como límite
    print("  (no hay en esta corrida ningún par con la MISMA dosis de dilución y distinto origen:")
    print("   ése es el control que este diseño no incluye — queda declarado como límite)")

    # ---------------- omnibus (bloques = grafo base) ----------------
    print("\n=== omnibus Friedman ===")
    for etiqueta, grupo in (("los 5 brazos", BRAZOS),
                            ("sólo el trío de densidad igualada", ("alin_dil", "perm_dil", "anti"))):
        mat = [[p[f"fraccion_masa_en_sumideros_{b}"] for p in pares] for b in grupo]
        fr = stats.friedmanchisquare(*mat)
        print(f"  frac_masa, {etiqueta:34s} chi2={fr.statistic:.3f} p={fr.pvalue:.4f}")

    return pares, contrastes


if __name__ == "__main__":
    main()

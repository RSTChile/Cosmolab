"""
cs090_fase5b_consolidar_20pares.py -- FASE V-B, escalado a ~20 pares TOTALES: junta los 8 pares
anteriores (cs090_fase5b_escala_v2_metricas_FINAL.csv, de FASE5B_investigacion_8sumideros_y_escala_CS.md)
con los pares NUEVOS de esta tarea (cs090_fase5b_escala_v3_metricas.csv) en un solo CSV consolidado, y
calcula la comparacion agregada final (fraccion de masa en sumideros, kappa_V) Clase III vs Clase I,
separando "todos" de "solo match exacto K=kcap" (deberia ser el mismo conjunto en esta tarea, ya que
TODOS los pares nuevos son exactos por diseno).

No modifica ningun archivo de datos anterior -- solo LEE cs090_fase5b_escala_v2_metricas_FINAL.csv (no
se toca) y escribe un CSV nuevo. No declara cierre ni veredicto -- solo tabula numeros.
"""
from __future__ import annotations
import csv
from collections import defaultdict

RUTA_V1V2 = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs090_fase5b_escala_v2_metricas_FINAL.csv"
RUTA_V3 = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs090_fase5b_escala_v3_metricas.csv"
RUTA_TOTAL = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs090_fase5b_TOTAL_20pares.csv"

CAMPOS_COMUNES = ["origen_tarea", "par", "rule_id", "rol", "clase", "seed", "K", "kcap",
                   "match_exacto_K_kcap", "fraccion_masa_en_sumideros", "n_sumideros",
                   "t_primer_sumidero", "masa_acretada_total", "kappa_v_agregado",
                   "kappa_v_medio_valido", "n_kappa_indefinidos", "carpeta"]


def cargar_v1v2():
    filas = []
    with open(RUTA_V1V2) as f:
        for row in csv.DictReader(f):
            filas.append(dict(
                origen_tarea="tarea_anterior_v1v2_8pares", par=row["par"], rule_id=row["rule_id"],
                rol=row["rol"], clase=row["clase"], seed=row["seed"], K=row["K"], kcap=row["kcap"],
                match_exacto_K_kcap=row["match_exacto_K_kcap"],
                fraccion_masa_en_sumideros=row["fraccion_masa_en_sumideros"],
                n_sumideros=row["n_sumideros"], t_primer_sumidero=row["t_primer_sumidero"],
                masa_acretada_total=row["masa_acretada_total"], kappa_v_agregado=row["kappa_v_agregado"],
                kappa_v_medio_valido=row["kappa_v_medio_valido"],
                n_kappa_indefinidos=row["n_kappa_indefinidos"], carpeta=row.get("carpeta_real", row["carpeta"])))
    return filas


def cargar_v3():
    filas = []
    with open(RUTA_V3) as f:
        for row in csv.DictReader(f):
            filas.append(dict(
                origen_tarea="tarea_actual_v3_escala20", par=row["par"], rule_id=row["rule_id"],
                rol=row["rol"], clase=row["clase"], seed=row["seed"], K=row["K"], kcap=row["kcap"],
                match_exacto_K_kcap=row["match_exacto_K_kcap"],
                fraccion_masa_en_sumideros=row["fraccion_masa_en_sumideros"],
                n_sumideros=row["n_sumideros"], t_primer_sumidero=row["t_primer_sumidero"],
                masa_acretada_total=row["masa_acretada_total"], kappa_v_agregado=row["kappa_v_agregado"],
                kappa_v_medio_valido=row["kappa_v_medio_valido"],
                n_kappa_indefinidos=row["n_kappa_indefinidos"], carpeta=row["carpeta"]))
    return filas


def agrupar_por_par(filas):
    por_par = defaultdict(dict)
    for f in filas:
        por_par[f["par"]][f["rol"]] = f
    return por_par


def comparacion_por_par(por_par):
    filas_cmp = []
    for par, roles in por_par.items():
        if "I" not in roles or "III" not in roles:
            continue
        fI, fIII = roles["I"], roles["III"]
        d_frac = float(fIII["fraccion_masa_en_sumideros"]) - float(fI["fraccion_masa_en_sumideros"])
        kv_I = fI["kappa_v_agregado"]
        kv_III = fIII["kappa_v_agregado"]
        d_kv = (float(kv_III) - float(kv_I)) if kv_I not in (None, "", "None") and kv_III not in (None, "", "None") else None
        filas_cmp.append(dict(par=par, origen_tarea=fI["origen_tarea"],
                               match_exacto=fI["match_exacto_K_kcap"],
                               K=fI["K"], kcap=fI["kcap"],
                               frac_I=float(fI["fraccion_masa_en_sumideros"]),
                               frac_III=float(fIII["fraccion_masa_en_sumideros"]),
                               d_frac=d_frac,
                               kv_I=float(kv_I) if kv_I not in (None, "", "None") else None,
                               kv_III=float(kv_III) if kv_III not in (None, "", "None") else None,
                               d_kv=d_kv))
    return filas_cmp


def main():
    filas_v1v2 = cargar_v1v2()
    filas_v3 = cargar_v3()
    todas = filas_v1v2 + filas_v3

    with open(RUTA_TOTAL, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS_COMUNES)
        w.writeheader()
        w.writerows(todas)
    print(f"[consolidado] {len(todas)} filas ({len(filas_v1v2)} tarea anterior + {len(filas_v3)} tarea "
          f"actual) -> {RUTA_TOTAL}")

    por_par = agrupar_por_par(todas)
    cmp_pares = comparacion_por_par(por_par)
    print(f"\n[pares completos I+III] {len(cmp_pares)}")

    def resumen(subset, etiqueta):
        n = len(subset)
        if n == 0:
            print(f"  {etiqueta}: n=0")
            return
        d_fracs = [c["d_frac"] for c in subset]
        d_kvs = [c["d_kv"] for c in subset if c["d_kv"] is not None]
        n_iii_mayor_frac = sum(1 for d in d_fracs if d > 0)
        n_iii_mayor_kv = sum(1 for d in d_kvs if d > 0)
        media_d_frac = sum(d_fracs) / n
        media_d_kv = sum(d_kvs) / len(d_kvs) if d_kvs else None
        print(f"  {etiqueta}: n={n}  media_Δfraccion(III-I)={media_d_frac:+.4f}  "
              f"III>I_fraccion={n_iii_mayor_frac}/{n}  "
              f"media_Δkappa_V(III-I)={media_d_kv:+.4f}  III>I_kappaV={n_iii_mayor_kv}/{len(d_kvs)}"
              if media_d_kv is not None else
              f"  {etiqueta}: n={n} media_Δfraccion={media_d_frac:+.4f} III>I_fraccion={n_iii_mayor_frac}/{n}")

    print("\n=== RESULTADO AGREGADO FINAL ===")
    resumen(cmp_pares, "TODOS los pares (n total)")
    exactos = [c for c in cmp_pares if str(c["match_exacto"]).lower() == "true"]
    resumen(exactos, "SOLO match exacto K=kcap")
    solo_nuevos = [c for c in cmp_pares if c["origen_tarea"] == "tarea_actual_v3_escala20"]
    resumen(solo_nuevos, "SOLO pares nuevos de esta tarea (n=12)")
    solo_anteriores = [c for c in cmp_pares if c["origen_tarea"] == "tarea_anterior_v1v2_8pares"]
    resumen(solo_anteriores, "SOLO los 8 pares de la tarea anterior (referencia)")

    print("\npar-por-par (todos):")
    for c in cmp_pares:
        print(f"  {c['par']:45s} K={c['K']} kcap={c['kcap']} exacto={c['match_exacto']:>5} "
              f"Δfrac={c['d_frac']:+.4f}  Δkv={c['d_kv']:+.4f}" if c['d_kv'] is not None else
              f"  {c['par']:45s} K={c['K']} kcap={c['kcap']} exacto={c['match_exacto']:>5} Δfrac={c['d_frac']:+.4f}")

    return cmp_pares


if __name__ == "__main__":
    main()

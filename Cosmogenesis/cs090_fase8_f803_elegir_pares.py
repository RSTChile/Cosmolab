"""
cs090_fase8_f803_elegir_pares.py — FASE VIII F8-03: elige, POR SELECCIÓN Y A CIEGAS DE LA MASA, el
emparejamiento `solap` ↔ `disp` de cada grafo base cuyo PICO LOCAL de densidad inicial queda más igual.
==========================================================================================================

CUÁNDO SE CORRE: **antes** de `cs090_fase8_f803_correr.py`. Este script no abre ni un solo dump de
Phantom — sólo lee el CSV de estructura, que trae el pico medido sobre la condición inicial. La elección
del par queda así congelada antes de que exista una sola masa, que es lo que hace legítimo el control.

QUÉ HACE
--------
Para cada grafo base hay `R` realizaciones de `solap` y `R` de `disp`, todas con el mismo N, las mismas
aristas, los mismos grados nodo por nodo y **el mismo T\***. Eso da `R×R` emparejamientos posibles, todos
igualmente válidos como contraste de apiñamiento. Se elige el que **minimiza |Δpico|**.

EL CRITERIO DE "IGUALADO", DECLARADO ACÁ Y NO DESPUÉS
------------------------------------------------------
El pico local no tiene una barra de error propia por corrida (es una medida determinista sobre una IC
determinista), pero sí tiene una **dispersión entre realizaciones del mismo brazo** — dos maquetas
armadas con el mismo criterio y distinta suerte caen en picos distintos. Esa dispersión es el "ruido
propio del pico" y es la vara honesta:

    sigma_pico = desviación estándar agrupada (pooled) del pico entre las R realizaciones de un mismo
                 brazo dentro de un mismo grafo base, promediada sobre brazos y grafos.

Un par se declara **IGUALADO** si `|Δpico| ≤ sigma_pico`, y **NO IGUALADO** si lo supera. Los dos grupos
se reportan por separado: un control fallido también es información, y además los pares no igualados
sirven de contraste interno (deberían mostrar el efecto entero).

SALIDA
------
  cs090_fase8_f803_pares_elegidos.csv   — un renglón por grafo base, con el par elegido, |Δpico|,
                                          Δsoporte, Δ de todas las medidas de apiñamiento y la etiqueta
                                          igualado / no_igualado.
  cs090_fase8_f803_emparejamientos.csv  — LOS R×R candidatos de cada grafo (para que se vea qué se
                                          descartó y por qué; sin ninguna masa a la vista).
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"

# el pico local declarado en F7-05 y en el generador; las otras dos son varas de robustez
PICO = "pico_p90_med"
PICOS_ALT = ("pico_cv", "pico_max_med", "pico_p99_med")

# medidas de apiñamiento que interesan (F8-01: la que queda en pie es el TAMAÑO DEL SOPORTE)
APIN = ("frac_aristas_en_triangulo", "tri_por_arista_media", "frac_aristas_multi_tri",
        "gini_tri_nodo", "tri_por_arista_max", "clustering_local", "modularidad_tri")


def cargar_estructura():
    """Se arma desde los `meta_regla.json` de las carpetas, no desde los CSV de estructura.

    Motivo práctico: el CSV de un shard se escribe recién al terminar sus 6 variantes, mientras que el
    `meta_regla.json` se escribe carpeta por carpeta y trae TODO lo que esta elección necesita (pico,
    apiñamiento, T, aristas, grados verificados, masa total). Así la elección no depende de que los 12
    procesos hayan terminado a la vez, y además usa exactamente el mismo archivo que después verifica
    el analizador. Los CSV de estructura quedan igual como crudo del experimento."""
    import json
    from pathlib import Path
    BASE = Path("/Users/alexis/phantom_cs073/bateria_fase8_f803_mismo_pico")
    filas = []
    for p in sorted(BASE.glob("*_f803_*/meta_regla.json")):
        m = json.loads(p.read_text())
        m["carpeta"] = str(p.parent)
        m["n_aristas"] = m["n_aristas_grafo_final"]
        m["grados_identicos"] = m["grados_identicos_al_original"]
        filas.append(m)
    if not filas:
        raise SystemExit("no hay carpetas f803 con meta_regla.json todavía")
    D = pd.DataFrame(filas)
    D["clave"] = D["rule_id"] + "_s" + D["seed"].astype(str)     # unión por (rule_id, seed), NUNCA rule_id
    return D


def main():
    D = cargar_estructura()
    print(f"filas de estructura: {len(D)}  grafos base: {D['clave'].nunique()}")

    # ---------- control de que lo que debía estar fijo, está fijo ----------
    avisos = []
    for clave, sub in D.groupby("clave"):
        if sub["n_triangulos"].nunique() != 1:
            avisos.append(f"{clave}: nº de triángulos NO idéntico entre variantes "
                          f"({sorted(sub['n_triangulos'].unique())})")
        if sub["n_aristas"].nunique() != 1:
            avisos.append(f"{clave}: nº de aristas NO idéntico ({sorted(sub['n_aristas'].unique())})")
        if not sub["grados_identicos"].all():
            avisos.append(f"{clave}: alguna variante NO conservó los grados")
        if sub["masa_total_ic"].round(6).nunique() > 1:
            avisos.append(f"{clave}: masa total de la IC distinta entre variantes")
    print("AVISOS de control fijo:", avisos if avisos else "ninguno")

    # ---------- sigma_pico: dispersión entre realizaciones del MISMO brazo ----------
    sd = []
    for (clave, brazo), sub in D.groupby(["clave", "brazo"]):
        if len(sub) > 1:
            sd.append(sub[PICO].std(ddof=1))
    sigma_pico = float(np.sqrt(np.mean(np.array(sd) ** 2)))      # pooled
    print(f"sigma_pico (ruido propio del pico entre realizaciones del mismo brazo) = {sigma_pico:.4f}"
          f"  [n grupos = {len(sd)}]")

    # ---------- todos los emparejamientos posibles ----------
    filas_all, filas_sel = [], []
    for clave, sub in D.groupby("clave"):
        S = sub[sub.brazo == "solap"].sort_values("realizacion")
        P = sub[sub.brazo == "disp"].sort_values("realizacion")
        cands = []
        for _, s in S.iterrows():
            for _, d in P.iterrows():
                fila = dict(clave=clave, rule_id=s["rule_id"], seed=int(s["seed"]), lote=s["lote"],
                            T_objetivo=int(s["T_objetivo"]),
                            var_solap=s["variante"], var_disp=d["variante"],
                            pico_solap=s[PICO], pico_disp=d[PICO],
                            d_pico=float(s[PICO] - d[PICO]), abs_d_pico=abs(float(s[PICO] - d[PICO])))
                for c in PICOS_ALT:
                    fila[f"d_{c}"] = float(s[c] - d[c])
                for c in APIN:
                    fila[f"{c}_solap"] = float(s[c])
                    fila[f"{c}_disp"] = float(d[c])
                    fila[f"d_{c}"] = float(s[c] - d[c])
                fila["carpeta_solap"] = s.get("carpeta")
                fila["carpeta_disp"] = d.get("carpeta")
                cands.append(fila)
        cands.sort(key=lambda f: f["abs_d_pico"])
        for r, f in enumerate(cands):
            f["rango_abs_d_pico"] = r
        filas_all.extend(cands)
        mejor = dict(cands[0])
        mejor["abs_d_pico_naive_mismo_indice"] = float(np.mean(
            [abs(f["d_pico"]) for f in cands if f["var_solap"][-1] == f["var_disp"][-1]]))
        mejor["abs_d_pico_mediana_candidatos"] = float(np.median([f["abs_d_pico"] for f in cands]))
        mejor["sigma_pico"] = sigma_pico
        mejor["igualado"] = bool(mejor["abs_d_pico"] <= sigma_pico)
        filas_sel.append(mejor)

    A = pd.DataFrame(filas_all)
    S = pd.DataFrame(filas_sel).sort_values("abs_d_pico")
    A.to_csv(os.path.join(HERE, "cs090_fase8_f803_emparejamientos.csv"), index=False)
    S.to_csv(os.path.join(HERE, "cs090_fase8_f803_pares_elegidos.csv"), index=False)

    cols = ["clave", "T_objetivo", "var_solap", "var_disp", "pico_solap", "pico_disp", "d_pico",
            "abs_d_pico_naive_mismo_indice", "d_frac_aristas_en_triangulo", "d_tri_por_arista_media",
            "igualado"]
    print("\n=== PAR ELEGIDO POR GRAFO (a ciegas de la masa) ===")
    print(S[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nigualados (|Δpico| <= {sigma_pico:.3f}): {int(S['igualado'].sum())}/{len(S)}")
    print(f"|Δpico| elegido: mediana {S['abs_d_pico'].median():.3f}  media {S['abs_d_pico'].mean():.3f}")
    print(f"|Δpico| ingenuo (mismo índice): mediana "
          f"{S['abs_d_pico_naive_mismo_indice'].median():.3f}  media "
          f"{S['abs_d_pico_naive_mismo_indice'].mean():.3f}")
    print(f"Δsoporte del par elegido (solap−disp): mediana "
          f"{S['d_frac_aristas_en_triangulo'].median():.4f}")


if __name__ == "__main__":
    main()

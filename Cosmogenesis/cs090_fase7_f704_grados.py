"""
cs090_fase7_f704_grados.py — FASE VII, F7-04: ¿CUÁNTO cambia la SECUENCIA DE GRADOS entre brazos?

POR QUÉ ESTE ARCHIVO EXISTE
---------------------------
F7-02 (`FASE7_F702_escalera_clustering_CS.md`) movió el clustering **fijando la secuencia de grados
nodo por nodo** (double-edge-swap: cada nodo pierde un vecino y gana otro). F7-04 hace lo contrario:
**quita** aristas, y quitar aristas distintas cambia el grado de los nodos tocados. Los dos experimentos
sólo pueden convivir si esa diferencia de diseño está medida, no supuesta.

Este archivo recomputa los 5 brazos de cada grafo base — reusando `cs090_fase7_f704_brazos.py`, sin
volver a correr layouts ni Phantom (es barato: sólo la dinámica del motor, ~2 s por grafo) — y mide,
brazo contra brazo:

  · cuántos nodos cambian de grado respecto del brazo `c2` y cuánto (media de |Δgrado|);
  · el desvío estándar de la distribución de grados de cada brazo (¿alguno queda más desparejo?);
  · la correlación de Spearman entre secuencias de grados de brazos distintos.

Suma de grados: idéntica por construcción en los 5 brazos (= 2·M, y M es idéntico — verificado en
`cs090_fase7_f704_brazos.py`). Lo que puede diferir es CÓMO se reparte esa suma.

Escribe `cs090_fase7_f704_grados.csv`. No corre Phantom, no modifica nada, no declara cierre.
"""
from __future__ import annotations
import csv
import sys

import numpy as np
from scipy import stats

HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, HERE)

import cs090_fase5_motor as MOT
import cs090_fase7_f704_brazos as F
from cs090_fase6_o3b_rewiring import grados, N_NODOS

RUTA_SALIDA = f"{HERE}/cs090_fase7_f704_grados.csv"


def main():
    with open(F.RUTA_SELECCION) as f:
        elegidos = list(csv.DictReader(f))

    filas = []
    for sel in elegidos:
        seed = int(sel["seed"])
        est = F.reproducir_dinamica_hasta_pre_poda(seed)
        p, edges, adj_pre = est["p"], est["edges"], est["adj_pre"]
        costo = F.vector_de_costos(edges, est["flip_count"], est["E_estado"], p["K"], est["triangles"])
        soporte = F.soporte_local_de_aristas(adj_pre, edges)
        conservar = MOT._costo_y_podar(edges, est["flip_count"], est["E_estado"], p["K"],
                                       est["triangles"], P=F.PERCENTIL_PODA)
        conservadas, n_cut, _, _ = F.brazos_de_corte(edges, costo, soporte, conservar, seed)

        g = {b: grados(F.construir_adj_desde_aristas(v, N_NODOS), N_NODOS)
             for b, v in conservadas.items()}
        sumas = {b: int(v.sum()) for b, v in g.items()}
        assert len(set(sumas.values())) == 1, f"{sel['rule_id']}: suma de grados distinta {sumas}"

        fila = dict(rule_id=sel["rule_id"], seed=seed, lote=sel["lote"], n_cut=n_cut,
                    suma_grados=sumas["c2"], grado_medio=sumas["c2"] / N_NODOS)
        for b in F.BRAZOS:
            fila[f"std_grado_{b}"] = float(g[b].std())
        for b in F.BRAZOS:
            if b == "c2":
                continue
            d = g["c2"].astype(int) - g[b].astype(int)
            fila[f"n_nodos_grado_distinto_c2_vs_{b}"] = int((d != 0).sum())
            fila[f"frac_nodos_grado_distinto_c2_vs_{b}"] = float((d != 0).mean())
            fila[f"media_abs_dgrado_c2_vs_{b}"] = float(np.abs(d).mean())
            fila[f"max_abs_dgrado_c2_vs_{b}"] = int(np.abs(d).max())
            fila[f"spearman_grados_c2_vs_{b}"] = float(stats.spearmanr(g["c2"], g[b]).statistic)
        filas.append(fila)
        print(f"[{sel['rule_id']}] n_cut={n_cut} grado_medio={fila['grado_medio']:.3f} | "
              + " ".join(f"{b}:{fila[f'frac_nodos_grado_distinto_c2_vs_{b}']*100:.1f}%"
                         for b in F.BRAZOS if b != "c2"), flush=True)

    with open(RUTA_SALIDA, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)

    print(f"\n=== resumen sobre {len(filas)} grafos base ===")
    print(f"  grado medio (idéntico en los 5 brazos por construcción): "
          f"{np.mean([f['grado_medio'] for f in filas]):.3f}")
    for b in F.BRAZOS:
        print(f"  std del grado, brazo {b:12s}: {np.mean([f[f'std_grado_{b}'] for f in filas]):.4f}")
    for b in F.BRAZOS:
        if b == "c2":
            continue
        print(f"  c2 vs {b:12s}: nodos con grado distinto "
              f"{100*np.mean([f[f'frac_nodos_grado_distinto_c2_vs_{b}'] for f in filas]):5.2f}% "
              f"| media |Δgrado| {np.mean([f[f'media_abs_dgrado_c2_vs_{b}'] for f in filas]):.4f} "
              f"| Spearman grados {np.mean([f[f'spearman_grados_c2_vs_{b}'] for f in filas]):.4f}")
    print(f"\n  -> {RUTA_SALIDA.split('/')[-1]}")
    return filas


if __name__ == "__main__":
    main()

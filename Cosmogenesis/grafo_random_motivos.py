"""
grafo_random_motivos.py — Paso 4 del encargo: mide motivos/triángulos/ciclos-4 DIRECTAMENTE sobre el
grafo Erdős-Rényi de control (N=2000, mismo n/m que la malla causal REAL) -- mismo método EXACTO que
`null3_motivos_directos.py` (reusa sus funciones `contar_triangulos_y_clustering`/`contar_ciclos_4`
importadas, no reescritas), para reportar el punto de comparación pedido: ¿cuántos triángulos tiene un
grafo Erdős-Rényi de este tamaño/densidad, comparado con los ~2780 de REAL y los ~2005 de
NULL-3(tol=0.2)?

No corre Phantom. No genera condiciones iniciales. No toca ningún archivo congelado ni ninguna carpeta
de batería anterior -- sólo lectura/importación.
"""
import time

import numpy as np

from null3_investigacion_preliminar import reconstruir_grafo_real, aristas_de
from null3_motivos_directos import contar_triangulos_y_clustering, contar_ciclos_4, reportar
from grafo_random_layout_generar_ic import generar_grafo_erdos_renyi

SEED = 701   # misma semilla que la primera corrida nueva de la batería del control random


def main():
    t0 = time.time()
    print("[1] reconstruyendo grafo causal REAL (N=2000, determinista, sólo para contar n/m)...",
          flush=True)
    adj_real, pos_real, n = reconstruir_grafo_real()
    n_aristas = len(aristas_de(adj_real, n))
    print(f"    n={n} n_aristas={n_aristas} tiempo={time.time()-t0:.2f}s\n", flush=True)

    print(f"[2] generando grafo random Erdős-Rényi G(n={n}, m={n_aristas}), seed={SEED} -- "
          f"INDEPENDIENTE de REAL (ninguna arista compartida)...", flush=True)
    adj_random, edge_set, intentos = generar_grafo_erdos_renyi(n, n_aristas, seed=SEED)
    print(f"    aristas generadas={len(edge_set)} intentos_rechazo={intentos}\n", flush=True)

    # verificación explícita de independencia: cuántas aristas coinciden por azar con REAL
    edges_real = set(tuple(sorted(e)) for e in aristas_de(adj_real, n))
    solapadas = len(edges_real & edge_set)
    print(f"[verificación] aristas que el grafo random comparte con REAL por puro azar: "
          f"{solapadas}/{n_aristas} ({100*solapadas/n_aristas:.3f}%) -- esperado ~0 si es "
          f"independiente (m/C(n,2) = {n_aristas/ (n*(n-1)/2):.5f} de probabilidad por arista)\n",
          flush=True)

    print("[3] contando triángulos / clustering / ciclos de 4 en REAL vs grafo random:\n", flush=True)
    r_real = reportar("REAL", adj_real, n)
    r_random = reportar(f"grafo random Erdős-Rényi (seed={SEED})", adj_random, n)

    print("\n[4] comparación explícita (REAL vs Erdős-Rényi):")
    dtri = 100 * (r_random["n_triangulos"] - r_real["n_triangulos"]) / r_real["n_triangulos"]
    print(f"    triángulos:          REAL={r_real['n_triangulos']}  ER={r_random['n_triangulos']}  "
          f"diff={dtri:+.1f}%")
    print(f"    clustering promedio: REAL={r_real['clustering_promedio']:.5f}  "
          f"ER={r_random['clustering_promedio']:.5f}")
    print(f"    clustering global:   REAL={r_real['clustering_global']:.5f}  "
          f"ER={r_random['clustering_global']:.5f}")
    print(f"    ciclos de 4:         REAL={r_real['ciclos_4']}  ER={r_random['ciclos_4']}")

    # referencia teórica: nº esperado de triángulos en G(n,m) = C(n,3) * p^3, p = m/C(n,2)
    p = n_aristas / (n * (n - 1) / 2)
    from math import comb
    triangulos_esperados_teoria = comb(n, 3) * p ** 3
    print(f"\n[5] referencia teórica ER: p={p:.6f}, triángulos esperados = C(n,3)*p^3 = "
          f"{triangulos_esperados_teoria:.2f} (fórmula estándar, no ajustada) -- grafo random medido: "
          f"{r_random['n_triangulos']} (consistencia esperada: ambos deben ser del mismo orden de "
          f"magnitud, pequeño, muy por debajo de REAL/NULL-3)")

    print(f"\n[TOTAL] {time.time()-t0:.2f}s", flush=True)
    return dict(real=r_real, random=r_random, n_aristas_solapadas=solapadas,
                triangulos_esperados_teoria=triangulos_esperados_teoria)


if __name__ == "__main__":
    main()

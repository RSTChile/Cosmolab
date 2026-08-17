"""
null3_motivos_directos.py — Parte A del encargo "robustecer NULL-3": medición DIRECTA de motivos/
triángulos/ciclos en el grafo causal REAL vs el grafo NULL-3 (double-edge-swap con filtro geométrico de
longitud, tol_relativa=0.2, seed=501 -- el mismo grafo que ya se usó para la batería NULL3_resultado_CS.md).

Por qué esto era el caveat pendiente: `NULL3_resultado_CS.md` infirió "los motivos/ciclos no importan"
INDIRECTAMENTE del contraste de sumideros (REAL≈NULL-3 en masa acretada) -- nunca midió motivos
directamente sobre el grafo. Este script cierra ese hueco: cuenta triángulos y coeficiente de clustering
(y, si es barato, ciclos de 4 nodos) en REAL y en NULL-3, para saber CUÁNTA estructura de orden superior
se destruyó de verdad.

No usa networkx (no está instalado en el venv del proyecto, ver `venv/`) -- implementación propia con
sets de adyacencia (triángulos, barato: k=4 en promedio, ~4945 aristas) y una multiplicación de matriz
booleana/entera con numpy para los ciclos de 4 nodos (n=2000, matmul denso trivial para numpy/BLAS).

No toca ningún archivo congelado -- sólo importa `reconstruir_grafo_real` y
`barajar_aristas_preservando_longitud`/`barajar_aristas_sin_restriccion` de
`null3_investigacion_preliminar.py` (ya validadas, no reescritas). No corre Phantom. No genera
condiciones iniciales. No toca ninguna carpeta de batería.
"""
import time

import numpy as np

from null3_investigacion_preliminar import (
    reconstruir_grafo_real, aristas_de, barajar_aristas_preservando_longitud,
    barajar_aristas_sin_restriccion,
)

SEED = 501           # misma semilla que usó NULL3_resultado_CS.md (Paso 1 / primera de la batería 501-508)
TOL_ORIGINAL = 0.2   # tol_relativa original de la batería NULL-3 ya cerrada


# ----------------------------------------------------------------------------------------------------
# Triángulos + coeficiente de clustering -- sets de adyacencia, O(n_aristas * grado promedio)
# ----------------------------------------------------------------------------------------------------
def contar_triangulos_y_clustering(adj: dict, n: int) -> dict:
    """adj: dict nodo -> set de vecinos (sin incluirse a sí mismo). Devuelve triángulos totales,
    coeficiente de clustering PROMEDIO (media de C_i por nodo, C_i=0 si grado<2, convención estándar
    igual a networkx.average_clustering) y GLOBAL/transitividad (3*triángulos / nº de tripletes
    conectados = suma de C(grado_i,2))."""
    grados = np.array([len(adj.get(i, ())) for i in range(n)])
    tri_por_nodo = np.zeros(n, dtype=np.int64)

    # Por cada arista (i,j) (cada arista aparece UNA vez en `edges`, ver aristas_de: j>i), cada vecino
    # común k cierra un triángulo {i,j,k}. Un triángulo tiene 3 aristas, así que aparece exactamente 3
    # veces en total sumando sobre TODAS las aristas -> triángulos_totales = n_hallazgos // 3. Cada vez
    # que aparece, se suma +1 a los 3 nodos del triángulo -- pero cada nodo del triángulo incide en 2 de
    # sus 3 aristas, así que tri_por_nodo[i] queda sobre-contado x2 -> se corrige dividiendo por 2.
    edges = aristas_de(adj, n)
    n_hallazgos = 0
    for (i, j) in edges:
        comunes = adj[i] & adj[j]
        n_hallazgos += len(comunes)
        for k in comunes:
            tri_por_nodo[i] += 1
            tri_por_nodo[j] += 1
            tri_por_nodo[k] += 1

    triangulos_totales = n_hallazgos // 3
    tri_por_nodo = tri_por_nodo // 2

    with np.errstate(divide="ignore", invalid="ignore"):
        max_posibles = grados * (grados - 1) / 2.0
        c_local = np.where(max_posibles > 0, tri_por_nodo / max_posibles, 0.0)
    clustering_promedio = float(c_local.mean())

    triples_conectados = float(max_posibles.sum())
    clustering_global = (3.0 * triangulos_totales / triples_conectados) if triples_conectados > 0 else 0.0

    return dict(n_triangulos=int(triangulos_totales), clustering_promedio=clustering_promedio,
                clustering_global=clustering_global, grado_medio=float(grados.mean()),
                triples_conectados=int(triples_conectados))


# ----------------------------------------------------------------------------------------------------
# Ciclos de 4 nodos (C4, subgrafo -- 4 vértices distintos formando un cuadrado, sin diagonales exigidas)
# vía A^2: si c(u,w) = nº de vecinos comunes de u,w (u!=w), entonces cada elección de 2 de esos vecinos
# comunes cierra un 4-ciclo u-x-w-y-u. Cada 4-ciclo tiene 2 diagonales {u,w}/{x,y}, así que se cuenta 2
# veces en la suma sobre TODOS los pares -- de ahí el /2. Identidad estándar de teoría de grafos, no
# inventada para este experimento.
# ----------------------------------------------------------------------------------------------------
def contar_ciclos_4(adj: dict, n: int) -> int:
    # float64 (no int32/int64) para que numpy use BLAS en el matmul -- numpy NO acelera matmul entero
    # con BLAS (sólo float32/float64), así que A@A con dtype entero es ~30x más lento a n=2000. Los
    # valores de A2 son cuentas de vecinos comunes (enteros pequeños, <<2^53) -- exactos en float64,
    # redondeados de vuelta a entero antes de usarlos.
    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in adj.get(i, ()):
            A[i, j] = 1.0
    A2 = np.rint(A @ A).astype(np.int64)
    iu = np.triu_indices(n, k=1)
    c = A2[iu]
    total = int((c * (c - 1) // 2).sum())
    return total // 2


def reportar(nombre: str, adj: dict, n: int, incluir_c4: bool = True) -> dict:
    t0 = time.time()
    r = contar_triangulos_y_clustering(adj, n)
    t_tri = time.time() - t0
    linea = (f"{nombre:34s} triángulos={r['n_triangulos']:5d}  clustering_prom={r['clustering_promedio']:.5f}  "
              f"clustering_global={r['clustering_global']:.5f}  grado_medio={r['grado_medio']:.3f}  "
              f"(tri: {t_tri:.2f}s)")
    if incluir_c4:
        t1 = time.time()
        c4 = contar_ciclos_4(adj, n)
        t_c4 = time.time() - t1
        r["ciclos_4"] = c4
        linea += f"  ciclos_4={c4}  (c4: {t_c4:.2f}s)"
    print(linea, flush=True)
    r["nombre"] = nombre
    return r


def main():
    t0 = time.time()
    print("[1] reconstruyendo grafo causal REAL (determinista, mismos parámetros que traducir_pool)...",
          flush=True)
    adj_real, pos_real, n = reconstruir_grafo_real()
    print(f"    n={n}  n_aristas={len(aristas_de(adj_real, n))}  tiempo={time.time()-t0:.2f}s\n", flush=True)

    print(f"[2] generando NULL-3 (double-edge-swap + filtro geométrico, tol_relativa={TOL_ORIGINAL}, "
          f"seed={SEED} -- MISMO grafo que produjo NULL3_resultado_CS.md)...", flush=True)
    adj_null3, aceptados, intentos = barajar_aristas_preservando_longitud(
        adj_real, n, pos_real, seed=SEED, tol_relativa=TOL_ORIGINAL)
    print(f"    swap aceptados/intentados={aceptados}/{intentos} ({100*aceptados/intentos:.2f}%)\n",
          flush=True)

    print(f"[3] generando swap SIN restricción de longitud (Maslov-Sneppen original, seed={SEED} -- "
          f"mecanismo de los NULL1-8 originales de CS073) para referencia lado a lado...", flush=True)
    adj_sin = barajar_aristas_sin_restriccion(adj_real, n, seed=SEED)
    print("", flush=True)

    print("[4] contando triángulos / clustering / ciclos de 4 en los tres grafos:\n", flush=True)
    r_real = reportar("REAL", adj_real, n)
    r_null3 = reportar(f"NULL-3 (tol_relativa={TOL_ORIGINAL})", adj_null3, n)
    r_sin = reportar("swap SIN restricción (NULL1-8 orig.)", adj_sin, n)

    print("\n[5] cuánto cambió NULL-3 respecto de REAL (motivos de orden superior):")
    dtri = 100 * (r_null3["n_triangulos"] - r_real["n_triangulos"]) / r_real["n_triangulos"]
    dclust_prom = 100 * (r_null3["clustering_promedio"] - r_real["clustering_promedio"]) / r_real["clustering_promedio"]
    dclust_glob = 100 * (r_null3["clustering_global"] - r_real["clustering_global"]) / r_real["clustering_global"]
    dc4 = 100 * (r_null3["ciclos_4"] - r_real["ciclos_4"]) / r_real["ciclos_4"] if r_real["ciclos_4"] > 0 else float("nan")
    print(f"    triángulos:          REAL={r_real['n_triangulos']}  NULL-3={r_null3['n_triangulos']}  "
          f"diff={dtri:+.1f}%")
    print(f"    clustering promedio: REAL={r_real['clustering_promedio']:.5f}  "
          f"NULL-3={r_null3['clustering_promedio']:.5f}  diff={dclust_prom:+.1f}%")
    print(f"    clustering global:   REAL={r_real['clustering_global']:.5f}  "
          f"NULL-3={r_null3['clustering_global']:.5f}  diff={dclust_glob:+.1f}%")
    print(f"    ciclos de 4:         REAL={r_real['ciclos_4']}  NULL-3={r_null3['ciclos_4']}  "
          f"diff={dc4:+.1f}%")

    print("\n    (referencia, swap SIN restricción de longitud, mismo seed):")
    dtri_sin = 100 * (r_sin["n_triangulos"] - r_real["n_triangulos"]) / r_real["n_triangulos"]
    print(f"    triángulos:          REAL={r_real['n_triangulos']}  sin_restr={r_sin['n_triangulos']}  "
          f"diff={dtri_sin:+.1f}%")

    print(f"\n[TOTAL] {time.time()-t0:.2f}s", flush=True)
    return dict(real=r_real, null3=r_null3, sin_restriccion=r_sin,
                swap_aceptados=aceptados, swap_intentos=intentos)


if __name__ == "__main__":
    main()

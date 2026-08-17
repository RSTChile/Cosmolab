"""
null4_verificar_invarianza_orden.py -- NULL-4, Fase II CS073, escalón "orden temporal de formación".

QUÉ HACE ESTE SCRIPT (léase antes de correr nada de Phantom): responde la pregunta que pide el encargo
de NULL-4 ANTES de gastar cómputo -- "¿la topología final + `layout_resortes` dependen del ORDEN en que
la malla causal se construyó, o sólo de la topología final como conjunto?" -- de forma empírica y
directa, reconstruyendo la malla causal REAL (N=2000, el mismo pool ya extraído en
`bateria_n2000/dens_bar.npy`) y pasándola por `layout_resortes` bajo varias reordenaciones distintas del
mismo conjunto de aristas.

No toca NINGÚN archivo congelado -- sólo IMPORTA `malla_causal_atomos`/`layout_resortes` de
`p_semilla_causal.py` (piezas ya validadas) y lee `dens_bar.npy`/`masa_bar.npy` ya extraídos en disco por
la batería original (no re-extrae el pool).

Hallazgo que este script busca confirmar o refutar (ver docstring de `layout_resortes` en
`p_semilla_causal.py`, líneas 95-137): la función construye `pos` inicial aleatoria (según `seed`), y
luego en cada una de las `iters` iteraciones calcula la fuerza de TODOS los nodos/aristas de forma
VECTORIZADA (numpy, suma por `np.add.at`) -- no hay ningún bucle secuencial que procese nodos/aristas
"uno por uno en algún orden" cuyo resultado dependa de cuál se procesó primero. La suma vectorizada es
conmutativa (hasta ruido de redondeo de punto flotante, ~1e-15 relativo) -- así que, en principio, el
ORDEN en que se insertan las aristas al `dict` de adyacencia `adj` no debería alterar el resultado más
allá de ese ruido de redondeo, muchos órdenes de magnitud por debajo de cualquier diferencia física
relevante (las posiciones tienen escala ~O(10-100), diferencias de sumideros se miden en unidades de masa
enteras).

Diseño del test: reconstruir el MISMO conjunto de aristas de la malla causal REAL, pero construir el
`dict` `adj` insertando esas aristas en 4 órdenes distintos (natural / invertido / 2 barajados al azar
con semillas distintas) -- así el iterador interno de `layout_resortes`
(`[(i,j) for i in range(n) for j in adj.get(i,()) if j>i]`) puede, en principio, recorrer las mismas
aristas en secuencias distintas dentro del bucle de construcción de `edges` (el orden de iteración de un
`set` de Python SÍ puede depender del orden de inserción cuando hay colisiones de hash en la tabla). Se
corre `layout_resortes` con el MISMO `seed_layout=12345` (la semilla real de la jerarquía) sobre las 4
versiones y se comparan las posiciones resultantes bit a bit.
"""
import numpy as np
import sys
sys.path.insert(0, "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis")

from cs072_modulos.piezas.p_semilla_causal import malla_causal_atomos, layout_resortes

RUTA_POOL = "/Users/alexis/phantom_cs073/bateria_n2000"
SEED_LAYOUT_REAL = 12345   # la misma semilla que usa toda la jerarquía (traducir_pool/null3_generar_ic/...)
D_CAUSAL, K_CAUSAL, SEED_EJES = 3, 4, 2000   # mismos parámetros que traducir_pool para REAL


def cargar_pool():
    dens_bar = np.load(f"{RUTA_POOL}/dens_bar.npy")
    masa_bar = np.load(f"{RUTA_POOL}/masa_bar.npy")
    return masa_bar, dens_bar


def adj_a_lista_aristas(adj, n):
    """Lista canónica de aristas (i<j) de un dict de adyacencia -- una por par, sin duplicar."""
    return [(i, j) for i in range(n) for j in adj.get(i, ()) if j > i]


def construir_adj_en_orden(lista_aristas, n, orden_idx):
    """Reconstruye un dict `adj` (mismo tipo que produce malla_causal_atomos: dict de sets) insertando
    las aristas de `lista_aristas` en el orden dado por `orden_idx` (permutación de índices sobre
    lista_aristas). El CONJUNTO final de aristas es idéntico sea cual sea `orden_idx` -- lo único que
    cambia es la SECUENCIA de inserción en cada `set()` de Python."""
    adj = {i: set() for i in range(n)}
    for k in orden_idx:
        i, j = lista_aristas[k]
        adj[i].add(j)
        adj[j].add(i)
    return adj


def main():
    print("=== NULL-4 -- verificación de invarianza de orden en layout_resortes (N=2000, malla REAL) ===\n")
    masa_bar, dens_bar = cargar_pool()
    n = len(dens_bar)
    lado = float(n) ** (1.0 / 3.0)
    print(f"n={n}  lado={lado:.6f}  masa_particula_media={masa_bar.mean():.6f}")

    adj_real, m = malla_causal_atomos(dens_bar, D=D_CAUSAL, k=K_CAUSAL, seed_ejes=SEED_EJES)
    lista = adj_a_lista_aristas(adj_real, n)
    n_aristas = len(lista)
    print(f"malla causal REAL reconstruida: {n_aristas} aristas (grado medio = {2*n_aristas/n:.4f})\n")

    # 4 órdenes de inserción del MISMO conjunto de aristas: natural, invertido, 2 barajados al azar
    rng1 = np.random.default_rng(7001)
    rng2 = np.random.default_rng(7002)
    ordenes = {
        "natural (orden de malla_causal_atomos)": list(range(n_aristas)),
        "invertido": list(range(n_aristas))[::-1],
        "barajado_seed7001": rng1.permutation(n_aristas).tolist(),
        "barajado_seed7002": rng2.permutation(n_aristas).tolist(),
    }

    posiciones = {}
    for nombre, orden in ordenes.items():
        adj_version = construir_adj_en_orden(lista, n, orden)
        # verificación de que el CONJUNTO de aristas es idéntico al de REAL (no sólo el nº)
        lista_version = sorted(adj_a_lista_aristas(adj_version, n))
        assert lista_version == sorted(lista), f"¡conjunto de aristas distinto en orden '{nombre}'!"
        pos = layout_resortes(adj_version, n, lado=lado, iters=100, seed=SEED_LAYOUT_REAL)
        posiciones[nombre] = pos
        print(f"  orden '{nombre}': layout_resortes corrido (topología idéntica verificada por conjunto)")

    print("\n=== Comparación de posiciones resultantes (todas vs. 'natural') ===")
    base_nombre = "natural (orden de malla_causal_atomos)"
    base = posiciones[base_nombre]
    r_base = np.linalg.norm(base, axis=1)
    print(f"  '{base_nombre}': r_mean={r_base.mean():.10f}  r_std={r_base.std():.10f}")
    peor_max_abs = 0.0
    for nombre, pos in posiciones.items():
        if nombre == base_nombre:
            continue
        diff = np.abs(pos - base)
        max_abs = diff.max()
        mean_abs = diff.mean()
        r = np.linalg.norm(pos, axis=1)
        peor_max_abs = max(peor_max_abs, max_abs)
        print(f"  '{nombre}': max|Δpos|={max_abs:.3e}  mean|Δpos|={mean_abs:.3e}  "
              f"r_mean={r.mean():.10f}  r_std={r.std():.10f}")

    escala_tipica = np.abs(base).mean()
    print(f"\nEscala típica de coordenada (|pos| media): {escala_tipica:.6f}")
    print(f"Peor diferencia absoluta observada entre órdenes: {peor_max_abs:.3e}")
    print(f"Razón (peor_diff / escala_tipica): {peor_max_abs/escala_tipica:.3e}")

    if peor_max_abs < 1e-6:
        print("\n>>> CONCLUSIÓN: las posiciones son IDÉNTICAS (a nivel de ruido de punto flotante, <1e-6 "
              "en unidades de coordenada ~O(1-10)) sea cual sea el orden de inserción de las aristas. "
              "layout_resortes NO depende del orden de construcción de la malla -- sólo de la topología "
              "final como CONJUNTO -- dado el mismo seed_layout.")
    else:
        print("\n>>> CONCLUSIÓN: las posiciones SÍ difieren de forma no trivial entre órdenes de "
              "inserción -- layout_resortes tiene alguna dependencia de orden no anticipada por el "
              "análisis de código. Habría que investigar la fuente antes de proceder.")


if __name__ == "__main__":
    main()

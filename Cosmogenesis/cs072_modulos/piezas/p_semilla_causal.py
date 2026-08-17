"""
p_semilla_causal.py — PIEZA: EL PUENTE (semilla dinámica desde la malla causal, CS073 v4).

Qué hace, en simple: `p_gravedad_general.posiciones_escenario()` sembraba posiciones UNIFORMES,
independientes de la densidad #23 — por eso REAL≈NULL en la primera corrida holística (nunca había
coherencia espacial que un barajado pudiera destruir). Esta pieza reemplaza esa siembra por una que SÍ
carga la única coherencia relacional que el motor realmente produce: la malla causal de las dos fases
de expansión (`_malla_causal`, proceso_sucesivo.py — la misma que usa `dimension_acoplada`), aplicada a
las densidades REALES de los átomos ya formados.

Distinción con el Paso A (que ya falló, `p02b_gravedad_general.py`, deprecado): Paso A embebía esa malla
de UNA VEZ (MDS, foto estática) para intentar leerle una dimensión 3D — negativo sólido. Esta pieza NO
embebe: usa la malla sólo como GRAFO DE VECINDAD (quién quedó causalmente cerca de quién) para sembrar
una posición INICIAL por layout de resortes (pares conectados arrancan próximos, el resto disperso) —
luego son p_expansion + p_gravedad_general + p_enfriamiento_H2 + p_materia_oscura_halo, en el mismo
bucle de siempre, los que DESPLIEGAN esa semilla en el tiempo. Proceso, no fotografía.

NO toca proceso_sucesivo.py/catalogo.py/estado.py/nucleo.py (todos congelados) — importa `_malla_causal`
directo (ya validada, no reimplementada) y duplica sólo el one-liner de construcción de ejes que
`dimension_acoplada` ya usa (mismo patrón que `p02b_gravedad_general.py`: reusar el algoritmo, no la
línea de import, para no tocar el archivo congelado).

Layout por resortes: Fruchterman-Reingold (1991) — el estándar de campo para grafos, no un método
inventado para este experimento (mismo criterio que ya se usó para el linking length de FoF, b=0.2).

NULL: barajado de ARISTAS por double-edge-swap (Maslov-Sneppen), preserva la secuencia de grados
exactamente -- destruye la topología específica (quién-con-quién) sin cambiar cuántas conexiones tiene
cada nodo. Distinto del NULL de #23 (que baraja VALORES de densidad): éste baraja la TOPOLOGÍA de la
malla ya construida, la salvedad que corrigió CC sobre "fases aleatorizadas" (lenguaje de Fourier que no
aplica a un grafo).
"""
import numpy as np
from scipy.spatial import cKDTree

from cs072_modulos.proceso_sucesivo import _malla_causal


def _ejes_desde_densidad(dens_arr, D, seed0):
    """MISMA receta que dimension_acoplada() (proceso_sucesivo.py, ~línea 98-99), duplicada aquí para
    no tocar ese archivo -- instrucción explícita del director (módulo nuevo, no tocar lo que funciona)."""
    n = len(dens_arr)
    return np.column_stack([dens_arr[np.random.default_rng(seed0 + eje).permutation(n)] for eje in range(D)])


def malla_causal_atomos(dens_bar, D=3, k=4, seed_ejes=2000):
    """Construye la malla causal REAL sobre los átomos ya formados -- mismo mecanismo de dos fases que
    ya mide dimension_acoplada, aplicado aquí como grafo de vecindad (no como embedding)."""
    n = len(dens_bar)
    V = _ejes_desde_densidad(dens_bar, D, seed_ejes)
    adj, m, _arranque = _malla_causal(V, min(k, max(n - 1, 1)))
    return adj, m


def barajar_aristas(adj, n, seed, factor_swaps=10):
    """NULL: double-edge-swap (Maslov-Sneppen) -- preserva la secuencia de grados EXACTA, destruye la
    topología específica. nº de intentos de swap = factor_swaps * nº de aristas (convención estándar del
    método, no ajustada a este experimento)."""
    rng = np.random.default_rng(seed)
    edges = [(i, j) for i in range(n) for j in adj[i] if j > i]
    if len(edges) < 2:
        return {i: set(adj[i]) for i in range(n)}
    edges = [list(e) for e in edges]
    n_intentos = factor_swaps * len(edges)
    edge_set = set(tuple(e) for e in edges)
    for _ in range(n_intentos):
        i1, i2 = rng.integers(0, len(edges), size=2)
        if i1 == i2:
            continue
        a, b = edges[i1]
        c, d = edges[i2]
        if len({a, b, c, d}) < 4:
            continue
        nuevo1 = tuple(sorted((a, d)))
        nuevo2 = tuple(sorted((c, b)))
        if nuevo1 in edge_set or nuevo2 in edge_set or nuevo1 == nuevo2:
            continue
        viejo1 = tuple(sorted((a, b)))
        viejo2 = tuple(sorted((c, d)))
        edge_set.discard(viejo1)
        edge_set.discard(viejo2)
        edge_set.add(nuevo1)
        edge_set.add(nuevo2)
        edges[i1] = [a, d]
        edges[i2] = [c, b]
    adj_null = {i: set() for i in range(n)}
    for (a, b) in edge_set:
        adj_null[a].add(b)
        adj_null[b].add(a)
    return adj_null


HFACT_SPH = 1.2   # misma convención SPH del resto del arco (== hfact_default de Phantom, kernel_cubic.f90)


def layout_resortes(adj, n, lado, dims=3, iters=100, seed=12345, sep_min=None):
    """Fruchterman-Reingold: posición inicial sembrada por el grafo (pares conectados arrancan
    próximos), no plantada a mano (G-SIN-SIEMBRA) -- emerge de la topología + una relajación física
    estándar. k_FR = (volumen/n)^(1/3) -- la distancia 'natural' de la fórmula estándar, no ajustada.

    sep_min=None (DEFAULT, adjudicación final CS 20-jul, INSTRUCCION_CC_hlocal_PARA_CC.md): NO se
    impone ningún piso aquí. El histograma de distancia al vecino más cercano no tiene hueco (ni en REAL
    ni en NULL) -- los pares casi-coincidentes son la cola de una estructura jerárquica genuina, común a
    ambos brazos, no una patología separable (ver INSTRUCCION_CC_sepmin_PARA_CC.md: sep_min="auto" con
    HFACT_SPH*k_fr destruía 1246/1250 pares a N=250 -- retirado, NO reintroducir). La resolución del
    problema de convergencia de Phantom fue del lado de Phantom (h inicial uniforme, grad-h nativo -- ver
    fase1_traducir_a_phantom.py), no de este layout. sep_min queda como parámetro por si algún día hace
    falta, pero el valor por defecto es None (sin efecto)."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, lado, size=(n, dims))
    if n < 2:
        return pos
    k_fr = (lado ** dims / n) ** (1.0 / dims)
    edges = np.array([(i, j) for i in range(n) for j in adj.get(i, ()) if j > i], dtype=int)
    paso = lado * 0.1
    for it in range(iters):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt(np.sum(delta ** 2, axis=-1)) + 1e-6
        np.fill_diagonal(dist, np.inf)
        f_rep = (k_fr ** 2 / dist)[:, :, None] * (delta / dist[:, :, None])
        desplaz = f_rep.sum(axis=1)
        if len(edges) > 0:
            ii, jj = edges[:, 0], edges[:, 1]
            d_edge = pos[ii] - pos[jj]
            dist_edge = np.sqrt(np.sum(d_edge ** 2, axis=-1)) + 1e-6
            f_attr = (dist_edge ** 2 / k_fr)[:, None] * (d_edge / dist_edge[:, None])
            np.add.at(desplaz, ii, -f_attr)
            np.add.at(desplaz, jj, f_attr)
        norm = np.linalg.norm(desplaz, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        paso_it = paso * (1.0 - it / iters)
        pos = pos + (desplaz / norm) * np.minimum(norm, paso_it)
        pos = _reflejar(pos, lado)

    sm = HFACT_SPH * k_fr if sep_min == "auto" else sep_min
    if sm is not None and sm > 0:
        pos = _imponer_separacion_minima(pos, lado, sm)
    return pos


def _imponer_separacion_minima(pos, lado, sep_min, max_iter=50):
    """Post-proceso simétrico: cualquier par por debajo de sep_min se separa a lo largo de su propia
    línea de unión, cada partícula la mitad de lo que falta -- no privilegia a ninguna, no cambia la
    dirección de la coherencia ya relajada, sólo respeta el piso de resolución SPH. Iterativo (separar un
    par puede acercar a un tercero) hasta max_iter o hasta que no queden violaciones."""
    n = len(pos)
    if n < 2:
        return pos
    for _ in range(max_iter):
        tree = cKDTree(pos)
        pares = tree.query_pairs(r=sep_min)
        if not pares:
            break
        for i, j in pares:
            d = pos[i] - pos[j]
            dist = np.linalg.norm(d)
            if dist < 1e-12:
                d = np.random.default_rng(i * 1000003 + j).normal(size=pos.shape[1])
                dist = np.linalg.norm(d)
            falta = (sep_min - dist) / 2.0
            direccion = d / dist
            pos[i] = pos[i] + direccion * falta
            pos[j] = pos[j] - direccion * falta
        pos = _reflejar(pos, lado)
    return pos


def _reflejar(pos, lado):
    """Frontera REFLECTANTE (Fruchterman-Reingold estándar) -- reemplaza el clip duro (INSTRUCCION_CC_
    layout_fix_PARA_CC.md): el clip apilaba partículas exactamente en el borde/esquinas (N=250: 246/250
    en el borde, 29/250 en esquinas duplicadas -- artefacto numérico, no la malla causal). La reflexión
    en onda triangular (módulo 2*lado) maneja cualquier magnitud de desborde de forma robusta, no sólo
    un rebote de un paso -- válvula de estabilidad genérica, no un caso especial."""
    span = lado
    y = np.mod(pos, 2.0 * span)
    return np.where(y > span, 2.0 * span - y, y)

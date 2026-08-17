#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cs090_layout_barnes_hut.py — INFRAESTRUCTURA: layout de resortes O(N log N), intercambiable con
`layout_resortes` (cs072_modulos/piezas/p_semilla_causal.py, CONGELADO — no se toca).
========================================================================================================

QUÉ PROBLEMA RESUELVE, EN SIMPLE
--------------------------------
El layout Fruchterman-Reingold que usa todo el proyecto hace, en cada una de sus 100 iteraciones, que
CADA partícula se empuje con TODAS las demás. Con N partículas eso son N×N empujones: al doblar N el
trabajo se cuadruplica. Medido en `FASE6_O3A_convergencia_resolucion_CS.md`: 98 s a N=2000, 656 s a
N=4000 (exponente 2,74 — peor que el 2,0 teórico porque los arreglos N×N×3 se salen de la caché), y
~73 min extrapolados por grafo a N=8000. Eso dejó la serie clavada en N=4000.

La analogía: para saber cuánto tira de mí la gente de una plaza, no necesito preguntarle una por una a
la multitud del fondo — me alcanza con tratar a ese grupo lejano como UN bulto en su centro. Sólo a los
que tengo al lado los cuento uno por uno. Eso es Barnes-Hut (1986): se arma una caja que se subdivide en
ocho (un "octree"), y para cada partícula se recorre el árbol de arriba hacia abajo; si una caja se ve
"chica" desde donde estoy (su ancho dividido por la distancia es menor que un umbral θ), se usa su centro
de masa y no se abre; si se ve "grande", se abre en sus ocho hijas y se sigue mirando. El costo pasa de
N² a N·log N.

QUÉ SE MANTIENE IDÉNTICO AL ORIGINAL (fidelidad de protocolo > elegancia)
------------------------------------------------------------------------
Todo salvo la forma de sumar la repulsión:
  * misma semilla y mismo consumo del RNG: `np.random.default_rng(seed).uniform(0, lado, (n, dims))`;
  * misma k_FR = (lado**dims / n)**(1/dims);
  * misma construcción de la lista de aristas (misma expresión, mismo orden de recorrido);
  * mismo término atractivo sobre aristas, con el mismo `np.add.at` (ya es O(M), no se toca);
  * mismo esquema de enfriamiento: paso = lado*0.1, paso_it = paso*(1 - it/iters), 100 iteraciones;
  * mismo cap del desplazamiento: (desplaz/norm) * min(norm, paso_it), con norm==0 → 1.0;
  * misma frontera reflectante `_reflejar` (se IMPORTA del módulo congelado, no se copia);
  * mismo post-proceso opcional `sep_min` (se IMPORTA `_imponer_separacion_minima`), default None;
  * misma constante HFACT_SPH para el modo sep_min="auto".

LA FÓRMULA EXACTA QUE HAY QUE REPRODUCIR
----------------------------------------
En el original (líneas 116-120 de p_semilla_causal.py):

    delta_ij = pos_i - pos_j
    dist_ij  = |delta_ij| + 1e-6          <-- el +1e-6 va ANTES de la diagonal-infinito
    f_rep_ij = k_FR² * delta_ij / dist_ij²
    desplaz_i = Σ_{j≠i} f_rep_ij

O sea: NO es exactamente una fuerza 1/r (el +1e-6 es un regularizador aditivo). La aproximación de
monopolo la respeta: una celda con `cnt` partículas y centro de masa `c` aporta

    cnt * k_FR² * (pos_i - c) / (|pos_i - c| + 1e-6)²

que es el mismo regularizador aplicado a la distancia al bulto. Con θ=0 ninguna celda se acepta nunca
(el criterio es `ancho < θ · distancia`, imposible con θ=0), el árbol se abre hasta las hojas y la suma
resultante es la MISMA suma partícula-a-partícula del original — el mismo conjunto de sumandos, en otro
orden. Ese es el test de que el árbol no tiene bugs.

CRITERIO DE APERTURA — por qué está escrito como multiplicación y no como división
----------------------------------------------------------------------------------
    aceptar  ⟺  ancho_celda < θ · distancia_al_centro_de_masa
Escrito así (y no como `ancho/distancia < θ`) queda bien definido cuando la partícula cae justo sobre el
centro de masa de la celda (distancia = 0): la celda NO se acepta, se abre — que es lo correcto, y
además evita una división por cero. La celda propia de la partícula siempre se abre por esta vía, y al
llegar a la hoja la partícula se excluye a sí misma explícitamente.

IMPLEMENTACIÓN — octree uniforme, recorrido vectorizado por niveles
-------------------------------------------------------------------
No hay recursión en Python (sería lentísimo). El árbol es un octree UNIFORME de profundidad D sobre la
caja [0, lado]³ (que es exactamente el dominio: `_reflejar` devuelve las posiciones a esa caja en cada
iteración). Para cada nivel L se precalculan, con `np.bincount`, el conteo y el centro de masa de cada
celda. El recorrido mantiene una lista de pares (partícula, celda) que se procesa NIVEL POR NIVEL, toda
de una vez con numpy:

    nivel L:  pares aceptados  -> se acumula su fuerza de monopolo y salen de la lista
              pares abiertos   -> se expanden a las (hasta 8) celdas hijas no vacías, y pasan al nivel L+1
    nivel D:  los pares abiertos se resuelven partícula-a-partícula con los miembros de la celda,
              excluyendo la propia partícula.

Las partículas objetivo se procesan en bloques (`chunk`) para acotar la memoria del arreglo de pares.
La acumulación se hace con `np.bincount` sobre el índice local del bloque (más rápido que `np.add.at`).

QUÉ *NO* GARANTIZA ESTE MÓDULO (dicho de frente)
------------------------------------------------
No garantiza igualdad bit a bit con el original ni siquiera con θ=0, porque suma los mismos números en
otro orden y la suma en punto flotante no es asociativa. La diferencia esperada es del orden del épsilon
de máquina (~1e-16 relativo por operación). Ese mismo orden de magnitud es el ruido propio que el
proyecto YA midió en el método: en `FASE6_O3B_control_rewiring_CS.md`, re-correr el layout original sobre
el MISMO grafo con la MISMA semilla dio diferencias de coordenada de ~1e-15 en 8 de 12 casos y de 15-23
unidades en 4 de 12 (la relajación FR amplifica caóticamente perturbaciones de nivel 1e-16). Por eso la
validación honesta se hace sobre las MÉTRICAS que el proyecto usa, comparadas contra ese piso de ruido,
y no sobre las coordenadas. Ver `INFRA_layout_barnes_hut_CS.md`.

USO
---
    from cs090_layout_barnes_hut import layout_barnes_hut
    pos = layout_barnes_hut(adj_dict, n, lado, iters=100, seed=12345, theta=0.5)

    # firma idéntica a layout_resortes + el parámetro theta (y opcionales de diagnóstico)

Autocomprobación rápida (compara contra el original a N chico):
    ./venv/bin/python cs090_layout_barnes_hut.py
"""
from __future__ import annotations

import math

import numpy as np

# Se IMPORTAN (no se copian) las piezas del módulo congelado, para que la frontera reflectante y el
# post-proceso de separación mínima sean literalmente el mismo código que usa el layout original.
from cs072_modulos.piezas.p_semilla_causal import (HFACT_SPH, _imponer_separacion_minima, _reflejar)

REG = 1e-6          # el mismo regularizador aditivo del original (p_semilla_causal.py línea 117)
OCUPACION_HOJA = 4  # partículas por hoja que se busca al elegir la profundidad del octree
PROF_MIN, PROF_MAX = 2, 6
MAX_PARES_BLOQUE = 4_000_000   # techo de memoria del arreglo de pares (partícula, celda) por bloque


# ==========================================================================================
# 1) El octree: conteos y centros de masa por nivel, calculados con bincount
# ==========================================================================================
def _profundidad_octree(n: int) -> int:
    """Profundidad D tal que la ocupación media de hoja sea ~OCUPACION_HOJA (8^D ≈ n/4)."""
    if n <= OCUPACION_HOJA:
        return PROF_MIN
    d = math.ceil(math.log(n / OCUPACION_HOJA, 8.0))
    return int(min(max(d, PROF_MIN), PROF_MAX))


def _construir_octree(pos: np.ndarray, lado: float, prof: int):
    """Devuelve, por nivel L = 1..prof, el conteo y el centro de masa de cada celda del octree uniforme.

    Las celdas se numeran por índice lineal (ix*2^L + iy)*2^L + iz. El índice de una celda en el nivel L
    se obtiene del índice del nivel `prof` desplazando bits (los octrees uniformes tienen esa propiedad),
    así que se cuantiza UNA sola vez, en el nivel más fino.
    """
    n = len(pos)
    m = 1 << prof                                     # celdas por eje en el nivel más fino
    q = np.clip((pos * (m / lado)).astype(np.int64), 0, m - 1)   # (n,3) índices enteros

    conteos, coms, celdas_finas = {}, {}, None
    for L in range(1, prof + 1):
        desp = prof - L
        mL = 1 << L
        ix, iy, iz = q[:, 0] >> desp, q[:, 1] >> desp, q[:, 2] >> desp
        cid = (ix * mL + iy) * mL + iz
        ncel = mL ** 3
        cnt = np.bincount(cid, minlength=ncel)
        suma = np.empty((ncel, 3))
        for d in range(3):
            suma[:, d] = np.bincount(cid, weights=pos[:, d], minlength=ncel)
        com = np.zeros((ncel, 3))
        vivas = cnt > 0
        com[vivas] = suma[vivas] / cnt[vivas, None]
        conteos[L] = cnt
        coms[L] = com
        if L == prof:
            celdas_finas = cid

    # miembros por celda de hoja, en formato CSR (orden + offsets), para las sumas directas
    orden = np.argsort(celdas_finas, kind="stable")
    cnt_f = conteos[prof]
    inicio = np.zeros(len(cnt_f) + 1, dtype=np.int64)
    np.cumsum(cnt_f, out=inicio[1:])
    return conteos, coms, orden, inicio


# ==========================================================================================
# 2) La repulsión Barnes-Hut
# ==========================================================================================
def repulsion_barnes_hut(pos: np.ndarray, k_fr: float, lado: float, theta: float,
                         prof: int | None = None, contadores: dict | None = None) -> np.ndarray:
    """Σ_{j≠i} k_FR² (x_i - x_j)/(|x_i - x_j| + 1e-6)², aproximada por monopolos con criterio θ.

    Con θ=0 no se acepta ninguna celda y el resultado es la suma exacta partícula-a-partícula (los
    mismos sumandos que el original, en otro orden).
    """
    n = len(pos)
    if n < 2:
        return np.zeros_like(pos)
    if prof is None:
        prof = _profundidad_octree(n)
    conteos, coms, orden, inicio = _construir_octree(pos, lado, prof)
    k2 = k_fr * k_fr

    # tamaño de bloque: acota el arreglo de pares en el peor caso (θ=0 abre TODAS las celdas)
    if theta <= 0.0:
        chunk = max(32, int(MAX_PARES_BLOQUE // max(1, 8 ** prof)))
    else:
        chunk = 2048
    chunk = int(min(n, chunk))

    raiz_vivas = np.flatnonzero(conteos[1] > 0)
    fuerza = np.empty_like(pos)
    n_monopolos = n_directas = 0

    for ini in range(0, n, chunk):
        fin = min(ini + chunk, n)
        nb = fin - ini
        p_loc = np.repeat(np.arange(nb, dtype=np.int64), len(raiz_vivas))
        celdas = np.tile(raiz_vivas, nb)
        acc = np.zeros((nb, 3))
        xs = pos[ini:fin]

        for L in range(1, prof + 1):
            if len(p_loc) == 0:
                break
            ancho = lado / (1 << L)
            dvec = xs[p_loc] - coms[L][celdas]
            dist = np.sqrt(np.einsum("ij,ij->i", dvec, dvec))
            # criterio de apertura escrito como producto: bien definido con dist == 0 (no acepta)
            # En la hoja rige el MISMO criterio: una hoja lejana se acepta como monopolo, una hoja
            # cercana (o la propia) se abre y se resuelve partícula a partícula.
            acepta = ancho < theta * dist

            if acepta.any():
                sel = np.flatnonzero(acepta)
                cnt = conteos[L][celdas[sel]].astype(np.float64)
                dd = dist[sel] + REG
                w = (k2 * cnt / (dd * dd))[:, None] * dvec[sel]
                for d in range(3):
                    acc[:, d] += np.bincount(p_loc[sel], weights=w[:, d], minlength=nb)
                n_monopolos += len(sel)

            abre = ~acepta
            p_ab, c_ab = p_loc[abre], celdas[abre]
            if L < prof:
                # expandir a las 8 hijas; quedarse con las no vacías
                mL1 = 1 << (L + 1)
                ix = c_ab // ((1 << L) * (1 << L))
                iy = (c_ab // (1 << L)) % (1 << L)
                iz = c_ab % (1 << L)
                hijas = (((2 * ix)[:, None] + np.array([0, 0, 0, 0, 1, 1, 1, 1])) * mL1
                         + ((2 * iy)[:, None] + np.array([0, 0, 1, 1, 0, 0, 1, 1]))) * mL1 \
                        + ((2 * iz)[:, None] + np.array([0, 1, 0, 1, 0, 1, 0, 1]))
                p_loc = np.repeat(p_ab, 8)
                celdas = hijas.ravel()
                viva = conteos[L + 1][celdas] > 0
                p_loc, celdas = p_loc[viva], celdas[viva]
                if len(p_loc) > 8 * MAX_PARES_BLOQUE:
                    raise MemoryError(f"lista de pares desbordada en nivel {L+1}: {len(p_loc)}")
            else:
                # HOJA: suma directa partícula-a-partícula con los miembros de la celda
                if len(p_ab):
                    largos = conteos[prof][c_ab]
                    p_rep = np.repeat(p_ab, largos)
                    off = np.repeat(inicio[c_ab], largos)
                    base = np.repeat(np.cumsum(largos) - largos, largos)
                    idx = orden[off + (np.arange(len(p_rep)) - base)]
                    real = idx != (p_rep + ini)          # excluir la propia partícula
                    p_rep, idx = p_rep[real], idx[real]
                    dv = xs[p_rep] - pos[idx]
                    dj = np.sqrt(np.einsum("ij,ij->i", dv, dv)) + REG
                    w = (k2 / (dj * dj))[:, None] * dv
                    for d in range(3):
                        acc[:, d] += np.bincount(p_rep, weights=w[:, d], minlength=nb)
                    n_directas += len(p_rep)
                p_loc = np.empty(0, dtype=np.int64)

        fuerza[ini:fin] = acc

    if contadores is not None:
        contadores.update(prof=prof, n_monopolos=n_monopolos, n_directas=n_directas,
                          interacciones=n_monopolos + n_directas, chunk=chunk)
    return fuerza


def repulsion_exacta(pos: np.ndarray, k_fr: float) -> np.ndarray:
    """La repulsión del original, tal cual (líneas 116-120 de p_semilla_causal.py). Se reproduce acá
    SÓLO como referencia de validación — el original no se modifica ni se llama a medias."""
    delta = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt(np.sum(delta ** 2, axis=-1)) + REG
    np.fill_diagonal(dist, np.inf)
    f_rep = (k_fr ** 2 / dist)[:, :, None] * (delta / dist[:, :, None])
    return f_rep.sum(axis=1)


def repulsion_exacta_reordenada(pos: np.ndarray, k_fr: float) -> np.ndarray:
    """MISMA suma exacta, MISMOS sumandos, sumados en orden inverso (j de N-1 a 0).

    Para qué sirve: es el CONTROL DE REDONDEO. Matemáticamente da lo mismo que `repulsion_exacta`; en
    punto flotante difiere en el último bit (~1e-16 relativo), porque la suma no es asociativa. Usada
    dentro del mismo bucle de 100 iteraciones, mide cuánto se separa el layout Fruchterman-Reingold de
    sí mismo cuando lo único que cambia es el redondeo — o sea, el PISO DE RUIDO PROPIO DEL MÉTODO,
    exactamente el fenómeno que `FASE6_O3B_control_rewiring_CS.md` observó por accidente (8/12 corridas
    con diferencia ~1e-15, 4/12 con diferencia de 15-23 unidades). Es el patrón de comparación honesto
    contra el cual hay que medir el error que introduce Barnes-Hut.
    """
    delta = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt(np.sum(delta ** 2, axis=-1)) + REG
    np.fill_diagonal(dist, np.inf)
    f_rep = (k_fr ** 2 / dist)[:, :, None] * (delta / dist[:, :, None])
    return f_rep[:, ::-1, :].sum(axis=1)


# ==========================================================================================
# 3) El layout completo — mismo protocolo, sólo cambia cómo se suma la repulsión
# ==========================================================================================
def layout_barnes_hut(adj, n, lado, dims=3, iters=100, seed=12345, sep_min=None,
                      theta=0.5, prof=None, diagnostico=None, metodo="bh"):
    """Reemplazo intercambiable de `layout_resortes` con repulsión Barnes-Hut O(N log N).

    Firma idéntica al original más `theta` (criterio de apertura; θ=0 ⇒ suma exacta), `prof` (forzar la
    profundidad del octree; None = automática) y `diagnostico` (dict que se rellena con contadores).

    dims != 3 no está soportado por el octree: en ese caso se cae a la suma exacta y se avisa en el
    diagnóstico, para no cambiar silenciosamente el resultado.

    `metodo` selecciona cómo se suma la repulsión, SIN tocar nada más del protocolo:
      "bh"                 — Barnes-Hut con el θ pedido (el modo de producción);
      "exacta"             — la suma N² del original (sirve para verificar que este bucle reproduce
                             `layout_resortes` bit a bit: debe dar diferencia 0.0);
      "exacta_reordenada"  — la misma suma N² en orden inverso (CONTROL DE REDONDEO / piso de ruido).
    """
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, lado, size=(n, dims))
    if n < 2:
        return pos
    k_fr = (lado ** dims / n) ** (1.0 / dims)
    edges = np.array([(i, j) for i in range(n) for j in adj.get(i, ()) if j > i], dtype=int)
    paso = lado * 0.1
    usar_bh = (dims == 3) and metodo == "bh"
    cont = {}
    for it in range(iters):
        if usar_bh:
            desplaz = repulsion_barnes_hut(pos, k_fr, lado, theta, prof=prof, contadores=cont)
        elif metodo == "exacta_reordenada":
            desplaz = repulsion_exacta_reordenada(pos, k_fr)
        else:
            desplaz = repulsion_exacta(pos, k_fr)
        if len(edges) > 0:
            ii, jj = edges[:, 0], edges[:, 1]
            d_edge = pos[ii] - pos[jj]
            dist_edge = np.sqrt(np.sum(d_edge ** 2, axis=-1)) + REG
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
    if diagnostico is not None:
        diagnostico.update(cont)
        diagnostico["usar_bh"] = usar_bh
        diagnostico["metodo"] = metodo
        diagnostico["theta"] = theta
        diagnostico["n_aristas"] = len(edges)
    return pos


# ==========================================================================================
# 4) Autocomprobación
# ==========================================================================================
if __name__ == "__main__":
    import time
    import sys
    sys.path.insert(0, "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis")
    from cs072_modulos.piezas.p_semilla_causal import layout_resortes

    rng = np.random.default_rng(7)
    N, LADO = 600, float(2000) ** (1 / 3)
    adj = {i: set() for i in range(N)}
    for _ in range(3 * N):
        a, b = rng.integers(0, N, 2)
        if a != b:
            adj[int(a)].add(int(b))
            adj[int(b)].add(int(a))

    # (a) una sola evaluación de fuerza: BH(θ=0) contra la suma exacta
    p0 = rng.uniform(0, LADO, size=(N, 3))
    k = (LADO ** 3 / N) ** (1 / 3)
    fe = repulsion_exacta(p0, k)
    for th in (0.0, 0.3, 0.5, 0.7, 1.0):
        fb = repulsion_barnes_hut(p0, k, LADO, th)
        err = np.linalg.norm(fb - fe, axis=1)
        print(f"  theta={th:<4} err_rel_medio={np.mean(err)/np.mean(np.linalg.norm(fe,axis=1)):.3e} "
              f"err_max_abs={err.max():.3e}")

    # (b) layout completo
    t0 = time.time(); pa = layout_resortes(adj, N, LADO, iters=100, seed=12345); ta = time.time() - t0
    t0 = time.time(); pb = layout_barnes_hut(adj, N, LADO, iters=100, seed=12345, theta=0.0)
    tb = time.time() - t0
    print(f"  layout N={N}: original {ta:.2f}s  BH(theta=0) {tb:.2f}s  "
          f"RMS={np.sqrt(np.mean(np.sum((pa-pb)**2,axis=1))):.3e}  max={np.abs(pa-pb).max():.3e}")

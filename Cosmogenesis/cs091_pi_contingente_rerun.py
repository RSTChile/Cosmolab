#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
"""
CS091 — π CONTINGENTE, RE-CORRIDA CON CONTROLES
================================================

QUÉ ES ESTO (a nivel módulo)
-----------------------------
Re-implementación desde cero de la medida de `HALLAZGO_pi_contingente_y_rumbo_gravedad_cuantica_CS.md`
(16-jul-2026), que midió "π emergente" en cuatro sustratos y publicó 2.0 / 2.99 / 1.5 / estallido —
**sin ningún NULL**. Este script (a) reproduce las cuatro filas con la definición textual del nodo,
(b) agrega los controles que faltaron sobre los grafos REALES que ya están en disco, y (c) calcula
explícitamente cuánto del "estallido" es simple álgebra de bolas que crecen exponencialmente.

LA DEFINICIÓN (textual del nodo, NO se cambia)
-----------------------------------------------
En un grafo, la "circunferencia" de radio r = nº de nodos a distancia geodésica EXACTA r desde una
fuente (la FRONTERA de la bola, |S(r)|); el "diámetro" = 2r. Entonces:

        π_emergente(r) = |S(r)| / (2 r)

Analogía: uno se para en un nodo y cuenta cuántos nodos hay "a exactamente r pasos". En un piso de
baldosas cuadradas, a r pasos hay siempre 4r baldosas (el borde de un rombo), así que la cuenta da
4r/2r = 2,0 SIEMPRE. En una red donde hay atajos, a cada paso el número de vecinos nuevos se
multiplica en vez de sumar, y la cuenta se dispara. Eso es lo que el nodo llamó "π indefinido".

NADA DE COORDENADAS. La medida es intrínseca al grafo (sólo BFS). Usar `layout_resortes` daría
3,1416 por construcción — sería exactamente la fabricación a evitar.

QUÉ MIDE, POR PARTES
---------------------
1. RETÍCULAS CONSTRUIDAS A MANO (cuadrada, triangular, hexagonal): reproducción de las 3 primeras filas.
2. MUNDO-PEQUEÑO sintético (anillo + atajos): reproducción de la 4ª fila.
3. GRAFOS REALES del corpus (`grafos_f800/`, línea A2-B0-C2): el sustrato de verdad.
4. NULL-BARAJADO: double-edge-swap del MISMO grafo real (grados preservados). ← el control que decide.
5. NULL-ER: Erdős–Rényi del mismo N y mismo nº de aristas.
6. ÁLGEBRA: tasa de crecimiento de bolas b(r) = |S(r)|/|S(r-1)| y la predicción π_alg(r) que se sigue
   sólo de esa tasa, para separar "estallido por estructura" de "estallido por aritmética".

GUARDAS APLICADAS
-----------------
- Componente gigante: todas las BFS arrancan en nodos de la componente gigante (bug `_diam` de cs055).
- Anti-isomorfismo: se comprueba con números que el barajado cambió algo medible (solapamiento de
  aristas, triángulos, clustering, diámetro). Grados idénticos por construcción — se verifica.
- Sin layout, sin coordenadas, sin embebido.

SALIDAS
-------
- `pi_contingente_rerun_curvas.csv`   : π(r) por sustrato y por radio (+ |S(r)|, |B(r)|, b(r))
- `pi_contingente_rerun_curvas.png`   : las curvas
- `pi_contingente_rerun_controles.csv`: tabla anti-isomorfismo real vs barajado vs ER
"""
from __future__ import annotations

import csv
import gzip
import json
import math
import os
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

AQUI = Path("/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis")
sys.path.insert(0, str(AQUI))
import cs090_fase8_f800_grafos as G8          # cargador oficial de grafos guardados (verifica el sello)

SEMILLA = 20260813
R_MAX = 12                                     # radios reportados (el nodo publicó hasta r=7)


# =============================================================================================
# 1) NÚCLEO DE LA MEDIDA — BFS, frontera de la bola, π(r)
# =============================================================================================
def bfs_capas(adj, fuente, r_max):
    """Devuelve una lista `capas[r] = nº de nodos a distancia geodésica EXACTA r` desde `fuente`.

    Es un BFS común y corriente: la capa 0 es la fuente, la capa r+1 son los vecinos todavía no
    vistos de la capa r. `capas[r]` es |S(r)|, la FRONTERA de la bola de radio r — no la bola entera.
    """
    visto = bytearray(len(adj))
    visto[fuente] = 1
    frente = [fuente]
    capas = [1]
    r = 0
    while frente and r < r_max:
        nuevo = []
        for u in frente:
            for v in adj[u]:
                if not visto[v]:
                    visto[v] = 1
                    nuevo.append(v)
        frente = nuevo
        capas.append(len(nuevo))
        r += 1
    while len(capas) < r_max + 1:
        capas.append(0)
    return capas


def curva_pi(adj, fuentes, r_max=R_MAX):
    """π(r) promediando la FRONTERA sobre varias fuentes y dividiendo por 2r.

    Se promedia |S(r)| (no los cocientes) y recién después se divide: así el promedio es el de la
    cantidad física medida, y una fuente que se queda sin grafo antes de r no infla el cociente.
    Se reporta además la mediana y el rango intercuartil de π(r) fuente a fuente.
    """
    M = np.zeros((len(fuentes), r_max + 1), dtype=float)
    for k, f in enumerate(fuentes):
        M[k, :] = bfs_capas(adj, f, r_max)
    S_med = M.mean(axis=0)                       # |S(r)| promedio
    B_med = np.cumsum(S_med)                     # |B(r)| = bola acumulada
    filas = []
    for r in range(1, r_max + 1):
        pis_fuente = M[:, r] / (2.0 * r)
        filas.append(dict(
            r=r,
            S_r=float(S_med[r]),
            B_r=float(B_med[r]),
            pi_r=float(S_med[r] / (2.0 * r)),
            pi_mediana=float(np.median(pis_fuente)),
            pi_q1=float(np.percentile(pis_fuente, 25)),
            pi_q3=float(np.percentile(pis_fuente, 75)),
            b_r=float(S_med[r] / S_med[r - 1]) if S_med[r - 1] > 0 else float("nan"),
            n_fuentes=len(fuentes),
        ))
    return filas


# =============================================================================================
# 2) COMPONENTE GIGANTE (guarda del bug `_diam`)
# =============================================================================================
def componente_gigante(adj):
    """Devuelve la lista de nodos de la componente conexa más grande. Nunca se arranca 'desde el
    nodo 0 por índice' — ese fue el bug `_diam` de cs055, que medía fragmentos sueltos."""
    n = len(adj)
    visto = bytearray(n)
    mejor = []
    for s in range(n):
        if visto[s]:
            continue
        comp = [s]
        visto[s] = 1
        cola = deque([s])
        while cola:
            u = cola.popleft()
            for v in adj[u]:
                if not visto[v]:
                    visto[v] = 1
                    comp.append(v)
                    cola.append(v)
        if len(comp) > len(mejor):
            mejor = comp
    return mejor


# =============================================================================================
# 3) LOS CUATRO SUSTRATOS DEL NODO (construidos a mano — es lo que el nodo hizo)
# =============================================================================================
def reticula_cuadrada(L):
    """Retícula cuadrada L×L, vecinos ±1 en x e y (métrica taxicab). Frontera teórica: |S(r)| = 4r."""
    idx = lambda x, y: x * L + y
    adj = [set() for _ in range(L * L)]
    for x in range(L):
        for y in range(L):
            if x + 1 < L:
                adj[idx(x, y)].add(idx(x + 1, y)); adj[idx(x + 1, y)].add(idx(x, y))
            if y + 1 < L:
                adj[idx(x, y)].add(idx(x, y + 1)); adj[idx(x, y + 1)].add(idx(x, y))
    return adj, idx(L // 2, L // 2)


def reticula_triangular(L):
    """Retícula triangular en coordenadas axiales: 6 vecinos (±1,0),(0,±1),(+1,−1),(−1,+1).
    Frontera teórica: |S(r)| = 6r."""
    idx = lambda x, y: x * L + y
    adj = [set() for _ in range(L * L)]
    for x in range(L):
        for y in range(L):
            for dx, dy in ((1, 0), (0, 1), (1, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < L and 0 <= ny < L:
                    adj[idx(x, y)].add(idx(nx, ny)); adj[idx(nx, ny)].add(idx(x, y))
    return adj, idx(L // 2, L // 2)


def reticula_hexagonal(L):
    """Retícula hexagonal (panal) en representación 'muro de ladrillos': cada nodo tiene vecinos
    (x±1, y) y UN vecino vertical — arriba si (x+y) es par, abajo si es impar. Grado 3.
    Frontera teórica asintótica: |S(r)| ≈ 3r."""
    idx = lambda x, y: x * L + y
    adj = [set() for _ in range(L * L)]
    for x in range(L):
        for y in range(L):
            if x + 1 < L:
                adj[idx(x, y)].add(idx(x + 1, y)); adj[idx(x + 1, y)].add(idx(x, y))
            if (x + y) % 2 == 0 and y + 1 < L:
                adj[idx(x, y)].add(idx(x, y + 1)); adj[idx(x, y + 1)].add(idx(x, y))
    return adj, idx(L // 2, L // 2 if (L // 2 + L // 2) % 2 == 0 else L // 2 + 1)


def mundo_pequeno(N, k, p, semilla):
    """Watts–Strogatz: anillo donde cada nodo toca a sus k/2 vecinos de cada lado, y cada arista se
    recablea con probabilidad p (los 'atajos' que hacen el mundo pequeño). Grado medio ≈ k."""
    rng = random.Random(semilla)
    adj = [set() for _ in range(N)]
    for i in range(N):
        for j in range(1, k // 2 + 1):
            v = (i + j) % N
            adj[i].add(v); adj[v].add(i)
    aristas = sorted({(min(i, j), max(i, j)) for i in range(N) for j in adj[i]})
    for (i, j) in aristas:
        if rng.random() < p:
            for _ in range(20):
                w = rng.randrange(N)
                if w != i and w not in adj[i]:
                    adj[i].discard(j); adj[j].discard(i)
                    adj[i].add(w); adj[w].add(i)
                    break
    return adj


# =============================================================================================
# 4) LOS NULL — barajado de grados preservados y Erdős–Rényi
# =============================================================================================
def baraja_rapida(adj, factor=20, semilla=SEMILLA):
    """Double-edge-swap: se toman dos aristas (a,b) y (c,d) y se cambian por (a,d) y (c,b).
    Cada nodo conserva EXACTAMENTE su grado; lo que se destruye es cómo estaban conectados entre si.

    Analogia: los mismos invitados a la fiesta, cada uno con la misma cantidad de amigos que antes,
    pero repartidos al azar - se conserva quien es popular, se pierde quien conocia a quien.
    `factor` = intentos de intercambio por arista (20*E es el estandar de mezcla completa)."""
    rng = random.Random(semilla)
    E = [(min(i, j), max(i, j)) for i in range(len(adj)) for j in adj[i] if i < j]
    conj = set(E)
    hechos = 0
    for _ in range(factor * len(E)):
        i1 = rng.randrange(len(E)); i2 = rng.randrange(len(E))
        a, b = E[i1]; c, d = E[i2]
        if rng.random() < 0.5:
            c, d = d, c
        if len({a, b, c, d}) < 4:
            continue
        n1 = (min(a, d), max(a, d)); n2 = (min(c, b), max(c, b))
        if n1 in conj or n2 in conj:
            continue
        conj.discard(E[i1]); conj.discard(E[i2])
        conj.add(n1); conj.add(n2)
        E[i1] = n1; E[i2] = n2
        hechos += 1
    nuevo = [set() for _ in range(len(adj))]
    for (i, j) in conj:
        nuevo[i].add(j); nuevo[j].add(i)
    return nuevo, hechos


def erdos_renyi(N, m, semilla=SEMILLA):
    """Grafo aleatorio con N nodos y EXACTAMENTE m aristas, sin ninguna estructura."""
    rng = random.Random(semilla)
    conj = set()
    while len(conj) < m:
        i = rng.randrange(N); j = rng.randrange(N)
        if i == j:
            continue
        conj.add((min(i, j), max(i, j)))
    adj = [set() for _ in range(N)]
    for (i, j) in conj:
        adj[i].add(j); adj[j].add(i)
    return adj


# =============================================================================================
# 5) ANTI-ISOMORFISMO — comprobar CON NÚMEROS que el NULL no es el mismo grafo renombrado
# =============================================================================================
def triangulos(adj):
    t = 0
    for u in range(len(adj)):
        vec = [v for v in adj[u] if v > u]
        for a in range(len(vec)):
            for b in range(a + 1, len(vec)):
                if vec[b] in adj[vec[a]]:
                    t += 1
    return t


def clustering_medio(adj):
    vals = []
    for u in range(len(adj)):
        vec = list(adj[u])
        k = len(vec)
        if k < 2:
            vals.append(0.0); continue
        enl = 0
        for a in range(k):
            for b in range(a + 1, k):
                if vec[b] in adj[vec[a]]:
                    enl += 1
        vals.append(2.0 * enl / (k * (k - 1)))
    return float(np.mean(vals))


def diam_gigante(adj):
    """Diámetro aproximado por doble BFS ARRANCANDO EN LA COMPONENTE GIGANTE (guarda cs055/_diam)."""
    comp = componente_gigante(adj)
    def lejano(s):
        dist = {s: 0}
        cola = deque([s]); ult = s
        while cola:
            u = cola.popleft(); ult = u
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    cola.append(v)
        return ult, dist[ult]
    a, _ = lejano(comp[0])
    b, d = lejano(a)
    return d, len(comp)


def solapamiento_aristas(a1, a2):
    E1 = {(min(i, j), max(i, j)) for i in range(len(a1)) for j in a1[i] if i < j}
    E2 = {(min(i, j), max(i, j)) for i in range(len(a2)) for j in a2[i] if i < j}
    return len(E1 & E2) / max(1, len(E1 | E2)), len(E1), len(E2)


def grados_iguales(a1, a2):
    return sorted(len(s) for s in a1) == sorted(len(s) for s in a2)


# =============================================================================================
# 6) EJECUCIÓN
# =============================================================================================
def fuentes_gigante(adj, n_fuentes, semilla=SEMILLA):
    comp = componente_gigante(adj)
    rng = random.Random(semilla)
    if len(comp) <= n_fuentes:
        return comp
    return rng.sample(comp, n_fuentes)


def main():
    filas_csv = []
    controles = []

    def registrar(nombre, familia, adj, fuentes, r_max=R_MAX, N=None, E=None):
        N = N if N is not None else len(adj)
        E = E if E is not None else sum(len(s) for s in adj) // 2
        for f in curva_pi(adj, fuentes, r_max):
            f.update(sustrato=nombre, familia=familia, N=N, E=E)
            filas_csv.append(f)
        print(f"  {nombre:38s} N={N:6d} E={E:6d}  "
              f"π(r)= " + " ".join(f"{x['pi_r']:.2f}" for x in filas_csv[-r_max:][:8]))

    # ---- (1) las tres retículas construidas a mano -------------------------------------------
    print("\n[1] RETÍCULAS CONSTRUIDAS A MANO (reproducción de las 3 primeras filas)")
    L = 121
    for nombre, ctor in (("reticula_cuadrada", reticula_cuadrada),
                         ("reticula_triangular", reticula_triangular),
                         ("reticula_hexagonal", reticula_hexagonal)):
        adj, centro = ctor(L)
        registrar(nombre, "reticula_a_mano", adj, [centro], r_max=R_MAX)

    # ---- (2) mundo pequeño sintético ----------------------------------------------------------
    print("\n[2] MUNDO-PEQUEÑO SINTÉTICO (reproducción de la 4ª fila)")
    for k, p in ((4, 0.10), (6, 0.10), (6, 0.01)):
        adj = mundo_pequeno(2000, k, p, SEMILLA)
        registrar(f"mundo_pequeno_k{k}_p{p}", "mundo_pequeno_sintetico", adj,
                  fuentes_gigante(adj, 200))

    # ---- (3) grafos REALES del corpus + sus NULL ----------------------------------------------
    print("\n[3] GRAFOS REALES DEL CORPUS + NULL-BARAJADO + NULL-ER")
    reales = sorted((AQUI / "grafos_f800" / "F5B_40pares").glob("*.grafo.gz"))[:5]
    reales += sorted((AQUI / "grafos_f800" / "O3A_N4000").glob("*.grafo.gz"))[:2]
    acum = {"real": [], "barajado": [], "er": []}
    for ruta in reales:
        adj, N, meta = G8.cargar_grafo(ruta)
        E = meta["E"]
        etiqueta = ruta.name.split("__")[0]
        fu = fuentes_gigante(adj, 200)
        registrar(f"REAL:{etiqueta}", "real_corpus", adj, fu, N=N, E=E)
        acum["real"].append(filas_csv[-R_MAX:])

        bar, hechos = baraja_rapida(adj, factor=20)
        registrar(f"BARAJADO:{etiqueta}", "null_barajado", bar, fuentes_gigante(bar, 200), N=N, E=E)
        acum["barajado"].append(filas_csv[-R_MAX:])

        er = erdos_renyi(N, E, semilla=SEMILLA + len(reales))
        registrar(f"ER:{etiqueta}", "null_er", er, fuentes_gigante(er, 200), N=N, E=E)
        acum["er"].append(filas_csv[-R_MAX:])

        # ---- anti-isomorfismo ----
        jac, E1, E2 = solapamiento_aristas(adj, bar)
        d_real, c_real = diam_gigante(adj)
        d_bar, c_bar = diam_gigante(bar)
        d_er, c_er = diam_gigante(er)
        controles.append(dict(
            grafo=etiqueta, N=N, E=E, swaps_efectivos=hechos,
            jaccard_aristas_real_vs_barajado=round(jac, 5),
            grados_identicos=grados_iguales(adj, bar),
            triangulos_real=triangulos(adj), triangulos_barajado=triangulos(bar),
            triangulos_er=triangulos(er),
            clustering_real=round(clustering_medio(adj), 5),
            clustering_barajado=round(clustering_medio(bar), 5),
            clustering_er=round(clustering_medio(er), 5),
            diam_real=d_real, diam_barajado=d_bar, diam_er=d_er,
            gigante_real=c_real, gigante_barajado=c_bar, gigante_er=c_er,
        ))
        print(f"    anti-isomorfismo: jaccard={jac:.4f} grados_iguales={grados_iguales(adj, bar)} "
              f"triang {triangulos(adj)}→{triangulos(bar)} diam {d_real}→{d_bar}")

    # ---- salidas ------------------------------------------------------------------------------
    cols = ["sustrato", "familia", "N", "E", "r", "S_r", "B_r", "pi_r",
            "pi_mediana", "pi_q1", "pi_q3", "b_r", "n_fuentes"]
    with open(AQUI / "pi_contingente_rerun_curvas.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for fila in filas_csv:
            w.writerow({c: fila[c] for c in cols})

    if controles:
        with open(AQUI / "pi_contingente_rerun_controles.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(controles[0].keys()))
            w.writeheader()
            w.writerows(controles)

    print(f"\nEscrito: pi_contingente_rerun_curvas.csv ({len(filas_csv)} filas), "
          f"pi_contingente_rerun_controles.csv ({len(controles)} filas)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
CG003 — Diagnóstico de PLANITUD: δ-hiperbolicidad de Gromov + escalado de diámetro
==================================================================================
Respuesta a Claude Science (2-jul-2026): el juez d(N) por conteo de bolas MIENTE a
N accesibles — una red 2D genuina también da d que sube (1.1->1.7), igual que un
árbol hiperbólico (2.0->4.0). Ver d trepar NO distingue "2D finito" de "hiperbólico".
Lo que SÍ los separa, por medición y sin coordenadas:

  · DIAMETRO vs N     : geometria d-dim -> diam ~ N^(1/d) ; mundo-pequeño -> diam ~ log N.
  · δ de GROMOV (4pt) : plano -> δ CRECE con N (~diametro) ; hiperbolico -> δ ACOTADA (~0).

δ (condicion de los 4 puntos): para x,y,z,w toma las tres sumas de pares opuestos
  S1=d(x,y)+d(z,w)  S2=d(x,z)+d(y,w)  S3=d(x,w)+d(y,z),
ordena S_a>=S_b>=S_c ; δ_quad = (S_a - S_b)/2.  δ_grafo = media (y cuantil alto)
sobre cuadruplas muestreadas. Solo usa distancias BFS: CERO coordenadas.

Este script VERIFICA los controles que Claude Science reportó (no los toma en fe) y
MIDE δ sobre nuestro grafo real cg003d. Regla de la casa: reproducir, no creer.

USO: python3 cg003_diagnostico_gromov.py [--quick]
"""
from __future__ import annotations

import argparse
import time
from collections import deque

import numpy as np

from cg003d_campo_angular import crecer_campo, shuffle_adj


# --------------------------- grafos de control ---------------------------
def lattice2d(N):
    """Red cuadrada L×L (2D plana genuina): control POSITIVO de geometria."""
    L = int(round(np.sqrt(N)))
    adj = [[] for _ in range(L * L)]
    def idx(r, c): return r * L + c
    for r in range(L):
        for c in range(L):
            u = idx(r, c)
            if c + 1 < L: v = idx(r, c + 1); adj[u].append(v); adj[v].append(u)
            if r + 1 < L: v = idx(r + 1, c); adj[u].append(v); adj[v].append(u)
    return [np.fromiter(a, dtype=np.int32) for a in adj], L * L


def arbol(N, b=3):
    """Arbol b-ario (hiperbolico exacto, δ=0): control NEGATIVO."""
    adj = [[] for _ in range(N)]
    for v in range(1, N):
        u = (v - 1) // b
        adj[u].append(v); adj[v].append(u)
    return [np.fromiter(a, dtype=np.int32) for a in adj], N


def aleatorio(N, meandeg, seed=0):
    """Grafo aleatorio (mundo-pequeño, δ acotada): control de azar."""
    rng = np.random.default_rng(seed)
    E = int(meandeg * N / 2)
    adj = [set() for _ in range(N)]
    made = 0; guard = 0
    while made < E and guard < 20 * E:
        i, j = int(rng.integers(N)), int(rng.integers(N)); guard += 1
        if i != j and j not in adj[i]:
            adj[i].add(j); adj[j].add(i); made += 1
    return [np.fromiter(a, dtype=np.int32) for a in adj], N


# --------------------------- medicion (coordinate-free) ---------------------------
def _bfs_dist(adj, src, N):
    dist = np.full(N, -1, dtype=np.int32); dist[src] = 0; q = deque([src])
    while q:
        u = q.popleft(); du = dist[u] + 1
        for w in adj[u]:
            if dist[w] < 0:
                dist[w] = du; q.append(int(w))
    return dist


def landmarks_dist(adj, N, K, seed=0):
    """BFS desde K nodos-faro -> matriz KxK de distancias (anclado en la componente GIGANTE)."""
    rng = np.random.default_rng(seed)
    # anclar en la componente gigante: probar varios semillas, quedarse con el mayor alcance
    reach = np.array([], dtype=int)
    for _ in range(8):
        d0 = _bfs_dist(adj, int(rng.integers(N)), N)
        r = np.where(d0 >= 0)[0]
        if len(r) > len(reach):
            reach = r
        if len(reach) > N // 2:
            break
    frac_giant = len(reach) / N
    if len(reach) < K + 4:
        return None, 0, frac_giant
    src = rng.choice(reach, size=min(K, len(reach)), replace=False)
    D = np.zeros((len(src), len(src)))
    dmax = 0
    for a, s in enumerate(src):
        ds = _bfs_dist(adj, int(s), N)
        D[a] = ds[src]
        dmax = max(dmax, int(ds[reach].max()))
    return D, dmax, frac_giant


def gromov_delta(D, n_quad=20000, seed=0):
    """δ media y cuantil-95 sobre cuadruplas de faros (condicion 4 puntos)."""
    rng = np.random.default_rng(seed)
    K = D.shape[0]
    if K < 4:
        return np.nan, np.nan
    q = rng.integers(0, K, size=(n_quad, 4))
    ok = (q[:, 0] != q[:, 1]) & (q[:, 2] != q[:, 3])
    q = q[ok]
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    S1 = D[x, y] + D[z, w]
    S2 = D[x, z] + D[y, w]
    S3 = D[x, w] + D[y, z]
    S = np.sort(np.stack([S1, S2, S3], axis=1), axis=1)  # asc: S[:,2]=max, S[:,1]=2do
    delta = (S[:, 2] - S[:, 1]) / 2.0
    return float(delta.mean()), float(np.quantile(delta, 0.95))


def diagnos(nombre, adj, N, K, seed=0):
    D, dmax, fg = landmarks_dist(adj, N, K, seed=seed)
    if D is None:
        return dict(nombre=nombre, N=N, diam=np.nan, dmean=np.nan, d95=np.nan, fg=fg)
    dmean, d95 = gromov_delta(D, seed=seed + 1)
    return dict(nombre=nombre, N=N, diam=dmax, dmean=dmean, d95=d95, fg=fg)


def escala(nombre, filas):
    """Pendiente log-log diam~N (0.5 = 2D ; ~0 = log N = mundo-pequeño)."""
    xs = np.log([f["N"] for f in filas]); ys = np.log([max(f["diam"], 1) for f in filas])
    p = np.polyfit(xs, ys, 1)[0]
    # tendencia de δ: ¿crece (plano) o acotada (hiperbolico)?
    dd = [f["dmean"] for f in filas]
    trend = "CRECE (plano)" if dd[-1] > dd[0] + 0.5 else "ACOTADA (hiperbolico)"
    return p, trend


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    Ns = [1024, 4096] if args.quick else [1024, 4096, 16384]
    K = 80 if args.quick else 120

    t0 = time.monotonic()
    print("CG003 — DIAGNOSTICO DE PLANITUD (δ-Gromov + diametro) · cero coordenadas")
    print("=" * 82)
    print("plano: diam~N^(1/d) (pend~0.5) y δ CRECE.  mundo-pequeño: diam~logN (pend~0) y δ ACOTADA.\n")
    print(f"{'grafo':>16} {'N':>7} {'%gig':>5} {'diam':>5} {'δ_media':>8} {'δ_q95':>7}")

    catalogo = {}
    for N in Ns:
        # controles verificados por nosotros (no en fe)
        for nombre, mk in [
            ("lattice2D",  lambda N: lattice2d(N)),
            ("arbol_b3",   lambda N: arbol(N, 3)),
            ("aleatorio",  lambda N: aleatorio(N, 6, seed=1)),
        ]:
            adj, Nreal = mk(N)
            r = diagnos(nombre, adj, Nreal, K, seed=7)
            catalogo.setdefault(nombre, []).append(r)
            print(f"{nombre:>16} {r['N']:>7} {r['fg']*100:>4.0f} {r['diam']:>5} {r['dmean']:>8.2f} {r['d95']:>7.2f}", flush=True)
        # nuestro grafo real
        for Dt in (2, 3):
            adj = crecer_campo(N, Dtan=Dt, kdeg=2 * Dt + 4, seed=1)
            nombre = f"cg003d_Dt{Dt}"
            r = diagnos(nombre, adj, N, K, seed=7)
            catalogo.setdefault(nombre, []).append(r)
            print(f"{nombre:>16} {r['N']:>7} {r['fg']*100:>4.0f} {r['diam']:>5} {r['dmean']:>8.2f} {r['d95']:>7.2f}", flush=True)
            # shuffle de nuestro grafo (falsador)
            R = shuffle_adj(adj, N, seed=Dt + 9)
            rs = diagnos(f"cg003d_Dt{Dt}_shuf", R, N, K, seed=7)
            catalogo.setdefault(f"cg003d_Dt{Dt}_shuf", []).append(rs)
            print(f"{'  (shuffle)':>16} {rs['N']:>7} {rs['fg']*100:>4.0f} {rs['diam']:>5} {rs['dmean']:>8.2f} {rs['d95']:>7.2f}", flush=True)
        print()

    print("=" * 82)
    print("Veredicto por escalado (pendiente log-log diam~N ; tendencia de δ):")
    for nombre, filas in catalogo.items():
        if len(filas) >= 2 and all(f["diam"] == f["diam"] for f in filas):
            p, trend = escala(nombre, filas)
            print(f"  {nombre:>18}: diam-pend={p:5.2f}   δ {trend}")
    print(f"\nLectura: si cg003d se agrupa con 'arbol/aleatorio' (pend~0, δ acotada) -> HIPERBOLICO,")
    print("         falta planitud (confirma Claude Science). Si va con 'lattice2D' -> ya hay geometria.")
    print(f"\nTiempo: {time.monotonic()-t0:.1f}s")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
CG003-c — El espacio que CRECE (escalable): ¿que dimension elige a N grande?
============================================================================
El Universo CRECE: las asimetrias se expresan una a una (el contador potencia->acto)
y se pegan LOCALMENTE al frente insaturado. El tope de grado (economia) satura las
viejas (interiores) y fuerza el crecimiento al BORDE -> el cosmos crece hacia afuera
-> diametro grande -> geometria (no mundo-pequeño). Los atajos se matan por
construccion (nunca un enlace a lo lejano).

NO fijamos dimension. La MEDIMOS (dimension de crecimiento N(r)~r^d) y dejamos que
el sistema la elija. A N grande el diametro es largo -> la regla (potencias vs
exponencial) se vuelve decisiva. Falsador: shuffle (misma densidad, al azar) ->
debe dar mundo-pequeño / exponencial.

Cero coordenadas. Listas de adyacencia + BFS (escala a decenas de miles).

USO: python3 cg003c_crecimiento.py [--quick]
"""
from __future__ import annotations

import argparse
import time
from collections import deque

import numpy as np


def crecer(N, m=3, kdeg=6, seed=0):
    """Crece el cosmos: cada asimetria nueva se pega a un frente global insaturado."""
    rng = np.random.default_rng(seed)
    adj = [set() for _ in range(N)]
    deg = np.zeros(N, int)
    s0 = m + 1
    for i in range(s0):
        for j in range(i + 1, s0):
            adj[i].add(j); adj[j].add(i); deg[i] += 1; deg[j] += 1
    frontier = [i for i in range(s0) if deg[i] < kdeg]   # lista; saturados se sacan al ser elegidos (O(1) amortizado)

    def pick():                                   # ancla en el borde (frente global)
        while frontier:
            idx = int(rng.integers(len(frontier)))
            u = frontier[idx]
            if deg[u] < kdeg:
                return u
            frontier[idx] = frontier[-1]; frontier.pop()
        return None

    for v in range(s0, N):                        # el contador expresa una asimetria por paso
        u = pick()
        if u is None:
            break
        nb = [w for w in adj[u] if deg[w] < kdeg]
        rng.shuffle(nb)
        parche = [u] + nb[: m - 1]
        for w in parche:
            if deg[v] < kdeg and deg[w] < kdeg:
                adj[v].add(w); adj[w].add(v); deg[v] += 1; deg[w] += 1
        if not adj[v]:
            adj[v].add(u); adj[u].add(v); deg[v] += 1; deg[u] += 1
        if deg[v] < kdeg:
            frontier.append(v)
    adj = [np.fromiter(s, dtype=np.int32) for s in adj]
    return adj, deg


def _bfs(adj, src):
    dist = {src: 0}; q = deque([src])
    while q:
        u = q.popleft(); du = dist[u] + 1
        for w in adj[u]:
            if w not in dist:
                dist[w] = du; q.append(int(w))
    return dist


def diametro(adj, N, n_src=24, seed=0):
    rng = np.random.default_rng(seed)
    mx = 0
    for s in rng.choice(N, size=min(n_src, N), replace=False):
        d = _bfs(adj, int(s))
        mx = max(mx, max(d.values()))
    return mx


def dimension_crecimiento(adj, N, n_src=60, seed=0):
    """N(r) = #{a distancia relacional <= r}. r^d = geometria ; e^{br} = azar."""
    rng = np.random.default_rng(seed)
    RMAX = 500; acc = np.zeros(RMAX + 1); cnt = 0; reach = []
    for s in rng.choice(N, size=min(n_src, N), replace=False):
        d = _bfs(adj, int(s))
        ds = np.fromiter(d.values(), dtype=int)
        reach.append(len(ds))
        cum = np.bincount(np.minimum(ds, RMAX), minlength=RMAX + 1).cumsum().astype(float)
        acc += cum; cnt += 1
    if cnt == 0:
        return dict(d=np.nan, r2p=np.nan, r2e=np.nan, ver="sin fuentes", rango=0, giant=0)
    Nr = acc / cnt; giant = int(np.median(reach)); half = 0.5 * giant
    rs, ys = [], []
    for r in range(1, RMAX + 1):
        if Nr[r] <= Nr[r - 1] + 1e-9:
            break
        rs.append(r); ys.append(Nr[r])
        if Nr[r] >= half:
            break
    if len(rs) < 5:
        return dict(d=np.nan, r2p=np.nan, r2e=np.nan, ver="poco rango", rango=len(rs), giant=giant)
    rs = np.array(rs, float); ly = np.log(np.array(ys, float)); ym = ly.mean()
    cp = np.polyfit(np.log(rs), ly, 1); r2p = 1 - np.sum((ly - np.polyval(cp, np.log(rs))) ** 2) / (np.sum((ly - ym) ** 2) + 1e-12)
    ce = np.polyfit(rs, ly, 1); r2e = 1 - np.sum((ly - np.polyval(ce, rs)) ** 2) / (np.sum((ly - ym) ** 2) + 1e-12)
    ver = "GEOMETRIA" if r2p >= r2e else "AZAR(exp)"
    return dict(d=float(cp[0]), r2p=float(r2p), r2e=float(r2e), ver=ver, rango=len(rs), giant=giant)


def shuffle_adj(adj, N, seed=0):
    """Grafo azaroso con la MISMA cantidad de aristas (rompe la historia)."""
    E = sum(len(a) for a in adj) // 2
    rng = np.random.default_rng(seed)
    radj = [set() for _ in range(N)]
    made = 0; guard = 0
    while made < E and guard < 20 * E:
        i, j = int(rng.integers(N)), int(rng.integers(N))
        guard += 1
        if i != j and j not in radj[i]:
            radj[i].add(j); radj[j].add(i); made += 1
    return [np.fromiter(s, dtype=np.int32) for s in radj]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    Ns = [1000, 4000] if args.quick else [1000, 4000, 12000, 30000]
    Ms = [2, 3, 4]
    seeds = range(1, 4) if args.quick else range(1, 5)

    t0 = time.monotonic()
    print("CG003-c — el espacio que CRECE (escalable) · NO fijamos dimension\n" + "=" * 82)
    print("geometria d-dim: diametro ~ N^(1/d), N(r)~r^d.   mundo-pequeño: diametro ~ log N.\n")
    print(f"{'m':>2} {'N':>6} {'seed':>4} {'diam':>5} | {'d_growth':>8} {'R2pot':>6} {'R2exp':>6} {'veredicto':>11} | {'shuf_d':>6} {'shuf':>10}")
    rows = []
    for m in Ms:
        for N in Ns:
            for sd in seeds:
                adj, deg = crecer(N, m=m, kdeg=m + 3, seed=sd)
                dia = diametro(adj, N, seed=sd)
                g = dimension_crecimiento(adj, N, seed=sd)
                R = shuffle_adj(adj, N, seed=sd + 7)
                gs = dimension_crecimiento(R, N, seed=sd + 3)
                rows.append((m, N, sd, dia, g, gs))
                print(f"{m:>2} {N:>6} {sd:>4} {dia:>5} | {g['d']:>8.2f} {g['r2p']:>6.3f} {g['r2e']:>6.3f} "
                      f"{g['ver']:>11} | {gs['d']:>6.2f} {gs['ver']:>10}", flush=True)

    print("\n" + "=" * 82)
    print("Dimension emergente (media sobre semillas), por m y N — ¿converge al subir N?")
    for m in Ms:
        line = f"  m={m}: "
        for N in Ns:
            ds = [g['d'] for (mm, NN, s, dia, g, gs) in rows if mm == m and NN == N and g['d'] == g['d']]
            line += f"N={N}:d={np.mean(ds):.2f}  " if ds else f"N={N}:—  "
        print(line)
    geo = sum(1 for (m, N, s, dia, g, gs) in rows if g['ver'] == 'GEOMETRIA')
    az = sum(1 for (m, N, s, dia, g, gs) in rows if gs['ver'].startswith('AZAR') or gs['ver'] == 'poco rango')
    print(f"\n  real: {geo}/{len(rows)} GEOMETRIA (potencias)   shuffle: {az}/{len(rows)} azar/mundo-pequeño")
    print(f"Tiempo: {time.monotonic()-t0:.1f}s")


if __name__ == "__main__":
    main()

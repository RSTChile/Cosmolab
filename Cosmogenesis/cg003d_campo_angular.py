#!/usr/bin/env python3
"""
CG003-d — El campo angular: ¿emerge el ESPACIO cuando el sustrato tiene DIRECCIONES?
====================================================================================
El muro de CG003a/b/c (medido, no supuesto): de pura RELACION sale tiempo/orden
(un hilo 1D, barato: cada enlace es un "despues"), pero NO espacio/extension. A
escala (cg003c, N hasta 30.000) la dimension de crecimiento TREPA con N sin
converger (m=2: 1.89->2.24->2.53->2.77) y R2_exp>R2_pot => mundo-pequeño, no
geometria. Una dimension REAL es N-independiente (como el grano κ_Δ del arco
CG002). Que d(N) suba monotona ES la firma de que no hay dimension seleccionada.

Diagnostico (teoria + dato): el espacio pide DIRECCIONES —ejes inconmensurables
que un grafo de relaciones no tiene—. Es el MISMO muro del spin y la quiralidad.
El unico portador de angulo es el CAMPO CONTINUO. Este experimento le da al
sustrato lo que al grafo le faltaba, y NADA MAS: una orientacion local.

QUE INYECTAMOS (lo minimo, y solo esto):
  · Cada asimetria (nodo) tiene un espacio TANGENTE local de dimension Dtan:
    sus enlaces salen en DIRECCIONES (vectores unitarios en R^Dtan). Eso es el
    "angulo" del campo — el grado de libertad que el grafo puro no tenia.
QUE NO INYECTAMOS (seguimos sin dibujar la caja):
  · CERO coordenadas globales. No hay posicion x,y,z de ningun nodo. Solo hay
    direcciones LOCALES relativas (el frente de cada nodo), como una conexion
    (transporte paralelo), nunca un sistema de referencia comun.
  · NO fijamos la dimension del espacio. La MEDIMOS (N(r)~r^d) y la dejamos elegir.

LA REGLA (una linea): un nodo reparte sus enlaces en direcciones MUTUAMENTE
SEPARADAS (exclusion angular > θ_min) — porque dos direcciones casi paralelas
son la MISMA direccion (economia: no gastar un eje repetido). Esa exclusion es
lo que hace las direcciones INCONMENSURABLES: crea ejes locales casi-ortogonales
-> coordinacion finita -> metrica -> geometria. Los atajos mueren solos (un enlace
lejano tendria que caer en una direccion ya ocupada).

HIPOTESIS FALSABLE (la apuesta): si el angulo era lo que faltaba, la dimension
emergente d CONVERGE a ~Dtan y se vuelve N-INDEPENDIENTE (a diferencia de cg003c).
  · d ~ Dtan estable, N-indep, shuffle lo destruye  -> EL ESPACIO EMERGE DEL CAMPO.
  · d sigue trepando con N                           -> el angulo tampoco basta;
    faltaria la metrica emergente, no solo la direccion (negativo honesto).

Cero coordenadas globales · listas de adyacencia + BFS (escala a decenas de miles).
USO: python3 cg003d_campo_angular.py [--quick]
"""
from __future__ import annotations

import argparse
import time
from collections import deque

import numpy as np


# ---------------------------------------------------------------------------
# Crecimiento del campo: cada nodo nuevo se pega en una DIRECCION libre del frente
# ---------------------------------------------------------------------------
def _rand_unit(rng, Dtan):
    v = rng.normal(0.0, 1.0, Dtan)
    return v / (np.linalg.norm(v) + 1e-12)


def crecer_campo(N, Dtan=2, kdeg=8, cos_min=0.5, m_cross=2, seed=0):
    """
    Crece el cosmos como un CAMPO con direcciones locales.
      · adj[i]      : vecinos de i
      · dirs[i]     : dict {vecino -> vector unitario de la direccion i->vecino en R^Dtan}
      · exclusion angular: i no acepta un enlace nuevo cuya direccion tenga
        cos(angulo) > cos_min con alguna direccion ya usada por i (ejes repetidos
        = mismo eje; se prohiben). cos_min alto => ejes muy separados => localidad rigida.
      · m_cross: ademas del enlace al padre u, v intenta cerrar el vecindario
        enlazando a vecinos de u que caen en direcciones ANGULARMENTE VECINAS
        (esto teje la 'tela' local -> dimension > 1; sin esto seria un arbol=1D).
    """
    rng = np.random.default_rng(seed)
    adj = [set() for _ in range(N)]
    dirs = [dict() for _ in range(N)]

    def libre(i, d, exclude=None):
        """¿cabe una direccion d en el nodo i sin repetir eje?"""
        if len(adj[i]) >= kdeg:
            return False
        for w, dw in dirs[i].items():
            if w == exclude:
                continue
            if float(np.dot(d, dw)) > cos_min:   # demasiado paralela a un eje ya usado
                return False
        return True

    def enlazar(i, j, d):
        adj[i].add(j); adj[j].add(i)
        dirs[i][j] = d; dirs[j][i] = -d          # la vuelta es la direccion opuesta (transporte)

    # semilla: un pequeño simplex con direcciones bien separadas
    s0 = Dtan + 1
    seed_dirs = [_rand_unit(rng, Dtan) for _ in range(s0)]
    for i in range(s0):
        for j in range(i + 1, s0):
            d = seed_dirs[j] - seed_dirs[i]
            d = d / (np.linalg.norm(d) + 1e-12)
            enlazar(i, j, d)

    frontier = [i for i in range(s0) if len(adj[i]) < kdeg]

    def pick():
        while frontier:
            idx = int(rng.integers(len(frontier)))
            u = frontier[idx]
            if len(adj[u]) < kdeg:
                return u
            frontier[idx] = frontier[-1]; frontier.pop()
        return None

    for v in range(s0, N):
        u = pick()
        if u is None:
            break
        # 1) direccion del enlace padre: una direccion NUEVA, separada de las de u
        d_new = None
        for _ in range(16):
            cand = _rand_unit(rng, Dtan)
            if libre(u, cand):
                d_new = cand; break
        if d_new is None:
            # u esta lleno de ejes; sacalo del frente y sigue
            continue
        enlazar(u, v, d_new)

        # 2) cerrar el vecindario: v se pega a vecinos de u cuya direccion (desde u)
        #    es ANGULARMENTE VECINA a d_new (comparten sector) -> tejido local
        cross = []
        for w, dw in dirs[u].items():
            if w == v:
                continue
            c = float(np.dot(dw, d_new))
            if c > 0.0:                          # mismo hemisferio angular = vecino cercano
                cross.append((c, w))
        cross.sort(reverse=True)
        for _, w in cross[:m_cross]:
            # direccion v->w consistente: aprox (dir u->w) - (dir u->v), renormalizada
            dv = dirs[u][w] - d_new
            nv = np.linalg.norm(dv)
            if nv < 1e-9:
                continue
            dv = dv / nv
            if libre(v, dv) and libre(w, -dv, exclude=None):
                enlazar(v, w, dv)

        if len(adj[v]) < kdeg:
            frontier.append(v)
        if len(adj[u]) < kdeg:
            frontier.append(u)

    adj = [np.fromiter(s, dtype=np.int32) for s in adj]
    return adj


# ---------------------------------------------------------------------------
# Medicion (coordinate-free): dimension de crecimiento + diametro
# ---------------------------------------------------------------------------
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
        if d:
            mx = max(mx, max(d.values()))
    return mx


def dimension_crecimiento(adj, N, n_src=60, seed=0):
    """N(r)=#{a distancia relacional<=r}. r^d=geometria ; e^{br}=azar(mundo-pequeño)."""
    rng = np.random.default_rng(seed)
    RMAX = 800; acc = np.zeros(RMAX + 1); cnt = 0; reach = []
    for s in rng.choice(N, size=min(n_src, N), replace=False):
        d = _bfs(adj, int(s))
        if len(d) < 10:
            continue
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
    cp = np.polyfit(np.log(rs), ly, 1)
    r2p = 1 - np.sum((ly - np.polyval(cp, np.log(rs))) ** 2) / (np.sum((ly - ym) ** 2) + 1e-12)
    ce = np.polyfit(rs, ly, 1)
    r2e = 1 - np.sum((ly - np.polyval(ce, rs)) ** 2) / (np.sum((ly - ym) ** 2) + 1e-12)
    ver = "GEOMETRIA" if r2p >= r2e else "AZAR(exp)"
    return dict(d=float(cp[0]), r2p=float(r2p), r2e=float(r2e), ver=ver, rango=len(rs), giant=giant)


def shuffle_adj(adj, N, seed=0):
    """Grafo azaroso con la MISMA cantidad de aristas (rompe la historia direccional)."""
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
    Dtans = [2, 3]
    Ns = [1000, 4000] if args.quick else [1000, 4000, 12000, 30000]
    seeds = range(1, 4) if args.quick else range(1, 5)

    t0 = time.monotonic()
    print("CG003-d — el CAMPO ANGULAR: el sustrato tiene DIRECCIONES (lo que al grafo le faltaba)")
    print("=" * 92)
    print("apuesta: d converge ~Dtan y N-INDEPENDIENTE (a diferencia de cg003c que trepaba).")
    print("geometria d-dim: N(r)~r^d, diametro~N^(1/d).  mundo-pequeño: diametro~log N.\n")
    print(f"{'Dt':>2} {'N':>6} {'sd':>3} {'diam':>5} | {'d_grow':>7} {'R2pot':>6} {'R2exp':>6} {'veredicto':>10} | {'shuf_d':>6} {'shuf':>10}")
    rows = []
    for Dt in Dtans:
        for N in Ns:
            for sd in seeds:
                adj = crecer_campo(N, Dtan=Dt, kdeg=2 * Dt + 4, seed=sd)
                dia = diametro(adj, N, seed=sd)
                g = dimension_crecimiento(adj, N, seed=sd)
                R = shuffle_adj(adj, N, seed=sd + 7)
                gs = dimension_crecimiento(R, N, seed=sd + 3)
                rows.append((Dt, N, sd, dia, g, gs))
                print(f"{Dt:>2} {N:>6} {sd:>3} {dia:>5} | {g['d']:>7.2f} {g['r2p']:>6.3f} {g['r2e']:>6.3f} "
                      f"{g['ver']:>10} | {gs['d']:>6.2f} {gs['ver']:>10}", flush=True)

    print("\n" + "=" * 92)
    print("Dimension emergente (media sobre semillas), por Dtan y N — ¿CONVERGE al subir N?")
    converge_todo = True
    for Dt in Dtans:
        vals = []
        line = f"  Dtan={Dt}: "
        for N in Ns:
            ds = [g['d'] for (D, NN, s, dia, g, gs) in rows if D == Dt and NN == N and g['d'] == g['d']]
            if ds:
                mu = np.mean(ds); vals.append(mu); line += f"N={N}:d={mu:.2f}  "
            else:
                line += f"N={N}:—  "
        print(line)
        if len(vals) >= 2:
            drift = abs(vals[-1] - vals[0])
            veredicto = "CONVERGE (N-indep)" if drift < 0.35 else f"TREPA (Δ={drift:.2f}) — NO converge"
            print(f"           -> {veredicto}")
            if drift >= 0.35:
                converge_todo = False
    geo = sum(1 for (D, N, s, dia, g, gs) in rows if g['ver'] == 'GEOMETRIA')
    az = sum(1 for (D, N, s, dia, g, gs) in rows if gs['ver'].startswith('AZAR'))
    print(f"\n  real: {geo}/{len(rows)} GEOMETRIA (potencias)   shuffle: {az}/{len(rows)} azar/mundo-pequeño")
    if converge_todo and geo >= 0.7 * len(rows):
        print("  => EL ESPACIO EMERGE DEL CAMPO: dimension estable, N-independiente, shuffle la destruye.")
    else:
        print("  => AUN NO: la dimension no se estabiliza / el shuffle no separa. El angulo no basto todavia.")
    print(f"\nTiempo: {time.monotonic()-t0:.1f}s")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
CG003-f — versión iPad AUTOCONTENIDA (un solo archivo, SOLO numpy)
=================================================================
Corrida de fondo pedida por Claude Science (2-jul-2026): la métrica de holonomía v1
está SANA (no-degenerada; entre 24 direcciones unas quedan más planas que otras, y
vale para Dtan=2 y 3). Falta la pregunta de fondo: al soltar a crecer un universo
entero con la regla, ¿emerge un espacio PLANO y EXTENDIDO, o sigue "todo amontonado
en un puntito"?

Este archivo funde en uno solo lo que en el Mac vivía en 3 (cg003d + diagnóstico de
Gromov + cg003f), para poder correrlo en el iPad Pro M1 (libre de ANIMA) con apps de
Python que traen numpy (a-Shell, Pythonista, Carnets, Pyto). NO necesita scipy ni nada
compilado extra: SOLO numpy.

TRES BRAZOS (idénticos salvo la regla, mismas semillas):
  · REGLA   = crecimiento con holonomía v1 activa (λ_H > 0).
  · CONTROL = exactamente lo mismo con la regla apagada (λ_H = 0).
  · AZAR    = shuffle del grafo REGLA (mismas aristas barajadas): el piso del azar.
El CONTROL es lo que convierte "salió plano" en "salió plano POR la regla".

QUÉ MIDE (primero que la medición esté sana, después la conclusión):
  · δ de Gromov (4 puntos): plano -> δ CRECE con N ; hiperbólico -> δ ACOTADA (~0).
  · diámetro vs N (pendiente log-log): plano ~N^(1/2) (pend~0.5) ; mundo-pequeño ~logN (~0).
  · dimensión de crecimiento N(r)~r^d (emergente, NO asignada) + veredicto GEOMETRIA/AZAR.
  · % componente gigante (para ver si se amontona en un puntito).
Ancla de sanidad: lattice2D (control POSITIVO de geometría) y árbol b3 (NEGATIVO, δ=0).

USO (en el iPad, terminal a-Shell o consola):
  python3 cg003f_ipad.py            # barrido por defecto (N hasta 16384, 2 semillas)
  python3 cg003f_ipad.py --quick    # rápido (N hasta 4096)
  python3 cg003f_ipad.py --full     # exhaustivo (3 semillas)
Copiá TODA la salida de consola y pegámela: yo la empaqueto para la auditoría de CS.
"""
from __future__ import annotations

import argparse
import time
from collections import deque

import numpy as np


# ===========================================================================
#  NÚCLEO DEL CAMPO ANGULAR  (de cg003d_campo_angular.py)
# ===========================================================================
def _rand_unit(rng, Dtan):
    v = rng.normal(0.0, 1.0, Dtan)
    return v / (np.linalg.norm(v) + 1e-12)


def crecer_campo(N, Dtan=2, kdeg=8, cos_min=0.5, m_cross=2, seed=0):
    """Crecimiento base (sin exergía ni holonomía): el CONTROL histórico cg003d."""
    rng = np.random.default_rng(seed)
    adj = [set() for _ in range(N)]
    dirs = [dict() for _ in range(N)]

    def libre(i, d, exclude=None):
        if len(adj[i]) >= kdeg:
            return False
        for w, dw in dirs[i].items():
            if w == exclude:
                continue
            if float(np.dot(d, dw)) > cos_min:
                return False
        return True

    def enlazar(i, j, d):
        adj[i].add(j); adj[j].add(i)
        dirs[i][j] = d; dirs[j][i] = -d

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
        d_new = None
        for _ in range(16):
            cand = _rand_unit(rng, Dtan)
            if libre(u, cand):
                d_new = cand; break
        if d_new is None:
            continue
        enlazar(u, v, d_new)
        cross = []
        for w, dw in dirs[u].items():
            if w == v:
                continue
            c = float(np.dot(dw, d_new))
            if c > 0.0:
                cross.append((c, w))
        cross.sort(reverse=True)
        for _, w in cross[:m_cross]:
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
    """N(r)=#{a distancia<=r}. r^d=geometria ; e^{br}=azar(mundo-pequeño)."""
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
    """BRAZO AZAR: mismas aristas barajadas (rompe la historia direccional)."""
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


# ===========================================================================
#  DIAGNÓSTICO DE PLANITUD: δ de Gromov  (de cg003_diagnostico_gromov.py)
# ===========================================================================
def lattice2d(N):
    """Red cuadrada L×L (2D plana genuina): control POSITIVO de la MEDICIÓN."""
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
    """Árbol b-ario (hiperbólico exacto, δ=0): control NEGATIVO de la MEDICIÓN."""
    adj = [[] for _ in range(N)]
    for v in range(1, N):
        u = (v - 1) // b
        adj[u].append(v); adj[v].append(u)
    return [np.fromiter(a, dtype=np.int32) for a in adj], N


def _bfs_dist(adj, src, N):
    dist = np.full(N, -1, dtype=np.int32); dist[src] = 0; q = deque([src])
    while q:
        u = q.popleft(); du = dist[u] + 1
        for w in adj[u]:
            if dist[w] < 0:
                dist[w] = du; q.append(int(w))
    return dist


def landmarks_dist(adj, N, K, seed=0):
    """BFS desde K faros anclados en la componente GIGANTE -> matriz KxK."""
    rng = np.random.default_rng(seed)
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
    """δ media y cuantil-95 sobre cuádruplas de faros (condición 4 puntos)."""
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
    S = np.sort(np.stack([S1, S2, S3], axis=1), axis=1)
    delta = (S[:, 2] - S[:, 1]) / 2.0
    return float(delta.mean()), float(np.quantile(delta, 0.95))


def diagnos(nombre, adj, N, K, seed=0):
    D, dmax, fg = landmarks_dist(adj, N, K, seed=seed)
    if D is None:
        return dict(nombre=nombre, N=N, diam=np.nan, dmean=np.nan, d95=np.nan, fg=fg)
    dmean, d95 = gromov_delta(D, seed=seed + 1)
    return dict(nombre=nombre, N=N, diam=dmax, dmean=dmean, d95=d95, fg=fg)


# ===========================================================================
#  HOLONOMÍA v1 + CRECIMIENTO CON EXERGÍA  (de cg003f_planitud_exergia.py)
# ===========================================================================
C_LINK = 0.04
INJECT = 0.08
D_DIFF = 0.18
GAMMA = 0.008
DIFFUSE_EVERY = 40


def _ang(a, b):
    return float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))


def cerrar_plano(duv, duw):
    dv = duw - duv
    nv = float(np.linalg.norm(dv))
    return dv / nv if nv > 1e-9 else None


def frustracion_ciclo(d_uv, d_uw, dv):
    """
    v1 — Holonomía del LAZO CERRADO u->v->w->u: defecto angular del triángulo,
    leído SÓLO de las direcciones reales (Σ ángulos interiores − π = curvatura
    encerrada). SÍ depende de dv (a diferencia del transporte por una arista, que
    en R² es trivial). Usa sólo ∠(·,·)=arccos(dot) -> vale en cualquier Dtan.
    """
    ang_u = _ang(d_uv, d_uw)
    ang_v = _ang(-d_uv, dv)
    ang_w = _ang(d_uw, dv)
    return abs((ang_u + ang_v + ang_w) - np.pi)


def _check_no_degenerada(seed=0, Dtan=2, n_dir=24):
    """Control (Claude Science): la métrica de selección debe VARIAR, no empatar."""
    rng = np.random.default_rng(seed)
    d_uv = _rand_unit(rng, Dtan)
    d_uw = _rand_unit(rng, Dtan)
    vals = [frustracion_ciclo(d_uv, d_uw, _rand_unit(rng, Dtan)) for _ in range(n_dir)]
    sd = float(np.std(vals))
    return sd, sd > 1e-6


def _pagar(e, nodes, cost):
    if cost <= 0:
        return True
    tot = sum(float(e[i]) for i in nodes)
    if tot < cost - 1e-12:
        return False
    for i in nodes:
        share = cost * float(e[i]) / (tot + 1e-12)
        e[i] = max(0.0, float(e[i]) - share)
    return True


def _difundir(e, adj, n_act):
    flow = np.zeros(n_act, dtype=np.float64)
    for u in range(n_act):
        eu = float(e[u])
        for w in adj[u]:
            if w <= u or w >= n_act:
                continue
            ew = float(e[w])
            f = D_DIFF * (eu - ew)
            flow[u] -= f
            flow[w] += f
    e[:n_act] = np.clip(e[:n_act] + flow, 0.0, None)
    e[:n_act] = e[:n_act] * (1.0 - GAMMA)


def _frac_gigante(adj, N):
    best = 0
    for s in range(min(N, 32)):
        dist = {s: 0}
        q = deque([s])
        while q:
            u = q.popleft()
            for w in adj[u]:
                if w not in dist:
                    dist[w] = dist[u] + 1
                    q.append(int(w))
        best = max(best, len(dist))
    return best / N


def crecer_campo_exergia(N, Dtan=2, kdeg=8, cos_min=0.5, m_cross=2, lambda_H=1.0, seed=0):
    """cg003d + costo de holonomía v1 pagado en exergía + difusión (2ª ley).
    lambda_H=0 -> BRAZO CONTROL (sin filtro).  lambda_H>0 -> BRAZO REGLA."""
    rng = np.random.default_rng(seed)
    adj = [set() for _ in range(N)]
    dirs = [dict() for _ in range(N)]
    e = np.zeros(N, dtype=np.float64)
    phi = np.zeros(N, dtype=np.float64)

    def libre(i, d, exclude=None):
        if len(adj[i]) >= kdeg:
            return False
        for w, dw in dirs[i].items():
            if w == exclude:
                continue
            if float(np.dot(d, dw)) > cos_min:
                return False
        return True

    def _psi2(d):
        return float(np.arctan2(d[1], d[0])) if Dtan == 2 else 0.0

    def enlazar(i, j, d, cost_nodes, cost, set_phi_j=None):
        if not _pagar(e, cost_nodes, cost):
            return False
        adj[i].add(j); adj[j].add(i)
        dirs[i][j] = d; dirs[j][i] = -d
        if set_phi_j is not None:
            phi[j] = set_phi_j % (2 * np.pi)
        return True

    s0 = Dtan + 1
    base = _rand_unit(rng, Dtan)
    perp = _rand_unit(rng, Dtan)
    perp = perp - np.dot(perp, base) * base
    perp = perp / (np.linalg.norm(perp) + 1e-12)
    seed_dirs = []
    for k in range(s0):
        th = 2.0 * np.pi * k / s0
        seed_dirs.append(np.cos(th) * base + np.sin(th) * perp)
    for i in range(s0):
        e[i] = INJECT
        for j in range(i + 1, s0):
            d = seed_dirs[j] - seed_dirs[i]
            d = d / (np.linalg.norm(d) + 1e-12)
            enlazar(i, j, d, [i, j], C_LINK, set_phi_j=(phi[i] + _psi2(d)) % (2 * np.pi))

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
        e[v] = INJECT
        d_new = None
        for _ in range(16):
            cand = _rand_unit(rng, Dtan)
            if libre(u, cand):
                d_new = cand; break
        if d_new is None:
            continue
        if not enlazar(u, v, d_new, [u, v], C_LINK,
                       set_phi_j=(phi[u] + _psi2(d_new)) % (2 * np.pi)):
            continue

        cross = []
        for w, dw in dirs[u].items():
            if w == v:
                continue
            c = float(np.dot(dw, d_new))
            if c > 0.0:
                cross.append((c, w))
        cross.sort(reverse=True)

        added = 0
        for _, w in cross:
            if added >= m_cross:
                break
            if lambda_H <= 0:
                dv = cerrar_plano(d_new, dirs[u][w])
                if dv is None or not libre(v, dv) or not libre(w, -dv):
                    continue
                if enlazar(v, w, dv, [u, v, w], C_LINK):
                    added += 1
                continue
            d_uw = dirs[u][w]
            cand = []
            dv0 = cerrar_plano(d_new, d_uw)
            if dv0 is not None and libre(v, dv0) and libre(w, -dv0):
                cand.append((frustracion_ciclo(d_new, d_uw, dv0), dv0))
            for _ in range(24):
                dtry = _rand_unit(rng, Dtan)
                if not libre(v, dtry) or not libre(w, -dtry):
                    continue
                cand.append((frustracion_ciclo(d_new, d_uw, dtry), dtry))
            if not cand:
                continue
            cand.sort(key=lambda x: x[0])
            H, dv = cand[0]
            cost = C_LINK + lambda_H * (H ** 2)
            if sum(float(e[i]) for i in (u, v, w)) < cost:
                continue
            if enlazar(v, w, dv, [u, v, w], cost):
                added += 1

        if len(adj[v]) < kdeg:
            frontier.append(v)
        if len(adj[u]) < kdeg:
            frontier.append(u)
        if v % DIFFUSE_EVERY == 0:
            _difundir(e, adj, v + 1)

    adj_out = [np.fromiter(s, dtype=np.int32) for s in adj]
    return adj_out, float(e.sum()), _frac_gigante(adj, N)


def _pend_diam(Ns, dias):
    if len(Ns) < 2:
        return np.nan
    xs = np.log(np.array(Ns, float))
    ys = np.log(np.maximum(np.array(dias, float), 1.0))
    return float(np.polyfit(xs, ys, 1)[0])


# ===========================================================================
#  MAIN — barrido de los 3 brazos
# ===========================================================================
def _medir(adj, N, K, sd):
    """diámetro + dimensión emergente + δ de Gromov, en una pasada."""
    dia = diametro(adj, N, seed=sd)
    g = dimension_crecimiento(adj, N, seed=sd)
    r = diagnos("x", adj, N, K, seed=sd + 11)
    return dia, g, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()

    if args.quick:
        Ns, seeds, K = [1024, 4096], [1, 2], 80
    elif args.full:
        Ns, seeds, K = [1024, 4096, 16384], [1, 2, 3], 120
    else:
        Ns, seeds, K = [1024, 4096, 16384], [1, 2], 120
    Dtans = [3, 2]           # Dtan=3 primero (donde la dimensión tiene aire para emerger)
    LAM = 2.0                # valor del BRAZO REGLA (λ_H>0)

    t0 = time.monotonic()
    print("CG003-f (iPad) — ¿emerge espacio PLANO y EXTENDIDO?  ·  3 brazos")
    print("=" * 100)

    # --- pre-vuelo: la métrica de selección no debe ser un empate ---
    print("Pre-vuelo · control de no-degeneración de la métrica (holonomía de ciclo):")
    for Dt in (2, 3):
        sd, ok = _check_no_degenerada(seed=0, Dtan=Dt)
        print(f"  Dtan={Dt}: std(24 dir)={sd:.3e}  ->  {'OK (orienta)' if ok else 'DEGENERADA'}")
        assert ok, f"MÉTRICA DEGENERADA en Dtan={Dt}"
    print()

    # --- paso 1 de CS: sanidad de la MEDICIÓN (anclas conocidas) ---
    print("Sanidad de la medición (δ debe: lattice2D CRECER con N ; árbol ~0):")
    print(f"  {'ancla':>12} {'N':>6} {'%gig':>5} {'diam':>5} {'δ_med':>7} {'δ_q95':>7} {'d_grow':>6} {'ver':>10}")
    for Nanc in (Ns[0], Ns[-1]):
        for nombre, mk in (("lattice2D", lambda n: lattice2d(n)), ("arbol_b3", lambda n: arbol(n, 3))):
            adj, Nr = mk(Nanc)
            dia, g, r = _medir(adj, Nr, K, 7)
            print(f"  {nombre:>12} {Nr:>6} {r['fg']*100:>4.0f} {dia:>5} {r['dmean']:>7.2f} "
                  f"{r['d95']:>7.2f} {g['d']:>6.2f} {g['ver']:>10}", flush=True)
    print()

    # --- barrido de los 3 brazos ---
    print(f"Barrido (REGLA λ_H={LAM} · CONTROL λ_H=0 · AZAR=shuffle de REGLA):")
    hdr = (f"{'brazo':>8} {'Dt':>2} {'N':>6} {'sd':>2} {'%gig':>5} {'diam':>5} "
           f"{'δ_med':>7} {'δ_q95':>7} {'d_grow':>6} {'ver':>10}")
    print(hdr); print("-" * len(hdr))

    # acumuladores para el resumen: data[(brazo,Dt,N)] = listas
    data = {}
    def push(br, Dt, N, dia, g, r):
        k = (br, Dt, N)
        data.setdefault(k, dict(diam=[], dmean=[], dgrow=[], fg=[], ver=[]))
        data[k]["diam"].append(dia); data[k]["dmean"].append(r["dmean"])
        data[k]["dgrow"].append(g["d"]); data[k]["fg"].append(r["fg"]); data[k]["ver"].append(g["ver"])

    for Dt in Dtans:
        for N in Ns:
            for sd in seeds:
                # REGLA
                adjR, exR, fgR = crecer_campo_exergia(N, Dtan=Dt, kdeg=2 * Dt + 4, lambda_H=LAM, seed=sd)
                diaR, gR, rR = _medir(adjR, N, K, sd)
                push("REGLA", Dt, N, diaR, gR, rR)
                print(f"{'REGLA':>8} {Dt:>2} {N:>6} {sd:>2} {rR['fg']*100:>4.0f} {diaR:>5} "
                      f"{rR['dmean']:>7.2f} {rR['d95']:>7.2f} {gR['d']:>6.2f} {gR['ver']:>10}", flush=True)
                # CONTROL
                adjC, exC, fgC = crecer_campo_exergia(N, Dtan=Dt, kdeg=2 * Dt + 4, lambda_H=0.0, seed=sd)
                diaC, gC, rC = _medir(adjC, N, K, sd)
                push("CONTROL", Dt, N, diaC, gC, rC)
                print(f"{'CONTROL':>8} {Dt:>2} {N:>6} {sd:>2} {rC['fg']*100:>4.0f} {diaC:>5} "
                      f"{rC['dmean']:>7.2f} {rC['d95']:>7.2f} {gC['d']:>6.2f} {gC['ver']:>10}", flush=True)
                # AZAR (shuffle de REGLA)
                adjZ = shuffle_adj(adjR, N, seed=sd + 7)
                diaZ, gZ, rZ = _medir(adjZ, N, K, sd)
                push("AZAR", Dt, N, diaZ, gZ, rZ)
                print(f"{'AZAR':>8} {Dt:>2} {N:>6} {sd:>2} {rZ['fg']*100:>4.0f} {diaZ:>5} "
                      f"{rZ['dmean']:>7.2f} {rZ['d95']:>7.2f} {gZ['d']:>6.2f} {gZ['ver']:>10}", flush=True)
            print()

    # --- resumen por brazo y Dtan ---
    print("=" * 100)
    print("RESUMEN (medias sobre semillas) — pendiente diam~N, δ al mayor N, dimensión por N:")
    def m(xs):
        xs = [x for x in xs if x == x]
        return float(np.mean(xs)) if xs else float("nan")
    for Dt in Dtans:
        print(f"\n  ── Dtan={Dt} ──")
        for br in ("REGLA", "CONTROL", "AZAR"):
            dias = [m(data[(br, Dt, N)]["diam"]) for N in Ns if (br, Dt, N) in data]
            pend = _pend_diam(Ns[:len(dias)], dias) if len(dias) >= 2 else float("nan")
            dvals = [m(data[(br, Dt, N)]["dmean"]) for N in Ns if (br, Dt, N) in data]
            dtrend = "CRECE(plano)" if len(dvals) >= 2 and dvals[-1] > dvals[0] + 0.5 else "ACOTADA(hiperb)"
            dims = "  ".join(f"N={N}:d={m(data[(br,Dt,N)]['dgrow']):.2f}" for N in Ns if (br, Dt, N) in data)
            print(f"    {br:>8}: diam-pend={pend:5.2f}  δ→{dtrend:>16}  dim[{dims}]")
    print("\nLECTURA (para CS):")
    print("  · REGLA con diam-pend→~0.5, δ CRECE y dim que CONVERGE a un valor definido, y")
    print("    DISTINTO de CONTROL  => el espacio plano emergió POR la regla (éxito afirmable).")
    print("  · REGLA sigue amontonada (pend~0, δ acotada, %gig<<100) => NO es fallo de la regla")
    print("    (ya está sana): es resultado físico del mecanismo -> activar cg003f-b (contrapeso).")
    print("  · REGLA ≈ CONTROL => la regla orienta local pero no cambia la geometría global.")
    print(f"\nTiempo total: {time.monotonic()-t0:.1f}s")


if __name__ == "__main__":
    main()

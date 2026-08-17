"""
CG003-f (CARNETS / iPad) — ¿emerge espacio PLANO y EXTENDIDO?  ·  3 brazos
=========================================================================
Corrida de fondo pedida por Claude Science. La métrica de holonomía v1 está SANA
(no-degenerada; entre 24 direcciones unas quedan más planas que otras; vale Dtan=2 y 3).
Pregunta de fondo: al soltar a crecer un universo con la regla, ¿emerge un espacio
PLANO y EXTENDIDO, o sigue "todo amontonado en un puntito"?

FORMATO CARNETS (igual que tu barrido CG001 de 1000 semillas):
  · Pega TODA esta celda en Carnets y Run All. SOLO numpy (scipy no hace falta).
  · GUARDA CADA MEDICIÓN al instante (CSV por fila). Si Carnets se suspende o se cae,
    NO se pierde lo hecho.
  · SE REANUDA SOLO: re-ejecuta la MISMA celda con el MISMO LOG y salta lo ya hecho.
  · El veredicto se RECALCULA desde el CSV cada vez.

TRES BRAZOS (idénticos salvo la regla, mismas semillas):
  · REGLA   = holonomía v1 activa (λ_H > 0).
  · CONTROL = lo mismo con la regla apagada (λ_H = 0).   <- el candado del experimento
  · AZAR    = shuffle del grafo REGLA (mismas aristas barajadas): el piso del azar.

QUÉ MIDE: δ de Gromov (plano->δ crece con N), diámetro vs N (pend~0.5=plano ; ~0=mundo
pequeño), dimensión emergente N(r)~r^d (+veredicto GEOMETRIA/AZAR), % componente gigante.
Ancla de sanidad de la MEDICIÓN: lattice2D (positivo) y árbol b3 (negativo, δ=0).

Cuando termine: Files -> On My iPad -> Carnets -> cg003f_ipad.csv (+ pégame la salida).
"""
from __future__ import annotations

import csv
import os
import time
from collections import deque

import numpy as np


# ============================ CONFIG (editar aquí) ============================
MODO = "default"          # "quick" (N<=4096, rápido) | "default" (N<=16384, 2 sem) | "full" (3 sem)
LOG  = "cg003f_ipad"      # MISMO nombre para reanudar. Cámbialo solo para empezar de cero.
LAM  = 2.0                # valor del BRAZO REGLA (λ_H>0)
# =============================================================================


# ===========================================================================
#  NÚCLEO DEL CAMPO ANGULAR  (de cg003d)
# ===========================================================================
def _rand_unit(rng, Dtan):
    v = rng.normal(0.0, 1.0, Dtan)
    return v / (np.linalg.norm(v) + 1e-12)


def crecer_campo(N, Dtan=2, kdeg=8, cos_min=0.5, m_cross=2, seed=0):
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
        return dict(d=np.nan, r2p=np.nan, r2e=np.nan, ver="sin fuentes", giant=0)
    Nr = acc / cnt; giant = int(np.median(reach)); half = 0.5 * giant
    rs, ys = [], []
    for r in range(1, RMAX + 1):
        if Nr[r] <= Nr[r - 1] + 1e-9:
            break
        rs.append(r); ys.append(Nr[r])
        if Nr[r] >= half:
            break
    if len(rs) < 5:
        return dict(d=np.nan, r2p=np.nan, r2e=np.nan, ver="poco rango", giant=giant)
    rs = np.array(rs, float); ly = np.log(np.array(ys, float)); ym = ly.mean()
    cp = np.polyfit(np.log(rs), ly, 1)
    r2p = 1 - np.sum((ly - np.polyval(cp, np.log(rs))) ** 2) / (np.sum((ly - ym) ** 2) + 1e-12)
    ce = np.polyfit(rs, ly, 1)
    r2e = 1 - np.sum((ly - np.polyval(ce, rs)) ** 2) / (np.sum((ly - ym) ** 2) + 1e-12)
    ver = "GEOMETRIA" if r2p >= r2e else "AZAR(exp)"
    return dict(d=float(cp[0]), r2p=float(r2p), r2e=float(r2e), ver=ver, giant=giant)


def shuffle_adj(adj, N, seed=0):
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
#  DIAGNÓSTICO DE PLANITUD: δ de Gromov  (de cg003_diagnostico_gromov)
# ===========================================================================
def lattice2d(N):
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
    rng = np.random.default_rng(seed)
    K = D.shape[0]
    if K < 4:
        return np.nan, np.nan
    q = rng.integers(0, K, size=(n_quad, 4))
    ok = (q[:, 0] != q[:, 1]) & (q[:, 2] != q[:, 3])
    q = q[ok]
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    S1 = D[x, y] + D[z, w]; S2 = D[x, z] + D[y, w]; S3 = D[x, w] + D[y, z]
    S = np.sort(np.stack([S1, S2, S3], axis=1), axis=1)
    delta = (S[:, 2] - S[:, 1]) / 2.0
    return float(delta.mean()), float(np.quantile(delta, 0.95))


def diagnos(adj, N, K, seed=0):
    D, dmax, fg = landmarks_dist(adj, N, K, seed=seed)
    if D is None:
        return dict(diam=np.nan, dmean=np.nan, d95=np.nan, fg=fg)
    dmean, d95 = gromov_delta(D, seed=seed + 1)
    return dict(diam=dmax, dmean=dmean, d95=d95, fg=fg)


# ===========================================================================
#  HOLONOMÍA v1 + CRECIMIENTO CON EXERGÍA  (de cg003f)
# ===========================================================================
C_LINK = 0.04; INJECT = 0.08; D_DIFF = 0.18; GAMMA = 0.008; DIFFUSE_EVERY = 40


def _ang(a, b):
    return float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))


def cerrar_plano(duv, duw):
    dv = duw - duv
    nv = float(np.linalg.norm(dv))
    return dv / nv if nv > 1e-9 else None


def frustracion_ciclo(d_uv, d_uw, dv):
    """v1: defecto angular del triángulo cerrado (Σ ángulos interiores − π). Depende de dv."""
    return abs((_ang(d_uv, d_uw) + _ang(-d_uv, dv) + _ang(d_uw, dv)) - np.pi)


def _check_no_degenerada(seed=0, Dtan=2, n_dir=24):
    rng = np.random.default_rng(seed)
    d_uv = _rand_unit(rng, Dtan); d_uw = _rand_unit(rng, Dtan)
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
        e[i] = max(0.0, float(e[i]) - cost * float(e[i]) / (tot + 1e-12))
    return True


def _difundir(e, adj, n_act):
    flow = np.zeros(n_act, dtype=np.float64)
    for u in range(n_act):
        eu = float(e[u])
        for w in adj[u]:
            if w <= u or w >= n_act:
                continue
            f = D_DIFF * (eu - float(e[w]))
            flow[u] -= f; flow[w] += f
    e[:n_act] = np.clip(e[:n_act] + flow, 0.0, None) * (1.0 - GAMMA)


def _frac_gigante(adj, N):
    best = 0
    for s in range(min(N, 32)):
        dist = {s: 0}; q = deque([s])
        while q:
            u = q.popleft()
            for w in adj[u]:
                if w not in dist:
                    dist[w] = dist[u] + 1; q.append(int(w))
        best = max(best, len(dist))
    return best / N


def crecer_campo_exergia(N, Dtan=2, kdeg=8, cos_min=0.5, m_cross=2, lambda_H=1.0, seed=0):
    """lambda_H=0 -> BRAZO CONTROL ; lambda_H>0 -> BRAZO REGLA."""
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

    return [np.fromiter(s, dtype=np.int32) for s in adj]


# ===========================================================================
#  MAIN — barrido de 3 brazos, con checkpoint + reanudación (estilo CG001)
# ===========================================================================
def _pend_diam(Ns, dias):
    Ns = [n for n, d in zip(Ns, dias) if d == d]
    dias = [d for d in dias if d == d]
    if len(Ns) < 2:
        return float("nan")
    xs = np.log(np.array(Ns, float)); ys = np.log(np.maximum(np.array(dias, float), 1.0))
    return float(np.polyfit(xs, ys, 1)[0])


def _medir(adj, N, K, sd):
    dia = diametro(adj, N, seed=sd)
    g = dimension_crecimiento(adj, N, seed=sd)
    r = diagnos(adj, N, K, seed=sd + 11)
    return dia, g, r


def main():
    if MODO == "quick":
        Ns, seeds, K = [1024, 4096], [1, 2], 80
    elif MODO == "full":
        Ns, seeds, K = [1024, 4096, 16384], [1, 2, 3], 120
    else:
        Ns, seeds, K = [1024, 4096, 16384], [1, 2], 120
    Dtans = [3, 2]                        # Dtan=3 primero (donde la dimensión tiene aire)
    csv_path = f"{LOG}.csv"
    cols = ["brazo", "Dt", "N", "seed", "fg", "diam", "dmean", "d95", "dgrow", "ver"]

    t0 = time.time()
    print("CG003-f (Carnets) — ¿emerge espacio PLANO y EXTENDIDO?  ·  3 brazos")
    print("=" * 96)

    # --- pre-vuelo: la métrica no debe ser un empate ---
    print("Pre-vuelo · no-degeneración de la métrica (holonomía de ciclo):")
    for Dt in (2, 3):
        sd, ok = _check_no_degenerada(seed=0, Dtan=Dt)
        print(f"  Dtan={Dt}: std(24 dir)={sd:.3e} -> {'OK (orienta)' if ok else 'DEGENERADA'}")
        assert ok, f"MÉTRICA DEGENERADA en Dtan={Dt}"

    # --- sanidad de la MEDICIÓN (paso 1 de CS): anclas conocidas ---
    print("\nSanidad de la medición (δ debe: lattice2D CRECER con N ; árbol ~0):")
    print(f"  {'ancla':>10} {'N':>6} {'%gig':>5} {'diam':>5} {'δ_med':>7} {'δ_q95':>7} {'d_grow':>6} {'ver':>10}")
    for Nanc in (Ns[0], Ns[-1]):
        for nombre, mk in (("lattice2D", lambda n: lattice2d(n)), ("arbol_b3", lambda n: arbol(n, 3))):
            adj, Nr = mk(Nanc)
            dia, g, r = _medir(adj, Nr, K, 7)
            print(f"  {nombre:>10} {Nr:>6} {r['fg']*100:>4.0f} {dia:>5} {r['dmean']:>7.2f} "
                  f"{r['d95']:>7.2f} {g['d']:>6.2f} {g['ver']:>10}", flush=True)

    # --- reanudación: leer lo ya hecho ---
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                done.add((row["brazo"], int(row["Dt"]), int(row["N"]), int(row["seed"])))

    unidades = [(Dt, N, sd) for Dt in Dtans for N in Ns for sd in seeds]
    total = len(unidades) * 3
    print(f"\nBarrido REGLA λ_H={LAM} · CONTROL λ_H=0 · AZAR=shuffle(REGLA)")
    print(f"unidades={len(unidades)}  filas totales={total}  ya hechas={len(done)}  faltan={total-len(done)}")
    print(f"LOG={csv_path} (reanuda con el mismo nombre)\n")
    hdr = (f"{'brazo':>8} {'Dt':>2} {'N':>6} {'sd':>2} {'%gig':>5} {'diam':>5} "
           f"{'δ_med':>7} {'δ_q95':>7} {'d_grow':>6} {'ver':>10}")
    print(hdr); print("-" * len(hdr))

    nuevo = not os.path.exists(csv_path)
    fcsv = open(csv_path, "a", newline=""); w = csv.writer(fcsv)
    if nuevo:
        w.writerow(cols); fcsv.flush()

    def emit(br, Dt, N, sd, dia, g, r):
        w.writerow([br, Dt, N, sd, r["fg"], dia, r["dmean"], r["d95"], g["d"], g["ver"]])
        fcsv.flush()                       # durabilidad: cada medición en disco
        print(f"{br:>8} {Dt:>2} {N:>6} {sd:>2} {r['fg']*100:>4.0f} {dia:>5} "
              f"{r['dmean']:>7.2f} {r['d95']:>7.2f} {g['d']:>6.2f} {g['ver']:>10}", flush=True)

    for (Dt, N, sd) in unidades:
        faltan_br = [b for b in ("REGLA", "CONTROL", "AZAR") if (b, Dt, N, sd) not in done]
        if not faltan_br:
            continue
        kdeg = 2 * Dt + 4
        adjR = crecer_campo_exergia(N, Dtan=Dt, kdeg=kdeg, lambda_H=LAM, seed=sd)
        if "REGLA" in faltan_br:
            dia, g, r = _medir(adjR, N, K, sd); emit("REGLA", Dt, N, sd, dia, g, r)
        if "CONTROL" in faltan_br:
            adjC = crecer_campo_exergia(N, Dtan=Dt, kdeg=kdeg, lambda_H=0.0, seed=sd)
            dia, g, r = _medir(adjC, N, K, sd); emit("CONTROL", Dt, N, sd, dia, g, r)
        if "AZAR" in faltan_br:
            adjZ = shuffle_adj(adjR, N, seed=sd + 7)
            dia, g, r = _medir(adjZ, N, K, sd); emit("AZAR", Dt, N, sd, dia, g, r)
    fcsv.close()

    # --- RESUMEN recalculado desde TODO el CSV ---
    filas = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            filas.append(row)
    def fnum(x):
        try:
            return float(x)
        except Exception:
            return float("nan")
    def mean_of(br, Dt, N, campo):
        xs = [fnum(r[campo]) for r in filas
              if r["brazo"] == br and int(r["Dt"]) == Dt and int(r["N"]) == N]
        xs = [x for x in xs if x == x]
        return float(np.mean(xs)) if xs else float("nan")

    print("\n" + "=" * 96)
    print("RESUMEN (medias sobre semillas) — pend diam~N, tendencia δ, dimensión por N, %gig:")
    for Dt in Dtans:
        print(f"\n  ── Dtan={Dt} ──")
        for br in ("REGLA", "CONTROL", "AZAR"):
            dias = [mean_of(br, Dt, N, "diam") for N in Ns]
            pend = _pend_diam(Ns, dias)
            dme = [mean_of(br, Dt, N, "dmean") for N in Ns]
            dme_ok = [x for x in dme if x == x]
            dtrend = "CRECE(plano)" if len(dme_ok) >= 2 and dme_ok[-1] > dme_ok[0] + 0.5 else "ACOTADA(hiperb)"
            dims = "  ".join(f"N={N}:d={mean_of(br,Dt,N,'dgrow'):.2f}" for N in Ns)
            gig = mean_of(br, Dt, Ns[-1], "fg") * 100
            print(f"    {br:>8}: diam-pend={pend:5.2f}  δ→{dtrend:>16}  %gig(N={Ns[-1]})={gig:4.0f}  [{dims}]")
    print("\nLECTURA (para CS):")
    print("  · REGLA: diam-pend→~0.5, δ CRECE, dim CONVERGE a valor definido, y DISTINTO de")
    print("    CONTROL  => espacio plano emergió POR la regla (éxito afirmable).")
    print("  · REGLA amontonada (pend~0, δ acotada, %gig<<100) => NO es fallo de la regla")
    print("    (ya sana): resultado físico del mecanismo -> activar cg003f-b (contrapeso).")
    print("  · REGLA ≈ CONTROL => la regla orienta local pero no cambia la geometría global.")
    print(f"\nTiempo: {(time.time()-t0)/60:.1f} min · CSV: {csv_path}")
    print("Files -> On My iPad -> Carnets")


main()

"""
CG004 (CARNETS / iPad) — ¿la DINÁMICA DE CRECIMIENTO (attach/frente) despliega lo plano?
========================================================================================
cg003f (holonomía) y cg003f-b (relajación/cirugía) dieron NEGATIVO: intervenir la métrica
o podar DESPUÉS no endereza nada. La evidencia converge al LOCUS aguas arriba: la regla de
crecimiento. Diagnóstico del código (crecer_full de cg003f-b):
  · pick() elige un nodo del frente UNIFORMEMENTE AL AZAR (no la superficie geométrica),
  · cada nodo nuevo cuelga de 1 arista primaria a ese u random + 2 cierres oportunistas
    con vecinos VIEJOS de u, NUNCA con nodos hermanos del mismo frente.
  => espina dorsal = árbol recursivo aleatorio (diam~logN, δ≈0) con tejido lateral débil.
  La holonomía/cirugía operan sobre esa espina ya formada -> no la deshacen.

HIPÓTESIS: lo plano necesita que el crecimiento (a) avance como FRENTE COHERENTE (cáscaras,
no inserción interior aleatoria) y (b) TEJA lateralmente los nodos del mismo frente ->
retícula (bola~r^d, coordinación ~hexagonal en 2D) en vez de árbol (~b^r).

CANDADO (la variable nueva es la DISCIPLINA DE ATTACH; todo lo demás idéntico):
  · CONTROL       = regla actual (frente random + 2 cross-links)  == cg003f, hiperbólico
  · FRENTE        = frente FIFO (crece el borde más viejo -> cáscaras)
  · TEJIDO        = enlaces laterales a nodos jóvenes del frente (cose la cáscara)
  · FRENTE+TEJIDO = ambos
  · AZAR          = shuffle(FRENTE+TEJIDO)   (null de mismo nº de aristas)

CRITERIO PRE-REGISTRADO (emergencia, NO impuesto; "φ que crece sin diferenciar no cuenta"):
Un brazo GANA solo si se SEPARA de CONTROL hacia lo plano:
  · δ_med CRECE con N (como lattice2D 2.18->8.88), no acotada (~0);
  · diam-pend sube hacia 1/Dt (0.5 en Dt=2), no ~0.08;
  · dim CONVERGE (deja de trepar con N);
  · %gig alto (no fragmenta) — si %gig<<100, NO leer diam/dim de ese brazo;
  · shuffle (AZAR) NO lo reproduce (lo destruye).
Si TODOS ≈ CONTROL (hiperbólico) => la disciplina de crecimiento tampoco es el lever:
  bajar aún más (dimensión del espacio de direcciones, cap de grado).

FORMATO CARNETS: solo numpy, flush por fila, se REANUDA con el mismo LOG. Dt=2, quick.
"""
from __future__ import annotations

import csv
import os
import time
from collections import deque

import numpy as np


# ============================ CONFIG (editar aquí) ============================
MODO = "quick"       # "quick" (Ns=[1024,4096,16384], 2 sem) | "mini" (Ns=[1024,4096], 2 sem)
LOG  = "cg004_attach"
LAM  = 2.0           # λ_H del crecimiento (mismo que cg003f/f-b; cg003f probó que es inerte)
TEJIDO_MAX = 4       # máx. enlaces laterales por nodo nuevo en los brazos con TEJIDO
# =============================================================================


def _rand_unit(rng, Dtan):
    v = rng.normal(0.0, 1.0, Dtan)
    return v / (np.linalg.norm(v) + 1e-12)


# ===========================================================================
#  MEDICIÓN coordinate-free (dimensión + diámetro)   [de cg003d, sin tocar]
# ===========================================================================
def _bfs(adj, src):
    dist = {src: 0}; q = deque([src])
    while q:
        u = q.popleft(); du = dist[u] + 1
        for w in adj[u]:
            if w not in dist:
                dist[w] = du; q.append(int(w))
    return dist


def diametro(adj, N, n_src=24, seed=0):
    rng = np.random.default_rng(seed); mx = 0
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
        return dict(d=np.nan, ver="sin fuentes")
    Nr = acc / cnt; giant = int(np.median(reach)); half = 0.5 * giant
    rs, ys = [], []
    for r in range(1, RMAX + 1):
        if Nr[r] <= Nr[r - 1] + 1e-9:
            break
        rs.append(r); ys.append(Nr[r])
        if Nr[r] >= half:
            break
    if len(rs) < 5:
        return dict(d=np.nan, ver="poco rango")
    rs = np.array(rs, float); ly = np.log(np.array(ys, float)); ym = ly.mean()
    cp = np.polyfit(np.log(rs), ly, 1)
    r2p = 1 - np.sum((ly - np.polyval(cp, np.log(rs))) ** 2) / (np.sum((ly - ym) ** 2) + 1e-12)
    ce = np.polyfit(rs, ly, 1)
    r2e = 1 - np.sum((ly - np.polyval(ce, rs)) ** 2) / (np.sum((ly - ym) ** 2) + 1e-12)
    ver = "GEOMETRIA" if r2p >= r2e else "AZAR(exp)"
    return dict(d=float(cp[0]), ver=ver)


def shuffle_adj(adj, N, seed=0):
    E = sum(len(a) for a in adj) // 2
    rng = np.random.default_rng(seed)
    radj = [set() for _ in range(N)]
    made = 0; guard = 0
    while made < E and guard < 20 * E:
        i, j = int(rng.integers(N)), int(rng.integers(N)); guard += 1
        if i != j and j not in radj[i]:
            radj[i].add(j); radj[j].add(i); made += 1
    return [np.fromiter(s, dtype=np.int32) for s in radj]


# ===========================================================================
#  δ de GROMOV  [de cg003_diagnostico_gromov, sin tocar]
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
    D = np.zeros((len(src), len(src))); dmax = 0
    for a, s in enumerate(src):
        ds = _bfs_dist(adj, int(s), N)
        D[a] = ds[src]; dmax = max(dmax, int(ds[reach].max()))
    return D, dmax, frac_giant


def gromov_delta(D, n_quad=20000, seed=0):
    rng = np.random.default_rng(seed); K = D.shape[0]
    if K < 4:
        return np.nan, np.nan
    q = rng.integers(0, K, size=(n_quad, 4))
    ok = (q[:, 0] != q[:, 1]) & (q[:, 2] != q[:, 3]); q = q[ok]
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
#  HOLONOMÍA v1 + EXERGÍA (compartido)  [de cg003f, sin tocar]
# ===========================================================================
C_LINK = 0.04; INJECT = 0.08; D_DIFF = 0.18; GAMMA = 0.008; DIFFUSE_EVERY = 40


def _ang(a, b):
    return float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))


def cerrar_plano(duv, duw):
    dv = duw - duv; nv = float(np.linalg.norm(dv))
    return dv / nv if nv > 1e-9 else None


def frustracion_ciclo(d_uv, d_uw, dv):
    """v1: defecto angular del triángulo cerrado (Σ ángulos interiores − π)."""
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


# ===========================================================================
#  CRECIMIENTO parametrizado por DISCIPLINA DE ATTACH
#    frente_fifo=False -> pick uniforme al azar (CONTROL == cg003f)
#    frente_fifo=True  -> FIFO (cáscaras)
#    tejido=True       -> enlaces laterales a cousins jóvenes del frente (retícula)
# ===========================================================================
def crecer(N, Dtan=2, kdeg=8, cos_min=0.5, m_cross=2, lambda_H=2.0,
           frente_fifo=False, tejido=False, tejido_max=4, seed=0):
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

    # ---- semilla-símplex (idéntica a cg003f) ----
    s0 = Dtan + 1
    base = _rand_unit(rng, Dtan); perp = _rand_unit(rng, Dtan)
    perp = perp - np.dot(perp, base) * base
    perp = perp / (np.linalg.norm(perp) + 1e-12)
    seed_dirs = []
    for k in range(s0):
        th = 2.0 * np.pi * k / s0
        seed_dirs.append(np.cos(th) * base + np.sin(th) * perp)
    for i in range(s0):
        e[i] = INJECT
        for j in range(i + 1, s0):
            d = seed_dirs[j] - seed_dirs[i]; d = d / (np.linalg.norm(d) + 1e-12)
            enlazar(i, j, d, [i, j], C_LINK, set_phi_j=(phi[i] + _psi2(d)) % (2 * np.pi))

    # ---- frente: FIFO (cáscaras) vs uniforme al azar ----
    if frente_fifo:
        frontier = deque(i for i in range(s0) if len(adj[i]) < kdeg)
        def pick():
            while frontier:
                u = frontier.popleft()
                if len(adj[u]) < kdeg:
                    return u
            return None
        def readd(x):
            if len(adj[x]) < kdeg:
                frontier.append(x)
    else:
        frontier = [i for i in range(s0) if len(adj[i]) < kdeg]
        def pick():
            while frontier:
                idx = int(rng.integers(len(frontier))); u = frontier[idx]
                if len(adj[u]) < kdeg:
                    return u
                frontier[idx] = frontier[-1]; frontier.pop()
            return None
        def readd(x):
            if len(adj[x]) < kdeg:
                frontier.append(x)

    for v in range(s0, N):
        u = pick()
        if u is None:
            break
        e[v] = INJECT
        # --- arista primaria (idéntica a cg003f) ---
        d_new = None
        for _ in range(16):
            cand = _rand_unit(rng, Dtan)
            if libre(u, cand):
                d_new = cand; break
        if d_new is None:
            continue                                  # u interior (sin dirección libre): sale del frente
        if not enlazar(u, v, d_new, [u, v], C_LINK, set_phi_j=(phi[u] + _psi2(d_new)) % (2 * np.pi)):
            continue
        # --- cross-links oportunistas (idéntico a cg003f) ---
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
            d_uw = dirs[u][w]; cand = []
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
            cand.sort(key=lambda x: x[0]); H, dv = cand[0]
            cost = C_LINK + lambda_H * (H ** 2)
            if sum(float(e[i]) for i in (u, v, w)) < cost:
                continue
            if enlazar(v, w, dv, [u, v, w], cost):
                added += 1

        # --- TEJIDO: cose v a 'cousins' JÓVENES del frente (triangula su vecindad) ---
        # Para cada w ya vecino de v, y cada x vecino de w con cupo (joven), cierra PLANO
        # el triángulo v-w-x. Ata nodos del mismo frente entre sí => retícula, no abanico.
        if tejido:
            added_t = 0
            for w in list(adj[v]):
                if added_t >= tejido_max:
                    break
                for x in list(adj[w]):
                    if added_t >= tejido_max:
                        break
                    if x == v or x in adj[v]:
                        continue
                    if len(adj[x]) >= kdeg:
                        continue                        # solo cousins jóvenes (frente)
                    dvx = cerrar_plano(dirs[w][v], dirs[w][x])   # dir v->x que cierra plano vía w
                    if dvx is None or not libre(v, dvx) or not libre(x, -dvx):
                        continue
                    H = frustracion_ciclo(dirs[w][v], dirs[w][x], dvx)
                    cost = C_LINK + lambda_H * (H ** 2)
                    if sum(float(e[i]) for i in (v, x)) < cost:
                        continue
                    if enlazar(v, x, dvx, [v, x], cost):
                        added_t += 1

        readd(v); readd(u)
        if v % DIFFUSE_EVERY == 0:
            _difundir(e, adj, v + 1)
    return adj, dirs


# ===========================================================================
#  MAIN — brazos de disciplina de attach, con checkpoint + reanudación
# ===========================================================================
def _pend_diam(Ns, dias):
    Ns2 = [n for n, d in zip(Ns, dias) if d == d]; dias = [d for d in dias if d == d]
    if len(Ns2) < 2:
        return float("nan")
    xs = np.log(np.array(Ns2, float)); ys = np.log(np.maximum(np.array(dias, float), 1.0))
    return float(np.polyfit(xs, ys, 1)[0])


def _finalizar(adj_sets):
    return [np.fromiter(s, dtype=np.int32) for s in adj_sets]


def _medir(adj, N, K, sd):
    dia = diametro(adj, N, seed=sd)
    g = dimension_crecimiento(adj, N, seed=sd)
    r = diagnos(adj, N, K, seed=sd + 11)
    return dia, g, r


# (brazo, frente_fifo, tejido)
BRAZOS = [
    ("CONTROL",       False, False),
    ("FRENTE",        True,  False),
    ("TEJIDO",        False, True),
    ("FRENTE+TEJIDO", True,  True),
]


def main():
    if MODO == "mini":
        Ns, seeds, K = [1024, 4096], [1, 2], 80
    else:  # quick
        Ns, seeds, K = [1024, 4096, 16384], [1, 2], 120
    Dtans = [2]
    csv_path = f"{LOG}.csv"
    cols = ["brazo", "Dt", "N", "seed", "fg", "diam", "dmean", "d95", "dgrow", "ver"]

    t0 = time.time()
    print("CG004 (Carnets) — ¿la DINÁMICA DE CRECIMIENTO (attach/frente) despliega lo plano?")
    print("=" * 100)
    print(f"brazos de attach · λ_H={LAM} · tejido_max={TEJIDO_MAX} · Dt=2 · MODO={MODO}")

    print("\nPre-vuelo · no-degeneración de la métrica:")
    for Dt in Dtans:
        sd, ok = _check_no_degenerada(seed=0, Dtan=Dt)
        print(f"  Dtan={Dt}: std(24 dir)={sd:.3e} -> {'OK' if ok else 'DEGENERADA'}")
        assert ok

    print("\nSanidad de la medición (lattice2D δ CRECE ; árbol δ=0):")
    print(f"  {'ancla':>10} {'N':>6} {'%gig':>5} {'diam':>5} {'δ_med':>7} {'d_grow':>6} {'ver':>10}")
    for Nanc in (Ns[0], Ns[-1]):
        for nombre, mk in (("lattice2D", lambda n: lattice2d(n)), ("arbol_b3", lambda n: arbol(n, 3))):
            adj, Nr = mk(Nanc)
            dia, g, r = _medir(adj, Nr, K, 7)
            print(f"  {nombre:>10} {Nr:>6} {r['fg']*100:>4.0f} {dia:>5} {r['dmean']:>7.2f} "
                  f"{g['d']:>6.2f} {g['ver']:>10}", flush=True)

    done = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                done.add((row["brazo"], int(row["Dt"]), int(row["N"]), int(row["seed"])))

    todos = [b[0] for b in BRAZOS] + ["AZAR"]
    unidades = [(Dt, N, sd) for Dt in Dtans for N in Ns for sd in seeds]
    total = len(unidades) * len(todos)
    print(f"\nBrazos: {' · '.join(todos)}")
    print(f"filas totales={total}  ya hechas={len(done)}  faltan={total-len(done)}   LOG={csv_path}\n")
    hdr = (f"{'brazo':>14} {'Dt':>2} {'N':>6} {'sd':>2} {'%gig':>5} {'diam':>5} "
           f"{'δ_med':>7} {'δ_q95':>7} {'d_grow':>6} {'ver':>10}")
    print(hdr); print("-" * len(hdr))

    nuevo = not os.path.exists(csv_path)
    fcsv = open(csv_path, "a", newline=""); w = csv.writer(fcsv)
    if nuevo:
        w.writerow(cols); fcsv.flush()

    def emit(br, Dt, N, sd, dia, g, r):
        w.writerow([br, Dt, N, sd, r["fg"], dia, r["dmean"], r["d95"], g["d"], g["ver"]])
        fcsv.flush()
        print(f"{br:>14} {Dt:>2} {N:>6} {sd:>2} {r['fg']*100:>4.0f} {dia:>5} "
              f"{r['dmean']:>7.2f} {r['d95']:>7.2f} {g['d']:>6.2f} {g['ver']:>10}", flush=True)

    for (Dt, N, sd) in unidades:
        faltan = [b for b in todos if (b, Dt, N, sd) not in done]
        if not faltan:
            continue
        kdeg = 2 * Dt + 4
        adjFT = None                                    # se guarda para el shuffle (AZAR)
        for (br, fifo, tej) in BRAZOS:
            need = br in faltan
            need_for_azar = (br == "FRENTE+TEJIDO") and ("AZAR" in faltan)
            if not (need or need_for_azar):
                continue
            adj_sets, _dirs = crecer(N, Dtan=Dt, kdeg=kdeg, lambda_H=LAM,
                                     frente_fifo=fifo, tejido=tej, tejido_max=TEJIDO_MAX, seed=sd)
            adjF = _finalizar(adj_sets)
            if br == "FRENTE+TEJIDO":
                adjFT = adjF
            if need:
                dia, g, r = _medir(adjF, N, K, sd); emit(br, Dt, N, sd, dia, g, r)
        if "AZAR" in faltan and adjFT is not None:
            adjZ = shuffle_adj(adjFT, N, seed=sd + 7)
            dia, g, r = _medir(adjZ, N, K, sd); emit("AZAR", Dt, N, sd, dia, g, r)
    fcsv.close()

    # --- RESUMEN desde el CSV ---
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
        xs = [fnum(r[campo]) for r in filas if r["brazo"] == br and int(r["Dt"]) == Dt and int(r["N"]) == N]
        xs = [x for x in xs if x == x]
        return float(np.mean(xs)) if xs else float("nan")

    print("\n" + "=" * 100)
    print("RESUMEN — ¿algún brazo se SEPARA de CONTROL hacia lo plano? (objetivo diam-pend→0.5)")
    for Dt in Dtans:
        print(f"\n  ── Dtan={Dt} ──")
        for br in todos:
            dias = [mean_of(br, Dt, N, "diam") for N in Ns]
            pend = _pend_diam(Ns, dias)
            dme = [mean_of(br, Dt, N, "dmean") for N in Ns]; dme = [x for x in dme if x == x]
            dtrend = "CRECE(plano)" if len(dme) >= 2 and dme[-1] > dme[0] + 0.5 else "ACOTADA(hiperb)"
            dims = "  ".join(f"N={N}:d={mean_of(br,Dt,N,'dgrow'):.2f}" for N in Ns)
            gig = mean_of(br, Dt, Ns[-1], "fg") * 100
            print(f"    {br:>14}: diam-pend={pend:5.2f}  δ→{dtrend:>16}  %gig(N={Ns[-1]})={gig:4.0f}  [{dims}]")
    print("\nLECTURA (pre-registrada):")
    print("  · GANA el brazo con diam-pend→~0.5, δ CRECE, dim CONVERGE, %gig alto, DISTINTO de")
    print("    CONTROL, y que AZAR (shuffle) NO reproduce => la disciplina de crecimiento despliega")
    print("    lo plano (éxito afirmable). El lever es el frente/attach.")
    print("  · Si TODOS ≈ CONTROL (hiperbólicos) => la disciplina de crecimiento tampoco basta:")
    print("    bajar aún más (dimensión del espacio de direcciones, cap de grado).")
    print("  · %gig(brazo) << 100 => ese brazo fragmentó: NO leer su diam/dim.")
    print(f"\nTiempo: {(time.time()-t0)/60:.1f} min · CSV: {csv_path}")


main()

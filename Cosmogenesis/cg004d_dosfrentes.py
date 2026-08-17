"""
CG004-d — TEST DE DOS FRENTES (mínimo, falsable) — pedido de CS
================================================================
Mecanismo global elegido (CS): RECONVERGENCIA DE FRENTES POR PEGADO. Cuando dos frentes se
encuentran con MARCOS CONSISTENTES (holonomía≈0 en el lazo, pero distancia de grafo grande), se
PEGAN — identifican/conectan nodos — en vez de pasarse de largo. Solo PEGAR cambia las distancias
(δ y dim); alinear rotaciones sobre grafo fijo es NO-OP. La consistencia de marcos debe volverse
métrica.

Antes de la maquinaria completa: TEST DE DOS FRENTES barato y falsable.
  · Dos frentes que avanzan uno hacia el otro (aquí: dos copias del mismo parche, enfrentadas).
    Marco de cada nodo = φ (ángulo transportado en paralelo desde la semilla; ya lo produce el
    crecimiento). El frente = nodos de la frontera (no saturados).
  · Al tocarse, dos brazos con EL MISMO conjunto de pegados candidatos:
      REGLA   = pegar frontera-A con frontera-B donde los MARCOS COINCIDEN (min |Δφ|).
      CONTROL = pegar la MISMA cantidad, al AZAR (sin criterio de marco).
  · Métrica que decide: ¿|S(r)| RECONVERGE (crecimiento sub-exponencial / polinómico) y la
    dimensión converge bajo REGLA y NO bajo CONTROL?

TRES CUERDAS (CS):
  1. No hornear: el criterio es SOLO consistencia de marco (φ). La dimensión debe EMERGER
     (convergencia de dim + diam-pend→1/Dt), no imponerse.
  2. Sobre-pegado = colapso (esfera/mundo-pequeño). Guard: %gig sano y que el diámetro NO colapse
     a ~log (un "plano" que colapsó no es victoria). Se barre la fracción pegada.
  3. Alcance: Dt∈{2,3}, ≥8 semillas desde el inicio.

Reusa el arnés de medición de cg004_attach.py.
"""
from __future__ import annotations

import csv
import importlib.util
import os
import time

import numpy as np
from collections import deque

_HERE = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(_HERE, "cg004_attach.py")).read().replace("\nmain()\n", "\n")
_M = {}
exec(compile(_src, "cg004_attach.py", "exec"), _M)
_rand_unit = _M["_rand_unit"]; cerrar_plano = _M["cerrar_plano"]; _pagar = _M["_pagar"]; _difundir = _M["_difundir"]
C_LINK = _M["C_LINK"]; INJECT = _M["INJECT"]; DIFFUSE_EVERY = _M["DIFFUSE_EVERY"]
diametro = _M["diametro"]; dimension_crecimiento = _M["dimension_crecimiento"]; diagnos = _M["diagnos"]
lattice2d = _M["lattice2d"]; arbol = _M["arbol"]; _check_no_degenerada = _M["_check_no_degenerada"]


# ============================ CONFIG ============================
LOG   = "cg004d_dosfrentes"
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8]
NHALF = [512, 2048, 8192]          # nodos por frente -> combinado ~1k, 4k, 16k
DTANS = [2, 3]
K     = 120
COSM  = 0.6                        # gate relajado (triángulos posibles), λ_H=0
MCROSS = 8
GLUE_FRAC = 1.0                    # fracción de la frontera que se pega (barrido de colapso: ver GLUE_SWEEP)
FRAME_TOL = None                   # None = emparejar por orden de φ (zip); si float, sólo pega si |Δφ|<tol
# ===============================================================


def grow_frame(N, Dtan=2, kdeg=8, cos_min=COSM, m_cross=MCROSS, seed=0):
    """Crecimiento con ciclos baratos (λ_H=0) que DEVUELVE también phi (marco transportado)."""
    rng = np.random.default_rng(seed)
    adj = [set() for _ in range(N)]; dirs = [dict() for _ in range(N)]
    e = np.zeros(N); phi = np.zeros(N)

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

    def enlazar(i, j, d, cn, cost, set_phi_j=None):
        if not _pagar(e, cn, cost):
            return False
        adj[i].add(j); adj[j].add(i); dirs[i][j] = d; dirs[j][i] = -d
        if set_phi_j is not None:
            phi[j] = set_phi_j % (2 * np.pi)
        return True

    s0 = Dtan + 1
    base = _rand_unit(rng, Dtan); perp = _rand_unit(rng, Dtan)
    perp = perp - np.dot(perp, base) * base; perp /= (np.linalg.norm(perp) + 1e-12)
    sd = [np.cos(2 * np.pi * k / s0) * base + np.sin(2 * np.pi * k / s0) * perp for k in range(s0)]
    for i in range(s0):
        e[i] = INJECT
        for j in range(i + 1, s0):
            d = sd[j] - sd[i]; d /= (np.linalg.norm(d) + 1e-12)
            enlazar(i, j, d, [i, j], C_LINK, set_phi_j=(phi[i] + _psi2(d)) % (2 * np.pi))
    frontier = [i for i in range(s0) if len(adj[i]) < kdeg]

    def pick():
        while frontier:
            idx = int(rng.integers(len(frontier))); u = frontier[idx]
            if len(adj[u]) < kdeg:
                return u
            frontier[idx] = frontier[-1]; frontier.pop()
        return None

    for v in range(s0, N):
        u = pick()
        if u is None:
            break
        e[v] = INJECT; d_new = None
        for _ in range(60):
            c = _rand_unit(rng, Dtan)
            if libre(u, c):
                d_new = c; break
        if d_new is None:
            continue
        if not enlazar(u, v, d_new, [u, v], C_LINK, set_phi_j=(phi[u] + _psi2(d_new)) % (2 * np.pi)):
            continue
        cross = sorted(((float(np.dot(dw, d_new)), w) for w, dw in dirs[u].items() if w != v and float(np.dot(dw, d_new)) > 0),
                       reverse=True)
        added = 0
        for _, w in cross:
            if added >= m_cross:
                break
            dv = cerrar_plano(d_new, dirs[u][w])
            if dv is None or not libre(v, dv) or not libre(w, -dv):
                continue
            if enlazar(v, w, dv, [u, v, w], C_LINK):
                added += 1
        if len(adj[v]) < kdeg:
            frontier.append(v)
        if len(adj[u]) < kdeg:
            frontier.append(u)
        if v % DIFFUSE_EVERY == 0:
            _difundir(e, adj, v + 1)
    active = [i for i in range(N) if adj[i]]
    front = [i for i in active if len(adj[i]) < kdeg]     # frontera = no saturados
    return adj, phi, active, front


def combinar_y_pegar(seed, Dtan, Nhalf, kdeg, modo, rng):
    """Dos copias enfrentadas del mismo parche; pega frontera-A a frontera-B.
    modo='REGLA' -> por marco (φ); modo='CONTROL' -> al azar. Devuelve adj combinado (list[np.array]) y N."""
    adjA, phiA, actA, frontA = grow_frame(Nhalf, Dtan=Dtan, kdeg=kdeg, seed=seed)
    # B = copia idéntica (mismo seed), enfrentada: reflexión -> φ_B = -φ_A
    off = Nhalf
    S = [set(int(x) for x in adjA[i]) for i in range(Nhalf)]           # A
    Scomb = [set() for _ in range(2 * Nhalf)]
    for i in range(Nhalf):
        for j in S[i]:
            Scomb[i].add(j)
        for j in adjA[i]:                                             # B (misma topología, offset)
            Scomb[off + i].add(off + int(j))
    phiB = (-phiA) % (2 * np.pi)                                       # reflexión niega el ángulo
    frontB = [off + i for i in frontA]

    # candidatos: frontera-A vs frontera-B, misma cantidad
    fa = list(frontA); fb = list(frontB)
    G = min(len(fa), len(fb))
    G = int(G * GLUE_FRAC)
    if G <= 0:
        return [np.fromiter(s, dtype=np.int32) for s in Scomb], 2 * Nhalf, 0
    if modo == "REGLA":
        # emparejar por MARCO: ordenar A por φ_A y B por (φ enfrentado = -φ_B_local) y zippear
        fa_sorted = sorted(fa, key=lambda i: phiA[i])
        fb_sorted = sorted(fb, key=lambda j: (-phiA[j - off]) % (2 * np.pi))  # φ del nodo B enfrentado
        pairs = list(zip(fa_sorted[:G], fb_sorted[:G]))
        if FRAME_TOL is not None:
            pairs = [(a, b) for a, b in pairs
                     if min((phiA[a] - ((-phiA[b - off]) % (2*np.pi))) % (2*np.pi),
                            ((-phiA[b - off]) % (2*np.pi) - phiA[a]) % (2*np.pi)) < FRAME_TOL]
    else:  # CONTROL: azar
        a_sel = list(rng.permutation(fa))[:G]
        b_sel = list(rng.permutation(fb))[:G]
        pairs = list(zip(a_sel, b_sel))
    ng = 0
    for a, b in pairs:
        if b not in Scomb[a]:
            Scomb[a].add(b); Scomb[b].add(a); ng += 1
    return [np.fromiter(s, dtype=np.int32) for s in Scomb], 2 * Nhalf, ng


def sphere_turnover(adj, N, n_src=20, seed=0):
    """¿|S(r)| reconverge? Devuelve razón de crecimiento medio S(r+1)/S(r) en el tramo creciente.
    ~exponencial(árbol/mundo-pequeño): ratio alto sostenido. ~plano: ratio -> 1 antes de dar vuelta."""
    rng = np.random.default_rng(seed)
    active = [i for i in range(N) if len(adj[i]) > 0]
    if len(active) < n_src:
        return float("nan")
    ratios = []
    for s in rng.choice(active, size=n_src, replace=False):
        dist = np.full(N, -1, np.int32); dist[s] = 0; q = deque([int(s)])
        while q:
            u = q.popleft()
            for w in adj[u]:
                if dist[w] < 0:
                    dist[w] = dist[u] + 1; q.append(int(w))
        prof = np.bincount(dist[dist >= 0])
        if len(prof) < 4:
            continue
        peak = int(np.argmax(prof))
        if peak >= 2:
            rr = [prof[r + 1] / prof[r] for r in range(1, peak) if prof[r] > 0]
            if rr:
                ratios.append(float(np.mean(rr)))
    return float(np.mean(ratios)) if ratios else float("nan")


def _medir(adj, N, sd):
    dia = diametro(adj, N, seed=sd)
    g = dimension_crecimiento(adj, N, seed=sd)
    r = diagnos(adj, N, K, seed=sd + 11)
    turn = sphere_turnover(adj, N, seed=sd + 5)
    return dia, g, r, turn


def _slope(xs, ys):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    m = np.isfinite(xs) & np.isfinite(ys)
    if m.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(xs[m]), np.log(np.maximum(ys[m], 1.0)), 1)[0])


def main():
    csv_path = f"{LOG}.csv"
    cols = ["brazo", "Dt", "Nhalf", "seed", "N", "ng", "fg", "diam", "dmean", "d95", "dgrow", "turn", "ver"]
    t0 = time.time()
    print("CG004-d — TEST DE DOS FRENTES (pegado por marco vs azar) — ¿reconverge |S(r)|?")
    print("=" * 100)
    print(f"parche: crecer(cos_min={COSM}, λ_H=0, m_cross={MCROSS}) · GLUE_FRAC={GLUE_FRAC} · Dt∈{DTANS} · {len(SEEDS)} semillas")

    print("\nPre-vuelo métrica:")
    for Dt in DTANS:
        sd, ok = _check_no_degenerada(seed=0, Dtan=Dt); print(f"  Dtan={Dt}: {sd:.3e} -> {'OK' if ok else 'DEG'}"); assert ok
    print("\nAnclas (turn=razón S(r+1)/S(r); plano→~1, árbol→alto):")
    for nm, mk in (("lattice2D", lambda n: lattice2d(n)), ("arbol_b3", lambda n: arbol(n, 3))):
        adj, Nr = mk(4096); dia, g, r, turn = _medir(adj, Nr, 7)
        print(f"  {nm:>10} N={Nr}: turn={turn:.2f} diam={dia} δ={r['dmean']:.2f} d_grow={g['d']:.2f} %gig={r['fg']*100:.0f}")

    done = set()
    if os.path.exists(csv_path):
        for row in csv.DictReader(open(csv_path, newline="")):
            done.add((row["brazo"], int(row["Dt"]), int(row["Nhalf"]), int(row["seed"])))
    todos = ["REGLA", "CONTROL"]
    unidades = [(Dt, Nh, sd) for Dt in DTANS for Nh in NHALF for sd in SEEDS]
    total = len(unidades) * len(todos)
    print(f"\nfilas={total} hechas={len(done)} faltan={total-len(done)}  LOG={csv_path}\n")
    hdr = (f"{'brazo':>7} {'Dt':>2} {'Nhalf':>6} {'sd':>2} {'N':>6} {'ng':>5} {'%gig':>5} {'diam':>6} "
           f"{'δ_med':>7} {'d_grow':>6} {'turn':>5} {'ver':>10}")
    print(hdr); print("-" * len(hdr))
    nuevo = not os.path.exists(csv_path)
    fcsv = open(csv_path, "a", newline=""); w = csv.writer(fcsv)
    if nuevo:
        w.writerow(cols); fcsv.flush()

    def emit(br, Dt, Nh, sd, N, ng, dia, g, r, turn):
        w.writerow([br, Dt, Nh, sd, N, ng, r["fg"], dia, r["dmean"], r["d95"], g["d"], turn, g["ver"]]); fcsv.flush()
        print(f"{br:>7} {Dt:>2} {Nh:>6} {sd:>2} {N:>6} {ng:>5} {r['fg']*100:>4.0f} {dia:>6} "
              f"{r['dmean']:>7.2f} {g['d']:>6.2f} {turn:>5.2f} {g['ver']:>10}", flush=True)

    for (Dt, Nh, sd) in unidades:
        faltan = [b for b in todos if (b, Dt, Nh, sd) not in done]
        if not faltan:
            continue
        kdeg = 2 * Dt + 4
        for br in faltan:
            rng = np.random.default_rng(1000 * sd + Nh + (0 if br == "REGLA" else 7))
            adj, N, ng = combinar_y_pegar(sd, Dt, Nh, kdeg, br, rng)
            dia, g, r, turn = _medir(adj, N, sd)
            emit(br, Dt, Nh, sd, N, ng, dia, g, r, turn)
    fcsv.close()

    # -------- RESUMEN --------
    rows = list(csv.DictReader(open(csv_path, newline="")))
    def fnum(x):
        try:
            return float(x)
        except Exception:
            return float("nan")
    def per_seed(br, Dt, Nh, campo):
        return {int(r["seed"]): fnum(r[campo]) for r in rows if r["brazo"] == br and int(r["Dt"]) == Dt and int(r["Nhalf"]) == Nh}

    print("\n" + "=" * 100)
    print("RESUMEN — ¿REGLA (pegado por marco) reconverge y CONTROL (azar) no?  objetivo plano: diam-pend→1/Dt, turn→~1")
    for Dt in DTANS:
        print(f"\n  ── Dtan={Dt} (obj diam-pend={1.0/Dt:.2f}) ──")
        for br in todos:
            pends = []
            for sd in SEEDS:
                dias = [per_seed(br, Dt, Nh, "diam").get(sd, np.nan) for Nh in NHALF]
                pends.append(_slope([2*x for x in NHALF], dias))
            pends = [p for p in pends if p == p]
            pm, ps = (np.mean(pends), np.std(pends)) if pends else (np.nan, np.nan)
            dg = [np.nanmean(list(per_seed(br, Dt, Nh, "dgrow").values())) for Nh in NHALF]
            dme = [np.nanmean(list(per_seed(br, Dt, Nh, "dmean").values())) for Nh in NHALF]
            turn = [np.nanmean(list(per_seed(br, Dt, Nh, "turn").values())) for Nh in NHALF]
            gig = np.nanmean(list(per_seed(br, Dt, NHALF[-1], "fg").values())) * 100
            dtr = "CRECE" if dme[-1] > dme[0] + 0.5 else "acotada"
            print(f"    {br:>7}: diam-pend={pm:5.2f}±{ps:.2f}  %gig={gig:3.0f}  δ→{dtr}  "
                  f"d_grow={'/'.join(f'{x:.2f}' for x in dg)}  turn={'/'.join(f'{x:.2f}' for x in turn)}")
    print("\nLECTURA (pre-registrada, cuerdas CS):")
    print("  · GANA si REGLA reconverge (turn→~1, diam-pend→1/Dt, dim CONVERGE) y CONTROL no,")
    print("    con %gig sano y diámetro que NO colapsa (si colapsa = esfera, no plano).")
    print("  · REGLA ≈ CONTROL => pegado-por-marco tampoco es el lever (lo sabemos con script chico).")
    print(f"\nTiempo: {(time.time()-t0)/60:.1f} min · CSV: {csv_path}")


main()

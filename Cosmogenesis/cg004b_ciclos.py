"""
CG004-b — ¿la ABUNDANCIA de CICLOS PLANOS BARATOS despliega lo plano?
====================================================================
Diagnóstico decisivo (CC, 3-jul): el crecimiento con λ_H=2.0 (el usado por cg003f, cg003f-b
y cg004) produce un ÁRBOL PURO (E/V=1.00, clustering=0.000, cero triángulos). Telemetría:
de 1593 intentos de cross-link, 0 exitosos (955 fallan por EXERGÍA, 638 por ángulo). El
costo de holonomía cost=C_LINK+λ_H·H² hace los cierres INASEQUIBLES => sin ciclos => máx.
hiperbólico. El "costo de holonomía selecciona lazos planos" SE INVIERTE: λ_H>0 los mata.
A λ_H=0 sí se forman ciclos (E/V=1.39, 678 cierres) pero pocos (el gate angular bloquea 705)
y sin presión a llenar el espacio -> sigue hiperbólico y ralo (grado medio ~2.8, no ~6).

HIPÓTESIS: la hiperbolicidad es un ARTEFACTO DE INANICIÓN DE CICLOS. Hacer los cierres PLANOS
baratos (λ_H=0) y ABUNDANTES (más vecinos candidatos, arreglar la inanición de crecimiento
para acercar la coordinación a ~6 = retícula triangular 2D) debería producir geometría plana.

CANDADO (la variable nueva = régimen de formación de ciclos; todo lo demás igual):
  · ÁRBOL(=CONTROL) = crecer(λ_H=2.0)  -> árbol puro, hiperbólico (cg003f)
  · CICLOS          = λ_H=0 + m_cross alto + anti-inanición (cierres planos baratos abundantes)
  · AZAR            = shuffle(CICLOS)

CRITERIO PRE-REGISTRADO (emergencia, NO impuesto):
GANA CICLOS solo si se SEPARA de ÁRBOL hacia lo plano:
  · clustering ALTO (triángulos de verdad; árbol=0, retícula~0.4) — condición necesaria;
  · δ_med CRECE con N (no acotada ~0);
  · diam-pend sube hacia 1/Dt (0.5 en Dt=2);
  · dim CONVERGE (deja de trepar);
  · %gig alto (no fragmenta); shuffle NO lo reproduce.
Si CICLOS tiene clustering alto pero SIGUE hiperbólico (δ~0, dim trepa) => los ciclos locales
NO bastan; la planitud exige consistencia GLOBAL de marcos que ninguna regla local da (la
"pared"): resultado, no fracaso.

Reusa el arnés de medición de cg004_attach.py (mismas δ/diam/dim/controles).
"""
from __future__ import annotations

import csv
import os
import time
from collections import deque

import numpy as np

# --- cargar el arnés de cg004_attach.py SIN correr su main() ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(_HERE, "cg004_attach.py")).read().replace("\nmain()\n", "\n")
_M = {}
exec(compile(_src, "cg004_attach.py", "exec"), _M)

_rand_unit = _M["_rand_unit"]; cerrar_plano = _M["cerrar_plano"]
frustracion_ciclo = _M["frustracion_ciclo"]; _pagar = _M["_pagar"]; _difundir = _M["_difundir"]
C_LINK = _M["C_LINK"]; INJECT = _M["INJECT"]; DIFFUSE_EVERY = _M["DIFFUSE_EVERY"]
diametro = _M["diametro"]; dimension_crecimiento = _M["dimension_crecimiento"]
diagnos = _M["diagnos"]; shuffle_adj = _M["shuffle_adj"]
lattice2d = _M["lattice2d"]; arbol = _M["arbol"]
_check_no_degenerada = _M["_check_no_degenerada"]; _pend_diam = _M["_pend_diam"]
crecer = _M["crecer"]          # CONTROL == árbol (λ_H=2.0), idéntico a cg003f


# ============================ CONFIG ============================
MODO = "quick"          # quick: Ns=[1024,4096,16384], 2 sem
LOG  = "cg004b_ciclos"
M_CROSS_CICLOS = 8      # candidatos de cierre por nodo nuevo (era 2)
PRIMARY_TRIES  = 60     # intentos de dirección primaria (era 16) -> menos inanición
PRIMARY_RETRY  = 6      # nodos-frente alternativos a probar si u está saturado
# ==============================================================


def clustering(adj, N, n=400, seed=0):
    """Coef. de clustering local medio (fracción de pares de vecinos conectados)."""
    S = [set(int(x) for x in a) for a in adj]
    active = [i for i in range(N) if S[i]]
    if not active:
        return 0.0
    rng = np.random.default_rng(seed)
    samp = rng.choice(active, size=min(n, len(active)), replace=False)
    cs = []
    for u in samp:
        nb = list(S[u])
        if len(nb) < 2:
            continue
        links = 0; tot = 0
        for i in range(len(nb)):
            for j in range(i + 1, len(nb)):
                tot += 1
                if nb[j] in S[int(nb[i])]:
                    links += 1
        if tot:
            cs.append(links / tot)
    return float(np.mean(cs)) if cs else 0.0


def edges_per_node(adj, N):
    deg = np.array([len(a) for a in adj]); active = deg > 0
    if active.sum() == 0:
        return 0.0, 0.0
    return float(deg[active].mean()), float(deg.sum() // 2) / float(active.sum())


def crecer_ciclos(N, Dtan=2, kdeg=8, cos_min=0.5, m_cross=8, seed=0,
                  primary_tries=60, primary_retry=6):
    """λ_H=0: cierres PLANOS baratos (cerrar_plano, cost=C_LINK) y ABUNDANTES (m_cross alto).
    Anti-inanición: más intentos de dirección primaria + reintento en otro nodo del frente."""
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

    # semilla-símplex (idéntica)
    s0 = Dtan + 1
    base = _rand_unit(rng, Dtan); perp = _rand_unit(rng, Dtan)
    perp = perp - np.dot(perp, base) * base
    perp = perp / (np.linalg.norm(perp) + 1e-12)
    seed_dirs = [np.cos(2 * np.pi * k / s0) * base + np.sin(2 * np.pi * k / s0) * perp
                 for k in range(s0)]
    for i in range(s0):
        e[i] = INJECT
        for j in range(i + 1, s0):
            d = seed_dirs[j] - seed_dirs[i]; d = d / (np.linalg.norm(d) + 1e-12)
            enlazar(i, j, d, [i, j], C_LINK, set_phi_j=(phi[i] + _psi2(d)) % (2 * np.pi))

    # frente como pila con reinyección (pop al usar, re-append si queda cupo)
    frontier = [i for i in range(s0) if len(adj[i]) < kdeg]

    def pick_free():
        """Devuelve (u, d_new) con dirección primaria libre, probando varios nodos-frente."""
        tried = 0
        while frontier and tried < primary_retry:
            u = frontier.pop()
            tried += 1
            if len(adj[u]) >= kdeg:
                continue                      # saturado por grado -> fuera del frente
            d_new = None
            for _ in range(primary_tries):
                cand = _rand_unit(rng, Dtan)
                if libre(u, cand):
                    d_new = cand; break
            if d_new is not None:
                return u, d_new
            # u sin dirección libre -> interior -> NO re-añadir (sale del frente)
        return None, None

    for v in range(s0, N):
        u, d_new = pick_free()
        if u is None:
            break
        e[v] = INJECT
        if not enlazar(u, v, d_new, [u, v], C_LINK, set_phi_j=(phi[u] + _psi2(d_new)) % (2 * np.pi)):
            continue
        # cierres PLANOS baratos y abundantes (λ_H=0): cerrar_plano, cost=C_LINK
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
    return adj, dirs


def _finalizar(adj_sets):
    return [np.fromiter(s, dtype=np.int32) for s in adj_sets]


def _medir(adj, N, K, sd):
    dia = diametro(adj, N, seed=sd)
    g = dimension_crecimiento(adj, N, seed=sd)
    r = diagnos(adj, N, K, seed=sd + 11)
    clu = clustering(adj, N, seed=sd + 3)
    _, ev = edges_per_node(adj, N)
    return dia, g, r, clu, ev


def main():
    Ns, seeds, K = [1024, 4096, 16384], [1, 2], 120
    Dtans = [2]
    csv_path = f"{LOG}.csv"
    cols = ["brazo", "Dt", "N", "seed", "fg", "clu", "ev", "diam", "dmean", "d95", "dgrow", "ver"]

    t0 = time.time()
    print("CG004-b — ¿la ABUNDANCIA de CICLOS PLANOS BARATOS despliega lo plano?")
    print("=" * 100)
    print(f"CICLOS: λ_H=0 · m_cross={M_CROSS_CICLOS} · primary_tries={PRIMARY_TRIES} · retry={PRIMARY_RETRY}")

    print("\nPre-vuelo · no-degeneración de la métrica:")
    for Dt in Dtans:
        sd, ok = _check_no_degenerada(seed=0, Dtan=Dt)
        print(f"  Dtan={Dt}: std(24 dir)={sd:.3e} -> {'OK' if ok else 'DEGENERADA'}")
        assert ok

    print("\nSanidad medición (lattice2D: δ CRECE, clu>0 ; árbol: δ=0, clu=0):")
    print(f"  {'ancla':>10} {'N':>6} {'clu':>5} {'ev':>4} {'%gig':>5} {'diam':>5} {'δ_med':>7} {'d_grow':>6} {'ver':>10}")
    for Nanc in (Ns[0], Ns[-1]):
        for nombre, mk in (("lattice2D", lambda n: lattice2d(n)), ("arbol_b3", lambda n: arbol(n, 3))):
            adj, Nr = mk(Nanc)
            dia, g, r, clu, ev = _medir(adj, Nr, K, 7)
            print(f"  {nombre:>10} {Nr:>6} {clu:>5.2f} {ev:>4.2f} {r['fg']*100:>4.0f} {dia:>5} "
                  f"{r['dmean']:>7.2f} {g['d']:>6.2f} {g['ver']:>10}", flush=True)

    done = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                done.add((row["brazo"], int(row["Dt"]), int(row["N"]), int(row["seed"])))

    todos = ["ARBOL", "CICLOS", "AZAR"]
    unidades = [(Dt, N, sd) for Dt in Dtans for N in Ns for sd in seeds]
    total = len(unidades) * len(todos)
    print(f"\nBrazos: {' · '.join(todos)}")
    print(f"filas totales={total}  ya hechas={len(done)}  faltan={total-len(done)}   LOG={csv_path}\n")
    hdr = (f"{'brazo':>8} {'Dt':>2} {'N':>6} {'sd':>2} {'clu':>5} {'ev':>5} {'%gig':>5} {'diam':>5} "
           f"{'δ_med':>7} {'δ_q95':>7} {'d_grow':>6} {'ver':>10}")
    print(hdr); print("-" * len(hdr))

    nuevo = not os.path.exists(csv_path)
    fcsv = open(csv_path, "a", newline=""); w = csv.writer(fcsv)
    if nuevo:
        w.writerow(cols); fcsv.flush()

    def emit(br, Dt, N, sd, dia, g, r, clu, ev):
        w.writerow([br, Dt, N, sd, r["fg"], clu, ev, dia, r["dmean"], r["d95"], g["d"], g["ver"]])
        fcsv.flush()
        print(f"{br:>8} {Dt:>2} {N:>6} {sd:>2} {clu:>5.2f} {ev:>5.2f} {r['fg']*100:>4.0f} {dia:>5} "
              f"{r['dmean']:>7.2f} {r['d95']:>7.2f} {g['d']:>6.2f} {g['ver']:>10}", flush=True)

    for (Dt, N, sd) in unidades:
        faltan = [b for b in todos if (b, Dt, N, sd) not in done]
        if not faltan:
            continue
        kdeg = 2 * Dt + 4
        adjCIC = None
        if "ARBOL" in faltan:
            a_sets, _ = crecer(N, Dtan=Dt, kdeg=kdeg, lambda_H=2.0, seed=sd)
            adjA = _finalizar(a_sets)
            dia, g, r, clu, ev = _medir(adjA, N, K, sd); emit("ARBOL", Dt, N, sd, dia, g, r, clu, ev)
        if "CICLOS" in faltan or "AZAR" in faltan:
            c_sets, _ = crecer_ciclos(N, Dtan=Dt, kdeg=kdeg, m_cross=M_CROSS_CICLOS, seed=sd,
                                      primary_tries=PRIMARY_TRIES, primary_retry=PRIMARY_RETRY)
            adjCIC = _finalizar(c_sets)
        if "CICLOS" in faltan:
            dia, g, r, clu, ev = _medir(adjCIC, N, K, sd); emit("CICLOS", Dt, N, sd, dia, g, r, clu, ev)
        if "AZAR" in faltan and adjCIC is not None:
            adjZ = shuffle_adj(adjCIC, N, seed=sd + 7)
            dia, g, r, clu, ev = _medir(adjZ, N, K, sd); emit("AZAR", Dt, N, sd, dia, g, r, clu, ev)
    fcsv.close()

    # --- RESUMEN ---
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
    print("RESUMEN — ¿CICLOS se SEPARA de ÁRBOL hacia lo plano? (objetivo diam-pend→0.5, δ CRECE)")
    for Dt in Dtans:
        print(f"\n  ── Dtan={Dt} ──")
        for br in todos:
            dias = [mean_of(br, Dt, N, "diam") for N in Ns]
            pend = _pend_diam(Ns, dias)
            dme = [mean_of(br, Dt, N, "dmean") for N in Ns]; dme = [x for x in dme if x == x]
            dtrend = "CRECE(plano)" if len(dme) >= 2 and dme[-1] > dme[0] + 0.5 else "ACOTADA(hiperb)"
            clu = mean_of(br, Dt, Ns[-1], "clu"); ev = mean_of(br, Dt, Ns[-1], "ev")
            dims = "  ".join(f"N={N}:d={mean_of(br,Dt,N,'dgrow'):.2f}" for N in Ns)
            gig = mean_of(br, Dt, Ns[-1], "fg") * 100
            print(f"    {br:>8}: clu={clu:.2f} ev={ev:.2f} diam-pend={pend:5.2f}  δ→{dtrend:>16}  "
                  f"%gig={gig:4.0f}  [{dims}]")
    print("\nLECTURA (pre-registrada):")
    print("  · CICLOS con clu ALTO + δ CRECE + diam-pend→~0.5 + dim CONVERGE + shuffle destruye")
    print("    => los ciclos planos baratos despliegan lo plano. El lever era la INANICIÓN DE CICLOS.")
    print("  · CICLOS con clu alto pero δ~0 y dim trepa => ciclos locales NO bastan; la planitud")
    print("    exige consistencia GLOBAL de marcos (la 'pared'): resultado, no fracaso.")
    print("  · clu(CICLOS)~0 => no logramos formar ciclos (revisar gate angular): experimento inválido.")
    print(f"\nTiempo: {(time.time()-t0)/60:.1f} min · CSV: {csv_path}")


main()

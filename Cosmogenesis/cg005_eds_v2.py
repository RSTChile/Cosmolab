"""
CG005 v2 — EDS + orden temporal (cand 3) + FUERZA RESIDUAL débil (cand 2). ¿Se liga el medio → plano?
=====================================================================================================
v1 mostró: el orden temporal del congelamiento CONFINA mejor (REGLA_T 89% vs NULL_T 62%) pero da un
GAS de hadrones aislados (%gig 2-15%): un vínculo entre 2 hadrones ya neutros no sirve a ninguna tríada
neutra → sin premio → el costo lo poda → el confinamiento DESPRENDE los lazos inter-hadrón. Falta el
COMPLEMENTO: la fuerza fuerte RESIDUAL que liga hadrones neutros en materia extendida.

Re-adjudicación de CS (readjudicacion_cg005_v2): SÍ a cand 3 + cand 2. El (2) es legítimo AHORA porque
el (3) YA derivó la ubicabilidad TEMPORAL (no métrica) que el (2) necesita. Diseño v2 (mismo andamio):

REGLA v2 = confinamiento (intacto) + RESIDUAL, con las 3 trampas del residual cerradas en el diseño:
  1. El residual RESPETA la ventana temporal (igual que el confinamiento): sólo liga hadrones cuyos
     constituyentes son temporalmente adyacentes (Yukawa, corto alcance) — no campo medio. Se cumple
     porque las aristas propuestas YA están restringidas a la ventana (allow[i]).
  2. Peso residual DÉBIL, fijado por FÍSICA antes de correr: liga POCOS lazos inter-hadrón sin fundir
     los hadrones. λ_res·min(h_i, CAP_RES), λ_res>c_bond (para que ligue contra el costo) pero ≪ λ
     (para no fundir), y CAP_RES chico (pocos vecinos → medio, no blob). NO se retunea.
  3. El residual SATURA por nodo (CAP_RES): cada hadrón se liga a POCOS vecinos-en-tiempo (coord ~pocos)
     — es lo que distingue un medio EXTENDIDO de un blob small-world.
  h_i = nº de vecinos CONFINADOS (t>0) de i. Premio residual sólo a nodos confinados (hadrones ligan
  hadrones). Recalculado por barrido (débil, granularidad de sweep basta); t (confinamiento) exacto por
  move.

TRES GUARDIANES PRE-REGISTRADOS (CS):
  1. NULL-temporal (anti-circularidad): con residual, NULL_T (ventanas al azar) debe SEGUIR sin
     extender hacia el plano. Si REGLA_T extiende y NULL_T no → generación de la historia real, no de
     secuenciar por secuenciar.
  2. Acercamiento al ancla plana (ÉXITO, no sólo conexión): %gig→~100 (dejar de ser gas) Y ADEMÁS
     turn↓→1.15, δ↑→2.18, diam↑. Conectar en un blob (small-world, δ≈0, turn alto) NO es éxito.
  3. ANTI-DISOLUCIÓN (nuevo): los hadrones siguen siendo hadrones — tríadas-neutras/nodo tras v2 ≈ las
     de v1 (~3.3), NO menos. Si el residual funde hadrones, el confinamiento de color se rompe (vuelve
     al gas de quarks) → residual demasiado fuerte, inválido.

CUERDA DURA: λ_res y CAP_RES fijados por física ANTES de correr; NO subir para "que salga plano" (Shannon).
Secuencia (CS): v2 con orden AL AZAR primero; si liga, añadir REGLA_E (orden por energía) como 3er brazo.
Reusa el arnés de cg004_attach.py. numpy-only.
"""
from __future__ import annotations

import os
import time
import math
from collections import deque

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(_HERE, "cg004_attach.py")).read().replace("\nmain()\n", "\n")
_M = {}
exec(compile(_src, "cg004_attach.py", "exec"), _M)
diametro = _M["diametro"]; dimension_crecimiento = _M["dimension_crecimiento"]; diagnos = _M["diagnos"]
lattice2d = _M["lattice2d"]; arbol = _M["arbol"]


# ============================ CONFIG ============================
N        = 450
C_BOND   = 1.0
LAMBDA   = 6.0          # confinamiento (color), intacto de v1
TAU      = 3.0
LAMBDA_RES = 2.0        # RESIDUAL: > c_bond (liga contra el costo) pero ≪ λ (no funde). FIJO por física.
CAP_RES  = 3            # saturación residual: cada hadrón liga POCOS vecinos-en-tiempo (medio, no blob)
W        = 8
T_HI     = 3.0
T_LO     = 0.04
SWEEPS   = 280
K_LM     = 120
SEEDS    = [1, 2, 3, 4]
TRI_V1   = 3.3          # referencia de confinamiento de v1 (guardián 3: no debe caer)
# ===============================================================


def _colores(n, rng):
    c = np.tile(np.arange(3), n // 3 + 1)[:n]
    rng.shuffle(c)
    return c.astype(np.int8)


def _neutra(ci, cj, ck):
    return (ci != cj) and (cj != ck) and (ci != ck)


def _ventanas(N, W, modo, rng):
    orden = rng.permutation(N)                  # secuencia de congelamiento (al azar en v2)
    pos = np.empty(N, dtype=np.int64); pos[orden] = np.arange(N)
    allow = [None] * N
    if modo == "REGLA_T":
        for i in range(N):
            p = pos[i]; lo, hi = max(0, p - W), min(N, p + W + 1)
            allow[i] = [int(orden[q]) for q in range(lo, hi) if orden[q] != i]
    else:
        for i in range(N):
            cand = rng.choice(N, size=min(2 * W, N - 1), replace=False)
            allow[i] = [int(x) for x in cand if x != i]
    return allow


def _contar_triadas(adj, color, N):
    t = np.zeros(N, dtype=np.int32)
    for i in range(N):
        vs = list(adj[i])
        for a in range(len(vs)):
            for b in range(a + 1, len(vs)):
                if vs[b] in adj[vs[a]] and _neutra(color[i], color[vs[a]], color[vs[b]]):
                    t[i] += 1
    return t


def _h_confinados(adj, t, N):
    """h[i] = nº de vecinos CONFINADOS (t>0) de i (para el premio residual)."""
    h = np.zeros(N, dtype=np.int32)
    for i in range(N):
        h[i] = sum(1 for j in adj[i] if t[j] > 0)
    return h


def cuajar(N, color, allow, rng):
    """Metropolis: confinamiento saturante (λ, color, exacto por move) + RESIDUAL débil saturante
    (λ_res, entre confinados, recalculado por barrido). Propuestas restringidas a la ventana."""
    adj = [set() for _ in range(N)]
    for i in range(N):
        for j in allow[i]:
            if i < j and rng.random() < (1.5 / max(len(allow[i]), 1)):
                adj[i].add(j); adj[j].add(i)
    t = _contar_triadas(adj, color, N)
    h = _h_confinados(adj, t, N)

    def f(x):
        return 1.0 - math.exp(-x / TAU)

    def res(hx, confined):
        return (LAMBDA_RES * min(hx, CAP_RES)) if confined else 0.0

    for s in range(SWEEPS):
        T = T_HI * (T_LO / T_HI) ** (s / max(SWEEPS - 1, 1))
        h = _h_confinados(adj, t, N)                     # residual: granularidad de barrido (débil)
        for _ in range(N):
            i = int(rng.integers(N)); ai = allow[i]
            if not ai:
                continue
            j = ai[int(rng.integers(len(ai)))]
            if i == j:
                continue
            existe = j in adj[i]
            comunes = adj[i] & adj[j]; ci, cj = color[i], color[j]
            K = [k for k in comunes if _neutra(ci, cj, color[k])]
            m = len(K); sgn = -1 if existe else +1
            ti2, tj2 = t[i] + sgn * m, t[j] + sgn * m
            # --- confinamiento (exacto) ---
            drew = LAMBDA * ((f(ti2) - f(t[i])) + (f(tj2) - f(t[j])))
            for k in K:
                drew += LAMBDA * (f(t[k] + sgn) - f(t[k]))
            # --- residual (efecto de arista, con h del barrido) ---
            dhi = sgn if tj2 > 0 else 0                   # j confinado (post) aporta a h_i
            dhj = sgn if ti2 > 0 else 0
            dres = (res(h[i] + dhi, ti2 > 0) - res(h[i], t[i] > 0)) + \
                   (res(h[j] + dhj, tj2 > 0) - res(h[j], t[j] > 0))
            dE = sgn * 2.0 * C_BOND - drew - dres
            if dE <= 0 or rng.random() < math.exp(-dE / max(T, 1e-9)):
                if existe:
                    adj[i].discard(j); adj[j].discard(i)
                else:
                    adj[i].add(j); adj[j].add(i)
                t[i] += sgn * m; t[j] += sgn * m
                for k in K:
                    t[k] += sgn
                h[i] += dhi; h[j] += dhj                  # actualización local (se recalibra al sweep)
    return adj


def sphere_turnover(adj, N, n_src=20, seed=0):
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


def _fin(adj):
    return [np.fromiter(s, dtype=np.int32) for s in adj]


def _coord(adj, color, N):
    grados = []; ntri = []; en = 0
    for i in range(N):
        vs = list(adj[i]); grados.append(len(vs)); tt = 0
        for a in range(len(vs)):
            for b in range(a + 1, len(vs)):
                if vs[b] in adj[vs[a]] and _neutra(color[i], color[vs[a]], color[vs[b]]):
                    tt += 1
        ntri.append(tt)
        if tt > 0:
            en += 1
    return float(np.mean(grados)), float(np.mean(ntri)), en / N


def _medir(adj, color, N, sd):
    adjF = _fin(adj)
    dia = diametro(adjF, N, seed=sd)
    g = dimension_crecimiento(adjF, N, seed=sd)
    r = diagnos(adjF, N, K_LM, seed=sd + 11)
    turn = sphere_turnover(adjF, N, seed=sd + 5)
    gmed, tri, frac = _coord(adj, color, N)
    return dict(diam=dia, dim=g["d"], delta=r["dmean"], fg=r["fg"], turn=turn,
                gmed=gmed, tri=tri, frac=frac, ver=g["ver"])


def main():
    t0 = time.time()
    print("CG005 v2 — orden temporal (cand 3) + RESIDUAL débil (cand 2): ¿se liga el medio hacia el PLANO?")
    print("=" * 104)
    print(f"N={N} · λ={LAMBDA} τ={TAU} (confinamiento) · λ_res={LAMBDA_RES} CAP_RES={CAP_RES} (residual, FIJO) · W={W} · {len(SEEDS)} sem")

    print("\nAnclas (éxito = ACERCARSE a lattice2D):")
    anc = {}
    for nm, mk in (("lattice2D", lambda n: lattice2d(n)), ("arbol_b3", lambda n: arbol(n, 3))):
        a, Nr = mk(1024); m = _medir([set(x.tolist()) for x in a], np.zeros(Nr, np.int8), Nr, 7)
        anc[nm] = m
        print(f"  {nm:>10}: δ={m['delta']:.2f} turn={m['turn']:.2f} diam={m['diam']} dim={m['dim']:.2f} %gig={m['fg']*100:.0f}")

    print("\n" + "-" * 104)
    print(f"  {'brazo':>8} {'sd':>2} {'%gig':>5} {'g_med':>6} {'tri/nod':>8} {'%entri':>7} "
          f"{'diam':>5} {'δ_med':>7} {'dim':>6} {'turn':>6} {'ver':>10}")
    print("  " + "-" * 92)
    acc = {"REGLA_T": [], "NULL_T": []}
    for sd in SEEDS:
        color = _colores(N, np.random.default_rng(2000 + sd))
        for brazo in ("REGLA_T", "NULL_T"):
            allow = _ventanas(N, W, brazo, np.random.default_rng(3000 + sd * 10 + (0 if brazo == "REGLA_T" else 5)))
            adj = cuajar(N, color, allow, np.random.default_rng(4000 + sd * 10 + (0 if brazo == "REGLA_T" else 5)))
            assert color.dtype == np.int8 and len(color) == N
            m = _medir(adj, color, N, sd); acc[brazo].append(m)
            print(f"  {brazo:>8} {sd:>2} {m['fg']*100:>4.0f} {m['gmed']:>6.2f} {m['tri']:>8.2f} "
                  f"{m['frac']*100:>6.0f} {m['diam']:>5} {m['delta']:>7.2f} {m['dim']:>6.2f} "
                  f"{m['turn']:>6.2f} {m['ver']:>10}", flush=True)

    def prom(br, c):
        xs = [m[c] for m in acc[br] if m[c] == m[c]]
        return float(np.mean(xs)) if xs else float("nan")
    L = anc["lattice2D"]
    print("\n" + "=" * 104)
    print("RESUMEN (criterio: ¿REGLA_T se vuelve CONEXO Y se acerca al plano, y NULL_T no?):")
    print(f"  {'ancla plana':>10}: δ={L['delta']:.2f} turn={L['turn']:.2f} diam={L['diam']} dim={L['dim']:.2f} %gig=100")
    for br in ("REGLA_T", "NULL_T"):
        print(f"  {br:>10}: %gig={prom(br,'fg')*100:3.0f}  δ={prom(br,'delta'):.2f}  turn={prom(br,'turn'):.2f}  "
              f"diam={prom(br,'diam'):.0f}  dim={prom(br,'dim'):.2f}  tri/nodo={prom(br,'tri'):.2f}  g_med={prom(br,'gmed'):.2f}")

    # ---- veredicto con los 3 guardianes ----
    print("\nVEREDICTO (3 guardianes pre-registrados):")
    gig_R = prom("REGLA_T", "fg"); gig_N = prom("NULL_T", "fg")
    tri_R = prom("REGLA_T", "tri"); turn_R = prom("REGLA_T", "turn"); d_R = prom("REGLA_T", "delta")
    # G3 anti-disolución
    g3 = tri_R >= 0.7 * TRI_V1
    print(f"  G3 anti-disolución: tri/nodo REGLA_T={tri_R:.2f} vs v1~{TRI_V1} → {'OK (hadrones intactos)' if g3 else 'FALLA (residual fundió hadrones)'}")
    # conexión
    conx = gig_R > 0.8
    print(f"  Conexión: %gig REGLA_T={gig_R*100:.0f} → {'CONEXO (dejó de ser gas)' if conx else 'sigue gas/fragmentado'}")
    # G2 acercamiento al plano vs blob
    hacia_plano = conx and (turn_R < 0.5 * (12.0)) and (d_R > 0.5) and (prom("REGLA_T", "diam") > 15)
    # G1 NULL-temporal
    g1 = (prom("NULL_T", "delta") < d_R - 0.3) or (gig_N < gig_R - 0.2)
    print(f"  G2 hacia el plano (no blob): turn_R={turn_R:.2f} δ_R={d_R:.2f} diam_R={prom('REGLA_T','diam'):.0f} → "
          f"{'ACERCA al plano' if hacia_plano else 'blob/no-plano o gas'}")
    print(f"  G1 NULL-temporal: REGLA_T {'SE SEPARA de' if g1 else 'NO se separa de'} NULL_T")
    print("\n  DESENLACE:")
    if conx and hacia_plano and g1 and g3:
        print("    ★ CONEXO + HACIA EL PLANO + separado de NULL + hadrones intactos = PRIMER POSITIVO DE")
        print("      GENERACIÓN de espacio del arco (no preservación). Auditar el triple con CS.")
    elif conx and not hacia_plano:
        print("    El residual LIGA pero da BLOB hiperbólico (no plano): el medio existe pero curvo →")
        print("      falta AÚN un ingrediente que fuerce planitud (aguas arriba), ahora con medio conexo.")
    elif not conx:
        print("    Sigue FRAGMENTANDO: el residual débil no ligó. Reportar sin subir λ_res (sería Shannon).")
    if not g3:
        print("    (Y G3 falló: el residual fundió hadrones — inválido, no leer geometría.)")
    print(f"\nTiempo: {(time.time()-t0)/60:.1f} min")


main()

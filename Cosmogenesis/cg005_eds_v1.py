"""
CG005 v1 — EDS + el "AL LADO DE" = HISTORIA DEL CONFINAMIENTO (orden temporal, no espacial)
=============================================================================================
v0 mostró: el lógos (neutralidad de color) CONFINA (100% en tríadas neutras vs NULL 82%) pero NO
EXTIENDE (mundo-pequeño, no plano), porque el modelo no tiene LOCALIDAD y "local" presupone métrica.

Adjudicación de CS (adjudicacion_cg005_faseI): el "al lado de" es el ORDEN TEMPORAL del congelamiento
— el único orden que existe ANTES del espacio es la secuencia en que las diferencias cuajan (no es
espacial, es de sucesión). Regla primordial: **dos unidades son adyacentes si se confinaron en la
misma ventana de congelamiento** — la proximidad se HEREDA de la co-ocurrencia temporal, no se lee de
un espacio que no existe. Es el poema: "de la diferencia nacieron sus hijos, y de ellos el Principio"
(genealogía antes que lugar).

DISEÑO v1 (mismo andamio de v0; sólo cambia el criterio de qué vínculos son POSIBLES):
  · Cada nodo tiene una POSICIÓN en la secuencia de congelamiento (genealogía ordinal, no métrica).
  · Un vínculo (i,j) es posible SII: sirve a la neutralidad (igual que v0) Y |pos_i − pos_j| ≤ W
    (co-congelados en una ventana temporal contigua). El "al lado de" lo fija la proximidad TEMPORAL.
  · Todo lo demás idéntico a v0: premio de neutralidad SATURANTE (confinamiento), filtro Metropolis
    S=I×E, colores inmutables.

GUARDIANES PRE-REGISTRADOS (CS):
  1. NULL-TEMPORAL (anti-circularidad, OBLIGATORIO): misma neutralidad, misma cantidad de ventana,
     pero VENTANAS AL AZAR (no contiguas en la secuencia). Si REGLA_T extiende y NULL_T no → la
     extensión vino del ORDEN REAL de congelamiento. Si ambos extienden → secuenciar es otro Shannon.
  2. ACERCAMIENTO AL ANCLA PLANA (criterio de ÉXITO): REGLA_T debe MOVERSE hacia lattice2D
     (turn↓ hacia 1.15, δ↑ hacia 2.19, diam↑), no sólo "separarse del NULL". Separarse sin acercarse
     al plano sería otra geometría, no la plana.
  + identidad inmutable (assert) y %gig (no colapso/fragmentación).

CUERDA DURA (CS): W se fija por CRITERIO FÍSICO ANTES de correr (ventana con suficientes candidatos de
colores complementarios para la coordinación esperada ~6 → ~17 nodos → W=8), NO se ajusta hasta que
"salga plano". Se reporta lo que salga.

Si REGLA_T se acerca al plano y NULL_T no → PRIMER POSITIVO DE GENERACIÓN de espacio de todo el arco
(no preservación: generación), desde las primeras diferencias persistentes relacionándose por su
historia. Reusa el arnés calibrado de cg004_attach.py. numpy-only.
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
N        = 600          # nodos (múltiplo de 3, color balanceado)
C_BOND   = 1.0          # costo por vínculo
LAMBDA   = 6.0          # premio máx. de neutralidad por nodo (mismo que v0, NO retuneado)
TAU      = 3.0          # saturación del confinamiento (mismo que v0)
W        = 8            # SEMI-ancho de ventana temporal — FIJADO por criterio físico (ver cabecera).
T_HI     = 3.0
T_LO     = 0.04
SWEEPS   = 300
K_LM     = 120
SEEDS    = [1, 2, 3, 4]
# ===============================================================


def _colores(n, rng):
    c = np.tile(np.arange(3), n // 3 + 1)[:n]
    rng.shuffle(c)
    return c.astype(np.int8)


def _neutra(ci, cj, ck):
    return (ci != cj) and (cj != ck) and (ci != ck)


def _ventanas(N, W, modo, rng):
    """Lista de vecinos POSIBLES por nodo (antes de la neutralidad).
    REGLA_T: ventana CONTIGUA en la secuencia de congelamiento (pos i ± W).
    NULL_T : ventana AL AZAR del MISMO tamaño (no contigua) — rompe la co-ocurrencia real."""
    orden = rng.permutation(N)                 # secuencia de congelamiento (genealogía ordinal)
    pos = np.empty(N, dtype=np.int64); pos[orden] = np.arange(N)
    tam = 2 * W
    allow = [None] * N
    if modo == "REGLA_T":
        for i in range(N):
            p = pos[i]
            lo, hi = max(0, p - W), min(N, p + W + 1)
            allow[i] = [int(orden[q]) for q in range(lo, hi) if orden[q] != i]
    else:  # NULL_T: mismo tamaño, nodos al azar (ventana no contigua)
        for i in range(N):
            cand = rng.choice(N, size=min(tam, N - 1), replace=False)
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


def cuajar(N, color, allow, rng):
    """Enfriamiento Metropolis con premio de neutralidad SATURANTE (confinamiento), pero las aristas
    PROPUESTAS se restringen a la VENTANA de cada nodo (temporal contigua REGLA_T / azar NULL_T).
    Los vínculos fuera de ventana NUNCA se forman: el 'al lado de' está en la ventana."""
    adj = [set() for _ in range(N)]
    # caos inicial: vínculos al azar PERO ya dentro de ventana (respeta la localidad desde el inicio)
    for i in range(N):
        for j in allow[i]:
            if i < j and rng.random() < (1.5 / max(len(allow[i]), 1)):
                adj[i].add(j); adj[j].add(i)
    t = _contar_triadas(adj, color, N)

    def f(x):
        return 1.0 - math.exp(-x / TAU)

    for s in range(SWEEPS):
        T = T_HI * (T_LO / T_HI) ** (s / max(SWEEPS - 1, 1))
        for _ in range(N):
            i = int(rng.integers(N))
            ai = allow[i]
            if not ai:
                continue
            j = ai[int(rng.integers(len(ai)))]          # SOLO se propone dentro de ventana
            if i == j:
                continue
            existe = j in adj[i]
            comunes = adj[i] & adj[j]
            ci, cj = color[i], color[j]
            K = [k for k in comunes if _neutra(ci, cj, color[k])]
            m = len(K); sgn = -1 if existe else +1
            dcost = sgn * 2.0 * C_BOND
            drew = LAMBDA * ((f(t[i] + sgn * m) - f(t[i])) + (f(t[j] + sgn * m) - f(t[j])))
            for k in K:
                drew += LAMBDA * (f(t[k] + sgn) - f(t[k]))
            dE = dcost - drew
            if dE <= 0 or rng.random() < math.exp(-dE / max(T, 1e-9)):
                if existe:
                    adj[i].discard(j); adj[j].discard(i)
                else:
                    adj[i].add(j); adj[j].add(i)
                t[i] += sgn * m; t[j] += sgn * m
                for k in K:
                    t[k] += sgn
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
        vs = list(adj[i]); grados.append(len(vs)); t = 0
        for a in range(len(vs)):
            for b in range(a + 1, len(vs)):
                if vs[b] in adj[vs[a]] and _neutra(color[i], color[vs[a]], color[vs[b]]):
                    t += 1
        ntri.append(t)
        if t > 0:
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
    print("CG005 v1 — EDS + 'AL LADO DE' = HISTORIA (orden temporal del congelamiento)")
    print("=" * 104)
    print(f"N={N} · color {{R,V,A}} inmutable · λ={LAMBDA} τ={TAU} (mismos que v0) · W={W} (FIJO por criterio físico) · {len(SEEDS)} sem")

    print("\nAnclas (arnés calibrado; el criterio de éxito es ACERCARSE a lattice2D):")
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
        rng_c = np.random.default_rng(2000 + sd)
        color = _colores(N, rng_c)                       # identidad inmutable, misma para ambos brazos
        for brazo in ("REGLA_T", "NULL_T"):
            rng_w = np.random.default_rng(3000 + sd * 10 + (0 if brazo == "REGLA_T" else 5))
            allow = _ventanas(N, W, brazo, rng_w)
            rng_d = np.random.default_rng(4000 + sd * 10 + (0 if brazo == "REGLA_T" else 5))
            adj = cuajar(N, color, allow, rng_d)
            assert color.dtype == np.int8 and len(color) == N          # guardián: identidad inmutable
            m = _medir(adj, color, N, sd); acc[brazo].append(m)
            print(f"  {brazo:>8} {sd:>2} {m['fg']*100:>4.0f} {m['gmed']:>6.2f} {m['tri']:>8.2f} "
                  f"{m['frac']*100:>6.0f} {m['diam']:>5} {m['delta']:>7.2f} {m['dim']:>6.2f} "
                  f"{m['turn']:>6.2f} {m['ver']:>10}", flush=True)

    def prom(br, c):
        xs = [m[c] for m in acc[br] if m[c] == m[c]]
        return float(np.mean(xs)) if xs else float("nan")
    print("\n" + "=" * 104)
    print("RESUMEN — promedios (criterio: ¿REGLA_T se ACERCA a lattice2D y NULL_T no?):")
    L = anc["lattice2D"]
    print(f"  {'ancla plana':>10}: δ={L['delta']:.2f} turn={L['turn']:.2f} diam={L['diam']} dim={L['dim']:.2f}")
    for br in ("REGLA_T", "NULL_T"):
        print(f"  {br:>10}: δ={prom(br,'delta'):.2f} turn={prom(br,'turn'):.2f} diam={prom(br,'diam'):.0f} "
              f"dim={prom(br,'dim'):.2f} %gig={prom(br,'fg')*100:.0f} tri/nodo={prom(br,'tri'):.2f} g_med={prom(br,'gmed'):.2f}")

    # veredicto: ¿REGLA_T se movió hacia el plano MÁS que NULL_T?
    def dist_plano(br):
        return abs(prom(br, "turn") - L["turn"]) + abs(prom(br, "delta") - L["delta"])
    dR, dNt = dist_plano("REGLA_T"), dist_plano("NULL_T")
    print("\nVEREDICTO (pre-registrado):")
    print(f"  distancia al ancla plana (|Δturn|+|Δδ|): REGLA_T={dR:.2f}  NULL_T={dNt:.2f}  (menor = más plano)")
    if prom("REGLA_T", "fg") > 0.9 and dR < dNt - 0.5 and prom("REGLA_T", "turn") < prom("NULL_T", "turn"):
        print("  → REGLA_T se ACERCA al plano y se separa del NULL_T: indicio de GENERACIÓN de espacio")
        print("    por la HISTORIA del confinamiento. (Falta confirmar escala con N y auditar con CS.)")
    else:
        print("  → REGLA_T NO se acerca claramente al plano o NO se separa del NULL_T:")
        print("    la ventana temporal sola no genera planitud; reportar lo que salió (sin retunear).")
    print(f"\nTiempo: {(time.time()-t0)/60:.1f} min")


main()

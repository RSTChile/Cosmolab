"""
CG004-f — BARRIDO DE CURVATURA CONTROLADA (test P-κ) · ETAPA 1: fabricar y VALIDAR sustratos
=============================================================================================
Adjudicación de CS (adjudicacion_cg004e_CS.md): (P) plano superado (filtro válido sobre sustrato
trivial). El siguiente NO es "crecer hiperbólico + pegar" (confound: si solo pegas donde la
holonomía YA es ≈0, refuerzas el campo hiperbólico, no lo aplanas; y no separas "el pegado no
funciona" de "el crecimiento ya arruinó los marcos"). El test correcto es el BARRIDO DE CURVATURA
CONTROLADA: sustratos con un KNOB κ que FIJAS (no emerge del crecimiento), de plano a hiperbólico;
cortar+repegar en cada κ; medir a qué curvatura el pegado deja de preservar. Convierte "¿bootstrapea?"
en una FRONTERA medible y separa "preservar" de "generar" sin confound.

ESTA ETAPA 1 sólo FABRICA y VALIDA los sustratos (B-antes-de-A sobre la propia construcción):
  · KNOB = tesselación regular {3,q} (triángulos, q por vértice):
      q=6  -> {3,6} retícula triangular EUCLÍDEA (κ=0, plano)
      q=7  -> {3,7} hiperbólica (κ<0 chico)
      q=8  -> {3,8} hiperbólica (κ<0 mayor)  ...
    (déficit angular por vértice = (6−q)·π/3 ; q=6→0, q≥7→negativo = hiperbólico). κ lo fija q.
  · Hiperbólicos: BFS de isometrías del disco de Poincaré (SU(1,1)) — robusto, dedup por proximidad.
  · MAPA DE DESARROLLO afín (equilátero): se desarrolla con esquinas de π/3 (triángulo euclídeo).
    Alrededor de un vértice de grado q el marco ROTA (q−6)·π/3 => holonomía afín != 0 para q!=6.

VALIDACIÓN pre-registrada (cuerdas 1 y 3 de CS):
  (a) MÉTRICA pasa de plano a hiperbólico con q:
        q=6  -> δ CRECE con N, turn→~1, diam-pend→~0.5   (plano)
        q≥7  -> δ ACOTADA, turn alto, diam-pend→~0        (hiperbólico)
  (b) defdev (defecto de cierre afín de lazo) es la SEÑAL, no un error:
        q=6  -> defdev ≈ 0   (desarrollo univaluado)
        q≥7  -> defdev > 0 y CRECE con (q−6)   (curvatura real en los marcos)
  (c) %gig ~ 100 en todos (si un sustrato fragmenta, es defecto de construcción).
Si (a)(b)(c) se cumplen, la familia de sustratos es SANA y se gana el derecho a la Etapa 2
(cortar+repegar). Si defdev=0 en un q≥7, la construcción está mal (como el no-op de TEJIDO) y NO
se pasa a Etapa 2.

numpy-only, reutiliza el arnés de medición calibrado de cg004_attach.py.
"""
from __future__ import annotations

import os
import time
import cmath
import math
from collections import deque

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(_HERE, "cg004_attach.py")).read().replace("\nmain()\n", "\n")
_M = {}
exec(compile(_src, "cg004_attach.py", "exec"), _M)
diametro = _M["diametro"]; dimension_crecimiento = _M["dimension_crecimiento"]; diagnos = _M["diagnos"]


# ============================ CONFIG ============================
QS       = [6, 7, 8]        # knob de curvatura: 6=plano, 7/8=hiperbólico
TARGET_N = [800, 3000]      # dos tamaños por sustrato para ver la tendencia de δ con N
K        = 100              # landmarks para δ de Gromov
# ===============================================================


# ===========================================================================
#  Isometrías del disco de Poincaré como SU(1,1): (a,b), |a|²−|b|²=1,
#  acción z -> (a z + b)/(conj(b) z + conj(a)).  Composición = producto matricial.
# ===========================================================================
def su_rot(theta):
    return (cmath.exp(1j * theta / 2.0), 0.0 + 0.0j)

def su_trans(t):                       # traslación real por disco-radio t (|t|<1)
    s = 1.0 / math.sqrt(1.0 - t * t)
    return (complex(s, 0.0), complex(t * s, 0.0))

def su_mul(g1, g2):
    a1, b1 = g1; a2, b2 = g2
    return (a1 * a2 + b1 * b2.conjugate(), a1 * b2 + b1 * a2.conjugate())

def su_apply(g, z):
    a, b = g
    return (a * z + b) / (b.conjugate() * z + a.conjugate())

def su_inv(g):
    a, b = g
    return (a.conjugate(), -b)

def hdist(z, w):                        # distancia hiperbólica en el disco
    num = abs(z - w) ** 2
    den = (1 - abs(z) ** 2) * (1 - abs(w) ** 2)
    return math.acosh(max(1.0, 1.0 + 2.0 * num / max(den, 1e-18)))


# ===========================================================================
#  SUSTRATO {3,6} EUCLÍDEO (plano) — retícula triangular
# ===========================================================================
def tri_euclidea(target_n):
    L = max(4, int(round(math.sqrt(target_n))))
    idx = {}; pos = []
    for i in range(L):
        for j in range(L):
            idx[(i, j)] = len(pos)
            pos.append(complex(i + 0.5 * j, (math.sqrt(3) / 2) * j))
    N = len(pos)
    adj = [set() for _ in range(N)]
    nb = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]   # 6 vecinos = grado 6 interior
    for (i, j), u in idx.items():
        for di, dj in nb:
            v = idx.get((i + di, j + dj))
            if v is not None:
                adj[u].add(v); adj[v].add(u)
    return adj, pos, N


# ===========================================================================
#  SUSTRATO {3,q} HIPERBÓLICO — BFS de isometrías del disco (dedup por proximidad)
# ===========================================================================
def tri_hiperbolica(q, target_n, max_gen=40):
    alpha = 2.0 * math.pi / q
    cosh_l = math.cos(alpha) / (1.0 - math.cos(alpha))         # lado equilátero {3,q}
    ell = math.acosh(cosh_l)
    rr = math.tanh(ell / 2.0)                                  # disco-radio del vecino
    pbase = [rr * cmath.exp(1j * 2 * math.pi * k / q) for k in range(q)]
    step = [su_mul(su_mul(su_rot(2 * math.pi * k / q), su_trans(rr)), su_rot(math.pi))
            for k in range(q)]                                 # isometría "ir al vecino k"
    tol = ell * 0.45

    pos = [complex(0, 0)]; giso = [(1 + 0j, 0j)]               # vértice 0 en el centro
    adj = [set()]
    # hash espacial grueso en coords del disco para dedup rápido
    def key(z): return (round(z.real, 4), round(z.imag, 4))
    buckets = {}
    def add_bucket(z, i):
        kx, ky = key(z)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                buckets.setdefault((kx + dx * 1e-4, ky + dy * 1e-4), [])
        buckets.setdefault((kx, ky), []).append(i)
    def find(z):
        kx, ky = key(z)
        best = -1
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for i in buckets.get((round(kx + dx * 1e-4, 4), round(ky + dy * 1e-4, 4)), []):
                    if hdist(pos[i], z) < tol:
                        return i
        return best
    add_bucket(pos[0], 0)

    frontier = deque([0]); gen = {0: 0}
    while frontier and len(pos) < target_n:
        u = frontier.popleft()
        if gen[u] > max_gen:
            continue
        for k in range(q):
            zc = su_apply(giso[u], pbase[k])
            if abs(zc) > 0.99999:
                continue
            w = find(zc)
            if w < 0:
                w = len(pos)
                pos.append(zc); giso.append(su_mul(giso[u], step[k])); adj.append(set())
                add_bucket(zc, w); gen[w] = gen[u] + 1
                if len(pos) < target_n:
                    frontier.append(w)
            adj[u].add(w); adj[w].add(u)
    N = len(pos)
    # --- pasada final: cerrar aristas LATERALES entre vértices ya existentes ---
    # (los vértices creados tras el corte del frente nunca procesaron sus vecinos -> grado 1;
    #  aquí conectamos a cualquier vecino {3,q} que YA exista, sin crear nodos nuevos)
    for u in range(N):
        for k in range(q):
            zc = su_apply(giso[u], pbase[k])
            if abs(zc) > 0.99999:
                continue
            w = find(zc)
            if w >= 0 and w != u:
                adj[u].add(w); adj[w].add(u)
    return adj, pos, N, giso


# ===========================================================================
#  ORDEN CÍCLICO de vecinos (rotación) + DESARROLLO afín equilátero (esquinas π/3)
#  defdev = defecto de cierre afín en aristas no-árbol (0 si plano; >0 si curvo).
# ===========================================================================
def orden_ciclico(adj, pos, N, giso=None):
    """Devuelve, por vértice, la lista de vecinos ordenada por ángulo (rotación del vértice)."""
    orden = []
    for u in range(N):
        vs = list(adj[u])
        if giso is not None:
            gi = su_inv(giso[u])
            ang = [math.atan2(*(lambda z: (z.imag, z.real))(su_apply(gi, pos[w]))) for w in vs]
        else:
            ang = [math.atan2((pos[w] - pos[u]).imag, (pos[w] - pos[u]).real) for w in vs]
        orden.append([w for _, w in sorted(zip(ang, vs))])
    return orden


def curvatura_discreta(adj, N):
    """Holonomía afín ROTACIONAL por vértice = déficit angular de Gauss-Bonnet (desarrollo
    equilátero, esquinas π/3): deficit(v) = |2π − n_triángulos(v)·(π/3)|.
      · interior plano (grado 6) → 6·π/3 = 2π → deficit 0.
      · interior {3,q} (grado q) → deficit = |6−q|·π/3 (crece con la curvatura).
    Un vértice es INTERIOR si su fan cierra (n_triángulos == grado); en el borde el fan es
    abierto (n_tri = grado−1) y no se cuenta (evita el artefacto de borde del desarrollo).
    Devuelve (deficit_medio_interior, deficit_max, fracción_interior)."""
    CORNER = math.pi / 3.0
    sumdef = 0.0; maxdef = 0.0; nint = 0
    for v in range(N):
        S = list(adj[v]); d = len(S)
        if d < 3:
            continue
        ntri = 0
        for i in range(d):
            ai = adj[S[i]]
            for j in range(i + 1, d):
                if S[j] in ai:
                    ntri += 1
        if ntri == d:                                  # fan cerrado => vértice interior
            defc = abs(2 * math.pi - d * CORNER)
            sumdef += defc; maxdef = max(maxdef, defc); nint += 1
    meandef = sumdef / nint if nint else float("nan")
    return meandef, maxdef, (nint / N if N else 0.0)


# ===========================================================================
#  MÉTRICA de reconvergencia |S(r)| (turn) — de cg004d
# ===========================================================================
def sphere_turnover(adj, N, n_src=20, seed=0):
    rng = np.random.default_rng(seed)
    active = [i for i in range(N) if len(adj[i]) > 0]
    if len(active) < n_src:
        return float("nan")
    ratios = []
    for s in rng.choice(active, size=n_src, replace=False):
        dist = np.full(N, -1, np.int32); dist[s] = 0; qd = deque([int(s)])
        while qd:
            u = qd.popleft()
            for w in adj[u]:
                if dist[w] < 0:
                    dist[w] = dist[u] + 1; qd.append(int(w))
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


def construir(q, target_n):
    if q == 6:
        adj, pos, N = tri_euclidea(target_n); giso = None
    else:
        adj, pos, N, giso = tri_hiperbolica(q, target_n)
    orden = orden_ciclico(adj, pos, N, giso)
    return adj, pos, N, orden, giso


def main():
    t0 = time.time()
    print("CG004-f ETAPA 1 — fabricar y VALIDAR sustratos de curvatura controlada {3,q}")
    print("=" * 100)
    print(f"q∈{QS} (6=plano, ≥7=hiperbólico) · tamaños≈{TARGET_N} · déficit/vértice=(6−q)·π/3")
    print("\nValidación pre-registrada: q=6 → δ CRECE, turn~1, defic≈0 ; q≥7 → δ ACOTADA, turn alto,")
    print("  defic(déficit angular interior, Gauss-Bonnet) ≈ |6−q|·π/3 → q7≈1.05, q8≈2.09 rad")
    hdr = (f"  {'q':>2} {'N':>6} {'grado*':>6} {'%gig':>5} {'diam':>6} {'δ_med':>7} {'d_grow':>6} "
           f"{'turn':>5} {'defic':>7} {'%int':>5} {'ver':>10}")
    print("\n" + hdr); print("  " + "-" * (len(hdr) - 2))
    resumen = {}
    for q in QS:
        fila = []
        for tn in TARGET_N:
            adj, pos, N, orden, _giso = construir(q, tn)
            gmed = np.median([len(a) for a in adj])
            defdev, defmax, fint = curvatura_discreta(adj, N)
            adjF = _fin(adj)
            dia = diametro(adjF, N, seed=7)
            g = dimension_crecimiento(adjF, N, seed=7)
            r = diagnos(adjF, N, K, seed=9)
            turn = sphere_turnover(adjF, N, seed=5)
            print(f"  {q:>2} {N:>6} {gmed:>6.1f} {r['fg']*100:>4.0f} {dia:>6} {r['dmean']:>7.2f} "
                  f"{g['d']:>6.2f} {turn:>5.2f} {defdev:>7.3f} {fint*100:>4.0f} {g['ver']:>10}", flush=True)
            fila.append(dict(N=N, delta=r["dmean"], turn=turn, diam=dia, defdev=defdev,
                             fg=r["fg"], dgrow=g["d"]))
        resumen[q] = fila

    print("\n" + "=" * 100)
    print("VEREDICTO DE VALIDACIÓN (¿la familia de sustratos es SANA para la Etapa 2?)")
    ok_metrica = True; ok_defdev = True; ok_gig = True
    d6 = resumen.get(6)
    for q in QS:
        f = resumen[q]
        d_lo, d_hi = f[0]["delta"], f[-1]["delta"]
        crece = d_hi > d_lo + 0.5
        dd = max(x["defdev"] for x in f)
        gig = min(x["fg"] for x in f) * 100
        if q == 6:
            m = "PLANO ok" if crece else "¡esperaba δ CRECE!"
            b = "ok (≈0)" if dd < 1e-6 else f"¡esperaba ≈0, dio {dd:.1e}!"
            ok_metrica &= crece; ok_defdev &= (dd < 1e-6)
        else:
            m = "HIPERB ok" if not crece else "¡esperaba δ ACOTADA!"
            b = f"ok (>0, {dd:.2e})" if dd > 1e-6 else "¡esperaba >0, dio 0!"
            ok_metrica &= (not crece); ok_defdev &= (dd > 1e-6)
        ok_gig &= (gig > 90)
        print(f"  q={q}: métrica δ {f[0]['delta']:.2f}→{f[-1]['delta']:.2f} [{m}] · "
              f"defdev={dd:.2e} [{b}] · %gig={gig:.0f}")
    # monotonía de defdev con q (curvatura)
    dds = [max(x["defdev"] for x in resumen[q]) for q in QS]
    mono = all(dds[i] <= dds[i + 1] + 1e-9 for i in range(len(dds) - 1))
    print(f"\n  defdev por q: {' '.join(f'{q}:{d:.2e}' for q, d in zip(QS, dds))}  "
          f"[{'CRECE con κ ✓' if mono else '¡NO monótona!'}]")
    veredicto = ok_metrica and ok_defdev and ok_gig and mono
    print("\n  " + ("✓ SUSTRATOS SANOS → derecho a Etapa 2 (cortar+repegar por κ)."
                     if veredicto else
                     "✗ CONSTRUCCIÓN DEFECTUOSA → NO pasar a Etapa 2; arreglar el sustrato primero."))
    print(f"\nTiempo: {(time.time()-t0)/60:.1f} min")


main()

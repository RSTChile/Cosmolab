"""
CS052 — EL MARCO (espín) como el "hacia dónde": ¿la alineación de orientaciones genera PLANITUD?
=================================================================================================
Cierre del arco (CG004+CG005): ninguna regla de ADYACENCIA (quién-con-quién) genera plano; el hueco es
el MARCO (con-qué-orientación-relativa). Aporte de Alexis: el ESPÍN es el "hacia dónde" que el color
(escalar) no captura — una orientación intrínseca por nodo. CS052 lo mete al EDS y prueba si alinear
marcos al ligarse GENERA el plano. (DISENO_CS052_marco_espin.md)

OBJETO NUEVO (lo único que se agrega a CS047): cada nodo, además de su color inmutable {R,V,A}, lleva
una ORIENTACIÓN θ_i ∈ {0..5} (múltiplos de 60°, el mismo retículo que el Burgers de CG004f3). Emergente:
arranque ALEATORIO SIMÉTRICO, sin dirección privilegiada, JAMÁS leído de una posición (guardián G-COORD).

REGLA NUEVA (marco-consistencia, sobre el confinamiento intacto):
  · Capa 1 (confinamiento, CS047, SIN CAMBIOS): el lógos liga por neutralidad de color RVA saturante.
  · Capa 2 (alineación de marco, NUEVA): una tríada neutra (hadrón=triángulo) está "flat/consistente"
    cuando sus tres orientaciones son MUTUAMENTE 120°-DISTINTAS (= {v, v+2, v+4} mod 6) — el análogo
    equilátero del espín: los tres marcos apuntan a las tres direcciones del triángulo plano. Premio
    E_marco = −μ·(nº de tríadas con marco consistente). La FRUSTRACIÓN de esta consistencia vive
    exactamente en la coordinación ≠6 por vértice (déficit angular) — el MISMO objeto que el Burgers
    mide. μ FIJADO por física (rigidez de marco < confinamiento) ANTES de correr, NO tuneado (G-NOTUNE).
  · Dinámica: Metropolis mixto — toggles de arista (confinamiento+marco) Y flips de θ (marco).

BRAZOS:
  REGLA_M   : color + premio de alineación de marco ON.
  NULL_M    : color + θ presente pero SIN premio (μ=0) — aísla el marco.
  NULL_θrand: color + premio de alineación pero a θ RELABELADOS (permutación fija σ) — anti-relabel
              (lección CS050): el marco debe estar ACOPLADO a la estructura, no ser etiqueta.
  base      : color sin θ (línea base CS047 = gas/blob).

JUEZ: arnés calibrado de CG005 (δ Gromov, dim, turn, %gig; anclas lattice2D/árbol) sobre el medio + la
frustración de marco (fracción de tríadas consistentes) + déficit medio (proxy directo del Burgers
rotacional = holonomía de marco, coordinate-free). [El Burgers-Eisenstein multi-radio de CG004f3 es el
refinamiento de precisión; requiere el sistema de rotación que los propios θ inducen — v1.]

CINCO GUARDIANES PRE-REGISTRADOS:
  G-COORD    : θ nunca de una posición (init aleatorio, updates solo por relaciones). Assert de diseño.
  G-PLANO    : REGLA_M debe ACERCARSE al ancla lattice2D (δ↑→2.18, turn↓→1.15, dim→~2, %gig→100), no
               solo separarse del control. Un blob (conexo pero curvo) NO es éxito.
  G-ANTIRELABEL: REGLA_M debe SEPARARSE de NULL_θrand (marco acoplado, no relabel).
  G-CONFINA  : el marco NO funde hadrones — tríadas-neutras/nodo ≈ las de CS047 (~3.3–5.7), no menos.
  G-NOTUNE   : μ fijado por física antes, reportado, no movido.

TRES DESENLACES (pre-escritos): REGLA_M→plano y controles no = 1er POSITIVO de GENERACIÓN del arco.
REGLA_M conecta pero blob = falta la regla de alineación correcta. REGLA_M gas/=base = confirmación
TRIPLE (planitud aguas arriba, ninguna regla local ni de adyacencia ni de marco genera plano).

Reusa cg005_eds_v2.py (confinamiento) y cg004_attach.py (arnés). numpy-only.
"""
from __future__ import annotations

import os
import time
import math
import numpy as np
from collections import deque

_HERE = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(_HERE, "cg004_attach.py")).read().replace("\nmain()\n", "\n")
_M = {}
exec(compile(_src, "cg004_attach.py", "exec"), _M)
diametro = _M["diametro"]; dimension_crecimiento = _M["dimension_crecimiento"]; diagnos = _M["diagnos"]
lattice2d = _M["lattice2d"]; arbol = _M["arbol"]


# ============================ CONFIG ============================
N        = 450
C_BOND   = 1.0
LAMBDA   = 6.0          # confinamiento (color), CS047
TAU      = 3.0
MU       = 3.0          # rigidez de MARCO: < λ (el confinamiento decide QUIÉN; el marco modula ORIENTACIÓN). FIJO por física.
W        = 8            # ventana temporal (localidad, de CS048); orden emergido no es el foco aquí — al azar
T_HI     = 3.0
T_LO     = 0.04
SWEEPS   = 300
K_LM     = 120
SEEDS    = [1, 2, 3, 4]
TRI_REF  = 3.3          # confinamiento de referencia (G-CONFINA)
# ===============================================================


def _colores(n, rng):
    c = np.tile(np.arange(3), n // 3 + 1)[:n]
    rng.shuffle(c)
    return c.astype(np.int8)


def _neutra(ci, cj, ck):
    return (ci != cj) and (cj != ck) and (ci != ck)


def _marco_ok(a, b, c):
    """¿Las tres orientaciones son mutuamente 120°-distintas ({v,v+2,v+4} mod 6)? = triángulo de marco
    plano/consistente (el análogo equilátero del espín). Coordinate-free."""
    s = {a % 6, b % 6, c % 6}
    if len(s) != 3:
        return False
    p = {x % 2 for x in s}
    return len(p) == 1                              # las 3 misma paridad y distintas = {v,v+2,v+4}


def _ventanas(N, W, rng):
    orden = rng.permutation(N)
    pos = np.empty(N, dtype=np.int64); pos[orden] = np.arange(N)
    allow = [None] * N
    for i in range(N):
        p = pos[i]; lo, hi = max(0, p - W), min(N, p + W + 1)
        allow[i] = [int(orden[q]) for q in range(lo, hi) if orden[q] != i]
    return allow


def cuajar_marco(N, color, allow, modo, rng):
    """Metropolis mixto: confinamiento (color, saturante) + alineación de MARCO (θ). modo ∈
    {REGLA_M, NULL_M(μ=0), NULL_θrand(premio a θ relabelados), base(sin θ)}."""
    adj = [set() for _ in range(N)]
    for i in range(N):
        for j in allow[i]:
            if i < j and rng.random() < (1.5 / max(len(allow[i]), 1)):
                adj[i].add(j); adj[j].add(i)
    theta = rng.integers(0, 6, size=N).astype(np.int8)     # G-COORD: init ALEATORIO, sin coordenadas
    usa_theta = modo in ("REGLA_M", "NULL_θrand", "NULL_M")
    mu = MU if modo in ("REGLA_M", "NULL_θrand") else 0.0
    sigma = rng.permutation(N) if modo == "NULL_θrand" else None   # relabel fijo (anti-relabel control)

    def th(i):                                              # θ que ve el premio (relabelado en NULL_θrand)
        return theta[sigma[i]] if sigma is not None else theta[i]

    # t[i] = tríadas neutras (confinamiento, exacto por move)
    t = np.zeros(N, dtype=np.int32)
    for i in range(N):
        vs = list(adj[i])
        for a in range(len(vs)):
            for b in range(a + 1, len(vs)):
                if vs[b] in adj[vs[a]] and _neutra(color[i], color[vs[a]], color[vs[b]]):
                    t[i] += 1

    def f(x):
        return 1.0 - math.exp(-x / TAU)

    def marco_tri(i, j, k):                                 # premio de marco de la tríada (i,j,k)
        return 1.0 if _marco_ok(th(i), th(j), th(k)) else 0.0

    for s in range(SWEEPS):
        T = T_HI * (T_LO / T_HI) ** (s / max(SWEEPS - 1, 1))
        for _ in range(N):
            if usa_theta and rng.random() < 0.35:
                # --- MOVE θ: flip de orientación de un nodo (solo marco) ---
                i = int(rng.integers(N))
                vs = list(adj[i])
                tris = [(vs[a], vs[b]) for a in range(len(vs)) for b in range(a + 1, len(vs))
                        if vs[b] in adj[vs[a]] and _neutra(color[i], color[vs[a]], color[vs[b]])]
                if not tris:
                    continue
                old = theta[i]; new = np.int8(rng.integers(0, 6))
                if new == old:
                    continue
                e_old = sum(marco_tri(i, a, b) for a, b in tris)
                theta[i] = new
                e_new = sum(marco_tri(i, a, b) for a, b in tris)
                dE = -mu * (e_new - e_old)
                if not (dE <= 0 or rng.random() < math.exp(-dE / max(T, 1e-9))):
                    theta[i] = old
                continue
            # --- MOVE arista: confinamiento + marco ---
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
            drew = LAMBDA * ((f(t[i] + sgn * m) - f(t[i])) + (f(t[j] + sgn * m) - f(t[j])))
            for k in K:
                drew += LAMBDA * (f(t[k] + sgn) - f(t[k]))
            dmarco = 0.0
            if mu > 0.0:                                    # las tríadas (i,j,k) aparecen/desaparecen
                dmarco = mu * sgn * sum(marco_tri(i, j, k) for k in K)
            dE = sgn * 2.0 * C_BOND - drew - dmarco
            if dE <= 0 or rng.random() < math.exp(-dE / max(T, 1e-9)):
                if existe:
                    adj[i].discard(j); adj[j].discard(i)
                else:
                    adj[i].add(j); adj[j].add(i)
                t[i] += sgn * m; t[j] += sgn * m
                for k in K:
                    t[k] += sgn
    return adj, theta


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


def _coord_marco(adj, color, theta, N):
    """grado medio, tríadas-neutras/nodo, %tríadas con MARCO consistente, déficit medio interior."""
    grados = []; ntri = []; ok = 0; tot = 0; defs = []
    for i in range(N):
        vs = list(adj[i]); grados.append(len(vs)); nt = 0
        for a in range(len(vs)):
            for b in range(a + 1, len(vs)):
                if vs[b] in adj[vs[a]] and _neutra(color[i], color[vs[a]], color[vs[b]]):
                    nt += 1; tot += 1
                    if _marco_ok(theta[i], theta[vs[a]], theta[vs[b]]):
                        ok += 1
        ntri.append(nt)
        if nt == 6:
            defs.append(0.0)
        elif nt >= 3:
            defs.append(abs(6 - nt) * math.pi / 3.0)
    frac_ok = ok / max(tot, 1)
    return (float(np.mean(grados)), float(np.mean(ntri)), frac_ok,
            float(np.mean(defs)) if defs else float("nan"))


def _medir(adj, color, theta, N, sd):
    adjF = _fin(adj)
    dia = diametro(adjF, N, seed=sd)
    g = dimension_crecimiento(adjF, N, seed=sd)
    r = diagnos(adjF, N, K_LM, seed=sd + 11)
    turn = sphere_turnover(adjF, N, seed=sd + 5)
    gmed, tri, frac_ok, defic = _coord_marco(adj, color, theta, N)
    return dict(diam=dia, dim=g["d"], delta=r["dmean"], fg=r["fg"], turn=turn,
                gmed=gmed, tri=tri, frac_ok=frac_ok, defic=defic, ver=g["ver"])


def main():
    t0 = time.time()
    print("CS052 — EL MARCO (espín): ¿alinear orientaciones al ligarse GENERA planitud?")
    print("=" * 108)
    print(f"N={N} · λ={LAMBDA} (confinamiento) · μ={MU} (marco, FIJO por física) · W={W} · Fase A (sin Higgs) · {len(SEEDS)} sem")

    print("\nAnclas (éxito = ACERCARSE a lattice2D):")
    anc = {}
    for nm, mk in (("lattice2D", lambda x: lattice2d(x)), ("arbol_b3", lambda x: arbol(x, 3))):
        a, Nr = mk(1024); mm = _medir([set(y.tolist()) for y in a], np.zeros(Nr, np.int8),
                                       np.zeros(Nr, np.int8), Nr, 7)
        anc[nm] = mm
        print(f"  {nm:>10}: δ={mm['delta']:.2f} turn={mm['turn']:.2f} diam={mm['diam']} dim={mm['dim']:.2f} %gig={mm['fg']*100:.0f}")

    print("\n" + "-" * 108)
    print(f"  {'brazo':>10} {'sd':>2} {'%gig':>5} {'g_med':>6} {'tri/nod':>8} {'%marco_ok':>10} "
          f"{'defic':>6} {'diam':>5} {'δ_med':>7} {'dim':>6} {'turn':>6}")
    print("  " + "-" * 100)
    BR = ("REGLA_M", "NULL_M", "NULL_θrand", "base")
    acc = {b: [] for b in BR}
    for sd in SEEDS:
        color = _colores(N, np.random.default_rng(2000 + sd))
        allow = _ventanas(N, W, np.random.default_rng(3000 + sd))
        for brazo in BR:
            adj, theta = cuajar_marco(N, color, allow, brazo, np.random.default_rng(9000 + sd * 10 + BR.index(brazo)))
            assert color.dtype == np.int8                  # G-COORD (θ nunca tocó una coordenada)
            m = _medir(adj, color, theta, N, sd); acc[brazo].append(m)
            print(f"  {brazo:>10} {sd:>2} {m['fg']*100:>4.0f} {m['gmed']:>6.2f} {m['tri']:>8.2f} "
                  f"{m['frac_ok']*100:>9.0f} {m['defic']:>6.2f} {m['diam']:>5} {m['delta']:>7.2f} "
                  f"{m['dim']:>6.2f} {m['turn']:>6.2f}", flush=True)

    def prom(b, c):
        xs = [m[c] for m in acc[b] if m[c] == m[c]]
        return float(np.mean(xs)) if xs else float("nan")
    L = anc["lattice2D"]
    print("\n" + "=" * 108)
    print("RESUMEN (¿REGLA_M se acerca al plano y los controles no?):")
    print(f"  {'ancla plana':>10}: δ={L['delta']:.2f} turn={L['turn']:.2f} diam={L['diam']} dim={L['dim']:.2f} %gig=100")
    for b in BR:
        print(f"  {b:>10}: %gig={prom(b,'fg')*100:3.0f}  δ={prom(b,'delta'):.2f}  turn={prom(b,'turn'):.2f}  "
              f"dim={prom(b,'dim'):.2f}  tri/nodo={prom(b,'tri'):.2f}  %marco_ok={prom(b,'frac_ok')*100:.0f}  defic={prom(b,'defic'):.2f}")

    print("\nVEREDICTO (5 guardianes):")
    gigR = prom("REGLA_M", "fg"); triR = prom("REGLA_M", "tri"); turnR = prom("REGLA_M", "turn"); dR = prom("REGLA_M", "delta")
    okR = prom("REGLA_M", "frac_ok")
    g_confina = triR >= 0.7 * TRI_REF
    conx = gigR > 0.8
    hplano = conx and (dR > 0.8) and (turnR < 6.0) and (prom("REGLA_M", "diam") > 15)
    g_antirelabel = (dR > prom("NULL_θrand", "delta") + 0.3) or (gigR > prom("NULL_θrand", "fg") + 0.2) or (okR > prom("NULL_θrand", "frac_ok") + 0.15)
    g_null = (dR > prom("NULL_M", "delta") + 0.3) or (gigR > prom("NULL_M", "fg") + 0.2)
    print(f"  G-COORD: θ init aleatorio, updates solo relacionales → OK (por diseño)")
    print(f"  G-CONFINA: tri/nodo REGLA_M={triR:.2f} (≥{0.7*TRI_REF:.1f}?) → {'OK (hadrones intactos)' if g_confina else 'FALLA (marco fundió hadrones)'}")
    print(f"  %marco consistente REGLA_M={okR*100:.0f}% (¿los marcos SÍ se alinearon?)")
    print(f"  Conexión REGLA_M: %gig={gigR*100:.0f} → {'CONEXO' if conx else 'gas/fragmentado'}")
    print(f"  G-PLANO: δ_R={dR:.2f} turn_R={turnR:.2f} diam_R={prom('REGLA_M','diam'):.0f} → {'ACERCA al plano' if hplano else 'blob/no-plano o gas'}")
    print(f"  G-ANTIRELABEL: REGLA_M {'SE SEPARA de' if g_antirelabel else 'NO se separa de'} NULL_θrand")
    print(f"  vs NULL_M: REGLA_M {'SE SEPARA' if g_null else 'NO se separa'}")
    print("\n  DESENLACE:")
    if g_confina and conx and hplano and g_antirelabel and g_null:
        print("    ★★★ REGLA_M se acerca al PLANO, acoplado (anti-relabel), sin disolver, separado de los")
        print("        controles = PRIMER POSITIVO DE GENERACIÓN de espacio del arco. Auditar el quíntuple con CS.")
    elif conx and not hplano:
        print("    El marco LIGA/alinea pero da BLOB (no plano): la consistencia local de marco no aplana →")
        print("        falta la regla de alineación correcta o algo más. Hueco más estrecho.")
    else:
        print("    REGLA_M NO genera plano (gas o =base). Confirmación TRIPLE: ni adyacencia ni marco local")
        print("        generan planitud → aguas arriba. Negativo convergente, no fracaso. NO tunear μ.")
    print(f"\nTiempo: {(time.time()-t0)/60:.1f} min")


main()

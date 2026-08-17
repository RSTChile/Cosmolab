"""
CG005 v3 — REGLA_E: el orden de congelamiento EMERGE (no se asigna). ¿Percola hacia el plano?
==============================================================================================
v2: local puro→gas, no-local→blob, plano=ninguno. Diagnóstico profundizado por CS: el residual local
no percoló porque el orden temporal era AL AZAR (rng.permutation) → desacopla la adyacencia temporal
de la estructura relacional ("co-congelados en ventana" eran vecinos en una BARAJA, sin razón
relacional). No invalida el candidato 3: invalida el orden al azar como su realización.

Adjudicación de CS (adjudicacion_cg005_v2_REGLAE): la energía de congelamiento NO se asigna al inicio
(sería horneado/arbitrario, un pseudo-espacio) — EMERGE. Definición limpia, coordinate-free, físicamente
el enfriamiento de Alexis puesto en lo relacional:

  >>> t_freeze(i) = el instante en que el nodo i completa su PRIMERA TRÍADA NEUTRA. <<<
  Un quark cuaja cuando ENCUENTRA sus colores complementarios; los que la hallan antes, cuajan antes.

Por qué es correcta: coordinate-free (solo relaciones de color, NUNCA posición → pasa el guardián duro);
ENDÓGENA (output de la dinámica, la simetría inicial se rompe sola por azar microscópico); acopla TIEMPO
con ESTRUCTURA (el que cuajó temprano lo hizo con vecinos concretos → sus co-congelados comparten
constituyentes reales → el residual liga hadrones estructuralmente cercanos → puede percolar RETÍCULA en
vez de gas). Ese acoplamiento es lo que la baraja destruía.

DISEÑO (dos fases, sin pre-asignar nada):
  Fase 1 NUCLEACIÓN: enfría el confinamiento SIN ventana (all-pairs); registra t_freeze(i) = paso de la
    1ª tríada neutra. Umbral = 1ª tríada (físico, NO tuneable). Orden emergido = argsort(t_freeze).
  Fase 2 LIGADO: el residual débil (v2, mismos parámetros) liga dentro de ventanas construidas del orden
    EMERGIDO (co-congelados = t_freeze cercanos, ahora acoplado a estructura).
  Tres brazos: REGLA_E (orden emergido) vs REGLA_T (orden al azar, v2) vs NULL_T (ventanas al azar).

CUATRO GUARDIANES (los 3 de v2 + anti-relabel):
  1. NULL-temporal: REGLA_T (azar) debe SEGUIR sin percolar al plano.
  2. Acercamiento al ancla plana (no blob): %gig→100 Y turn↓→1.15, δ↑→2.18, dim→~2. Blob (δ≈0, turn alto)
     NO es éxito.
  3. Anti-disolución: hadrones intactos (tri/nodo ~3.3–3.8).
  4. ANTI-RELABEL (nuevo, imprescindible): t_freeze endógeno debe estar CORRELACIONADO con la estructura
     — los co-congelados comparten más aristas de nucleación que el azar. Si REGLA_E fuera solo una
     permutación disfrazada (misma estadística que la baraja), la capa no aportó. Mido:
     tasa(arista_nucleación | co-congelados) / tasa(azar). Debe ser ≫ 1 en REGLA_E y ≈1 en REGLA_T.

CUERDA GRANDE (CS): puede que NINGUNA regla local genere planitud (la planitud es AGUAS ARRIBA, como
mostró CG004). Si REGLA_E también falla → misma pared R7 desde el lado EDS = confirmación por dos caminos
independientes de que el lever está en GENERAR marcos, no en ligar. Vale tanto un positivo como ese
negativo convergente. λ_res y umbral FIJOS por física, no tunear.

Reutiliza cg005_eds_v2.py (binding) y cg004_attach.py (arnés). numpy-only.
"""
from __future__ import annotations

import os
import time
import math
import numpy as np
from collections import deque

_HERE = os.path.dirname(os.path.abspath(__file__))
_src2 = open(os.path.join(_HERE, "cg005_eds_v2.py")).read().replace("\nmain()\n", "\n")
_V2 = {"__file__": os.path.join(_HERE, "cg005_eds_v2.py")}
exec(compile(_src2, "cg005_eds_v2.py", "exec"), _V2)
_colores = _V2["_colores"]; _neutra = _V2["_neutra"]; cuajar = _V2["cuajar"]; _medir = _V2["_medir"]
lattice2d = _V2["lattice2d"]; arbol = _V2["arbol"]
LAMBDA = _V2["LAMBDA"]; TAU = _V2["TAU"]; C_BOND = _V2["C_BOND"]

# ============================ CONFIG ============================
N        = 450
W        = 8
NUC_SWEEPS = 160        # nucleación (genera el orden emergido)
SEEDS    = [1, 2, 3, 4]
TRI_V1   = 3.3
# (λ_res, CAP_RES, TAU, LAMBDA, C_BOND: heredados de v2, sin tocar)
# ===============================================================


def _contar_triadas(adj, color, N):
    t = np.zeros(N, dtype=np.int32)
    for i in range(N):
        vs = list(adj[i])
        for a in range(len(vs)):
            for b in range(a + 1, len(vs)):
                if vs[b] in adj[vs[a]] and _neutra(color[i], color[vs[a]], color[vs[b]]):
                    t[i] += 1
    return t


def nucleacion(N, color, rng, sweeps):
    """Fase 1: confinamiento SIN ventana (all-pairs). Registra t_freeze(i) = paso de la 1ª tríada neutra
    (S=I·E local cruza umbral). Devuelve (t_freeze, adj_nuc). Umbral = 1ª tríada, físico, no tuneable."""
    adj = [set() for _ in range(N)]
    for _ in range(int(2.0 * N)):
        i, j = int(rng.integers(N)), int(rng.integers(N))
        if i != j:
            adj[i].add(j); adj[j].add(i)
    t = _contar_triadas(adj, color, N)
    t_freeze = np.full(N, -1, dtype=np.int64)          # -1 = aún no congela
    for i in range(N):
        if t[i] > 0:
            t_freeze[i] = 0
    step = 0

    def f(x):
        return 1.0 - math.exp(-x / TAU)

    for s in range(sweeps):
        T = 3.0 * (0.04 / 3.0) ** (s / max(sweeps - 1, 1))
        for _ in range(N):
            step += 1
            i = int(rng.integers(N)); j = int(rng.integers(N))
            if i == j:
                continue
            existe = j in adj[i]
            comunes = adj[i] & adj[j]; ci, cj = color[i], color[j]
            K = [k for k in comunes if _neutra(ci, cj, color[k])]
            m = len(K); sgn = -1 if existe else +1
            ti2, tj2 = t[i] + sgn * m, t[j] + sgn * m
            drew = LAMBDA * ((f(ti2) - f(t[i])) + (f(tj2) - f(t[j])))
            for k in K:
                drew += LAMBDA * (f(t[k] + sgn) - f(t[k]))
            dE = sgn * 2.0 * C_BOND - drew
            if dE <= 0 or rng.random() < math.exp(-dE / max(T, 1e-9)):
                if existe:
                    adj[i].discard(j); adj[j].discard(i)
                else:
                    adj[i].add(j); adj[j].add(i)
                t[i] += sgn * m; t[j] += sgn * m
                for k in K:
                    t[k] += sgn
                for x in (i, j):                        # registrar 1ª nucleación
                    if t[x] > 0 and t_freeze[x] < 0:
                        t_freeze[x] = step
                for k in K:
                    if t[k] > 0 and t_freeze[k] < 0:
                        t_freeze[k] = step
    t_freeze[t_freeze < 0] = step + 1                   # nunca congelaron → al final del orden
    return t_freeze, adj


def _ventanas_desde_orden(orden, N, W):
    """allow[i] = nodos dentro de W posiciones de i EN EL ORDEN dado (contiguo)."""
    pos = np.empty(N, dtype=np.int64); pos[orden] = np.arange(N)
    allow = [None] * N
    for i in range(N):
        p = pos[i]; lo, hi = max(0, p - W), min(N, p + W + 1)
        allow[i] = [int(orden[q]) for q in range(lo, hi) if orden[q] != i]
    return allow, pos


def _ventanas_azar(N, W, rng):
    allow = [None] * N
    for i in range(N):
        cand = rng.choice(N, size=min(2 * W, N - 1), replace=False)
        allow[i] = [int(x) for x in cand if x != i]
    return allow


def g4_anti_relabel(pos, adj_nuc, N, W, rng, n=4000):
    """¿El orden (pos) está ACOPLADO a la estructura de nucleación? Tasa de arista_nuc entre pares
    CO-CONGELADOS (|Δpos|≤W) vs pares al AZAR. Ratio≫1 → acoplamiento real; ≈1 → permutación disfrazada."""
    orden = np.empty(N, dtype=np.int64); orden[pos] = np.arange(N)
    e_co = n_co = e_az = n_az = 0
    for _ in range(n):
        i = int(rng.integers(N)); dp = int(rng.integers(1, W + 1)) * (1 if rng.random() < 0.5 else -1)
        p = pos[i] + dp
        if 0 <= p < N:
            j = int(orden[p]); n_co += 1; e_co += (1 if j in adj_nuc[i] else 0)
        a, b = int(rng.integers(N)), int(rng.integers(N))
        if a != b:
            n_az += 1; e_az += (1 if b in adj_nuc[a] else 0)
    r_co = e_co / max(n_co, 1); r_az = e_az / max(n_az, 1)
    return r_co, r_az, (r_co / max(r_az, 1e-9))


def main():
    t0 = time.time()
    print("CG005 v3 — REGLA_E: el orden de congelamiento EMERGE (1ª tríada neutra). ¿Percola al PLANO?")
    print("=" * 104)
    print(f"N={N} · W={W} · nucleación {NUC_SWEEPS} sweeps (all-pairs) · λ_res/CAP_RES heredados de v2 · {len(SEEDS)} sem")

    print("\nAnclas (éxito = ACERCARSE a lattice2D):")
    anc = {}
    for nm, mk in (("lattice2D", lambda x: lattice2d(x)), ("arbol_b3", lambda x: arbol(x, 3))):
        a, Nr = mk(1024); mm = _medir([set(y.tolist()) for y in a], np.zeros(Nr, np.int8), Nr, 7)
        anc[nm] = mm
        print(f"  {nm:>10}: δ={mm['delta']:.2f} turn={mm['turn']:.2f} diam={mm['diam']} dim={mm['dim']:.2f} %gig={mm['fg']*100:.0f}")

    print("\n" + "-" * 104)
    print(f"  {'brazo':>8} {'sd':>2} {'%gig':>5} {'g_med':>6} {'tri/nod':>8} {'diam':>5} {'δ_med':>7} "
          f"{'dim':>6} {'turn':>6} {'G4 ratio':>9} {'ver':>10}")
    print("  " + "-" * 96)
    acc = {"REGLA_E": [], "REGLA_T": [], "NULL_T": []}
    g4acc = {"REGLA_E": [], "REGLA_T": []}
    for sd in SEEDS:
        color = _colores(N, np.random.default_rng(2000 + sd))
        # --- Fase 1: nucleación (orden emergido) ---
        t_freeze, adj_nuc = nucleacion(N, color, np.random.default_rng(5000 + sd), NUC_SWEEPS)
        orden_E = np.argsort(t_freeze, kind="stable")
        allow_E, pos_E = _ventanas_desde_orden(orden_E, N, W)
        # orden al azar (REGLA_T = v2)
        orden_T = np.random.default_rng(6000 + sd).permutation(N)
        allow_T, pos_T = _ventanas_desde_orden(orden_T, N, W)
        # ventanas al azar (NULL_T)
        allow_Nt = _ventanas_azar(N, W, np.random.default_rng(7000 + sd))
        # G4 anti-relabel (E acoplado? T no debería)
        rgen = np.random.default_rng(8000 + sd)
        _, _, ratioE = g4_anti_relabel(pos_E, adj_nuc, N, W, rgen)
        _, _, ratioT = g4_anti_relabel(pos_T, adj_nuc, N, W, rgen)
        g4acc["REGLA_E"].append(ratioE); g4acc["REGLA_T"].append(ratioT)

        for brazo, allow, ratio in (("REGLA_E", allow_E, ratioE), ("REGLA_T", allow_T, ratioT),
                                    ("NULL_T", allow_Nt, float("nan"))):
            adj = cuajar(N, color, allow, np.random.default_rng(9000 + sd * 10 + hash(brazo) % 7))
            assert color.dtype == np.int8
            m = _medir(adj, color, N, sd); acc[brazo].append(m)
            print(f"  {brazo:>8} {sd:>2} {m['fg']*100:>4.0f} {m['gmed']:>6.2f} {m['tri']:>8.2f} "
                  f"{m['diam']:>5} {m['delta']:>7.2f} {m['dim']:>6.2f} {m['turn']:>6.2f} "
                  f"{ratio:>9.2f} {m['ver']:>10}", flush=True)

    def prom(br, c):
        xs = [m[c] for m in acc[br] if m[c] == m[c]]
        return float(np.mean(xs)) if xs else float("nan")
    L = anc["lattice2D"]
    print("\n" + "=" * 104)
    print("RESUMEN (criterio: ¿REGLA_E percola Y se acerca al plano, y REGLA_T/NULL_T no?):")
    print(f"  {'ancla plana':>10}: δ={L['delta']:.2f} turn={L['turn']:.2f} diam={L['diam']} dim={L['dim']:.2f} %gig=100")
    for br in ("REGLA_E", "REGLA_T", "NULL_T"):
        extra = f"  G4-ratio={np.mean(g4acc[br]):.2f}" if br in g4acc else ""
        print(f"  {br:>10}: %gig={prom(br,'fg')*100:3.0f}  δ={prom(br,'delta'):.2f}  turn={prom(br,'turn'):.2f}  "
              f"diam={prom(br,'diam'):.0f}  dim={prom(br,'dim'):.2f}  tri/nodo={prom(br,'tri'):.2f}{extra}")

    print("\nVEREDICTO (4 guardianes):")
    gigE = prom("REGLA_E", "fg"); triE = prom("REGLA_E", "tri"); turnE = prom("REGLA_E", "turn"); dE = prom("REGLA_E", "delta")
    g4 = np.mean(g4acc["REGLA_E"]) > 1.5 and np.mean(g4acc["REGLA_T"]) < 1.5
    g3 = triE >= 0.7 * TRI_V1
    conx = gigE > 0.8
    hplano = conx and (turnE < 6.0) and (dE > 0.8) and (prom("REGLA_E", "diam") > 15)
    g1 = (gigE > prom("REGLA_T", "fg") + 0.2) or (dE > prom("REGLA_T", "delta") + 0.3)
    print(f"  G4 anti-relabel: ratio_E={np.mean(g4acc['REGLA_E']):.2f} (≫1?) ratio_T={np.mean(g4acc['REGLA_T']):.2f} (≈1?) → "
          f"{'OK (orden endógeno ACOPLADO a estructura)' if g4 else 'FALLA (relabel/no acoplado)'}")
    print(f"  G3 anti-disolución: tri/nodo_E={triE:.2f} → {'OK' if g3 else 'FALLA (fundió hadrones)'}")
    print(f"  Conexión REGLA_E: %gig={gigE*100:.0f} → {'CONEXO' if conx else 'gas/fragmentado'}")
    print(f"  G2 hacia el plano: δ_E={dE:.2f} turn_E={turnE:.2f} diam_E={prom('REGLA_E','diam'):.0f} → "
          f"{'ACERCA al plano' if hplano else 'blob/no-plano o gas'}")
    print(f"  G1 vs REGLA_T (azar): REGLA_E {'SE SEPARA' if g1 else 'NO se separa'}")
    print("\n  DESENLACE:")
    if g4 and g3 and conx and hplano and g1:
        print("    ★★ REGLA_E percola HACIA EL PLANO, acoplado (G4), sin disolver, separado del azar =")
        print("       PRIMER POSITIVO DE GENERACIÓN de espacio del arco. Auditar el cuádruple con CS.")
    elif not g4:
        print("    G4 falló: el orden endógeno NO está acoplado → REGLA_E sería relabel; la capa no aportó.")
    elif conx and not hplano:
        print("    REGLA_E liga pero da BLOB (no plano): el orden endógeno tampoco aplana; hueco más estrecho.")
    else:
        print("    REGLA_E NO percola al plano. Convergente con CG004: la planitud es AGUAS ARRIBA (pared R7")
        print("    desde el lado EDS). Negativo convergente = confirmación, no fracaso. NO tunear.")
    print(f"\nTiempo: {(time.time()-t0)/60:.1f} min")


main()

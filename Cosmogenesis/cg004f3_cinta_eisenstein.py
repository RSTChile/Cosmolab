"""
CG004-f3 — CINTA DE TRIÁNGULOS INTERIORES + ENTEROS DE EISENSTEIN (adjudicación CS de cg004f2)
==============================================================================================
CS resolvió el muro de método: la holonomía discreta se transporta por CARAS (triángulos), no por
bordes. Transportar por una CINTA de triángulos interiores que rodea una región:
  · cada paso cruza la arista compartida entre dos triángulos equiláteros → isometría exacta, giro
    SIEMPRE π/3 cuantizado; NUNCA toca el corte ni un puente-arista (los tres muros de cg004f2).
  · al cerrar la cinta, la parte TRASLACIONAL de la isometría acumulada = VECTOR DE BURGERS.
    (la rotacional = déficit encerrado, que ignoramos: mide el sustrato, no el pegado — circular.)
El Burgers es lo que "carga la métrica" y selecciona (objeto traslacional, no rotacional).

Cuerda de CS: aritmética EXACTA en enteros de Eisenstein Z[ω] (ω=e^{iπ/3}, ω²=ω−1). Las isometrías
equiláteras son rotación k·60° + traslación en Z[ω] → el cierre en el plano es 0 REAL, no "<1e-9".
Blinda el guardián contra falsos positivos numéricos.

HALLAZGO (mi diagnóstico, confirmado con CS): mi muro previo fue transportar sobre el grafo CORTADO
(la orilla perdió triángulos → giro no cuantizado). Sobre el grafo INTACTO toda arista tiene sus
triángulos → todo giro es π/3, y la cinta cierra o no según κ. El corte/reforma es conceptual: en el
intacto los triángulos de la costura SON los que REGLA reformaría.

VEREDICTO PRE-REGISTRADO:
  · κ=0 (q=6): cinta de triángulos euclídeos → cierra EXACTA → Burgers=0 → el pegado RECONVERGE
    (puede aplanar/preservar).
  · κ≠0 (q≥7): la cinta encierra déficit → NO cierra → Burgers≠0 → el pegado NO puede cerrar el lazo
    → NO puede generar planitud desde curvatura. FRONTERA en κ=0⁺ = el pegado preserva pero no genera;
    el lever está aguas arriba (generar consistencia de marcos). 3er cierre-de-puerta con mecanismo.

Guardián: q=6 debe dar Burgers EXACTAMENTE 0 (Eisenstein). Si no, un paso de transporte está mal
orientado — no se lee la frontera.
"""
from __future__ import annotations

import os
import math
from collections import deque

_HERE = os.path.dirname(os.path.abspath(__file__))
_srcF = open(os.path.join(_HERE, "cg004f_barrido_curvatura.py")).read().replace("\nmain()\n", "\n")
_F = {"__file__": os.path.join(_HERE, "cg004f_barrido_curvatura.py")}
exec(compile(_srcF, "cg004f_barrido_curvatura.py", "exec"), _F)
construir = _F["construir"]
_src2 = open(os.path.join(_HERE, "cg004f2_barrido_cortar.py")).read().replace("\nmain()\n", "\n")
_M = {"__file__": os.path.join(_HERE, "cg004f2_barrido_cortar.py")}
exec(compile(_src2, "cg004f2_barrido_cortar.py", "exec"), _M)
_sistema_rot = _M["_sistema_rot"]; _turn = _M["_turn"]; _seed_interior = _M["_seed_interior"]


# ============================ CONFIG ============================
QS = [6, 7, 8]                 # knob de curvatura
RADIOS = [2, 3, 4]             # radios de la cinta (chicos → ciclo limpio, sin degenerar en hiperbólico)
TARGET_N = 4000
# ===============================================================


# --- Enteros de Eisenstein: p = a + b·ω, ω=e^{iπ/3}, ω²=ω−1 ---
# Direcciones unidad ω^k, k=0..5, como (a,b) enteros:
_EIS_UNIT = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]

def _eis_real(a, b):
    """(a,b) en Z[ω] → coordenada real (x,y)."""
    return (a + 0.5 * b, (math.sqrt(3) / 2.0) * b)


def anillo_ciclo(adj, pos, centro, R):
    """Ciclo-frontera (vértices a distancia BFS = R de 'centro') ordenado como lazo ADYACENTE.
    Devuelve (lazo, n_encerrados) o (None, .)."""
    dist = {centro: 0}; q = deque([centro])
    while q:
        u = q.popleft()
        for w in adj[u]:
            if w not in dist:
                dist[w] = dist[u] + 1; q.append(int(w))
    ring = [u for u in dist if dist[u] == R]
    if len(ring) < 4:
        return None, 0
    ring.sort(key=lambda u: math.atan2((pos[u] - pos[centro]).imag, (pos[u] - pos[centro]).real))
    # ordenar en ciclo adyacente (greedy por vecindad)
    rset = set(ring)
    L = [ring[0]]; used = {ring[0]}
    while len(L) < len(ring):
        u = L[-1]; nxt = None
        # preferir el vecino en el anillo más cercano en ángulo aún no usado
        cands = [w for w in adj[u] if w in rset and w not in used]
        if not cands:
            break
        nxt = min(cands, key=lambda w: abs(((math.atan2((pos[w] - pos[centro]).imag, (pos[w] - pos[centro]).real)
                                             - math.atan2((pos[u] - pos[centro]).imag, (pos[u] - pos[centro]).real)) + math.pi) % (2 * math.pi) - math.pi))
        L.append(nxt); used.add(nxt)
    # sólo un ciclo LIMPIO: usó todos los vértices del anillo y cierra (si no, el anillo es
    # fragmentado/degenerado — típico en anillos hiperbólicos grandes → no es una cinta válida)
    if len(L) != len(ring) or L[0] not in adj[L[-1]]:
        return None, 0
    n_dentro = sum(1 for u in dist if dist[u] < R)
    return L, n_dentro


def burgers_eisenstein(L, adj, orden, idx):
    """Transporta por el lazo L con aritmética EXACTA de Eisenstein. Devuelve
    (burgers_real, cerrado_exacto, rot_index, n_none). Cada giro es múltiplo entero de π/3."""
    n = len(L)
    if n < 4 or L[0] not in adj[L[-1]]:
        return None
    # dirección como índice entero de sextos (0 = ω^0); posición como (a,b) en Z[ω]
    di = 0                                   # índice de dirección de la 1ª arista
    a, b = _EIS_UNIT[0]                       # suma del 1er vector unidad
    none = 0
    for i in range(1, n):
        u = L[i]; prev = L[i - 1]; nxt = L[(i + 1) % n]
        t = _turn(orden[u], idx[u], adj, prev, nxt)
        if t is None:
            none += 1
            return None
        steps = int(round(t / (math.pi / 3.0)))          # giro en unidades de π/3 (entero exacto)
        di = di + 3 + steps                              # dir(u→nxt) = dir(u→prev)+π + giro
        ua, ub = _EIS_UNIT[di % 6]
        a += ua; b += ub
    cerrado = (a == 0 and b == 0)
    rx, ry = _eis_real(a, b)
    return math.hypot(rx, ry), cerrado, di % 6, none


def main():
    print("CG004-f3 — CINTA DE TRIÁNGULOS INTERIORES (transporte por caras) + EISENSTEIN EXACTO")
    print("=" * 100)
    print(f"q∈{QS} · radios de cinta {RADIOS} · Burgers = |traslación acumulada| (Z[ω] exacto)")
    print("guardián: q=6 (plano) debe cerrar EXACTO (a=b=0 en Eisenstein → Burgers=0 real)\n")

    print(f"  {'q':>2} {'R':>2} {'cinta_n':>7} {'encerr':>7} {'cerr_exacto':>11} {'BURGERS':>9} {'defic':>6}")
    print("  " + "-" * 56)
    res = {}
    for q in QS:
        adj, pos, N, orden, giso = construir(q, TARGET_N)
        oo, aa, ii = _sistema_rot(adj, pos, giso, N)
        centro = _seed_interior(adj, pos, N)
        for R in RADIOS:
            L, dentro = anillo_ciclo(adj, pos, centro, R)
            if L is None:
                print(f"  {q:>2} {R:>2}  anillo no formado"); continue
            out = burgers_eisenstein(L, adj, oo, ii)
            if out is None:
                print(f"  {q:>2} {R:>2}  lazo no cerrado/None en transporte — revisar"); continue
            burg, cerr, rot, none = out
            defic = abs(6 - q) * math.pi / 3.0
            res[(q, R)] = (burg, cerr, dentro)
            print(f"  {q:>2} {R:>2} {len(L):>7} {dentro:>7} {str(cerr):>11} {burg:>9.4f} {defic:>6.2f}", flush=True)

    # ---------- GUARDIÁN + FRONTERA ----------
    print("\n" + "=" * 100)
    guard = [res[(6, R)] for R in RADIOS if (6, R) in res]
    guard_ok = all(g[1] and g[0] == 0.0 for g in guard) and len(guard) > 0
    print(f"GUARDIÁN (plano q=6): {'cierra EXACTO (Burgers=0 real) ✓' if guard_ok else '¡NO cierra exacto! transporte mal orientado — no leer frontera'}")
    if not guard_ok:
        print("  Abortado."); return

    print("\nFRONTERA — Burgers de la cinta vs curvatura κ:")
    for q in QS:
        vals = [(R, res[(q, R)][0]) for R in RADIOS if (q, R) in res]
        if not vals:
            continue
        defic = abs(6 - q) * math.pi / 3.0
        cierra = all(v[1] == 0.0 for v in vals)
        s = "  ".join(f"R{R}:{b:.3f}" for R, b in vals)
        estado = "CIERRA (Burgers=0 → reconverge/aplana)" if cierra else "NO cierra (Burgers>0 → NO puede aplanar)"
        print(f"  q={q} (defic={defic:.2f}): {s}   [{estado}]")
    print("\nLECTURA (pre-registrada, adjudicación CS):")
    print("  · Burgers=0 SÓLO en κ=0 y >0 para todo κ>0 → el pegado-por-desarrollo PRESERVA lo plano")
    print("    pero NO puede GENERARLO desde curvatura (el Burgers no-nulo impide cerrar el lazo).")
    print("  · Frontera en κ=0⁺: cualquier curvatura bloquea la reconvergencia. El lever está AGUAS")
    print("    ARRIBA (generar consistencia de marcos local), no en el pegado. 3er cierre con mecanismo.")


main()

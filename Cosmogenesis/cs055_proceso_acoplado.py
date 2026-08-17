"""
CS055 — El PROCESO ACOPLADO: enfriamiento + gravedad-con-caída + confinamiento + despliegue, TODO a la vez
=========================================================================================================
Planteo de Alexis: "Es un PROCESO, no una sucesión de sucesos. Medir cada cosa por separado nunca dará 3D
—lo vimos—. Metamos todas las variables que ocurrían JUNTAS (el calor que baja, la gravedad que cae con la
distancia, el confinamiento) y veamos qué universo sale. Al inicio podría ser cualquier cosa —3D, 4D, 10D—;
contrasto contra el único que existe. Puede salir 3D por ser el único viable, o puede que no. Por eso
experimento." (No se parametriza para que salga 3D — se corre para saber.)

Los aislados fueron PARCIALES y se corrigen mutuamente:
  · gravedad-con-alcance (CS054-v2) empuja la dimensión ABAJO (elige 2D: menos conectividad sobrevive el filo).
  · confinamiento de color (CS047): un barión junta R+V+A neutro; esos tríos podrían no saturar cómodos en
    2D → empujar ARRIBA. HIPÓTESIS de Alexis, a probar — NO impuesta.
  · enfriamiento: el reloj que las enciende a la vez.
HIPÓTESIS PRE-REGISTRADA (arriesgada, puede fallar): 3D es el FILO donde la gravedad (↓) y el confinamiento
(↑) se atrapan — un filo que SOLO existe cuando actúan JUNTAS. O sale 2D, o 4D, o no converge. El experimento
decide.

LAS 4 PIEZAS (todas en CADA paso, no en fases), moduladas por T(t) que baja (reloj global, curva física fija):
  1. ENFRIAMIENTO T(t): decae geométrico. NO conoce la dimensión.
  2. GRAVEDAD con caída por DISTANCIA DE GRAFO (CS054-v2): peso ∝ ρ_i·ρ_j/d_ij^α, d_ij=SALTOS (BFS), jamás
     coordenada. Intensidad escala con T (al inicio la gravedad era enorme).
  3. CONFINAMIENTO de color: nodos con color {R,V,A}; cuando T<UMBRAL, se premian tríos neutros R+V+A
     (hadronización). La regla es SOLO neutralidad de color — NUNCA menciona "3D" (G-CONFIN-CIEGO-A-DIM).
  4. DESPLIEGUE: remueve vínculos (estira/diluye).

BRAZOS (mismo arnés): acoplado / G-NULL(color barajado) / gravedad-sola / confinamiento-solo.
GUARDIANES (ingeniería del código): G-NO-PRESUPONER-ESPACIO (toda distancia por BFS), G-CONFIN-CIEGO-A-DIM
(confinamiento solo ve color), G-TASAS-FIJAS (fijas por física, se reporta el patrón — robustez), G-NULL,
G-APAGADO (aislados de referencia). Se mide la TRAYECTORIA (dim a lo largo del enfriamiento), no solo el final.

Reusa cs053 (ensemble, medidor) y cs054_v2 (gravedad con alcance). numpy-only.
"""
from __future__ import annotations

import os
import math
import numpy as np
from collections import deque

_HERE = os.path.dirname(os.path.abspath(__file__))
_s = open(os.path.join(_HERE, "cs053_persistencia_geometria.py")).read().replace("\nmain()\n", "\n")
_C = {"__file__": os.path.join(_HERE, "cs053_persistencia_geometria.py")}
exec(compile(_s, "cs053_persistencia_geometria.py", "exec"), _C)
medir_geom = _C["medir_geom"]
cadena = _C["cadena"]; cuadrada2d = _C["cuadrada2d"]; cubica3d = _C["cubica3d"]
hipercubica4d = _C["hipercubica4d"]; _tri_euclidea = _C["_tri_euclidea"]; _tri_hip = _C["_tri_hip"]; _arbol = _C["_arbol"]


# ============================ CONFIG (todas por física, fijas ANTES) ============================
T_HI, T_LO = 3.0, 0.04     # curva de enfriamiento (geométrica)
T_CONF     = 1.0           # umbral: T<T_CONF enciende el confinamiento (hadronización)
T_STEPS    = 10
G_RATE     = 0.06          # gravedad (contrae, escala con T)
C_RATE     = 0.06          # confinamiento (forma tríos neutros)
H_RATE     = 0.08          # despliegue (estira)
ALPHA      = 2             # caída de la gravedad (cuadrado inverso emergente)
D_MAX      = 2             # alcance de la gravedad (saltos, local — de la lección de CS054-v2)
SAT        = 6             # saturación del confinamiento (tríos neutros por nodo)
NREP       = 3
# ================================================================================================


def _ens():
    tipos = [("cadena_d1", lambda: cadena(500)), ("cuadr_d2pl", lambda: cuadrada2d(576)),
             ("tri_d2pl", _tri_euclidea), ("hip37_d2cv", lambda: _tri_hip(7)),
             ("cubo_d3pl", lambda: cubica3d(512)), ("hcubo_d4pl", lambda: hipercubica4d(625)),
             ("arbol_cv", _arbol)]
    out = []
    for _ in range(NREP):
        out += tipos
    return out


def _colores(N, rng):
    c = np.tile(np.arange(3), N // 3 + 1)[:N].astype(np.int8); rng.shuffle(c); return c


def _giant(adj, N):
    seen = np.zeros(N, bool); best = 0
    for s in range(N):
        if not seen[s] and adj[s]:
            q = deque([s]); seen[s] = True; c = 0
            while q:
                u = q.popleft(); c += 1
                for w in adj[u]:
                    if not seen[w]:
                        seen[w] = True; q.append(int(w))
            best = max(best, c)
    return best / N


def _diam(adj, N):
    src = next((i for i in range(N) if adj[i]), 0)
    def far(s):
        d = {s: 0}; q = deque([s]); f = s
        while q:
            u = q.popleft()
            for w in adj[u]:
                if w not in d:
                    d[w] = d[u] + 1; q.append(int(w))
                    if d[w] > d[f]:
                        f = w
        return f, d[f]
    a, _ = far(src); _, dd = far(a); return dd


def _grav_paso(adj, N, rng, T):
    """Gravedad con caída por distancia de grafo (CS054-v2). d_ij = SALTOS (BFS), JAMÁS coordenada."""
    E = sum(len(a) for a in adj) // 2
    rho = np.array([len(a) for a in adj], float)
    if rho.sum() <= 0:
        return
    nadd = max(1, int(G_RATE * (0.5 + T) * E))          # intensidad escala con T
    srcs = rng.choice(N, size=nadd, p=rho / rho.sum())
    for i in srcs:
        i = int(i); dist = {i: 0}; q = deque([i])
        while q:
            u = q.popleft()
            if dist[u] >= D_MAX:
                continue
            for w in adj[u]:
                if w not in dist:
                    dist[w] = dist[u] + 1; q.append(int(w))
        cand = [(j, d) for j, d in dist.items() if d >= 2]
        if not cand:
            continue
        w = np.array([rho[j] / (d ** ALPHA) for j, d in cand])
        if w.sum() <= 0:
            continue
        j = cand[int(rng.choice(len(cand), p=w / w.sum()))][0]
        adj[i].add(j); adj[j].add(i)


def _confin_paso(adj, N, col, t, rng):
    """Confinamiento: forma TRÍOS NEUTROS R+V+A (hadronización). Solo ve COLOR — jamás dimensión
    (G-CONFIN-CIEGO-A-DIM). Satura por nodo (SAT). Un trío = i + par (j,k) de los otros dos colores, j~k."""
    E = sum(len(a) for a in adj) // 2
    nc = max(1, int(C_RATE * E))
    for _ in range(nc):
        i = int(rng.integers(N))
        if t[i] >= SAT:
            continue
        ci = col[i]
        # candidatos a 2 saltos: buscar par (j,k) de colores complementarios ya vinculados (j~k)
        vecinos2 = set()
        for u in adj[i]:
            vecinos2.add(u)
            for w in adj[u]:
                vecinos2.add(w)
        vecinos2.discard(i)
        otros = [x for x in vecinos2 if col[x] != ci]
        hecho = False
        for a in otros:
            if hecho:
                break
            for b in adj[a]:
                if b != i and col[b] != ci and col[b] != col[a] and b in vecinos2:
                    # {ci, col[a], col[b]} = {R,V,A} neutro → cerrar el trío
                    adj[i].add(a); adj[a].add(i); adj[i].add(b); adj[b].add(i)
                    t[i] += 1; t[a] = t.get(a, 0) + 1 if isinstance(t, dict) else t[a]
                    hecho = True
                    break


def _despliegue_paso(adj, N, rng):
    edges = [(i, j) for i in range(N) for j in adj[i] if i < j]
    if edges:
        rng.shuffle(edges)
        for (i, j) in edges[:int(H_RATE * len(edges))]:
            adj[i].discard(j); adj[j].discard(i)


def proceso(adj0, N, color, rng, gravedad=True, confinamiento=True, despliegue=True, null_color=False):
    """UN bucle temporal: en CADA paso, las 4 fuerzas a la vez, moduladas por T(t) que baja."""
    adj = [set(a) for a in adj0]
    col = color.copy()
    if null_color:
        rng.shuffle(col)                                 # G-NULL: color barajado (confin ciego a estructura)
    t = np.zeros(N, dtype=np.int32)
    traj = []
    for step in range(T_STEPS):
        T = T_HI * (T_LO / T_HI) ** (step / max(T_STEPS - 1, 1))
        if sum(len(a) for a in adj) < 4:
            break
        if gravedad:
            _grav_paso(adj, N, rng, T)
        if confinamiento and T < T_CONF:                 # el enfriamiento ENCIENDE el confinamiento
            _confin_paso(adj, N, col, t, rng)
        if despliegue:
            _despliegue_paso(adj, N, rng)
        if step in (0, T_STEPS // 2, T_STEPS - 1):
            traj.append((round(T, 2), round(_giant(adj, N), 2), _diam(adj, N)))
    return adj, traj


def _vive(adj, N):
    return (_giant(adj, N) > 0.5) and (_diam(adj, N) > 1.6 * math.log2(max(N, 2)))


def main():
    rng = np.random.default_rng(2029)
    print("CS055 — PROCESO ACOPLADO: enfriamiento + gravedad-con-caída + confinamiento + despliegue, JUNTOS")
    print("=" * 110)
    print(f"T:{T_HI}→{T_LO} · T_conf={T_CONF} · G={G_RATE} C={C_RATE} H={H_RATE} α={ALPHA} D_MAX={D_MAX} SAT={SAT} (FIJOS)")
    print("HIPÓTESIS pre-registrada: gravedad tira dim ABAJO (2D), confinamiento ARRIBA → 3D podría ser el filo. O no.")

    espec = _ens()
    BR = ("acoplado", "G_NULL", "grav_sola", "confin_solo")
    tipos = {}
    print(f"\n  {'geom':>11} {'dim0':>5} | " + " ".join(f"{b:>11}" for b in BR))
    print("  " + "-" * 74)
    traj_ej = None
    for nombre, build in espec:
        try:
            adj, N = build()
        except Exception:
            continue
        color = _colores(N, np.random.default_rng(int(rng.integers(1 << 30))))
        dim0, _, _ = medir_geom(adj, N)
        res = {}
        for br in BR:
            r = np.random.default_rng(int(rng.integers(1 << 30)))
            kw = dict(gravedad=True, confinamiento=True, despliegue=True, null_color=False)
            if br == "G_NULL":
                kw["null_color"] = True
            elif br == "grav_sola":
                kw["confinamiento"] = False
            elif br == "confin_solo":
                kw["gravedad"] = False
            a, traj = proceso(adj, N, color, r, **kw)
            res[br] = _vive(a, N)
            if br == "acoplado" and nombre == "cubo_d3pl" and traj_ej is None:
                traj_ej = (nombre, traj)
        tipos.setdefault(nombre, {"n": 0, "dim0": dim0, **{b: 0 for b in BR}})
        tipos[nombre]["n"] += 1
        for b in BR:
            tipos[nombre][b] += int(res[b])
        print(f"  {nombre:>11} {dim0:>5.2f} | " + " ".join(f"{'VIVE' if res[b] else 'muere':>11}" for b in BR), flush=True)

    print("\n" + "=" * 110)
    print("SUPERVIVIENTES por tipo (fracción) — ¿el ACOPLADO cambia el patrón de la gravedad-sola?:")
    print(f"  {'geom':>11} {'dim0':>5} | " + " ".join(f"{b:>10}" for b in BR))
    for nm in sorted(tipos):
        t = tipos[nm]
        print(f"  {nm:>11} {t['dim0']:>5.2f} | " + " ".join(f"{t[b]}/{t['n']:<8}" for b in BR))

    if traj_ej:
        print(f"\nTRAYECTORIA (ej: {traj_ej[0]}, acoplado) — (T, %gig, diam) al enfriar:")
        print("  " + "  ".join(f"T={T}:gig={g},diam={d}" for T, g, d in traj_ej[1]))

    # ---- veredicto (por tipos, no por el contador roto) ----
    print("\n" + "=" * 110)
    print("VEREDICTO (por TIPOS de retículo — la dimensión verdadera):")
    def frac(nm, br):
        return tipos[nm][br] / tipos[nm]["n"] if nm in tipos else 0
    d2 = [nm for nm in tipos if "d2" in nm]; d3 = [nm for nm in tipos if "d3" in nm]; d4 = [nm for nm in tipos if "d4" in nm]
    def viv(brs, br):
        return sum(tipos[nm][br] for nm in brs), sum(tipos[nm]["n"] for nm in brs)
    for br in BR:
        v2 = viv(d2, br); v3 = viv(d3, br); v4 = viv(d4, br)
        print(f"  {br:>11}: 2D {v2[0]}/{v2[1]}   3D {v3[0]}/{v3[1]}   4D {v4[0]}/{v4[1]}")
    print("\n  LECTURA (pre-registrada):")
    a3 = viv(d3, "acoplado"); g3 = viv(d3, "grav_sola"); a2 = viv(d2, "acoplado"); a4 = viv(d4, "acoplado")
    print(f"  · ¿el confinamiento RESCATA 3D? acoplado 3D={a3[0]}/{a3[1]} vs grav_sola 3D={g3[0]}/{g3[1]}.")
    if a3[0] > g3[0] and a3[0] >= a2[0] and a3[0] >= a4[0]:
        print("    → El ACOPLADO deja 3D poblado (más que gravedad-sola) y ≥ que 2D/4D → 3D EMERGE del PROCESO.")
        print("      3D no lo elige un ingrediente: lo elige la tensión gravedad↓ vs confinamiento↑. (Auditar G-NULL.)")
    elif a2[0] > a3[0]:
        print("    → El acoplado sigue prefiriendo 2D (como la gravedad sola): el confinamiento NO subió a 3D.")
        print("      Falsación honesta: el acoplamiento propuesto no basta para fijar 3D. Falta otra pieza.")
    else:
        print("    → Patrón mixto/otro: reportar cuál dimensión domina y si G-NULL lo reproduce (=confin inerte).")
    print("  · G-NULL (color barajado): si su patrón dimensional = acoplado → el confinamiento fue inerte.")
    print("  · G-NO-PRESUPONER-ESPACIO: toda distancia por BFS/saltos. G-CONFIN-CIEGO-A-DIM: confin solo vio color.")


main()

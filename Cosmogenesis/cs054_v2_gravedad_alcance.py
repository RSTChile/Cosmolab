"""
CS054-v2 — La GRAVEDAD CON ALCANCE (cuadrado inverso SIN espacio): ¿ahora queda 3D-plano?
=========================================================================================
CS054 (v1) falló acotadamente: mi gravedad ligaba nodos densos con CUALQUIER otro nodo denso, sin
importar cuántos saltos los separaban → fuerza uniforme e infinita → colapsa TODO a un blob. Como dijo
ALEXIS —y es la hipótesis entera—:

    "Si la gravedad fuese igual en todas partes siempre, no habría universo posible."

Una gravedad sin ALCANCE garantiza el colapso; casi una tautología. La pieza que faltaba es el DECAIMIENTO
—lo que hace el cuadrado inverso: atenuar con la separación—, PERO medido sin espacio: la fórmula 1/r²
presupone una distancia euclidiana r (prohibido, G-NO-PRESUPONER-ESPACIO); lo que SÍ va es atenuar con la
DISTANCIA DE GRAFO (saltos de vínculo), la única distancia que emerge del vínculo atado, sin coordenadas.

CS054-v2 (un solo cambio respecto de v1, todo lo demás heredado): la gravedad liga nodos densos con nodos
densos CERCANOS EN EL GRAFO, con peso que DECAE con la distancia de grafo d_ij:
    peso(i,j) ∝ ρ_i · ρ_j · 1/(d_ij)^α ,  d_ij = saltos de vínculo (BFS), α≈2 (cuadrado inverso emergente).
Regiones separadas por muchos saltos casi no se atraen → estructuras extendidas sobreviven, no todo colapsa.
d_ij se computa por BFS sobre el grafo, JAMÁS de una coordenada (assert G-NO-PRESUPONER-ESPACIO intacto).

GUARDIÁN NUEVO — G-ALCANCE: con atenuación puesta, la gravedad-sola debe seguir curvando LOCALMENTE
(G-BALANCE) pero ya NO colapsar TODO a un blob único — debe dejar CÚMULOS separados (como la materia real:
galaxias, no un punto). Si con atenuación sigue dando un blob único, la atenuación no funcionó.

α es FÍSICA, fijado ANTES (=2 por el cuadrado inverso). Se BARRE α∈{1,2,3} como ROBUSTEZ del patrón, NO
como perilla: si "¿sobrevive 3D-plano?" es igual para α=1,2,3, es robusto; si solo sale a un α afinado, es
horneado y se descarta (G-NO-TUNE).

BRAZOS: con_gravedad(α) / sin_gravedad (CS053) / gravedad_sola(α) (G-ALCANCE) / G-NULL (adición al azar).
DESENLACES (los tres): (1) con alcance → sobrevive 3D-plano, distinto de CS053 y azar → CONFIRMACIÓN (lo
que faltaba era el decaimiento). (2) sigue multitud → la gravedad tampoco selecciona ni con alcance.
(3) cúmulos pero no 3D-plano → el alcance evita el colapso (avance) pero falta otra pieza.

Reusa cs053 (ensemble, medidor). numpy-only.
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
medir_geom = _C["medir_geom"]; persiste_S = _C["persiste_S"]; THETA = _C["THETA"]
cadena = _C["cadena"]; cuadrada2d = _C["cuadrada2d"]; cubica3d = _C["cubica3d"]
hipercubica4d = _C["hipercubica4d"]; _tri_euclidea = _C["_tri_euclidea"]; _tri_hip = _C["_tri_hip"]; _arbol = _C["_arbol"]


# ============================ CONFIG ============================
G_RATE = 0.10          # fuerza gravitatoria (FIJO)
H_RATE = 0.10          # despliegue/expansión (FIJO, = G_RATE)
T_DIN  = 6             # pasos de la dinámica
D_MAX  = 2             # alcance máximo (saltos) — GENUINAMENTE LOCAL: solo 2-hop, impide la cascada al blob
                       # (D_MAX=4 se auto-amplificaba: los atajos encogían el diámetro → colapso. G-ALCANCE lo cazó)
ALPHAS = [1, 2, 3]     # barrido de robustez del exponente (2 = cuadrado inverso); NO perilla
NREP   = 3
# ===============================================================


def _ens():
    tipos = [
        ("cadena_d1", lambda: cadena(700)), ("cuadr_d2pl", lambda: cuadrada2d(900)),
        ("tri_d2pl", _tri_euclidea), ("hip37_d2cv", lambda: _tri_hip(7)),
        ("hip38_d2cv", lambda: _tri_hip(8)), ("cubo_d3pl", lambda: cubica3d(729)),
        ("hcubo_d4pl", lambda: hipercubica4d(625)), ("arbol_cv", _arbol),
    ]
    out = []
    for _ in range(NREP):
        out += tipos
    return out


def _giant(adj, N):
    seen = np.zeros(N, bool); best = 0; ncomp = 0
    for s in range(N):
        if not seen[s] and adj[s]:
            q = deque([s]); seen[s] = True; c = 0; ncomp += 1
            while q:
                u = q.popleft(); c += 1
                for w in adj[u]:
                    if not seen[w]:
                        seen[w] = True; q.append(int(w))
            best = max(best, c)
    return best / N, ncomp


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


def dinamica_alcance(adj0, N, rng, alpha, gravedad=True, despliegue=True, null=False):
    """Gravedad CON ALCANCE: liga i (∝ρ) con j CERCANO (BFS ≤D_MAX), peso ∝ ρ_j/d_ij^α. d_ij = SALTOS de
    grafo (BFS), jamás coordenada. G_NULL: liga al azar. Despliegue: remueve al azar (estira)."""
    adj = [set(a) for a in adj0]
    for _ in range(T_DIN):
        E = sum(len(a) for a in adj) // 2
        if E < 2:
            break
        # --- contracción con alcance (gravedad) o al azar (null) ---
        nadd = max(1, int(G_RATE * E))
        rho = np.array([len(a) for a in adj], float)
        if null:
            ii = rng.integers(0, N, size=nadd); jj = rng.integers(0, N, size=nadd)
            for a, b in zip(ii, jj):
                a, b = int(a), int(b)
                if a != b:
                    adj[a].add(b); adj[b].add(a)
        elif gravedad and rho.sum() > 0:
            srcs = rng.choice(N, size=nadd, p=rho / rho.sum())     # fuente ∝ densidad
            for i in srcs:
                i = int(i)
                dist = {i: 0}; q = deque([i])                       # BFS de ALCANCE (≤ D_MAX saltos)
                while q:
                    u = q.popleft()
                    if dist[u] >= D_MAX:
                        continue
                    for w in adj[u]:
                        if w not in dist:
                            dist[w] = dist[u] + 1; q.append(int(w))
                cand = [(j, d) for j, d in dist.items() if d >= 2]  # d≥2: no re-agregar vecino
                if not cand:
                    continue
                w = np.array([rho[j] / (d ** alpha) for j, d in cand])  # peso ∝ ρ_j / d^α (cuadrado inverso)
                if w.sum() <= 0:
                    continue
                k = int(rng.choice(len(cand), p=w / w.sum()))
                j = cand[k][0]
                adj[i].add(j); adj[j].add(i)
        # --- despliegue ---
        if despliegue:
            edges = [(i, j) for i in range(N) for j in adj[i] if i < j]
            if edges:
                rng.shuffle(edges)
                for (i, j) in edges[:int(H_RATE * len(edges))]:
                    adj[i].discard(j); adj[j].discard(i)
    return adj


def _vive_extendido(adj, N):
    gf, ncomp = _giant(adj, N); dia = _diam(adj, N)
    return (gf > 0.5) and (dia > 1.6 * math.log2(max(N, 2))), gf, ncomp, dia


def main():
    rng = np.random.default_rng(2028)
    print("CS054-v2 — GRAVEDAD CON ALCANCE (cuadrado inverso SIN espacio, por saltos de grafo)")
    print("=" * 108)
    print(f"G_RATE={G_RATE} H_RATE={H_RATE} T={T_DIN} D_MAX={D_MAX} saltos · α∈{ALPHAS} (robustez, NO perilla)")
    print("Hipótesis de Alexis: 'gravedad igual en todas partes = sin universo'. El decaimiento es la pieza.")

    espec = _ens()
    # precomputar geometrías (dim0,turn0) y CS053(sin_grav) una vez
    base = []
    for nombre, build in espec:
        try:
            adj, N = build()
        except Exception:
            continue
        dim0, _, turn0 = medir_geom(adj, N)
        S, _, _ = persiste_S(adj, N, np.random.default_rng(int(rng.integers(1 << 30))))
        base.append((nombre, adj, N, dim0, turn0, S > THETA))

    # ---- G-ALCANCE: gravedad_sola con atenuación deja CÚMULOS (no blob único)? ----
    print("\n── G-ALCANCE · gravedad-sola CON atenuación (α=2, sin despliegue): ¿cúmulos o blob único? ──")
    print(f"  {'geom':>11} {'%gig':>5} {'#comp':>6} {'diam':>5}  (blob único = %gig~1 & diam chico)")
    for nombre, adj, N, d0, t0, _ in base[:8]:
        a2 = dinamica_alcance(adj, N, np.random.default_rng(7), 2, gravedad=True, despliegue=False)
        gf, nc = _giant(a2, N); dia = _diam(a2, N)
        print(f"  {nombre:>11} {gf*100:>4.0f} {nc:>6} {dia:>5}")

    # ---- barrido α: patrón de supervivencia con_gravedad ----
    print("\n── con_gravedad (con alcance) — supervivientes por α (robustez del patrón) ──")
    tipos = {}
    for nombre, adj, N, d0, t0, sg in base:
        tipos.setdefault(nombre, {"n": 0, "dim0": d0, "turn0": t0, "sin": 0})
        tipos[nombre]["n"] += 1; tipos[nombre]["sin"] += int(sg)
    for al in ALPHAS:
        for nombre, adj, N, d0, t0, sg in base:
            v, gf, nc, dia = _vive_extendido(
                dinamica_alcance(adj, N, np.random.default_rng(int(rng.integers(1 << 30))), al,
                                 gravedad=True, despliegue=True), N)
            tipos[nombre][f"cg{al}"] = tipos[nombre].get(f"cg{al}", 0) + int(v)
    hdr = f"  {'geom':>11} {'dim0':>5} {'turn0':>6} {'sinGrav':>8}"
    for al in ALPHAS:
        hdr += f" {'cgα=' + str(al):>7}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for nm in sorted(tipos):
        t = tipos[nm]
        row = f"  {nm:>11} {t['dim0']:>5.2f} {t['turn0']:>6.2f} {t['sin']}/{t['n']:<6}"
        for al in ALPHAS:
            row += f" {t.get('cg'+str(al),0)}/{t['n']:<5}"
        print(row)

    # ---- veredicto ----
    print("\n" + "=" * 108)
    print("VEREDICTO (¿la gravedad CON alcance selecciona 3D-plano, robusto en α?):")
    def es_d3plano(nm, t):
        return abs(t["dim0"] - 2.0) < 0.7 and t["turn0"] < 1.5
    for al in ALPHAS:
        viv = [(nm, t) for nm, t in tipos.items() if t.get('cg'+str(al), 0) > 0]
        d3 = sum(t.get('cg'+str(al), 0) for nm, t in tipos.items() if es_d3plano(nm, t))
        tot = sum(t.get('cg'+str(al), 0) for nm, t in tipos.items())
        otros = tot - d3
        print(f"  α={al}: sobreviven {tot} (d≈3-plano={d3}, otros={otros}); tipos vivos: {[nm for nm,_ in viv]}")
    print("\n  LECTURA:")
    print("  · Si con alcance sobrevive SOLO 3D-plano y es igual para α=1,2,3 → CONFIRMACIÓN robusta: el")
    print("    decaimiento era la pieza (avance real sobre CS054). · Si deja CÚMULOS extendidos variados sin")
    print("    privilegiar d=3 → el alcance evita el colapso (avance) pero no fija la dimensión. · Si solo")
    print("    sale a un α → horneado, se descarta. NO tunear α ni tasas buscando 3D.")
    print("  · G-ALCANCE: si gravedad-sola dejó cúmulos (no blob único), el alcance funciona (vs CS054 que")
    print("    colapsaba todo). G-NO-PRESUPONER-ESPACIO: d_ij por BFS (saltos), jamás coordenada.")


main()

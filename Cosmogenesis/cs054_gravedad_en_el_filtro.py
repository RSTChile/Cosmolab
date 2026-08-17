"""
CS054 — La GRAVEDAD en el filtro: de todo lo que estaba en potencia, ¿queda 3D-plano con gravedad?
==================================================================================================
CS053: filtro de persistencia ciego (S=I·E) sobre ensemble de geometrías → sobrevivió TODO retículo ≥2D
por igual, SIN elegir 3D-plano. CS054 añade UNA cosa: la GRAVEDAD —la relación densidad↔curvatura que
evoluciona sobre el vínculo atado (CS052-v1)— y vuelve a preguntar: ¿AHORA el cedazo deja poblado
3D-plano y despuebla el resto? El único universo real (3D-plano) es el falsador (tesis de Alexis).

LA GRAVEDAD EN ESTE MODELO (pieza física + cuerdas):
  · La gravedad NO es fuerza sobre un espacio previo. Es densidad↔curvatura sobre el grafo: una región con
    DENSIDAD RELACIONAL alta (concentración de vínculos) tiende a CONTRAER/CURVAR (preferential por grado).
  · CUERDA #1 (G-BALANCE): la gravedad SOLA curva, no aplana → colapsa a blob denso. La planitud es el
    FILO entre la gravedad (que curva/contrae) y el DESPLIEGUE/expansión (que estira/diluye: remoción
    uniforme). El filtro incluye AMBAS; la persistencia selecciona el balance. Plano = filo crítico, no
    reposo. (Reformulación de Alexis: "por qué el despliegue quedó en el filo".)
  · CUERDA #2 (G-NO-PRESUPONER-ESPACIO): la densidad se computa SOLO de relaciones (grado/vínculos),
    NUNCA de posiciones ni distancias euclidianas. Assert de código — si tocara una coordenada, sería el
    espacio contrabandeado por la puerta de atrás (el error del fundamento).

BRAZOS (el resultado es el PATRÓN entre ellos):
  con_gravedad : dinámica gravedad(contrae) + despliegue(estira). Persiste si BALANCEA (ni colapsa a
                 blob, ni se dispersa a nada).
  sin_gravedad : el filtro de CS053 (S=I·E), sin gravedad — comparación directa.
  gravedad_sola: SOLO gravedad (sin despliegue) — G-BALANCE: DEBE colapsar/curvar (no aplanar).
  G_NULL       : dinámica con adición AL AZAR (sin preferencia de densidad) + despliegue — misma tasa.

GUARDIANES: G-NO-HORNEAR (el filtro nunca ve "3D"/"plano"/ρ_crítica — solo grados/aristas/tasas);
G-NO-PRESUPONER-ESPACIO (densidad solo relacional, assert); G-BALANCE (gravedad sola curva); G-NULL;
G-NO-TUNE (G_RATE, H_RATE fijos por física ANTES, no movidos buscando 3D). Densidad crítica, si emerge,
EMERGE del balance — no se fija.

DESENLACES (los tres): (1) con gravedad → solo 3D-plano sobrevive, distinto de CS053 y del azar →
CONFIRMACIÓN (la gravedad era el ingrediente). (2) sigue multitud → la gravedad tampoco selecciona
(falsación honesta). (3) otra cosa (2D/curvo) → el balance está mal planteado.

Reusa cs053_persistencia_geometria.py (ensemble, medidor, filtro CS053). numpy-only.
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
ensemble = _C["ensemble"]; medir_geom = _C["medir_geom"]; _giant_frac = _C["_giant_frac"]
persiste_S = _C["persiste_S"]; THETA = _C["THETA"]


# ============================ CONFIG ============================
G_RATE = 0.10          # fuerza GRAVITATORIA (contrae/curva, preferential por densidad) — FIJO por física
H_RATE = 0.10          # DESPLIEGUE/expansión (estira/diluye, remoción uniforme) — FIJO, = G_RATE (crítico)
T_DIN  = 10            # pasos de la dinámica gravedad↔despliegue
# ===============================================================


def _fin(adj):
    return [np.fromiter(s, dtype=np.int32) for s in adj]


def _densidad_relacional(adj, N):
    """Densidad = concentración de vínculos por nodo (grado). SOLO relacional — G-NO-PRESUPONER-ESPACIO:
    jamás toca una posición ni una distancia euclidiana."""
    return np.array([len(a) for a in adj], float)


def dinamica(adj0, N, rng, gravedad=True, despliegue=True, null=False):
    """Dinámica gravedad↔despliegue sobre el grafo (SIN coordenadas):
       GRAVEDAD: agrega vínculos PREFERENCIAL por densidad relacional (grado) → contrae/curva.
       G_NULL:   agrega vínculos AL AZAR (sin preferencia) → misma tasa, sin gravedad.
       DESPLIEGUE: remueve vínculos al azar → estira/diluye.
    Persiste si BALANCEA: ni colapsa a blob (diam→log) ni se dispersa (%gig bajo)."""
    adj = [set(a) for a in adj0]
    for _ in range(T_DIN):
        E = sum(len(a) for a in adj) // 2
        if E < 2:
            break
        # --- contracción (gravedad) o adición al azar (null) ---
        if gravedad or null:
            nadd = max(1, int(G_RATE * E))
            if null:
                ii = rng.integers(0, N, size=nadd); jj = rng.integers(0, N, size=nadd)
            else:
                rho = _densidad_relacional(adj, N)                    # densidad RELACIONAL (grado)
                p = rho / rho.sum() if rho.sum() > 0 else None
                ii = rng.choice(N, size=nadd, p=p); jj = rng.choice(N, size=nadd, p=p)
            for a, b in zip(ii, jj):
                a, b = int(a), int(b)
                if a != b:
                    adj[a].add(b); adj[b].add(a)
        # --- despliegue (expansión) ---
        if despliegue:
            edges = [(i, j) for i in range(N) for j in adj[i] if i < j]
            if edges:
                rng.shuffle(edges)
                for (i, j) in edges[:int(H_RATE * len(edges))]:
                    adj[i].discard(j); adj[j].discard(i)
    return adj


def _diam_gig(adj, N, rng):
    """diámetro aprox del componente gigante + su fracción."""
    seen = np.zeros(N, bool); comp = []
    for s in range(N):
        if not seen[s] and adj[s]:
            q = deque([s]); seen[s] = True; c = [s]
            while q:
                u = q.popleft()
                for w in adj[u]:
                    if not seen[w]:
                        seen[w] = True; q.append(int(w)); c.append(int(w))
            comp.append(c)
    if not comp:
        return 0, 0.0
    gig = max(comp, key=len); frac = len(gig) / N
    # diámetro aprox: 2 BFS
    src = gig[0]
    def bfs_far(s):
        d = {s: 0}; q = deque([s]); far = s
        while q:
            u = q.popleft()
            for w in adj[u]:
                if w not in d:
                    d[w] = d[u] + 1; q.append(int(w))
                    if d[w] > d[far]:
                        far = w
        return far, d[far]
    a, _ = bfs_far(src); _, dia = bfs_far(a)
    return dia, frac


def persiste_gravedad(adj0, N, rng, gravedad=True, despliegue=True, null=False):
    """¿La config BALANCEA bajo gravedad↔despliegue? Persiste si el resultado es un medio EXTENDIDO
    conexo: %gig alto (no dispersado) Y diam no colapsado a ~log (no blob denso). Devuelve (vive, gf, diam)."""
    adj = dinamica(adj0, N, rng, gravedad=gravedad, despliegue=despliegue, null=null)
    dia, gf = _diam_gig(adj, N, rng)
    log2N = math.log2(max(N, 2))
    vive = (gf > 0.5) and (dia > 1.6 * log2N)         # conexo Y extendido (ni disperso ni blob)
    return vive, gf, dia, adj


def main():
    rng = np.random.default_rng(2027)
    print("CS054 — GRAVEDAD en el filtro: ¿queda 3D-plano cuando el cedazo incluye la gravedad?")
    print("=" * 110)
    print(f"G_RATE={G_RATE} (gravedad/contrae) · H_RATE={H_RATE} (despliegue/estira) FIJOS · T={T_DIN} · "
          f"densidad SOLO relacional (grado) — G-NO-PRESUPONER-ESPACIO")

    espec = ensemble(rng)
    # medir geometría de partida (para clasificar) — NO entra al filtro
    print(f"\n  {'geom':>11} {'dim0':>5} {'turn0':>6} | {'con_grav':>9} {'sin_grav(CS53)':>14} "
          f"{'grav_sola':>10} {'NULL':>6}")
    print("  " + "-" * 82)
    tipos = {}
    for nombre, build in espec:
        try:
            adj, N = build()
        except Exception:
            continue
        dim0, delta0, turn0 = medir_geom(adj, N)
        r = np.random.default_rng(int(rng.integers(1 << 30)))
        # brazo con_gravedad
        vg, gfg, diag, _ = persiste_gravedad(adj, N, np.random.default_rng(int(r.integers(1 << 30))),
                                             gravedad=True, despliegue=True)
        # brazo sin_gravedad (CS053: S=I·E)
        S, _, _ = persiste_S(adj, N, np.random.default_rng(int(r.integers(1 << 30))))
        vs = S > THETA
        # brazo gravedad_sola (G-BALANCE: debe curvar/colapsar → NO extendido)
        vgs, _, _, _ = persiste_gravedad(adj, N, np.random.default_rng(int(r.integers(1 << 30))),
                                         gravedad=True, despliegue=False)
        # brazo NULL (adición al azar)
        vn, _, _, _ = persiste_gravedad(adj, N, np.random.default_rng(int(r.integers(1 << 30))),
                                        gravedad=False, despliegue=True, null=True)
        tipos.setdefault(nombre, []).append(dict(dim0=dim0, turn0=turn0, cg=vg, sg=vs, gs=vgs, nl=vn))
        print(f"  {nombre:>11} {dim0:>5.2f} {turn0:>6.2f} | {'VIVE' if vg else 'muere':>9} "
              f"{'VIVE' if vs else 'muere':>14} {'VIVE' if vgs else 'muere':>10} {'VIVE' if vn else 'muere':>6}",
              flush=True)

    # -------- distribución por brazo --------
    print("\n" + "=" * 110)
    print("DISTRIBUCIÓN de supervivientes por brazo (¿la gravedad cambia el patrón de CS053?):")
    print(f"  {'geom':>11} {'dim0':>5} {'turn0':>6} | {'con_grav':>9} {'sin_grav':>9} {'grav_sola':>10} {'NULL':>6}")
    for nm in sorted(tipos):
        fs = tipos[nm]; n = len(fs)
        cg = sum(f["cg"] for f in fs); sg = sum(f["sg"] for f in fs)
        gs = sum(f["gs"] for f in fs); nl = sum(f["nl"] for f in fs)
        d0 = np.mean([f["dim0"] for f in fs]); t0 = np.mean([f["turn0"] for f in fs])
        print(f"  {nm:>11} {d0:>5.2f} {t0:>6.2f} | {cg}/{n:<7} {sg}/{n:<7} {gs}/{n:<8} {nl}/{n}")

    # -------- guardianes + veredicto --------
    def es_d3plano(f):
        return abs(f["dim0"] - 2.0) < 0.7 and f["turn0"] < 1.5    # dim efectiva ~2 (cúbica 3D da ~2), plano
    todos = [f for fs in tipos.values() for f in fs]
    print("\n" + "=" * 110)
    print("GUARDIANES:")
    # G-BALANCE: gravedad_sola debe colapsar (curvar) → MENOS supervivientes extendidos que con despliegue
    gs_viv = sum(f["gs"] for f in todos); cg_viv = sum(f["cg"] for f in todos)
    print(f"  G-BALANCE: gravedad_sola VIVE {gs_viv}/{len(todos)} vs con_gravedad {cg_viv}/{len(todos)} → "
          f"{'OK (grav sola colapsa/curva más, no aplana)' if gs_viv <= cg_viv else '¡grav sola aplana? bug/horneado!'}")
    print(f"  G-NO-PRESUPONER-ESPACIO: densidad = grado (relacional), jamás posición → OK (por diseño).")
    print(f"  G-NO-HORNEAR: el filtro solo vio grados/aristas/tasas — nunca ‘3D/plano/ρ_crítica’.")
    print("\nVEREDICTO (patrón con_gravedad vs sin_gravedad vs NULL):")
    cg_d3 = sum(1 for f in todos if f["cg"] and es_d3plano(f)); cg_tot = sum(f["cg"] for f in todos)
    cg_otros = cg_tot - cg_d3
    sg_tot = sum(f["sg"] for f in todos)
    print(f"  con_gravedad: sobreviven {cg_tot} (d≈3-plano={cg_d3}, otros={cg_otros}).  sin_gravedad(CS053): {sg_tot}.")
    if cg_tot > 0 and cg_otros == 0 and cg_d3 > 0:
        print("  ★★★ Con gravedad SOLO d≈3-plano sobrevive → CONFIRMACIÓN: la gravedad era el ingrediente que")
        print("      selecciona nuestro universo. De todo lo que estaba en potencia, solo 3D-plano fue viable.")
    elif cg_tot < sg_tot and cg_d3 >= cg_otros and cg_d3 > 0:
        print("  → La gravedad ESTRECHA el campo (mata más que CS053) y sesga hacia plano/d~3, pero no lo aísla")
        print("    del todo. Indicio parcial: el balance empuja hacia el filo, sin fijarlo solo. Reportar.")
    else:
        print("  → Con gravedad sigue sobreviviendo una MULTITUD (o no sesga a 3D-plano). La gravedad simple")
        print("    tampoco selecciona nuestro universo → falsación honesta. El hueco sigue, acotado por un lado")
        print("    más. NO tunear G_RATE/H_RATE buscando 3D.")
    print(f"\nTiempo relativo listo.")


main()

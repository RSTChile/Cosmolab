"""
CS056 — LAS CUATRO FUERZAS en el proceso acoplado: gravedad + fuerte(confinamiento) + EM + débil
=================================================================================================
Planteo de Alexis: faltaban DOS fuerzas, y cada una hace algo que ninguna previa hace:
  · ELECTROMAGNETISMO — la ÚNICA que REPELE (cargas iguales se repelen, opuestas se atraen). La gravedad
    solo atrae → colapsó a 2D (CS055). El EM puede FRENAR ese colapso empujando aparte → candidato a "lo
    que sostiene la estructura extendida" = la pregunta del filo en 3D. La pieza más prometedora.
  · FUERZA DÉBIL — TRANSMUTA (cambia el tipo, no liga ni repele): un nodo cambia color/carga → deja al
    sistema ESCAPAR de un 2D metaestable hacia un 3D estable. Rango corto, prob baja.

CS055 probó 2 fuerzas a razón irreal 1:1 → la gravedad dominó → 2D. CS056 pone las CUATRO y BARRE la razón
gravedad:fuerte para MAPEAR el paisaje (no forzar 3D), marcando el valor físico.

HONESTIDAD DE MODELADO (dicha de frente): las intensidades reales abarcan 38 órdenes (fuerte 1 · EM 1/137 ·
débil 1e-6 · gravedad 1e-38). En pasos finitos, una fuerza 1e-38 no hace NADA → el "punto físico" del modelo
= gravedad DESPRECIABLE = régimen dominado por el confinamiento. Lo que el barrido mapea LIMPIO es el CRUCE:
a qué razón gravedad:fuerte cambia la dimensión superviviente. Y la pregunta nueva y nítida: ¿el EM (que
repele) rescata las dimensiones altas del colapso gravitatorio? Eso SÍ se contrasta de verdad.

LAS 6 PIEZAS (cada paso, T bajando): enfriamiento · gravedad(contrae, cae 1/d^α, escala·r) · fuerte/
confinamiento(tríos R+V+A bajo umbral) · EM(carga {+,−}: repele igual/atrae opuesto, 1/d²) · débil
(transmuta tipo, prob baja) · despliegue. Toda distancia por SALTOS de grafo (G-NO-PRESUPONER-ESPACIO).
Ninguna fuerza ve "3D" (G-CIEGO-A-DIM). BRAZOS: grav+conf(=CS055) vs 4fuerzas — para AISLAR el EM+débil.
G-NULL: color/carga barajados. Se mide dim POR TIPOS.

DESENLACES: (1) el EM rescata 3D del colapso a intensidad razonable → la repulsión era la pieza. (2) 3D
solo a razones irreales → negativo. (3) ni con 4 → falta el espín/contingencia. Todo informa.

Reusa cs055 (proceso, ensemble, gravedad, confinamiento, despliegue, medidor). numpy-only.
"""
from __future__ import annotations

import os
import math
import numpy as np
from collections import deque

_HERE = os.path.dirname(os.path.abspath(__file__))
_s = open(os.path.join(_HERE, "cs055_proceso_acoplado.py")).read().replace("\nmain()\n", "\n")
_C5 = {"__file__": os.path.join(_HERE, "cs055_proceso_acoplado.py")}
exec(compile(_s, "cs055_proceso_acoplado.py", "exec"), _C5)
medir_geom = _C5["medir_geom"]; _giant = _C5["_giant"]; _diam = _C5["_diam"]; _vive = _C5["_vive"]
_colores = _C5["_colores"]; _grav_paso = _C5["_grav_paso"]; _confin_paso = _C5["_confin_paso"]
_despliegue_paso = _C5["_despliegue_paso"]
cadena = _C5["cadena"]; cuadrada2d = _C5["cuadrada2d"]; cubica3d = _C5["cubica3d"]
hipercubica4d = _C5["hipercubica4d"]; _tri_euclidea = _C5["_tri_euclidea"]; _tri_hip = _C5["_tri_hip"]; _arbol = _C5["_arbol"]
T_HI = _C5["T_HI"]; T_LO = _C5["T_LO"]; T_CONF = _C5["T_CONF"]; T_STEPS = _C5["T_STEPS"]
G_RATE = _C5["G_RATE"]; ALPHA = _C5["ALPHA"]; D_MAX = _C5["D_MAX"]


# ============================ CONFIG ============================
EM_RATE  = 0.06        # electromagnetismo (atrae opuestos / repele iguales) — presente para AISLAR su efecto
W_PROB   = 0.003       # débil: prob de transmutar el tipo de un nodo por paso (rango corto, ínfima)
R_SWEEP  = [1.0, 0.3, 0.1, 0.03, 0.01, 0.0]   # razón gravedad:fuerte (barrido de MAPEO; 0.0 = punto físico ~despreciable)
NREP     = 2
# ===============================================================


def _ens():
    tipos = [("cuadr_d2pl", lambda: cuadrada2d(529)), ("tri_d2pl", _tri_euclidea),
             ("cubo_d3pl", lambda: cubica3d(512)), ("hcubo_d4pl", lambda: hipercubica4d(625)),
             ("hip37_d2cv", lambda: _tri_hip(7))]
    out = []
    for _ in range(NREP):
        out += tipos
    return out


def _em_paso(adj, N, carga, deg0, rng):
    """Electromagnetismo: ATRAE opuestos (agrega vínculo entre cargas opuestas cercanas, 1/d²) y REPELE
    iguales. La repulsión FRENA EL COLAPSO — modelo JUSTO: solo alivia donde la gravedad COMPRIMIÓ (grado
    por encima del basal), nunca erosiona un retículo prístino (que sería un test amañado en contra)."""
    E = sum(len(a) for a in adj) // 2
    if E < 2:
        return
    # --- repulsión: solo entre cargas IGUALES y solo donde AMBOS extremos están sobre-comprimidos ---
    mismas = [(i, j) for i in range(N) for j in adj[i]
              if i < j and carga[i] == carga[j] and len(adj[i]) > deg0[i] and len(adj[j]) > deg0[j]]
    if mismas:
        rng.shuffle(mismas)
        for (i, j) in mismas[:int(EM_RATE * len(mismas)) + 1]:
            adj[i].discard(j); adj[j].discard(i)
    # --- atracción: agregar vínculos entre cargas OPUESTAS cercanas (BFS≤D_MAX), peso 1/d² ---
    nadd = max(1, int(EM_RATE * E))
    for _ in range(nadd):
        i = int(rng.integers(N))
        if not adj[i]:
            continue
        dist = {i: 0}; q = deque([i])
        while q:
            u = q.popleft()
            if dist[u] >= D_MAX:
                continue
            for w in adj[u]:
                if w not in dist:
                    dist[w] = dist[u] + 1; q.append(int(w))
        cand = [(j, d) for j, d in dist.items() if d >= 2 and carga[j] != carga[i]]
        if not cand:
            continue
        w = np.array([1.0 / (d ** 2) for j, d in cand])
        j = cand[int(rng.choice(len(cand), p=w / w.sum()))][0]
        adj[i].add(j); adj[j].add(i)


def _debil_paso(N, color, carga, rng):
    """Débil: transmuta el TIPO (color o carga) de nodos al azar, prob baja. Deja escapar metaestables."""
    n = np.random.default_rng(int(rng.integers(1 << 30)))
    flip = np.where(n.random(N) < W_PROB)[0]
    for i in flip:
        if n.random() < 0.5:
            color[i] = np.int8(n.integers(3))
        else:
            carga[i] = np.int8(1 if carga[i] == 0 else 0)


def proceso4(adj0, N, color0, carga0, rng, r_grav, con_em, con_debil, null=False):
    """Proceso de las 6 piezas. r_grav = razón gravedad:fuerte. con_em/con_debil activan EM/débil."""
    adj = [set(a) for a in adj0]; col = color0.copy(); car = carga0.copy()
    deg0 = [len(a) for a in adj]                       # grado basal (para el EM justo: solo alivia compresión)
    if null:
        rng.shuffle(col); rng.shuffle(car)
    t = np.zeros(N, dtype=np.int32)
    for step in range(T_STEPS):
        T = T_HI * (T_LO / T_HI) ** (step / max(T_STEPS - 1, 1))
        if sum(len(a) for a in adj) < 4:
            break
        if r_grav > 0:                                    # gravedad escalada por la razón
            for _ in range(max(1, int(round(r_grav * 1)))):
                _grav_paso(adj, N, rng, T * r_grav)       # intensidad ∝ r
        if T < T_CONF:                                    # fuerte/confinamiento (referencia = 1)
            _confin_paso(adj, N, col, t, rng)
        if con_em:
            _em_paso(adj, N, car, deg0, rng)
        if con_debil:
            _debil_paso(N, col, car, rng)
        _despliegue_paso(adj, N, rng)
    return adj


def main():
    rng = np.random.default_rng(2030)
    print("CS056 — LAS CUATRO FUERZAS: gravedad + fuerte + EM(repele) + débil, barriendo la razón gravedad:fuerte")
    print("=" * 112)
    print(f"EM_RATE={EM_RATE} W_PROB={W_PROB} · barrido r(grav:fuerte)={R_SWEEP} (0.0=punto físico, gravedad despreciable)")
    print("HONESTIDAD: intensidades reales = 38 órdenes → en el modelo el punto físico = gravedad OFF = régimen confinamiento.")
    print("PREGUNTA NUEVA: ¿el EM (que REPELE) rescata las dim altas del colapso gravitatorio que ganó en CS055?")
    print("\nPREDICCIÓN CIEGA (pre-registrada, antes de leer): a r=1 la gravedad domina→2D (CS055); al bajar r o")
    print("añadir EM, la repulsión frena el colapso → dim altas (3D/4D/hiperb) sobreviven. Si sale SOLO 3D en algún")
    print("r → sorpresa fuerte. Probable: multitud de dim altas (EM frena colapso pero no aísla 3D). El dato decide.\n")

    # G-APAGADO: los 4 subconjuntos para AISLAR qué aporta cada fuerza nueva (cem, cdeb)
    ARMS = [("grav+conf", False, False), ("conf+EM", True, False),
            ("conf+débil", False, True), ("4fuerzas", True, True)]
    espec = _ens()
    tipos_por_r = {}   # (r, arm) -> {tipo: [vivos, n]}
    for r in R_SWEEP:
        for arm, _, _ in ARMS:
            tipos_por_r[(r, arm)] = {}
        for nombre, build in espec:
            try:
                adj, N = build()
            except Exception:
                continue
            color = _colores(N, np.random.default_rng(int(rng.integers(1 << 30))))
            carga = (np.arange(N) % 2).astype(np.int8); np.random.default_rng(int(rng.integers(1 << 30))).shuffle(carga)
            for arm, cem, cdeb in ARMS:
                a = proceso4(adj, N, color, carga, np.random.default_rng(int(rng.integers(1 << 30))),
                             r, cem, cdeb)
                d = tipos_por_r[(r, arm)].setdefault(nombre, [0, 0])
                d[0] += int(_vive(a, N)); d[1] += 1

    # ---- paisaje: dim superviviente por r y por brazo ----
    def dimclass(nm):
        return "2D" if "d2" in nm else ("3D" if "d3" in nm else ("4D" if "d4" in nm else "?"))
    print("PAISAJE — supervivientes por dimensión, por r(grav:fuerte) y brazo (¿el EM rescata dim altas?):")
    print(f"  {'r(g:f)':>7} {'brazo':>11} | {'2D':>6} {'3D':>6} {'4D':>6}   (0.0 = punto físico)")
    print("  " + "-" * 60)
    for r in R_SWEEP:
        for arm, _, _ in ARMS:
            agg = {"2D": [0, 0], "3D": [0, 0], "4D": [0, 0]}
            for nm, (v, n) in tipos_por_r[(r, arm)].items():
                c = dimclass(nm)
                if c in agg:
                    agg[c][0] += v; agg[c][1] += n
            marca = "  ← físico" if r == 0.0 else ""
            print(f"  {r:>7} {arm:>11} | "
                  f"{agg['2D'][0]}/{agg['2D'][1]:<4} {agg['3D'][0]}/{agg['3D'][1]:<4} {agg['4D'][0]}/{agg['4D'][1]:<4}{marca}",
                  flush=True)

    # ---- veredicto ----
    print("\n" + "=" * 112)
    print("VEREDICTO (mapa del paisaje, punto físico marcado):")
    # ¿el EM cambió algo? comparar 4fuerzas vs grav+conf en 3D/4D a cada r
    def d3(r, arm):
        return sum(v for nm, (v, n) in tipos_por_r[(r, arm)].items() if dimclass(nm) == "3D")
    def d4(r, arm):
        return sum(v for nm, (v, n) in tipos_por_r[(r, arm)].items() if dimclass(nm) == "4D")
    print("  ¿El EM (repulsión) rescata dim altas del colapso? (3D+4D: 4fuerzas vs grav+conf, por r)")
    rescate = False
    for r in R_SWEEP:
        alto_4f = d3(r, "4fuerzas") + d4(r, "4fuerzas")
        alto_gc = d3(r, "grav+conf") + d4(r, "grav+conf")
        if alto_4f > alto_gc:
            rescate = True
        print(f"    r={r}: 4fuerzas dim-altas={alto_4f}, grav+conf={alto_gc}  {'← EM RESCATA' if alto_4f > alto_gc else ''}")
    print("\n  LECTURA:")
    if rescate:
        print("  · El EM (repulsión) SÍ rescata dim altas del colapso gravitatorio en algún r → la repulsión")
        print("    era una pieza real (frena el colapso a 2D). ¿Aísla 3D o deja multitud? ver el paisaje.")
    else:
        print("  · El EM NO cambió el patrón (4fuerzas ≈ grav+conf) → la repulsión, así modelada, no frenó el")
        print("    colapso; el colapso gravitatorio a 2D lo decide la gravedad, el EM no lo revierte aquí.")
    print("  · Punto físico (r=0, gravedad despreciable): régimen confinamiento — ver si deja SOLO 3D o multitud.")
    print("    (Recordar: 38 órdenes no se simulan literal; el punto físico = gravedad OFF, no un balance fino.)")
    print("  · G-NO-PRESUPONER-ESPACIO ✓ (distancias por saltos). G-CIEGO-A-DIM ✓ (ninguna fuerza vio la dim).")


main()

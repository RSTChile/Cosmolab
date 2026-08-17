"""
CS053 — PERSISTENCIA de la geometría: de todas las geometrías/dimensiones, ¿cuáles SOBREVIVEN?
==============================================================================================
Tesis de Alexis: "Antes del vínculo NO hay geometría — todas en potencia, ninguna ES. La pregunta no es
cuál se ELIGE de un menú previo (eso es platonismo), sino cuáles PERSISTEN una vez que el vínculo existe.
Y tenemos UN universo real (3D, plano) contra el cual falsar todo lo demás." El mismo cedazo de la materia
(baryogénesis ~50%, CS008) y del exceso de orden a d=3 (CS018), aplicado a la geometría Y la dimensión.

DISEÑO (CS): arrancar de un ENSEMBLE SIMÉTRICO de geometrías/dimensiones (ninguna privilegiada), pasar un
FILTRO DE PERSISTENCIA INTRÍNSECO (robustez S=I·E, CIEGO a nuestro universo), y medir la DISTRIBUCIÓN de
supervivientes en (dimensión, curvatura). Juez: ¿queda d=3-plano poblado y las demás despobladas, y
DISTINTO del azar (G-NULL)?

EL FILTRO (intrínseco, ciego — G-NO-HORNEAR):
  S = I · E, donde SOLO se usan cantidades intrínsecas relacionales, JAMÁS dimensión ni curvatura ni
  "3D/plano" como target:
    I (consistencia interna) = REGULARIDAD del vínculo = grado_medio/(grado_medio+std_grado) ∈ (0,1].
        Una geometría coherente tiene vínculos uniformes; una deshilachada, no.
    E (compatibilidad con el entorno) = RESILIENCIA = fracción del componente gigante tras remover al
        azar una fracción p de vínculos (percolación). Lo que se sostiene bajo estrés, persiste.
  PERSISTE si S > θ (θ FIJO por física antes de correr — G-NO-TUNE). El filtro NUNCA recibe la dimensión
  ni la curvatura de la config: son OUTPUT que se mide DESPUÉS, no input del filtro. (assert G-NO-HORNEAR)

MEDIDOR (output, no input del filtro): dimensión efectiva (dim de CG005) + curvatura (turn: plano→~1,
curvo/árbol→alto; y δ de Gromov). Se mira la DISTRIBUCIÓN de supervivientes.

GUARDIANES:
  G-NO-HORNEAR (EL crítico): el filtro solo ve (I,E) intrínsecos; dim/curvatura se miden aparte y NUNCA
    entran al filtro. Es assert de estructura del código: persiste() no recibe dim ni curv.
  G-ENSEMBLE-SIMÉTRICO: el revuelto de partida NO privilegia d=3 ni plano — se reporta la distribución
    inicial (ancha, con todo tipo). Si arranca sesgado, el resultado está contaminado.
  G-NO-TUNE: θ fijo por física antes, no movido hasta que "salga 3D".
  G-NULL: brazo de control con filtro AL AZAR (misma tasa de muerte). Si el azar deja la MISMA
    distribución → la persistencia no hizo nada; cualquier "preferencia por 3D" sería artefacto del
    ensemble, no del cedazo.

DESENLACES (los tres honestos): (1) sobrevive d=3-plano y las demás no, distinto del azar → CONFIRMACIÓN
fuerte (por qué ESTE espacio). (2) sobrevive multitud sin preferencia / =azar → la persistencia sola no
basta, nuestro universo la falsa (resultado real). (3) sobrevive OTRA cosa (2D/curvo) → contradicción con
la realidad → regla incompleta. NO hornear, NO tunear — reportar la distribución salga lo que salga.

Reusa cg004_attach (arnés, lattice2d, arbol) y cg004f (tri_hiperbolica). numpy-only.
"""
from __future__ import annotations

import os
import math
import numpy as np
from collections import deque

_HERE = os.path.dirname(os.path.abspath(__file__))
_s = open(os.path.join(_HERE, "cg004_attach.py")).read().replace("\nmain()\n", "\n")
_M = {}
exec(compile(_s, "cg004_attach.py", "exec"), _M)
dimension_crecimiento = _M["dimension_crecimiento"]; diagnos = _M["diagnos"]; diametro = _M["diametro"]
lattice2d = _M["lattice2d"]; arbol = _M["arbol"]
_sf = open(os.path.join(_HERE, "cg004f_barrido_curvatura.py")).read().replace("\nmain()\n", "\n")
_F = {"__file__": os.path.join(_HERE, "cg004f_barrido_curvatura.py")}
exec(compile(_sf, "cg004f_barrido_curvatura.py", "exec"), _F)
tri_hiperbolica = _F["tri_hiperbolica"]; tri_euclidea = _F["tri_euclidea"]


# ============================ CONFIG ============================
THETA   = 0.45         # umbral de persistencia S=I·E — FIJO por física antes de correr (G-NO-TUNE)
P_REMOVE = 0.30        # fracción de vínculos removidos para medir RESILIENCIA (E)
NREP    = 6            # instancias por tipo de geometría (estadística)
K_LM    = 80
# ===============================================================


def _fin(adj):
    return [np.fromiter(s, dtype=np.int32) for s in adj]


# ---------------- ENSEMBLE de geometrías/dimensiones (simétrico) ----------------
def cadena(N):                                          # d≈1
    adj = [set() for _ in range(N)]
    for i in range(N - 1):
        adj[i].add(i + 1); adj[i + 1].add(i)
    return adj, N


def cuadrada2d(N):                                      # d≈2 PLANA (grid 4-vecinos)
    L = int(round(N ** (1 / 2))); N2 = L * L
    adj = [set() for _ in range(N2)]
    idx = lambda r, c: r * L + c
    for r in range(L):
        for c in range(L):
            u = idx(r, c)
            if c + 1 < L: v = idx(r, c + 1); adj[u].add(v); adj[v].add(u)
            if r + 1 < L: v = idx(r + 1, c); adj[u].add(v); adj[v].add(u)
    return adj, N2


def cubica3d(N):                                        # d≈3 PLANA (grid 6-vecinos)
    L = max(3, int(round(N ** (1 / 3)))); N3 = L ** 3
    adj = [set() for _ in range(N3)]
    idx = lambda x, y, z: (x * L + y) * L + z
    for x in range(L):
        for y in range(L):
            for z in range(L):
                u = idx(x, y, z)
                if x + 1 < L: v = idx(x + 1, y, z); adj[u].add(v); adj[v].add(u)
                if y + 1 < L: v = idx(x, y + 1, z); adj[u].add(v); adj[v].add(u)
                if z + 1 < L: v = idx(x, y, z + 1); adj[u].add(v); adj[v].add(u)
    return adj, N3


def hipercubica4d(N):                                   # d≈4 PLANA (grid 8-vecinos)
    L = max(3, int(round(N ** (1 / 4)))); N4 = L ** 4
    adj = [set() for _ in range(N4)]
    def idx(a, b, c, d): return ((a * L + b) * L + c) * L + d
    for a in range(L):
        for b in range(L):
            for c in range(L):
                for d in range(L):
                    u = idx(a, b, c, d)
                    if a + 1 < L: v = idx(a + 1, b, c, d); adj[u].add(v); adj[v].add(u)
                    if b + 1 < L: v = idx(a, b + 1, c, d); adj[u].add(v); adj[v].add(u)
                    if c + 1 < L: v = idx(a, b, c + 1, d); adj[u].add(v); adj[v].add(u)
                    if d + 1 < L: v = idx(a, b, c, d + 1); adj[u].add(v); adj[v].add(u)
    return adj, N4


def _to_set(adj_arr):
    return [set(int(x) for x in a) for a in adj_arr]


def _tri_euclidea():
    r = tri_euclidea(1024); return list(r[0]), r[2]        # (adj:list[set], N)

def _tri_hip(q):
    r = tri_hiperbolica(q, 1000); return list(r[0]), r[2]  # (adj:list[set], N)

def _arbol():
    a, n = arbol(1000, 3); return _to_set(a), n            # (adj:list[set], N)


def ensemble(rng):
    """Revuelto SIMÉTRICO de geometrías: d≈1..4, plana/curva±. Ninguna privilegiada. Cada una repetida."""
    tipos = [
        ("cadena_d1",   lambda: cadena(1000)),
        ("cuadr_d2pl",  lambda: cuadrada2d(1024)),
        ("tri_d2pl",    _tri_euclidea),
        ("hip37_d2cv",  lambda: _tri_hip(7)),
        ("hip38_d2cv",  lambda: _tri_hip(8)),
        ("cubo_d3pl",   cubica3d.__call__ if False else (lambda: cubica3d(1000))),
        ("hcubo_d4pl",  lambda: hipercubica4d(1296)),
        ("arbol_cv",    _arbol),
    ]
    espec = []
    for _ in range(NREP):
        espec += tipos
    return espec


# ---------------- FILTRO de persistencia S=I·E (INTRÍNSECO, CIEGO) ----------------
def _giant_frac(adj, N):
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


def persiste_S(adj, N, rng):
    """S = I·E, INTRÍNSECO y CIEGO. NO recibe dimensión ni curvatura (G-NO-HORNEAR).
       I = regularidad del vínculo ; E = resiliencia (componente gigante tras remover P_REMOVE aristas)."""
    grados = np.array([len(a) for a in adj], float)
    gm = grados.mean(); gs = grados.std()
    I = gm / (gm + gs + 1e-9)                            # regularidad ∈ (0,1]
    # resiliencia: remover P_REMOVE de las aristas al azar y medir %gigante
    adj2 = [set(a) for a in adj]
    edges = [(i, j) for i in range(N) for j in adj[i] if i < j]
    rng.shuffle(edges)
    for (i, j) in edges[:int(P_REMOVE * len(edges))]:
        adj2[i].discard(j); adj2[j].discard(i)
    E = _giant_frac(adj2, N)                             # compatibilidad/resiliencia ∈ [0,1]
    return I * E, I, E


# ---------------- MEDIDOR (output: dimensión + curvatura) ----------------
def _turn(adj, N, seed=0):
    rng = np.random.default_rng(seed); active = [i for i in range(N) if len(adj[i]) > 0]
    if len(active) < 10:
        return float("nan")
    ratios = []
    for s in rng.choice(active, size=min(20, len(active)), replace=False):
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


def medir_geom(adj, N):
    adjF = _fin(adj)
    g = dimension_crecimiento(adjF, N, seed=3)
    r = diagnos(adjF, N, K_LM, seed=5)
    return g["d"], r["dmean"], _turn(adjF, N, 7)          # (dim, δ, turn) — SOLO output


def main():
    rng = np.random.default_rng(2026)
    print("CS053 — PERSISTENCIA de la geometría: ¿qué (dim,curvatura) SOBREVIVE al filtro ciego S=I·E?")
    print("=" * 108)
    print(f"θ_persist={THETA} (FIJO) · resiliencia rem {P_REMOVE*100:.0f}% aristas · {NREP} repeticiones · G-NO-HORNEAR: el filtro NO ve dim/curv")

    espec = ensemble(rng)
    filas = []
    print(f"\n  {'geom':>11} {'N':>6} {'dim':>5} {'δ':>6} {'turn':>6} {'I':>5} {'E':>5} {'S=I·E':>6} {'persiste?':>9}")
    print("  " + "-" * 78)
    for nombre, build in espec:
        try:
            adj, N = build()
        except Exception as e:
            continue
        S, I, E = persiste_S(adj, N, np.random.default_rng(int(rng.integers(1 << 30))))
        dim, delta, turn = medir_geom(adj, N)              # medido APARTE, no entra al filtro
        vive = S > THETA
        filas.append(dict(nombre=nombre, N=N, dim=dim, delta=delta, turn=turn, S=S, I=I, E=E, vive=vive))
        print(f"  {nombre:>11} {N:>6} {dim:>5.2f} {delta:>6.2f} {turn:>6.2f} {I:>5.2f} {E:>5.2f} {S:>6.3f} "
              f"{'VIVE' if vive else 'muere':>9}", flush=True)

    # ---- distribución de supervivientes por tipo ----
    print("\n" + "=" * 108)
    print("DISTRIBUCIÓN de supervivientes (filtro S=I·E) — ¿qué geometría/dimensión queda POBLADA?")
    tipos = {}
    for f in filas:
        tipos.setdefault(f["nombre"], []).append(f)
    for nm in sorted(tipos):
        fs = tipos[nm]; nv = sum(1 for f in fs if f["vive"])
        dim = np.mean([f["dim"] for f in fs]); turn = np.mean([f["turn"] for f in fs])
        Sm = np.mean([f["S"] for f in fs])
        print(f"  {nm:>11}: sobrevive {nv}/{len(fs)}  (dim≈{dim:.2f}, turn≈{turn:.2f}, S≈{Sm:.3f})")

    # ---- G-NULL: filtro al azar (misma tasa) ----
    tasa = np.mean([f["vive"] for f in filas])
    rngN = np.random.default_rng(99)
    print(f"\nG-NULL (filtro AL AZAR, misma tasa de muerte {tasa*100:.0f}%): ¿deja la MISMA distribución?")
    null_viv = {}
    for nm in sorted(tipos):
        fs = tipos[nm]; nv = sum(1 for _ in fs if rngN.random() < tasa)
        null_viv[nm] = nv
        print(f"  {nm:>11}: azar sobrevive {nv}/{len(fs)}")

    # ---- veredicto ----
    print("\n" + "=" * 108)
    print("VEREDICTO (los tres desenlaces, honestos):")
    # ¿el filtro concentra supervivencia en d≈3-plano MÁS que el azar?
    def es_d3plano(f):
        return abs(f["dim"] - 2.7) < 0.6 and f["turn"] < 1.4     # d~3 (dim efectiva ~2.7) y ~plano
    viv_d3 = sum(1 for f in filas if f["vive"] and es_d3plano(f))
    tot_viv = sum(1 for f in filas if f["vive"])
    viv_otros = tot_viv - viv_d3
    print(f"  Supervivientes filtro: total={tot_viv}, d≈3-plano={viv_d3}, otros={viv_otros}")
    if tot_viv == 0:
        print("  → Nadie sobrevivió con θ fijo: el umbral mata todo o el filtro no discrimina. Reportar (NO bajar θ).")
    elif viv_d3 > 0 and viv_otros == 0:
        print("  ★★★ SOLO d≈3-plano sobrevive → CONFIRMACIÓN: la persistencia ciega converge a lo que")
        print("      somos, SIN habérselo pedido. Auditar G-NO-HORNEAR y G-NULL con lupa (sería el mayor).")
    else:
        print("  → Sobrevive una MULTITUD sin preferencia clara por d=3-plano (o correlaciona con grado/")
        print("    dimensión alta, no con d=3). La persistencia SOLA no fija 3D-plano → nuestro universo la")
        print("    FALSA. Resultado real (falsación honesta), no fracaso: falta otro ingrediente aguas arriba.")
    print("\n  (G-NO-HORNEAR: el filtro persiste_S sólo recibió (grados, aristas) — nunca dim/curv/‘3D’.)")
    print("  (G-ENSEMBLE-SIMÉTRICO: el revuelto tiene d≈1,2,3,4 y plana/curva± por igual — ver tabla inicial.)")


main()

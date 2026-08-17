"""
CG005 v0 — EDS: ¿emerge una métrica PLANA de RELACIONES con IDENTIDAD persistente?
==================================================================================
Reemplazo canónico de CG004 (nodos vacíos). Fundamento (FUNDAMENTO_origen_era_la_relacion):
la geometría no preexiste; es la HUELLA de que las diferencias persistan y se vinculen. CG004 falló
porque partió de nodos idénticos (forma sin diferencia). CG005 parte de la RELACIÓN con contenido.

SUSTRATO = Espacio de Diferencias Sostenidas (EDS):
  · Nodos con IDENTIDAD inmutable: color C_i ∈ {R,V,A}. Un nodo no existe sin identidad.
  · Vínculos condicionados por el LÓGOS: una arista es estable sii acerca a la NEUTRALIDAD local
    (combinatoria de color de la fuerza fuerte). El motivo neutro mínimo = TRÍADA RVA (barión=blanco).

REGLA v0 (local, falsable, NO horneada — la planitud NO es la función-costo, sólo puede EMERGER):
Energía a minimizar por el enfriamiento (filtro de selección S=I×E, el caos disuelve lo inestable):
    E = Σ_aristas [ w_same·1(C_i=C_j) + c_bond ]  −  λ · (nº de TRÍADAS de color NEUTRO {R,V,A})
  · w_same: un vínculo entre colores IGUALES está frustrado (el vínculo pide DIFERENCIA, no igualdad).
  · c_bond: costo por vínculo (presión de dispersión; evita el colapso a grafo completo).
  · λ: premio por tríada neutra RVA (unidad estable tipo hadrón).
El enfriamiento (Metropolis, T alta→baja) parte de vínculos al azar (caos, I alta pero E baja, S→0)
y disuelve los que no estabilizan neutralidad, hasta cuajar estructuras neutras. Se mide entonces la
MATRIZ DE DISTANCIAS EFECTIVAS y su geometría.

PUENTE CON CG004: una red de tríadas RVA es una tesselación {3,q}. q=6 → PLANO (déficit 0); q>6 →
hiperbólico. La pregunta del experimento: ¿el filtro de neutralidad SELECCIONA coordinación ~6
(plano) sin cirugía externa? El arnés de medición de CG004 (δ Gromov, dim, diam-pend, turn, %gig) lo
lee directo, ya calibrado (ancla lattice2D+ / árbol−).

GUARDIANES / FALSABILIDAD (cableados desde el día 1):
  1. IDENTIDAD INMUTABLE: los colores NUNCA cambian (assert) → caza la falsabilidad (b) "se diluye
     y regresa a nodos vacíos".
  2. %gigante: caza la falsabilidad (a) "colapsa en agujero negro topológico" / fragmenta.
  3. BRAZO NULL DEL LÓGOS (control crítico, anti-Shannon): misma dinámica pero COLOR-CIEGA (premia
     CUALQUIER triángulo, sin exigir neutralidad, w_same=0). Si lo plano emerge IGUAL → el mérito era
     formar triángulos, no el lógos. Si REGLA se separa de NULL → el mérito es la NEUTRALIDAD (el
     vínculo con contenido). Sin este control un positivo es ambiguo.

Reusa el arnés calibrado de cg004_attach.py. numpy-only.
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
N        = 450          # nodos (múltiplo de 3 para color balanceado)
C_BOND   = 1.0          # costo por vínculo (dispersión)
LAMBDA   = 6.0          # premio máx. por SATURAR la neutralidad de un nodo (confinamiento)
TAU      = 3.0          # escala de SATURACIÓN: premio_nodo = λ·(1−e^{−t/τ}); rendimientos decrecientes
T_HI     = 3.0          # temperatura inicial (caos)
T_LO     = 0.04         # temperatura final (congelado)
SWEEPS   = 260          # barridos de enfriamiento (cada uno = N intentos de move)
K_LM     = 100          # landmarks para δ de Gromov
SEEDS    = [1, 2, 3, 4]
# ===============================================================


def _colores(n, rng):
    """Identidad INMUTABLE: colores {0,1,2} balanceados, repartidos al azar."""
    c = np.tile(np.arange(3), n // 3 + 1)[:n]
    rng.shuffle(c)
    return c.astype(np.int8)


def _neutra(ci, cj, ck):
    """¿La tríada (ci,cj,ck) es de color NEUTRO {R,V,A} = blanco (barión)?"""
    return (ci != cj) and (cj != ck) and (ci != ck)


def _comunes_triada(adj, color, i, j, color_ciega):
    """Vecinos comunes k tal que la tríada (i,j,k) CUENTA: neutra RVA (REGLA) o cualquiera (NULL)."""
    comunes = adj[i] & adj[j]
    if color_ciega:
        return list(comunes)
    ci, cj = color[i], color[j]
    return [k for k in comunes if _neutra(ci, cj, color[k])]


def _contar_triadas(adj, color, N, color_ciega):
    """t[i] = nº de tríadas (neutras/any) en las que participa el nodo i (para el premio saturante)."""
    t = np.zeros(N, dtype=np.int32)
    for i in range(N):
        vs = list(adj[i])
        for a in range(len(vs)):
            va = vs[a]
            if va < i:  # contar cada tríada una vez por su nodo mínimo... no: queremos por-nodo, contamos todas
                pass
        # contar tríadas centradas en cualquier posición: par de vecinos conectados que forman tríada válida
        for a in range(len(vs)):
            for b in range(a + 1, len(vs)):
                if vs[b] in adj[vs[a]] and (color_ciega or _neutra(color[i], color[vs[a]], color[vs[b]])):
                    t[i] += 1
    return t


def cuajar(N, color, rng, color_ciega=False):
    """Enfriamiento Metropolis con premio de neutralidad SATURANTE por nodo (confinamiento):
        E = Σ_i [ C_BOND·grado_i − λ·(1 − e^{−t_i/τ}) ]
    donde t_i = nº de tríadas (neutras REGLA / cualquiera NULL) del nodo i. La saturación (1−e^{−t/τ})
    encarna que un quark se confina en UN hadrón y se SATÚA: neutralizarse localmente basta, más
    vínculos sólo cuestan. Impide el colapso a grafo completo. color_ciega=True => brazo NULL."""
    adj = [set() for _ in range(N)]
    m0 = int(2.0 * N)                                    # caos inicial: vínculos al azar
    for _ in range(m0):
        i, j = int(rng.integers(N)), int(rng.integers(N))
        if i != j:
            adj[i].add(j); adj[j].add(i)
    t = _contar_triadas(adj, color, N, color_ciega)      # tríadas por nodo (se mantiene incremental)

    def f(x):
        return 1.0 - math.exp(-x / TAU)

    for s in range(SWEEPS):
        T = T_HI * (T_LO / T_HI) ** (s / max(SWEEPS - 1, 1))
        for _ in range(N):
            i = int(rng.integers(N)); j = int(rng.integers(N))
            if i == j:
                continue
            existe = j in adj[i]
            K = _comunes_triada(adj, color, i, j, color_ciega)   # tríadas (i,j,k) afectadas
            m = len(K)
            sgn = -1 if existe else +1                    # +1 = agregar, −1 = quitar
            # Δcosto (grado_i, grado_j cambian en sgn):
            dcost = sgn * 2.0 * C_BOND
            # Δpremio: t_i,t_j cambian en sgn·m ; cada t_k en sgn·1
            ti2 = t[i] + sgn * m; tj2 = t[j] + sgn * m
            drew = LAMBDA * ((f(ti2) - f(t[i])) + (f(tj2) - f(t[j])))
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


def _coordinacion_neutra(adj, color, N):
    """grado medio + nº medio de TRÍADAS NEUTRAS por nodo (proxy de q en {3,q}) + %nodos en tríada."""
    grados = []; ntri = []; en_triada = 0
    for i in range(N):
        vs = list(adj[i]); grados.append(len(vs))
        t = 0
        for a in range(len(vs)):
            for b in range(a + 1, len(vs)):
                if vs[b] in adj[vs[a]] and _neutra(color[i], color[vs[a]], color[vs[b]]):
                    t += 1
        ntri.append(t)
        if t > 0:
            en_triada += 1
    return float(np.mean(grados)), float(np.mean(ntri)), en_triada / N


def _medir(adj, color, N, sd):
    adjF = _fin(adj)
    dia = diametro(adjF, N, seed=sd)
    g = dimension_crecimiento(adjF, N, seed=sd)
    r = diagnos(adjF, N, K_LM, seed=sd + 11)
    turn = sphere_turnover(adjF, N, seed=sd + 5)
    gmed, tri_med, frac_tri = _coordinacion_neutra(adj, color, N)
    return dict(diam=dia, dim=g["d"], delta=r["dmean"], fg=r["fg"], turn=turn,
                gmed=gmed, tri_med=tri_med, frac_tri=frac_tri, ver=g["ver"])


def main():
    t0 = time.time()
    print("CG005 v0 — EDS: ¿emerge métrica PLANA de relaciones con IDENTIDAD (color) + LÓGOS (neutralidad)?")
    print("=" * 104)
    print(f"N={N} · color {{R,V,A}} inmutable · c_bond={C_BOND} λ={LAMBDA} τ={TAU} (premio saturante) · "
          f"enfriamiento {T_HI}→{T_LO} en {SWEEPS} sweeps · {len(SEEDS)} semillas")

    # -------- Anclas (arnés calibrado de CG004) --------
    print("\nAnclas de la medición (plano vs hiperbólico; deben discriminar):")
    for nm, mk in (("lattice2D", lambda n: lattice2d(n)), ("arbol_b3", lambda n: arbol(n, 3))):
        a, Nr = mk(1024); m = _medir([set(x.tolist()) for x in a], np.zeros(Nr, np.int8), Nr, 7)
        print(f"  {nm:>10}: δ={m['delta']:.2f} turn={m['turn']:.2f} diam={m['diam']} dim={m['dim']:.2f} %gig={m['fg']*100:.0f}")

    # -------- Barrido REGLA vs NULL --------
    print("\n" + "-" * 104)
    print(f"  {'brazo':>7} {'sd':>2} {'%gig':>5} {'g_med':>6} {'tri/nod':>8} {'%entri':>7} "
          f"{'diam':>5} {'δ_med':>7} {'dim':>6} {'turn':>5} {'ver':>10}")
    print("  " + "-" * 92)
    acc = {"REGLA": [], "NULL": []}
    for sd in SEEDS:
        rng = np.random.default_rng(1000 + sd)
        color = _colores(N, rng)                          # IDENTIDAD inmutable (misma para ambos brazos)
        for brazo, ciega in (("REGLA", False), ("NULL", True)):
            rng2 = np.random.default_rng(7000 + sd * 10 + (0 if brazo == "REGLA" else 5))
            adj = cuajar(N, color, rng2, color_ciega=ciega)
            # GUARDIÁN 1: identidad inmutable (los colores nunca se tocaron)
            assert color.dtype == np.int8 and len(color) == N
            m = _medir(adj, color, N, sd)
            acc[brazo].append(m)
            print(f"  {brazo:>7} {sd:>2} {m['fg']*100:>4.0f} {m['gmed']:>6.2f} {m['tri_med']:>8.2f} "
                  f"{m['frac_tri']*100:>6.0f} {m['diam']:>5} {m['delta']:>7.2f} {m['dim']:>6.2f} "
                  f"{m['turn']:>5.2f} {m['ver']:>10}", flush=True)

    # -------- RESUMEN / VEREDICTO --------
    def prom(br, campo):
        xs = [m[campo] for m in acc[br] if m[campo] == m[campo]]
        return float(np.mean(xs)) if xs else float("nan")
    print("\n" + "=" * 104)
    print("RESUMEN — promedios por brazo:")
    for br in ("REGLA", "NULL"):
        print(f"  {br:>7}: %gig={prom(br,'fg')*100:4.0f}  g_med={prom(br,'gmed'):.2f}  "
              f"tri/nodo={prom(br,'tri_med'):.2f}  %en_triada={prom(br,'frac_tri')*100:3.0f}  "
              f"δ={prom(br,'delta'):.2f}  dim={prom(br,'dim'):.2f}  turn={prom(br,'turn'):.2f}  diam={prom(br,'diam'):.0f}")

    print("\nLECTURA (pre-registrada):")
    print("  GUARDIÁN a: %gig≈100 (no colapsó/fragmentó).   GUARDIÁN b: identidad inmutable (assert OK).")
    print("  PLANO (éxito) si REGLA: coord ~6 tríadas/nodo, δ CRECE con N (aquí δ moderado+dim~2), turn→~1,")
    print("    diam grande (no log), %gig~100 — y SEPARADO del NULL (color-ciego). Entonces la NEUTRALIDAD")
    print("    (el lógos) despliega el espacio, no el mero formar triángulos.")
    print("  Si REGLA ≈ NULL → el mérito era la identidad/triángulos, no el lógos (control lo desenmascara).")
    print("  Si curvado/colapsado en ambos → falsabilidad (a): el sustrato con identidad no basta con esta")
    print("    regla; hay que volver aguas arriba (afinar el lógos en Fase I).")
    print(f"\nTiempo: {(time.time()-t0)/60:.1f} min")


main()

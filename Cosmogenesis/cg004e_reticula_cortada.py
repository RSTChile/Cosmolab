"""
CG004-e — TEST (P) PRESERVAR: RE-PEGAR UNA RETÍCULA CORTADA por MAPA DE DESARROLLO
==================================================================================
Adjudicación de CS (adjudicacion_cg004d_CS.md): el test de dos frentes quedó ANULADO por
construcción (dos copias apiladas => sin interfaz espacial). El criterio correcto NO es la
holonomía ROTACIONAL (en el plano es 0 en todas partes => no selecciona nada) sino el MAPA DE
DESARROLLO / holonomía AFÍN: integrar las direcciones `dirs` (rotación + TRASLACIÓN) a lo largo
de cada camino => la traslación desarrollada ES el desplazamiento => carga la MÉTRICA (pasa el
filtro "solo cuenta si cambia distancias").

Y el PRIMER test no es dos frentes: es RE-PEGAR UNA RETÍCULA CORTADA. Se descompone la pregunta:
  (P) PRESERVAR/COMPLETAR: ¿el pegado-por-desarrollo re-cierra planitud YA presente pero cortada?
  (B) BOOTSTRAP        : ¿el pegado GENERA planitud desde crecimiento hiperbólico?  (más caro)
(P) es prerrequisito barato de (B). Este script hace (P).

CONSTRUCCIÓN (limpia, no-horneada):
  1. lattice2D LxL (plano conocido; ancla ya calibrada turn≈1.09, diam-pend≈0.5, δ CRECE).
     Direcciones `dirs` explícitas de la retícula (±x, ±y) = la CONEXIÓN.
  2. CORTE por una costura vertical entre columnas c0 y c0+1: se quitan TODAS las aristas
     horizontales que la cruzan MENOS una BISAGRA (fila r=0). Resultado: las dos orillas quedan
     LEJOS en distancia de grafo (~2L, hay que rodear por la bisagra) pero su posición
     DESARROLLADA es adyacente (offset (+1,0)). Interfaz espacial REAL (lo que faltó en dos-frentes).
     La bisagra mantiene UN solo marco de desarrollo (una sola semilla) => offsets gauge-invariantes
     (cuerda 2 de CS): comparamos offsets RELATIVOS, nunca posiciones absolutas.
  3. RE-PEGAR (mismo conjunto de nodos candidatos = orilla-izq La x orilla-der Ra, mismo nº de
     pegados G en ambos brazos):
       REGLA   = pega a<->b donde el offset DESARROLLADO dev[b]-dev[a] ≈ (+1,0) (dev-adyacente).
                 El mapa de desarrollo ENCUENTRA los pares; NO se le dicen las coords verdaderas.
       CONTROL = pega G pares al AZAR de La x Ra (mismo nº, sin criterio de desarrollo).
  4. Referencias medidas: INTACTA (objetivo) y CORTADA (herida, sin re-pegar).

MAPA DE DESARROLLO (honesto, cuerdas de CS):
  · dev[·] se obtiene INTEGRANDO `dirs` sobre el árbol BFS desde una semilla (no se leen coords).
  · Cuerda 1 (path-dependence): en sustrato PLANO el desarrollo es univaluado; lo VERIFICAMOS
    midiendo el defecto de cierre afín de cada arista no-árbol (debe ser ~0). Ese ~0 ES la
    afirmación "el sustrato local es plano" — condición del caso (P).
  · Cuerda 2 (gauge/semilla): el criterio compara offsets RELATIVOS dev[b]-dev[a] (invariante al
    marco-semilla), nunca posiciones absolutas.

VEREDICTO PRE-REGISTRADO:
  GANA (mecanismo de pegado VÁLIDO para preservar) si REGLA RESTAURA lo plano
  (turn→~1.09, diam-pend→~0.5, δ CRECE con N, %gig~100, diámetro NO colapsa) mientras
  CONTROL COLAPSA (turn→2+, diam→~log, δ acotada) — con REGLA claramente separado de CONTROL.
  Si REGLA NO re-completa ni una retícula cortada => el pegado no es operación válida:
  puerta cerrada con mecanismo (fin de esta rama). Si (P) PASA, se gana el derecho a (B).

Reusa el arnés de medición calibrado de cg004_attach.py.  numpy-only, flush por fila, reanuda.
"""
from __future__ import annotations

import csv
import os
import time
from collections import deque

import numpy as np

# --- reusar el arnés de medición ya calibrado (sin re-ejecutar su main) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(_HERE, "cg004_attach.py")).read().replace("\nmain()\n", "\n")
_M = {}
exec(compile(_src, "cg004_attach.py", "exec"), _M)
diametro = _M["diametro"]; dimension_crecimiento = _M["dimension_crecimiento"]; diagnos = _M["diagnos"]
lattice2d = _M["lattice2d"]; arbol = _M["arbol"]


# ============================ CONFIG (editar aquí) ============================
LOG   = "cg004e_reticula_cortada"
LS    = [32, 64, 128]        # lado L -> N = L*L = 1024, 4096, 16384
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8]
K     = 120                  # landmarks para δ de Gromov
REGLA_TOL = 0.5             # tolerancia del match por desarrollo (|offset-(+1,0)| < tol)
# =============================================================================


# ===========================================================================
#  RETÍCULA con DIRECCIONES explícitas (la conexión) + CORTE con bisagra
# ===========================================================================
def lattice_con_dirs(L):
    """Retícula LxL como adj(list[set]) + dirs(list[dict node->vec unitario]).
    idx = r*L + c ; +x=(1,0) col c+1 ; +y=(0,1) fila r+1. dirs = la CONEXIÓN plana."""
    N = L * L
    adj = [set() for _ in range(N)]
    dirs = [dict() for _ in range(N)]
    def idx(r, c): return r * L + c
    EX = np.array([1.0, 0.0]); EY = np.array([0.0, 1.0])
    for r in range(L):
        for c in range(L):
            u = idx(r, c)
            if c + 1 < L:
                v = idx(r, c + 1)
                adj[u].add(v); adj[v].add(u); dirs[u][v] = EX.copy(); dirs[v][u] = -EX
            if r + 1 < L:
                v = idx(r + 1, c)
                adj[u].add(v); adj[v].add(u); dirs[u][v] = EY.copy(); dirs[v][u] = -EY
    return adj, dirs, N


def cortar_costura(adj, dirs, L, hinge_rows=(0,)):
    """Quita las aristas horizontales que cruzan la costura c0|c0+1, salvo las bisagras.
    Devuelve (c0, cut_pairs, La, Ra): pares cortados y las dos orillas (izq col c0, der col c0+1)."""
    c0 = L // 2 - 1
    def idx(r, c): return r * L + c
    cut_pairs = []; La = []; Ra = []
    for r in range(L):
        a = idx(r, c0); b = idx(r, c0 + 1)
        La.append(a); Ra.append(b)
        if r in hinge_rows:
            continue                                   # bisagra: mantiene UN marco de desarrollo
        if b in adj[a]:
            adj[a].discard(b); adj[b].discard(a)
            dirs[a].pop(b, None); dirs[b].pop(a, None)
            cut_pairs.append((a, b))
    return c0, cut_pairs, La, Ra


# ===========================================================================
#  MAPA DE DESARROLLO: integra `dirs` sobre el árbol BFS desde una semilla.
#  (Cuerda 1) verifica cierre afín de aristas no-árbol => sustrato plano si ~0.
#  (Cuerda 2) sólo se usan offsets RELATIVOS dev[b]-dev[a] (gauge-invariante).
# ===========================================================================
def desarrollar(adj, dirs, N, semilla=0):
    dev = np.full((N, 2), np.nan)
    seen = np.zeros(N, bool)
    # arrancar en la semilla o, si no tiene aristas, en el primer nodo con aristas
    src = semilla if len(adj[semilla]) else next((i for i in range(N) if adj[i]), semilla)
    dev[src] = (0.0, 0.0); seen[src] = True
    q = deque([src])
    while q:
        u = q.popleft()
        for w in adj[u]:
            if not seen[w]:
                dev[w] = dev[u] + dirs[u][w]           # integra rotación+traslación
                seen[w] = True; q.append(int(w))
    # cierre afín (path-independence en sustrato plano): defecto de aristas no-árbol
    defmax = 0.0
    for u in range(N):
        if not seen[u]:
            continue
        for w in adj[u]:
            if w > u and seen[w]:
                d = float(np.linalg.norm((dev[w] - dev[u]) - dirs[u][w]))
                if d > defmax:
                    defmax = d
    return dev, defmax


def repegar(adj_base, dev, La, Ra, cut_pairs, modo, rng, tol=REGLA_TOL):
    """Devuelve adj (list[set]) re-pegado. G = nº de pegados = nº de cortes (mismo en ambos brazos).
    REGLA: pega orilla-izq a<->b donde dev[b]-dev[a] ≈ (+1,0). CONTROL: G pares al azar de La x Ra."""
    adj = [set(s) for s in adj_base]
    G = len(cut_pairs)
    target = np.array([1.0, 0.0])                      # el offset de la arista que faltaba
    ng = 0
    if modo == "REGLA":
        Ra_arr = np.array(Ra)
        devR = dev[Ra_arr]
        for a in La:
            if not np.all(np.isfinite(dev[a])):
                continue
            off = devR - (dev[a] + target)             # 0 en el par verdadero (dev-adyacente)
            k = int(np.argmin(np.einsum("ij,ij->i", off, off)))
            b = int(Ra_arr[k])
            if np.linalg.norm((dev[b] - dev[a]) - target) < tol and b not in adj[a] and a != b:
                adj[a].add(b); adj[b].add(a); ng += 1
    else:                                              # CONTROL: mismo nº, azar, mismos pools
        aa = list(rng.permutation(La)); bb = list(rng.permutation(Ra))
        for a, b in zip(aa[:G], bb[:G]):
            a = int(a); b = int(b)
            if a != b and b not in adj[a]:
                adj[a].add(b); adj[b].add(a); ng += 1
    return adj, ng


# ===========================================================================
#  MÉTRICA de reconvergencia |S(r)| (turn) — de cg004d, sin tocar
# ===========================================================================
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


def _fin(adj_sets):
    return [np.fromiter(s, dtype=np.int32) for s in adj_sets]


def _medir(adj, N, sd):
    dia = diametro(adj, N, seed=sd)
    g = dimension_crecimiento(adj, N, seed=sd)
    r = diagnos(adj, N, K, seed=sd + 11)
    turn = sphere_turnover(adj, N, seed=sd + 5)
    return dia, g, r, turn


def _slope(xs, ys):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    m = np.isfinite(xs) & np.isfinite(ys)
    if m.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(xs[m]), np.log(np.maximum(ys[m], 1.0)), 1)[0])


BRAZOS = ["INTACTA", "CORTADA", "REGLA", "CONTROL"]


def main():
    csv_path = f"{LOG}.csv"
    cols = ["brazo", "L", "N", "seed", "ng", "defdev", "fg", "diam", "dmean", "d95", "dgrow", "turn", "ver"]
    t0 = time.time()
    print("CG004-e — RE-PEGAR RETÍCULA CORTADA por MAPA DE DESARROLLO (test P = preservar)")
    print("=" * 100)
    print(f"L∈{LS} (N=L²) · {len(SEEDS)} semillas · REGLA=dev-adyacente vs CONTROL=azar (mismo nº) · tol={REGLA_TOL}")

    print("\nAnclas (turn=razón S(r+1)/S(r); plano→~1, árbol→alto):")
    for nm, mk in (("lattice2D", lambda n: lattice2d(n)), ("arbol_b3", lambda n: arbol(n, 3))):
        adj, Nr = mk(4096); dia, g, r, turn = _medir(adj, Nr, 7)
        print(f"  {nm:>10} N={Nr}: turn={turn:.2f} diam={dia} δ={r['dmean']:.2f} d_grow={g['d']:.2f} %gig={r['fg']*100:.0f}")

    done = set()
    if os.path.exists(csv_path):
        for row in csv.DictReader(open(csv_path, newline="")):
            done.add((row["brazo"], int(row["L"]), int(row["seed"])))
    unidades = [(L, sd) for L in LS for sd in SEEDS]
    total = len(unidades) * len(BRAZOS)
    print(f"\nfilas={total} hechas={len(done)} faltan={total-len(done)}  LOG={csv_path}\n")
    hdr = (f"{'brazo':>8} {'L':>4} {'N':>6} {'sd':>2} {'ng':>4} {'defdev':>7} {'%gig':>5} {'diam':>6} "
           f"{'δ_med':>7} {'d_grow':>6} {'turn':>5} {'ver':>10}")
    print(hdr); print("-" * len(hdr))
    nuevo = not os.path.exists(csv_path)
    fcsv = open(csv_path, "a", newline=""); w = csv.writer(fcsv)
    if nuevo:
        w.writerow(cols); fcsv.flush()

    def emit(br, L, N, sd, ng, defdev, dia, g, r, turn):
        w.writerow([br, L, N, sd, ng, f"{defdev:.2e}", r["fg"], dia, r["dmean"], r["d95"], g["d"], turn, g["ver"]])
        fcsv.flush()
        print(f"{br:>8} {L:>4} {N:>6} {sd:>2} {ng:>4} {defdev:>7.1e} {r['fg']*100:>4.0f} {dia:>6} "
              f"{r['dmean']:>7.2f} {g['d']:>6.2f} {turn:>5.2f} {g['ver']:>10}", flush=True)

    for (L, sd) in unidades:
        faltan = [b for b in BRAZOS if (b, L, sd) not in done]
        if not faltan:
            continue
        N = L * L
        # construir intacta -> cortar -> desarrollar (una vez por (L,sd))
        adj0, dirs0, _ = lattice_con_dirs(L)
        adj_int = [set(s) for s in adj0]                       # copia intacta ANTES de cortar
        c0, cut_pairs, La, Ra = cortar_costura(adj0, dirs0, L)  # adj0/dirs0 quedan CORTADOS
        dev, defdev = desarrollar(adj0, dirs0, N, semilla=0)
        rng = np.random.default_rng(1000 * sd + L)

        for br in faltan:
            if br == "INTACTA":
                adj = adj_int; ng = 0; dd = 0.0
            elif br == "CORTADA":
                adj = adj0; ng = 0; dd = defdev
            elif br == "REGLA":
                adj, ng = repegar(adj0, dev, La, Ra, cut_pairs, "REGLA", rng); dd = defdev
            else:  # CONTROL
                adj, ng = repegar(adj0, dev, La, Ra, cut_pairs, "CONTROL", rng); dd = defdev
            adjF = _fin(adj)
            dia, g, r, turn = _medir(adjF, N, sd)
            emit(br, L, N, sd, ng, dd, dia, g, r, turn)
    fcsv.close()

    # -------- RESUMEN --------
    rows = list(csv.DictReader(open(csv_path, newline="")))
    def fnum(x):
        try:
            return float(x)
        except Exception:
            return float("nan")
    def per_seed(br, L, campo):
        return {int(r["seed"]): fnum(r[campo]) for r in rows if r["brazo"] == br and int(r["L"]) == L}

    print("\n" + "=" * 100)
    print("RESUMEN — ¿REGLA (dev-adyacente) RESTAURA lo plano y CONTROL (azar) COLAPSA?")
    print("objetivo plano: diam-pend→0.5, turn→~1.09, δ CRECE con N, %gig~100")
    for br in BRAZOS:
        pends = []
        for sd in SEEDS:
            dias = [per_seed(br, L, "diam").get(sd, np.nan) for L in LS]
            pends.append(_slope([L * L for L in LS], dias))
        pends = [p for p in pends if p == p]
        pm, ps = (np.mean(pends), np.std(pends)) if pends else (np.nan, np.nan)
        dme = [np.nanmean(list(per_seed(br, L, "dmean").values())) for L in LS]
        turn = [np.nanmean(list(per_seed(br, L, "turn").values())) for L in LS]
        dg = [np.nanmean(list(per_seed(br, L, "dgrow").values())) for L in LS]
        gig = np.nanmean(list(per_seed(br, LS[-1], "fg").values())) * 100
        dtr = "CRECE(plano)" if len(dme) >= 2 and dme[-1] > dme[0] + 0.5 else "acotada(hiperb)"
        print(f"  {br:>8}: diam-pend={pm:5.2f}±{ps:.2f}  %gig={gig:3.0f}  δ→{dtr:>16}  "
              f"δ={'/'.join(f'{x:.1f}' for x in dme)}  turn={'/'.join(f'{x:.2f}' for x in turn)}  "
              f"d_grow={'/'.join(f'{x:.2f}' for x in dg)}")
    print("\nLECTURA (pre-registrada, adjudicación CS):")
    print("  · GANA (P) si REGLA≈INTACTA (turn→~1.09, diam-pend→0.5, δ CRECE) y CONTROL COLAPSA")
    print("    (turn→2+, diam→log, δ acotada). Entonces el pegado-por-desarrollo PRESERVA => derecho a (B).")
    print("  · Si REGLA NO restaura (≈CORTADA o ≈CONTROL) => el pegado no es operación válida:")
    print("    puerta cerrada con mecanismo (script chico, negativo honesto).")
    print("  · defdev (defecto de cierre afín) DEBE ser ~0: confirma sustrato plano (cuerda 1).")
    print(f"\nTiempo: {(time.time()-t0)/60:.1f} min · CSV: {csv_path}")


main()
